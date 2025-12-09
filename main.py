from comet_ml import Experiment  # важно импортировать до torch

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# 1. Настройки и Comet ML
# ==========================================
API_KEY = "tlPyEqcWoRIJ9LyqK7782UYRC"
PROJECT_NAME = "sarah-cifar10-experiments"

experiment = Experiment(
    api_key=API_KEY,
    project_name=PROJECT_NAME,
    workspace="zukep102",
    auto_output_logging="simple",
)

experiment.set_name("SARAH-ResNet18-CIFAR10")

# Базовые параметры
BATCH_SIZE    = 64
BASE_LR       = 0.002        # базовый learning rate
WEIGHT_DECAY  = 5e-4         # стандартный L2 для CIFAR10+ResNet
EPOCHS        = 250
CLIP_NORM     = 1.0          # максимум для ||v_t||
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment.log_parameters({
    "batch_size": BATCH_SIZE,
    "base_learning_rate": BASE_LR,
    "weight_decay": WEIGHT_DECAY,
    "epochs": EPOCHS,
    "optimizer": "SARAH_Permutation",
    "model": "ResNet18_CIFAR",
    "clip_norm": CLIP_NORM,
})

# ==========================================
# 2. Данные
# ==========================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# Лоадер для полного градиента (фиксированный порядок)
train_loader_full = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# ==========================================
# 3. Модель
# ==========================================
def get_cifar_resnet18():
    model = torchvision.models.resnet18(weights=None)
    # адаптация под 32x32
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

model = get_cifar_resnet18().to(DEVICE)

# ==========================================
# 4. Вспомогательные функции SARAH
# ==========================================
criterion = nn.CrossEntropyLoss()

def compute_full_gradient(model, loader):
    """
    Полный градиент v0 и честные train_loss/train_acc.
    Считаем в eval(), чтобы не дёргать BatchNorm-статистики.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]

    with torch.set_grad_enabled(True):
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            grads = torch.autograd.grad(loss, model.parameters())
            for i, g in enumerate(grads):
                accumulated_grads[i] += g.detach() * inputs.size(0)

    for i in range(len(accumulated_grads)):
        accumulated_grads[i] /= total

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return accumulated_grads, avg_loss, acc

def get_batch_grads(model, inputs, targets, params_state=None):
    """
    Градиент для одного батча при заданных весах.
    Если params_state не None — считаем grad(w_prev),
    а затем восстанавливаем исходные веса.
    """
    original_state = None
    if params_state is not None:
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(params_state)

    model.train()  # BN использует статистику текущего батча

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    grads = torch.autograd.grad(loss, model.parameters())

    if original_state is not None:
        model.load_state_dict(original_state)

    return [g.detach() for g in grads]

def update_weights(model, update_vecs, lr, weight_decay):
    """
    Обновление: w <- (1 - lr*wd)*w - lr*v_t
    """
    with torch.no_grad():
        for param, vec in zip(model.parameters(), update_vecs):
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.sub_(lr * vec)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100.0 * correct / total

# ==========================================
# 5. Обучение SARAH + Random Reshuffling
# ==========================================
print(f"Starting SARAH training on {DEVICE}...")

for epoch in range(EPOCHS):
    # простое расписание шага обучения
    if epoch < 15:
        lr_epoch = BASE_LR
    elif epoch < 30:
        lr_epoch = BASE_LR * 0.3
    else:
        lr_epoch = BASE_LR * 0.1

    print(f"\nEpoch {epoch+1}/{EPOCHS} (lr = {lr_epoch:.5f})")

    # новый DataLoader с перестановкой на каждую эпоху
    train_loader_perm = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # 1. полный градиент v0 и метрики на w0
    print("Computing full gradient (v0)...")
    v0, train_loss, train_acc = compute_full_gradient(model, train_loader_full)

    # сохраняем w0
    w0_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 2. первый шаг: w1 = w0 - lr_epoch * v0
    update_weights(model, v0, lr_epoch, WEIGHT_DECAY)
    v_prev = v0
    w_prev_state = w0_state

    # 3. внутренний цикл SARAH по перестановке батчей
    print("Inner loop...")
    last_grad_norm = 0.0
    for inputs, targets in tqdm(train_loader_perm):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # градиент в текущей точке w_t
        grad_curr = get_batch_grads(model, inputs, targets, params_state=None)
        # градиент в предыдущей точке w_{t-1} на том же батче
        grad_prev = get_batch_grads(model, inputs, targets, params_state=w_prev_state)

        # v_t = grad_curr - grad_prev + v_{t-1}
        v_curr = []
        norm_sq = 0.0
        for g_c, g_p, v_p in zip(grad_curr, grad_prev, v_prev):
            diff = g_c - g_p + v_p
            v_curr.append(diff)
            norm_sq += diff.norm().item() ** 2
        total_norm = norm_sq ** 0.5
        last_grad_norm = total_norm

        # gradient clipping по норме v_t
        if total_norm > CLIP_NORM:
            scale = CLIP_NORM / (total_norm + 1e-6)
            v_curr = [v * scale for v in v_curr]

        # сохраняем текущее w_t как w_{t-1} для следующего шага
        w_prev_state = {k: v.clone() for k, v in model.state_dict().items()}

        # шаг обновления
        update_weights(model, v_curr, lr_epoch, WEIGHT_DECAY)
        v_prev = v_curr

    # 4. валидация после inner-loop
    val_loss, val_acc = evaluate(model, test_loader)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    experiment.log_metrics(
        {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "grad_norm": last_grad_norm,
            "lr_epoch": lr_epoch,
        },
        step=epoch,
    )

print("Training finished.")
experiment.end()