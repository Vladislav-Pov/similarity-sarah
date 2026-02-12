import copy
import math
import random

import optuna
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torchvision.models import resnet18

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

# ==========================================
# 1. Settings
# ==========================================
API_KEY = "tlPyEqcWoRIJ9LyqK7782UYRC"
PROJECT_NAME = "sarah-cifar10-experiments"
WORKSPACE = "zukep102"
EXPERIMENT_NAME = "SARAH-ResNet18-CIFAR10"

USE_COMET = True
COMET_LOG_TUNING = False

RUN_TUNING = True
RUN_FINAL_TRAIN = True

TUNING_TRIALS = 100
TUNING_EPOCHS = 60
FINAL_EPOCHS = 200

DATA_ROOT = "./data"
NUM_WORKERS = 2
TEST_BATCH_SIZE = 256
VAL_SPLIT = 0.1
SEED = 42

DEFAULT_PARAMS = {
    "sarah_lr": 0.02,
    "weight_decay": 5e-4,
    "warmup_epochs": 10,
    "min_lr": 1e-4,
    "label_smoothing": 0.1,
    "max_grad_norm": 5.0,
    "batch_size": 128,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_create_experiment(run_name):
    if not USE_COMET or Experiment is None:
        return None
    experiment = Experiment(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
        auto_output_logging="simple",
    )
    experiment.set_name(run_name)
    return experiment


def log_metrics(experiment, metrics, step):
    if experiment is None:
        return
    experiment.log_metrics(metrics, step=step)


def finalize_params(params):
    final_params = params.copy()
    if "min_lr_ratio" in final_params:
        final_params["min_lr"] = final_params["sarah_lr"] * final_params.pop("min_lr_ratio")
    return final_params


def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def build_criterions(label_smoothing):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_sum = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)
    return criterion, criterion_sum


def build_dataloaders(batch_size, val_split, seed):
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_test,
    )
    num_train = len(train_dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator).tolist()
    split = int(num_train * (1 - val_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return trainloader, valloader

def build_full_trainloader(batch_size):
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    return torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )


def build_testloader():
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_test,
    )
    return torch.utils.data.DataLoader(
        testset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


def zero_grads(model_to_zero):
    for param in model_to_zero.parameters():
        if param.grad is not None:
            param.grad.zero_()


def clone_grads(model_to_clone):
    grads = []
    for param in model_to_clone.parameters():
        if param.grad is None:
            grads.append(None)
        else:
            grads.append(param.grad.detach().clone())
    return grads


def apply_weight_decay(grads, model_for_decay, decay):
    if decay == 0:
        return grads
    decayed = []
    for grad, param in zip(grads, model_for_decay.parameters()):
        if grad is None:
            decayed.append(None)
        else:
            decayed.append(grad + decay * param.detach())
    return decayed


def clip_grads(grads, max_norm):
    if max_norm is None or max_norm <= 0:
        return grads
    total_norm_sq = torch.zeros(1, device=device)
    for grad in grads:
        if grad is None:
            continue
        total_norm_sq += grad.pow(2).sum()
    total_norm = total_norm_sq.sqrt()
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        clipped = []
        for grad in grads:
            if grad is None:
                clipped.append(None)
            else:
                clipped.append(grad * scale)
        return clipped
    return grads


def apply_update(params, grads, lr):
    with torch.no_grad():
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            param.add_(grad, alpha=-lr)


def compute_full_grad(model_for_grad, loader, loss_fn_sum, decay):
    model_for_grad.train()
    for param in model_for_grad.parameters():
        param.grad = None
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_for_grad(inputs)
        loss = loss_fn_sum(outputs, targets)
        loss.backward()
        total += targets.size(0)
    full_grads = []
    for param in model_for_grad.parameters():
        if param.grad is None:
            full_grads.append(None)
        else:
            full_grads.append(param.grad.detach().clone() / total)
    return apply_weight_decay(full_grads, model_for_grad, decay)


def train_epoch(model, trainloader, criterion, criterion_sum, weight_decay, max_grad_norm, lr):
    model.train()
    full_grads = compute_full_grad(model, trainloader, criterion_sum, weight_decay)
    full_grads = clip_grads(full_grads, max_grad_norm)
    model_prev = copy.deepcopy(model).to(device)

    apply_update(model.parameters(), full_grads, lr)
    v_prev = full_grads

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        zero_grads(model)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        grad_cur = apply_weight_decay(clone_grads(model), model, weight_decay)

        zero_grads(model_prev)
        outputs_prev = model_prev(inputs)
        loss_prev = criterion(outputs_prev, targets)
        loss_prev.backward()
        grad_prev = apply_weight_decay(clone_grads(model_prev), model_prev, weight_decay)
        v_t = []
        for g_cur, g_prev, v_old in zip(grad_cur, grad_prev, v_prev):
            if g_cur is None:
                v_t.append(None)
            else:
                v_t.append(g_cur - g_prev + v_old)
        v_t = clip_grads(v_t, max_grad_norm)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        model_prev.load_state_dict(model.state_dict())
        apply_update(model.parameters(), v_t, lr)
        v_prev = v_t

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    return train_loss, train_acc

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    eval_loss = loss_sum / total
    eval_acc = 100.0 * correct / total
    return eval_loss, eval_acc


def get_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def train_loop(
    model,
    trainloader,
    eval_loader,
    criterion,
    criterion_sum,
    params,
    total_epochs,
    eval_name,
    experiment=None,
    trial=None,
):
    best_eval_acc = 0.0
    eval_label = eval_name.capitalize() if eval_name else "Eval"

    for epoch in range(1, total_epochs + 1):
        lr = get_lr(
            epoch,
            total_epochs,
            params["warmup_epochs"],
            params["sarah_lr"],
            params["min_lr"],
        )
        train_loss, train_acc = train_epoch(
            model,
            trainloader,
            criterion,
            criterion_sum,
            params["weight_decay"],
            params["max_grad_norm"],
            lr,
        )
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr": lr,
        }
        if eval_name:
            metrics[f"{eval_name}_loss"] = eval_loss
            metrics[f"{eval_name}_acc"] = eval_acc
        log_metrics(experiment, metrics, step=epoch)

        if trial is not None:
            trial.report(eval_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        print(
            f"Epoch {epoch} Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
            f"{eval_label} Loss: {eval_loss:.4f} Acc: {eval_acc:.2f}%"
        )

    return best_eval_acc


def objective(trial):
    params = {
        "sarah_lr": trial.suggest_float("sarah_lr", 1e-3, 0.2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 20),
        "min_lr_ratio": trial.suggest_float("min_lr_ratio", 1e-3, 1e-1, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 10.0),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    params = finalize_params(params)

    trainloader, valloader = build_dataloaders(params["batch_size"], VAL_SPLIT, SEED)
    model = build_model().to(device)
    criterion, criterion_sum = build_criterions(params["label_smoothing"])
    experiment = None
    if COMET_LOG_TUNING:
        experiment = maybe_create_experiment(f"{EXPERIMENT_NAME}-trial-{trial.number}")

    best_val_acc = train_loop(
        model,
        trainloader,
        valloader,
        criterion,
        criterion_sum,
        params,
        TUNING_EPOCHS,
        eval_name="val",
        experiment=experiment,
        trial=trial,
    )
    if experiment is not None:
        experiment.end()
    return best_val_acc


def train_with_params(params, total_epochs, run_name):
    trainloader = build_full_trainloader(params["batch_size"])
    testloader = build_testloader()
    model = build_model().to(device)
    criterion, criterion_sum = build_criterions(params["label_smoothing"])

    experiment = maybe_create_experiment(run_name)
    best_test_acc = train_loop(
        model,
        trainloader,
        testloader,
        criterion,
        criterion_sum,
        params,
        total_epochs,
        eval_name="test",
        experiment=experiment,
    )
    if experiment is not None:
        experiment.end()
    return best_test_acc


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    if RUN_TUNING:
        sampler = TPESampler(seed=SEED)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=TUNING_TRIALS)

        best_params = finalize_params(study.best_trial.params)
        print(f"Best val acc: {study.best_value:.2f}%")
        print(f"Best params: {best_params}")

        if RUN_FINAL_TRAIN:
            train_with_params(best_params, FINAL_EPOCHS, f"{EXPERIMENT_NAME}-best")
    else:
        train_with_params(DEFAULT_PARAMS, FINAL_EPOCHS, EXPERIMENT_NAME)


if __name__ == "__main__":
    main()