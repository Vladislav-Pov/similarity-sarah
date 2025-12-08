from comet_ml import Experiment

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -------------------------
# Hyperparameters
# -------------------------
NUM_CLIENTS = 10
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# -------------------------
# COMET ML (опционально)
# -------------------------
experiment = Experiment(
    api_key="tlPyEqcWoRIJ9LyqK7782UYRC",
    project_name="federated-sarah",
    workspace="zukep102",
)
experiment.set_name("SARAH-ResNet18-CIFAR10-named-grads")


# -------------------------
# CIFAR-10 DATASET
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Handle uneven split
total = len(dataset)
base = total // NUM_CLIENTS
lengths = [base] * NUM_CLIENTS
remainder = total - base * NUM_CLIENTS
for i in range(remainder):
    lengths[i] += 1

client_subsets = random_split(dataset, lengths)


# -------------------------
# Helpers
# -------------------------
def gradients_to_dict(model):
    """Return dict mapping parameter names -> grad tensors (detached clones)."""
    gdict = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            gdict[name] = torch.zeros_like(param.data)
        else:
            gdict[name] = param.grad.detach().clone()
    return gdict

def l2_norm_of_dict(d):
    s = 0.0
    for v in d.values():
        s += (v.norm() ** 2)
    return torch.sqrt(s).item()


# -------------------------
# SARAH Client
# -------------------------
class SARAHClient:
    def __init__(self, client_id, subset):
        self.client_id = client_id
        self.dataset = subset
        self.dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        self.current_weights = None  # dict (cpu tensors)
        self.previous_weights = None # dict (cpu tensors)
        self.v_prev = None           # dict (cpu tensors) or None
        self.need_full_grad = False

    def receive_weights(self, current_weights, previous_weights, v_prev, need_full_grad):
        # Expect current_weights / previous_weights to be dict state_dict (cpu tensors)
        self.current_weights = {k: v.clone().detach() for k, v in current_weights.items()} if current_weights else None
        self.previous_weights = {k: v.clone().detach() for k, v in previous_weights.items()} if previous_weights else None
        self.v_prev = {k: v.clone().detach() for k, v in v_prev.items()} if v_prev else None
        self.need_full_grad = need_full_grad

    def compute_gradient(self):
        """
        Returns:
            grad_dict: dict mapping param_name -> gradient tensor (cpu tensors)
        """
        # build a fresh model and load current weights (move to DEVICE)
        model = models.resnet18(num_classes=10).to(DEVICE)
        model.train()

        # load_state_dict with tensors moved to DEVICE
        if self.current_weights is None:
            # nothing to do
            return {name: torch.zeros_like(p) for name, p in model.state_dict().items()}

        model.load_state_dict({k: v.to(DEVICE) for k, v in self.current_weights.items()})

        criterion = nn.CrossEntropyLoss()

        # ---------- FULL GRADIENT ----------
        if self.need_full_grad:
            full_grad_dict = None
            num_batches = 0

            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                grads = gradients_to_dict(model)  # tensors on DEVICE

                # move grads to CPU for aggregation / sending
                grads = {k: v.detach().cpu() for k, v in grads.items()}

                if full_grad_dict is None:
                    full_grad_dict = {k: v.clone() for k, v in grads.items()}
                else:
                    for k in full_grad_dict:
                        full_grad_dict[k] += grads[k]
                num_batches += 1

            if num_batches == 0:
                # empty subset
                return {k: torch.zeros_like(v) for k, v in self.current_weights.items()}

            # average and return
            for k in full_grad_dict:
                full_grad_dict[k] /= float(num_batches)

            # debug norm
            print(f"Client {self.client_id} FULL grad norm: {l2_norm_of_dict(full_grad_dict):.6f}")

            return full_grad_dict  # cpu tensors

        # ---------- RECURSIVE GRADIENT (v_new = avg(grad_cur - grad_prev) + v_prev) ----------
        else:
            # Need previous_weights and v_prev available
            if (self.previous_weights is None) or (self.v_prev is None):
                # fallback to zero dict
                return {k: torch.zeros_like(v) for k, v in self.current_weights.items()}

            acc_diff = None
            num_batches = 0

            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # grad current
                model.load_state_dict({k: v.to(DEVICE) for k, v in self.current_weights.items()})
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                grads_cur = gradients_to_dict(model)  # DEVICE
                grads_cur = {k: v.detach().cpu() for k, v in grads_cur.items()}

                # grad previous
                model.load_state_dict({k: v.to(DEVICE) for k, v in self.previous_weights.items()})
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                grads_prev = gradients_to_dict(model)  # DEVICE
                grads_prev = {k: v.detach().cpu() for k, v in grads_prev.items()}

                # diff
                diff = {k: (grads_cur[k] - grads_prev[k]) for k in grads_cur}

                if acc_diff is None:
                    acc_diff = {k: diff[k].clone() for k in diff}
                else:
                    for k in acc_diff:
                        acc_diff[k] += diff[k]

                num_batches += 1

            if num_batches == 0:
                return {k: torch.zeros_like(v) for k, v in self.current_weights.items()}

            for k in acc_diff:
                acc_diff[k] /= float(num_batches)

            # add v_prev (both are CPU tensors)
            v_new = {k: acc_diff[k] + self.v_prev[k] for k in acc_diff}

            print(f"Client {self.client_id} RECURSIVE grad (v_new) norm: {l2_norm_of_dict(v_new):.6f}")

            return v_new  # cpu tensors


# -------------------------
# SARAH Server
# -------------------------
class SARAHServer:
    def __init__(self, model, clients, lr):
        self.model = model
        self.clients = clients
        self.learning_rate = lr
        # store current_weights as cpu tensors (state_dict)
        self.current_weights = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        self.previous_weights = None
        self.v_prev = None  # dict of cpu tensors

    def send_weights_to_client(self, client_id, need_full_grad):
        client = self.clients[client_id]

        cw = {k: v.clone().detach() for k, v in self.current_weights.items()}
        pw = {k: v.clone().detach() for k, v in self.previous_weights.items()} if self.previous_weights else None
        vp = {k: v.clone().detach() for k, v in self.v_prev.items()} if self.v_prev else None

        client.receive_weights(cw, pw, vp, need_full_grad)

    def receive_gradient_from_client(self, gradient):
        # gradient: dict of cpu tensors coming from client
        # store as cpu clones
        self.v_prev = {k: g.clone().detach() for k, g in gradient.items()}

    def update_model(self, gradient):
        """
        gradient: dict of cpu tensors (parameter_name -> grad)
        We apply update: sd[name] = sd[name] - lr * grad.to(sd[name].device)
        """
        with torch.no_grad():
            sd = self.model.state_dict()
            # apply update by key
            for k, g in gradient.items():
                if k not in sd:
                    raise KeyError(f"Gradient contains key {k} not in model.state_dict()")
                sd_val = sd[k]
                # move gradient to same device as parameter and apply
                sd[k] = sd_val - self.learning_rate * g.to(sd_val.device)
            # load updated state dict back into model
            self.model.load_state_dict(sd)

        # update snapshots (store cpu copies)
        self.previous_weights = {k: v.clone().detach().cpu() for k, v in self.current_weights.items()}
        self.current_weights = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}


# -------------------------
# Evaluation function
# -------------------------
def evaluate_model(model, data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# -------------------------
# Init global model & SARAH
# -------------------------
global_model = models.resnet18(num_classes=10).to(DEVICE)
clients = [SARAHClient(i, client_subsets[i]) for i in range(NUM_CLIENTS)]
server = SARAHServer(global_model, clients, lr=LEARNING_RATE)

train_loader_for_logging = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------
# TRAINING LOOP
# -------------------------
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    permutation = torch.randperm(NUM_CLIENTS).tolist()
    # snapshot previous weights before starting federated round
    server.previous_weights = {k: v.clone().detach() for k, v in server.current_weights.items()}
    server.v_prev = None

    # Federated SARAH steps
    for i, client_id in enumerate(permutation):
        need_full_grad = (i == 0)
        server.send_weights_to_client(client_id, need_full_grad)
        gradient = clients[client_id].compute_gradient()   # dict (cpu tensors)
        server.receive_gradient_from_client(gradient)       # server.v_prev = gradient
        server.update_model(gradient)                       # apply by keys

    # Evaluate and log
    train_loss, train_acc = evaluate_model(global_model, train_loader_for_logging)
    val_loss, val_acc = evaluate_model(global_model, val_loader)

    print(f" Train loss={train_loss:.4f}  acc={train_acc:.2f}%")
    print(f" Valid loss={val_loss:.4f}  acc={val_acc:.2f}%")

    experiment.log_metric("train_loss", train_loss, epoch=epoch)
    experiment.log_metric("val_loss", val_loss, epoch=epoch)
    experiment.log_metric("train_acc", train_acc, epoch=epoch)
    experiment.log_metric("val_acc", val_acc, epoch=epoch)


# -------------------------
# Final test accuracy
# -------------------------
print("\nFinal Test Evaluation:")
test_loss, test_acc = evaluate_model(global_model, val_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
