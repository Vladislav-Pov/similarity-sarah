"""
Distributed client-server SARAH implementation with K clients.

Each client holds a disjoint subset of training data and computes local gradients.
The server maintains the global model and SARAH state (v_t, s_t).
"""

import copy
import random
import torch
import torch.nn as nn
from typing import List, Tuple


class Client:
    """
    Client holding a local dataset subset.
    Computes gradients on request but never updates model weights.
    """
    
    def __init__(self, dataloader, device):
        """
        Args:
            dataloader: PyTorch DataLoader with client's local data subset
            device: torch device (cuda/cpu)
        """
        self.dataloader = dataloader
        self.device = device
        self.data_iter = iter(self.dataloader)
    
    def sample_batch(self):
        """Sample a batch from local data (cycles if exhausted)."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch
    
    def compute_grad(self, model, batch, criterion):
        """
        Compute gradient on given batch with given model.
        
        Args:
            model: PyTorch model
            batch: (inputs, targets) — will be moved to device
            criterion: loss function
        
        Returns:
            (grads, batch_loss, batch_correct, batch_total)
        """
        model.train()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        grads = []
        for param in model.parameters():
            if param.grad is None:
                grads.append(None)
            else:
                grads.append(param.grad.detach().clone())
        
        with torch.no_grad():
            _, predicted = outputs.max(1)
            batch_loss = loss.item()
            batch_correct = predicted.eq(targets).sum().item()
            batch_total = targets.size(0)
        
        return grads, batch_loss, batch_correct, batch_total
    
    def compute_full_grad(self, model, criterion_sum):
        """
        Compute full gradient over entire local dataset.
        
        Args:
            model: PyTorch model
            criterion_sum: loss function with reduction='sum'
        
        Returns:
            (grads, num_samples): List of gradient tensors and total sample count
        """
        model.train()
        for param in model.parameters():
            param.grad = None
        
        total_samples = 0
        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = criterion_sum(outputs, targets)
            loss.backward()
            total_samples += targets.size(0)
        
        # Clone gradients
        grads = []
        for param in model.parameters():
            if param.grad is None:
                grads.append(None)
            else:
                grads.append(param.grad.detach().clone())
        
        return grads, total_samples


class Server:
    """
    Server maintaining the global model and SARAH state.
    """
    
    def __init__(self, model, device, weight_decay=0.0, max_grad_norm=None):
        """
        Args:
            model: Global PyTorch model
            device: torch device
            weight_decay: L2 regularization coefficient
            max_grad_norm: gradient clipping threshold (None = no clipping)
        """
        self.model = model
        self.device = device
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # SARAH state
        self.v_t = None  # Recursive variance-reduced gradient
        self.s_t = None  # Momentum buffer
        
        # Previous model for computing grad_xprev
        self.model_prev = None
    
    def _apply_weight_decay(self, grads, model=None):
        """Add weight decay to gradients: grad + lambda * w."""
        if self.weight_decay == 0:
            return grads
        if model is None:
            model = self.model
        
        decayed = []
        for grad, param in zip(grads, model.parameters()):
            if grad is None:
                decayed.append(None)
            else:
                decayed.append(grad + self.weight_decay * param.detach())
        return decayed
    
    def _clip_grads(self, grads):
        """Clip gradients by global norm."""
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return grads
        
        total_norm_sq = torch.zeros(1, device=self.device)
        for grad in grads:
            if grad is None:
                continue
            total_norm_sq += grad.pow(2).sum()
        
        total_norm = total_norm_sq.sqrt()
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-6)
            clipped = []
            for grad in grads:
                if grad is None:
                    clipped.append(None)
                else:
                    clipped.append(grad * scale)
            return clipped
        return grads
    
    def _apply_update(self, grads, lr):
        """Apply gradient update: w = w - lr * grads."""
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue
                param.add_(grad, alpha=-lr)
    
    def initialize_sarah(self, full_grad_avg, lr):
        """
        Initialize SARAH state at the beginning of an epoch.
        
        Args:
            full_grad_avg: Averaged full gradient from all clients
            lr: Learning rate
        
        Formula:
            v_0 = full_grad_avg (with weight decay and clipping)
            s_0 = v_0
            x_1 = x_0 - lr * v_0
        """
        # Apply weight decay and clipping
        v_0 = self._apply_weight_decay(full_grad_avg)
        v_0 = self._clip_grads(v_0)
        
        # Initialize SARAH state
        self.v_t = [g.detach().clone() if g is not None else None for g in v_0]
        self.s_t = [g.detach().clone() if g is not None else None for g in v_0]
        
        # Store model_prev before first update
        self.model_prev = copy.deepcopy(self.model).to(self.device)
        
        # First step: x_1 = x_0 - lr * v_0
        self._apply_update(v_0, lr)
    
    def apply_sarah_step(self, grad_xt, grad_xprev, lr, beta):
        """
        Apply one SARAH step with momentum (variant 2).
        
        Args:
            grad_xt: Gradient at current model x_t on a batch
            grad_xprev: Gradient at previous model x_{t-1} on the same batch
            lr: Learning rate
            beta: Momentum coefficient
        
        Formula:
            v_t = grad_xt - grad_xprev + v_{t-1}
            s_t = beta * s_{t-1} + v_t
            x_{t+1} = x_t - lr * s_t
        """
        # Apply weight decay: grad_xt with current model, grad_xprev with previous model
        grad_xt = self._apply_weight_decay(grad_xt)
        grad_xprev = self._apply_weight_decay(grad_xprev, model=self.model_prev)
        
        # SARAH recursive update: v_t = grad_xt - grad_xprev + v_{t-1}
        v_t = []
        for g_cur, g_prev, v_old in zip(grad_xt, grad_xprev, self.v_t):
            if g_cur is None:
                v_t.append(None)
            else:
                v_t.append(g_cur - g_prev + v_old)
        
        # Clip v_t
        v_t = self._clip_grads(v_t)
        
        # Momentum update: s_t = beta * s_{t-1} + v_t
        if beta > 0:
            s_t = []
            for s_old, v in zip(self.s_t, v_t):
                if v is None:
                    s_t.append(None)
                else:
                    s_t.append((beta * s_old + v).detach().clone())
        else:
            s_t = [v.detach().clone() if v is not None else None for v in v_t]
        
        # Store current model as previous for next iteration
        self.model_prev.load_state_dict(self.model.state_dict())
        
        # Parameter update: x_{t+1} = x_t - lr * s_t
        self._apply_update(s_t, lr)
        
        # Update state
        self.v_t = v_t
        self.s_t = s_t
    
    def get_model(self):
        """Return current global model."""
        return self.model
    
    def get_prev_model(self):
        """Return previous model (for computing grad_xprev)."""
        return self.model_prev


def create_clients(dataset, num_clients, batch_size, device, num_workers=2, drop_last=True):
    """
    Split dataset into K clients with disjoint data subsets.
    
    Args:
        dataset: PyTorch Dataset
        num_clients: Number of clients (K)
        batch_size: Batch size for each client
        device: torch device
        num_workers: DataLoader num_workers
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        List of Client objects
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Split indices into K disjoint subsets
    client_size = total_size // num_clients
    clients = []
    
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]
        
        # Create subset and dataloader
        subset = torch.utils.data.Subset(dataset, client_indices)
        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        
        clients.append(Client(dataloader, device))
    
    return clients


def train_epoch_distributed(
    clients: List[Client],
    server: Server,
    criterion,
    criterion_sum,
    lr: float,
    momentum: float = 0.0
) -> Tuple[float, float]:
    """
    Train one epoch using distributed SARAH with K clients.
    
    Args:
        clients: List of K Client objects
        server: Server object with global model
        criterion: Loss function (mean reduction)
        criterion_sum: Loss function (sum reduction)
        lr: Learning rate
        momentum: Momentum coefficient (beta)
    
    Returns:
        (train_loss, train_acc): Average loss and accuracy over all batches
    """
    K = len(clients)
    model = server.get_model()
    model.train()
    
    # ==========================================
    # Step 1: Distributed full gradient computation
    # ==========================================
    print("Computing distributed full gradient...")
    full_grads_list = []
    total_samples = 0
    
    for client in clients:
        grads, num_samples = client.compute_full_grad(model, criterion_sum)
        full_grads_list.append(grads)
        total_samples += num_samples
    
    # Aggregate by averaging: v_0 = (1/K) * sum_i grad_i / n_i, normalized by total samples
    # Each client returns unnormalized sum, so we sum all and divide by total_samples
    full_grad_avg = []
    for param_idx in range(len(full_grads_list[0])):
        if full_grads_list[0][param_idx] is None:
            full_grad_avg.append(None)
        else:
            summed = sum(client_grads[param_idx] for client_grads in full_grads_list)
            full_grad_avg.append(summed / total_samples)
    
    # ==========================================
    # Step 2: Initialize SARAH (v_0, s_0, first step)
    # ==========================================
    server.initialize_sarah(full_grad_avg, lr)
    
    # ==========================================
    # Step 3: Random permutation of clients
    # ==========================================
    client_order = list(range(K))
    random.shuffle(client_order)
    
    # ==========================================
    # Step 4: Iterate over clients in random order
    # ==========================================
    running_loss = 0.0
    correct = 0
    total = 0
    
    for t, client_idx in enumerate(client_order):
        client = clients[client_idx]
        batch = client.sample_batch()
        
        # Compute grad_xt at current model (also returns metrics BEFORE update)
        grad_xt, batch_loss, batch_correct, batch_total = client.compute_grad(
            model, batch, criterion
        )
        
        # Compute grad_xprev at previous model
        model_prev = server.get_prev_model()
        grad_xprev, _, _, _ = client.compute_grad(model_prev, batch, criterion)
        
        # Collect metrics (at x_t, before the step — matches centralized version)
        running_loss += batch_loss * batch_total
        correct += batch_correct
        total += batch_total
        
        # Server applies SARAH step: x_{t+1} = x_t - lr * s_t
        server.apply_sarah_step(grad_xt, grad_xprev, lr, momentum)
    
    train_loss = running_loss / total if total > 0 else 0.0
    train_acc = 100.0 * correct / total if total > 0 else 0.0
    
    return train_loss, train_acc


# ==========================================
# Example integration into main training loop
# ==========================================

def train_loop_distributed(
    train_dataset,
    eval_loader,
    model,
    criterion,
    criterion_sum,
    params,
    total_epochs,
    num_clients,
    device,
    eval_name="val",
    experiment=None,
    trial=None,
):
    """
    Distributed training loop for multiple epochs.
    
    Args:
        train_dataset: Full training dataset (will be split into clients)
        eval_loader: Validation/test loader
        model: PyTorch model
        criterion: Loss function (mean reduction)
        criterion_sum: Loss function (sum reduction)
        params: Dict with hyperparameters (sarah_lr, weight_decay, max_grad_norm, 
                batch_size, momentum, warmup_epochs, min_lr)
        total_epochs: Number of epochs
        num_clients: K (number of clients)
        device: torch device
        eval_name: Name for logging ("val" or "test")
        experiment: Comet ML experiment (optional)
        trial: Optuna trial (optional)
    
    Returns:
        best_eval_acc: Best evaluation accuracy
    """
    from main import get_lr, evaluate, log_metrics
    import optuna
    
    # Create server
    server = Server(
        model=model,
        device=device,
        weight_decay=params["weight_decay"],
        max_grad_norm=params["max_grad_norm"],
    )
    
    # Create clients once before all epochs
    clients = create_clients(
        train_dataset,
        num_clients=num_clients,
        batch_size=params["batch_size"],
        device=device,
        num_workers=2,
        drop_last=True,
    )
    
    best_eval_acc = 0.0
    eval_label = eval_name.capitalize() if eval_name else "Eval"
    
    for epoch in range(1, total_epochs + 1):
        # Learning rate schedule
        lr = get_lr(
            epoch,
            total_epochs,
            params["warmup_epochs"],
            params["sarah_lr"],
            params["min_lr"],
        )
        
        # Distributed training epoch (using existing clients)
        train_loss, train_acc = train_epoch_distributed(
            clients=clients,
            server=server,
            criterion=criterion,
            criterion_sum=criterion_sum,
            lr=lr,
            momentum=params.get("momentum", 0.0),
        )
        
        # Evaluation
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
        
        # Logging
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
