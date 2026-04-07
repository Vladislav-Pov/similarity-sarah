"""Parameter manipulation and gradient computation utilities."""

from __future__ import annotations

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ParamList = list[torch.Tensor]


def set_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_params(model: nn.Module) -> ParamList:
    """Return a detached clone of all model parameters."""
    return [p.data.clone() for p in model.parameters()]


def set_params(model: nn.Module, params: ParamList) -> None:
    """Set model parameters from a list of tensors (in-place copy)."""
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def zeros_like_params(reference: Union[nn.Module, ParamList]) -> ParamList:
    """Create zero-valued parameter list matching the reference shape."""
    if isinstance(reference, nn.Module):
        return [torch.zeros_like(p) for p in reference.parameters()]
    return [torch.zeros_like(p) for p in reference]


def clone_params(params: ParamList) -> ParamList:
    """Clone a parameter list."""
    return [p.clone() for p in params]


def add_params_(target: ParamList, source: ParamList, alpha: float = 1.0) -> None:
    """In-place: target += alpha * source."""
    for t, s in zip(target, source):
        t.add_(s, alpha=alpha)


def scale_params_(params: ParamList, alpha: float) -> None:
    """In-place: params *= alpha."""
    for p in params:
        p.mul_(alpha)


def compute_param_norm_sq(params: ParamList) -> float:
    """Compute squared L2 norm of parameter list."""
    return sum(p.square().sum().item() for p in params)


def compute_full_gradient(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> ParamList:
    """Compute the exact average gradient of *loss_fn* over the entire dataset.

    The model must already have its parameters set to the desired evaluation
    point before calling this function.  No augmentation should be used in
    *loader* to guarantee deterministic gradients.
    """
    params = list(model.parameters())
    total_grad: list[torch.Tensor] = [torch.zeros_like(p) for p in params]
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        grads = torch.autograd.grad(loss, params)
        bs = x.size(0)
        for tg, g in zip(total_grad, grads):
            tg.add_(g, alpha=bs)
        total_samples += bs

    for tg in total_grad:
        tg.div_(total_samples)

    return total_grad
