"""Proximal operator solvers for the server's local objective.

The interface ``ProxSolver.step`` computes (approximately)

    w_new = prox_{θ·f₁}(w − θ·v)  =  argmin_w { f₁(w) + 1/(2θ)‖w − z‖² }

where z = w_current − θ·v.  Subclass ``ProxSolver`` to plug in a different
solver (e.g. L-BFGS, exact solver for quadratics, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.utils import ParamList, get_params, set_params


class ProxSolver(ABC):
    """Abstract base for proximal-operator solvers."""

    @abstractmethod
    def step(
        self,
        model: nn.Module,
        v: ParamList,
        theta: float,
        server_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        """Compute w ≈ prox_{θf₁}(w − θv) and update *model* in-place."""
        ...


class InexactProxSGD(ProxSolver):
    """Approximate the proximal operator with several SGD steps.

    Starting from z = w − θv, run *num_steps* stochastic gradient steps on
    the proximal objective  f₁(w) + 1/(2θ)‖w − z‖²  using mini-batches from
    the server's data loader.
    """

    def __init__(self, num_steps: int, lr: float) -> None:
        self.num_steps = num_steps
        self.lr = lr

    def step(
        self,
        model: nn.Module,
        v: ParamList,
        theta: float,
        server_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        w = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w, v)]
        set_params(model, z)

        server_iter = iter(server_loader)
        for _ in range(self.num_steps):
            try:
                x, y = next(server_iter)
            except StopIteration:
                server_iter = iter(server_loader)
                x, y = next(server_iter)

            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            grads = torch.autograd.grad(loss, list(model.parameters()))

            with torch.no_grad():
                for p, g, zi in zip(model.parameters(), grads, z):
                    p.data.sub_(self.lr * (g + (p.data - zi) / theta))
