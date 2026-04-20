"""Proximal operator solvers for the server's local objective.

The interface ``ProxSolver.step`` computes (approximately)

    w_new = prox_{θ·f₁}(w − θ·v)
          = argmin_w { f₁(w) + 1/(2θ) · ‖w − z‖² }

where ``z = w_current − θ · v``.  Subclass :class:`ProxSolver` to plug in a
different solver (L-BFGS, an exact solver for quadratics, an Adam-based
inexact prox, etc.).

Every concrete solver returns a small dictionary of *diagnostics* (norm of
the prox subproblem gradient at the first / last inner step, mean prox
loss, …).  The runner forwards them to the metrics logger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

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
    ) -> Mapping[str, float]:
        """Approximately solve ``w ≈ prox_{θ f₁}(w − θ v)`` and update *model*.

        Returns:
            Diagnostics dictionary (e.g. gradient norms of the prox
            subproblem at the first/last inner step).
        """


def _prox_grad_norm(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    z: ParamList,
    theta: float,
) -> float:
    """L2 norm of  ∇f₁(w) + (w − z)/θ   (gradient of the prox objective)."""
    s = 0.0
    for p, g, zi in zip(params, grads, z):
        delta = g + (p.data - zi) / theta
        s += delta.square().sum().item()
    return float(s ** 0.5)


class InexactProxSGD(ProxSolver):
    """Approximate the proximal operator with several SGD steps.

    Starting from ``z = w − θ v``, run ``num_steps`` mini-batch SGD steps
    on the proximal objective

        f₁(w) + 1/(2θ) · ‖w − z‖² + (weight_decay / 2) · ‖w‖²

    using mini-batches drawn from the server data loader.

    Optional Polyak-momentum and weight-decay arguments make the inexact
    prox much more effective for deep networks (e.g. ResNet) where pure
    vanilla SGD converges very slowly.
    """

    def __init__(
        self,
        num_steps: int,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.num_steps = num_steps
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(
        self,
        model: nn.Module,
        v: ParamList,
        theta: float,
        server_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Mapping[str, float]:
        w = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w, v)]
        set_params(model, z)

        params = list(model.parameters())
        velocity: list[torch.Tensor] | None = None
        if self.momentum > 0:
            velocity = [torch.zeros_like(p) for p in params]

        first_norm: float | None = None
        last_norm: float | None = None
        loss_sum = 0.0
        loss_count = 0

        server_iter = iter(server_loader)
        for it in range(self.num_steps):
            try:
                x, y = next(server_iter)
            except StopIteration:
                server_iter = iter(server_loader)
                x, y = next(server_iter)

            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            grads = torch.autograd.grad(loss, params)

            grad_norm = _prox_grad_norm(params, list(grads), z, theta)
            if first_norm is None:
                first_norm = grad_norm
            last_norm = grad_norm
            loss_sum += float(loss.detach().item())
            loss_count += 1

            with torch.no_grad():
                for i, (p, g, zi) in enumerate(zip(params, grads, z)):
                    direction = g + (p.data - zi) / theta
                    if self.weight_decay > 0:
                        direction = direction + self.weight_decay * p.data
                    if velocity is not None:
                        velocity[i].mul_(self.momentum).add_(direction)
                        p.data.sub_(velocity[i], alpha=self.lr)
                    else:
                        p.data.sub_(direction, alpha=self.lr)

        return {
            "prox_grad_norm_first": float(first_norm) if first_norm is not None else 0.0,
            "prox_grad_norm_last": float(last_norm) if last_norm is not None else 0.0,
            "prox_loss_mean": loss_sum / max(loss_count, 1),
            "prox_inner_steps": float(self.num_steps),
        }


class InexactProxAdam(ProxSolver):
    """Adam-based inexact prox solver.

    Often noticeably more robust than vanilla SGD for non-convex deep models.
    """

    def __init__(
        self,
        num_steps: int,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.num_steps = num_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(
        self,
        model: nn.Module,
        v: ParamList,
        theta: float,
        server_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Mapping[str, float]:
        w = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w, v)]
        set_params(model, z)

        params = list(model.parameters())
        # Per-call Adam state (ephemeral inside one prox call).
        m_state = [torch.zeros_like(p) for p in params]
        v_state = [torch.zeros_like(p) for p in params]
        b1, b2 = self.betas

        first_norm: float | None = None
        last_norm: float | None = None
        loss_sum = 0.0
        loss_count = 0

        server_iter = iter(server_loader)
        for t in range(1, self.num_steps + 1):
            try:
                x, y = next(server_iter)
            except StopIteration:
                server_iter = iter(server_loader)
                x, y = next(server_iter)

            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            grads = torch.autograd.grad(loss, params)

            grad_norm = _prox_grad_norm(params, list(grads), z, theta)
            if first_norm is None:
                first_norm = grad_norm
            last_norm = grad_norm
            loss_sum += float(loss.detach().item())
            loss_count += 1

            bias1 = 1 - b1 ** t
            bias2 = 1 - b2 ** t

            with torch.no_grad():
                for i, (p, g, zi) in enumerate(zip(params, grads, z)):
                    direction = g + (p.data - zi) / theta
                    if self.weight_decay > 0:
                        direction = direction + self.weight_decay * p.data
                    m_state[i].mul_(b1).add_(direction, alpha=1 - b1)
                    v_state[i].mul_(b2).addcmul_(direction, direction, value=1 - b2)
                    m_hat = m_state[i] / bias1
                    v_hat = v_state[i] / bias2
                    p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

        return {
            "prox_grad_norm_first": float(first_norm) if first_norm is not None else 0.0,
            "prox_grad_norm_last": float(last_norm) if last_norm is not None else 0.0,
            "prox_loss_mean": loss_sum / max(loss_count, 1),
            "prox_inner_steps": float(self.num_steps),
        }
