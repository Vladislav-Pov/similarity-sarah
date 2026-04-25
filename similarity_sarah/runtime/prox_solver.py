"""Proximal operator solvers for the server's local objective.

The interface ``ProxSolver.step`` computes (approximately)

    w_new ≈ prox_{θ·f₁}(w_outer − θ·v)
         = argmin_w { ⟨v, w⟩ + 1/(2θ)·‖w − w_outer‖² + f₁(w) }
         = argmin_w { f₁(w) + 1/(2θ)·‖w − z‖² },   z := w_outer − θ·v.

The two ``argmin`` formulations are equivalent (complete the square in the
linear ``⟨v, w⟩`` term).  Inexact solvers run a few SGD/Adam steps on the
proximal gradient

    ∇Φ(w) = ∇f₁(w) + (w − z)/θ
         = ∇f₁(w) + v + (w − w_outer)/θ,

starting from ``w = w_outer`` (the current outer iterate).  The prox can
therefore only improve the model w.r.t. Φ — by construction we never end up
arbitrarily worse than where we entered.  Subclass :class:`ProxSolver` to
plug in a different solver (L-BFGS, an exact solver for quadratics, an
Adam-based inexact prox, etc.).

Every concrete solver returns a small dictionary of *diagnostics* tuned
for "did the prox subproblem actually get solved":

    prox_grad_norm_first   ‖∇Φ(w_0_prox)‖  on a fixed eval batch
    prox_grad_norm_last    ‖∇Φ(w_final)‖   on the same fixed eval batch
    prox_grad_norm_ratio   last / max(first, 1e-12)     — close to 0 ⇔ well-solved
    prox_obj_decrease      Φ(w_0_prox) − Φ(w_final)     — should be positive
    prox_inner_steps       configured ``num_steps``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.utils import ParamList, get_params


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
        eval_loader: DataLoader | None = None,
    ) -> Mapping[str, float]:
        """Approximately solve ``w ≈ prox_{θ f₁}(w − θ v)`` and update *model*.

        ``server_loader`` is iterated for the inner prox training steps (may be
        augmented / stochastic).  ``eval_loader`` — if given — provides the
        fixed deterministic minibatch used for the diagnostics; otherwise a
        batch is drawn from ``server_loader``.

        Returns:
            Diagnostics dictionary — see module docstring for keys.
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


def _prox_diag_on_batch(
    params: list[torch.Tensor],
    z: ParamList,
    theta: float,
    model: nn.Module,
    loss_fn: nn.Module,
    xy: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    """Return (‖∇Φ(w)‖, Φ(w)) on a fixed minibatch.

    The proximal objective is

        Φ(w) = f₁(w) + ‖w − z‖² / (2θ),

    where ``w`` is the current model state and ``z = w_outer − θ·v`` is the
    fixed target point of the prox subproblem.
    """
    x, y = xy
    x, y = x.to(device), y.to(device)
    output = model(x)
    loss = loss_fn(output, y)
    grads = torch.autograd.grad(loss, params)
    grad_norm = _prox_grad_norm(params, list(grads), z, theta)
    penalty_sq = 0.0
    for p, zi in zip(params, z):
        penalty_sq += (p.data - zi).square().sum().item()
    obj = float(loss.detach().item()) + 0.5 * penalty_sq / max(theta, 1e-12)
    return grad_norm, obj


def _prox_z_dist_rel(
    params: list[torch.Tensor],
    z: ParamList,
    theta: float,
    v_norm: float,
) -> float:
    s = 0.0
    for p, zi in zip(params, z):
        s += (p.data - zi).square().sum().item()
    return (s ** 0.5) / max(theta * v_norm, 1e-12)


def _fresh_eval_xy(
    eval_loader: DataLoader | None,
    server_loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = eval_loader if eval_loader is not None else server_loader
    return next(iter(loader))


def _build_diag_payload(
    first_grad_norm: float,
    first_obj: float,
    last_grad_norm: float,
    last_obj: float,
    num_steps: int,
) -> dict[str, float]:
    return {
        "prox_grad_norm_first": float(first_grad_norm),
        "prox_grad_norm_last": float(last_grad_norm),
        "prox_grad_norm_ratio": float(last_grad_norm / max(first_grad_norm, 1e-12)),
        "prox_obj_decrease": float(first_obj - last_obj),
        "prox_inner_steps": float(num_steps),
    }


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
        eval_loader: DataLoader | None = None,
    ) -> Mapping[str, float]:
        # Capture the outer iterate w_outer (the current model state) and the
        # prox target z = w_outer − θ·v.  We do NOT move the model: SGD warm-
        # starts from w_outer so the prox improves the iterate from the place
        # the outer algorithm just visited (safer than starting at z when the
        # inner solver only runs a few inexact steps).
        w_outer = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w_outer, v)]

        params = list(model.parameters())
        velocity: list[torch.Tensor] | None = None
        if self.momentum > 0:
            velocity = [torch.zeros_like(p) for p in params]

        # Fixed eval batch for diagnostics (sampled once, reused at start + end).
        eval_xy = _fresh_eval_xy(eval_loader, server_loader)
        first_grad_norm, first_obj = _prox_diag_on_batch(
            params, z, theta, model, loss_fn, eval_xy, device,
        )

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
            grads = torch.autograd.grad(loss, params)

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

        last_grad_norm, last_obj = _prox_diag_on_batch(
            params, z, theta, model, loss_fn, eval_xy, device,
        )
        return _build_diag_payload(
            first_grad_norm, first_obj,
            last_grad_norm, last_obj,
            self.num_steps,
        )


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
        eval_loader: DataLoader | None = None,
    ) -> Mapping[str, float]:
        # Warm-start at w_outer (model is already there); see InexactProxSGD.
        w_outer = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w_outer, v)]

        params = list(model.parameters())
        m_state = [torch.zeros_like(p) for p in params]
        v_state = [torch.zeros_like(p) for p in params]
        b1, b2 = self.betas

        eval_xy = _fresh_eval_xy(eval_loader, server_loader)
        first_grad_norm, first_obj = _prox_diag_on_batch(
            params, z, theta, model, loss_fn, eval_xy, device,
        )

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

        last_grad_norm, last_obj = _prox_diag_on_batch(
            params, z, theta, model, loss_fn, eval_xy, device,
        )
        return _build_diag_payload(
            first_grad_norm, first_obj,
            last_grad_norm, last_obj,
            self.num_steps,
        )
