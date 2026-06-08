"""Shared diagnostics and clipping helpers for the proximal solvers.

The proximal subproblem solved (approximately) by every solver is

    w_new ~= prox_{theta f1}(w_outer - theta v)
          = argmin_w  f1(w) + 1/(2 theta) ||w - z||^2,   z := w_outer - theta v,

with proximal gradient ``grad Phi(w) = grad f1(w) + (w - z)/theta``. These
helpers compute the convergence diagnostics on a fixed eval batch and the
direction-clipping variants the SGD/Adam/AccVRS solvers share.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.core.params import ParamList


def alpha_at(t: int, num_steps: int, schedule: str) -> float:
    """Coefficient in front of ``v`` inside the inner-step direction.

    ``constant`` always returns ``1.0`` (canonical prox). ``linear`` returns
    ``1 - t/(num_steps - 1)`` so it decays 1 -> 0 across the inner loop.
    """
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        denom = max(num_steps - 1, 1)
        return 1.0 - float(t) / float(denom)
    raise ValueError(f"Unknown v_schedule: {schedule!r}")


def prox_grad_norm(
    params: list[torch.Tensor], grads: list[torch.Tensor], z: ParamList, theta: float
) -> float:
    """L2 norm of ``grad f1(w) + (w - z)/theta`` (gradient of the prox objective)."""
    s = 0.0
    for p, g, zi in zip(params, grads, z):
        delta = g + (p.data - zi) / theta
        s += delta.square().sum().item()
    return float(s**0.5)


def diag_on_batches(
    params: list[torch.Tensor],
    z: ParamList,
    theta: float,
    model: nn.Module,
    loss_fn: nn.Module,
    xys: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(||grad Phi(w)||, Phi(w))`` on a fixed list of minibatches.

    Gradients of ``f1`` are accumulated sample-weighted across ``xys`` before
    the norm is taken, giving a lower-variance estimate than a single batch.
    """
    accum_grad: list[torch.Tensor] = [torch.zeros_like(p) for p in params]
    loss_sum_weighted = 0.0
    total_count = 0
    for x, y in xys:
        x, y = x.to(device), y.to(device)
        bs = x.size(0)
        output = model(x)
        loss = loss_fn(output, y)
        grads = torch.autograd.grad(loss, params)
        for a, g in zip(accum_grad, grads):
            a.add_(g, alpha=bs)
        loss_sum_weighted += float(loss.detach().item()) * bs
        total_count += bs

    if total_count == 0:
        raise ValueError("xys is empty - at least one eval batch is required")

    for a in accum_grad:
        a.div_(total_count)
    avg_loss = loss_sum_weighted / total_count

    grad_norm = prox_grad_norm(params, accum_grad, z, theta)
    penalty_sq = 0.0
    for p, zi in zip(params, z):
        penalty_sq += (p.data - zi).square().sum().item()
    obj = avg_loss + 0.5 * penalty_sq / max(theta, 1e-12)
    return grad_norm, obj


def fresh_eval_xys(
    eval_loader: DataLoader | None, server_loader: DataLoader, n_batches: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-materialise ``n_batches`` eval minibatches (reused at start and end)."""
    if n_batches < 1:
        raise ValueError(f"n_batches must be >= 1, got {n_batches}")
    loader = eval_loader if eval_loader is not None else server_loader
    it = iter(loader)
    xys: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_batches):
        try:
            xys.append(next(it))
        except StopIteration:
            it = iter(loader)
            xys.append(next(it))
    return xys


def build_diag_payload(
    first_grad_norm: float,
    first_obj: float,
    last_grad_norm: float,
    last_obj: float,
    num_steps: int,
    clip_count: int = 0,
) -> dict[str, float]:
    """Assemble the standard prox diagnostics dictionary."""
    return {
        "prox_grad_norm_first": float(first_grad_norm),
        "prox_grad_norm_last": float(last_grad_norm),
        "prox_grad_norm_ratio": float(last_grad_norm / max(first_grad_norm, 1e-12)),
        "prox_obj_decrease": float(first_obj - last_obj),
        "prox_inner_steps": float(num_steps),
        "prox_clip_frac": float(clip_count) / float(max(num_steps, 1)),
    }


def clip_l2_norm_(directions: list[torch.Tensor], max_norm: float) -> bool:
    """In-place global L2-norm clip across the direction list (InexactProx)."""
    if max_norm <= 0:
        return False
    total_sq = 0.0
    for d in directions:
        total_sq += d.square().sum().item()
    total = total_sq**0.5
    if total > max_norm:
        scale = max_norm / max(total, 1e-12)
        for d in directions:
            d.mul_(scale)
        return True
    return False


def clip_per_element_(directions: list[torch.Tensor], clip_value: float) -> bool:
    """In-place per-element clamp of each direction tensor (AccVRS)."""
    if clip_value <= 0:
        return False
    fired = False
    for d in directions:
        if not fired and d.abs().max().item() > clip_value:
            fired = True
        d.clamp_(-clip_value, clip_value)
    return fired


def direction_norm(directions: list[torch.Tensor]) -> float:
    """Global L2 norm across a direction list (no allocation)."""
    s = 0.0
    for d in directions:
        s += d.square().sum().item()
    return float(s**0.5)
