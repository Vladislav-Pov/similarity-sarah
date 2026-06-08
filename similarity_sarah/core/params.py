"""Model <-> parameter-list bridging and norm reductions.

A :data:`ParamList` is a plain ``list[torch.Tensor]`` snapshot of a model's
parameters, the currency the distributed algorithms pass around (estimators
``v``/``tilde_v``, iterate copies ``w_t``/``w_{t-1}``). In-place arithmetic is
delegated to the bit-safe fused ops in :mod:`similarity_sarah.core.foreach`;
the norm reductions keep an explicit per-tensor ``.item()`` summation so their
value is bit-identical to the pre-rewrite ``utils`` implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from similarity_sarah.core import foreach

ParamList = list[torch.Tensor]


def get_params(model: nn.Module) -> ParamList:
    """Return a detached clone of all model parameters."""
    return [p.data.clone() for p in model.parameters()]


def set_params(model: nn.Module, params: ParamList) -> None:
    """Copy ``params`` into the model's parameters in place."""
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def zeros_like_params(reference: nn.Module | ParamList) -> ParamList:
    """Zero-valued parameter list matching ``reference``'s shapes."""
    if isinstance(reference, nn.Module):
        return [torch.zeros_like(p) for p in reference.parameters()]
    return [torch.zeros_like(p) for p in reference]


def clone_params(params: ParamList) -> ParamList:
    """Clone a parameter list."""
    return [p.clone() for p in params]


def add_params_(target: ParamList, source: ParamList, alpha: float = 1.0) -> None:
    """In-place ``target += alpha * source`` (fused)."""
    foreach.add_(target, source, alpha=alpha)


def scale_params_(params: ParamList, alpha: float) -> None:
    """In-place ``params *= alpha`` (fused)."""
    foreach.scale_(params, alpha)


def compute_param_norm_sq(params: ParamList) -> float:
    """Squared L2 norm of a parameter list (per-tensor summation)."""
    return sum(p.square().sum().item() for p in params)


def compute_param_norm(params: ParamList) -> float:
    """L2 norm of a parameter list."""
    return float(compute_param_norm_sq(params) ** 0.5)


def diff_param_norm(a: ParamList, b: ParamList) -> float:
    """L2 norm of ``a - b``."""
    s = 0.0
    for ai, bi in zip(a, b):
        s += (ai - bi).square().sum().item()
    return float(s**0.5)
