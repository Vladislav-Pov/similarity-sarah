"""Gradient computation over data loaders.

Two primitives the distributed algorithms rely on:

* :func:`compute_batch_gradient` — mean gradient on a single minibatch. Pass a
  pre-sampled ``xy`` so two evaluations at different parameters share the same
  samples; this is what makes the SARAH telescope actually telescope (the
  "same minibatch at ``w_t`` and ``w_{t-1}``" invariant).
* :func:`compute_full_gradient` — exact average gradient over a whole loader,
  used by SVRS's anchor refresh.

Ported verbatim (bit-for-bit) from the pre-rewrite ``utils`` module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.core.params import ParamList


def compute_full_gradient(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> ParamList:
    """Exact average gradient of ``loss_fn`` over the entire ``loader``.

    The model must already sit at the desired evaluation point. ``loader`` must
    be deterministic (no augmentation) for the gradient to be well defined.
    """
    params = list(model.parameters())
    total_grad: ParamList = [torch.zeros_like(p) for p in params]
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


def compute_batch_gradient(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    xy: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> ParamList:
    """Mean gradient of ``loss_fn`` over a single minibatch.

    If ``xy`` is ``None`` the first batch of ``iter(loader)`` is used (a random
    minibatch when ``shuffle=True``). Otherwise the gradient is taken on the
    given ``(x, y)`` — pass this when two evaluations at different parameters
    must share the same samples (gradient differences).
    """
    params = list(model.parameters())
    if xy is None:
        x, y = next(iter(loader))
    else:
        x, y = xy
    x, y = x.to(device), y.to(device)
    output = model(x)
    loss = loss_fn(output, y)
    return list(torch.autograd.grad(loss, params))
