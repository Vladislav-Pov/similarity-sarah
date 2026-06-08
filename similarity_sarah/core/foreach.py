"""Bit-safe fused tensor-list arithmetic via ``torch._foreach_*``.

These wrappers replace the per-tensor Python ``zip`` loops on the hot path.
They act elementwise over independent tensors (no cross-tensor reduction), so
each result is bit-identical to the equivalent per-tensor loop — verified in
``tests/test_foreach.py``. Reductions whose result depends on summation order
(the norms) deliberately stay in :mod:`similarity_sarah.core.params`, not here.
"""

from __future__ import annotations

import torch

TensorList = list[torch.Tensor]


def add_(dst: TensorList, src: TensorList, *, alpha: float = 1.0) -> None:
    """In-place ``dst[i] += alpha * src[i]`` for every ``i``."""
    if not dst:
        return
    torch._foreach_add_(dst, src, alpha=alpha)


def scale_(params: TensorList, factor: float) -> None:
    """In-place ``params[i] *= factor`` for every ``i``."""
    if not params:
        return
    torch._foreach_mul_(params, factor)


def sub(a: TensorList, b: TensorList) -> TensorList:
    """Out-of-place ``[a[i] - b[i] for i]``."""
    if not a:
        return []
    return list(torch._foreach_sub(a, b))
