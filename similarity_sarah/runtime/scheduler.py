"""Client batch scheduler: permutation-based non-overlapping batches."""

from __future__ import annotations

import torch


def sample_client_batches(
    num_clients: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Return non-overlapping batches of client indices for one epoch.

    A fresh random permutation of ``[0, num_clients)`` is split into consecutive
    chunks of size ``batch_size``; the last chunk may be smaller.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    perm = torch.randperm(num_clients, generator=generator).tolist()
    return [perm[start : start + batch_size]
            for start in range(0, num_clients, batch_size)]
