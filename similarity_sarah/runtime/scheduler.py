"""Client batch scheduler: permutation-based non-overlapping batches."""

from __future__ import annotations

import torch


class ClientBatchScheduler:
    """Generate non-overlapping client batches per epoch.

    For each epoch a fresh random permutation of client indices
    ``[0, num_clients)`` is drawn and split into consecutive batches of size
    *batch_size*.  The last batch may be smaller.

    When ``batch_size == 1`` this reduces to sampling without replacement
    (a random permutation of single-element batches).
    """

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}")

        self.num_clients = num_clients
        self.batch_size = batch_size
        self.generator = generator

    def get_epoch_batches(self) -> list[list[int]]:
        """Return a list of non-overlapping client-index batches for one epoch."""
        perm = torch.randperm(self.num_clients, generator=self.generator).tolist()
        batches: list[list[int]] = []
        for start in range(0, self.num_clients, self.batch_size):
            batches.append(perm[start : start + self.batch_size])
        return batches
