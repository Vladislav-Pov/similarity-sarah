"""Data partitioning strategies for server–client distribution."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset


def create_partition(
    dataset: Dataset,
    num_partitions: int,
    cfg: DictConfig,
) -> list[Subset]:
    """Partition *dataset* into *num_partitions* non-overlapping subsets.

    The first partition is assigned to the server; the remaining partitions
    are assigned to clients 0 … num_clients-1.

    Returns:
        List of ``Subset`` objects, one per partition.
    """
    if cfg.name == "uniform":
        return _uniform_partition(dataset, num_partitions)
    raise ValueError(f"Unknown partition strategy: {cfg.name}")


def _uniform_partition(dataset: Dataset, num_partitions: int) -> list[Subset]:
    """Random uniform non-overlapping partition."""
    n = len(dataset)  # type: ignore[arg-type]
    indices = torch.randperm(n).tolist()

    base_size = n // num_partitions
    remainder = n % num_partitions

    partitions: list[Subset] = []
    offset = 0
    for i in range(num_partitions):
        size = base_size + (1 if i < remainder else 0)
        partitions.append(Subset(dataset, indices[offset : offset + size]))
        offset += size

    return partitions
