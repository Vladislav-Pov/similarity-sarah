"""Data partitioning strategies for server–client distribution."""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


def create_partition(
    dataset: Dataset,
    num_partitions: int,
    cfg: DictConfig,
) -> list[Subset]:
    """Partition *dataset* into *num_partitions* non-overlapping subsets.

    The first partition is assigned to the server; the remaining partitions
    are assigned to clients 0 … num_clients-1.

    For ``cfg.name == "uniform"``:
        * ``cfg.server_fraction`` (optional, in (0, 1)) — share of the data
          handed to the server's partition; the rest is split equally across
          the remaining ``num_partitions - 1`` clients.
        * If ``server_fraction`` is missing or ``null``, fall back to the
          fully-equal split.

    Returns:
        List of ``Subset`` objects, one per partition.  ``partitions[0]`` is
        the server, ``partitions[1:]`` are clients.
    """
    if cfg.name == "uniform":
        server_fraction = OmegaConf.select(cfg, "server_fraction", default=None)
        if server_fraction is None:
            return _uniform_partition(dataset, num_partitions)
        return _server_weighted_partition(
            dataset, num_partitions, float(server_fraction),
        )
    raise ValueError(f"Unknown partition strategy: {cfg.name}")


def _uniform_partition(dataset: Dataset, num_partitions: int) -> list[Subset]:
    """Random uniform non-overlapping partition (every partition same size)."""
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

    logger.info(
        "uniform partition: %d partitions, sizes=%s",
        num_partitions, [len(p) for p in partitions],
    )
    return partitions


def _server_weighted_partition(
    dataset: Dataset,
    num_partitions: int,
    server_fraction: float,
) -> list[Subset]:
    """Random non-overlapping partition with the server holding a heavier share.

    The server (``partitions[0]``) gets ``round(n * server_fraction)`` samples;
    the remaining ``n - server_size`` samples are split as equally as possible
    across the ``num_partitions - 1`` clients.
    """
    if num_partitions < 2:
        raise ValueError(
            f"server-weighted partition requires >= 2 partitions, got {num_partitions}",
        )
    if not 0.0 < server_fraction < 1.0:
        raise ValueError(
            f"server_fraction must lie in (0, 1), got {server_fraction}",
        )

    n = len(dataset)  # type: ignore[arg-type]
    indices = torch.randperm(n).tolist()

    server_size = int(round(n * server_fraction))
    # Keep at least one sample for every client.
    num_clients = num_partitions - 1
    if server_size > n - num_clients:
        server_size = n - num_clients
    if server_size < 1:
        server_size = 1

    remainder = n - server_size
    base_size = remainder // num_clients
    extra = remainder % num_clients

    partitions: list[Subset] = []
    partitions.append(Subset(dataset, indices[:server_size]))

    offset = server_size
    for i in range(num_clients):
        size = base_size + (1 if i < extra else 0)
        partitions.append(Subset(dataset, indices[offset : offset + size]))
        offset += size

    logger.info(
        "server-weighted partition: server=%d (%.1f%%), %d clients of size %d (+%d extra)",
        server_size, 100.0 * server_size / n,
        num_clients, base_size, extra,
    )
    return partitions
