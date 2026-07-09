"""Data partitioning strategies for server–client distribution."""

from __future__ import annotations

import logging

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset, TensorDataset

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

    For ``cfg.name == "dirichlet"`` — label-heterogeneous (non-IID) clients:
        * ``cfg.alpha`` — Dirichlet concentration.  Small α (≈0.1) → highly
          skewed clients (each holds few classes); large α (≈100) → nearly
          uniform / IID clients.
        * ``cfg.server_fraction`` (optional, in (0, 1)) — IID share handed to
          the server before the remaining pool is split across clients via a
          Dirichlet draw.  ``null`` → server gets an equal ``1/num_partitions``
          IID slice.
        * ``cfg.seed`` (optional) — overrides the seed used for the Dirichlet
          draw / shuffles; ``null`` derives it from the global RNG (which is
          already seeded from ``cfg.seed`` at experiment start), so runs stay
          reproducible.

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
    if cfg.name == "dirichlet":
        alpha = float(cfg.alpha)
        server_fraction = OmegaConf.select(cfg, "server_fraction", default=None)
        seed = OmegaConf.select(cfg, "seed", default=None)
        return _dirichlet_partition(
            dataset,
            num_partitions,
            alpha=alpha,
            server_fraction=None if server_fraction is None else float(server_fraction),
            seed=None if seed is None else int(seed),
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


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    """Return an integer label per sample of *dataset* (in dataset order).

    Walks nested ``Subset`` wrappers down to a root that exposes labels
    cheaply (torchvision ``.targets`` or ``TensorDataset.tensors[1]``) to
    avoid materialising / transforming every image.  Falls back to
    ``dataset[i][1]`` only when no cheap path exists.
    """
    # Unwrap Subset chain, composing index maps outward → inward.
    chains: list[list[int]] = []
    root: Dataset = dataset
    while isinstance(root, Subset):
        chains.append(list(root.indices))
        root = root.dataset

    root_targets: np.ndarray | None = None
    if isinstance(root, TensorDataset) and len(root.tensors) >= 2:
        root_targets = root.tensors[1].numpy()
    elif hasattr(root, "targets"):
        root_targets = np.asarray(root.targets)  # type: ignore[attr-defined]

    n = len(dataset)  # type: ignore[arg-type]
    if root_targets is None:
        # Slow path: pull the label element from each item.
        return np.asarray([int(dataset[i][1]) for i in range(n)])

    # Map each position in `dataset` back to a root index.
    positions = np.arange(n)
    for chain in reversed(chains):  # innermost first, then outward
        positions = np.asarray(chain, dtype=np.int64)[positions]
    return root_targets[positions].astype(np.int64)


def _dirichlet_partition(
    dataset: Dataset,
    num_partitions: int,
    alpha: float,
    server_fraction: float | None,
    seed: int | None,
) -> list[Subset]:
    """Non-IID partition: IID server slice + Dirichlet(α) split across clients.

    The server (``partitions[0]``) receives an IID random share so the
    "server is a powerful / representative node" assumption still holds; the
    remaining pool is dealt to the ``num_partitions - 1`` clients with a
    per-class Dirichlet(α) draw, producing label-skewed clients.
    """
    if num_partitions < 2:
        raise ValueError(
            f"dirichlet partition requires >= 2 partitions, got {num_partitions}",
        )
    if alpha <= 0.0:
        raise ValueError(f"dirichlet alpha must be > 0, got {alpha}")

    num_clients = num_partitions - 1
    labels = _dataset_labels(dataset)
    n = len(labels)

    # A dedicated RNG keeps the Dirichlet draw independent of torch state.
    # When no explicit seed is given, derive one from the (already-seeded)
    # global RNG so the split stays reproducible across runs.
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    # --- IID server slice -------------------------------------------------
    perm = rng.permutation(n)
    if server_fraction is None:
        server_size = n // num_partitions
    else:
        if not 0.0 < server_fraction < 1.0:
            raise ValueError(
                f"server_fraction must lie in (0, 1), got {server_fraction}",
            )
        server_size = int(round(n * server_fraction))
    # Leave at least one sample per client.
    server_size = max(1, min(server_size, n - num_clients))

    server_positions = perm[:server_size]
    client_pool = perm[server_size:]

    # --- Dirichlet split of the remaining pool across clients ------------
    pool_labels = labels[client_pool]
    classes = np.unique(pool_labels)
    n_classes = len(classes)

    # class_counts[k] = #samples of class k left in the pool
    class_counts = np.array([(pool_labels == c).sum() for c in classes])

    # Draw proportions (n_classes × num_clients); scale to integer counts.
    proportions = rng.dirichlet(alpha * np.ones(num_clients), n_classes)
    counts = (proportions * class_counts[:, np.newaxis]).astype(int)  # per class×client

    # Integer truncation loses a few samples per class; hand the leftovers to
    # the least-loaded clients so every sample of every class is assigned.
    per_client_total = counts.sum(axis=0)
    for k in range(n_classes):
        leftover = int(class_counts[k] - counts[k].sum())
        for _ in range(leftover):
            client_idx = int(np.argmin(per_client_total))
            counts[k, client_idx] += 1
            per_client_total[client_idx] += 1

    # Deal actual sample positions class by class.
    client_positions: list[list[int]] = [[] for _ in range(num_clients)]
    for k, c in enumerate(classes):
        pos_c = client_pool[pool_labels == c]
        rng.shuffle(pos_c)
        start = 0
        for client in range(num_clients):
            take = int(counts[k, client])
            if take:
                client_positions[client].extend(pos_c[start : start + take].tolist())
                start += take

    partitions: list[Subset] = [Subset(dataset, server_positions.tolist())]
    for client in range(num_clients):
        partitions.append(Subset(dataset, client_positions[client]))

    client_sizes = [len(p) for p in partitions[1:]]
    logger.info(
        "dirichlet partition (alpha=%.3g): server=%d (%.1f%%, IID), "
        "%d clients sizes=%d…%d",
        alpha, server_size, 100.0 * server_size / n,
        num_clients, min(client_sizes), max(client_sizes),
    )
    return partitions
