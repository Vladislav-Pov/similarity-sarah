"""Build the federated train/eval loaders from a :class:`RunSpec`.

Preserves the two-loader server split: ``server_grad_loader`` (deterministic,
feeds the SARAH gradients) and ``server_prox_loader`` (feeds the inexact prox
solver). Server-side augmentation is out of scope for the submission, so both
use the deterministic transform.

Bit-reproducibility: when ``spec.runtime.deterministic`` is False (the
reference path) loaders take no explicit ``generator``, so the shuffle order is
drawn from the global RNG exactly as the pre-rewrite runner did. When True,
each loader is seeded for full reproducibility under ``num_workers > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from similarity_sarah.core.repro import seed_worker
from similarity_sarah.data.datasets import (
    load_augmented_train,
    load_dataset,
    split_train_val,
)
from similarity_sarah.data.partition import create_partition
from similarity_sarah.spec import RunSpec


def _abs_indices(dataset: Dataset) -> list[int]:
    """Resolve a (possibly nested) ``Subset`` to absolute indices into the base."""
    if isinstance(dataset, Subset):
        parent = _abs_indices(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    return list(range(len(dataset)))  # type: ignore[arg-type]


@dataclass
class FederatedData:
    """The loaders a run needs.

    ``train_eval_loader`` is a deterministic, no-augmentation loader over a
    subset of the (pooled) training data — used only to log train loss/accuracy
    each eval step, so over/under-fitting is visible against val.
    """

    server_grad_loader: DataLoader
    server_prox_loader: DataLoader
    client_loaders: list[DataLoader]
    val_loader: DataLoader
    test_loader: DataLoader
    train_eval_loader: DataLoader


def build_federated_data(
    spec: RunSpec, *, generator: torch.Generator | None = None
) -> FederatedData:
    """Construct the federated loaders in the pre-rewrite (data -> model) order."""
    data_cfg = OmegaConf.create(
        {
            "name": spec.data.name,
            "data_dir": spec.data.data_dir,
            "n_train": spec.data.n_train,
            "n_test": spec.data.n_test,
        }
    )
    train, test = load_dataset(data_cfg)
    train, val = split_train_val(train, spec.data.val_fraction)

    part_cfg = OmegaConf.create(
        {"name": spec.partition.name, "server_fraction": spec.partition.server_fraction}
    )
    partitions = create_partition(train, spec.num_clients + 1, part_cfg)

    # Optional train augmentation on the *node gradient* loaders (server_grad +
    # clients). Safe for the SARAH telescope: section 11 reuses the same
    # materialised minibatch at w_t and w_{t-1}, so the augmented sample is fixed
    # within an inner step; it only varies across steps. ``None`` ⇒ the dataset
    # has no augmentation pipeline (e.g. synthetic), so fall back to deterministic.
    aug_full = load_augmented_train(data_cfg) if spec.data.augment_train else None

    def _node_ds(partition: Dataset) -> Dataset:
        if aug_full is None:
            return partition
        return Subset(aug_full, _abs_indices(partition))

    rt = spec.runtime
    nw = rt.num_workers
    eval_bs = (
        max(
            rt.batch_size_server_grad,
            rt.batch_size_server_prox,
            rt.batch_size_data_clients,
            rt.batch_size,
        )
        * 2
    )
    worker_init = seed_worker if generator is not None else None

    server_grad = DataLoader(
        _node_ds(partitions[0]), batch_size=rt.batch_size_server_grad, shuffle=True,
        num_workers=nw, generator=generator, worker_init_fn=worker_init,
    )
    server_prox = DataLoader(
        partitions[0], batch_size=rt.batch_size_server_prox, shuffle=True,
        num_workers=nw, generator=generator, worker_init_fn=worker_init,
    )
    clients = [
        DataLoader(
            _node_ds(p), batch_size=rt.batch_size_data_clients, shuffle=True,
            num_workers=nw, generator=generator, worker_init_fn=worker_init,
        )
        for p in partitions[1:]
    ]
    test_loader = DataLoader(test, batch_size=eval_bs, shuffle=False, num_workers=nw)
    val_loader = DataLoader(val, batch_size=eval_bs, shuffle=False, num_workers=nw)

    # Train-set eval loader (deterministic, no augmentation): a fixed subset of
    # the pooled train, capped for speed. Lets the loop log train loss/accuracy.
    n_train_eval = min(len(train), 10000)  # type: ignore[arg-type]
    train_eval_loader = DataLoader(
        Subset(train, list(range(n_train_eval))),
        batch_size=eval_bs, shuffle=False, num_workers=nw,
    )
    return FederatedData(
        server_grad, server_prox, clients, val_loader, test_loader, train_eval_loader
    )
