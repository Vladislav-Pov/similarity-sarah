"""Deterministic "fast oracle" for NFG-SS bit-reproducibility.

Runs one epoch of the *real* NFG-SS + AccVRS-SGD code on a tiny synthetic,
CPU-only configuration and returns a bit-sensitive signature: the SHA-256 of
the final model weights plus the epoch metrics dict. It is cheap enough for CI
yet exercises the exact hot path — the SARAH telescope, the deliberate D1
``1/(n*B)`` scaling, the carry-over anchor, and the AccVRS inner solver
(auto-gamma0, convex-blend momentum, early-stop).

The data preparation (synthetic tensors, train/val split, server-weighted
partition) is inlined here rather than imported from ``similarity_sarah.data``
so the oracle depends on ``torch`` alone — the algorithm/solver path is
omegaconf-free, and the local dev box has no Hydra stack. The inlined logic
mirrors ``data.datasets._synthetic_cifar10`` / ``data.partition`` exactly.

This is the *local development* oracle, captured from the pre-rewrite code and
re-checked after every milestone. The full CIFAR-10 / ResNet-18 reference
configs (`docs/reference_runs/*.json`) are GPU jobs beyond the 5-minute local
budget; reproduce them on the original hardware from baseline commit 4ca4873.

Important: ``prox_v_schedule`` is a NO-OP on the AccVRS path
(``AccvrsBatchSGDProx`` neither accepts nor reads it), so ``constant`` and
``linear`` at the same ``num_steps`` are bit-identical here by construction —
asserted in ``test_golden.py``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from similarity_sarah.algorithms.batched_nfg_sarah import BatchedNoFullGradSARAH
from similarity_sarah.core.repro import set_seed
from similarity_sarah.models.simple_cnn import SimpleCNN
from similarity_sarah.runtime.prox_solver_accvrs import AccvrsBatchSGDProx

SEED = 42
DEVICE = "cpu"
BATCH_SIZE = 64
N_TRAIN = 500
N_TEST = 100
VAL_FRACTION = 0.1
NUM_PARTITIONS = 11  # 1 server + 10 clients
SERVER_FRACTION = 0.5


@dataclass(frozen=True)
class FastVariant:
    """One fast-oracle configuration, mirroring a reference-run file."""

    name: str
    prox_v_schedule: str
    prox_num_steps: int


# Mirror the three docs/reference_runs/*.json files on the fast config.
FAST_VARIANTS: tuple[FastVariant, ...] = (
    FastVariant("fast_constant_4steps", "constant", 4),
    FastVariant("fast_linear_4steps", "linear", 4),
    FastVariant("fast_linear_5steps", "linear", 5),
)


def _synthetic_train() -> Dataset:
    """Random CIFAR-shaped train set (mirrors ``datasets._synthetic_cifar10``).

    The (unused) test set is materialised too so the RNG-draw order matches the
    real loader pipeline.
    """
    train = TensorDataset(
        torch.randn(N_TRAIN, 3, 32, 32), torch.randint(0, 10, (N_TRAIN,))
    )
    _ = TensorDataset(torch.randn(N_TEST, 3, 32, 32), torch.randint(0, 10, (N_TEST,)))
    return train


def _server_weighted_partition(dataset: Dataset, num_partitions: int) -> list[Subset]:
    """Server-heavy partition (mirrors ``data.partition._server_weighted_partition``)."""
    n = len(dataset)  # type: ignore[arg-type]
    indices = torch.randperm(n).tolist()
    server_size = int(round(n * SERVER_FRACTION))
    num_clients = num_partitions - 1
    server_size = max(1, min(server_size, n - num_clients))

    remainder = n - server_size
    base_size = remainder // num_clients
    extra = remainder % num_clients

    parts = [Subset(dataset, indices[:server_size])]
    offset = server_size
    for i in range(num_clients):
        size = base_size + (1 if i < extra else 0)
        parts.append(Subset(dataset, indices[offset : offset + size]))
        offset += size
    return parts


def _weight_sha256(model: nn.Module) -> str:
    """SHA-256 over the raw float32 bytes of all parameters, in order."""
    digest = hashlib.sha256()
    for p in model.parameters():
        arr = p.detach().to("cpu", torch.float32).contiguous().numpy()
        digest.update(arr.tobytes())
    return digest.hexdigest()


def build_and_run(variant: FastVariant) -> dict[str, object]:
    """Run one NFG-SS epoch on the fast config and return its signature.

    Returns
    -------
    dict
        ``{"weight_sha256": str, "metrics": dict[str, float]}``.
    """
    logging.getLogger("similarity_sarah").setLevel(logging.WARNING)
    set_seed(SEED, deterministic=True)
    device = torch.device(DEVICE)

    # Build in the same order as ``Runner`` (data -> model -> run) so the RNG
    # draw sequence is faithful to the real pipeline.
    train = _synthetic_train()
    n_val = int(len(train) * VAL_FRACTION)  # type: ignore[arg-type]
    train, _val = random_split(train, [len(train) - n_val, n_val])  # type: ignore[arg-type]
    partitions = _server_weighted_partition(train, NUM_PARTITIONS)

    server_grad = DataLoader(partitions[0], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    server_prox = DataLoader(partitions[0], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    clients = [
        DataLoader(p, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        for p in partitions[1:]
    ]

    model = SimpleCNN(num_classes=10).to(device)

    solver = AccvrsBatchSGDProx(
        num_steps=variant.prox_num_steps,
        lr=None,
        L1=200.0,
        lr_factor=0.1,
        weight_decay=0.1,
        momentum=0.9,
        grad_clip=0.0,
        inner_decay_factor=1.0,
        inner_decay_period=None,
        early_stop_ratio=1e-4,
        include_linear_term=False,
        eval_batches=2,
    )
    algo = BatchedNoFullGradSARAH(theta=0.2, batch_size_clients=1, prox_solver=solver)
    algo.initialize(
        model=model,
        server_grad_loader=server_grad,
        server_prox_loader=server_prox,
        client_loaders=clients,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
    )
    metrics = {k: float(v) for k, v in algo.run_epoch(0).items()}
    return {"weight_sha256": _weight_sha256(model), "metrics": metrics}
