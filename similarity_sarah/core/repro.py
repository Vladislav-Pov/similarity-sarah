"""Reproducibility primitives: global seeding and deterministic execution.

The reference NFG-SS runs require that a fixed seed maps to a fixed result.
:func:`set_seed` seeds every RNG the pipeline touches; passing
``deterministic=True`` additionally forces PyTorch onto deterministic kernels
and pins cuDNN, trading a little speed for bit-stable trajectories.

Notes
-----
``np.random`` is still seeded for parity with the pre-rewrite ``utils.set_seed``
so the RNG-draw order is unchanged; the numpy dependency is slated for removal
in a later cleanup milestone.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed all RNGs used by the pipeline.

    Parameters
    ----------
    seed:
        Seed shared by ``random``, ``numpy``, and ``torch`` (CPU and CUDA).
    deterministic:
        If ``True``, also enable PyTorch deterministic algorithms and pin
        cuDNN. Some kernels lack a deterministic implementation; those emit a
        warning rather than raising (``warn_only=True``) so that MPS/CUDA runs
        do not crash. Leave ``False`` to reproduce the legacy (non-pinned)
        behaviour of the pre-rewrite code.

    Notes
    -----
    ``deterministic=True`` is a process-wide switch: it affects every
    subsequent PyTorch call in the interpreter, not just the next run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic CUDA matmul; harmless elsewhere.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU :class:`torch.Generator` seeded with ``seed``.

    Use this to give a ``DataLoader`` an explicit ``generator=`` so its shuffle
    order does not depend on (or perturb) the global RNG state.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that makes worker RNGs reproducible.

    PyTorch derives each worker's base seed from the main process generator;
    we propagate it to ``numpy`` and ``random`` so augmentation/shuffling in
    workers is reproducible across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
