"""Base interface for distributed optimisation algorithms.

Every concrete algorithm subclasses :class:`BaseAlgorithm` and implements

    initialize(...)        # bind a model, data loaders, loss, device
    run_epoch(...)         # execute one outer epoch (returns metrics)

For more fine-grained training loops, override :meth:`run_step`.

Adding a new algorithm only requires:

    1.  ``similarity_sarah/algorithms/<name>.py``  with a subclass.
    2.  ``configs/algorithm/<name>.yaml``  with hyper-parameters.
    3.  Register it in :class:`Runner._setup_algorithm`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAlgorithm(ABC):
    """Abstract base class for distributed optimisation algorithms."""

    @abstractmethod
    def initialize(
        self,
        model: nn.Module,
        server_grad_loader: DataLoader,
        server_prox_loader: DataLoader,
        client_loaders: list[DataLoader],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        """Bind the algorithm to a model, data, and device.

        ``server_grad_loader`` is the deterministic loader used to evaluate
        ∇f₁ inside the SARAH recursion (must give reproducible gradients for a
        fixed minibatch).  ``server_prox_loader`` is the loader used inside the
        inexact proximal solver (may be augmented).
        """

    @abstractmethod
    def run_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one full epoch and return a metrics dict."""

    def run_step(self, step: int) -> dict[str, float]:
        """Execute a single inner step (override for fine-grained control)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_step",
        )
