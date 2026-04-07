"""Base interface for distributed optimization algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAlgorithm(ABC):
    """Abstract base class for distributed optimisation algorithms.

    Lifecycle:
        1. ``__init__``  — store hyper-parameters.
        2. ``initialize`` — receive the model, data loaders, and device.
        3. ``run_epoch``  — called repeatedly by the runner.

    To add a new algorithm, subclass ``BaseAlgorithm`` and implement
    ``initialize`` and ``run_epoch``.  Optionally override ``run_step`` for
    finer-grained control.
    """

    @abstractmethod
    def initialize(
        self,
        model: nn.Module,
        server_loader: DataLoader,
        client_loaders: list[DataLoader],
        loss_fn: nn.Module,
        device: "torch.device",
    ) -> None:
        """Bind the algorithm to a model, data, and device."""
        ...

    @abstractmethod
    def run_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one full epoch and return a metrics dict."""
        ...

    def run_step(self, step: int) -> dict[str, float]:
        """Execute a single inner step (optional, for fine-grained control)."""
        raise NotImplementedError


# Needed for the type annotation in initialize; avoid circular import.
import torch  # noqa: E402
