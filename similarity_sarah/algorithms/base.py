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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.registry import Registry

if TYPE_CHECKING:
    from similarity_sarah.spec import RunSpec


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


@dataclass
class AlgorithmCtx:
    """The model, data, loss, and device an algorithm runs against.

    ``server_grad_loader`` is the deterministic loader for the SARAH gradients
    (no augmentation); ``server_prox_loader`` feeds the inexact prox solver
    (may be augmented).
    """

    model: nn.Module
    server_grad_loader: DataLoader
    server_prox_loader: DataLoader
    client_loaders: list[DataLoader]
    loss_fn: nn.Module
    device: torch.device

    @property
    def num_clients(self) -> int:
        return len(self.client_loaders)

    @property
    def total_nodes(self) -> int:
        """The paper's ``n``: server ``f1`` plus one objective per client."""
        return self.num_clients + 1


class Algorithm(ABC):
    """Registry-driven base for the rewritten distributed algorithms.

    A new baseline is two files — ``algorithms/<name>.py`` (a subclass that
    self-registers with :data:`ALGORITHMS` and implements ``from_spec``) and
    ``configs/algorithm/<name>.yaml`` — with no edits to the runner.
    """

    @abstractmethod
    def bind(self, ctx: AlgorithmCtx) -> None:
        """Bind the algorithm to a run context (model, loaders, loss, device)."""

    @abstractmethod
    def run_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one outer epoch and return a metrics dict."""

    @classmethod
    def from_spec(cls, spec: RunSpec) -> Algorithm:
        """Build an algorithm from a :class:`RunSpec` (overridden by subclasses)."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_spec")

    def initialize(
        self,
        model: nn.Module,
        server_grad_loader: DataLoader,
        server_prox_loader: DataLoader,
        client_loaders: list[DataLoader],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        """Legacy six-argument bind, for the pre-rewrite runner (removed in M7)."""
        self.bind(
            AlgorithmCtx(
                model, server_grad_loader, server_prox_loader,
                client_loaders, loss_fn, device,
            )
        )


ALGORITHMS: Registry[Algorithm] = Registry("algorithm")
