"""Base interface for the rewritten distributed algorithms.

A new baseline is two files — ``algorithms/<name>.py`` (an :class:`Algorithm`
subclass that self-registers with :data:`ALGORITHMS` and implements
``from_spec``) and ``configs/algorithm/<name>.yaml`` — with no edits to the
runner.
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


@dataclass
class AlgorithmCtx:
    """The model, data, loss, and device an algorithm runs against.

    ``server_grad_loader`` is the deterministic loader for the SARAH gradients
    (no augmentation); ``server_prox_loader`` feeds the inexact prox solver.
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
    """Registry-driven base for the distributed algorithms."""

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


ALGORITHMS: Registry[Algorithm] = Registry("algorithm")
