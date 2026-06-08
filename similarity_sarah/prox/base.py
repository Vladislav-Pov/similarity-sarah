"""Proximal-solver interface, registry, and factory.

A :class:`ProxSolver` approximately solves ``w ~= prox_{theta f1}(w - theta v)``
and updates the model in place, returning a small diagnostics dict. Concrete
solvers self-register with :data:`PROX_SOLVERS` and build from a
:class:`~similarity_sarah.spec.ProxSpec` via :meth:`ProxSolver.from_spec`;
:func:`build_prox_solver` resolves a spec to a solver by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.core.params import ParamList
from similarity_sarah.registry import Registry
from similarity_sarah.spec import ProxSpec


class ProxSolver(ABC):
    """Abstract base for inexact proximal-operator solvers."""

    #: Inner learning rate. ``None`` means auto-derived (AccVRS); the runner's
    #: prox-lr schedule reads and writes this attribute.
    lr: float | None

    @abstractmethod
    def step(
        self,
        model: nn.Module,
        v: ParamList,
        theta: float,
        server_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        eval_loader: DataLoader | None = None,
    ) -> Mapping[str, float]:
        """Approximately solve the prox subproblem and update ``model``.

        ``server_loader`` feeds the inner training steps (may be augmented);
        ``eval_loader`` provides the fixed deterministic batch for diagnostics
        (falling back to ``server_loader``). Returns the diagnostics dict.
        """

    @classmethod
    def from_spec(cls, spec: ProxSpec) -> ProxSolver:
        """Build a solver from a :class:`ProxSpec` (overridden by subclasses)."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_spec")


PROX_SOLVERS: Registry[ProxSolver] = Registry("prox solver")


def build_prox_solver(spec: ProxSpec) -> ProxSolver:
    """Resolve and construct the proximal solver named by ``spec.kind``."""
    return PROX_SOLVERS.get(spec.kind).from_spec(spec)
