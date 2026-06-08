"""Inexact proximal solvers for the server's local objective ``f1``.

Importing this package registers every in-scope solver with
:data:`PROX_SOLVERS` (``sgd``, ``adam``, ``accvrs_batch_sgd``). Build one from a
config via :func:`build_prox_solver`.
"""

from similarity_sarah.prox.accvrs import AccvrsBatchSGD
from similarity_sarah.prox.base import PROX_SOLVERS, ProxSolver, build_prox_solver
from similarity_sarah.prox.inexact import InexactProxAdam, InexactProxSGD

__all__ = [
    "PROX_SOLVERS",
    "AccvrsBatchSGD",
    "InexactProxAdam",
    "InexactProxSGD",
    "ProxSolver",
    "build_prox_solver",
]
