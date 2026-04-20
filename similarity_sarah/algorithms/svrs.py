"""SVRS — Server-side Variance-Reduced Similarity baseline.

Reference: "SVRS" (Khaled & Jin, 2023) — https://arxiv.org/pdf/2304.07504

Compared to *Batched No Full Grad SARAH* the algorithm differs in three
places:

1. At the start of each outer round it computes (or refreshes) a
   *reference* gradient estimator at an anchor point ``w_ref``.
2. The inner-loop correction is built relative to ``w_ref`` rather than
   to ``w_{t-1}`` (no recursive coupling between consecutive inner
   steps).
3. The proximal step is taken on the server's local function ``f₁``
   exactly like in our main algorithm, so the existing
   :class:`ProxSolver` can be re-used.

This file provides the *plumbing* (config wiring, registration, base
class implementation) so that experimenting with the full algorithm only
requires filling in :meth:`run_epoch`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.runtime.prox_solver import ProxSolver
from similarity_sarah.runtime.scheduler import ClientBatchScheduler
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    clone_params,
    compute_full_gradient,
    compute_param_norm,
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class SVRS(BaseAlgorithm):
    """SVRS baseline (similarity-based variance reduction)."""

    def __init__(
        self,
        theta: float,
        batch_size_clients: int,
        prox_solver: ProxSolver,
    ) -> None:
        self.theta = theta
        self.batch_size_clients = batch_size_clients
        self.prox_solver = prox_solver

        self.model: nn.Module | None = None
        self.server_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.scheduler: ClientBatchScheduler | None = None
        self.num_clients: int = 0
        self.total_nodes: int = 0

        # Anchor point and its global-similarity gradient estimator.
        self._w_ref: ParamList = []
        self._g_ref: ParamList = []

    def initialize(
        self,
        model: nn.Module,
        server_loader: DataLoader,
        client_loaders: list[DataLoader],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.server_loader = server_loader
        self.client_loaders = client_loaders
        self.loss_fn = loss_fn
        self.device = device
        self.num_clients = len(client_loaders)
        self.total_nodes = self.num_clients + 1
        self.scheduler = ClientBatchScheduler(
            self.num_clients, self.batch_size_clients,
        )
        self._w_ref = get_params(model)
        self._g_ref = zeros_like_params(model)

    # ------------------------------------------------------------------
    def _refresh_anchor(self) -> None:
        """Compute ``g_ref = (1/n) Σ_i ∇(f_i − f₁)(w_ref)``."""
        n = self.total_nodes
        set_params(self.model, self._w_ref)
        g_ref = zeros_like_params(self.model)
        grad_f1 = compute_full_gradient(
            self.model, self.server_loader, self.loss_fn, self.device,
        )
        for loader in self.client_loaders:
            g = compute_full_gradient(
                self.model, loader, self.loss_fn, self.device,
            )
            for tg, gi, g1 in zip(g_ref, g, grad_f1):
                tg.add_(gi - g1, alpha=1.0 / n)
        self._g_ref = g_ref

    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.scheduler is not None and self.model is not None

        # Refresh anchor at the start of every outer round.
        self._w_ref = get_params(self.model)
        self._refresh_anchor()

        batches = self.scheduler.get_epoch_batches()
        K = len(batches)
        n = self.total_nodes
        B = self.batch_size_clients

        for t, batch in enumerate(batches, start=1):
            B_actual = len(batch)
            w_curr = get_params(self.model)

            # ── stochastic similarity correction at w_t ──────────────
            grad_f1_curr = compute_full_gradient(
                self.model, self.server_loader, self.loss_fn, self.device,
            )
            sum_diff_curr = zeros_like_params(self.model)
            for cid in batch:
                g_curr = compute_full_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                for sd, gc, g1 in zip(sum_diff_curr, g_curr, grad_f1_curr):
                    sd.add_(gc - g1)

            set_params(self.model, self._w_ref)
            grad_f1_ref = compute_full_gradient(
                self.model, self.server_loader, self.loss_fn, self.device,
            )
            sum_diff_ref = zeros_like_params(self.model)
            for cid in batch:
                g_ref_b = compute_full_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                for sd, gr, g1 in zip(sum_diff_ref, g_ref_b, grad_f1_ref):
                    sd.add_(gr - g1)

            # v_t = g_ref + (1/B) ( Σ_curr - Σ_ref )
            v = clone_params(self._g_ref)
            for vi, sdc, sdr in zip(v, sum_diff_curr, sum_diff_ref):
                vi.add_(sdc - sdr, alpha=1.0 / B_actual)

            set_params(self.model, w_curr)
            self.prox_solver.step(
                self.model, v, self.theta,
                self.server_loader, self.loss_fn, self.device,
            )

            logger.debug(
                "SVRS epoch %d step %d/%d ‖v‖=%.3e",
                epoch, t, K, compute_param_norm(v),
            )

        return {"epoch": float(epoch), "inner_steps": float(K)}
