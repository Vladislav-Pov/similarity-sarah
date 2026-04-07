"""Distributed SARAH baseline with full-gradient computation.

At the start of every epoch the server computes the exact global gradient
(all nodes participate).  Inner steps then use a standard SARAH recursive
correction with partial client sampling and a plain gradient-descent step
(no proximal operator).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.runtime.scheduler import ClientBatchScheduler
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    clone_params,
    compute_full_gradient,
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class DistributedSARAH(BaseAlgorithm):
    """Distributed SARAH with full gradient at epoch start.

    Epoch structure:
        v₀ = ∇f(w₀)  =  (1/n) Σᵢ ∇fᵢ(w₀)           ← all nodes
        for t = 0 … K−1:
            w_{t+1} = wₜ − η·vₜ
            Δ_server = ∇f₁(w_{t+1}) − ∇f₁(wₜ)
            Δ_clients = (1/|Bₜ|) Σ_{i∈Bₜ} [∇fᵢ(w_{t+1}) − ∇fᵢ(wₜ)]
            v_{t+1} = vₜ + (1/n)·Δ_server + ((n−1)/n)·Δ_clients
    """

    def __init__(
        self,
        lr: float,
        batch_size_clients: int,
    ) -> None:
        self.lr = lr
        self.batch_size_clients = batch_size_clients

        self.model: nn.Module | None = None
        self.server_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.scheduler: ClientBatchScheduler | None = None
        self.num_clients: int = 0
        self.total_nodes: int = 0

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

    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.scheduler is not None

        # ── full gradient at epoch start (all nodes) ────────────────
        v = self._compute_global_gradient()

        batches = self.scheduler.get_epoch_batches()
        K = len(batches)
        n = self.total_nodes

        for t, batch in enumerate(batches):
            w_old = get_params(self.model)

            # gradient step: w_{t+1} = wₜ − η·vₜ
            w_new = [wo - self.lr * vi for wo, vi in zip(w_old, v)]
            set_params(self.model, w_new)

            # ── server correction (always computed, local) ──────────
            set_params(self.model, w_new)
            grad_f1_new = compute_full_gradient(
                self.model, self.server_loader, self.loss_fn, self.device,
            )
            set_params(self.model, w_old)
            grad_f1_old = compute_full_gradient(
                self.model, self.server_loader, self.loss_fn, self.device,
            )
            delta_server = [gn - go for gn, go in zip(grad_f1_new, grad_f1_old)]

            # ── client correction (sampled batch) ───────────────────
            B_size = len(batch)
            sum_delta_clients = zeros_like_params(self.model)
            for cid in batch:
                set_params(self.model, w_new)
                g_new = compute_full_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                set_params(self.model, w_old)
                g_old = compute_full_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                for sd, gn, go in zip(sum_delta_clients, g_new, g_old):
                    sd.add_(gn - go)

            # v_{t+1} = vₜ + (1/n)·Δ_server + ((n-1)/(n·B))·Σ Δ_clients
            for vi, ds, dc in zip(v, delta_server, sum_delta_clients):
                vi.add_(ds, alpha=1.0 / n)
                vi.add_(dc, alpha=(n - 1) / (n * B_size))

            set_params(self.model, w_new)
            logger.debug("Epoch %d  step %d/%d done", epoch, t + 1, K)

        return {"epoch": epoch, "inner_steps": K}

    # ------------------------------------------------------------------
    def _compute_global_gradient(self) -> ParamList:
        """(1/n) Σᵢ₌₁ⁿ ∇fᵢ(w)  —  requires participation of every node."""
        n = self.total_nodes
        global_grad = zeros_like_params(self.model)

        grad_server = compute_full_gradient(
            self.model, self.server_loader, self.loss_fn, self.device,
        )
        add_params_(global_grad, grad_server, alpha=1.0 / n)

        for loader in self.client_loaders:
            g = compute_full_gradient(
                self.model, loader, self.loss_fn, self.device,
            )
            add_params_(global_grad, g, alpha=1.0 / n)

        return global_grad
