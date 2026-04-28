"""Distributed SARAH baseline.

At the start of every epoch the server aggregates *exact* local full
gradients from every node (one full pass per loader, accumulating
sample-weighted gradients).  Inner steps use a SARAH-style correction
with partial client sampling and a plain gradient-descent step (no
proximal operator).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.runtime.scheduler import sample_client_batches
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    clone_params,
    compute_batch_gradient,
    compute_full_gradient,
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class DistributedSARAH(BaseAlgorithm):
    """Distributed SARAH with full-gradient anchor at epoch start.

    Epoch structure:
        v₀ = (1/n) Σᵢ ∇fᵢ(w₀)   (FULL local gradient on each node — one
                                  pass over the entire local partition).
        for t = 0 … K−1:
            w_{t+1} = wₜ − η·vₜ
            Δ_server = ∇f₁(w_{t+1}) − ∇f₁(wₜ)        (one minibatch on server,
                                                       same xy at both points)
            Δ_clients = (1/|Bₜ|) Σ_{i∈Bₜ} [∇fᵢ(w_{t+1}) − ∇fᵢ(wₜ)]
                                                      (one minibatch per
                                                       sampled client, same
                                                       xy at both points)
            v_{t+1} = vₜ + (1/n)·Δ_server + ((n−1)/(n·B))·Σ Δ_clients
    """

    def __init__(
        self,
        lr: float,
        batch_size_clients: int,
    ) -> None:
        self.lr = lr
        self.batch_size_clients = batch_size_clients

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.num_clients: int = 0
        self.total_nodes: int = 0

    def initialize(
        self,
        model: nn.Module,
        server_grad_loader: DataLoader,
        server_prox_loader: DataLoader,
        client_loaders: list[DataLoader],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.server_grad_loader = server_grad_loader
        # Kept for parity with other algorithms; this baseline uses only the
        # deterministic gradient loader (no prox operator here).
        self.server_prox_loader = server_prox_loader
        self.client_loaders = client_loaders
        self.loss_fn = loss_fn
        self.device = device
        self.num_clients = len(client_loaders)
        self.total_nodes = self.num_clients + 1

    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        # ── global gradient estimate at epoch start (one batch per node) ─
        v = self._compute_global_gradient()

        batches = sample_client_batches(self.num_clients, self.batch_size_clients)
        K = len(batches)
        n = self.total_nodes

        for t, batch in enumerate(batches):
            w_old = get_params(self.model)

            # gradient step: w_{t+1} = wₜ − η·vₜ
            w_new = [wo - self.lr * vi for wo, vi in zip(w_old, v)]
            set_params(self.model, w_new)

            # ── server correction (same minibatch at w_new and w_old) ─
            srv_xy = next(iter(self.server_grad_loader))
            set_params(self.model, w_new)
            grad_f1_new = compute_batch_gradient(
                self.model, self.server_grad_loader, self.loss_fn, self.device,
                xy=srv_xy,
            )
            set_params(self.model, w_old)
            grad_f1_old = compute_batch_gradient(
                self.model, self.server_grad_loader, self.loss_fn, self.device,
                xy=srv_xy,
            )
            delta_server = [gn - go for gn, go in zip(grad_f1_new, grad_f1_old)]

            # ── client correction (sampled batch) ───────────────────
            B_size = len(batch)
            sum_delta_clients = zeros_like_params(self.model)
            for cid in batch:
                cli_xy = next(iter(self.client_loaders[cid]))
                set_params(self.model, w_new)
                g_new = compute_batch_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                    xy=cli_xy,
                )
                set_params(self.model, w_old)
                g_old = compute_batch_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                    xy=cli_xy,
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
        """(1/n) Σᵢ ∇fᵢ(w) — *exact* local full gradient on every node.

        Each call to ``compute_full_gradient(loader, ...)`` iterates the
        loader to exhaustion (one full pass) and accumulates per-batch
        gradients weighted by batch size, then divides by total samples —
        equivalent to the deterministic per-node gradient ``∇fᵢ(w)``.

        The matching pseudocode line is

            g_i^k = ∇f_i(x^k)            ← FULL local gradient

        With deterministic per-node gradients, ``v₀`` has zero variance
        from anchor estimation; the only stochasticity left is in the
        recursive Δ-updates.
        """
        n = self.total_nodes
        global_grad = zeros_like_params(self.model)

        grad_server = compute_full_gradient(
            self.model, self.server_grad_loader, self.loss_fn, self.device,
        )
        add_params_(global_grad, grad_server, alpha=1.0 / n)

        for loader in self.client_loaders:
            g = compute_full_gradient(
                self.model, loader, self.loss_fn, self.device,
            )
            add_params_(global_grad, g, alpha=1.0 / n)

        return global_grad
