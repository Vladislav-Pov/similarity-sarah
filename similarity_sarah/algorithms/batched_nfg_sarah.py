"""Batched No Full Grad SARAH — the main algorithm.

Implements Algorithm 1 from the paper.  Key properties:
  - No synchronised full-gradient computation;
  - Recursive SARAH-type estimators (v, tilde_v);
  - Proximal updates on the server's local objective f₁;
  - Non-overlapping batched client sampling per epoch.
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
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class BatchedNoFullGradSARAH(BaseAlgorithm):
    """Batched No Full Grad SARAH.

    Pseudocode (one epoch, s fixed):
        v₀ = v^(s)                                      (carry-over)
        w₁ = prox_{θf₁}(w₀ − θv₀)
        for t = 1 … K:
            tilde_v_{t+1} = ((t−1)/t) tilde_v_t
                          + (1/(tB)) Σ_{i∈Bₜ} ∇(fᵢ−f₁)(wₜ)
            vₜ = v_{t−1}
               + (1/(nB)) Σ_{i∈Bₜ} [∇(fᵢ−f₁)(wₜ) − ∇(fᵢ−f₁)(w_{t−1})]
            w_{t+1} = prox_{θf₁}(wₜ − θvₜ)
        v^(s+1) = tilde_v_{K+1}
    """

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
        self.v_epoch: ParamList = []
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
        self.total_nodes = self.num_clients + 1 # maybe without +1

        self.scheduler = ClientBatchScheduler(
            self.num_clients, self.batch_size_clients,
        )
        self.v_epoch = zeros_like_params(model)

    # ------------------------------------------------------------------
    # Main epoch logic
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.scheduler is not None
        batches = self.scheduler.get_epoch_batches()
        K = len(batches)

        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(self.model)

        # Cache w₀ and ∇f₁(w₀) before the first prox step
        w_prev = get_params(self.model)
        grad_f1_prev = compute_full_gradient(
            self.model, self.server_loader, self.loss_fn, self.device,
        ) ## it is possible to count with stochastic gradient 

        # w₁ = prox_{θf₁}(w₀ − θv₀)
        self.prox_solver.step(
            self.model, v, self.theta,
            self.server_loader, self.loss_fn, self.device,
        )

        B = self.batch_size_clients  # paper's B (fixed denominator)

        for t in range(1, K + 1):
            batch = batches[t - 1]
            w_curr = get_params(self.model)

            # ── gradients at wₜ (model is already at wₜ) ────────────
            grad_f1_curr = compute_full_gradient(
                self.model, self.server_loader, self.loss_fn, self.device,
            )
            sum_client_grads_curr = zeros_like_params(self.model)
            for cid in batch:
                g = compute_full_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                add_params_(sum_client_grads_curr, g)

            # ── gradients at w_{t−1} ────────────────────────────────
            set_params(self.model, w_prev)
            sum_client_grads_prev = zeros_like_params(self.model)
            for cid in batch:
                g = compute_full_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                add_params_(sum_client_grads_prev, g)

            # ── form Σ(∇fᵢ − ∇f₁) at wₜ and w_{t−1} ──────────────
            B_actual = len(batch)
            sum_diff_curr = zeros_like_params(self.model)
            sum_diff_prev = zeros_like_params(self.model)
            for sd_c, sc, g1c in zip(
                sum_diff_curr, sum_client_grads_curr, grad_f1_curr,
            ):
                sd_c.copy_(sc - B_actual * g1c)
            for sd_p, sp, g1p in zip(
                sum_diff_prev, sum_client_grads_prev, grad_f1_prev,
            ):
                sd_p.copy_(sp - B_actual * g1p)

            # ── update tilde_v ──────────────────────────────────────
            ratio = (t - 1) / t
            coeff_tilde = 1.0 / (t * B)
            for tv, sdc in zip(tilde_v, sum_diff_curr):
                tv.mul_(ratio).add_(sdc, alpha=coeff_tilde)

            # ── update v (SARAH correction) ─────────────────────────
            n = self.total_nodes
            coeff_v = 1.0 / (n * B)
            for vi, sdc, sdp in zip(v, sum_diff_curr, sum_diff_prev):
                vi.add_(sdc - sdp, alpha=coeff_v)

            # ── prepare next iteration ──────────────────────────────
            w_prev = w_curr
            grad_f1_prev = grad_f1_curr

            # w_{t+1} = prox_{θf₁}(wₜ − θvₜ)
            set_params(self.model, w_curr)
            self.prox_solver.step(
                self.model, v, self.theta,
                self.server_loader, self.loss_fn, self.device,
            )

            logger.debug("Epoch %d  step %d/%d done", epoch, t, K)

        # v^{(s+1)} = tilde_v_{K+1}
        self.v_epoch = tilde_v

        return {"epoch": epoch, "inner_steps": K}
