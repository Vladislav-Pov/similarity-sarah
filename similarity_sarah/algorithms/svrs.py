"""SVRS — Server-side Variance-Reduced Similarity baseline.

Reference: Lin, Karagulyan, Richtárik, "Stochastic Distributed
Optimization under Average Second-order Similarity: Algorithms and
Analysis" — https://arxiv.org/pdf/2304.07504  (Algorithm 1, SVRS¹ᵉᵖ).

Compared to *Batched No Full Grad SARAH* the algorithm differs in:

1. At the start of each outer round we compute the deterministic
   anchor ``g_ref = (1/n) Σᵢ ∇(fᵢ − f₁)(w_ref)`` (full local
   gradients on each node — see ``_refresh_anchor``).
2. The epoch length ``T ∼ Geom(p)`` with ``p = 1/num_clients``;
   ``E[T] = num_clients``.  This replaces the deterministic
   permutation-based pass we used previously.
3. Each inner step samples ONE client ``i_t ∼ Unif([num_clients])``
   (Algorithm 1 of the paper has no client-batching) and builds
   ``v_t = g_ref + ∇(f_{i_t} − f₁)(w_t) − ∇(f_{i_t} − f₁)(w_ref)``.
4. The proximal step is taken on the server's local function ``f₁``
   exactly like in BNFG, so the existing :class:`ProxSolver` is
   re-used.

The SARAH-style same-batch telescope ``∇f_i(w_t) − ∇f_i(w_ref)`` uses
one fixed minibatch per client per step (pre-sampled and reused at
both points) — without that, the correction would behave like a fresh
stochastic gradient with no variance reduction.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.runtime.prox_solver import ProxSolver
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    clone_params,
    compute_batch_gradient,
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
        # Algorithm 1 of Lin et al. 2023 has *one* client per inner step
        # (i_t ∼ Unif([n])).  We keep ``batch_size_clients`` as a no-op
        # field for API parity with BNFG / distributed_sarah, but force
        # B = 1 internally and warn if the user asked for more.
        if batch_size_clients != 1:
            logger.warning(
                "SVRS: batch_size_clients=%d ignored — Algorithm 1 of the "
                "original paper samples exactly one client per inner step "
                "(B=1).  Forcing B=1.",
                batch_size_clients,
            )
        self.batch_size_clients = 1
        self.prox_solver = prox_solver

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.num_clients: int = 0
        self.total_nodes: int = 0

        # Anchor point and its global-similarity gradient estimator.
        self._w_ref: ParamList = []
        self._g_ref: ParamList = []

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
        self.server_prox_loader = server_prox_loader
        self.client_loaders = client_loaders
        self.loss_fn = loss_fn
        self.device = device
        self.num_clients = len(client_loaders)
        self.total_nodes = self.num_clients + 1
        self._w_ref = get_params(model)
        self._g_ref = zeros_like_params(model)

    # ------------------------------------------------------------------
    def _refresh_anchor(self) -> None:
        """Compute ``g_ref = (1/n) Σ_i ∇(f_i − f₁)(w_ref)`` exactly.

        Uses :func:`compute_full_gradient` (one full pass per loader) so that
        ``g_ref`` is deterministic — no minibatch noise carries into ``v_t``
        through the anchor.  Matches the paper's ``∇f(w_0)`` precomputation
        (just shifted by ``∇f_1(w_0)``); see Algorithm 1 of Khaled & Jin 2023.
        """
        n = self.total_nodes
        set_params(self.model, self._w_ref)
        g_ref = zeros_like_params(self.model)
        grad_f1 = compute_full_gradient(
            self.model, self.server_grad_loader, self.loss_fn, self.device,
        )
        for loader in self.client_loaders:
            g = compute_full_gradient(
                self.model, loader, self.loss_fn, self.device,
            )
            for tg, gi, g1 in zip(g_ref, g, grad_f1):
                tg.add_(gi - g1, alpha=1.0 / n)
        self._g_ref = g_ref

    def _sample_epoch_length(self) -> int:
        """Draw T ∼ Geom(p = 1/num_clients), shifted so that T ≥ 1.

        Matches Algorithm 1 of Lin et al. 2023 (SVRS¹ᵉᵖ): the inner-loop
        horizon between anchor refreshes is a geometric random variable
        with ``E[T] = 1/p = num_clients``.  This replaces the deterministic
        permutation-based pass we used before (which always covered every
        client exactly once per epoch).

        Using ``num_clients`` for ``p`` (rather than ``total_nodes``)
        because in our setup the server is always queried inside the prox
        and is never one of the i_t-sampled components.
        """
        p = torch.tensor(1.0 / max(self.num_clients, 1))
        raw = torch.distributions.Geometric(probs=p).sample().item()
        return min(int(raw) + 1, 4)

    def _sample_one_client(self) -> int:
        """``i_t ∼ Unif([num_clients])`` — a single uniform draw."""
        return int(torch.randint(low=0, high=self.num_clients, size=(1,)).item())

    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None

        # Refresh anchor at the start of every outer round.
        self._w_ref = get_params(self.model)
        self._refresh_anchor()

        # Random epoch length per the original SVRS paper.
        T = self._sample_epoch_length()
        logger.debug(
            "SVRS epoch %d: sampled T=%d (E[T]=%d)", epoch, T, self.num_clients,
        )

        prox_first_norms: list[float] = []
        prox_last_norms: list[float] = []
        prox_ratios: list[float] = []
        prox_obj_decreases: list[float] = []
        prox_clip_fracs: list[float] = []
        prox_inner_steps: list[float] = []

        for t in range(1, T + 1):
            cid = self._sample_one_client()
            w_curr = get_params(self.model)

            # Pre-sample one minibatch per node, reused at w_curr and w_ref.
            srv_xy = next(iter(self.server_grad_loader))
            cli_xy = next(iter(self.client_loaders[cid]))

            # ── ∇(f_i − f₁)(w_t) on shared minibatch ─────────────────
            grad_f1_curr = compute_batch_gradient(
                self.model, self.server_grad_loader, self.loss_fn, self.device,
                xy=srv_xy,
            )
            g_curr = compute_batch_gradient(
                self.model, self.client_loaders[cid],
                self.loss_fn, self.device, xy=cli_xy,
            )
            diff_curr = [gc - g1 for gc, g1 in zip(g_curr, grad_f1_curr)]

            # ── ∇(f_i − f₁)(w_ref) on the SAME minibatch ─────────────
            set_params(self.model, self._w_ref)
            grad_f1_ref = compute_batch_gradient(
                self.model, self.server_grad_loader, self.loss_fn, self.device,
                xy=srv_xy,
            )
            g_ref_b = compute_batch_gradient(
                self.model, self.client_loaders[cid],
                self.loss_fn, self.device, xy=cli_xy,
            )
            diff_ref = [gr - g1 for gr, g1 in zip(g_ref_b, grad_f1_ref)]

            # v_t = g_ref + (∇(f_i − f₁)(w_t) − ∇(f_i − f₁)(w_ref))
            # — single client, no /B division.
            v = clone_params(self._g_ref)
            for vi, dc, dr in zip(v, diff_curr, diff_ref):
                vi.add_(dc - dr)

            set_params(self.model, w_curr)
            prox_diag = self.prox_solver.step(
                self.model, v, self.theta,
                self.server_prox_loader, self.loss_fn, self.device,
                eval_loader=self.server_grad_loader,
            )

            prox_first_norms.append(float(prox_diag["prox_grad_norm_first"]))
            prox_last_norms.append(float(prox_diag["prox_grad_norm_last"]))
            prox_ratios.append(float(prox_diag["prox_grad_norm_ratio"]))
            prox_obj_decreases.append(float(prox_diag["prox_obj_decrease"]))
            prox_clip_fracs.append(float(prox_diag.get("prox_clip_frac", 0.0)))
            prox_inner_steps.append(float(prox_diag.get("prox_inner_steps", 0.0)))

            logger.debug(
                "SVRS epoch %d step %d/%d  cid=%d  ‖v‖=%.3e  "
                "prox_ratio=%.3f  clip_frac=%.2f",
                epoch, t, T, cid, compute_param_norm(v),
                prox_ratios[-1], prox_clip_fracs[-1],
            )

        def _avg(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        return {
            "epoch": float(epoch),
            "inner_steps": float(T),
            "prox_grad_norm_first_mean": _avg(prox_first_norms),
            "prox_grad_norm_last_mean": _avg(prox_last_norms),
            "prox_grad_norm_ratio_mean": _avg(prox_ratios),
            "prox_obj_decrease_mean": _avg(prox_obj_decreases),
            "prox_clip_frac_mean": _avg(prox_clip_fracs),
            "prox_inner_steps_mean": _avg(prox_inner_steps),
        }
