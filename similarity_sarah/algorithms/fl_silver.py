"""FL-SILVER baseline — Federated variance-reduced local training under
average second-order similarity, with per-client gradient memory.

Reference: Algorithm 2 ``FL-SILVER(x⁰, η, p, b, T, K, r)`` from the paper
https://openreview.net/pdf?id=pOgMluzEIH .

Unlike the SARAH-style server-driven methods already in this repo
(``batched_nfg_sarah``, ``svrs``, ``distributed_sarah``) — where the server
holds the model and clients only ship per-batch gradients — FL-SILVER keeps a
**persistent per-client gradient memory** ``y_i`` (an estimate of ∇f_i at the
last point client ``i`` was refreshed) and runs the SARAH variance-reduction
*locally* on a single sampled client each round.

Structure of one round ``t`` (Algorithm 2 lines 5–20):

    g_bar = (1/P) Σ_{i=1}^P y_i^{t-1}          # server broadcasts the aggregate
    sample one client i_t
    x^{t,0} = x^{t-1},  z^{t,0} = 0            # local SARAH corrector starts at 0
    for k = 1 … K:                            # K local steps on client i_t
        x^{t,k} = x^{t,k-1} − η·(g_bar + z^{t,k-1}) + ξ^{t,k}
        sample local minibatch J (size b)
        z^{t,k} = z^{t,k-1}
                  + (1/b) Σ_{j∈J} (∇f_{i_t,j}(x^{t,k}) − ∇f_{i_t,j}(x^{t,k-1}))
    x^t = x^{t,K}
    sample p clients I^t and refresh their memory at x^t (lines 15–18):
        y_i^t   = (1/Kb) Σ ∇f_{i,j}(x^t)                      # fresh anchor
        Δy_i^t  = (1/Kb) Σ (∇f_{i,j}(x^t) − ∇f_{i,j}(x^{t-1})) # same batch
    memory update (lines 19–20):
        y_i^t = y_i^{t-1} + (1/p) Σ_{i∈I^t} Δy_i^t     for i ∉ I^t
    server aggregate updated accordingly.

The **similarity assumption** is what makes line 20 legal: because the local
functions are second-order similar, the gradient *change* between x^{t-1} and
x^t is nearly the same for every client, so the ``(P − p)`` clients that were
*not* re-queried this round can advance their memory by the mean observed
change ``(1/p) Σ_{i∈I^t} Δy_i`` instead of paying a full re-evaluation.

``ξ^{t,k}`` is the algorithm's injected Gaussian perturbation (the Langevin /
sampling term).  For a pure-optimization baseline directly comparable to the
other methods leave ``noise_std = 0.0`` (ξ ≡ 0); set it > 0 to recover the
sampling variant, where ξ^{t,k} ~ √(2η)·noise_std·N(0, I).

Within one local step the *same* minibatch ``J`` is reused at ``x^{t,k}`` and
``x^{t,k-1}`` (and, for the refresh, the same anchor minibatches are reused at
``x^t`` and ``x^{t-1}``) so that the SARAH difference telescopes into a
low-variance estimate — mirroring the ``xy``-reuse convention used elsewhere in
this repo.

Memory footprint
----------------
With ``d`` = number of model parameters and ``P`` participants, FL-SILVER
stores the full client-memory table ``{y_i}_{i=1}^P`` plus the running
aggregate — i.e. ``(P + 1)·d`` floats of *persistent* state, versus the
``O(d)`` (a handful of param-vectors) carried by ``batched_nfg_sarah`` /
``svrs`` / ``distributed_sarah``.  See the module-level constant
:data:`PERSISTENT_PARAM_VECTORS` and ``docs`` for the concrete numbers.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    clone_params,
    compute_batch_gradient,
    compute_param_norm,
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class FLSilver(BaseAlgorithm):
    """FL-SILVER: local SARAH variance reduction with per-client memory."""

    def __init__(
        self,
        lr: float,
        num_local_steps: int,
        num_refresh_clients: int,
        rounds_per_epoch: int,
        anchor_batches: int = 1,
        noise_std: float = 0.0,
        include_server: bool = True,
    ) -> None:
        # η — local step size.
        self.lr = float(lr)
        # K — number of local SARAH steps per round.
        self.num_local_steps = int(num_local_steps)
        # p — number of client memories refreshed per round (|I^t|).
        self.num_refresh_clients = int(num_refresh_clients)
        # T — number of rounds per outer epoch.
        self.rounds_per_epoch = int(rounds_per_epoch)
        # Number of minibatches averaged for the anchor y_i (approximates the
        # paper's "size Kb" full-anchor batch; 1 ⇒ a single minibatch).
        self.anchor_batches = max(1, int(anchor_batches))
        # Std of the injected Gaussian perturbation ξ (0 ⇒ pure optimization).
        self.noise_std = float(noise_std)
        # Whether the server's own local function f₁ participates as one of the
        # P nodes (keeps parity with distributed_sarah's total_nodes = M + 1
        # convention and uses the server partition's data).
        self.include_server = bool(include_server)

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")

        # Participant loaders: [server] + clients when include_server else clients.
        self.participant_loaders: list[DataLoader] = []
        self.num_participants: int = 0  # P

        # Persistent state (the FL-SILVER memory).
        self._y: list[ParamList] = []      # per-participant memory {y_i}
        self._y_sum: ParamList = []        # Σ_i y_i  (server aggregate)
        self._initialized: bool = False

    # ------------------------------------------------------------------
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
        # Kept for API parity; FL-SILVER has no proximal operator.
        self.server_prox_loader = server_prox_loader
        self.client_loaders = client_loaders
        self.loss_fn = loss_fn
        self.device = device

        if self.include_server:
            self.participant_loaders = [server_grad_loader, *client_loaders]
        else:
            self.participant_loaders = list(client_loaders)
        self.num_participants = len(self.participant_loaders)

        if self.num_refresh_clients > self.num_participants:
            logger.warning(
                "FL-SILVER: num_refresh_clients=%d > P=%d; clamping to P.",
                self.num_refresh_clients, self.num_participants,
            )
            self.num_refresh_clients = self.num_participants

        # Memory is built lazily on the first ``run_epoch`` so it anchors at the
        # actual starting weights (which may come from a checkpoint).
        self._y = []
        self._y_sum = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------
    def _sample_anchor_batches(
        self, loader: DataLoader,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-sample ``anchor_batches`` minibatches from *loader*."""
        it = iter(loader)
        xys: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.anchor_batches):
            try:
                xys.append(next(it))
            except StopIteration:
                it = iter(loader)
                xys.append(next(it))
        return xys

    def _avg_grad_on(
        self,
        loader: DataLoader,
        xys: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> ParamList:
        """Mean gradient over the given fixed minibatches at the current w.

        Approximates the paper's ``(1/Kb) Σ_{j∈J} ∇f_{i,j}(w)`` anchor: with
        ``anchor_batches`` minibatches of size ``b`` this averages ``anchor_batches·b``
        samples, so ``anchor_batches`` plays the role of ``K`` in "size Kb".
        """
        acc = zeros_like_params(self.model)
        for xy in xys:
            g = compute_batch_gradient(
                self.model, loader, self.loss_fn, self.device, xy=xy,
            )
            add_params_(acc, g, alpha=1.0 / len(xys))
        return acc

    def _init_memory(self) -> None:
        """Lines 1–4: y_i⁰ = (1/Kb) Σ ∇f_{i,j}(x⁰) for every participant."""
        w0 = get_params(self.model)
        self._y = []
        self._y_sum = zeros_like_params(self.model)
        for loader in self.participant_loaders:
            set_params(self.model, w0)
            xys = self._sample_anchor_batches(loader)
            y_i = self._avg_grad_on(loader, xys)
            self._y.append(y_i)
            add_params_(self._y_sum, y_i)
        set_params(self.model, w0)
        self._initialized = True
        logger.info(
            "FL-SILVER memory initialised: P=%d participants, "
            "persistent state = (P+1)·d = %d param-vectors, ‖(1/P)Σy_i‖=%.4e",
            self.num_participants, self.num_participants + 1,
            compute_param_norm(self._y_sum) / max(self.num_participants, 1),
        )

    def _maybe_noise(self, reference: ParamList) -> ParamList | None:
        """ξ^{t,k} ~ √(2η)·noise_std·N(0, I), or None when disabled."""
        if self.noise_std <= 0.0:
            return None
        scale = (2.0 * self.lr) ** 0.5 * self.noise_std
        return [scale * torch.randn_like(p) for p in reference]

    # ------------------------------------------------------------------
    # Local SARAH training on one client (lines 8–13).
    # ------------------------------------------------------------------
    def _local_train(self, i_t: int, g_bar: ParamList) -> tuple[ParamList, float]:
        """K local steps of variance-reduced SGD on client ``i_t``.

        Returns the final iterate ``x^t`` and the terminal ‖z^{t,K}‖ (for
        diagnostics).  The model is left at ``x^t``.
        """
        loader = self.participant_loaders[i_t]
        x_cur = get_params(self.model)  # x^{t,0} = x^{t-1}
        z = zeros_like_params(self.model)  # z^{t,0} = 0

        for _k in range(self.num_local_steps):
            # x^{t,k} = x^{t,k-1} − η·(g_bar + z^{t,k-1}) + ξ^{t,k}
            noise = self._maybe_noise(x_cur)
            x_new = [
                xc - self.lr * (gb + zi)
                for xc, gb, zi in zip(x_cur, g_bar, z)
            ]
            if noise is not None:
                for xn, ns in zip(x_new, noise):
                    xn.add_(ns)

            # Same minibatch J at x^{t,k} and x^{t,k-1} → the SARAH difference
            # telescopes (low-variance corrector).
            xy = next(iter(loader))
            set_params(self.model, x_new)
            g_new = compute_batch_gradient(
                self.model, loader, self.loss_fn, self.device, xy=xy,
            )
            set_params(self.model, x_cur)
            g_old = compute_batch_gradient(
                self.model, loader, self.loss_fn, self.device, xy=xy,
            )
            # z^{t,k} = z^{t,k-1} + (∇f(x^{t,k}) − ∇f(x^{t,k-1}))
            for zi, gn, go in zip(z, g_new, g_old):
                zi.add_(gn - go)

            x_cur = x_new

        set_params(self.model, x_cur)  # leave model at x^t = x^{t,K}
        return x_cur, compute_param_norm(z)

    # ------------------------------------------------------------------
    # Memory refresh (lines 14–20).
    # ------------------------------------------------------------------
    def _refresh_memory(self, x_t: ParamList, x_prev: ParamList) -> None:
        """Refresh p sampled clients at ``x_t`` and shift the rest by mean Δ."""
        P = self.num_participants
        p = self.num_refresh_clients
        refreshed = torch.randperm(P)[:p].tolist()

        fresh_y: dict[int, ParamList] = {}
        avg_delta = zeros_like_params(self.model)  # (1/p) Σ Δy_i

        for i in refreshed:
            loader = self.participant_loaders[i]
            xys = self._sample_anchor_batches(loader)
            # y_i^t = (1/Kb) Σ ∇f_{i,j}(x^t)   (line 17)
            set_params(self.model, x_t)
            y_new = self._avg_grad_on(loader, xys)
            # Δy_i^t = y_i(x^t) − y_i(x^{t-1}) on the SAME batches (line 18)
            set_params(self.model, x_prev)
            y_prev = self._avg_grad_on(loader, xys)
            fresh_y[i] = y_new
            for a, yn, yp in zip(avg_delta, y_new, y_prev):
                a.add_((yn - yp), alpha=1.0 / p)

        # ── Update the aggregate Σ_i y_i (server side) ──────────────────
        #   i ∈ I^t : y_i ← y_new                    (fresh anchor)
        #   i ∉ I^t : y_i ← y_i + avg_delta          (similarity shift, line 20)
        # New Σ = old Σ + (P − p)·avg_delta + Σ_{i∈I^t}(y_new − y_old).
        add_params_(self._y_sum, avg_delta, alpha=float(P - p))
        for i in refreshed:
            for s, yn, yo in zip(self._y_sum, fresh_y[i], self._y[i]):
                s.add_(yn - yo)

        # ── Update the per-client memory table ──────────────────────────
        refreshed_set = set(refreshed)
        for i in range(P):
            if i in refreshed_set:
                self._y[i] = fresh_y[i]
            else:
                add_params_(self._y[i], avg_delta)

        set_params(self.model, x_t)  # leave model at x^t

    # ------------------------------------------------------------------
    # Main epoch.
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None
        if not self._initialized:
            self._init_memory()

        P = self.num_participants
        T = self.rounds_per_epoch

        z_norms: list[float] = []
        step_norms: list[float] = []
        g_bar_norms: list[float] = []

        for t in range(T):
            # g_bar = (1/P) Σ_i y_i^{t-1}  (broadcast anchor, fixed for round)
            g_bar = clone_params(self._y_sum)
            for gb in g_bar:
                gb.mul_(1.0 / P)
            g_bar_norms.append(compute_param_norm(g_bar))

            i_t = int(torch.randint(low=0, high=P, size=(1,)).item())
            x_prev = get_params(self.model)  # x^{t-1}

            x_t, z_norm = self._local_train(i_t, g_bar)
            z_norms.append(z_norm)
            step_norms.append(
                compute_param_norm([xt - xp for xt, xp in zip(x_t, x_prev)]),
            )

            self._refresh_memory(x_t, x_prev)

            logger.debug(
                "FL-SILVER epoch %d round %d/%d  i_t=%d  ‖g_bar‖=%.3e  "
                "‖z‖=%.3e  ‖x^t-x^{t-1}‖=%.3e",
                epoch, t + 1, T, i_t,
                g_bar_norms[-1], z_norm, step_norms[-1],
            )

        def _avg(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        return {
            "epoch": float(epoch),
            "rounds": float(T),
            "inner_steps": float(T * self.num_local_steps),
            "g_bar_norm_mean": _avg(g_bar_norms),
            "z_norm_mean": _avg(z_norms),
            "step_norm_mean": _avg(step_norms),
            "step_norm_last": float(step_norms[-1]) if step_norms else 0.0,
            "y_sum_norm": compute_param_norm(self._y_sum),
            "param_norm": compute_param_norm(get_params(self.model)),
        }
