"""Batched No Full Grad SARAH — the main algorithm.

Implements *Batched No Full Grad SARAH* (Algorithm 1):

    Input  : initial point w_0^{(0)} ∈ ℝ^d
             initial gradient estimators tilde_v_1^{(0)} = 0, v_0^{(0)} = 0
    Param  : stepsize θ, batch size B (clients per inner step)

    For each epoch s = 0, 1, …, S:
        Sample batches B_1^{(s)}, …, B_K^{(s)},  K = ⌈n / B⌉
        v_0^{(s)}   = v^{(s)}                                 (carry over)
        w_1^{(s)}   = prox_{θ f₁}( w_0^{(s)} − θ · v_0^{(s)} )

        For t = 1, 2, …, K:
            tilde_v_{t+1}^{(s)} = ((t-1)/t) · tilde_v_t^{(s)}
                                 + (1/(t·B)) · Σ_{i ∈ B_t} ∇(f_i − f₁)(w_t^{(s)})

            v_t^{(s)}            = v_{t-1}^{(s)}
                                 + (1/(n·B)) · Σ_{i ∈ B_t}
                                     [ ∇(f_i − f₁)(w_t^{(s)})
                                       − ∇(f_i − f₁)(w_{t-1}^{(s)}) ]

            w_{t+1}^{(s)}        = prox_{θ f₁}( w_t^{(s)} − θ · v_t^{(s)} )

        w_0^{(s+1)}     = w_{K+1}^{(s)}
        tilde_v_1^{(s+1)} = 0
        v^{(s+1)}        = tilde_v_{K+1}^{(s)}

Within one inner step the *same* minibatch is reused at ``w_t`` and
``w_{t-1}`` for every client and for the server, so that the SARAH
difference ∇(f_i − f₁)(w_t) − ∇(f_i − f₁)(w_{t-1}) actually telescopes
(otherwise the recursion becomes a noisy SGD estimator with no variance
reduction).  Concretely, we pre-sample one ``srv_xy`` from the server's
*gradient* loader and one ``cli_xy[cid]`` per client in the batch, then
evaluate all four gradient quantities on those fixed tensors.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.runtime.prox_solver import ProxSolver
from similarity_sarah.runtime.scheduler import sample_client_batches
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    clone_params,
    compute_batch_gradient,
    compute_full_gradient,
    compute_param_norm,
    diff_param_norm,
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class BatchedNoFullGradSARAH(BaseAlgorithm):
    """Batched No Full Grad SARAH.

    The server holds the global model and performs every update step;
    clients only return per-batch gradients ∇f_i(w_t), ∇f_i(w_{t-1})
    computed on the same local minibatch.  No full-gradient
    synchronisation is required.

    Args:
        theta: Proximal step size θ.
        batch_size_clients: Number of clients sampled per inner step (B).
        prox_solver: Concrete :class:`ProxSolver` used for prox_{θ f₁}.
        update_v_tilde_in_the_end: If True, override the recursive
            tilde_v at end of epoch with a fresh stochastic anchor
            computed at the final iterate w_final:

                v_epoch_new = (1/n) Σ_i ∇(f_i − f_1)(w_final)

            using ONE minibatch per client (and one minibatch on the
            server) — i.e. the same ``compute_batch_gradient`` used by
            the recursive update, just evaluated once at the
            end-of-epoch point.  Cost: 1 + M extra mini-batch
            forward+backward passes per epoch.  The recursive
            ``tilde_v`` is *still* computed during the epoch — only its
            role as "carry-over to next epoch" is replaced.  When False
            (default) keeps the literal pseudocode behaviour.
        clip_number_of_clients_with_reshuffle: If True, an epoch
            traverses only ``clip_clients_per_epoch`` clients (default
            3) instead of all of them.  Clients are picked from a
            *persistent* random permutation of ``[0, num_clients)``: the
            first ``num_clients // clip_clients_per_epoch`` epochs use
            consecutive non-overlapping slices of length
            ``clip_clients_per_epoch`` from that permutation; once the
            permutation is exhausted, a fresh one is sampled (the
            leftover ``num_clients % clip_clients_per_epoch`` clients
            from each permutation are skipped — matches the user spec
            "первые 9 клиентов, и делим их на три группы").  At the end
            of every epoch the recursive ``tilde_v`` is multiplied by
            ``n / clip_clients_per_epoch`` so its magnitude scales as
            if it had been built from all ``n`` clients.  Requires
            ``batch_size_clients == 1``.  Default False.
        clip_clients_per_epoch: Group size for the flag above (default 3).
        log_deviation: If True, at the start of every epoch compute the
            *exact* lemma reference anchor
            ``g_exact = ∇(f − f_1)(w_0^{(s)})
                     = (1/n) Σ_{i=2..n} (∇f_i − ∇f_1)(w_0^{(s)})``
            via :func:`compute_full_gradient` on every node, and log the
            deviation of the carry-over estimator
            ``v_0^{(s)} = self.v_epoch`` from this exact value in three
            complementary flavours:

              * absolute      ``‖v_0 − g_exact‖`` (and its square),
              * proportional  ``‖v_0 − g_exact‖ / ‖g_exact‖``,
              * directional   ``cos∠(v_0, g_exact)`` and the angle in °.

            The reference is *average gradient minus server gradient*
            ``∇(f − f_1)`` — the exact object the SARAH recursion tries
            to estimate and the one bounded by the epoch-initial
            variance lemma — *not* the plain full gradient ``∇f``.
            Useful as a diagnostic for how close the running-mean
            estimator stays to the true value over training.  Cost: one
            full pass over the server's grad loader plus one full pass
            per client per epoch (i.e. an SVRS-style anchor refresh, but
            only for logging — not used in the update).  Default False.
        log_deviation_inner: If True, additionally measure the same
            deviation of the *running* estimator ``v_t`` from
            ``∇(f − f_1)(w_t)`` at **every inner step** ``t`` of the
            epoch (evaluated at the current iterate ``w_t`` just before
            the prox move).  Per-epoch aggregates (mean / max / last of
            the proportional and angular deviation, plus the start-of-
            epoch point) are returned as scalar metrics, and the full
            within-epoch curve is stashed in ``self.last_inner_deviation``
            so the runner can push it to W&B as a table + line plots.
            Implies ``log_deviation`` for the start-of-epoch point.
            *Very* expensive — one exact full-gradient anchor per inner
            step (K SVRS-style refreshes per epoch) — meant only for
            paper-figure / diagnostic runs.  Default False.
    """

    def __init__(
        self,
        theta: float,
        batch_size_clients: int,
        prox_solver: ProxSolver,
        update_v_tilde_in_the_end: bool = False,
        clip_number_of_clients_with_reshuffle: bool = False,
        clip_clients_per_epoch: int = 3,
        log_deviation: bool = False,
        log_deviation_inner: bool = False,
    ) -> None:
        self.theta = theta
        self.batch_size_clients = batch_size_clients
        self.prox_solver = prox_solver
        self.update_v_tilde_in_the_end = bool(update_v_tilde_in_the_end)
        self.clip_number_of_clients_with_reshuffle = bool(
            clip_number_of_clients_with_reshuffle
        )
        self.clip_clients_per_epoch = int(clip_clients_per_epoch)
        self.log_deviation_inner = bool(log_deviation_inner)
        # Inner-step logging needs the start-of-epoch anchor too.
        self.log_deviation = bool(log_deviation) or self.log_deviation_inner
        # Within-epoch deviation curve of the most recent epoch, stashed for
        # the runner to forward to W&B (list of per-inner-step metric dicts,
        # or None).  See ``log_deviation_inner``.
        self.last_inner_deviation: list[dict[str, float]] | None = None

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.v_epoch: ParamList = []
        self.num_clients: int = 0
        self.total_nodes: int = 0

        # Persistent permutation state for ``clip_number_of_clients_with_reshuffle``.
        self._client_permutation: list[int] = []
        self._client_perm_offset: int = 0

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
        # ``n`` in the pseudocode = total number of functions (server's f₁
        # plus one per client).
        self.total_nodes = self.num_clients + 1

        self.v_epoch = zeros_like_params(model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _client_grad_sum(
        self,
        client_ids: list[int],
        xys: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> ParamList:
        """Σ_{i ∈ batch} ∇f_i(w) at the current model state.

        If ``xys`` is given, uses one pre-sampled minibatch per client
        (required by the SARAH telescope — same samples at the two
        evaluation points within one inner step).
        """
        acc = zeros_like_params(self.model)
        if xys is None:
            for cid in client_ids:
                g = compute_batch_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                add_params_(acc, g)
        else:
            for cid, xy in zip(client_ids, xys):
                g = compute_batch_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                    xy=xy,
                )
                add_params_(acc, g)
        return acc

    def _server_batch_grad(
        self,
        xy: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ParamList:
        return compute_batch_gradient(
            self.model, self.server_grad_loader,
            self.loss_fn, self.device,
            xy=xy,
        )

    # ------------------------------------------------------------------
    # End-of-epoch fresh anchor for v_tilde (optional, see __init__ docstring).
    # ------------------------------------------------------------------
    def _compute_end_of_epoch_v_tilde(self) -> ParamList:
        """Recompute ``v_tilde`` at the end of an epoch using a *single
        minibatch* per node at the current model state ``w_final``:

            v_tilde_new = (1/n) Σ_{i=2..n} ∇(f_i − f_1)(w_final)

        Each client runs ``compute_batch_gradient`` on one fresh minibatch
        from its local loader; the server does the same on its grad
        loader.  This is a *stochastic* one-shot estimate of
        ``∇f − ∇f_1`` (matches the spirit of the recursive update
        during the epoch, which also uses one minibatch per client).
        Cheap (1+M forward+backward passes per epoch) — much lighter
        than a full-gradient anchor but noisier.

        Server contribution ``∇(f_1 − f_1) = 0`` is implicit, so we sum
        only over clients with weight ``1/n`` (algebraic identity:
        ``∇f − ∇f_1 = (1/n) Σ_{i=2..n} (∇f_i − ∇f_1)``).
        """
        n = self.total_nodes
        # ∇f_1(w_final) on one minibatch from the deterministic server loader.
        grad_f1 = compute_batch_gradient(
            self.model, self.server_grad_loader,
            self.loss_fn, self.device,
        )
        # (1/n) Σ_{i=2..n} ∇(f_i − f_1)(w_final), one minibatch per client.
        accum = zeros_like_params(self.model)
        for client_loader in self.client_loaders:
            grad_i = compute_batch_gradient(
                self.model, client_loader, self.loss_fn, self.device,
            )
            for a, gi, g1 in zip(accum, grad_i, grad_f1):
                a.add_(gi - g1, alpha=1.0 / n)
        return accum

    # ------------------------------------------------------------------
    # Exact full-gradient anchor (diagnostic only — see ``log_deviation``).
    # ------------------------------------------------------------------
    def _compute_exact_full_grad_diff(self) -> ParamList:
        """``g_exact = (1/n) Σ_{i=2..n} (∇f_i − ∇f_1)(w_current)`` exactly.

        One full deterministic pass over the server's grad loader and
        each client's loader — same construction as SVRS's
        ``_refresh_anchor``.  Does not modify the model parameters.
        """
        n = self.total_nodes
        grad_f1 = compute_full_gradient(
            self.model, self.server_grad_loader, self.loss_fn, self.device,
        )
        accum = zeros_like_params(self.model)
        for client_loader in self.client_loaders:
            grad_i = compute_full_gradient(
                self.model, client_loader, self.loss_fn, self.device,
            )
            for a, gi, g1 in zip(accum, grad_i, grad_f1):
                a.add_(gi - g1, alpha=1.0 / n)
        return accum

    # ------------------------------------------------------------------
    # Deviation of an estimator ``v`` from the lemma reference ``g``.
    # ------------------------------------------------------------------
    @staticmethod
    def _deviation_metrics(v: ParamList, g: ParamList) -> dict[str, float]:
        """Deviation of estimator ``v`` from reference ``g = ∇(f − f_1)``.

        Returns the three complementary flavours used throughout the
        deviation diagnostics:

          * ``dev_norm`` / ``dev_sq`` — absolute ``‖v − g‖`` (and square),
          * ``dev_rel``               — proportion ``‖v − g‖ / ‖g‖``,
          * ``cos_sim`` / ``angle_deg`` — direction (cosine and angle in °).

        Also reports ``est_norm = ‖v‖`` and ``ref_norm = ‖g‖``.  When a
        norm is zero the ratio / angle are ``nan`` (W&B renders a gap).
        """
        diff_sq = 0.0
        dot = 0.0
        v_sq = 0.0
        g_sq = 0.0
        for vi, gi in zip(v, g):
            diff_sq += (vi - gi).square().sum().item()
            dot += (vi * gi).sum().item()
            v_sq += vi.square().sum().item()
            g_sq += gi.square().sum().item()

        est_norm = float(v_sq ** 0.5)
        ref_norm = float(g_sq ** 0.5)
        dev_norm = float(diff_sq ** 0.5)
        dev_rel = dev_norm / ref_norm if ref_norm > 0.0 else float("nan")
        denom = est_norm * ref_norm
        if denom > 0.0:
            cos_sim = max(-1.0, min(1.0, dot / denom))
            angle_deg = math.degrees(math.acos(cos_sim))
        else:
            cos_sim = float("nan")
            angle_deg = float("nan")
        return {
            "dev_sq": float(diff_sq),
            "dev_norm": dev_norm,
            "dev_rel": float(dev_rel),
            "cos_sim": float(cos_sim),
            "angle_deg": float(angle_deg),
            "est_norm": est_norm,
            "ref_norm": ref_norm,
        }

    # ------------------------------------------------------------------
    # Persistent-permutation slicing for ``clip_number_of_clients_with_reshuffle``.
    # ------------------------------------------------------------------
    def _next_clipped_client_slice(self) -> list[int]:
        """Return the next ``clip_clients_per_epoch`` client indices.

        Uses ``self._client_permutation`` as a persistent shuffle: each
        call advances by ``clip_clients_per_epoch`` consecutive indices.
        When the remaining tail is shorter than the slice size, a fresh
        permutation is sampled (the tail itself is *skipped* — matches
        the user's spec "первые 9 клиентов, делим на три группы").
        """
        K = self.clip_clients_per_epoch
        if (
            not self._client_permutation
            or self._client_perm_offset + K > len(self._client_permutation)
        ):
            self._client_permutation = torch.randperm(self.num_clients).tolist()
            self._client_perm_offset = 0
        slice_ = self._client_permutation[
            self._client_perm_offset : self._client_perm_offset + K
        ]
        self._client_perm_offset += K
        return slice_

    # ------------------------------------------------------------------
    # Main epoch
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None

        if self.clip_number_of_clients_with_reshuffle:
            if self.batch_size_clients != 1:
                raise ValueError(
                    "clip_number_of_clients_with_reshuffle requires "
                    "batch_size_clients == 1, got "
                    f"{self.batch_size_clients}"
                )
            client_slice = self._next_clipped_client_slice()
            batches = [[cid] for cid in client_slice]
            logger.info(
                "Epoch %d clipping clients (%d/%d): perm_offset=%d, slice=%s",
                epoch, len(client_slice), self.num_clients,
                self._client_perm_offset - len(client_slice), client_slice,
            )
        else:
            batches = sample_client_batches(
                self.num_clients, self.batch_size_clients,
            )
        K = len(batches)
        n = self.total_nodes
        B = self.batch_size_clients  # paper's *fixed* B (used in coefficients)

        # tilde_v_1^{(s)} = 0,  v_0^{(s)} = v^{(s)} (carry over)
        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(self.model)

        # ── Optional diagnostic: deviation of v_0^{(s)} from the lemma
        #    reference g = ∇(f − f_1)(w_0^{(s)}) ────────────────────────
        # Compares the carry-over estimator to the *exact* average-minus-
        # server gradient anchor at the start-of-epoch iterate (the object
        # bounded by the epoch-initial variance lemma).  Model is currently
        # at w_0^{(s)} so the gradient is taken at the right point.  Skipped
        # by default — flag is for diagnostic runs only.
        v0_norm_at_start = compute_param_norm(v)
        v0_metrics: dict[str, float] = {}
        # Within-epoch curve: start-of-epoch point (inner_step 0) then one
        # row per inner step (filled below when ``log_deviation_inner``).
        deviation_series: list[dict[str, float]] = []
        if self.log_deviation:
            exact_grad_diff = self._compute_exact_full_grad_diff()
            v0_metrics = self._deviation_metrics(v, exact_grad_diff)
            deviation_series.append({"inner_step": 0.0, **v0_metrics})
            logger.info(
                "Epoch %d deviation@start: ‖v_0‖=%.4e  ‖g‖=%.4e  "
                "‖v_0-g‖=%.4e  rel=%.4f  cos=%.4f  angle=%.1f°",
                epoch, v0_metrics["est_norm"], v0_metrics["ref_norm"],
                v0_metrics["dev_norm"], v0_metrics["dev_rel"],
                v0_metrics["cos_sim"], v0_metrics["angle_deg"],
            )

        # w_0 for the first inner step (used as w_{t-1} when t=1).
        w_prev = get_params(self.model)

        # w_1^{(s)} = prox_{θ f₁}( w_0^{(s)} − θ · v_0^{(s)} )
        prox_diag_first = self.prox_solver.step(
            self.model, v, self.theta,
            self.server_prox_loader, self.loss_fn, self.device,
            eval_loader=self.server_grad_loader,
        )
        logger.info(
            "Epoch %d step 0/%d (initial prox): prox_grad_first=%.4f  prox_grad_last=%.4f  "
            "ratio=%.3f  clip_frac=%.2f",
            epoch, K,
            prox_diag_first["prox_grad_norm_first"],
            prox_diag_first["prox_grad_norm_last"],
            prox_diag_first["prox_grad_norm_ratio"],
            prox_diag_first.get("prox_clip_frac", 0.0),
        )

        # Aggregated diagnostics across the epoch.
        prox_first_norms: list[float] = [prox_diag_first["prox_grad_norm_first"]]
        prox_last_norms: list[float] = [prox_diag_first["prox_grad_norm_last"]]
        prox_ratios: list[float] = [prox_diag_first["prox_grad_norm_ratio"]]
        prox_obj_decreases: list[float] = [prox_diag_first["prox_obj_decrease"]]
        prox_clip_fracs: list[float] = [prox_diag_first.get("prox_clip_frac", 0.0)]
        step_norms: list[float] = []

        for t in range(1, K + 1):
            batch = batches[t - 1]
            B_actual = len(batch)
            w_curr = get_params(self.model)  # = w_t^{(s)}

            # ── pre-sample one minibatch per server/client for this step ──
            # Reused at both w_t and w_{t-1} so that the SARAH difference
            # telescopes into a low-variance estimate.
            srv_xy = next(iter(self.server_grad_loader))
            cli_xys = [next(iter(self.client_loaders[cid])) for cid in batch]

            # ── gradients at w_t (model is already at w_t) ───────────
            grad_f1_curr = self._server_batch_grad(xy=srv_xy)
            sum_client_grads_curr = self._client_grad_sum(batch, xys=cli_xys)

            # ── gradients at w_{t-1} on the SAME minibatches ─────────
            set_params(self.model, w_prev)
            grad_f1_prev = self._server_batch_grad(xy=srv_xy)
            sum_client_grads_prev = self._client_grad_sum(batch, xys=cli_xys)

            # Σ_{i ∈ B_t}( ∇f_i − ∇f₁ )(w_t) and (w_{t-1})
            sum_diff_curr = [
                sc - B_actual * g1
                for sc, g1 in zip(sum_client_grads_curr, grad_f1_curr)
            ]
            sum_diff_prev = [
                sp - B_actual * g1
                for sp, g1 in zip(sum_client_grads_prev, grad_f1_prev)
            ]

            # ── update tilde_v_{t+1} = ((t-1)/t) tilde_v_t + (1/(tB)) Σ_t
            ratio = (t - 1) / t
            coeff_tilde = 1.0 / (t * B)
            for tv, sdc in zip(tilde_v, sum_diff_curr):
                tv.mul_(ratio).add_(sdc, alpha=coeff_tilde)

            # ── update v_t = v_{t-1} + (1/(nB)) Σ (curr − prev) ──────
            coeff_v = 1.0 / (n * B)
            for vi, sdc, sdp in zip(v, sum_diff_curr, sum_diff_prev):
                vi.add_(sdc - sdp, alpha=coeff_v)

            # Roll forward bookkeeping for the next iteration.
            w_prev = w_curr

            # w_{t+1} = prox_{θ f₁}( w_t − θ v_t )
            set_params(self.model, w_curr)

            # ── Optional diagnostic: deviation of the running estimator
            #    v_t from ∇(f − f_1)(w_t) at the current iterate ───────
            # Model is at w_curr = w_t here (restored above, prox has not
            # moved it yet), so the exact anchor is taken at the right
            # point.  ``_compute_exact_full_grad_diff`` only reads
            # gradients (autograd.grad) — it does not mutate params.
            if self.log_deviation_inner:
                g_t = self._compute_exact_full_grad_diff()
                m_t = self._deviation_metrics(v, g_t)
                deviation_series.append({"inner_step": float(t), **m_t})

            prox_diag = self.prox_solver.step(
                self.model, v, self.theta,
                self.server_prox_loader, self.loss_fn, self.device,
                eval_loader=self.server_grad_loader,
            )

            prox_first_norms.append(prox_diag["prox_grad_norm_first"])
            prox_last_norms.append(prox_diag["prox_grad_norm_last"])
            prox_ratios.append(prox_diag["prox_grad_norm_ratio"])
            prox_obj_decreases.append(prox_diag["prox_obj_decrease"])
            prox_clip_fracs.append(prox_diag.get("prox_clip_frac", 0.0))
            step_norms.append(diff_param_norm(get_params(self.model), w_curr))

            logger.info(
                "Epoch %d step %d/%d: prox_grad_first=%.4f  prox_grad_last=%.4f  "
                "ratio=%.3f  clip_frac=%.2f  ‖v‖=%.3e  ‖w_{t+1}-w_t‖=%.3e",
                epoch, t, K,
                prox_first_norms[-1], prox_last_norms[-1],
                prox_ratios[-1], prox_clip_fracs[-1],
                compute_param_norm(v), step_norms[-1],
            )

        # v^{(s+1)} carry-over to next epoch.
        # Default path: v_epoch = tilde_v_{K+1} (literal pseudocode).
        # Alternative: refresh v_tilde at the very end with exact
        # full-gradient values at w_final (see _compute_end_of_epoch_v_tilde).
        if self.update_v_tilde_in_the_end:
            v_tilde_recursive_norm = compute_param_norm(tilde_v)
            self.v_epoch = self._compute_end_of_epoch_v_tilde()
            v_tilde_refreshed_norm = compute_param_norm(self.v_epoch)
            logger.info(
                "Epoch %d end-of-epoch v_tilde refresh: "
                "‖tilde_v_recursive‖=%.4f → ‖v_tilde_refreshed‖=%.4f",
                epoch, v_tilde_recursive_norm, v_tilde_refreshed_norm,
            )
        elif self.clip_number_of_clients_with_reshuffle:
            # tilde_v built from only ``clip_clients_per_epoch`` clients;
            # rescale by ``n / clip_clients_per_epoch`` so its magnitude
            # matches a full-traversal carry-over.
            scale = n / self.clip_clients_per_epoch
            scaled = clone_params(tilde_v)
            for t_ in scaled:
                t_.mul_(scale)
            logger.info(
                "Epoch %d clipped v_tilde rescale x%.4f: "
                "‖tilde_v_recursive‖=%.4f → ‖v_epoch‖=%.4f",
                epoch, scale,
                compute_param_norm(tilde_v), compute_param_norm(scaled),
            )
            self.v_epoch = scaled
        else:
            self.v_epoch = tilde_v

        def _avg(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        metrics: dict[str, float] = {
            "epoch": float(epoch),
            "inner_steps": float(K),
            "v_norm": compute_param_norm(v),
            "tilde_v_norm": compute_param_norm(tilde_v),
            "v_epoch_norm": compute_param_norm(self.v_epoch),
            "v_tilde_refreshed": float(self.update_v_tilde_in_the_end),
            "param_norm": compute_param_norm(get_params(self.model)),
            "step_norm_mean": _avg(step_norms),
            "step_norm_last": float(step_norms[-1]) if step_norms else 0.0,
            "prox_grad_norm_first_mean": _avg(prox_first_norms),
            "prox_grad_norm_last_mean": _avg(prox_last_norms),
            "prox_grad_norm_ratio_mean": _avg(prox_ratios),
            "prox_obj_decrease_mean": _avg(prox_obj_decreases),
            "prox_clip_frac_mean": _avg(prox_clip_fracs),
            "v0_norm_at_start": v0_norm_at_start,
        }

        # ── Start-of-epoch deviation of v_0 from g = ∇(f-f_1)(w_0). ──
        # Backward-compatible keys (…_sq, …_norm, exact_grad_diff_norm) plus
        # the proportional and directional flavours.  0.0 / nan when
        # ``log_deviation`` is False (cheap default).
        if v0_metrics:
            metrics.update({
                "v0_deviation_sq": v0_metrics["dev_sq"],
                "v0_deviation_norm": v0_metrics["dev_norm"],
                "exact_grad_diff_norm": v0_metrics["ref_norm"],
                "v0_deviation_rel": v0_metrics["dev_rel"],
                "v0_cos_sim": v0_metrics["cos_sim"],
                "v0_angle_deg": v0_metrics["angle_deg"],
            })
        else:
            metrics.update({
                "v0_deviation_sq": 0.0,
                "v0_deviation_norm": 0.0,
                "exact_grad_diff_norm": 0.0,
            })

        # ── Within-epoch deviation aggregates (log_deviation_inner). ──
        # Aggregate over the inner steps t = 1..K (the start-of-epoch point
        # at inner_step 0 is reported separately by the v0_* keys above).
        inner_rows = [r for r in deviation_series if r["inner_step"] > 0.0]
        if inner_rows:
            rel = [r["dev_rel"] for r in inner_rows]
            cos = [r["cos_sim"] for r in inner_rows]
            ang = [r["angle_deg"] for r in inner_rows]
            dnorm = [r["dev_norm"] for r in inner_rows]
            metrics.update({
                "inner_dev_rel_mean": _avg(rel),
                "inner_dev_rel_max": float(max(rel)),
                "inner_dev_rel_last": float(rel[-1]),
                "inner_cos_sim_mean": _avg(cos),
                "inner_cos_sim_min": float(min(cos)),
                "inner_cos_sim_last": float(cos[-1]),
                "inner_angle_deg_mean": _avg(ang),
                "inner_angle_deg_max": float(max(ang)),
                "inner_angle_deg_last": float(ang[-1]),
                "inner_dev_norm_mean": _avg(dnorm),
                "inner_dev_norm_last": float(dnorm[-1]),
            })

        # Stash the full within-epoch curve for the runner to push to W&B
        # as a table + line plots (see ``log_deviation_inner``).  Only when
        # inner logging actually ran; the pure start-of-epoch point is
        # already covered by the v0_* scalars.
        self.last_inner_deviation = deviation_series if inner_rows else None

        return metrics
