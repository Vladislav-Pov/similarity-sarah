"""NFG-SS ablations — NOT in the submission.

Three out-of-scope switches kept off the hot path (``REWRITE_BRIEF.md``):

* ``update_v_tilde_in_the_end`` — replace the carry-over running mean with a
  one-shot estimate at the final iterate.
* ``clip_number_of_clients_with_reshuffle`` — traverse only
  ``clip_clients_per_epoch`` clients from a persistent permutation, rescaling
  the running mean by ``n / clip``.
* ``log_deviation`` — log the squared deviation of the carry-over anchor from
  the exact full-gradient anchor (doubles epoch cost).

Constructed manually (no registry / ``from_spec``); see this package's README.
The default path (all flags off) is bit-identical to :class:`NFGSS`.
"""

from __future__ import annotations

import logging

import torch

from similarity_sarah.algorithms.nfg_ss import NFGSS, _avg
from similarity_sarah.core import foreach
from similarity_sarah.core.grads import compute_batch_gradient, compute_full_gradient
from similarity_sarah.core.params import (
    ParamList,
    add_params_,
    clone_params,
    compute_param_norm,
    diff_param_norm,
    get_params,
    scale_params_,
    set_params,
    zeros_like_params,
)
from similarity_sarah.prox import ProxSolver

logger = logging.getLogger(__name__)


class NfgSSAblations(NFGSS):
    """NFG-SS with the three out-of-scope ablation switches."""

    def __init__(
        self,
        theta: float,
        batch_size_clients: int,
        prox_solver: ProxSolver,
        *,
        update_v_tilde_in_the_end: bool = False,
        clip_number_of_clients_with_reshuffle: bool = False,
        clip_clients_per_epoch: int = 3,
        log_deviation: bool = False,
    ) -> None:
        super().__init__(theta, batch_size_clients, prox_solver)
        self.update_v_tilde_in_the_end = bool(update_v_tilde_in_the_end)
        self.clip_number_of_clients_with_reshuffle = bool(
            clip_number_of_clients_with_reshuffle
        )
        self.clip_clients_per_epoch = int(clip_clients_per_epoch)
        self.log_deviation = bool(log_deviation)
        self._client_permutation: list[int] = []
        self._client_perm_offset: int = 0

    def _compute_end_of_epoch_v_tilde(self) -> ParamList:
        ctx = self._ctx
        assert ctx is not None
        n = ctx.total_nodes
        grad_f1 = compute_batch_gradient(
            ctx.model, ctx.server_grad_loader, ctx.loss_fn, ctx.device
        )
        accum = zeros_like_params(ctx.model)
        for loader in ctx.client_loaders:
            grad_i = compute_batch_gradient(ctx.model, loader, ctx.loss_fn, ctx.device)
            for a, gi, g1 in zip(accum, grad_i, grad_f1):
                a.add_(gi - g1, alpha=1.0 / n)
        return accum

    def _compute_exact_full_grad_diff(self) -> ParamList:
        ctx = self._ctx
        assert ctx is not None
        n = ctx.total_nodes
        grad_f1 = compute_full_gradient(
            ctx.model, ctx.server_grad_loader, ctx.loss_fn, ctx.device
        )
        accum = zeros_like_params(ctx.model)
        for loader in ctx.client_loaders:
            grad_i = compute_full_gradient(ctx.model, loader, ctx.loss_fn, ctx.device)
            for a, gi, g1 in zip(accum, grad_i, grad_f1):
                a.add_(gi - g1, alpha=1.0 / n)
        return accum

    def _next_clipped_client_slice(self) -> list[int]:
        ctx = self._ctx
        assert ctx is not None
        k = self.clip_clients_per_epoch
        if (
            not self._client_permutation
            or self._client_perm_offset + k > len(self._client_permutation)
        ):
            self._client_permutation = torch.randperm(ctx.num_clients).tolist()
            self._client_perm_offset = 0
        sl = self._client_permutation[self._client_perm_offset : self._client_perm_offset + k]
        self._client_perm_offset += k
        return sl

    def run_epoch(self, epoch: int) -> dict[str, float]:
        ctx = self._ctx
        assert ctx is not None, "call bind(ctx) before run_epoch"
        model = ctx.model
        n = ctx.total_nodes
        B = self.batch_size_clients

        if self.clip_number_of_clients_with_reshuffle:
            if B != 1:
                raise ValueError(
                    "clip_number_of_clients_with_reshuffle requires "
                    f"batch_size_clients == 1, got {B}"
                )
            batches = [[cid] for cid in self._next_clipped_client_slice()]
        else:
            from similarity_sarah.runtime.scheduler import sample_client_batches

            batches = sample_client_batches(ctx.num_clients, B)
        K = len(batches)

        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(model)

        if self.log_deviation:
            exact = self._compute_exact_full_grad_diff()
            dev_sq = sum((vi - ei).square().sum().item() for vi, ei in zip(v, exact))
            logger.info("Epoch %d log_deviation: ||v_0 - grad(f-f1)(w_0)||^2=%.4e", epoch, dev_sq)

        w_prev = get_params(model)
        diag0 = self._prox(v)
        prox_first = [diag0["prox_grad_norm_first"]]
        prox_last = [diag0["prox_grad_norm_last"]]
        prox_ratio = [diag0["prox_grad_norm_ratio"]]
        prox_obj = [diag0["prox_obj_decrease"]]
        step_norms: list[float] = []

        for t in range(1, K + 1):
            batch = batches[t - 1]
            b_actual = len(batch)
            w_curr = get_params(model)
            srv_xy = next(iter(ctx.server_grad_loader))
            cli_xys = [next(iter(ctx.client_loaders[cid])) for cid in batch]

            grad_f1_curr = self._server_grad(srv_xy)
            sum_client_curr = self._client_grad_sum(batch, cli_xys)
            set_params(model, w_prev)
            grad_f1_prev = self._server_grad(srv_xy)
            sum_client_prev = self._client_grad_sum(batch, cli_xys)

            sum_diff_curr = [sc - b_actual * g1 for sc, g1 in zip(sum_client_curr, grad_f1_curr)]
            sum_diff_prev = [sp - b_actual * g1 for sp, g1 in zip(sum_client_prev, grad_f1_prev)]

            scale_params_(tilde_v, (t - 1) / t)
            add_params_(tilde_v, sum_diff_curr, alpha=1.0 / (t * B))
            add_params_(v, foreach.sub(sum_diff_curr, sum_diff_prev), alpha=1.0 / (n * B))

            w_prev = w_curr
            set_params(model, w_curr)
            diag = self._prox(v)
            prox_first.append(diag["prox_grad_norm_first"])
            prox_last.append(diag["prox_grad_norm_last"])
            prox_ratio.append(diag["prox_grad_norm_ratio"])
            prox_obj.append(diag["prox_obj_decrease"])
            step_norms.append(diff_param_norm(get_params(model), w_curr))

        if self.update_v_tilde_in_the_end:
            self.v_epoch = self._compute_end_of_epoch_v_tilde()
        elif self.clip_number_of_clients_with_reshuffle:
            scaled = clone_params(tilde_v)
            scale_params_(scaled, n / self.clip_clients_per_epoch)
            self.v_epoch = scaled
        else:
            self.v_epoch = tilde_v

        return {
            "epoch": float(epoch),
            "inner_steps": float(K),
            "v_norm": compute_param_norm(v),
            "tilde_v_norm": compute_param_norm(tilde_v),
            "v_epoch_norm": compute_param_norm(self.v_epoch),
            "param_norm": compute_param_norm(get_params(model)),
            "step_norm_mean": _avg(step_norms),
            "prox_grad_norm_first_mean": _avg(prox_first),
            "prox_grad_norm_last_mean": _avg(prox_last),
            "prox_grad_norm_ratio_mean": _avg(prox_ratio),
            "prox_obj_decrease_mean": _avg(prox_obj),
        }
