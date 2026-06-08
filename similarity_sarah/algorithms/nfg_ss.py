"""NFG-SS: NoFullGrad SARAH Similarity (Algorithm 1).

The server holds the global model and performs every update; clients only
return per-batch gradients. No full gradient is ever computed: each epoch's
estimator is seeded from the previous epoch's within-epoch running mean.

Per outer epoch ``s`` (clients partitioned into disjoint batches ``B_1..B_K``):

    v_0      = v^{(s)}                                   # carry-over anchor
    w_1      = prox_{theta f1}(w_0 - theta v_0)
    for t = 1..K:
        tilde_v_{t+1} = (t-1)/t * tilde_v_t
                        + 1/(t*B) * sum_{i in B_t} grad(f_i - f1)(w_t)
        v_t   = v_{t-1}
                + 1/(n*B) * sum_{i in B_t} [ grad(f_i - f1)(w_t)
                                             - grad(f_i - f1)(w_{t-1}) ]
        w_{t+1} = prox_{theta f1}(w_t - theta v_t)
    v^{(s+1)} = tilde_v_{K+1}

Two load-bearing details, both deliberately preserved bit-for-bit:

* The SARAH increment uses ``1/(n*B)`` (``coeff_v`` below). Algorithm 1 line 9
  in the paper writes ``1/b``; this codebase intentionally keeps ``1/(n*B)`` so
  the published reference runs reproduce exactly. DO NOT change it.
* Within one inner step the *same* minibatch is reused at ``w_t`` and
  ``w_{t-1}`` for every node, so the SARAH difference telescopes instead of
  degenerating into noisy SGD (see ``docs/instructions.md`` section 11).
"""

from __future__ import annotations

import torch.nn as nn

from similarity_sarah.algorithms.base import ALGORITHMS, Algorithm, AlgorithmCtx
from similarity_sarah.core import foreach
from similarity_sarah.core.grads import compute_batch_gradient
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
from similarity_sarah.prox import ProxSolver, build_prox_solver
from similarity_sarah.runtime.scheduler import sample_client_batches
from similarity_sarah.spec import RunSpec


def _avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


@ALGORITHMS.register("nfg_ss", "batched_nfg_sarah")
class NFGSS(Algorithm):
    """No-full-gradient SARAH under second-order similarity (Algorithm 1)."""

    def __init__(
        self, theta: float, batch_size_clients: int, prox_solver: ProxSolver
    ) -> None:
        self.theta = float(theta)
        self.batch_size_clients = int(batch_size_clients)
        self.prox_solver = prox_solver
        self._ctx: AlgorithmCtx | None = None
        self.v_epoch: ParamList = []

    @classmethod
    def from_spec(cls, spec: RunSpec) -> NFGSS:
        if spec.prox is None:
            raise ValueError("nfg_ss requires a prox-solver configuration")
        if spec.theta is None:
            raise ValueError("nfg_ss requires 'theta'")
        return cls(
            theta=spec.theta,
            batch_size_clients=spec.batch_size_clients,
            prox_solver=build_prox_solver(spec.prox),
        )

    def bind(self, ctx: AlgorithmCtx) -> None:
        self._ctx = ctx
        # tilde_v_1^{(0)} = 0, v_0^{(0)} = 0 (Algorithm 1 input).
        self.v_epoch = zeros_like_params(ctx.model)

    # -- helpers --------------------------------------------------------------
    def _client_grad_sum(self, client_ids: list[int], xys: list) -> ParamList:
        """sum_{i in batch} grad f_i(w) on the given per-client minibatches."""
        ctx = self._ctx
        assert ctx is not None
        acc = zeros_like_params(ctx.model)
        for cid, xy in zip(client_ids, xys):
            g = compute_batch_gradient(
                ctx.model, ctx.client_loaders[cid], ctx.loss_fn, ctx.device, xy=xy
            )
            add_params_(acc, g)
        return acc

    def _server_grad(self, xy) -> ParamList:
        ctx = self._ctx
        assert ctx is not None
        return compute_batch_gradient(
            ctx.model, ctx.server_grad_loader, ctx.loss_fn, ctx.device, xy=xy
        )

    def _prox(self, v: ParamList) -> dict:
        ctx = self._ctx
        assert ctx is not None
        return dict(
            self.prox_solver.step(
                ctx.model, v, self.theta,
                ctx.server_prox_loader, ctx.loss_fn, ctx.device,
                eval_loader=ctx.server_grad_loader,
            )
        )

    # -- one outer epoch ------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        ctx = self._ctx
        assert ctx is not None, "call bind(ctx) before run_epoch"
        model: nn.Module = ctx.model
        n = ctx.total_nodes
        B = self.batch_size_clients

        batches = sample_client_batches(ctx.num_clients, B)
        K = len(batches)

        # v_0 = v^{(s)} carry-over; tilde_v_1 = 0.
        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(model)

        # w_1 = prox_{theta f1}(w_0 - theta v_0).
        w_prev = get_params(model)
        diag0 = self._prox(v)

        prox_first = [diag0["prox_grad_norm_first"]]
        prox_last = [diag0["prox_grad_norm_last"]]
        prox_ratio = [diag0["prox_grad_norm_ratio"]]
        prox_obj = [diag0["prox_obj_decrease"]]
        prox_clip = [diag0.get("prox_clip_frac", 0.0)]
        step_norms: list[float] = []

        for t in range(1, K + 1):
            batch = batches[t - 1]
            b_actual = len(batch)
            w_curr = get_params(model)

            # Same minibatch reused at w_t and w_{t-1} (section 11 invariant).
            srv_xy = next(iter(ctx.server_grad_loader))
            cli_xys = [next(iter(ctx.client_loaders[cid])) for cid in batch]

            grad_f1_curr = self._server_grad(srv_xy)
            sum_client_curr = self._client_grad_sum(batch, cli_xys)
            set_params(model, w_prev)
            grad_f1_prev = self._server_grad(srv_xy)
            sum_client_prev = self._client_grad_sum(batch, cli_xys)

            # sum_{i in B_t} grad(f_i - f1)(.)  at w_t and w_{t-1}.
            sum_diff_curr = [
                sc - b_actual * g1 for sc, g1 in zip(sum_client_curr, grad_f1_curr)
            ]
            sum_diff_prev = [
                sp - b_actual * g1 for sp, g1 in zip(sum_client_prev, grad_f1_prev)
            ]

            # tilde_v_{t+1} = (t-1)/t * tilde_v_t + 1/(t*B) * sum_diff_curr.
            scale_params_(tilde_v, (t - 1) / t)
            add_params_(tilde_v, sum_diff_curr, alpha=1.0 / (t * B))

            # v_t = v_{t-1} + coeff_v * (sum_diff_curr - sum_diff_prev).
            # coeff_v = 1/(n*B) is a deliberate, bit-reproduced choice; the
            # paper's Algorithm 1 line 9 uses 1/b. DO NOT change this.
            coeff_v = 1.0 / (n * B)
            add_params_(v, foreach.sub(sum_diff_curr, sum_diff_prev), alpha=coeff_v)

            # w_{t+1} = prox_{theta f1}(w_t - theta v_t).
            w_prev = w_curr
            set_params(model, w_curr)
            diag = self._prox(v)

            prox_first.append(diag["prox_grad_norm_first"])
            prox_last.append(diag["prox_grad_norm_last"])
            prox_ratio.append(diag["prox_grad_norm_ratio"])
            prox_obj.append(diag["prox_obj_decrease"])
            prox_clip.append(diag.get("prox_clip_frac", 0.0))
            step_norms.append(diff_param_norm(get_params(model), w_curr))

        # Hand off to next epoch — no full gradient.
        self.v_epoch = tilde_v

        return {
            "epoch": float(epoch),
            "inner_steps": float(K),
            "v_norm": compute_param_norm(v),
            "tilde_v_norm": compute_param_norm(tilde_v),
            "v_epoch_norm": compute_param_norm(self.v_epoch),
            "param_norm": compute_param_norm(get_params(model)),
            "step_norm_mean": _avg(step_norms),
            "step_norm_last": float(step_norms[-1]) if step_norms else 0.0,
            "prox_grad_norm_first_mean": _avg(prox_first),
            "prox_grad_norm_last_mean": _avg(prox_last),
            "prox_grad_norm_ratio_mean": _avg(prox_ratio),
            "prox_obj_decrease_mean": _avg(prox_obj),
            "prox_clip_frac_mean": _avg(prox_clip),
        }
