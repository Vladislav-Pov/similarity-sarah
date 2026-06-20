"""Distributed NoFullGrad SARAH with a local-step prox on ``f1`` (literal spec).

Same no-full-gradient SARAH estimator as
:class:`~similarity_sarah.algorithms.distributed_sarah.DistributedSARAH`
(running-mean carry-over over ``grad f_i``, never a full gradient), but the
plain gradient step is replaced by an **inexact proximal step on the server's
``f1``** solved with local SGD steps (the prox solver, exactly as in NFG-SS):

    w_{t+1} = argmin_w { (1/2θ) ‖w − w_t‖² + f1(w) } = prox_{θ f1}(w_t)

Per the specified pseudocode the proximal subproblem drops the linear term, so
the SARAH direction ``v_t`` is NOT injected into the prox (it is passed as 0).

NOTE — degeneracy of the literal spec. Because the ⟨v_t, ·⟩ term is removed,
``v_t`` (and the running-mean carry-over ``tilde_v``) do **not** influence the
iterate: the ``w``-trajectory is driven purely by the ``f1``-prox, so the
method converges towards ``argmin f1`` (the client gradients never enter ``w``).
``v_t`` / ``tilde_v`` are still maintained exactly as written so the carry-over
matches the pseudocode and re-enabling the linear term is a one-line change
(pass ``v`` instead of zeros to ``_prox``). This is implemented faithfully to
the spec as a baseline/ablation.

Per outer epoch ``s`` (clients partitioned into disjoint batches ``B_1..B_K``):

    g_t      = 1/n * grad(f1)(w_t)
               + (n-1)/(n*b) * sum_{i in B_t} grad(f_i)(w_t)   # ~= grad(f)(w_t)
    v_0      = v^{(s)}                                          # carry-over anchor
    w_1      = prox_{θ f1}(w_0)
    for t = 1..K:
        tilde_v_{t+1} = (t-1)/t * tilde_v_t + 1/t * g_t        # running mean
        v_t   = v_{t-1} + (g_t - g_{t-1})                      # SARAH telescope
        w_{t+1} = prox_{θ f1}(w_t)                             # local-step prox
    v^{(s+1)} = tilde_v_{K+1}

The *same* minibatch is reused at ``w_t`` and ``w_{t-1}`` for every node
(``docs/instructions.md`` section 11).
"""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)


def _avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


@ALGORITHMS.register("distributed_sarah_local_steps")
class DistributedSARAHLocalSteps(Algorithm):
    """NoFullGrad SARAH estimator + local-step ``f1`` prox (no linear term)."""

    def __init__(
        self,
        theta: float,
        batch_size_clients: int,
        prox_solver: ProxSolver,
        include_server: bool = True,
    ) -> None:
        self.theta = float(theta)
        self.batch_size_clients = int(batch_size_clients)
        self.prox_solver = prox_solver
        self.include_server = bool(include_server)
        self._ctx: AlgorithmCtx | None = None
        self.v_epoch: ParamList = []

    @classmethod
    def from_spec(cls, spec: RunSpec) -> DistributedSARAHLocalSteps:
        if spec.prox is None:
            raise ValueError("distributed_sarah_local_steps requires a prox-solver config")
        if spec.theta is None:
            raise ValueError("distributed_sarah_local_steps requires 'theta'")
        return cls(
            theta=spec.theta,
            batch_size_clients=spec.batch_size_clients,
            prox_solver=build_prox_solver(spec.prox),
            include_server=spec.include_server,
        )

    def bind(self, ctx: AlgorithmCtx) -> None:
        self._ctx = ctx
        self.v_epoch = zeros_like_params(ctx.model)

    # -- helpers --------------------------------------------------------------
    def _client_grad_sum(self, client_ids: list[int], xys: list) -> ParamList:
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

    def _global_grad(
        self, grad_f1: ParamList, sum_client: ParamList, b_actual: int, n: int
    ) -> ParamList:
        ctx = self._ctx
        assert ctx is not None
        g = zeros_like_params(ctx.model)
        if self.include_server:
            add_params_(g, grad_f1, alpha=1.0 / n)
            add_params_(g, sum_client, alpha=(n - 1) / (n * b_actual))
        else:
            add_params_(g, sum_client, alpha=1.0 / b_actual)
        return g

    def _prox(self, v: ParamList) -> dict:
        """prox_{theta f1}(w - theta*v) on the current model; v=0 ⇒ prox(w)."""
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
        model = ctx.model
        n = ctx.total_nodes
        B = self.batch_size_clients

        batches = sample_client_batches(ctx.num_clients, B)
        K = len(batches)

        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(model)
        # The prox drops the linear term ⟨v, ·⟩ (literal spec): solve prox(w_t).
        zero_v = zeros_like_params(model)

        # w_1 = prox_{theta f1}(w_0).
        w_prev = get_params(model)
        diag0 = self._prox(zero_v)
        logger.info(
            "epoch %d step 0/%d (init): prox_grad %.2f→%.2f",
            epoch, K, diag0["prox_grad_norm_first"], diag0["prox_grad_norm_last"],
        )

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

            g_curr = self._global_grad(grad_f1_curr, sum_client_curr, b_actual, n)
            g_prev = self._global_grad(grad_f1_prev, sum_client_prev, b_actual, n)

            # tilde_v_{t+1} = (t-1)/t * tilde_v_t + 1/t * g_curr  (carry-over).
            scale_params_(tilde_v, (t - 1) / t)
            add_params_(tilde_v, g_curr, alpha=1.0 / t)

            # v_t = v_{t-1} + (g_curr - g_prev)  (SARAH telescope; inert here).
            add_params_(v, foreach.sub(g_curr, g_prev), alpha=1.0)

            # w_{t+1} = prox_{theta f1}(w_t)  (no linear term, per spec).
            w_prev = w_curr
            set_params(model, w_curr)
            diag = self._prox(zero_v)

            prox_first.append(diag["prox_grad_norm_first"])
            prox_last.append(diag["prox_grad_norm_last"])
            prox_ratio.append(diag["prox_grad_norm_ratio"])
            prox_obj.append(diag["prox_obj_decrease"])
            prox_clip.append(diag.get("prox_clip_frac", 0.0))
            step_norms.append(diff_param_norm(get_params(model), w_curr))
            logger.info(
                "epoch %d step %d/%d: prox_grad %.2f→%.2f  ‖step‖=%.1e",
                epoch, t, K, diag["prox_grad_norm_first"], diag["prox_grad_norm_last"],
                step_norms[-1],
            )

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
