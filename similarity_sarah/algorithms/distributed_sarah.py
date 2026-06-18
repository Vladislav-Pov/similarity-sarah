"""Distributed NoFullGrad SARAH baseline (Medyakov-style transplant).

Plain SARAH carried over to the distributed setting: a variance-reduced
estimator that **never computes a full gradient**. Each epoch's anchor is
seeded from the previous epoch's within-epoch running mean -- the same
no-full-gradient carry-over as :class:`~similarity_sarah.algorithms.nfg_ss.NFGSS`
-- but this baseline drops the ``f1`` prox entirely: it estimates the *full*
gradient ``grad f`` (not ``grad(f - f1)``) and takes a plain gradient step.

It is the communication-matched counterpart of NFG-SS. Per inner step every
sampled client returns gradients at ``w_t`` and ``w_{t-1}`` (the same
two-gradient SARAH footprint), and the server contributes its own ``grad f1``
locally -- free under the communication metric, exactly as in NFG-SS. The gap
between the two therefore isolates what the ``f1``-proximal step under
second-order similarity buys, holding communication fixed.

Per outer epoch ``s`` (clients partitioned into disjoint batches ``B_1..B_K``):

    g_t      = 1/n * grad(f1)(w_t)
               + (n-1)/(n*b) * sum_{i in B_t} grad(f_i)(w_t)   # ~= grad(f)(w_t)
    v_0      = v^{(s)}                                          # carry-over anchor
    w_1      = w_0 - lr * (v_0 + wd * w_0)
    for t = 1..K:
        tilde_v_{t+1} = (t-1)/t * tilde_v_t + 1/t * g_t        # running mean
        v_t   = v_{t-1} + (g_t - g_{t-1})                      # SARAH telescope
        w_{t+1} = w_t - lr * (v_t + wd * w_t)                  # plain step + wd
    v^{(s+1)} = tilde_v_{K+1}

The server's ``f1`` is queried every inner step (weight ``1/n``) so the
estimator targets the *same* objective ``f = 1/n sum_i f_i`` that NFG-SS
minimises; with ``include_server=False`` the estimate covers the clients only.
The *same* minibatch is reused at ``w_t`` and ``w_{t-1}`` for every node
(``docs/instructions.md`` section 11) so the SARAH difference telescopes
instead of degenerating into noisy SGD.
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
from similarity_sarah.runtime.scheduler import sample_client_batches
from similarity_sarah.spec import RunSpec

logger = logging.getLogger(__name__)


def _avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


@ALGORITHMS.register("distributed_sarah")
class DistributedSARAH(Algorithm):
    """No-full-gradient SARAH with a plain step (no ``f1`` prox)."""

    def __init__(
        self,
        lr: float,
        batch_size_clients: int,
        weight_decay: float = 0.0,
        include_server: bool = True,
    ) -> None:
        self.lr = float(lr)
        self.batch_size_clients = int(batch_size_clients)
        self.weight_decay = float(weight_decay)
        self.include_server = bool(include_server)
        self._ctx: AlgorithmCtx | None = None
        self.v_epoch: ParamList = []

    @classmethod
    def from_spec(cls, spec: RunSpec) -> DistributedSARAH:
        if spec.lr is None:
            raise ValueError("distributed_sarah requires 'lr'")
        return cls(
            lr=spec.lr,
            batch_size_clients=spec.batch_size_clients,
            weight_decay=spec.weight_decay,
            include_server=spec.include_server,
        )

    def bind(self, ctx: AlgorithmCtx) -> None:
        self._ctx = ctx
        # v_0^{(0)} = 0 (no full gradient at the first epoch either).
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

    def _global_grad(
        self, grad_f1: ParamList, sum_client: ParamList, b_actual: int, n: int
    ) -> ParamList:
        """Unbiased estimate of grad f(w): server weight 1/n, clients (n-1)/(n*b)."""
        ctx = self._ctx
        assert ctx is not None
        g = zeros_like_params(ctx.model)
        if self.include_server:
            add_params_(g, grad_f1, alpha=1.0 / n)
            add_params_(g, sum_client, alpha=(n - 1) / (n * b_actual))
        else:
            add_params_(g, sum_client, alpha=1.0 / b_actual)
        return g

    def _step(self, w: ParamList, v: ParamList) -> None:
        """w <- w - lr * (v + wd * w)  (plain SARAH step with weight decay)."""
        ctx = self._ctx
        assert ctx is not None
        lr, wd = self.lr, self.weight_decay
        set_params(ctx.model, [wi - lr * (vi + wd * wi) for wi, vi in zip(w, v)])

    # -- one outer epoch ------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        ctx = self._ctx
        assert ctx is not None, "call bind(ctx) before run_epoch"
        model = ctx.model
        n = ctx.total_nodes
        B = self.batch_size_clients

        batches = sample_client_batches(ctx.num_clients, B)
        K = len(batches)

        # v_0 = v^{(s)} carry-over; tilde_v_1 = 0.
        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(model)

        # w_1 = w_0 - lr * (v_0 + wd * w_0).
        w_prev = get_params(model)
        self._step(w_prev, v)

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

            # Full-gradient estimates grad(f)(w_t), grad(f)(w_{t-1}).
            g_curr = self._global_grad(grad_f1_curr, sum_client_curr, b_actual, n)
            g_prev = self._global_grad(grad_f1_prev, sum_client_prev, b_actual, n)

            # tilde_v_{t+1} = (t-1)/t * tilde_v_t + 1/t * g_curr.
            scale_params_(tilde_v, (t - 1) / t)
            add_params_(tilde_v, g_curr, alpha=1.0 / t)

            # v_t = v_{t-1} + (g_curr - g_prev)  (SARAH telescope on grad f).
            add_params_(v, foreach.sub(g_curr, g_prev), alpha=1.0)

            # w_{t+1} = w_t - lr * (v_t + wd * w_t).
            w_prev = w_curr
            self._step(w_curr, v)
            step_norms.append(diff_param_norm(get_params(model), w_curr))
            logger.info(
                "epoch %d step %d/%d: ‖v‖=%.1e  ‖step‖=%.1e",
                epoch, t, K, compute_param_norm(v), step_norms[-1],
            )

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
        }
