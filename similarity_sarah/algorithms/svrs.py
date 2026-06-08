"""SVRS baseline (Lin et al. 2023, arXiv:2304.07504, Algorithm 1, SVRS^{1ep}).

Difference from NFG-SS: at the start of each outer round SVRS computes a
*deterministic* full-gradient anchor ``g_ref = (1/n) sum_i grad(f_i - f1)(w_ref)``
(``_refresh_anchor``), then runs ``T ~ Geom(1/num_clients)`` inner steps, each
sampling one client and forming ``v_t = g_ref + grad(f_i - f1)(w_t) -
grad(f_i - f1)(w_ref)`` on a shared minibatch. The prox step on ``f1`` is
identical to NFG-SS, so the same :class:`ProxSolver` is reused. The O(n) anchor
refresh is the cost SVRS pays that NFG-SS avoids.
"""

from __future__ import annotations

import logging

import torch

from similarity_sarah.algorithms.base import ALGORITHMS, Algorithm, AlgorithmCtx
from similarity_sarah.core.grads import compute_batch_gradient, compute_full_gradient
from similarity_sarah.core.params import (
    ParamList,
    clone_params,
    compute_param_norm,
    get_params,
    set_params,
    zeros_like_params,
)
from similarity_sarah.prox import ProxSolver, build_prox_solver
from similarity_sarah.spec import RunSpec

logger = logging.getLogger(__name__)


@ALGORITHMS.register("svrs")
class SVRS(Algorithm):
    """Server-side variance-reduced similarity with a full-gradient anchor."""

    def __init__(
        self, theta: float, batch_size_clients: int, prox_solver: ProxSolver
    ) -> None:
        self.theta = float(theta)
        if batch_size_clients != 1:
            logger.warning(
                "SVRS: batch_size_clients=%d ignored — Algorithm 1 samples one "
                "client per inner step (B=1). Forcing B=1.",
                batch_size_clients,
            )
        self.batch_size_clients = 1
        self.prox_solver = prox_solver
        self._ctx: AlgorithmCtx | None = None
        self._w_ref: ParamList = []
        self._g_ref: ParamList = []

    @classmethod
    def from_spec(cls, spec: RunSpec) -> SVRS:
        if spec.prox is None:
            raise ValueError("svrs requires a prox-solver configuration")
        if spec.theta is None:
            raise ValueError("svrs requires 'theta'")
        return cls(
            theta=spec.theta,
            batch_size_clients=spec.batch_size_clients,
            prox_solver=build_prox_solver(spec.prox),
        )

    def bind(self, ctx: AlgorithmCtx) -> None:
        self._ctx = ctx
        self._w_ref = get_params(ctx.model)
        self._g_ref = zeros_like_params(ctx.model)

    def _refresh_anchor(self) -> None:
        """g_ref = (1/n) sum_i grad(f_i - f1)(w_ref), exact full gradients."""
        ctx = self._ctx
        assert ctx is not None
        n = ctx.total_nodes
        set_params(ctx.model, self._w_ref)
        g_ref = zeros_like_params(ctx.model)
        grad_f1 = compute_full_gradient(
            ctx.model, ctx.server_grad_loader, ctx.loss_fn, ctx.device
        )
        for loader in ctx.client_loaders:
            g = compute_full_gradient(ctx.model, loader, ctx.loss_fn, ctx.device)
            for tg, gi, g1 in zip(g_ref, g, grad_f1):
                tg.add_(gi - g1, alpha=1.0 / n)
        self._g_ref = g_ref

    def _sample_epoch_length(self) -> int:
        """T ~ Geom(1/num_clients), shifted so T >= 1 (E[T] = num_clients)."""
        ctx = self._ctx
        assert ctx is not None
        p = torch.tensor(1.0 / max(ctx.num_clients, 1))
        raw = torch.distributions.Geometric(probs=p).sample().item()
        return int(raw) + 1

    def _sample_one_client(self) -> int:
        ctx = self._ctx
        assert ctx is not None
        return int(torch.randint(low=0, high=ctx.num_clients, size=(1,)).item())

    def run_epoch(self, epoch: int) -> dict[str, float]:
        ctx = self._ctx
        assert ctx is not None, "call bind(ctx) before run_epoch"
        model = ctx.model

        self._w_ref = get_params(model)
        self._refresh_anchor()

        T = self._sample_epoch_length()
        v_norm_last = 0.0
        for _ in range(T):
            cid = self._sample_one_client()
            w_curr = get_params(model)

            # Same minibatch per node, reused at w_t and w_ref (telescope).
            srv_xy = next(iter(ctx.server_grad_loader))
            cli_xy = next(iter(ctx.client_loaders[cid]))

            grad_f1_curr = compute_batch_gradient(
                model, ctx.server_grad_loader, ctx.loss_fn, ctx.device, xy=srv_xy
            )
            g_curr = compute_batch_gradient(
                model, ctx.client_loaders[cid], ctx.loss_fn, ctx.device, xy=cli_xy
            )
            diff_curr = [gc - g1 for gc, g1 in zip(g_curr, grad_f1_curr)]

            set_params(model, self._w_ref)
            grad_f1_ref = compute_batch_gradient(
                model, ctx.server_grad_loader, ctx.loss_fn, ctx.device, xy=srv_xy
            )
            g_ref_b = compute_batch_gradient(
                model, ctx.client_loaders[cid], ctx.loss_fn, ctx.device, xy=cli_xy
            )
            diff_ref = [gr - g1 for gr, g1 in zip(g_ref_b, grad_f1_ref)]

            # v_t = g_ref + (diff_curr - diff_ref); single client, no /B.
            v = clone_params(self._g_ref)
            for vi, dc, dr in zip(v, diff_curr, diff_ref):
                vi.add_(dc - dr)
            v_norm_last = compute_param_norm(v)

            set_params(model, w_curr)
            self.prox_solver.step(
                model, v, self.theta,
                ctx.server_prox_loader, ctx.loss_fn, ctx.device,
                eval_loader=ctx.server_grad_loader,
            )

        return {
            "epoch": float(epoch),
            "inner_steps": float(T),
            "v_norm": v_norm_last,
            "param_norm": compute_param_norm(get_params(model)),
        }
