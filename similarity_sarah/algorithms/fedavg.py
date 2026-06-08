"""FedAvg baseline (single-gradient "unfair" flavour, McMahan et al. 2017).

Each round samples ``B`` clients, every participant returns ONE stochastic
gradient at ``w_r`` (no local steps), the server averages them (including its
own ``grad f1`` when ``include_server``) and takes a single SGD step
``w_{r+1} = w_r - lr * v_r``. This is synchronous minibatch SGD — a clean
communication-matched lower-bound baseline.
"""

from __future__ import annotations

import logging

from similarity_sarah.algorithms.base import ALGORITHMS, Algorithm, AlgorithmCtx
from similarity_sarah.core.grads import compute_batch_gradient
from similarity_sarah.core.params import (
    ParamList,
    add_params_,
    compute_param_norm,
    get_params,
    set_params,
    zeros_like_params,
)
from similarity_sarah.runtime.scheduler import sample_client_batches
from similarity_sarah.spec import RunSpec

logger = logging.getLogger(__name__)


@ALGORITHMS.register("fedavg")
class FedAvg(Algorithm):
    """Synchronous minibatch SGD across sampled clients (no local steps)."""

    def __init__(
        self, lr: float, batch_size_clients: int, include_server: bool = True
    ) -> None:
        self.lr = float(lr)
        self.batch_size_clients = int(batch_size_clients)
        self.include_server = bool(include_server)
        self._ctx: AlgorithmCtx | None = None

    @classmethod
    def from_spec(cls, spec: RunSpec) -> FedAvg:
        if spec.lr is None:
            raise ValueError("fedavg requires 'lr'")
        return cls(
            lr=spec.lr,
            batch_size_clients=spec.batch_size_clients,
            include_server=spec.include_server,
        )

    def bind(self, ctx: AlgorithmCtx) -> None:
        self._ctx = ctx

    def run_epoch(self, epoch: int) -> dict[str, float]:
        ctx = self._ctx
        assert ctx is not None, "call bind(ctx) before run_epoch"
        model = ctx.model
        batches = sample_client_batches(ctx.num_clients, self.batch_size_clients)
        K = len(batches)

        for batch in batches:
            v: ParamList = zeros_like_params(model)
            n_participants = len(batch) + (1 if self.include_server else 0)

            if self.include_server:
                g_srv = compute_batch_gradient(
                    model, ctx.server_grad_loader, ctx.loss_fn, ctx.device
                )
                add_params_(v, g_srv, alpha=1.0 / n_participants)

            for cid in batch:
                g_cli = compute_batch_gradient(
                    model, ctx.client_loaders[cid], ctx.loss_fn, ctx.device
                )
                add_params_(v, g_cli, alpha=1.0 / n_participants)

            w = get_params(model)
            set_params(model, [wi - self.lr * vi for wi, vi in zip(w, v)])

        return {
            "epoch": float(epoch),
            "inner_steps": float(K),
            "param_norm": compute_param_norm(get_params(model)),
        }
