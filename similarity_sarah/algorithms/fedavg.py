"""FedAvg baseline (single-gradient flavour, "unfair" FedAvg).

Each communication round:
    1. Server samples a subset S_r of B clients.
    2. Server broadcasts w_r to the sampled clients.
    3. Each client computes ONE stochastic gradient on a local minibatch
       (no inner loop / local steps).
    4. Server aggregates: v_r = (1/|nodes|) Σ ∇f_i(w_r; minibatch_i)
       The server's own f_1 is included by default (matches our other
       baselines where the server is always queried).
    5. Server updates: w_{r+1} = w_r − lr · v_r.

This is equivalent to synchronous mini-batch SGD on the aggregated
gradient — no communication advantage over plain distributed SGD, hence
"unfair FedAvg".  To get a *real* FedAvg with multiple local steps per
round, see the standard formulation (this implementation deliberately
omits it to serve as a clean lower-bound baseline).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.runtime.scheduler import sample_client_batches
from similarity_sarah.utils import (
    ParamList,
    add_params_,
    compute_batch_gradient,
    get_params,
    set_params,
    zeros_like_params,
)

logger = logging.getLogger(__name__)


class FedAvg(BaseAlgorithm):
    """Synchronous mini-batch SGD across sampled clients (no local steps).

    Args:
        lr: Server step size η.
        batch_size_clients: Number of clients sampled per round (B).
        include_server: If True, the server also contributes ∇f_1(w_r) to
            the average (matches our SVRS / distributed_sarah convention,
            where the server is a special always-on participant).
    """

    def __init__(
        self,
        lr: float,
        batch_size_clients: int,
        include_server: bool = True,
    ) -> None:
        self.lr = lr
        self.batch_size_clients = batch_size_clients
        self.include_server = include_server

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.num_clients: int = 0
        self.total_nodes: int = 0

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
        # Kept for parity; FedAvg uses only the deterministic grad loader.
        self.server_prox_loader = server_prox_loader
        self.client_loaders = client_loaders
        self.loss_fn = loss_fn
        self.device = device
        self.num_clients = len(client_loaders)
        self.total_nodes = self.num_clients + 1

    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None
        batches = sample_client_batches(self.num_clients, self.batch_size_clients)
        K = len(batches)

        for t, batch in enumerate(batches):
            # Aggregate one stochastic gradient per participant at w_r.
            v: ParamList = zeros_like_params(self.model)
            n_participants = len(batch) + (1 if self.include_server else 0)

            if self.include_server:
                g_srv = compute_batch_gradient(
                    self.model, self.server_grad_loader,
                    self.loss_fn, self.device,
                )
                add_params_(v, g_srv, alpha=1.0 / n_participants)

            for cid in batch:
                g_cli = compute_batch_gradient(
                    self.model, self.client_loaders[cid],
                    self.loss_fn, self.device,
                )
                add_params_(v, g_cli, alpha=1.0 / n_participants)

            # Server step: w_{r+1} = w_r − lr · v_r
            w = get_params(self.model)
            w_new = [wi - self.lr * vi for wi, vi in zip(w, v)]
            set_params(self.model, w_new)

            logger.debug(
                "FedAvg epoch %d round %d/%d: averaged %d gradients",
                epoch, t + 1, K, n_participants,
            )

        return {"epoch": float(epoch), "inner_steps": float(K)}
