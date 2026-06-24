"""FedAvg baseline (single-gradient flavour, "unfair" FedAvg).

Each communication round:
    1. Server samples a subset S_r of B clients.
    2. Server broadcasts w_r to the sampled clients.
    3. Each client computes ONE stochastic gradient on a local minibatch
       (no inner loop / local steps).
    4. Server aggregates: v_r = (1/|batch|) Σ ∇f_i(w_r; minibatch_i)
       If ``include_server`` is True, the server's f_1 is treated as an
       extra participant in the epoch sampling (appears in one batch, not
       every batch).
    5. Server applies a single optimizer step at w_r using v_r as the
       gradient (plain SGD, SGD with Polyak momentum, or Adam).

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
from similarity_sarah.utils import compute_batch_gradient

logger = logging.getLogger(__name__)


class FedAvg(BaseAlgorithm):
    """Synchronous mini-batch SGD across sampled clients (no local steps).

    Args:
        lr: Server step size η (passed to the underlying ``torch.optim``).
        batch_size_clients: Number of clients sampled per round (B).
        include_server: If True, the server participates as an extra node
            in epoch sampling, contributing its gradient only when sampled.
        adam_like: If True, use ``torch.optim.Adam`` instead of SGD.  In
            this case ``momentum`` is ignored and Adam's ``betas=(0.9,
            0.999)`` are used.
        momentum: Polyak (heavy-ball) momentum coefficient for SGD.  Set to
            ``0.0`` (default) for vanilla SGD.  Ignored when
            ``adam_like=True``.
    """

    def __init__(
        self,
        lr: float,
        batch_size_clients: int,
        include_server: bool = True,
        adam_like: bool = False,
        momentum: float = 0.0,
    ) -> None:
        self.lr = lr
        self.batch_size_clients = batch_size_clients
        self.include_server = include_server
        self.adam_like = adam_like
        self.momentum = momentum

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.num_clients: int = 0
        self.total_nodes: int = 0
        self.optimizer: torch.optim.Optimizer | None = None

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
        self.total_nodes = self.num_clients + (1 if self.include_server else 0)

        if self.adam_like:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, betas=(0.9, 0.999),
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=self.momentum,
            )

    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None
        batches = sample_client_batches(self.total_nodes, self.batch_size_clients)
        K = len(batches)

        params = list(self.model.parameters())
        server_idx = self.num_clients if self.include_server else None

        for t, batch in enumerate(batches):
            n_participants = len(batch)
            # Clear gradients; we will create/accumulate ``p.grad`` manually
            # from the averaged client/server gradients.
            self.optimizer.zero_grad(set_to_none=True)

            for pid in batch:
                if server_idx is not None and pid == server_idx:
                    g = compute_batch_gradient(
                        self.model, self.server_grad_loader,
                        self.loss_fn, self.device,
                    )
                else:
                    g = compute_batch_gradient(
                        self.model, self.client_loaders[pid],
                        self.loss_fn, self.device,
                    )
                for p, gi in zip(params, g):
                    if p.grad is None:
                        p.grad = gi.detach().clone()
                        p.grad.mul_(1.0 / n_participants)
                    else:
                        p.grad.add_(gi, alpha=1.0 / n_participants)

            # Server step driven by torch.optim (SGD / SGD+momentum / Adam).
            self.optimizer.step()

            logger.debug(
                "FedAvg epoch %d round %d/%d: averaged %d gradients",
                epoch, t + 1, K, n_participants,
            )

        return {"epoch": float(epoch), "inner_steps": float(K)}
