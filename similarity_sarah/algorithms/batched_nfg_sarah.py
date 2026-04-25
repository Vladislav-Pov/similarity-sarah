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
    """

    def __init__(
        self,
        theta: float,
        batch_size_clients: int,
        prox_solver: ProxSolver,
    ) -> None:
        self.theta = theta
        self.batch_size_clients = batch_size_clients
        self.prox_solver = prox_solver

        self.model: nn.Module | None = None
        self.server_grad_loader: DataLoader | None = None
        self.server_prox_loader: DataLoader | None = None
        self.client_loaders: list[DataLoader] = []
        self.loss_fn: nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.v_epoch: ParamList = []
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
    # Main epoch
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None

        batches = sample_client_batches(self.num_clients, self.batch_size_clients)
        K = len(batches)
        n = self.total_nodes
        B = self.batch_size_clients  # paper's *fixed* B (used in coefficients)

        # tilde_v_1^{(s)} = 0,  v_0^{(s)} = v^{(s)} (carry over)
        v = clone_params(self.v_epoch)
        tilde_v = zeros_like_params(self.model)

        # w_0 for the first inner step (used as w_{t-1} when t=1).
        w_prev = get_params(self.model)

        # w_1^{(s)} = prox_{θ f₁}( w_0^{(s)} − θ · v_0^{(s)} )
        prox_diag_first = self.prox_solver.step(
            self.model, v, self.theta,
            self.server_prox_loader, self.loss_fn, self.device,
            eval_loader=self.server_grad_loader,
        )

        # Aggregated diagnostics across the epoch.
        prox_first_norms: list[float] = [prox_diag_first["prox_grad_norm_first"]]
        prox_last_norms: list[float] = [prox_diag_first["prox_grad_norm_last"]]
        prox_ratios: list[float] = [prox_diag_first["prox_grad_norm_ratio"]]
        prox_obj_decreases: list[float] = [prox_diag_first["prox_obj_decrease"]]
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
            prox_diag = self.prox_solver.step(
                self.model, v, self.theta,
                self.server_prox_loader, self.loss_fn, self.device,
                eval_loader=self.server_grad_loader,
            )

            prox_first_norms.append(prox_diag["prox_grad_norm_first"])
            prox_last_norms.append(prox_diag["prox_grad_norm_last"])
            prox_ratios.append(prox_diag["prox_grad_norm_ratio"])
            prox_obj_decreases.append(prox_diag["prox_obj_decrease"])
            step_norms.append(diff_param_norm(get_params(self.model), w_curr))

            logger.debug(
                "Epoch %d step %d/%d: ‖v‖=%.3e ‖tilde_v‖=%.3e ‖w_{t+1}-w_t‖=%.3e "
                "prox_ratio=%.3f",
                epoch, t, K,
                compute_param_norm(v), compute_param_norm(tilde_v),
                step_norms[-1], prox_ratios[-1],
            )

        # v^{(s+1)} = tilde_v_{K+1}
        self.v_epoch = tilde_v

        def _avg(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        return {
            "epoch": float(epoch),
            "inner_steps": float(K),
            "v_norm": compute_param_norm(v),
            "tilde_v_norm": compute_param_norm(tilde_v),
            "param_norm": compute_param_norm(get_params(self.model)),
            "step_norm_mean": _avg(step_norms),
            "step_norm_last": float(step_norms[-1]) if step_norms else 0.0,
            "prox_grad_norm_first_mean": _avg(prox_first_norms),
            "prox_grad_norm_last_mean": _avg(prox_last_norms),
            "prox_grad_norm_ratio_mean": _avg(prox_ratios),
            "prox_obj_decrease_mean": _avg(prox_obj_decreases),
        }
