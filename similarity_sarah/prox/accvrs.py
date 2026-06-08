"""AccVRS-style inexact proximal solver (the best-known inner solver).

Ported from ``AccVRS/resnet_exp/new_alg.py:argminA`` (Batch_SGD branch). Unlike
:class:`InexactProxSGD`, it warm-starts at ``z = w_outer - theta v`` and runs a
direction that *omits* the linear ``theta v`` term (it is encoded only via the
warm-start); the SGD fixed point is therefore ``prox_{theta f1}(w_outer)``,
v-free, which is by design since only a few inner epochs run. Key features:
auto ``gamma_0 = (1/(2L)) * lr_factor`` with ``L = 1 + theta * L1``,
convex-blend momentum, per-element clamp, multiplicative weight decay, periodic
LR decay, and early stop on ``||d||/||d_first|| < early_stop_ratio``.

``num_steps`` counts *full passes* over ``server_loader`` (not iterations).
``prox_v_schedule`` is intentionally NOT consumed here: ``v`` enters solely via
the warm-start, so ``constant`` and ``linear`` are identical on this path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.core.params import ParamList, get_params, set_params
from similarity_sarah.prox import _diag
from similarity_sarah.prox.base import PROX_SOLVERS, ProxSolver
from similarity_sarah.spec import ProxSpec


class _AccvrsBatchBase(ProxSolver):
    """Common scaffolding for AccVRS Batch_SGD / Batch_Adam variants."""

    def __init__(
        self,
        num_steps: int,
        lr: float | None = None,
        L1: float = 200.0,
        lr_factor: float = 1.0,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        grad_clip: float = 0.0,
        inner_decay_factor: float = 0.9,
        inner_decay_period: int | None = None,
        early_stop_ratio: float = 1e-3,
        include_linear_term: bool = False,
        eval_batches: int = 1,
    ) -> None:
        self.num_steps = int(num_steps)
        self.lr = None if lr is None else float(lr)
        self.L1 = float(L1)
        self.lr_factor = float(lr_factor)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.grad_clip = float(grad_clip)
        self.inner_decay_factor = float(inner_decay_factor)
        self.inner_decay_period = (
            int(inner_decay_period) if inner_decay_period is not None else None
        )
        self.early_stop_ratio = float(early_stop_ratio)
        self.include_linear_term = bool(include_linear_term)
        self.eval_batches = max(1, int(eval_batches))

    # -- subclass hooks -------------------------------------------------------
    def _init_state(self, params: list[torch.Tensor]) -> dict:
        return {}

    def _apply_update(
        self,
        params: list[torch.Tensor],
        direction: list[torch.Tensor],
        lr: float,
        state: dict,
        step_idx: int,
    ) -> None:
        raise NotImplementedError

    # -- helpers --------------------------------------------------------------
    def _resolve_lr(self, theta: float) -> float:
        if self.lr is not None:
            return self.lr * self.lr_factor
        L = 1.0 + theta * self.L1
        return (1.0 / (2.0 * L)) * self.lr_factor

    # -- ProxSolver interface -------------------------------------------------
    def step(
        self,
        model: nn.Module,
        v: ParamList,
        theta: float,
        server_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        eval_loader: DataLoader | None = None,
    ) -> Mapping[str, float]:
        w_outer = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w_outer, v)]
        set_params(model, z)
        params = cast("list[torch.Tensor]", list(model.parameters()))

        eval_xys = _diag.fresh_eval_xys(eval_loader, server_loader, self.eval_batches)
        first_grad_norm, first_obj = _diag.diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device
        )

        try:
            num_batches_per_epoch = len(server_loader)
        except TypeError:
            num_batches_per_epoch = 1
        decay_period = (
            self.inner_decay_period
            if self.inner_decay_period is not None
            else max(1, num_batches_per_epoch // 5)
        )

        lr = self._resolve_lr(theta)
        state = self._init_state(params)

        prev_direction: list[torch.Tensor] | None = None
        start_norm_inner = 0.0
        cur_norm_inner = 0.0
        last_norm_inner = 0.0
        clip_count = 0
        total_steps = 0
        broken = False

        for _ in range(self.num_steps):
            if broken:
                break
            server_iter = iter(server_loader)
            for _ in range(num_batches_per_epoch):
                try:
                    x, y = next(server_iter)
                except StopIteration:
                    break
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = loss_fn(output, y)
                grads = torch.autograd.grad(loss, params)

                with torch.no_grad():
                    # d = (param - w_outer) + theta * grad f1; the linear
                    # <theta v, .> term is encoded via the warm-start z, NOT here
                    # (unless include_linear_term).
                    d = [
                        (p.data - wo) + theta * g
                        for p, wo, g in zip(params, w_outer, grads)
                    ]
                    if self.include_linear_term:
                        for di, vi in zip(d, v):
                            di.add_(vi, alpha=theta)

                    if self.momentum > 0 and prev_direction is not None:
                        d = [
                            self.momentum * pd + (1.0 - self.momentum) * di
                            for pd, di in zip(prev_direction, d)
                        ]
                    prev_direction = [di.clone() for di in d]

                    if self.grad_clip > 0 and _diag.clip_per_element_(d, self.grad_clip):
                        clip_count += 1

                    if total_steps == 0:
                        start_norm_inner = _diag.direction_norm(d)
                        cur_norm_inner = start_norm_inner

                    self._apply_update(params, d, lr, state, total_steps + 1)

                total_steps += 1

                if total_steps % decay_period == 0:
                    cur_norm_inner = _diag.direction_norm(d)
                    last_norm_inner = cur_norm_inner
                    lr *= self.inner_decay_factor
                    ratio = cur_norm_inner / max(start_norm_inner, 1e-12)
                    if ratio < self.early_stop_ratio:
                        broken = True
                        break

        if last_norm_inner == 0.0 and prev_direction is not None:
            last_norm_inner = _diag.direction_norm(prev_direction)
        last_frac_inner = last_norm_inner / max(start_norm_inner, 1e-12)

        last_grad_norm, last_obj = _diag.diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device
        )

        payload = _diag.build_diag_payload(
            first_grad_norm, first_obj, last_grad_norm, last_obj,
            self.num_steps, clip_count=clip_count,
        )
        payload["prox_inner_steps"] = float(total_steps)
        payload["prox_inner_norm_first"] = float(start_norm_inner)
        payload["prox_inner_norm_last"] = float(last_norm_inner)
        payload["prox_inner_norm_ratio"] = float(last_frac_inner)
        return payload


@PROX_SOLVERS.register("accvrs_batch_sgd", "accvrs_sgd")
class AccvrsBatchSGD(_AccvrsBatchBase):
    """AccVRS Batch_SGD: ``param <- param * (1 - wd*lr) - lr * d``."""

    def _apply_update(self, params, direction, lr, state, step_idx):
        decay_mult = 1.0 - self.weight_decay * lr
        for p, d in zip(params, direction):
            if self.weight_decay > 0:
                p.data.mul_(decay_mult)
            p.data.add_(d, alpha=-lr)

    @classmethod
    def from_spec(cls, spec: ProxSpec) -> ProxSolver:
        return cls(
            num_steps=spec.num_steps,
            lr=spec.lr,
            L1=spec.L1,
            lr_factor=spec.lr_factor,
            weight_decay=spec.weight_decay,
            momentum=spec.momentum,
            grad_clip=spec.grad_clip,
            inner_decay_factor=spec.inner_decay_factor,
            inner_decay_period=spec.inner_decay_period,
            early_stop_ratio=spec.early_stop_ratio,
            include_linear_term=spec.include_linear_term,
            eval_batches=spec.eval_batches,
        )
