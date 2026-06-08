"""Inexact proximal solvers driven by the proximal gradient.

Both :class:`InexactProxSGD` and :class:`InexactProxAdam` warm-start at the
outer iterate ``w_outer`` and take ``num_steps`` minibatch steps on

    grad Phi_t(w) = grad f1(w) + (w - w_outer)/theta + alpha_t * v   (+ wd * w),

differing only in how a per-parameter direction becomes a parameter update
(Polyak-momentum SGD vs Adam moments). The shared scaffolding lives in
:class:`_InexactProxBase`; subclasses supply ``_init_state`` and ``_update``.
The ``alpha_t`` schedule (``constant``/``linear``) is honoured here — it is a
no-op only on the AccVRS path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.core.params import ParamList, get_params
from similarity_sarah.prox import _diag
from similarity_sarah.prox.base import PROX_SOLVERS, ProxSolver
from similarity_sarah.spec import ProxSpec


class _InexactProxBase(ProxSolver):
    """Shared inner-loop scaffolding for the SGD/Adam inexact solvers."""

    def __init__(
        self,
        num_steps: int,
        lr: float,
        *,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        eval_batches: int = 1,
        v_schedule: str = "constant",
    ) -> None:
        self.num_steps = int(num_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.eval_batches = max(1, int(eval_batches))
        self.v_schedule = v_schedule

    # -- subclass hooks -------------------------------------------------------
    def _init_state(self, params: list[torch.Tensor]) -> dict:
        return {}

    def _update(
        self,
        params: list[torch.Tensor],
        directions: list[torch.Tensor],
        state: dict,
        step_idx: int,
    ) -> None:
        raise NotImplementedError

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
        # Warm-start from w_outer (model is already there). z is the canonical
        # prox target, used only for diagnostics; the direction uses the
        # algebraic form so z need not be materialised each iteration.
        w_outer = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w_outer, v)]
        params = cast("list[torch.Tensor]", list(model.parameters()))
        state = self._init_state(params)

        eval_xys = _diag.fresh_eval_xys(eval_loader, server_loader, self.eval_batches)
        first_grad_norm, first_obj = _diag.diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device
        )

        clip_count = 0
        server_iter = iter(server_loader)
        for step_idx in range(self.num_steps):
            try:
                x, y = next(server_iter)
            except StopIteration:
                server_iter = iter(server_loader)
                x, y = next(server_iter)

            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            grads = torch.autograd.grad(loss, params)
            alpha = _diag.alpha_at(step_idx, self.num_steps, self.v_schedule)

            with torch.no_grad():
                directions: list[torch.Tensor] = []
                for p, g, w_out_i, v_i in zip(params, grads, w_outer, v):
                    d = g + (p.data - w_out_i) / theta + alpha * v_i
                    if self.weight_decay > 0:
                        d = d + self.weight_decay * p.data
                    directions.append(d)

                if self.grad_clip > 0 and _diag.clip_l2_norm_(directions, self.grad_clip):
                    clip_count += 1

                self._update(params, directions, state, step_idx)

        last_grad_norm, last_obj = _diag.diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device
        )
        return _diag.build_diag_payload(
            first_grad_norm, first_obj, last_grad_norm, last_obj,
            self.num_steps, clip_count=clip_count,
        )


@PROX_SOLVERS.register("sgd")
class InexactProxSGD(_InexactProxBase):
    """Inexact prox via SGD with optional Polyak momentum."""

    def __init__(
        self,
        num_steps: int,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        eval_batches: int = 1,
        v_schedule: str = "constant",
    ) -> None:
        super().__init__(
            num_steps, lr, weight_decay=weight_decay, grad_clip=grad_clip,
            eval_batches=eval_batches, v_schedule=v_schedule,
        )
        self.momentum = float(momentum)

    def _init_state(self, params: list[torch.Tensor]) -> dict:
        velocity = [torch.zeros_like(p) for p in params] if self.momentum > 0 else None
        return {"velocity": velocity}

    def _update(self, params, directions, state, step_idx):
        velocity = state["velocity"]
        for i, (p, d) in enumerate(zip(params, directions)):
            if velocity is not None:
                velocity[i].mul_(self.momentum).add_(d)
                p.data.sub_(velocity[i], alpha=self.lr)
            else:
                p.data.sub_(d, alpha=self.lr)

    @classmethod
    def from_spec(cls, spec: ProxSpec) -> ProxSolver:
        if spec.lr is None:
            raise ValueError("prox_solver=sgd requires an explicit prox_lr")
        return cls(
            num_steps=spec.num_steps, lr=spec.lr, momentum=spec.momentum,
            weight_decay=spec.weight_decay, grad_clip=spec.grad_clip,
            eval_batches=spec.eval_batches, v_schedule=spec.v_schedule,
        )


@PROX_SOLVERS.register("adam")
class InexactProxAdam(_InexactProxBase):
    """Inexact prox via Adam moments on the proximal gradient."""

    def __init__(
        self,
        num_steps: int,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        eval_batches: int = 1,
        v_schedule: str = "constant",
    ) -> None:
        super().__init__(
            num_steps, lr, weight_decay=weight_decay, grad_clip=grad_clip,
            eval_batches=eval_batches, v_schedule=v_schedule,
        )
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)

    def _init_state(self, params: list[torch.Tensor]) -> dict:
        return {
            "m": [torch.zeros_like(p) for p in params],
            "v": [torch.zeros_like(p) for p in params],
        }

    def _update(self, params, directions, state, step_idx):
        b1, b2 = self.betas
        t = step_idx + 1
        bias1 = 1 - b1**t
        bias2 = 1 - b2**t
        m_state, v_state = state["m"], state["v"]
        for i, (p, d) in enumerate(zip(params, directions)):
            m_state[i].mul_(b1).add_(d, alpha=1 - b1)
            v_state[i].mul_(b2).addcmul_(d, d, value=1 - b2)
            m_hat = m_state[i] / bias1
            v_hat = v_state[i] / bias2
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    @classmethod
    def from_spec(cls, spec: ProxSpec) -> ProxSolver:
        if spec.lr is None:
            raise ValueError("prox_solver=adam requires an explicit prox_lr")
        return cls(
            num_steps=spec.num_steps, lr=spec.lr, betas=spec.adam_betas,
            weight_decay=spec.weight_decay, grad_clip=spec.grad_clip,
            eval_batches=spec.eval_batches, v_schedule=spec.v_schedule,
        )
