"""AccVRS-style inexact proximal solvers.

Ported from ``AccVRS/resnet_exp/new_alg.py:argminA`` (Batch_SGD / Batch_Adam
branches), which empirically converges faster on ResNet-18 / CIFAR than our
:class:`InexactProxSGD` / :class:`InexactProxAdam`.  Solves the same
proximal subproblem the rest of the codebase expects:

    w_new ≈ prox_{θ f₁}(w_outer − θ v)
         = argmin_w  { ⟨θ v, w⟩ + (1/2) ‖w − w_outer‖² + θ f₁(w) }
         = argmin_w  { (1/(2θ)) ‖w − z‖² + f₁(w) },   z := w_outer − θ v.

Algorithm (mini-batch SGD / Adam) — mirrors their Batch_SGD branch literally:

    1. Warm-start at  z = w_outer − θ v   (encodes the linear ⟨θv, w⟩ term).
    2. Loop ``num_steps`` epochs over ``server_loader``.  Per minibatch:

         d = (param − w_outer) + θ · ∇f₁(param; batch)

       NOTE: this is the gradient of  Ψ(w) = (1/2)‖w−w_outer‖² + θ f₁(w),
       i.e. the linear ⟨θv, ·⟩ term is *intentionally absent* — its effect
       is encoded only via the warm-start.  At the warm-start the direction
       evaluates to  −θv + θ ∇f₁(z),  but the SGD fixed point is
       prox_{θ f₁}(w_outer)  (independent of v).  In practice their
       experiments use FEW inner epochs (``num_steps=4``), so the iterate
       stays in the v-shifted neighbourhood of z; this is by design.

       Pass ``include_linear_term=True`` to add  +θv  back to ``d`` — the
       SGD then converges to the *true* argmin  prox_{θf₁}(w_outer − θv)
       (matching :class:`InexactProxSGD`).  Use this if you plan to run
       many inner steps and want exact prox semantics.

    3. Optional Polyak momentum on  d  (their default 0.9 in argminA-Batch_SGD).
    4. Optional per-element clipping (``torch.clamp``).
    5. Optional weight-decay shrink:  param ← param·(1 − wd·lr) − lr·d.
    6. Apply solver-specific update (SGD: subtract; Adam: m/v moments).
    7. ``lr ×= inner_decay_factor`` every ``inner_decay_period`` steps.
    8. Early-stop when  ‖d_now‖ / ‖d_first‖ < ``early_stop_ratio``.

Diagnostics (consistent with our other solvers, plus AccVRS-specific):

    prox_grad_norm_first / _last / _ratio       on the fixed eval batch
    prox_obj_decrease                            Φ before − Φ after on eval
    prox_inner_steps                              actual inner-step count
    prox_clip_frac                                fraction of clipped steps
    prox_inner_norm_first / _last / _ratio        their original metric
                                                  (norms taken on the running
                                                  training batch — noisier
                                                  than the eval-batch one)
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from similarity_sarah.utils import ParamList, get_params, set_params
from similarity_sarah.runtime.prox_solver import (
    ProxSolver,
    _build_diag_payload,
    _fresh_eval_xys,
    _prox_diag_on_batches,
)


def _clip_per_element_(
    directions: list[torch.Tensor], clip_value: float,
) -> bool:
    """In-place per-element clamp of each tensor in ``directions``.

    Mirrors AccVRS's ``g = [torch.clip(gi, -c, c) for gi in g]``.  Returns
    ``True`` if any element actually exceeded ±clip_value (cheap probe
    via ``.abs().max()``), ``False`` otherwise.
    """
    if clip_value <= 0:
        return False
    fired = False
    for d in directions:
        if not fired:
            if d.abs().max().item() > clip_value:
                fired = True
        d.clamp_(-clip_value, clip_value)
    return fired


def _direction_norm(directions: list[torch.Tensor]) -> float:
    """Global L2 norm across a list of tensors (no allocation)."""
    s = 0.0
    for d in directions:
        s += d.square().sum().item()
    return float(s ** 0.5)


class _AccvrsBatchBase(ProxSolver):
    """Common scaffolding for AccVRS Batch_SGD / Batch_Adam variants.

    Subclasses override :meth:`_init_state` / :meth:`_apply_update`.
    """

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
    ):
        # ``num_steps`` here means *number of full passes* over the server
        # loader — matching their ``argmin_max_iter``.  Total per-batch
        # inner iterations = num_steps × len(server_loader).
        self.num_steps = int(num_steps)
        # ``lr`` may be None ⇒ auto-derive  γ₀ = (1/(2L)) · lr_factor,
        # where  L = 1 + θ · L1.
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

    # ── subclass hooks ─────────────────────────────────────────────────
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

    # ── helpers ────────────────────────────────────────────────────────
    def _resolve_lr(self, theta: float) -> float:
        if self.lr is not None:
            return self.lr * self.lr_factor
        L = 1.0 + theta * self.L1
        return (1.0 / (2.0 * L)) * self.lr_factor

    # ── ProxSolver interface ───────────────────────────────────────────
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
        # Snapshot the outer iterate w_outer (current model state) and the
        # warm-start point  z = w_outer − θ v  (encodes the linear term).
        w_outer = get_params(model)
        z = [wi - theta * vi for wi, vi in zip(w_outer, v)]
        set_params(model, z)
        params = list(model.parameters())

        # Pre-sample fixed eval batches for the standard diagnostic
        # (matches our InexactProxSGD/Adam logging).
        eval_xys = _fresh_eval_xys(eval_loader, server_loader, self.eval_batches)
        first_grad_norm, first_obj = _prox_diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device,
        )

        # Inner-loop setup.
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

        # Run num_steps full epochs (with possible early stop).
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
                    # AccVRS-faithful direction:
                    #   d = (param − w_outer) + θ · ∇f₁(param; batch).
                    # The linear ⟨θv, ·⟩ term is encoded via the warm-start
                    # z = w_outer − θv (NOT in d).  Set
                    # include_linear_term=True to add  +θv  back to d so the
                    # SGD fixed point is the actual prox_{θf₁}(w_outer − θv).
                    d = [
                        (p.data - wo) + theta * g
                        for p, wo, g in zip(params, w_outer, grads)
                    ]
                    if self.include_linear_term:
                        for di, vi in zip(d, v):
                            di.add_(vi, alpha=theta)

                    # Polyak momentum on the direction (their default 0.9).
                    if self.momentum > 0 and prev_direction is not None:
                        d = [
                            self.momentum * pd + di
                            for pd, di in zip(prev_direction, d)
                        ]
                    prev_direction = [di.clone() for di in d]

                    # Per-element clip (their `clip_grads`).
                    if self.grad_clip > 0:
                        if _clip_per_element_(d, self.grad_clip):
                            clip_count += 1

                    # Capture starting direction norm on iteration 0 (their
                    # original ``start_norm`` metric — measured on the running
                    # training batch, so noisier than the eval-batch one).
                    if total_steps == 0:
                        start_norm_inner = _direction_norm(d)
                        cur_norm_inner = start_norm_inner

                    # Solver-specific update.
                    self._apply_update(params, d, lr, state, total_steps + 1)

                total_steps += 1

                # Periodic norm check + LR decay + early stop.
                if total_steps % decay_period == 0:
                    cur_norm_inner = _direction_norm(d)
                    last_norm_inner = cur_norm_inner
                    lr *= self.inner_decay_factor
                    ratio = cur_norm_inner / max(start_norm_inner, 1e-12)
                    if ratio < self.early_stop_ratio:
                        broken = True
                        break

        # Final norm fallback if the last `total_steps` didn't hit a
        # decay_period boundary.
        if last_norm_inner == 0.0 and prev_direction is not None:
            last_norm_inner = _direction_norm(prev_direction)
        last_frac_inner = last_norm_inner / max(start_norm_inner, 1e-12)

        # Final eval-batch diagnostic.
        last_grad_norm, last_obj = _prox_diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device,
        )

        payload = _build_diag_payload(
            first_grad_norm, first_obj,
            last_grad_norm, last_obj,
            self.num_steps,
            clip_count=clip_count,
        )
        # Override the configured num_steps with the actual count taken.
        payload["prox_inner_steps"] = float(total_steps)
        # AccVRS-specific extras: training-batch direction-norm tracking.
        payload["prox_inner_norm_first"] = float(start_norm_inner)
        payload["prox_inner_norm_last"] = float(last_norm_inner)
        payload["prox_inner_norm_ratio"] = float(last_frac_inner)
        return payload


class AccvrsBatchSGDProx(_AccvrsBatchBase):
    """AccVRS Batch_SGD inexact prox solver.

    Inner update:  ``param ← param · (1 − wd·lr) − lr · d``.
    """

    def _apply_update(
        self,
        params: list[torch.Tensor],
        direction: list[torch.Tensor],
        lr: float,
        state: dict,
        step_idx: int,
    ) -> None:
        decay_mult = 1.0 - self.weight_decay * lr
        for p, d in zip(params, direction):
            if self.weight_decay > 0:
                p.data.mul_(decay_mult)
            p.data.add_(d, alpha=-lr)


class AccvrsBatchAdamProx(_AccvrsBatchBase):
    """AccVRS Batch_Adam inexact prox solver.

    Standard Adam-with-bias-correction on the same direction ``d`` as
    Batch_SGD; ``weight_decay`` (if > 0) is applied as a multiplicative
    parameter shrink before the Adam update (AdamW-style decoupled).
    """

    def __init__(
        self,
        *args,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)

    def _init_state(self, params: list[torch.Tensor]) -> dict:
        return {
            "m": [torch.zeros_like(p) for p in params],
            "v": [torch.zeros_like(p) for p in params],
        }

    def _apply_update(
        self,
        params: list[torch.Tensor],
        direction: list[torch.Tensor],
        lr: float,
        state: dict,
        step_idx: int,
    ) -> None:
        b1, b2 = self.betas
        m_state = state["m"]
        v_state = state["v"]
        bias1 = 1.0 - b1 ** step_idx
        bias2 = 1.0 - b2 ** step_idx
        decay_mult = 1.0 - self.weight_decay * lr
        for i, (p, d) in enumerate(zip(params, direction)):
            m_state[i].mul_(b1).add_(d, alpha=1.0 - b1)
            v_state[i].mul_(b2).addcmul_(d, d, value=1.0 - b2)
            m_hat = m_state[i] / bias1
            v_hat = v_state[i] / bias2
            if self.weight_decay > 0:
                p.data.mul_(decay_mult)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)


class AccXtraGradBatchSGDProx(ProxSolver):
    """Batch_SGD argmin from AccXtraGrad (direct port).

    Mirrors the ``optimizer_name == "Batch_SGD"`` branch from AccXtraGrad's
    ``argmin``: warm-start at ``z = w_outer − θ v``, then run SGD on

        d = (w − w_outer) + θ · ∇f₁(w),

    with EMA-style momentum, periodic LR decay, and early-stop on
    ``‖d‖ / ‖d_first‖``.  This omits the linear ``θ v`` term inside the
    direction — consistent with the AccXtraGrad snippet.
    """

    def __init__(
        self,
        num_steps: int,
        lr: float | None = None,
        L1: float = 200.0,
        lr_factor: float = 1.0,
        momentum: float = 0.9,
        inner_decay_factor: float = 0.9,
        inner_decay_period: int | None = None,
        early_stop_ratio: float = 1e-3,
        eval_batches: int = 1,
    ) -> None:
        self.num_steps = int(num_steps)
        self.lr = None if lr is None else float(lr)
        self.L1 = float(L1)
        self.lr_factor = float(lr_factor)
        self.momentum = float(momentum)
        self.inner_decay_factor = float(inner_decay_factor)
        self.inner_decay_period = (
            int(inner_decay_period) if inner_decay_period is not None else None
        )
        self.early_stop_ratio = float(early_stop_ratio)
        self.eval_batches = max(1, int(eval_batches))

    def _resolve_lr(self, theta: float) -> float:
        if self.lr is not None:
            return self.lr * self.lr_factor
        L = 1.0 + theta * self.L1
        return (1.0 / (2.0 * L)) * self.lr_factor

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
        params = list(model.parameters())

        eval_xys = _fresh_eval_xys(eval_loader, server_loader, self.eval_batches)
        first_grad_norm, first_obj = _prox_diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device,
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
        prev_direction: list[torch.Tensor] | None = None
        start_norm_inner = 0.0
        last_norm_inner = 0.0
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
                    direction = [
                        (p.data - wo) + theta * g
                        for p, wo, g in zip(params, w_outer, grads)
                    ]
                    if self.momentum > 0 and prev_direction is not None:
                        direction = [
                            self.momentum * pd + (1.0 - self.momentum) * di
                            for pd, di in zip(prev_direction, direction)
                        ]
                    prev_direction = [di.clone() for di in direction]

                    if total_steps == 0:
                        start_norm_inner = _direction_norm(direction)

                    for p, d in zip(params, direction):
                        p.data.add_(d, alpha=-lr)

                total_steps += 1

                if total_steps % decay_period == 0:
                    last_norm_inner = _direction_norm(direction)
                    lr *= self.inner_decay_factor
                    ratio = last_norm_inner / max(start_norm_inner, 1e-12)
                    if ratio < self.early_stop_ratio:
                        broken = True
                        break

        if last_norm_inner == 0.0 and prev_direction is not None:
            last_norm_inner = _direction_norm(prev_direction)
        last_frac_inner = last_norm_inner / max(start_norm_inner, 1e-12)

        last_grad_norm, last_obj = _prox_diag_on_batches(
            params, z, theta, model, loss_fn, eval_xys, device,
        )

        payload = _build_diag_payload(
            first_grad_norm, first_obj,
            last_grad_norm, last_obj,
            self.num_steps,
        )
        payload["prox_inner_steps"] = float(total_steps)
        payload["prox_inner_norm_first"] = float(start_norm_inner)
        payload["prox_inner_norm_last"] = float(last_norm_inner)
        payload["prox_inner_norm_ratio"] = float(last_frac_inner)
        return payload
