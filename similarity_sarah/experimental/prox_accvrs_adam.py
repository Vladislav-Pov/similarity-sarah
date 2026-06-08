"""AccVRS Batch_Adam inexact prox solver — NOT in the submission.

Tried during tuning; it did not improve over :class:`AccvrsBatchSGD`, so it is
out of scope (``REWRITE_BRIEF.md``). Kept here as an opt-in: importing this
module registers ``accvrs_batch_adam`` with ``PROX_SOLVERS``. The main hot path
never imports it.
"""

from __future__ import annotations

import torch

from similarity_sarah.prox.accvrs import _AccvrsBatchBase
from similarity_sarah.prox.base import PROX_SOLVERS, ProxSolver
from similarity_sarah.spec import ProxSpec


@PROX_SOLVERS.register("accvrs_batch_adam", "accvrs_adam")
class AccvrsBatchAdam(_AccvrsBatchBase):
    """AccVRS direction with a standard bias-corrected Adam update.

    ``weight_decay`` (if > 0) is applied as a multiplicative parameter shrink
    before the Adam step (AdamW-style decoupled).
    """

    def __init__(
        self,
        *args,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)

    def _init_state(self, params: list[torch.Tensor]) -> dict:
        return {
            "m": [torch.zeros_like(p) for p in params],
            "v": [torch.zeros_like(p) for p in params],
        }

    def _apply_update(self, params, direction, lr, state, step_idx):
        b1, b2 = self.betas
        m_state, v_state = state["m"], state["v"]
        bias1 = 1.0 - b1**step_idx
        bias2 = 1.0 - b2**step_idx
        decay_mult = 1.0 - self.weight_decay * lr
        for i, (p, d) in enumerate(zip(params, direction)):
            m_state[i].mul_(b1).add_(d, alpha=1.0 - b1)
            v_state[i].mul_(b2).addcmul_(d, d, value=1.0 - b2)
            m_hat = m_state[i] / bias1
            v_hat = v_state[i] / bias2
            if self.weight_decay > 0:
                p.data.mul_(decay_mult)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)

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
            betas=spec.adam_betas,
        )
