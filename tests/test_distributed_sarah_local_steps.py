"""Invariants for the Distributed SARAH + local-step-prox baseline."""

import math

from similarity_sarah.algorithms.base import AlgorithmCtx
from similarity_sarah.algorithms.distributed_sarah_local_steps import (
    DistributedSARAHLocalSteps,
)
from similarity_sarah.core.params import compute_param_norm, get_params
from similarity_sarah.prox import AccvrsBatchSGD
from tests.golden_oracle import build_fast_setup


def _solver() -> AccvrsBatchSGD:
    return AccvrsBatchSGD(
        num_steps=4, lr=None, L1=200.0, lr_factor=0.1, weight_decay=0.1,
        momentum=0.9, grad_clip=0.0, inner_decay_factor=1.0, inner_decay_period=None,
        early_stop_ratio=1e-4, include_linear_term=False, eval_batches=2,
    )


def _algo():
    model, sg, sp, clients, loss_fn, device = build_fast_setup()
    algo = DistributedSARAHLocalSteps(
        theta=0.2, batch_size_clients=1, prox_solver=_solver()
    )
    algo.bind(AlgorithmCtx(model, sg, sp, clients, loss_fn, device))
    return algo, model


def test_runs_and_metrics_finite():
    algo, _ = _algo()
    out = algo.run_epoch(0)
    assert out["inner_steps"] == 10.0
    assert math.isfinite(out["param_norm"])
    assert math.isfinite(out["prox_grad_norm_last_mean"])


def test_prox_moves_weights():
    # The f1-prox is the active step, so the iterate must change each epoch.
    algo, model = _algo()
    before = compute_param_norm(get_params(model))
    algo.run_epoch(0)
    assert compute_param_norm(get_params(model)) != before


def test_carryover_anchor_seeded_from_running_mean():
    # tilde_v carry-over is still maintained (even though it is inert per spec).
    algo, _ = _algo()
    assert compute_param_norm(algo.v_epoch) == 0.0
    algo.run_epoch(0)
    assert compute_param_norm(algo.v_epoch) > 0.0
