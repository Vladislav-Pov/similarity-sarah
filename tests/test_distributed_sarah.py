"""Invariants for the Distributed NoFullGrad SARAH baseline."""

import math

from similarity_sarah.algorithms.base import AlgorithmCtx
from similarity_sarah.algorithms.distributed_sarah import DistributedSARAH
from similarity_sarah.core.params import compute_param_norm, get_params
from tests.golden_oracle import build_fast_setup


def _algo(lr: float, weight_decay: float = 0.0, include_server: bool = True):
    model, sg, sp, clients, loss_fn, device = build_fast_setup()
    algo = DistributedSARAH(
        lr=lr,
        batch_size_clients=1,
        weight_decay=weight_decay,
        include_server=include_server,
    )
    algo.bind(AlgorithmCtx(model, sg, sp, clients, loss_fn, device))
    return algo, model


def test_runs_and_metrics_finite():
    algo, model = _algo(lr=0.05)
    out = algo.run_epoch(0)
    # 11 partitions = 1 server + 10 clients, B=1 -> 10 inner steps.
    assert out["inner_steps"] == 10.0
    assert math.isfinite(out["v_norm"])
    assert math.isfinite(compute_param_norm(get_params(model)))


def test_lr_zero_leaves_weights_unchanged():
    # With lr=0 every step is w <- w, so the weights never move (exactly).
    algo, model = _algo(lr=0.0, weight_decay=1e-3)
    before = compute_param_norm(get_params(model))
    algo.run_epoch(0)
    assert compute_param_norm(get_params(model)) == before


def test_carryover_anchor_seeded_from_running_mean():
    # v_epoch (the no-full-gradient carry-over) starts at 0 and is a nonzero
    # running mean after one epoch — this is what the next epoch's v_0 reuses.
    algo, _ = _algo(lr=0.05)
    assert compute_param_norm(algo.v_epoch) == 0.0
    algo.run_epoch(0)
    assert compute_param_norm(algo.v_epoch) > 0.0


def test_cosine_lr_schedule():
    algo = DistributedSARAH(
        lr=0.1, batch_size_clients=1, num_epochs=10,
        lr_schedule="cosine", lr_min_factor=0.0,
    )
    assert algo._epoch_lr(0) == 0.1                # epoch 0 → lr
    assert abs(algo._epoch_lr(9)) < 1e-6           # last epoch → lr * lr_min_factor
    assert 0.0 < algo._epoch_lr(5) < 0.1           # decays in between
    const = DistributedSARAH(lr=0.1, batch_size_clients=1, num_epochs=10)
    assert const._epoch_lr(0) == const._epoch_lr(9) == 0.1  # constant is default


def test_weight_decay_changes_trajectory():
    # Non-zero weight decay must move the iterate to a different point than wd=0.
    plain, plain_model = _algo(lr=0.05, weight_decay=0.0)
    decayed, decayed_model = _algo(lr=0.05, weight_decay=1e-2)
    plain.run_epoch(0)
    decayed.run_epoch(0)
    assert compute_param_norm(get_params(plain_model)) != compute_param_norm(
        get_params(decayed_model)
    )
