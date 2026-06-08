"""NFG-SS: bit-reproduction of the goldens + the load-bearing invariants."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from similarity_sarah.algorithms.base import AlgorithmCtx
from similarity_sarah.algorithms.nfg_ss import NFGSS
from similarity_sarah.core.grads import compute_batch_gradient
from similarity_sarah.core.params import (
    compute_param_norm,
    get_params,
    set_params,
)
from similarity_sarah.core.repro import set_seed
from similarity_sarah.prox import AccvrsBatchSGD
from similarity_sarah.prox.base import ProxSolver
from tests.golden_oracle import FAST_VARIANTS, _weight_sha256, build_fast_setup

GOLDEN_DIR = Path(__file__).parent / "golden"
_CPU = torch.device("cpu")
_DUMMY_DIAG = {
    "prox_grad_norm_first": 0.0,
    "prox_grad_norm_last": 0.0,
    "prox_grad_norm_ratio": 0.0,
    "prox_obj_decrease": 0.0,
    "prox_inner_steps": 0.0,
    "prox_clip_frac": 0.0,
}


class _IdentityProx(ProxSolver):
    """Test solver that leaves the model unchanged."""

    lr = None

    def step(self, model, v, theta, server_loader, loss_fn, device, eval_loader=None) -> Mapping[str, float]:
        return dict(_DUMMY_DIAG)


class _ShiftProx(ProxSolver):
    """Test solver that adds a fixed constant to every parameter."""

    lr = None

    def __init__(self, c: float) -> None:
        self.c = c

    def step(self, model, v, theta, server_loader, loss_fn, device, eval_loader=None) -> Mapping[str, float]:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(self.c)
        return dict(_DUMMY_DIAG)


def _new_accvrs(num_steps: int) -> AccvrsBatchSGD:
    return AccvrsBatchSGD(
        num_steps=num_steps, lr=None, L1=200.0, lr_factor=0.1, weight_decay=0.1,
        momentum=0.9, grad_clip=0.0, inner_decay_factor=1.0, inner_decay_period=None,
        early_stop_ratio=1e-4, include_linear_term=False, eval_batches=2,
    )


@pytest.mark.parametrize("variant", FAST_VARIANTS, ids=lambda v: v.name)
def test_new_nfg_ss_matches_golden(variant):
    """The rewritten NFG-SS reproduces the pre-rewrite goldens bit-for-bit."""
    golden = json.loads((GOLDEN_DIR / f"{variant.name}.json").read_text())
    model, sg, sp, clients, loss_fn, device = build_fast_setup()
    algo = NFGSS(theta=0.2, batch_size_clients=1, prox_solver=_new_accvrs(variant.prox_num_steps))
    algo.bind(AlgorithmCtx(model, sg, sp, clients, loss_fn, device))
    metrics = {k: float(v) for k, v in algo.run_epoch(0).items()}

    assert _weight_sha256(model) == golden["weight_sha256"], variant.name
    for key, val in metrics.items():
        assert val == pytest.approx(golden["metrics"][key], rel=1e-9, abs=1e-12), key


def _small_ctx(num_clients: int = 3, seed: int = 0) -> AlgorithmCtx:
    set_seed(seed, deterministic=True)
    model = nn.Linear(6, 3)

    def loader(n: int) -> DataLoader:
        dataset = TensorDataset(torch.randn(n, 6), torch.randint(0, 3, (n,)))
        return DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    server = loader(40)
    clients = [loader(20) for _ in range(num_clients)]
    return AlgorithmCtx(model, server, server, clients, nn.CrossEntropyLoss(), _CPU)


def test_telescope_zero_increment_when_prox_does_not_move():
    """Same-minibatch SARAH increment is exactly 0 when w_t == w_{t-1}.

    Guards the section-11 invariant: if a *different* minibatch were used at
    w_{t-1}, the increment would be nonzero and ``v_norm`` would not be 0.
    """
    ctx = _small_ctx()
    algo = NFGSS(theta=0.1, batch_size_clients=1, prox_solver=_IdentityProx())
    algo.bind(ctx)
    metrics = algo.run_epoch(0)
    assert metrics["v_norm"] == 0.0
    assert metrics["tilde_v_norm"] > 0.0  # running mean still accumulates


def test_running_mean_carryover():
    """v^{(s+1)} is the within-epoch running mean (no full gradient refresh)."""
    ctx = _small_ctx()
    algo = NFGSS(theta=0.1, batch_size_clients=1, prox_solver=_IdentityProx())
    assert compute_param_norm(algo.v_epoch) == 0.0  # tilde_v_1^{(0)} = 0
    algo.bind(ctx)
    assert compute_param_norm(algo.v_epoch) == 0.0  # still zero after bind
    metrics = algo.run_epoch(0)
    assert compute_param_norm(algo.v_epoch) == pytest.approx(metrics["tilde_v_norm"])
    assert metrics["v_epoch_norm"] == pytest.approx(metrics["tilde_v_norm"])


def test_sarah_increment_uses_one_over_nB():
    """The SARAH increment prefactor is 1/(n*B), not the paper's 1/b.

    With one client (n=2, B=1) the prefactor is 1/2. A shift-prox makes
    w_1 = w_0 + c so the increment is nonzero, and deterministic full-batch
    loaders let us re-derive v independently.
    """
    set_seed(1, deterministic=True)
    model = nn.Linear(5, 2)
    server_ds = TensorDataset(torch.randn(12, 5), torch.randint(0, 2, (12,)))
    client_ds = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
    server = DataLoader(server_ds, batch_size=64, shuffle=False, num_workers=0)
    client = DataLoader(client_ds, batch_size=64, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    c = 0.01

    w0 = get_params(model)
    set_params(model, w0)
    g1_prev = compute_batch_gradient(model, server, loss_fn, _CPU)
    gi_prev = compute_batch_gradient(model, client, loss_fn, _CPU)
    set_params(model, [w + c for w in w0])
    g1_curr = compute_batch_gradient(model, server, loss_fn, _CPU)
    gi_curr = compute_batch_gradient(model, client, loss_fn, _CPU)
    diff_prev = [gi - g1 for gi, g1 in zip(gi_prev, g1_prev)]
    diff_curr = [gi - g1 for gi, g1 in zip(gi_curr, g1_curr)]
    expected_v = [0.5 * (dc - dp) for dc, dp in zip(diff_curr, diff_prev)]
    expected_norm = compute_param_norm(expected_v)

    set_params(model, w0)
    ctx = AlgorithmCtx(model, server, server, [client], loss_fn, _CPU)
    algo = NFGSS(theta=0.3, batch_size_clients=1, prox_solver=_ShiftProx(c))
    algo.bind(ctx)
    metrics = algo.run_epoch(0)
    assert metrics["v_norm"] == pytest.approx(expected_norm, rel=1e-6, abs=1e-9)
