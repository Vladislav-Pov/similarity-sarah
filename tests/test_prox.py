"""Prox solvers: auto-gamma0, prox-does-not-worsen-Phi, v_schedule, from_spec."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from similarity_sarah.core.repro import set_seed
from similarity_sarah.prox import AccvrsBatchSGD, InexactProxSGD, build_prox_solver
from similarity_sarah.spec import parse_run_spec

_DEVICE = torch.device("cpu")
_THETA = 0.2
_REF = (
    Path(__file__).resolve().parent.parent
    / "docs" / "reference_runs" / "best_run_constant_4steps.json"
)


def _build_problem(seed: int):
    set_seed(seed, deterministic=True)
    model = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 4))
    dataset = TensorDataset(torch.randn(64, 12), torch.randint(0, 4, (64,)))
    server_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    v = [torch.randn_like(p) * 0.01 for p in model.parameters()]
    return model, v, server_loader, eval_loader, nn.CrossEntropyLoss()


def _run(make_solver, seed: int):
    model, v, server_loader, eval_loader, loss_fn = _build_problem(seed)
    make_solver().step(
        model, v, _THETA, server_loader, loss_fn, _DEVICE, eval_loader=eval_loader
    )
    return [p.detach().clone() for p in model.parameters()]


def test_accvrs_auto_gamma0():
    solver = AccvrsBatchSGD(num_steps=4, lr=None, L1=200.0, lr_factor=0.1)
    # L = 1 + 0.2*200 = 41 ; gamma0 = (1/(2*41)) * 0.1 = 0.1/82.
    assert solver._resolve_lr(_THETA) == pytest.approx(0.1 / 82.0, rel=0, abs=1e-15)


def test_prox_does_not_worsen_phi():
    """Convex f1 + full batch + small lr => Phi decreases (brief invariant)."""
    set_seed(0, deterministic=True)
    model = nn.Linear(8, 3)
    dataset = TensorDataset(torch.randn(40, 8), torch.randint(0, 3, (40,)))
    full = DataLoader(dataset, batch_size=40, shuffle=False, num_workers=0)
    v = [torch.randn_like(p) * 0.05 for p in model.parameters()]
    solver = InexactProxSGD(num_steps=25, lr=0.02, eval_batches=1)
    diag = solver.step(model, v, 0.5, full, nn.CrossEntropyLoss(), _DEVICE, eval_loader=full)
    assert diag["prox_obj_decrease"] >= 0.0


def test_v_schedule_affects_inexact_sgd():
    """`constant` and `linear` differ for InexactProx (unlike the AccVRS path)."""
    wc = _run(lambda: InexactProxSGD(num_steps=5, lr=0.05, v_schedule="constant"), 3)
    wl = _run(lambda: InexactProxSGD(num_steps=5, lr=0.05, v_schedule="linear"), 3)
    assert any(not torch.equal(a, b) for a, b in zip(wc, wl))


def test_build_prox_solver_from_reference_spec():
    raw = json.loads(_REF.read_text())
    raw.pop("_meta", None)
    spec = parse_run_spec(OmegaConf.create(raw))
    solver = build_prox_solver(spec.prox)
    assert isinstance(solver, AccvrsBatchSGD)
    assert solver.lr is None
    assert solver.num_steps == 4
    assert solver.momentum == 0.9
    assert solver._resolve_lr(_THETA) == pytest.approx(0.1 / 82.0)
