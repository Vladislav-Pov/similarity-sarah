"""End-to-end pipeline: loaders, baselines, and a full-stack golden match."""

import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from similarity_sarah.algorithms.base import ALGORITHMS, AlgorithmCtx
from similarity_sarah.core.repro import make_generator, set_seed
from similarity_sarah.data.loaders import build_federated_data
from similarity_sarah.models import build_model
from similarity_sarah.runtime.loop import run_experiment
from similarity_sarah.spec import parse_run_spec
from similarity_sarah.tasks.classification import ClassificationTask
from tests.golden_oracle import _weight_sha256

GOLDEN_DIR = Path(__file__).parent / "golden"
_CPU = torch.device("cpu")

_BASE_RUNTIME = {
    "device": "cpu",
    "deterministic": False,
    "eval_every": 999,
    "num_workers": 0,
    "batch_size": 64,
    "large_batch_size": 64,
    "batch_size_server_grad": 64,
    "batch_size_server_prox": 64,
    "batch_size_data_clients": 64,
    "prox_lr_schedule": {"kind": "constant"},
}
_DATA = {"name": "synthetic", "val_fraction": 0.1, "n_train": 500, "n_test": 100}
_PARTITION = {"name": "uniform", "server_fraction": 0.5}
_MODEL = {"name": "simple_cnn", "num_classes": 10}


def _nfg_cfg(**runtime_overrides):
    runtime = {**_BASE_RUNTIME, **runtime_overrides}
    return OmegaConf.create(
        {
            "seed": 42,
            "algorithm": {
                "name": "nfg_ss", "theta": 0.2, "num_epochs": 1, "num_clients": 10,
                "batch_size_clients": 1, "prox_solver": "accvrs_batch_sgd",
                "prox_lr": None, "prox_L1": 200, "prox_lr_factor": 0.1,
                "prox_num_steps": 4, "prox_momentum": 0.9, "prox_weight_decay": 0.1,
                "prox_grad_clip": 0, "prox_eval_batches": 2, "prox_inner_decay_factor": 1,
                "prox_inner_decay_period": None, "prox_include_linear_term": False,
                "prox_inner_early_stop_ratio": 1e-4, "prox_v_schedule": "constant",
            },
            "data": _DATA, "model": _MODEL, "partition": _PARTITION, "runtime": runtime,
        }
    )


def _baseline_cfg(algorithm: dict):
    return OmegaConf.create(
        {
            "seed": 42, "algorithm": algorithm, "data": _DATA, "model": _MODEL,
            "partition": _PARTITION, "runtime": _BASE_RUNTIME,
        }
    )


def _build_and_run(spec, algo_name: str):
    set_seed(spec.seed, deterministic=spec.runtime.deterministic)
    data = build_federated_data(spec)
    model = build_model(spec.model.name, spec.model.num_classes)
    task = ClassificationTask()
    algo = ALGORITHMS.get(algo_name).from_spec(spec)
    algo.bind(
        AlgorithmCtx(
            model, data.server_grad_loader, data.server_prox_loader,
            data.client_loaders, task.loss_fn, _CPU,
        )
    )
    metrics = algo.run_epoch(0)
    return _weight_sha256(model), metrics


def test_build_federated_data_structure():
    spec = parse_run_spec(_nfg_cfg())
    set_seed(spec.seed)
    data = build_federated_data(spec)
    assert len(data.client_loaders) == 10
    assert data.val_loader is not None
    assert data.test_loader is not None
    x, _ = next(iter(data.server_grad_loader))
    assert x.shape[0] == 64


def test_loaders_reproducible_same_seed():
    spec = parse_run_spec(_nfg_cfg())

    def first_batch():
        set_seed(spec.seed)
        data = build_federated_data(spec)
        return next(iter(data.server_grad_loader))[0]

    assert torch.equal(first_batch(), first_batch())


def test_seeded_generator_is_reproducible():
    spec = parse_run_spec(_nfg_cfg(deterministic=True))

    def first_batch():
        set_seed(spec.seed, deterministic=True)
        data = build_federated_data(spec, generator=make_generator(spec.seed))
        return next(iter(data.server_grad_loader))[0]

    assert torch.equal(first_batch(), first_batch())


def test_full_stack_nfg_ss_matches_golden():
    """spec -> loaders -> model -> algo reproduces the fast golden bit-for-bit."""
    spec = parse_run_spec(_nfg_cfg())
    sha, _ = _build_and_run(spec, "nfg_ss")
    golden = json.loads((GOLDEN_DIR / "fast_constant_4steps.json").read_text())
    assert sha == golden["weight_sha256"]


def test_svrs_runs_and_is_deterministic():
    cfg = _baseline_cfg(
        {
            "name": "svrs", "theta": 0.5, "num_epochs": 1, "num_clients": 10,
            "batch_size_clients": 1, "prox_solver": "sgd", "prox_lr": 0.01,
            "prox_num_steps": 5, "prox_momentum": 0.0, "prox_eval_batches": 1,
        }
    )
    spec = parse_run_spec(cfg)
    sha1, m1 = _build_and_run(spec, "svrs")
    sha2, _ = _build_and_run(spec, "svrs")
    assert sha1 == sha2
    assert m1["inner_steps"] >= 1
    assert math.isfinite(m1["param_norm"])


def test_fedavg_runs_and_is_deterministic():
    cfg = _baseline_cfg(
        {
            "name": "fedavg", "lr": 0.01, "num_epochs": 1, "num_clients": 10,
            "batch_size_clients": 1, "include_server": True,
        }
    )
    spec = parse_run_spec(cfg)
    sha1, m1 = _build_and_run(spec, "fedavg")
    sha2, _ = _build_and_run(spec, "fedavg")
    assert sha1 == sha2
    assert m1["inner_steps"] == 10
    assert math.isfinite(m1["param_norm"])


def test_run_experiment_completes_with_eval():
    spec = parse_run_spec(_nfg_cfg(eval_every=1))
    summary = run_experiment(spec)
    assert "best_val_accuracy" in summary
    assert "best_val_loss" in summary
