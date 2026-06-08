"""``spec.parse_run_spec`` must reproduce the pre-rewrite config values.

Pinned against the three ``docs/reference_runs/*.json`` snapshots plus a
minimal config that exercises the ``OmegaConf.select`` defaults.
"""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from similarity_sarah.spec import parse_run_spec

REF_DIR = Path(__file__).resolve().parent.parent / "docs" / "reference_runs"
REF_FILES = [
    "best_run_constant_4steps.json",
    "best_run_linear_4steps.json",
    "best_run_linear_5steps.json",
]


def _load_cfg(name: str):
    raw = json.loads((REF_DIR / name).read_text())
    raw.pop("_meta", None)
    return OmegaConf.create(raw)


@pytest.mark.parametrize("name", REF_FILES)
def test_parse_reference_config(name):
    spec = parse_run_spec(_load_cfg(name))

    assert spec.seed == 42
    assert spec.algorithm_name == "batched_nfg_sarah"
    assert spec.theta == 0.2
    assert spec.lr is None
    assert spec.num_clients == 10
    assert spec.batch_size_clients == 1
    assert spec.partition.server_fraction == 0.5
    assert spec.data.name == "cifar10"
    assert spec.data.val_fraction == 0.2
    assert spec.model.name == "resnet18_32x32"
    assert spec.runtime.batch_size == 512
    assert spec.runtime.batch_size_server_grad == 512
    assert spec.runtime.num_workers == 4
    assert spec.runtime.deterministic is False

    prox = spec.prox
    assert prox is not None
    assert prox.kind == "accvrs_batch_sgd"
    assert prox.lr is None
    assert prox.L1 == 200.0
    assert prox.lr_factor == 0.1
    assert prox.momentum == 0.9
    assert prox.weight_decay == 0.1
    assert prox.grad_clip == 0.0
    assert prox.include_linear_term is False
    assert prox.early_stop_ratio == 1e-4
    assert prox.eval_batches == 4
    assert prox.inner_decay_period is None


def test_num_steps_and_schedule_differ_by_file():
    specs = {name: parse_run_spec(_load_cfg(name)) for name in REF_FILES}
    assert specs["best_run_constant_4steps.json"].prox.num_steps == 4
    assert specs["best_run_linear_4steps.json"].prox.num_steps == 4
    assert specs["best_run_linear_5steps.json"].prox.num_steps == 5
    assert specs["best_run_constant_4steps.json"].prox.v_schedule == "constant"
    assert specs["best_run_linear_4steps.json"].prox.v_schedule == "linear"


def test_defaults_when_fields_absent():
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "algorithm": {
                "name": "fedavg",
                "num_epochs": 1,
                "num_clients": 2,
                "batch_size_clients": 1,
                "lr": 0.1,
            },
            "data": {"name": "synthetic"},
            "model": {"name": "simple_cnn"},
            "partition": {"name": "uniform"},
            "runtime": {},
        }
    )
    spec = parse_run_spec(cfg)
    assert spec.prox is None  # fedavg has no prox subproblem
    assert spec.lr == 0.1
    assert spec.theta is None
    assert spec.runtime.batch_size == 512
    assert spec.runtime.large_batch_size == 512  # mirrors batch_size default
    assert spec.runtime.batch_size_server_prox == 512
    assert spec.partition.server_fraction is None
    assert spec.data.val_fraction == 0.0
    assert spec.runtime.prox_lr_schedule is None
