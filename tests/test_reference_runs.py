"""Reference-config integrity: the three snapshots + best.yaml build correctly.

The full CIFAR-10 / ResNet-18 reference runs are GPU jobs beyond the local
5-minute budget; their bit-reproduction is exercised on synthetic data by the
fast golden (``test_nfg_ss``/``test_pipeline``). Here we verify the rewritten
code parses and constructs the exact reference configurations — same algorithm,
same AccVRS solver, same auto-gamma0.
"""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from similarity_sarah.algorithms.base import ALGORITHMS
from similarity_sarah.algorithms.nfg_ss import NFGSS
from similarity_sarah.prox import AccvrsBatchSGD, build_prox_solver
from similarity_sarah.spec import parse_prox_spec, parse_run_spec

_ROOT = Path(__file__).resolve().parent.parent
_REF_DIR = _ROOT / "docs" / "reference_runs"
_BEST_YAML = _ROOT / "configs" / "algorithm" / "best.yaml"
_REF_FILES = [
    "best_run_constant_4steps.json",
    "best_run_linear_4steps.json",
    "best_run_linear_5steps.json",
]
_GAMMA0 = 0.1 / 82.0  # (1 / (2 * (1 + 0.2 * 200))) * 0.1


def _ref_cfg(name: str):
    raw = json.loads((_REF_DIR / name).read_text())
    raw.pop("_meta", None)
    return OmegaConf.create(raw)


@pytest.mark.parametrize("name", _REF_FILES)
def test_reference_config_builds_to_nfg_ss(name):
    spec = parse_run_spec(_ref_cfg(name))
    algo = ALGORITHMS.get(spec.algorithm_name).from_spec(spec)
    assert isinstance(algo, NFGSS)
    assert algo.theta == 0.2
    assert algo.batch_size_clients == 1

    solver = algo.prox_solver
    assert isinstance(solver, AccvrsBatchSGD)
    assert solver.lr is None
    assert solver._resolve_lr(0.2) == pytest.approx(_GAMMA0)


def test_best_yaml_encodes_the_winning_formula():
    algo_cfg = OmegaConf.load(_BEST_YAML)
    assert algo_cfg.name in ("nfg_ss", "batched_nfg_sarah")
    assert float(algo_cfg.theta) == 0.2
    assert int(algo_cfg.batch_size_clients) == 1

    prox = parse_prox_spec(algo_cfg)
    assert prox.kind == "accvrs_batch_sgd"
    assert prox.lr is None
    assert prox.L1 == 200.0
    assert prox.lr_factor == 0.1
    assert prox.num_steps == 4
    assert prox.momentum == 0.9
    assert prox.weight_decay == 0.1
    assert prox.grad_clip == 0.0
    assert prox.include_linear_term is False
    assert prox.early_stop_ratio == 1e-4

    solver = build_prox_solver(prox)
    assert isinstance(solver, AccvrsBatchSGD)
    assert solver._resolve_lr(float(algo_cfg.theta)) == pytest.approx(_GAMMA0)
