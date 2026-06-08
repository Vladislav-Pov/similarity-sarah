"""Bit-reproducibility gate against the committed fast-oracle goldens.

At M1 these fixtures are captured from the pre-rewrite code; the same oracle is
re-pointed at the rewritten code in later milestones. A mismatch means the
rewrite changed the NFG-SS / AccVRS numerical trajectory.
"""

import json
from pathlib import Path

import pytest

from tests.golden_oracle import FAST_VARIANTS, build_and_run

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.mark.parametrize("variant", FAST_VARIANTS, ids=lambda v: v.name)
def test_fast_oracle_matches_golden(variant):
    golden_path = GOLDEN_DIR / f"{variant.name}.json"
    assert golden_path.exists(), (
        f"missing golden {golden_path}; run `python scripts/capture_golden.py`"
    )
    golden = json.loads(golden_path.read_text())
    result = build_and_run(variant)

    assert result["weight_sha256"] == golden["weight_sha256"], (
        f"{variant.name}: final-weight hash drifted from golden"
    )
    # The rewritten code emits a clean subset of the original diagnostics; each
    # emitted metric must still match the captured golden value.
    for key, got in result["metrics"].items():
        assert got == pytest.approx(golden["metrics"][key], rel=1e-9, abs=1e-12), key


def test_v_schedule_is_noop_on_accvrs_path():
    """`constant` and `linear` at the same num_steps are bit-identical here.

    Documents that ``AccvrsBatchSGD`` ignores ``prox_v_schedule``.
    """
    const = json.loads((GOLDEN_DIR / "fast_constant_4steps.json").read_text())
    linear = json.loads((GOLDEN_DIR / "fast_linear_4steps.json").read_text())
    assert const["weight_sha256"] == linear["weight_sha256"]
