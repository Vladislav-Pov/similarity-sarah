"""Capture the fast-oracle golden fixtures used by ``tests/test_golden.py``.

Run from the repo root *before* the rewrite touches the hot path:

    python scripts/capture_golden.py

Writes one ``tests/golden/<variant>.json`` per fast variant. Each holds the
final-weight SHA-256 and the epoch metrics of one NFG-SS epoch on the tiny
synthetic / SimpleCNN / CPU configuration (seconds to run).

The full CIFAR-10 / ResNet-18 reference configs are GPU jobs beyond the local
5-minute budget and are not captured here; reproduce them on the original
hardware from baseline commit 4ca4873.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests.golden_oracle import FAST_VARIANTS, build_and_run  # noqa: E402

GOLDEN_DIR = _REPO_ROOT / "tests" / "golden"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for variant in FAST_VARIANTS:
        result = build_and_run(variant)
        payload = {
            "_meta": {
                "variant": variant.name,
                "prox_v_schedule": variant.prox_v_schedule,
                "prox_num_steps": variant.prox_num_steps,
                "source": "pre-rewrite code @ baseline 4ca4873",
                "config": "synthetic / SimpleCNN / CPU / 1 epoch / seed 42",
                "note": "prox_v_schedule is a no-op on the AccVRS path",
            },
            "weight_sha256": result["weight_sha256"],
            "metrics": result["metrics"],
        }
        out_path = GOLDEN_DIR / f"{variant.name}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path.relative_to(_REPO_ROOT)}  sha={result['weight_sha256'][:12]}…")


if __name__ == "__main__":
    main()
