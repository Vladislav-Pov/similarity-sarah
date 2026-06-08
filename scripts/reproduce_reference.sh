#!/usr/bin/env bash
# Reproduce the best-known NFG-SS reference runs (CIFAR-10 / ResNet-18 / 11 nodes).
#
# configs/algorithm/best.yaml encodes the winning AccVRS formula. The three
# docs/reference_runs/*.json snapshots differ ONLY in prox_num_steps (4 vs 5) —
# prox_v_schedule is a no-op on the AccVRS path — so this sweeps num_steps in
# {4, 5} across the requested seeds.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/reproduce_reference.sh
#   CUDA_VISIBLE_DEVICES=0 SEEDS="0 1 2" WANDB=true ./scripts/reproduce_reference.sh
#
# Env vars:
#   SEEDS   space-separated seeds            (default: "0 1 2")
#   WANDB   "true" to log to W&B             (default: false)
#   EPOCHS  override algorithm.num_epochs    (default: best.yaml = 50)
#   PYTHON  python binary                    (default: python)
#
# Bit-reproducibility vs the pre-rewrite code: run this on baseline commit
# (git checkout 4ca4873) and on this branch with the same seed, then diff the
# final metrics. The fast synthetic bit-repro is already in `pytest`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
WANDB="${WANDB:-false}"

COMMON=("algorithm=best" "runtime.wandb.enabled=${WANDB}")
[[ -n "${EPOCHS:-}" ]] && COMMON+=("algorithm.num_epochs=${EPOCHS}")

echo "── reproduce reference NFG-SS runs"
echo "── seeds: ${SEEDS}   wandb: ${WANDB}   GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "── python: $("$PYTHON" -V 2>&1)"

for seed in ${SEEDS}; do
  for ns in 4 5; do
    name="nfg_ss-ns${ns}-seed${seed}"
    echo
    echo "=== ${name} (prox_num_steps=${ns}, seed=${seed}) ==="
    "$PYTHON" main.py "${COMMON[@]}" \
      "algorithm.prox_num_steps=${ns}" \
      "seed=${seed}" \
      "runtime.wandb.name=${name}"
  done
done

echo
echo "✓ done."
