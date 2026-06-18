#!/usr/bin/env bash
# Distributed NoFullGrad SARAH baseline (Medyakov-style: running-mean carry-over
# anchor, plain step, no f1 prox). Communication-matched to NFG-SS.
#
# Hyperparameters: lr (step size) and weight_decay — that's it (no prox knobs).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/distributed_sarah.sh            # single run
#   CUDA_VISIBLE_DEVICES=0 MODE=optuna ./scripts/distributed_sarah.sh
#   CUDA_VISIBLE_DEVICES=0 MODE=grid   ./scripts/distributed_sarah.sh
#
# Env vars:
#   MODE            run | optuna | grid          (default: run)
#   LR              step size for a single run    (default: 0.01)
#   WD              weight decay for a single run  (default: 0.0)
#   SERVER_FRACTION partition.server_fraction      (default: 0.5; null = equal split)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-run}"
LR="${LR:-0.01}"
WD="${WD:-0.0}"
SERVER_FRACTION="${SERVER_FRACTION:-0.5}"

case "$MODE" in
  run)
    python main.py algorithm=distributed_sarah \
      algorithm.lr="${LR}" algorithm.weight_decay="${WD}" \
      partition.server_fraction="${SERVER_FRACTION}" \
      runtime.wandb.enabled=true runtime.wandb.project="distributed-sarah" \
      runtime.wandb.name="dist-lr${LR}-wd${WD}"
    ;;
  optuna)
    # TPE + median pruner over lr (1e-4 … 1.0) and weight_decay (1e-6 … 1e-3).
    python main.py search=optuna_distsarah \
      partition.server_fraction="${SERVER_FRACTION}" \
      runtime.wandb.enabled=true
    ;;
  grid)
    # 8 lr × 4 weight_decay = 32 configs.
    python main.py search=grid_distsarah \
      partition.server_fraction="${SERVER_FRACTION}" \
      runtime.wandb.enabled=true
    ;;
  *)
    echo "unknown MODE='${MODE}' (expected run|optuna|grid)" >&2
    exit 1
    ;;
esac
