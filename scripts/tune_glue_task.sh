#!/usr/bin/env bash
# Tune ALL methods for ONE GLUE task inside a SINGLE process.
#
# Each method is its OWN independent Optuna study (own DB, own best params);
# they run one after another here — this is the "algorithms cycle within one
# launch" unit.  Launch several of these in parallel across tasks, one per GPU
# (see tune_glue_parallel.sh or the CUDA_VISIBLE_DEVICES examples below).
#
# Usage:
#   scripts/tune_glue_task.sh rte
#   METHODS="bnfg svrs" scripts/tune_glue_task.sh mrpc runtime.batch_size=16
#   CUDA_VISIBLE_DEVICES=0 scripts/tune_glue_task.sh rte     # pin to one GPU
set -euo pipefail

task="${1:?usage: tune_glue_task.sh <task> [extra hydra overrides...]}"
shift || true
METHODS="${METHODS:-bnfg svrs fedavg}"

# W&B on by default.  Optionally set the project name via PROJECT=<name>.
# During a SEARCH the effective project is `search.wandb_project` (not
# `runtime.wandb.project`), so we set both to cover search and plain runs.
baked=(runtime.wandb.enabled=true)
if [[ -n "${PROJECT:-}" ]]; then
  baked+=("search.wandb_project=${PROJECT}" "runtime.wandb.project=${PROJECT}")
fi

for method in $METHODS; do
  echo "=========================================================="
  echo "  [task=${task}]  Optuna method=${method}"
  echo "=========================================================="
  # `baked` first, then "$@" — extra CLI overrides win (last occurrence).
  python main.py \
    +experiment=glue_roberta_lora \
    data.task="${task}" \
    "search=optuna_glue_${method}" \
    "${baked[@]}" \
    "$@"
done
echo "[task=${task}] all methods done"
