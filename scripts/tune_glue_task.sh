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

for method in $METHODS; do
  echo "=========================================================="
  echo "  [task=${task}]  Optuna method=${method}"
  echo "=========================================================="
  python main.py \
    +experiment=glue_roberta_lora \
    data.task="${task}" \
    "search=optuna_glue_${method}" \
    "$@"
done
echo "[task=${task}] all methods done"
