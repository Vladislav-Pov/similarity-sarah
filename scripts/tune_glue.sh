#!/usr/bin/env bash
# Independent per-task Optuna tuning for every method on GLUE.
#
# Runs one SEPARATE Optuna study per (task, method): each writes its own
# resumable DB (optuna_studies/glue-<method>-<task>.db) and logs to W&B, so
# every task gets its own best hyper-parameters (the correct GLUE protocol).
#
# Usage:
#   scripts/tune_glue.sh                      # default tasks × methods
#   TASKS="rte mrpc" METHODS="bnfg" scripts/tune_glue.sh   # subset
#   CUDA_VISIBLE_DEVICES=1 scripts/tune_glue.sh            # pick a GPU
#
# Extra Hydra overrides can be appended, e.g.:
#   scripts/tune_glue.sh runtime.batch_size=16 search.num_trials=50
set -euo pipefail

# Ordered small→large so quick tasks surface results first.
TASKS="${TASKS:-rte mrpc cola sst2}"
METHODS="${METHODS:-bnfg svrs fedavg}"

for task in $TASKS; do
  for method in $METHODS; do
    echo "==================================================================="
    echo "  Optuna: method=${method}  task=${task}"
    echo "==================================================================="
    python main.py \
      +experiment=glue_roberta_lora \
      data.task="${task}" \
      "search=optuna_glue_${method}" \
      "$@"
  done
done
