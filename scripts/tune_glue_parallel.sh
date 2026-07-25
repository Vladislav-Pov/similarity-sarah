#!/usr/bin/env bash
# Fan out per-task tuning across GPUs: one background process PER TASK, each
# pinned to its own GPU, each running all methods (separate studies) for that
# task via tune_glue_task.sh.
#
# This realises: "parallel across tasks (separate launches) + algorithms
# cycling within each launch".
#
# Usage:
#   scripts/tune_glue_parallel.sh
#   TASKS="rte mrpc" GPUS="0 1" scripts/tune_glue_parallel.sh
#   TASKS="rte mrpc cola sst2" GPUS="0 1 2 3" scripts/tune_glue_parallel.sh runtime.batch_size=16
#
# Parallelism is SAFE: every (task, method) has a distinct study DB and W&B
# run, so there is no SQLite lock or file contention between processes.
# NOTE: each parallel task needs its OWN GPU — sharing one GPU across tasks
# will OOM.  If TASKS > GPUS, run the extra tasks in a second wave.
set -euo pipefail

read -r -a TASKS <<< "${TASKS:-rte mrpc cola sst2}"
read -r -a GPUS  <<< "${GPUS:-0 1 2 3}"
here="$(cd "$(dirname "$0")" && pwd)"

if (( ${#TASKS[@]} > ${#GPUS[@]} )); then
  echo "WARNING: ${#TASKS[@]} tasks but only ${#GPUS[@]} GPUs — some tasks will"
  echo "         share a GPU and may OOM.  Consider running in waves." >&2
fi

pids=()
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  logf="tune_glue_${task}.log"
  echo "launch task=${task} on GPU ${gpu}  ->  ${logf}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup bash "${here}/tune_glue_task.sh" "${task}" "$@" \
    > "${logf}" 2>&1 &
  pids+=("$!")
done

echo "launched ${#pids[@]} task processes: ${pids[*]}"
wait
echo "all task processes finished"
