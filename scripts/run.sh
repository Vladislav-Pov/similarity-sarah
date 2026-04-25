#!/usr/bin/env bash
# Driver script for similarity-sarah experiments.
#
# Usage:
#   GPU=0 ./scripts/run.sh baseline
#   GPU=1 ./scripts/run.sh stage1-wd
#   GPU=2 ./scripts/run.sh stage1-numsteps
#   GPU=0,1 ./scripts/run.sh grid
#   GPU=3 ./scripts/run.sh optuna
#   GPU=0 ./scripts/run.sh single algorithm.theta=0.02 runtime.wandb.enabled=true
#
# Environment variables:
#   GPU              CUDA device ids (default: 0).  Forwarded to CUDA_VISIBLE_DEVICES.
#   WANDB            "true" to enable wandb on every run (default: true for sweeps,
#                    inherits config for single).
#   WANDB_GROUP      optional W&B group label (default: "<stage>-<timestamp>").
#   EPOCHS           override algorithm.num_epochs.
#   EXTRA            extra Hydra overrides appended to every run (string).

set -euo pipefail

# ── Setup ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# Activate venv if present (adjust path if you keep yours elsewhere).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

PYTHON="${PYTHON:-python}"
WANDB="${WANDB:-true}"
TS="$(date +%Y%m%d-%H%M%S)"

STAGE="${1:-baseline}"
shift || true   # allow extra positional Hydra overrides

GROUP="${WANDB_GROUP:-${STAGE}-${TS}}"

# Common Hydra overrides.
COMMON=(
    "runtime.wandb.enabled=${WANDB}"
    "runtime.wandb.group=${GROUP}"
)
[[ -n "${EPOCHS:-}" ]] && COMMON+=("algorithm.num_epochs=${EPOCHS}")
[[ -n "${EXTRA:-}"  ]] && read -ra EXTRA_ARR <<< "$EXTRA" && COMMON+=("${EXTRA_ARR[@]}")

echo "── stage: $STAGE   GPU=$CUDA_VISIBLE_DEVICES   group=$GROUP"
echo "── repo:  $REPO_ROOT"
echo "── python: $($PYTHON -V 2>&1)"

run() {
    local name="$1"; shift
    echo
    echo "=== $name ==="
    echo "$PYTHON main.py $* runtime.wandb.name=${name}"
    "$PYTHON" main.py "$@" "runtime.wandb.name=${name}"
}

# ── Stages ───────────────────────────────────────────────────────────────
case "$STAGE" in

    # 0. Plain baseline run on the *current* config in configs/algorithm/.
    baseline)
        run "${STAGE}/bnfg-default" "${COMMON[@]}"
        ;;

    # 1.1 Weight-decay ablation (single variable).
    stage1-wd)
        for wd in 0.0 5.0e-4; do
            run "${STAGE}/wd=${wd}" "${COMMON[@]}" \
                "algorithm.prox_weight_decay=${wd}"
        done
        ;;

    # 1.2 prox_num_steps ablation.
    stage1-numsteps)
        for ns in 30 50 100 200; do
            run "${STAGE}/ns=${ns}" "${COMMON[@]}" \
                "algorithm.prox_num_steps=${ns}" \
                "algorithm.prox_weight_decay=0.0"
        done
        ;;

    # 1.3 prox_lr ablation (Adam).
    stage1-lr)
        for lr in 0.01 0.02 0.05 0.1; do
            run "${STAGE}/lr=${lr}" "${COMMON[@]}" \
                "algorithm.prox_lr=${lr}" \
                "algorithm.prox_weight_decay=0.0"
        done
        ;;

    # 1.4 SGD-with-momentum prox solver instead of Adam.
    stage1-sgd)
        for lr in 0.005 0.01 0.02 0.05; do
            run "${STAGE}/sgd-lr=${lr}" "${COMMON[@]}" \
                "algorithm.prox_solver=sgd" \
                "algorithm.prox_lr=${lr}" \
                "algorithm.prox_momentum=0.9" \
                "algorithm.prox_weight_decay=0.0"
        done
        ;;

    # 2. theta sweep (re-run after fixing prox in stage 1).
    stage2-theta)
        for th in 0.01 0.02 0.05 0.1; do
            run "${STAGE}/theta=${th}" "${COMMON[@]}" \
                "algorithm.theta=${th}" \
                "algorithm.prox_weight_decay=0.0"
        done
        ;;

    # 6. SVRS baseline on the same data.
    svrs-baseline)
        run "${STAGE}/svrs" "${COMMON[@]}" \
            "algorithm=svrs" \
            "algorithm.theta=0.02"
        ;;

    # Grid search (per configs/search/grid.yaml; override from CLI freely).
    grid)
        "$PYTHON" main.py "search=grid" "${COMMON[@]}" "$@"
        ;;

    # Optuna search.
    optuna)
        "$PYTHON" main.py "search=optuna" "${COMMON[@]}" "$@"
        ;;

    # Free-form single run with arbitrary Hydra overrides.
    single)
        run "${STAGE}/${TS}" "${COMMON[@]}" "$@"
        ;;

    *)
        echo "Unknown stage: $STAGE"
        echo "Available: baseline | stage1-wd | stage1-numsteps | stage1-lr | stage1-sgd"
        echo "           stage2-theta | svrs-baseline | grid | optuna | single"
        exit 1
        ;;
esac
