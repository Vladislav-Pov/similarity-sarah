#!/usr/bin/env bash
# Run a list of tuning experiments, one after another.
#
# Each row in CONFIGS below = one experiment: NAME | HYDRA_OVERRIDES.
# Overrides are semicolon-separated.  Common overrides (epochs, wandb)
# are added automatically.  Edit the list to add/remove experiments.
#
# Usage:
#   GPU=0 ./scripts/tune.sh                      # run every entry
#   GPU=1 EPOCHS=50 ./scripts/tune.sh            # all entries, 50 epochs each
#   GPU=0 ./scripts/tune.sh theta=1.0 mom=0.5    # only the named entries
#
# Env vars:
#   GPU           CUDA device id  (default: 0)
#   EPOCHS        epochs per run  (default: 30)
#   WANDB_GROUP   W&B group label (default: tune-<timestamp>)
#   PYTHON        python binary   (default: python)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-30}"
GROUP="${WANDB_GROUP:-tune-$(date +%Y%m%d-%H%M%S)}"

# ── Experiments ────────────────────────────────────────────────────────
# Each row: "NAME | hydra.override=... ; another.override=... ; ..."
# Whitespace around '|' and ';' is trimmed.  Anything not overridden
# falls back to configs/algorithm/batched_nfg_sarah.yaml (currently
# theta=0.5, prox_lr=0.005, prox_num_steps=80, prox_momentum=0,
# prox_grad_clip=2.0, prox_eval_batches=8, prox_v_schedule=constant).
CONFIGS=(
  "theta=1.0    | algorithm.theta=1.0  ; algorithm.prox_lr=0.005"
  "theta=2.0    | algorithm.theta=2.0  ; algorithm.prox_lr=0.003"
  "theta=0.2    | algorithm.theta=0.2  ; algorithm.prox_lr=0.012"
  "numsteps=160 | algorithm.prox_num_steps=160"
  "mom=0.5      | algorithm.prox_momentum=0.5"
  "vlinear      | algorithm.prox_v_schedule=linear"
)

# ── Run loop ───────────────────────────────────────────────────────────
SELECTED=("$@")

want() {
  [[ ${#SELECTED[@]} -eq 0 ]] && return 0
  for s in "${SELECTED[@]}"; do [[ "$1" == "$s" ]] && return 0; done
  return 1
}

trim() { echo "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

echo "── tune sweep"
echo "── group:  $GROUP"
echo "── epochs: $EPOCHS"
echo "── GPU:    $CUDA_VISIBLE_DEVICES"

for row in "${CONFIGS[@]}"; do
  name="$(trim "${row%%|*}")"
  overrides="${row#*|}"

  want "$name" || continue

  IFS=';' read -ra raw_args <<< "$overrides"
  args=()
  for a in "${raw_args[@]}"; do args+=("$(trim "$a")"); done

  echo
  echo "▶ $name"
  printf '   %s\n' "${args[@]}"

  "$PYTHON" main.py \
      "${args[@]}" \
      "algorithm.num_epochs=$EPOCHS" \
      "runtime.wandb.enabled=true" \
      "runtime.wandb.group=$GROUP" \
      "runtime.wandb.name=$name" \
    || echo "   ✗ $name failed — continuing"
done

echo
echo "✓ done.  Compare in W&B under group: $GROUP"
