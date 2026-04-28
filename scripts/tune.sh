#!/usr/bin/env bash
# Run a list of tuning experiments, one after another.
#
# Each row in CONFIGS below = one experiment: NAME | HYDRA_OVERRIDES.
# Overrides are semicolon-separated.  Common overrides (epochs, wandb)
# are added automatically.  Edit the list to add/remove experiments.
#
# NAME must not contain '=' (Hydra grammar uses '=' as key/value
# separator; we pass NAME via runtime.wandb.name).  Use '-' or '_'.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/tune.sh                 # run every entry
#   CUDA_VISIBLE_DEVICES=7 EPOCHS=50 ./scripts/tune.sh       # all entries, 50 epochs each
#   CUDA_VISIBLE_DEVICES=0 ./scripts/tune.sh theta-1.0 mom-0.5  # only the named entries
#
# Env vars:
#   CUDA_VISIBLE_DEVICES   GPU id(s) — inherited from caller (set externally).
#   EPOCHS                 epochs per run  (default: 30)
#   WANDB_GROUP            W&B group label (default: tune-<timestamp>)
#   PYTHON                 python binary   (default: python)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

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
  # Round 1 — baseline grid (already done):
  #   theta-1.0     mediocre
  #   theta-2.0     worse
  #   theta-0.2     BEST so far (≈0.654)
  #   vlinear       good (combine in round 2)
  # Round 2 — explore around theta=0.2 + combine with vlinear:
  "theta-0.1            | algorithm.theta=0.1  ; algorithm.prox_lr=0.020"
  "theta-0.3            | algorithm.theta=0.3  ; algorithm.prox_lr=0.008"
  "theta-0.2-lr0.020    | algorithm.theta=0.2  ; algorithm.prox_lr=0.020"
  "theta-0.2-numsteps160| algorithm.theta=0.2  ; algorithm.prox_lr=0.012 ; algorithm.prox_num_steps=160"
  "theta-0.2-vlinear    | algorithm.theta=0.2  ; algorithm.prox_lr=0.012 ; algorithm.prox_v_schedule=linear"
  "theta-0.5-vlinear    | algorithm.prox_v_schedule=linear"
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
echo "── CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

for row in "${CONFIGS[@]}"; do
  name="$(trim "${row%%|*}")"
  overrides="${row#*|}"

  want "$name" || continue

  if [[ "$name" == *"="* ]]; then
    echo "✗ $name contains '=' — Hydra rejects that in values, skipping"
    continue
  fi

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
