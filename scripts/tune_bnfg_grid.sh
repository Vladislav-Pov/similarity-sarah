#!/usr/bin/env bash
# bnfg — grid hyperparameter tuning.
# Set the GPU via CUDA_VISIBLE_DEVICES; append any Hydra overrides as args.
cd "$(dirname "$0")/.."
python main.py search=grid_bnfg runtime.wandb.enabled=true "$@"
