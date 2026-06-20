#!/usr/bin/env bash
# distsarah_local — grid hyperparameter tuning.
# Set the GPU via CUDA_VISIBLE_DEVICES; append any Hydra overrides as args.
cd "$(dirname "$0")/.."
python main.py search=grid_distsarah_local runtime.wandb.enabled=true "$@"
