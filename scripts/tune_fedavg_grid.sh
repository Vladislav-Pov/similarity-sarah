#!/usr/bin/env bash
# fedavg — grid hyperparameter tuning.
# Set the GPU via CUDA_VISIBLE_DEVICES; append any Hydra overrides as args.
cd "$(dirname "$0")/.."
python main.py search=grid_fedavg runtime.wandb.enabled=true "$@"
