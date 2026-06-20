#!/usr/bin/env bash
# svrs — optuna hyperparameter tuning.
# Set the GPU via CUDA_VISIBLE_DEVICES; append any Hydra overrides as args.
cd "$(dirname "$0")/.."
python main.py search=optuna_svrs runtime.wandb.enabled=true "$@"
