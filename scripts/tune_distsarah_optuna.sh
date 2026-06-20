#!/usr/bin/env bash
# distsarah — optuna tuning (cosine lr schedule + train augmentation on node loaders).
# Set the GPU via CUDA_VISIBLE_DEVICES; append any Hydra overrides as args.
cd "$(dirname "$0")/.."
python main.py search=optuna_distsarah data.augment_train=true runtime.wandb.enabled=true "$@"
