#!/usr/bin/env bash
# distsarah — Optuna tuning over server_fraction (null / 0.2 / 0.35 / 0.5).
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 python main.py search=grid_distsarah data.augment_train=true runtime.wandb.enabled=true partition.server_fraction=null search.study_name=distsarah-gridsf-null

CUDA_VISIBLE_DEVICES=1 python main.py search=grid_distsarah data.augment_train=true runtime.wandb.enabled=true partition.server_fraction=0.2 search.study_name=distsarah-gridsf-0.2

CUDA_VISIBLE_DEVICES=2 python main.py search=grid_distsarah data.augment_train=true runtime.wandb.enabled=true partition.server_fraction=0.35 search.study_name=distsarah-gridsf-0.35

CUDA_VISIBLE_DEVICES=3 python main.py search=grid_distsarah data.augment_train=true runtime.wandb.enabled=true partition.server_fraction=0.5 search.study_name=distsarah-gridsf-0.5
