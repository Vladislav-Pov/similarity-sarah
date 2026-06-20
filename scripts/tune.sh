#!/usr/bin/env bash
# Tuning is split into one plain script per method x search backend.
# Set the GPU with CUDA_VISIBLE_DEVICES and run the one you want; append any
# Hydra overrides as args.
#
#   ./scripts/tune_bnfg_optuna.sh             ./scripts/tune_bnfg_grid.sh
#   ./scripts/tune_svrs_optuna.sh             ./scripts/tune_svrs_grid.sh
#   ./scripts/tune_fedavg_optuna.sh           ./scripts/tune_fedavg_grid.sh
#   ./scripts/tune_distsarah_optuna.sh        ./scripts/tune_distsarah_grid.sh
#   ./scripts/tune_distsarah_local_optuna.sh  ./scripts/tune_distsarah_local_grid.sh
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/tune_bnfg_optuna.sh partition.server_fraction=null
echo "Pick a per-method tuning script:"
ls -1 "$(dirname "$0")"/tune_*_optuna.sh "$(dirname "$0")"/tune_*_grid.sh
