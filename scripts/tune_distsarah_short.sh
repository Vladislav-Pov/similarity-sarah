#!/usr/bin/env bash
# distsarah — SHORT tune (150 epochs) as a proxy for the 500-epoch final run.
# Cosine is calibrated to the FINAL horizon (lr_t_max=500) so each trial sees
# the same lr trajectory it would in the real 500-run's first 150 epochs.
# Rank by val accuracy at epoch 150; then run the winner at num_epochs=500.
# Sweeps lr x momentum, weight_decay fixed at 1e-4. Single runs, logged to W&B.
# Set the GPU before running, e.g.  CUDA_VISIBLE_DEVICES=0 ./scripts/tune_distsarah_short.sh
cd "$(dirname "$0")/.."

python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.002 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.9-lr0.002

python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.005 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.9-lr0.005

python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.01 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.9-lr0.01

python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.02 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.9-lr0.02

python main.py algorithm=distributed_sarah algorithm.momentum=0.95 algorithm.lr=0.002 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.95-lr0.002

python main.py algorithm=distributed_sarah algorithm.momentum=0.95 algorithm.lr=0.005 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.95-lr0.005

python main.py algorithm=distributed_sarah algorithm.momentum=0.95 algorithm.lr=0.01 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.95-lr0.01

python main.py algorithm=distributed_sarah algorithm.momentum=0.95 algorithm.lr=0.02 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 algorithm.lr_t_max=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-tune-mom0.95-lr0.02
