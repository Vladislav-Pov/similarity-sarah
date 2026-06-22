#!/usr/bin/env bash
# distsarah — momentum sweep (heavy-ball on v_t). Single runs, logged to W&B
# (project distributed_sarah). 3 momentum x 3 lr, weight_decay=1e-4, 150 epochs,
# cosine lr (default). Runs sequentially. Set the GPU before running, e.g.:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/sweep_distsarah_momentum.sh
# No train augmentation here (isolate the fit); append data.augment_train=true to try it.
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.01 algorithm.weight_decay=1e-4 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-mom0.9-lr0.01

CUDA_VISIBLE_DEVICES=1 python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.005 algorithm.weight_decay=1e-4 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-mom0.9-lr0.005

CUDA_VISIBLE_DEVICES=2 python main.py algorithm=distributed_sarah algorithm.momentum=0.9 algorithm.lr=0.001 algorithm.weight_decay=1e-4 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-mom0.9-lr0.001

CUDA_VISIBLE_DEVICES=3 python main.py algorithm=distributed_sarah algorithm.momentum=0.95 algorithm.lr=0.01 algorithm.weight_decay=1e-4 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-mom0.95-lr0.01

CUDA_VISIBLE_DEVICES=4 python main.py algorithm=distributed_sarah algorithm.momentum=0.95 algorithm.lr=0.005 algorithm.weight_decay=1e-4 algorithm.num_epochs=150 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-mom0.95-lr0.005

