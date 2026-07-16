#!/usr/bin/env bash

cd "$(dirname "$0")/.."

# CUDA_VISIBLE_DEVICES=1 python main.py algorithm=distributed_sarah algorithm.lr=0.0030285251536058856 algorithm.momentum=0.9 algorithm.weight_decay=1.8662266976517965e-05 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t0

# CUDA_VISIBLE_DEVICES=1 python main.py algorithm=distributed_sarah algorithm.lr=0.0022978642024960873 algorithm.momentum=0.9 algorithm.weight_decay=0.0002019296662831928 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t16

# CUDA_VISIBLE_DEVICES=1 python main.py algorithm=distributed_sarah algorithm.lr=0.0017211901064667712 algorithm.momentum=0.9 algorithm.weight_decay=6.765080011368816e-05 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t12



#####

# CUDA_VISIBLE_DEVICES=4 python main.py algorithm=distributed_sarah algorithm.lr=0.003428951768374952 algorithm.momentum=0.95 algorithm.weight_decay=0.00036516279017854075 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t29

# CUDA_VISIBLE_DEVICES=4 python main.py algorithm=distributed_sarah algorithm.lr=0.004830576584733714 algorithm.momentum=0.95 algorithm.weight_decay=0.00013829366234897962 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t39

# CUDA_VISIBLE_DEVICES=4 python main.py algorithm=distributed_sarah algorithm.lr=0.00461393415567276 algorithm.momentum=0.95 algorithm.weight_decay=0.00028297538477968466 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t41


##### NEW (train-loss теперь логируется): диагностика t29 — augmentation A/B, 500 эпох, GPU 4
##### смотри <algo>/train/loss vs <algo>/val/loss: если train сильно ниже val -> оверфит (аугментация поможет); если train≈val -> недофит (мало апдейтов).
CUDA_VISIBLE_DEVICES=2 python main.py algorithm=distributed_sarah algorithm.lr=0.003428951768374952 algorithm.momentum=0.95 algorithm.weight_decay=0.00036516279017854075 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-t29-noaug

CUDA_VISIBLE_DEVICES=2 python main.py algorithm=distributed_sarah algorithm.lr=0.003428951768374952 algorithm.momentum=0.95 algorithm.weight_decay=0.00036516279017854075 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-t29-aug
