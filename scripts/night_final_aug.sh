#!/usr/bin/env bash
# НОЧЬ, файл 1 — payoff: лучшие найденные конфиги на 500 эпох С аугментацией.
# Аугментация в диагностике помогла (модель переобучалась), поэтому здесь augment_train=true.
# GPU задаётся при запуске:  CUDA_VISIBLE_DEVICES=6 ./scripts/night_final_aug.sh
cd "$(dirname "$0")/.."

# A: dsarah-momtune2 best  (lr=0.00232, mom=0.95, wd=8.7e-6)
python main.py algorithm=distributed_sarah algorithm.lr=0.002320098264131071 algorithm.momentum=0.95 algorithm.weight_decay=8.669724543579625e-06 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500aug-momtune2best

# B: dsarah-momtune / t29  (lr=0.00343, mom=0.95, wd=3.65e-4)
python main.py algorithm=distributed_sarah algorithm.lr=0.003428951768374952 algorithm.momentum=0.95 algorithm.weight_decay=0.00036516279017854075 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500aug-t29
