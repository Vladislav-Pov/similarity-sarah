#!/usr/bin/env bash
# НОЧЬ, файл 3 — свип weight_decay на 500 эпох С аугментацией (фикс lr/mom, варьируем wd).
# Прошлый no-aug тюн дал крошечный wd (8.7e-6); с аугментацией оптимум wd может сдвинуться —
# проверяем 1e-4 / 5e-4 / 1e-3. lr/mom взяты из лучшего конфига.
# GPU при запуске:  CUDA_VISIBLE_DEVICES=2 ./scripts/night_wd_aug.sh
cd "$(dirname "$0")/.."

python main.py algorithm=distributed_sarah algorithm.lr=0.002320098264131071 algorithm.momentum=0.95 algorithm.weight_decay=1e-4 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500aug-wd1e-4

python main.py algorithm=distributed_sarah algorithm.lr=0.002320098264131071 algorithm.momentum=0.95 algorithm.weight_decay=5e-4 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500aug-wd5e-4

python main.py algorithm=distributed_sarah algorithm.lr=0.002320098264131071 algorithm.momentum=0.95 algorithm.weight_decay=1e-3 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500aug-wd1e-3
