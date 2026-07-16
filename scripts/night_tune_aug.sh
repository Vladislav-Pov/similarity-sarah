#!/usr/bin/env bash
# НОЧЬ, файл 2 — свежий Optuna-тюн С аугментацией (прошлые тюны были без неё).
# 150 эпох, косинус на 500 (lr_t_max=500 в configs/search/optuna_distsarah.yaml);
# ищет lr + momentum + wd. Новая студия dsarah-momtune-aug.
# GPU при запуске:  CUDA_VISIBLE_DEVICES=5 ./scripts/night_tune_aug.sh
# Лучшее:
#   python -c "import optuna; s=optuna.load_study(study_name='dsarah-momtune-aug', storage='sqlite:///nfg_optuna/dsarah-momtune-aug.db'); print('best', s.best_value); print(s.best_params)"
cd "$(dirname "$0")/.."
python main.py search=optuna_distsarah search.study_name=dsarah-momtune-aug data.augment_train=true
