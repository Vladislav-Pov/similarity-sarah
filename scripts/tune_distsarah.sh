#!/usr/bin/env bash
# Короткий Optuna-тюн distributed_sarah: 150 эпох, косинус откалиброван на 500
# (lr_t_max=500 в configs/search/optuna_distsarah.yaml). Ищет lr + momentum; wd
# фиксирован. Трайлы НЕ пишутся в W&B (no-op logger) — лучшее смотри из студии:
#   python -c "import optuna; s=optuna.load_study(study_name='dsarah-momtune2', storage='sqlite:///nfg_optuna/dsarah-momtune2.db'); print('best', s.best_value); print(s.best_params)"
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=4 python main.py search=optuna_distsarah search.study_name=dsarah-momtune2
