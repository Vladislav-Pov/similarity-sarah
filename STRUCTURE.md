# Структура проекта (main.py)

**Цель:** обучение ResNet18 на CIFAR-10 оптимизатором SARAH с опциональным momentum (variant 2) и тюнингом гиперпараметров через Optuna.

---

## Запуск
- `python main.py`
- Если `RUN_TUNING=True`: сначала Optuna (N trials), затем при `RUN_FINAL_TRAIN=True` — дообучение с лучшими параметрами.
- Если `RUN_TUNING=False`: один прогон с `DEFAULT_PARAMS`.

---

## Блоки кода (по порядку)

1. **Конфиг** — флаги (USE_COMET, RUN_TUNING, RUN_FINAL_TRAIN), TUNING_TRIALS/EPOCHS, FINAL_EPOCHS, DATA_ROOT, VAL_SPLIT, SEED, DEFAULT_PARAMS (sarah_lr, weight_decay, warmup_epochs, min_lr, label_smoothing, max_grad_norm, batch_size, momentum).

2. **Трансформы** — train: RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(CIFAR-10). Test/val: ToTensor(), Normalize.

3. **Утилиты** — set_seed; maybe_create_experiment (Comet); log_metrics; finalize_params (min_lr_ratio → min_lr).

4. **Модель и данные** — build_model (ResNet18 под 32×32, 10 классов); build_criterions(label_smoothing); build_dataloaders(batch_size, val_split, seed) → train/val; build_full_trainloader; build_testloader.

5. **SARAH + momentum (низкоуровневые)** — zero_grads, clone_grads, apply_weight_decay, clip_grads, apply_update, zeros_like_grads, momentum_step(m_old, v_t, beta) → m_new = beta*m_old + v_t.

6. **SARAH-эпоха** — compute_full_grad(model, loader, loss_sum, decay); train_epoch(...): в начале эпохи полный градиент v_0, s_0=v_0, шаг x_1=x_0-η*v_0; в цикле по батчам v_t = ∇f_i(x_t)-∇f_i(x_{t-1})+v_{t-1}, при momentum s_t=β*s_{t-1}+v_t и шаг по s_t, иначе по v_t; gradient clipping по max_grad_norm.

7. **Оценка и LR** — evaluate(model, loader, criterion); get_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr): warmup линейно, потом cosine decay до min_lr.

8. **Верхний уровень обучения** — train_loop(model, trainloader, eval_loader, criterion, criterion_sum, params, total_epochs, eval_name, experiment, trial): цикл эпох, get_lr → train_epoch → evaluate, логи в Comet, при trial — report(eval_acc) и should_prune().

9. **Optuna** — objective(trial): suggest гиперпараметров (sarah_lr, weight_decay, label_smoothing, warmup_epochs, min_lr_ratio, max_grad_norm, momentum), batch_size=256 фиксирован; finalize_params; build_dataloaders/build_model/criterions; train_loop(..., TUNING_EPOCHS, eval_name="val", trial=trial); return best_val_acc.

10. **Финальный прогон** — train_with_params(params, total_epochs, run_name): полный train loader, test loader, train_loop(..., eval_name="test", без trial).

11. **main()** — set_seed, cudnn.benchmark; при RUN_TUNING: TPESampler + MedianPruner, create_study(maximize), optimize(objective, n_trials), best_params["batch_size"]=256, при RUN_FINAL_TRAIN — train_with_params(best_params, FINAL_EPOCHS); иначе train_with_params(DEFAULT_PARAMS, FINAL_EPOCHS).

---

## Ключевые формулы SARAH + momentum (variant 2)
- Начало эпохи: v_0 = full_grad, s_0 = v_0; первый шаг: x_1 = x_0 - η*v_0.
- Внутри эпохи: v_t = ∇f_i(x_t) - ∇f_i(x_{t-1}) + v_{t-1}; s_t = β*s_{t-1} + v_t; x_{t+1} = x_t - η*s_t.
- Градиенты с weight decay и (опционально) clipping по max_norm.
