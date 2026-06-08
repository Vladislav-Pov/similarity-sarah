# Rewrite Brief — similarity-sarah experiments

Этот файл — постановка задачи для предстоящего rewrite экспериментальной части.
`paper.pdf` — источник истины по алгоритму и теореме; `instructions.md` —
описание текущей реализации; этот файл — что мы хотим **после** rewrite.

## Цель

Переписать или существенно оптимизировать экспериментальную часть
similarity-sarah под submission-quality.

## Критерии готовности

1. **Производительность.** На CIFAR-10 / ResNet-18 / 11 узлов / 1 GPU
   одна эпоха NFG-SS не медленнее, чем сейчас, при битовой
   воспроизводимости. Идеал — быстрее за счёт vectorisation /
   mixed precision / уменьшения python overhead.

2. **Читаемость.** Новый разработчик понимает любую часть алгоритма
   за 5 минут чтения:
   - один файл = одна ответственность;
   - type hints везде;
   - короткие docstrings (numpy-style);
   - никакого dead code, закомментированных альтернатив, debug-флагов;
   - unit-тесты на ключевые инварианты (SARAH-телескоп, running-mean,
     prox не ухудшает Φ).

3. **Расширяемость.** Добавить новый baseline = два файла:
   `algorithms/<name>.py` + `configs/algorithm/<name>.yaml`.
   Никаких правок в runner, utils, metrics-logger.

## Scope (ВКЛЮЧАЕМ)

### Алгоритм
- NFG-SS строго по `paper.pdf §4 Algorithm 1`.

### Прокс-решатели (три варианта, все нужны)
1. **`InexactProxSGD`** — SGD с опциональным Polyak momentum, L2 grad clip.
2. **`InexactProxAdam`** — Adam.
3. **`AccvrsBatchSGD`** — порт из AccVRS-репозитория
   (см. `similarity_sarah/runtime/prox_solver_accvrs.py` в текущей версии).
   **Это самый важный из трёх** — именно его best-known конфиги (см.
   `docs/reference_runs/`). Включает авто-вывод γ₀ через `prox_L1` +
   `prox_lr_factor`, опциональный inner-loop step decay, early-stop по
   ratio ‖∇Φ‖/‖∇Φ₀‖.

### Schedule для v внутри прокса
- `prox_v_schedule`: `constant` И `linear`. Оба нужны — best-known
  конфиги используют каждый.

### Baselines
- **SVRS** (Lin et al.~2023) — с честным full-grad anchor.
- **FedAvg** (McMahan et al.~2017) — communication-matched.

### Эксперимент
- CIFAR-10 + ResNet-18 + 11 узлов + 3 сида.
- WandB logging; reproducibility (deterministic ops, fixed seeds).
- Smoke-test минимально воспроизводит один из reference-конфигов.

## Reference: best-known configurations

В `docs/reference_runs/` лежат три эталонных конфига с прошлой версии.
**Изучи их при rewrite** — они должны быть воспроизводимы из новой
кодовой базы битово.

Общая формула победителя:
- `prox_solver=accvrs_batch_sgd`
- `prox_lr=null` (auto-derived через `prox_L1=200`, `prox_lr_factor=0.1`)
- `prox_num_steps ∈ {4, 5}`
- `prox_momentum=0.9`
- `prox_weight_decay=0.1`
- `prox_grad_clip=0`
- `prox_include_linear_term=false`
- `prox_inner_early_stop_ratio=0.0001`
- `theta=0.2`
- `batch_size_clients=1`

В rewrite создай `configs/algorithm/best.yaml` под эту формулу с явным
выбором между `constant`/`linear` schedule.

## Non-scope (НЕ ВКЛЮЧАЕМ)

- `clip_number_of_clients_with_reshuffle` — ablation, не входит в paper.
- `update_v_tilde_in_the_end` — ablation, не входит в paper.
- `log_deviation` diagnostic — diagnostic only.
- `AccvrsBatchAdamProx` — был, не дал улучшения, выкидываем.
- `FedProx`, `Scaffold`, `distributed_sarah` — не сравниваемся.
- Augmentation на server prox-loader (если не критично для accuracy).

Если эти фичи нужно сохранить — выноси в отдельный `experimental/`
модуль с README «не вошло в submission». В hot-path они жить не должны.

## Допускается

- Web-поиск (WebSearch / WebFetch): как современные работы решают
  inexact prox для distributed SARAH-style методов; чистые reference-
  имплементации distributed VR (FedML, Flower, FedScale); 2025-2026
  обновления для AccVRS-like методов.
- `torch.compile`, mixed precision, foreach-оптимизаторы — если дают
  выигрыш и не ломают reproducibility.
- Замена тяжёлых зависимостей на лёгкие, если осмыслено.

## Запрещается

- Менять формулировку алгоритма (расхождение с `paper.pdf` — fatal).
- Терять reproducibility (тот же seed → тот же результат до bit).
- Запускать обучение >5 минут без явного «ок».
- Pip-install без согласования.
- Git push.
- Удалять или менять `docs/reference_runs/` без согласования —
  это эталон.
