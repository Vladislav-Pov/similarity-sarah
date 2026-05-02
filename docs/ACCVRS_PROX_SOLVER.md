# AccVRS-style Inexact Prox Solvers

Ported from `AccVRS/resnet_exp/new_alg.py:argminA` into `similarity_sarah/runtime/prox_solver_accvrs.py`. Lets BNFG-SARAH (and any other algorithm using `ProxSolver`) reuse the inner-loop solver that converged faster on ResNet-18/CIFAR in the AccVRS experiments.

## Что они решают (то же что и наш `InexactProxSGD/Adam`)

$$w_{new}\ \approx\ \mathrm{prox}_{\theta f_1}(w_{outer} - \theta v) = \arg\min_w \Bigl\{\langle \theta v, w\rangle + \tfrac{1}{2}\|w - w_{outer}\|^2 + \theta f_1(w)\Bigr\}.$$

## 1) Какие версии оптимизатора у них есть и какие я перенёс

В оригинальном `argminA` (см. `AccVRS/resnet_exp/new_alg.py:138-343`) есть **6 веток** по `optimizer_name`:

| ветка | тип | что делает | перенёс? |
|---|---|---|---|
| **Batch_SGD** | mini-batch | их основной — SGD с Polyak-моментом на серверных батчах + LR-decay + early-stop | ✅ `AccvrsBatchSGDProx` |
| **Batch_Adam** | mini-batch | то же что Batch_SGD, но Adam-апдейт вместо SGD | ✅ `AccvrsBatchAdamProx` |
| **Batch_SGD_lib** | mini-batch | как Batch_SGD, но через `torch.optim.SGD` с переименованием параметров; `lambd` зашит как `momentum` в `torch.optim.SGD` (странно). По их же комментариям не лучше Batch_SGD | ❌ skip — confused semantics |
| **Nesterov** | full-batch | их кастомный Nesterov на полной `A`/`dA` цели | ❌ skip — full-batch требует пройти весь serverNN×шагов; на ResNet/CIFAR неэффективно |
| **SGD** | full-batch | их кастомный full-batch SGD `sgd_alg` | ❌ skip — то же |
| **Adam** | full-batch | их кастомный `adam_minimize_func` | ❌ skip — то же |

В нашем коде живут **`AccvrsBatchSGDProx`** и **`AccvrsBatchAdamProx`** — те две ветки, которые они реально используют в `experiment_p03.py` и `experiment_p05_argmin4.py`.

## 2) Какие фичи у их inner-loop'а

Шаги одного inner step (mini-batch) — точно по их Batch_SGD:

```
loop num_steps × len(server_loader) iterations:
    sample (x, y) from server_loader
    grad = ∇f_1(param; (x, y))                       # стохастический server grad
    d = (param − w_outer) + θ·grad                    # ★ см. примечание про θv ниже

    if momentum > 0 and prev_d is not None:
        d = momentum·prev_d + (1−momentum)·d         # Polyak на direction
    prev_d = d

    if grad_clip > 0:
        d = torch.clamp(d, −grad_clip, grad_clip)    # per-element clip

    param ← param·(1 − wd·lr) − lr·d                  # SGD + multiplicative WD
                                                       # (Adam — то же на m/v)

    every inner_decay_period iters:
        cur_norm = ‖d‖
        lr ×= inner_decay_factor                      # хардкод 0.9 у них
        if cur_norm / start_norm < early_stop_ratio:
            break                                      # хардкод 1e-3 у них
```

Что перенесено пословно:

| фича | в нашей реализации | их оригинал |
|---|---|---|
| Warm-start `z = w_outer − θ·v` | ✅ обязателен | то же |
| Polyak momentum на direction | ✅ `momentum=0.9` рекомендация | `self.momentum=0.9` в их config'ах |
| Per-element clip (`torch.clamp`) | ✅ `grad_clip > 0` | `clip_grads > 0` |
| Multiplicative weight decay `param·(1 − wd·lr)` | ✅ `weight_decay > 0` | `self.argmin_lambd > 0` |
| LR decay `lr ×= 0.9` каждые `period` шагов | ✅ `inner_decay_factor`, `inner_decay_period` | хардкод `0.9` каждые `server_num_batch//5` |
| Early-stop по ratio | ✅ `early_stop_ratio=1e-3` | хардкод `0.001` |
| Auto-LR `γ₀ = 1/(2(1 + θ·L1))` | ✅ если `lr=null` | `gamma = 1/(2L)` |
| `lr_factor` (их `linear_schedule`) | ✅ `lr_factor` | `theta_factor[k]` |

## 3) Что у них в коде есть, а у меня НЕТ (и почему)

### **Главная особенность**: их direction **НЕ** содержит `θ·v`

Это самое тонкое место. Выписываю буквально.

Их `A^t_θ(x) = ⟨θv, x⟩ + (1/2)‖x − w_{outer}‖² + θ·f_1(x)`. Тогда

$$\nabla A(x) = \theta v + (x - w_{outer}) + \theta \nabla f_1(x).$$

Их Batch_SGD считает direction:
```python
d = (param − w_outer) + θ · ∇f_1(param)              # ← НЕТ θ·v !
```

То есть direction = `∇A(x) − θ·v`. Линейный `θ·v`-член **отсутствует** в шаге, он зашит **только в warm-start** `z = w_outer − θ·v`.

Что это значит:
- **Стационарная точка** их SGD: `(w − w_outer) + θ·∇f_1(w) = 0`, т.е. `w = w_outer − θ·∇f_1(w)` = `prox_{θf_1}(w_{outer})` — **не зависит от v**.
- При **большом** `num_steps` итерат уезжает к `prox_{θf_1}(w_{outer})`, и информация про `v` теряется.
- При **малом** `num_steps` (их `argmin_max_iter=4` → ~1200 итераций для них; для нас будет 4×17 = 68) итерат остаётся в окрестности warm-start'а, где `v`-эффект сохранён.

**По умолчанию** я перенёс это поведение буквально (`include_linear_term=False`). У них это работает потому что:
1. Малый `argmin_max_iter` (4 эпохи).
2. Warm-start фактически делает «один большой шаг по `v`», и SGD-итерации лишь немного «полируют» это с помощью `f_1`.

Если хочешь, чтобы SGD сходился к **истинному** argmin'у `A^t_θ` (= `prox_{θf_1}(w_{outer} − θv)`), включи флаг:

```yaml
prox_include_linear_term: true
```

тогда direction становится `(param − w_outer) + θ·∇f_1(param) + θ·v`, что точно равно `∇A(param)`. Это **наша** семантика — то же, что делает `InexactProxSGD`. С большим `num_steps` ratio-метрика будет лучше сходиться, и поведение перестанет зависеть от warm-start'а.

### Их `Batch_SGD_lib` (с `torch.optim.SGD`)

У них там `optimizer = torch.optim.SGD(model.parameters(), lr=gamma, momentum=self.lambd)` — `self.lambd` (weight decay в моей терминологии) подсунут как `momentum` для torch'овой SGD. Это путаница в их коде, не баг алгоритма. На результаты сильно похоже на Batch_SGD, переносить не имеет смысла.

### Полнобатчевые **Nesterov / SGD / Adam**

У них `argminA` для них вычисляет `A(x)` целиком и `dA(x)` на полном датасете → один inner-step требует прохода по всему `server_size` (~15K сэмплов). На ResNet-18 это ~30 секунд **на один inner-step** на GPU. Невыполнимо при `num_steps ≥ 50`. Поэтому пропускаю.

Если очень хочется — Nesterov-вариант на нашем существующем `InexactProxSGD` можно эмулировать через `prox_momentum=0.9, prox_grad_clip=0`, это почти то же самое.

## 4) Конфиг и dispatch

`configs/algorithm/batched_nfg_sarah.yaml` — внизу закомментированный блок:

```yaml
prox_solver: accvrs_batch_sgd     # или accvrs_batch_adam
prox_num_steps: 4                  # ← теперь это *эпохи* серверного лоадера
prox_lr: null                      # null = auto: γ₀ = 1/(2L)·lr_factor
prox_L1: 200.0                     # Lipschitz ∇f_1 (для auto-LR)
prox_lr_factor: 1.0                # внешний множитель на γ₀
prox_momentum: 0.9                 # Polyak на direction
prox_weight_decay: 0.1             # их `argmin_lambd`
prox_grad_clip: 0.0                # per-element clamp; 0 = off
prox_inner_decay_factor: 0.9       # γ ×= это каждые decay_period
prox_inner_decay_period: null      # null ⇒ len(server_loader) // 5
prox_inner_early_stop_ratio: 0.001 # early-stop когда ‖d‖/‖d_first‖ < этого
prox_include_linear_term: false    # см. секцию выше
```

В runner.py добавлены ветки:
- `prox_solver: accvrs_batch_sgd` (или `accvrs_sgd`) → `AccvrsBatchSGDProx`
- `prox_solver: accvrs_batch_adam` (или `accvrs_adam`) → `AccvrsBatchAdamProx`

Существующий dispatch для `sgd` и `adam` (наши `InexactProxSGD/Adam`) не тронут — все старые конфиги работают как раньше.

## 5) Какие версии попробовать первым делом

### Эксперимент A — буквальный AccVRS Batch_SGD

Ровно их рабочий конфиг (`experiment_p05_argmin4.py`):

```yaml
prox_solver: accvrs_batch_sgd
prox_num_steps: 4
prox_lr: null                      # auto
prox_L1: 200.0
prox_lr_factor: 1.0
prox_momentum: 0.9
prox_weight_decay: 0.1
prox_grad_clip: 0.0
prox_include_linear_term: false   # их оригинал
```

Это **должно** работать на их данных. Если на наших данных prox-метрики плохие — попробуй Эксперимент B.

### Эксперимент B — то же, но с истинным argmin (мат-корректная версия)

```yaml
prox_solver: accvrs_batch_sgd
prox_num_steps: 4
prox_include_linear_term: true     # ← вкл!
prox_lr_factor: 1.0
prox_momentum: 0.9
prox_weight_decay: 0.1
```

Это уже эквивалент нашему `InexactProxSGD` с момент=0.9 и weight_decay=0.1, но с их `lr × 0.9 каждые period шагов` и early-stop.

### Эксперимент C — Adam-вариант

```yaml
prox_solver: accvrs_batch_adam
prox_num_steps: 4
prox_lr: 1.0e-3
prox_adam_beta1: 0.9
prox_adam_beta2: 0.999
prox_weight_decay: 0.1
prox_grad_clip: 0.0
```

Adam **с** `lr_factor`, `inner_decay_factor`, `early_stop_ratio` — все как у Batch_SGD, плюс Adam-моменты. Без Polyak момента (Adam свой имеет).

### Эксперимент D — больше `num_steps`

Их `num_steps=4` соответствует `4×len(server_loader)` итерациям. У них `len(server_loader)=300`, итого **1200**. У нас `len(server_loader)≈17` (`server_size=22500/batch=256`), `num_steps=4` даёт всего **68** итераций — мало.

```yaml
prox_solver: accvrs_batch_sgd
prox_num_steps: 50                 # ← 50 эпох server_loader = ~850 inner-step
prox_include_linear_term: true     # с большим num_steps это критично
prox_inner_early_stop_ratio: 1e-4  # пусть бежит дольше до сходимости
prox_lr: null
prox_lr_factor: 1.0
prox_momentum: 0.9
prox_weight_decay: 0.1
```

Это уже похоже на наш текущий `prox_num_steps=80`, только с auto-LR от `L1` и автодекеем `lr×0.9`.

## 6) Особенности и подводные камни

### Семантика `prox_num_steps` поменялась

Для **`sgd` / `adam`** (наши): `prox_num_steps` = total inner-step'ов. Их 80.

Для **`accvrs_batch_sgd` / `accvrs_batch_adam`**: `prox_num_steps` = **число полных проходов** по `server_loader`. Total iterations = `prox_num_steps × len(server_loader)`. С нашим `server_loader` (~17 батчей при batch_size=256, server_size=22500) и `num_steps=4` это всего ~68. Если хочешь столько же реальной работы как при `prox_num_steps=80` у нас — ставь `prox_num_steps=5` (`5×17=85`).

### Auto-LR через `prox_L1`

Их `L1=200` — Lipschitz `∇h_1` на их сетапе CIFAR. У нас сетап другой (другая партиция, augment, batch sizes). Если auto-LR `1/(2(1+θ·L1))` даёт расходимость — увеличить `L1` (получится меньший lr) или явно задать `prox_lr`.

При `θ=0.5` и `L1=200`: `lr_auto = 1/(2·101) ≈ 0.005` — близко к нашему текущему `prox_lr=0.005`. Для других `θ` пересчитывай.

### `prox_grad_clip` — другая семантика

В нашем `InexactProxSGD/Adam` это **глобальный L2-clip** (как `clip_grad_norm_`). В AccVRS-вариантах — **per-element clamp** (как `torch.clamp`). Числовые значения **не сравнимы** между двумя стилями. У них в `experiment_p03/p05` стоит `clip_grads=0` (выкл). Я бы тоже начал с 0 и включал только при наблюдаемых spike'ах в loss'е.

### `momentum` для AccvrsBatchAdam

В Adam собственные моменты β1/β2 — Polyak момент на direction поверх Adam практически бесполезен. Я **принудительно** ставлю `momentum=0` для AccvrsBatchAdam в `runner.py` — даже если в конфиге задано иное. Если очень нужно — править явно.

### `early_stop_ratio`

При `1e-3` (их хардкод) inner-loop часто рано останавливается на наших задачах (потому что у нас меньше `len(server_loader)` чем у них → меньше `decay_period` → быстрее накапливаются проверки). Если видишь `prox_inner_steps_actual << num_steps × len(server_loader)` — попробуй `early_stop_ratio=1e-4` или `1e-5`, чтобы не обрывать преждевременно.

### Логирование

В W&B попадут стандартные ключи (как в наших InexactProx*):
- `prox_grad_norm_first_mean / _last_mean / _ratio_mean` — на фиксированном eval-батче (как наши solver'ы)
- `prox_obj_decrease_mean`
- `prox_clip_frac_mean`

**Плюс** AccVRS-специфичные:
- `prox_inner_norm_first / _last / _ratio` — их «нативная» метрика (норма direction'а в самом начале/конце inner-loop'а, считается на текущем training batch'е, поэтому шумнее)
- `prox_inner_steps` — реальное число inner-итераций (с учётом early-stop)

Сравнивай разные solver'ы по `prox_grad_norm_ratio_mean` и `val/accuracy` — это устойчивые метрики через все 4 семейства (sgd, adam, accvrs_batch_sgd, accvrs_batch_adam).

## 7) Запуск

Никаких новых флагов в `tune.sh` или search-yaml не нужно. Просто переключи `prox_solver` в обычном алгоритм-конфиге:

```bash
# Один прогон
CUDA_VISIBLE_DEVICES=0 python main.py \
    algorithm.prox_solver=accvrs_batch_sgd \
    algorithm.prox_num_steps=4 \
    algorithm.prox_lr=null \
    algorithm.prox_L1=200.0 \
    algorithm.prox_momentum=0.9 \
    algorithm.prox_weight_decay=0.1 \
    runtime.wandb.enabled=true \
    runtime.wandb.name="accvrs-bsgd-baseline"

# Сравнительный sweep между нашим SGD и их Batch_SGD
CUDA_VISIBLE_DEVICES=0 python main.py search=optuna_bnfg \
    "+search.batched_nfg_sarah.prox_solver={type: categorical, choices: [sgd, accvrs_batch_sgd]}" \
    runtime.wandb.enabled=true
```

(второе — пример; чтобы сравнить категориальный choices между ветками solver'а в Optuna).

## TL;DR

| хочу | поставь |
|---|---|
| Их «как у них» Batch_SGD | `prox_solver=accvrs_batch_sgd, prox_num_steps=4, prox_lr=null, prox_momentum=0.9, prox_weight_decay=0.1, prox_include_linear_term=false` |
| Их Batch_SGD, но с честным argmin (мат-корректно) | то же + `prox_include_linear_term=true` |
| Их Batch_Adam | `prox_solver=accvrs_batch_adam, prox_num_steps=4, prox_lr=1e-3, prox_weight_decay=0.1` |
| Глубокий argmin (как наш `prox_num_steps=80`) | `accvrs_batch_sgd, prox_num_steps=50, prox_include_linear_term=true, prox_inner_early_stop_ratio=1e-4` |
