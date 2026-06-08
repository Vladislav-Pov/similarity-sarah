# Instructions для экспериментальной секции NFG-SS

Этот документ описывает экспериментальную часть статьи «Non-Convex Distributed Optimization without Full Client Participation under Second-Order Similarity» (метод **NFG-SS**). Используйте его вместе с PDF статьи: тут — внутренности реализации, гиперпараметры, прокс-решатели и протокол; в PDF — теоретическое обоснование и итоговые цифры.

---

## 1. Алгоритм NFG-SS (псевдокод)

### Обозначения

| Символ | Смысл |
|--------|-------|
| $n$ | общее число узлов: 1 сервер ($f_1$) + $n-1$ клиентов ($f_2,\ldots,f_n$) |
| $f(x) = \tfrac{1}{n}\sum_{i=1}^n f_i(x)$ | глобальная цель |
| $\delta$ | константа second-order similarity, $\lVert\nabla^2 f_i - \nabla^2 f\rVert \le \delta$ |
| $\theta > 0$ | шаг прокс-обновления |
| $b$ | client batch size (число клиентов в одном внутреннем шаге) |
| $S$ | число внешних эпох |
| $B_t^{(s)}$ | подмножество клиентов на шаге $t$ эпохи $s$; $\{B_t^{(s)}\}_{t=1}^{\lceil n/b\rceil}$ — разбиение всех клиентов |
| $v_t^{(s)}$ | SARAH-телескоп — recursive low-variance estimator для $\nabla(f - f_1)(w_t)$ |
| $\tilde v_t^{(s)}$ | running mean — усреднение клиентских градиентов в эпохе, используется как бесплатный анкер для следующей эпохи |

### Algorithm 1: NoFullGrad SARAH Similarity (NFG-SS)

```
Input:  w_0^{(0)} ∈ R^d;   ṽ_1^{(0)} = 0,   v_0^{(0)} = 0
Params: stepsize θ > 0,  batch size b

for s = 0, 1, …, S do
    # 1. Sample disjoint client batches that together cover {2, …, n}
    Sample B_1^{(s)}, …, B_{⌈n/b⌉}^{(s)}

    # 2. Carry-over: previous epoch's running mean becomes anchor
    v_0^{(s)} ← v^{(s)}

    # 3. Initial prox step at w_0^{(s)}
    w_1^{(s)} ← prox_{θ f_1}( w_0^{(s)} − θ · v_0^{(s)} )

    for t = 1, 2, …, ⌈n/b⌉ do
        # (a) Update running mean: weighted average of per-client gradients
        ṽ_{t+1}^{(s)} ← (t-1)/t · ṽ_t^{(s)}
                        + 1/(t b) · Σ_{i ∈ B_t^{(s)}} ∇(f_i − f_1)(w_t^{(s)})

        # (b) SARAH telescope: variance reduction along trajectory
        v_t^{(s)} ← v_{t-1}^{(s)}
                    + 1/b · Σ_{i ∈ B_t^{(s)}} [
                          ∇(f_i − f_1)(w_t^{(s)})
                        − ∇(f_i − f_1)(w_{t-1}^{(s)})
                      ]

        # (c) Inexact prox step on the server's local f_1
        w_{t+1}^{(s)} ← argmin_w { ⟨v_t^{(s)}, w⟩
                                  + (1/(2θ)) ‖w − w_t^{(s)}‖²
                                  + f_1(w) }
    end for

    # 4. Hand off to next epoch — no full gradient!
    w_0^{(s+1)}     ← w_{⌈n/b⌉+1}^{(s)}
    ṽ_1^{(s+1)}    ← 0
    v^{(s+1)}      ← ṽ_{⌈n/b⌉+1}^{(s)}     ← THIS is the key: no fresh ∇f
end for
```

Ключевая особенность — строка после `for`-end: переход к новой эпохе берёт running-mean из предыдущей вместо вычисления полного градиента. Стоимость одного раунда — `2b` векторов вверх (рассылка $w_t, w_{t-1}$) + `2b` векторов вниз (приём двух градиентов), то есть `4b` векторов на внутренний шаг. На эпоху — $4n$ векторов (а не $4n + n$ как у SVRS, который ещё платит $n$ за refresh).

---

## 2. Что мы тюним

### 2.1 Параметры внешнего цикла (НФГ-SS-specific)

В `configs/algorithm/batched_nfg_sarah.yaml`:

| Поле | Назначение | Типичный диапазон |
|------|-----------|-------------------|
| `theta` | шаг прокс-обновления $\theta$ в Algorithm 1 | `[0.1, 0.5]` для CIFAR-10 |
| `batch_size_clients` | $b$ — число клиентов на один внутренний шаг | $1$ (для теории), до $5$ для практики |
| `num_epochs` | $S$ — число внешних эпох | $50$–$200$ |
| `num_clients` | число клиентов $n-1$ (без сервера) | $10$ для основного эксперимента |

### 2.2 Параметры прокс-решателя (inner argmin)

| Поле | Назначение |
|------|-----------|
| `prox_solver` | какой решатель использовать: `sgd`, `adam`, `accvrs_batch_sgd`, `accvrs_batch_adam` |
| `prox_num_steps` | $K$ — число внутренних SGD/Adam-шагов на один прокс-вызов |
| `prox_lr` | learning rate inner-solver |
| `prox_momentum` | Polyak momentum для SGD (или AccVRS-SGD); 0 = vanilla SGD |
| `prox_weight_decay` | L2 weight decay внутри inner solver |
| `prox_grad_clip` | глобальный L2-клиппинг внутреннего направления $\nabla\Phi$; `0` = выключено |
| `prox_eval_batches` | сколько mini-batches усреднять для диагностики `prox_grad_norm_first/last` |
| `prox_v_schedule` | `constant` (всегда $\alpha=1$) или `linear` (decay $1\to 0$ через $K$ шагов) |

Расширенные настройки для AccVRS-вариантов:

| Поле | Назначение |
|------|-----------|
| `prox_L1` | оценка $L$-smoothness $f_1$ для авто-вывода $\gamma_0 = 1/(2L)$ |
| `prox_lr_factor` | множитель для авто-вычисленного $\gamma_0$ |
| `prox_inner_decay_factor`, `prox_inner_decay_period` | step decay внутри AccVRS |
| `prox_inner_early_stop_ratio` | ранний выход при достижении target $\|\nabla\Phi\|$ |
| `prox_include_linear_term` | переключатель: warm-start vs. явное добавление $\theta v$ |

### 2.3 Ablation switches (опциональные эксперименты)

| Флаг | Что делает |
|------|-----------|
| `update_v_tilde_in_the_end` | вместо carry-over $\tilde v^{(s)}$ пересчитать его one-shot оценкой на финальном $w_{\text{final}}$ эпохи (1 minibatch/клиент, $1+M$ дополнительных backprops) |
| `clip_number_of_clients_with_reshuffle` | вместо обхода всех клиентов в эпохе обходить только $K < n-1$ клиентов из персистентной перестановки; $\tilde v$ умножается на $n/K$ в конце эпохи |
| `clip_clients_per_epoch` | $K$ для флага выше (по умолчанию 3) |
| `log_deviation` | каждую эпоху логирует $\lVert v_0^{(s)} - \nabla(f - f_1)(w_0^{(s)})\rVert^2$, сравнивая carry-over с точным анкером (тратит дополнительные $n$ полных проходов — для диагностических runs) |

### 2.4 Параметры runtime/data

В `configs/runtime/default.yaml`:

| Поле | Назначение |
|------|-----------|
| `batch_size` | размер base mini-batch (для inner SGD внутри прокса) |
| `large_batch_size` | размер для low-variance оценок (SARAH telescope, full grad anchor) |
| `batch_size_server_grad` | mini-batch на сервере для $\nabla f_1$ в SARAH-update |
| `batch_size_server_prox` | mini-batch на сервере внутри прокс-решателя |
| `batch_size_data_clients` | mini-batch на клиенте для $\nabla f_i$ |
| `eval_every` | через сколько эпох эвалить на val/test |
| `prox_lr_schedule.kind` | `constant`, `cosine` или `step` — расписание `prox_lr` по эпохам |

---

## 3. Как решаем argmin (внутренний прокс)

### 3.1 Прокс-задача

На каждом внутреннем шаге $t$ нужно решить:
$$
w_{t+1} \;\approx\; \mathrm{prox}_{\theta f_1}\bigl(w_t - \theta v_t\bigr)
\;=\; \arg\min_{w \in \mathbb{R}^d} \Phi(w),
$$
где (две эквивалентные формы — отличаются только константой):
$$
\Phi(w) \;=\; \langle v_t, w\rangle + \tfrac{1}{2\theta}\|w - w_t\|^2 + f_1(w)
\;=\; \tfrac{1}{2\theta}\|w - z\|^2 + f_1(w),
\quad z := w_t - \theta v_t.
$$

Градиент прокс-цели:
$$
\nabla\Phi(w) \;=\; \nabla f_1(w) + v_t + \tfrac{1}{\theta}(w - w_t).
$$

### 3.2 Инициализация и стартовая точка

Inner-solver всегда стартует с $w^{(0)}_{\text{prox}} = w_t$ (текущая внешняя итерация) — это гарантирует, что прокс **не может ухудшить** $\Phi$: на старте $\Phi(w_t) = \langle v_t, w_t\rangle + f_1(w_t)$, и любое уменьшение — это улучшение.

### 3.3 Диагностики (выводятся каждый прокс-вызов)

| Метрика | Смысл |
|---------|-------|
| `prox_grad_norm_first` | $\|\nabla\Phi(w^{(0)}_{\text{prox}})\|$ на фиксированном eval-batch |
| `prox_grad_norm_last` | $\|\nabla\Phi(w^{(K)}_{\text{prox}})\|$ на том же batch |
| `prox_grad_norm_ratio` | `last / max(first, 1e-12)` — ближе к 0 ⇔ прокс хорошо решён |
| `prox_obj_decrease` | $\Phi(w^{(0)}) - \Phi(w^{(K)})$ — должно быть положительно |
| `prox_clip_frac` | доля шагов, где сработал `prox_grad_clip` |

Ratio $\approx 1$ при `prox_v_schedule=linear` нормален — там схема не таргетит каноническую цель, а ползёт между ней и плоским $f_1$-min'ом. В этом режиме надо следить за `prox_obj_decrease` и `val/accuracy`.

### 3.4 Доступные inner-solvers

#### (1) `InexactProxSGD` (default) — рекомендован

В `similarity_sarah/runtime/prox_solver.py:278`.

Делает $K = $ `prox_num_steps` шагов SGD-with-Polyak-momentum по $\nabla\Phi$:
$$
m_{k+1} = \beta\,m_k + \nabla\Phi_{\xi}(w^{(k)}), \qquad
w^{(k+1)} = w^{(k)} - \eta\,m_{k+1},
$$
где $\eta=$ `prox_lr`, $\beta=$ `prox_momentum`. Опционально — weight decay в gradient (стандартный L2), opt. глобальный L2-клиппинг.

**Почему рекомендован:** даёт лучшую финальную accuracy на CIFAR-10 (~0.79), стабилен при $\theta\sim 0.2$, $\eta\sim 0.005$–$0.02$, $K\sim 80$–$120$.

#### (2) `InexactProxAdam`

В `prox_solver.py:402`.

Тот же $\nabla\Phi$, но обновление через Adam:
$$
w^{(k+1)} = w^{(k)} - \eta \cdot \widehat m_{k+1} / (\sqrt{\widehat V_{k+1}} + \varepsilon).
$$
Параметры: `prox_adam_beta1`, `prox_adam_beta2`. На наших задачах Adam учится агрессивнее, но менее стабилен.

#### (3) `AccvrsBatchSGDProx`

В `prox_solver_accvrs.py:296`. Порт inner-solver'а из AccVRS-репозитория (Lin et al.).

Отличия от `InexactProxSGD`:
- `prox_num_steps` означает не «итерации», а «эпохи по server-loader'у» (по умолчанию ≈4).
- `prox_lr` может быть `null` → автоматически $\gamma_0 = (1/(2L))\cdot$`prox_lr_factor`, где $L = 1 + \theta\cdot$`prox_L1`.
- `prox_grad_clip` — element-wise `torch.clamp` (а не L2-norm clip), как в оригинальном AccVRS.
- Опционально — step decay по периодам.

Полезен как sanity-check, что наш метод не зависит от конкретного inner-solver'а.

#### (4) `AccvrsBatchAdamProx`

В `prox_solver_accvrs.py:317`. Тот же AccVRS-фреймворк, но Adam-обновление вместо SGD.

### 3.5 Схема выбора решателя

| Сценарий | Что брать |
|----------|----------|
| Baseline / main run | `prox_solver=sgd` + наши best HPs (см. §6) |
| Sanity check (метод не зависит от solver'a) | `prox_solver=accvrs_batch_sgd` |
| Большие модели / нестабильное обучение | `prox_solver=adam` |
| Когда $f_1$ почти выпуклая | `prox_solver=accvrs_batch_sgd` с авто-$\gamma_0$ |

---

## 4. Сравниваемые методы (baseline'ы)

### 4.1 NFG-SS (наш) — `batched_nfg_sarah`

См. §1. Без full gradient, similarity-aware, non-convex.

### 4.2 SVRS — `svrs`

В `similarity_sarah/algorithms/svrs.py`. Алгоритм 1 из Lin et al.~2023.

**Принципиальное отличие:** в начале каждой эпохи вычисляется **точный** anchor
$$g_{\mathrm{ref}} = \tfrac{1}{n}\sum_i \nabla(f_i - f_1)(w_{\mathrm{ref}})$$
через `compute_full_gradient` по всему датасету каждого клиента. Внутри эпохи — random sampling одного клиента ($T \sim \mathrm{Geom}(1/n)$ шагов в среднем).

**Communication cost:** на каждый refresh — $\mathcal{O}(n)$ векторов; внутри эпохи — $4$ векторов на шаг (как у NFG-SS).

Стартовая позиция в нашем сравнении — **identical setup**: тот же inner solver, тот же partition, тот же $\theta$.

### 4.3 FedAvg — `fedavg`

В `similarity_sarah/algorithms/fedavg.py`. Стандартный FedAvg McMahan et al.~2017.

**Communication-matched:** число transmitted vectors за обучение приравнивается к NFG-SS. Это **намеренно щедрый baseline** — мы отказываемся от стандартного full-aggregation schedule в его пользу, чтобы изолировать вклад similarity-aware variance reduction.

### 4.4 Distributed SARAH — `distributed_sarah`

В `similarity_sarah/algorithms/distributed_sarah.py`. Полнее как «sanity baseline» — distributed-вариант SARAH с честным full-grad refresh в начале эпохи. Не цитируется в paper'е, но удобно держать под рукой для ablation'ов.

---

## 5. Метрики и протокол эксперимента

### 5.1 X-axis: cost measure

**Число transmitted vectors** между сервером и клиентами (сумма обоих направлений). Per-epoch budget:

| Метод | Vectors / epoch |
|-------|-----------------|
| NFG-SS | $4(n-1)$ — для $n-1=10$ клиентов: 40 (4 вектора на inner step, K=10 inner steps) |
| SVRS | $4(n-1)$ inner + $2(n-1)$ anchor refresh = $\sim$60 при честной формулировке |
| FedAvg | $2(n-1)$ — для 10 клиентов: 20 |

В paper'е мы выровняли бюджет 2000 transmitted vectors = 50 эпох NFG-SS = 50 эпох SVRS (без anchor) = 100 эпох FedAvg.

### 5.2 Y-axis

- **test/accuracy** — основная метрика (для CIFAR-10 на validation split out-of-fold).
- Дополнительно логируется test/loss, train/loss, $\|v_t\|$, $\|\tilde v_t\|$ — см. namespace `<algo>/{train,val,test}/<metric>` в W&B.

### 5.3 Seeds и доверительные ленты

- **3 random seeds** на метод (`seed=0,1,2` в `configs/config.yaml`).
- Кривая = median по сидам.
- Лента = min/max envelope (это **не** cross-seed CI, а observed range; в caption явно так и пишется).
- Альтернатива — `BAND="std"` (mean ± 1·std), но при $n=3$ std-оценка шумная — рекомендую min/max.

### 5.4 Tuning protocol

- **TPE search** через Optuna (categorical choices).
- 100 trials на метод.
- Pruner: median, `n_startup_trials=10`, `n_warmup_epochs=20`, `interval_epochs=1`.
- Сетка для NFG-SS (CIFAR-10 / ResNet-18, $b=1$):

| Параметр | Choices |
|----------|---------|
| `theta` | `[0.1, 0.2, 0.3, 0.5]` |
| `prox_lr` | `[0.0025, 0.005, 0.007, 0.012, 0.014]` |
| `prox_num_steps` | `[80, 120]` |
| `prox_v_schedule` | `[constant, linear]` |

Скоринг — best `val/accuracy`. Лучший trial эвалится на test split.

---

## 6. Setup эксперимента (основной run)

| Параметр | Значение |
|----------|---------|
| Датасет | CIFAR-10 |
| Модель | ResNet-18 (adapted to 32×32 input) |
| Loss | Cross-entropy |
| Узлы | $n=11$: 1 сервер + 10 клиентов |
| Partition | uniform random |
| Server fraction | $1/11$ |
| Val fraction (из train) | 5% |
| Optimizer (inner) | SGD с Polyak momentum |
| Best theta | 0.19 |
| Best prox_lr | 0.0073 |
| Best prox_num_steps | 86 |
| Best prox_momentum | 0.0 |
| Best prox_grad_clip | 2.0 |
| Best prox_eval_batches | 4 |
| Best prox_v_schedule | constant |
| epoch budget | 50 для NFG-SS/SVRS; 100 для FedAvg |
| seeds | 0, 1, 2 |

(Цифры — после Optuna search; см. `configs/algorithm/batched_nfg_sarah.yaml`.)

---

## 7. Reproducibility

### 7.1 Запустить основной NFG-SS run

```bash
python main.py algorithm=batched_nfg_sarah seed=0
```

### 7.2 Запустить hyperparameter search (Optuna)

```bash
python main.py search=optuna_bnfg
```

Конфиг поиска: `configs/search/optuna_bnfg.yaml`.

### 7.3 Запустить ablation с clipping

```bash
python main.py search=optuna_bnfg_clip
```

См. `configs/search/optuna_bnfg_clip.yaml` — там `clip_number_of_clients_with_reshuffle=true`, `clip_clients_per_epoch ∈ {2, 3}`.

### 7.4 Запустить baseline'ы

```bash
python main.py algorithm=svrs    seed=0
python main.py algorithm=fedavg  seed=0
```

### 7.5 Построить сравнительный график

См. `scripts/plot_methods_comparison.py` — задаёшь W&B URLs в `RUNS` (по 3 сида на метод) и получаешь test accuracy vs transmitted vectors с min/max лентой.

Структура `RunSpec`:
```python
RunSpec(label, url, algo_namespace, vectors_per_epoch, max_epoch, color, marker)
```

`max_epoch` клипает run на нужном бюджете; `vectors_per_epoch` — для оси X.

---

## 8. Ablation studies (опциональные, могут пойти в Appendix)

1. **`clip_number_of_clients_with_reshuffle`** — проверка робастности: что если на эпоху обходить не все клиенты, а 2–3? Carry-over $\tilde v$ масштабируется на $n/K$. Цель — показать, что метод не ломается при ещё более агрессивном partial participation.

2. **`update_v_tilde_in_the_end`** — сравнить carry-over running-mean (наш дефолт) с one-shot оценкой на $w_{\mathrm{final}}$. Дополнительная стоимость — $1+M$ batch-gradients. Цель — показать, что recursive running-mean не хуже более «честной» one-shot оценки.

3. **`log_deviation` диагностика** — измерить $\|v_0^{(s)} - \nabla(f - f_1)(w_0^{(s)})\|^2$ в течение обучения. Цель — показать, что bias оценщика остаётся ограниченным (соответствует Лемме 2 в paper'е).

4. **Чувствительность к $\theta$** — sweep по `[0.05, 0.1, 0.2, 0.3, 0.5, 1.0]` при фиксированных остальных HPs.

5. **Чувствительность к $b$** — `batch_size_clients ∈ {1, 2, 5}`; теория покрывает $b=1$, но эмпирически $b>1$ может ускорять.

---

## 9. Что писать в caption / тексте экспериментов

### Что обязательно упомянуть

- $n$, partition, модель.
- Identical setup для NFG-SS и SVRS (тот же inner solver, тот же $\theta$, тот же seed).
- Каким образом SVRS считает full gradient (через `compute_full_gradient` по всему локальному датасету) и что эти $\mathcal{O}(n)$ векторов **включены** в его положение по оси X.
- FedAvg в communication-matched режиме.
- 3 seeds, median + min/max.

### Чего избегать

- Не называть min/max envelope «confidence interval» — это не CI, а observed range.
- Не выпрашивать $\delta$ из числа: его нельзя замерить экспериментально на ResNet-18, формула $\delta = \mathcal{O}(L/\sqrt m)$ — только мотивация.
- Не сравнивать с SILVER/SPAM эмпирически — у них другие задачи (federated/cross-device); они закрывают theoretical gap, а не practical.

---

## 10. Полезные ссылки в репозитории

- Алгоритм: `similarity_sarah/algorithms/batched_nfg_sarah.py`
- Прокс-решатели: `similarity_sarah/runtime/prox_solver.py`, `prox_solver_accvrs.py`
- Baseline: `similarity_sarah/algorithms/svrs.py`, `fedavg.py`
- Конфиги: `configs/algorithm/`, `configs/search/`
- График: `scripts/plot_methods_comparison.py`
- Точная формулировка прокс-задачи и диагностик: docstring `prox_solver.py:1-60`
- AccVRS-портированные solvers: `docs/ACCVRS_PROX_SOLVER.md`

---

## 11. Frequently overlooked details (для самопроверки)

- **Один и тот же minibatch** используется при вычислении $\nabla f_i(w_t)$ и $\nabla f_i(w_{t-1})$ — иначе SARAH-телескоп вырождается в SGD без variance reduction. Реализация: `srv_xy = next(iter(...))`, `cli_xys = [next(iter(...)) for cid in batch]` — потом передаются как `xy=` в `compute_batch_gradient`.
- **Серверная аугментация** разрешена ТОЛЬКО для `server_prox_loader` (используется внутри прокса). `server_grad_loader` должен быть детерминистичным, иначе SARAH-разность зашумится.
- **`batch_size_clients` в SVRS** игнорируется — Algorithm 1 у Lin et al. семплирует ровно одного клиента на шаг, мы форсируем $B=1$ и предупреждаем в лог.
- **`update_v_tilde_in_the_end=true`** и **`clip_number_of_clients_with_reshuffle=true`** — взаимоисключающие в дефолтной логике: при первом включенном втором флаг скейлинга $n/K$ не применяется.
- **`log_deviation=true`** удваивает время эпохи (полный градиент по всем клиентам + сервер). Использовать только для диагностических runs, отключать в основных.

---

## 12. Известные ограничения (для секции Limitations)

1. **Финальная сложность $\mathcal{O}(n^2\delta^2/(b\varepsilon^2))$** во втором слагаемом — нетривиальная плата за carry-over оценщик. При $b=n^\beta$ с $\beta>0$ второе слагаемое подавляется, но это требует $b>1$.
2. **Полное покрытие клиентов** в эпохе требуется для bias-bound (`$\tilde v_{\lceil n/b\rceil+1}$ как «почти» среднее $\nabla f_i$). Эксперимент с `clip_number_of_clients_with_reshuffle` показывает практическую устойчивость, но теории под него ещё нет.
3. **Стохастический $\nabla f_1$** на сервере (через `prox_solver`) — анализ предполагает доступ к точному $\nabla f_1$ внутри прокса. На практике inexact prox работает, но строго это lemma 1 в paper'е не покрывает.
4. **CIFAR-10 — не federated benchmark.** Для NeurIPS-уровня нужен дополнительный эксперимент на FEMNIST или Shakespeare с реалистичным non-i.i.d. partition.
