# Reference runs — лучшие конфигурации NFG-SS

Тут лежат три эталонных снимка best-known конфигов с грид-поисков начала
мая 2026. Все три — с одного коммита `aef4317`, на CIFAR-10 / ResNet-18,
с 11 узлами (1 сервер + 10 клиентов) и `batch_size_clients=1`.

При rewrite экспериментальной части их надо **изучить** и убедиться,
что новая реализация умеет воспроизвести их побитово (с тем же seed).

## Общие наблюдения (что выиграло на гридах)

| Гиперпараметр | Best value | Комментарий |
|---|---|---|
| `prox_solver` | `accvrs_batch_sgd` | AccVRS-стиль обходит обычный `InexactProxSGD` и `InexactProxAdam` |
| `prox_lr` | `null` (auto) | Автодеривация: γ₀ = (1/(2L))·`prox_lr_factor` с L = 1 + θ·`prox_L1` |
| `prox_L1` | `200` | Оценка L-smoothness серверной части f₁ |
| `prox_lr_factor` | `0.1` | Множитель к auto-γ₀ |
| `prox_num_steps` | `4` или `5` | Малое количество inner-шагов AccVRS-style |
| `prox_momentum` | `0.9` | Polyak momentum в inner-SGD; победил среди {0.5, 0.9, 0.95} |
| `prox_weight_decay` | `0.1` | Победил среди {0.01, 0.05, 0.1} |
| `prox_grad_clip` | `0` | Клиппинг выключен — на AccVRS-пути это не нужно |
| `prox_v_schedule` | `constant` или `linear` | Оба работают; linear даёт чуть ниже final accuracy, но ровнее сходимость |
| `prox_include_linear_term` | `false` | Канонический AccVRS — линейный член ⟨θv,·⟩ кодируется только через warm-start z = w − θv |
| `prox_inner_early_stop_ratio` | `0.0001` | Ранний выход inner-loop при ‖∇Φ‖/‖∇Φ₀‖ < этого |
| `theta` | `0.2` | Внешний прокс-шаг |
| `prox_eval_batches` | `4` | Для диагностики prox_grad_norm |

## Файлы

| Файл | Что отличается |
|---|---|
| `best_run_constant_4steps.json` | `prox_v_schedule=constant`, `prox_num_steps=4` |
| `best_run_linear_4steps.json` | `prox_v_schedule=linear`,   `prox_num_steps=4` |
| `best_run_linear_5steps.json` | `prox_v_schedule=linear`,   `prox_num_steps=5` |

## Что это значит для rewrite

1. **AccVRS-style inner-solver обязателен.** Старый репозиторий держит
   его в `similarity_sarah/runtime/prox_solver_accvrs.py`. При rewrite
   его нужно перенести с минимальными правками — он критичен для качества.
2. **Авто-вывод γ₀ (через `prox_lr=null` + `prox_L1` + `prox_lr_factor`)
   надо сохранить** — это main feature AccVRS-пути.
3. **`prox_v_schedule="linear"` входит в scope**, в отличие от того, что
   я раньше предлагал отбросить — потому что best-run-конфиги его
   используют.
4. **Smoke-test первого milestone'а** должен проверить ровно эту
   конфигурацию (одна из трёх выше) и получить такой же выход на 1 эпохе,
   что и до rewrite.
