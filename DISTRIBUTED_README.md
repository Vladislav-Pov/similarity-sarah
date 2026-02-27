# Distributed SARAH Implementation

## Файлы

1. **`distributed_sarah.py`** — реализация Client-Server архитектуры для SARAH
2. **`example_distributed.py`** — примеры использования и интеграция в main.py

---

## Архитектура

### Client (K клиентов)
- Каждый клиент держит свой **непересекающийся** subset данных (DataLoader)
- Методы:
  - `sample_batch()` — сэмплирует батч из локальных данных
  - `compute_grad(model, batch)` — вычисляет градиент по батчу для данной модели
  - `compute_full_grad(model)` — вычисляет полный градиент по всем локальным данным
- **Клиенты НЕ обновляют веса** — только вычисляют градиенты

### Server (1 сервер)
- Держит глобальную модель и SARAH-состояние: `v_t` (рекурсивный градиент) и `s_t` (momentum)
- Методы:
  - `initialize_sarah(full_grad_avg, lr)` — инициализация в начале эпохи: v_0, s_0, первый шаг
  - `apply_sarah_step(grad_xt, grad_xprev, lr, beta)` — применяет SARAH+momentum шаг:
    ```
    v_t = grad_xt - grad_xprev + v_{t-1}
    s_t = β*s_{t-1} + v_t
    x_{t+1} = x_t - η*s_t
    ```
  - Применяет **weight decay** и **gradient clipping** на стороне сервера
- Хранит `model_prev` для вычисления `grad_xprev`

---

## Протокол обучения (одна эпоха)

1. **Разбиение данных** — split train dataset на K непересекающихся подмножеств, создать K клиентов

2. **Начало эпохи — distributed full gradient:**
   - Каждый клиент вычисляет локальный полный градиент
   - Сервер усредняет: `v_0 = (1/K) * sum_i grad_i` (с учётом числа сэмплов)
   - Инициализация: `s_0 = v_0` (momentum variant 2)
   - Первый шаг: `x_1 = x_0 - η*v_0`

3. **Случайная перестановка клиентов:** `π_1, ..., π_K`

4. **Итерации по клиентам (t=1..K):**
   - Клиент `π_t` сэмплирует батч
   - Вычисляет:
     - `grad_xt` = градиент по батчу в текущей модели `x_t`
     - `grad_xprev` = градиент по тому же батчу в предыдущей модели `x_{t-1}`
   - Отправляет оба градиента на сервер
   - Сервер обновляет:
     ```
     v_t = grad_xt - grad_xprev + v_{t-1}
     s_t = β*s_{t-1} + v_t
     x_{t+1} = x_t - η*s_t
     ```
   - Сохраняет текущую модель как предыдущую для следующей итерации

5. **Клиенты никогда не обновляют веса** — только сервер применяет обновления

---

## Использование

### Вариант 1: Замена в train_loop (main.py)

```python
from distributed_sarah import create_clients, train_epoch_distributed, Server

# Вместо train_epoch используем train_epoch_distributed
server = Server(model, device, weight_decay, max_grad_norm)

clients = create_clients(
    dataset=train_dataset,
    num_clients=10,
    batch_size=params["batch_size"],
    device=device
)

train_loss, train_acc = train_epoch_distributed(
    clients=clients,
    server=server,
    criterion=criterion,
    criterion_sum=criterion_sum,
    lr=lr,
    momentum=params.get("momentum", 0.0)
)
```

### Вариант 2: Использовать train_loop_distributed

```python
from distributed_sarah import train_loop_distributed

best_acc = train_loop_distributed(
    train_dataset=train_dataset,
    eval_loader=testloader,
    model=model,
    criterion=criterion,
    criterion_sum=criterion_sum,
    params=DEFAULT_PARAMS,
    total_epochs=200,
    num_clients=10,
    device=device,
    eval_name="test"
)
```

### Запуск примеров

```bash
python example_distributed.py
```

---

## Отличия от централизованной версии

| Централизованная | Распределённая |
|------------------|----------------|
| Один DataLoader на весь train | K DataLoader'ов (по одному на клиента) |
| Полный градиент = один проход по всем данным | Полный градиент = агрегация K локальных полных градиентов |
| Каждый шаг = один батч из общего DataLoader | Каждый шаг = один батч от случайного клиента |
| model, model_prev в train_epoch | model, model_prev в Server |
| Обновление весов внутри train_epoch | Обновление весов только на сервере |

---

## Параметры

- **K (num_clients)** — число клиентов (например, 10)
- Остальные гиперпараметры (sarah_lr, weight_decay, momentum, max_grad_norm, batch_size) — те же, что в централизованной версии

---

## Проверка корректности

Распределённая версия должна давать **близкие результаты** к централизованной (при K=1 и одинаковых seed — идентичные).

Различия возможны из-за:
- Случайного порядка клиентов в каждой эпохе
- Разного порядка батчей (при K>1)
