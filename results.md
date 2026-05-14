# results.md - журнал результатов проекта emg-sampling

Документ заполняется вручную по ходу работы. После каждого запуска добавляйте строку в журнал и при необходимости обновляйте сводные таблицы.

## 1) Общий статус проекта

Дата последнего обновления: `YYYY-MM-DD`

| Поле | Значение |
|---|---|
| Текущая стадия | `Stage 1 / Stage 2 / Stage 3 / Stage 4 / Stage 5 / Stage 6` |
| Основная цель текущей стадии |  |
| Что уже завершено |  |
| Что в работе |  |
| Что заблокировано |  |
| Ссылка на основной план | `README.md` / `todo.md` |

Короткий прогресс по стадиям:

- [ ] Stage 1 - Baseline pipeline (24 канала)
- [ ] Stage 2 - Channel reduction (random/ranking/greedy)
- [ ] Stage 3 - Feature sweep (basic vs extended_td)
- [ ] Stage 4 - Contextual bandit
- [ ] Stage 5 - GUI MVP
- [ ] Stage 6 - Анализ и выводы

---

## 2) Журнал запусков экспериментов

Правило заполнения:
- 1 запуск = 1 строка.
- Поле "Конфиг" заполнять кратко, но конкретно: окна, фильтрация, число каналов, feature set, метод выбора.
- В "Артефакты" указывать пути к CSV/графикам/моделям, созданным запуском.

| Дата | Команда | Стадия | Конфиг | Ключевые метрики | Артефакты | Комментарий |
|---|---|---|---|---|---|---|
| `YYYY-MM-DD HH:MM` | ``python scripts/run_baseline.py`` | `Stage 1` | `win=200ms; filtered=on; channels=24; features=basic` | `acc=...; macro_f1=...; latency_ms=...` | ``results/tables/...`` |  |

---

## 3) Таблицы итогов по стадиям

### 3.1 Baseline (Stage 1)

| Конфигурация | Channels | Window (ms) | Filtering | Features | Accuracy | Macro-F1 | Processing time (ms) | Источник |
|---|---:|---:|---|---|---:|---:|---:|---|
| baseline_01 | 24 |  | on/off | basic |  |  |  | ``results/tables/baseline_results.csv`` |

### 3.2 Channel sweep (Stage 2)

| Method | Channels count | Selected channels | Window (ms) | Filtering | Features | Accuracy | Macro-F1 | Processing time (ms) | Feature vector size | Источник |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---|
| random | 3 |  |  |  | basic/extended |  |  |  |  | ``results/tables/...`` |
| ranking | 6 |  |  |  | basic/extended |  |  |  |  | ``results/tables/...`` |
| greedy | 12 |  |  |  | basic/extended |  |  |  |  | ``results/tables/...`` |
| full | 24 | all |  |  | basic/extended |  |  |  |  | ``results/tables/...`` |

### 3.3 Feature sweep (Stage 3)

| Channels count | Feature set | Feature vector size | Window (ms) | Filtering | Accuracy | Macro-F1 | Processing time (ms) | Прирост/падение к basic | Источник |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| 3 | basic |  |  |  |  |  |  |  | ``results/tables/...`` |
| 3 | extended_td |  |  |  |  |  |  |  | ``results/tables/...`` |
| 24 | basic |  |  |  |  |  |  |  | ``results/tables/...`` |
| 24 | extended_td |  |  |  |  |  |  |  | ``results/tables/...`` |

### 3.4 Bandit (Stage 4)

| Policy | Action space | Reward formula (alpha/beta/gamma) | Avg reward | Accuracy | Macro-F1 | Avg channels | Avg latency (ms) | Avg feature size | Источник |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| fixed_24ch |  |  |  |  |  | 24 |  |  | ``results/tables/...`` |
| fixed_12ch |  |  |  |  |  | 12 |  |  | ``results/tables/...`` |
| fixed_6ch |  |  |  |  |  | 6 |  |  | ``results/tables/...`` |
| bandit |  |  |  |  |  |  |  |  | ``results/tables/...`` |

---

## 4) Наблюдения и инсайты

Фиксируйте только проверяемые выводы на основе конкретных запусков.

| Дата | Наблюдение | На основе каких запусков/файлов | Практический вывод |
|---|---|---|---|
| `YYYY-MM-DD` |  |  |  |

Подсказки:
- Что влияет на Macro-F1 сильнее всего.
- Где компромисс между качеством и задержкой.
- Как меняется стабильность при уменьшении каналов.
- Когда расширенные признаки оправданы по времени.

---

## 5) Проблемы и решения

| Дата | Проблема | Симптом | Причина | Решение | Статус |
|---|---|---|---|---|---|
| `YYYY-MM-DD` |  |  |  |  | `open / resolved` |

Подсказки:
- Фиксируйте команды/пути, если проблема воспроизводится не всегда.
- Для resolved проблем кратко пишите, что изменили и где.

---

## 6) Следующие шаги

Шаблон на ближайшую итерацию (заполняется перед новой серией запусков):

| Приоритет | Задача | Критерий готовности | Команда/скрипт | Ожидаемый артефакт | Статус |
|---|---|---|---|---|---|
| P0 |  |  |  |  | `todo / in_progress / done` |
| P1 |  |  |  |  | `todo / in_progress / done` |
| P2 |  |  |  |  | `todo / in_progress / done` |

Короткий чек перед запуском:
- [ ] Данные доступны (`data/raw/grabmyo`)
- [ ] Индекс готов (`data/processed/grabmyo_index_4classes.csv`)
- [ ] Команда запуска зафиксирована заранее
- [ ] Путь артефактов заранее определен
- [ ] После запуска строка добавлена в "Журнал запусков"
