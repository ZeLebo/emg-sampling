# results.md - журнал результатов проекта emg-sampling

Документ заполняется вручную по ходу работы. После каждого запуска добавляйте строку в журнал и при необходимости обновляйте сводные таблицы.

## 1) Общий статус проекта

Дата последнего обновления: `2026-05-14`

| Поле | Значение |
|---|---|
| Текущая стадия | `Stage 2` |
| Основная цель текущей стадии | Подготовить воспроизводимую инфраструктуру экспериментов по сокращению числа каналов EMG на GRABMyo |
| Что уже завершено | Проверка структуры проекта, подтверждение наличия GRABMyo, восстановление editable install, подтверждение импорта `src/emg_sampling/__init__.py`, обновление `todo.md` под текущую сессию |
| Что в работе | P0-документация и подготовка модулей для channel reduction |
| Что заблокировано | `uv sync` в текущей среде ограничен доступом к глобальному cache и сетевыми ограничениями PyPI |
| Ссылка на основной план | `goal.md`, `todo.md` |

Короткий прогресс по стадиям:

- [x] Stage 1 - Baseline pipeline (24 канала)
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
| `2026-05-14 10:59` | ``git status --short --branch`` | `Stage 2` | `repo audit before channel reduction` | `branch=master ahead 1; worktree clean except untracked goal.md` | - | Начальная проверка автономной сессии |
| `2026-05-14 11:03` | ``.\.venv\Scripts\python.exe -m ensurepip --upgrade`` | `Stage 2` | `restore pip in local .venv` | `pip installed successfully` | - | Потребовалось для локальной editable-установки без пересоздания окружения |
| `2026-05-14 11:05` | ``.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation`` | `Stage 2` | `editable install from local sources; offline build` | `install=ok` | - | Сработало только с `--no-build-isolation` из-за сетевых ограничений |
| `2026-05-14 11:05` | ``.\.venv\Scripts\python.exe -c "import emg_sampling; print(emg_sampling.__file__)"`` | `Stage 2` | `import verification` | `path=src/emg_sampling/__init__.py` | - | Конфликт пакетов не обнаружен |

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
| `2026-05-14` | Проект можно использовать без пересоздания окружения, если выполнять editable-установку из локальной `.venv` с флагом `--no-build-isolation`. | Команды установки и импорта в журнале запусков | Для дальнейших прогонов в этой среде не нужно зависеть от `uv sync`, пока зависимости уже присутствуют локально. |

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
| `2026-05-14` | `uv sync` не завершается в текущей среде | Ошибка доступа к `C:\Users\sarta\AppData\Local\uv\cache` и сетевой отказ при запросах к PyPI | Ограничения sandbox/сети и прав на глобальный cache | Использована локальная `.venv`; `pip` восстановлен через `ensurepip`; editable install выполнен с `--no-build-isolation` | `resolved` |
| `2026-05-14` | Импорт `emg_sampling` изначально не работал | `ModuleNotFoundError` при вызове `.venv\Scripts\python.exe -c ...` | Пакет не был установлен в окружение | Выполнена локальная editable-установка и подтвержден путь `src/emg_sampling/__init__.py` | `resolved` |

Подсказки:
- Фиксируйте команды/пути, если проблема воспроизводится не всегда.
- Для resolved проблем кратко пишите, что изменили и где.

---

## 6) Следующие шаги

Шаблон на ближайшую итерацию (заполняется перед новой серией запусков):

| Приоритет | Задача | Критерий готовности | Команда/скрипт | Ожидаемый артефакт | Статус |
|---|---|---|---|---|---|
| P0 | Завершить чистку README и журнала перед экспериментами | `README.md` и `results.md` обновлены, импорт подтвержден, сделан коммит `chore: finalize project layout and docs` | ``.\.venv\Scripts\python.exe main.py`` | Обновленные документы и первый коммит сессии | `in_progress` |
| P1 | Реализовать `selection/` и `run_channel_sweep.py` | Есть random/ranking/greedy/full channel sweep с CSV и минимум одним графиком | ``.\.venv\Scripts\python.exe scripts/run_channel_sweep.py --quick`` | ``results/tables/channel_sweep_results.csv`` | `todo` |
| P2 | Подготовить feature sweep после channel sweep | Есть `basic` и `extended_td`, сохранены CSV и графики | ``.\.venv\Scripts\python.exe scripts/run_feature_sweep.py --quick`` | ``results/tables/feature_sweep_results.csv`` | `todo` |

Короткий чек перед запуском:
- [ ] Данные доступны (`data/raw/grabmyo`)
- [ ] Индекс готов (`data/processed/grabmyo_index_4classes.csv`)
- [ ] Команда запуска зафиксирована заранее
- [ ] Путь артефактов заранее определен
- [ ] После запуска строка добавлена в "Журнал запусков"
