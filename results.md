# results.md - журнал результатов проекта emg-sampling

Документ заполняется вручную по ходу работы. После каждого запуска добавляйте строку в журнал и при необходимости обновляйте сводные таблицы.

## 1) Общий статус проекта

Дата последнего обновления: `2026-05-14`

| Поле | Значение |
|---|---|
| Текущая стадия | `Stage 2` |
| Основная цель текущей стадии | Подготовить воспроизводимую инфраструктуру экспериментов по сокращению числа каналов EMG на GRABMyo |
| Что уже завершено | Проверка структуры проекта, подтверждение наличия GRABMyo, восстановление editable install, подтверждение импорта `src/emg_sampling/__init__.py`, сборка индекса 4 классов, реализация `selection/` и `channel_sweep`, быстрый прогон `channel_sweep --quick` с CSV и графиками |
| Что в работе | Анализ quick-результатов channel reduction и подготовка следующих экспериментов |
| Что заблокировано | Полный запуск Stage 2 еще не выполнен; `uv sync` в текущей среде ограничен доступом к глобальному cache и сетевыми ограничениями PyPI |
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
| `2026-05-14 11:11` | ``.\.venv\Scripts\python.exe scripts/make_index_4classes.py`` | `Stage 2` | `GRABMyo 4-class index build` | `rows=3612; participants=43; class_balance=903 per class` | ``data/processed/grabmyo_index_4classes.csv`` | Индекс успешно собран из локально доступных сырых данных |
| `2026-05-14 11:16` | ``.\.venv\Scripts\python.exe scripts/run_channel_sweep.py --quick --methods full random ranking greedy --channel-counts 3 4 6 8 12 16 24 --random-repeats 3`` | `Stage 2` | `quick subset; win=200ms; filtered=off; features=basic; methods=full/random/ranking/greedy` | `best_macro_f1=0.992277 (greedy, 12ch); full_24_macro_f1=0.889097` | ``results/tables/channel_sweep_results.csv``, ``results/plots/channel_sweep_*.png`` | Quick subset: train participants `1,2,3,4`, test participants `42,43`, total records `48` |

---

## 3) Таблицы итогов по стадиям

### 3.1 Baseline (Stage 1)

| Конфигурация | Channels | Window (ms) | Filtering | Features | Accuracy | Macro-F1 | Processing time (ms) | Источник |
|---|---:|---:|---|---|---:|---:|---:|---|
| baseline_01 | 24 |  | on/off | basic |  |  |  | ``results/tables/baseline_results.csv`` |

### 3.2 Channel sweep (Stage 2)

| Method | Channels count | Selected channels | Window (ms) | Filtering | Features | Accuracy | Macro-F1 | Processing time (ms) | Feature vector size | Источник |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---|
| random | 8 | `[5, 10, 11, 12, 14, 15, 16, 17]` | 200 | off | basic | 0.921392 | 0.921988 | 0.039450 | 32 | ``results/tables/channel_sweep_results.csv`` |
| ranking | 16 | `[14, 6, 15, 7, 5, 13, 4, 11, 1, 9, 12, 3, 0, 2, 10, 19]` | 200 | off | basic | 0.914948 | 0.915771 | 0.047717 | 64 | ``results/tables/channel_sweep_results.csv`` |
| greedy | 12 | `[14, 11, 20, 0, 21, 22, 23, 12, 4, 2, 10, 6]` | 200 | off | basic | 0.992268 | 0.992277 | 0.042854 | 48 | ``results/tables/channel_sweep_results.csv`` |
| full | 24 | all | 200 | off | basic | 0.887887 | 0.889097 | 0.057908 | 96 | ``results/tables/channel_sweep_results.csv`` |

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
| `2026-05-14` | В сырых WFDB-записях обнаружены 32 сигнала с именами `F*`, `W*`, `U*`, тогда как проектная постановка требует baseline на 24 каналах. Для согласования Stage 2 временно зафиксирован проектный 24-канальный набор. | Инспекция `wfdb.rdsamp(...)`, `data/raw/grabmyo/1.1.0/readme.txt`, код `resolve_project_channel_indices()` | Полный Stage 2 следует считать воспроизводимым относительно этого 24-канального project subset, а происхождение исключенных каналов нужно отдельно уточнить перед текстом диплома. |
| `2026-05-14` | На quick subset greedy selection дает существенно лучший Macro-F1, чем full-24 baseline и single-channel ranking. | ``results/tables/channel_sweep_results.csv`` и график ``results/plots/channel_sweep_macro_f1.png`` | Быстрый следующий шаг - перепроверить greedy и ranking на более крупном subset и с filtered-сигналом, чтобы отделить реальный выигрыш от эффекта малого числа участников. |

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
| `2026-05-14` | Несовпадение числа каналов в данных и в постановке эксперимента | Быстрый sweep сначала использовал 32 канала вместо ожидаемых 24 | В WFDB-файлах есть дополнительные сигналы `U1-U4`, а из EMG-разметки без дополнительного уточнения получается больше 24 каналов | Введена функция `resolve_project_channel_indices()` и зафиксирован project subset на 24 канала для всех baseline-like экспериментов | `resolved` |

Подсказки:
- Фиксируйте команды/пути, если проблема воспроизводится не всегда.
- Для resolved проблем кратко пишите, что изменили и где.

---

## 6) Следующие шаги

Шаблон на ближайшую итерацию (заполняется перед новой серией запусков):

| Приоритет | Задача | Критерий готовности | Команда/скрипт | Ожидаемый артефакт | Статус |
|---|---|---|---|---|---|
| P0 | Завершить чистку README и журнала перед экспериментами | `README.md` и `results.md` обновлены, импорт подтвержден, сделан коммит `chore: finalize project layout and docs` | ``.\.venv\Scripts\python.exe main.py`` | Обновленные документы и первый коммит сессии | `done` |
| P1 | Реализовать `selection/` и `run_channel_sweep.py` | Есть random/ranking/greedy/full channel sweep с CSV и минимум одним графиком | ``.\.venv\Scripts\python.exe scripts/run_channel_sweep.py --quick`` | ``results/tables/channel_sweep_results.csv`` | `done` |
| P2 | Подготовить feature sweep после channel sweep | Есть `basic` и `extended_td`, сохранены CSV и графики | ``.\.venv\Scripts\python.exe scripts/run_feature_sweep.py --quick`` | ``results/tables/feature_sweep_results.csv`` | `in_progress` |

Короткий чек перед запуском:
- [ ] Данные доступны (`data/raw/grabmyo`)
- [ ] Индекс готов (`data/processed/grabmyo_index_4classes.csv`)
- [ ] Команда запуска зафиксирована заранее
- [ ] Путь артефактов заранее определен
- [ ] После запуска строка добавлена в "Журнал запусков"

## 7) Текстовые заготовки для отчета

### Stage 2 - quick proof of concept

Эксперимент по сокращению числа каналов был проведен в быстром режиме для проверки работоспособности инфраструктуры `channel_sweep` и первичной оценки чувствительности качества распознавания к размеру подмножества каналов. Использовался упрощенный subset датасета GRABMyo: 4 обучающих участника (`1,2,3,4`) и 2 тестовых участника (`42,43`), по 2 записи на комбинацию `participant x class`, всего 48 записей. Классы оставались теми же, что и в baseline: `WF`, `WE`, `HO`, `HC`.

Во всех конфигурациях использовались окна 200 ms, необработанный сигнал и базовый набор признаков `MAV`, `RMS`, `WL`, `ZC`. Были реализованы и проверены четыре режима сравнения: `full`, `random`, `ranking`, `greedy`. Для `random` выполнялось 3 повтора на каждое число каналов. Результаты сохранялись в `results/tables/channel_sweep_results.csv`, а сводные графики - в `results/plots/channel_sweep_*.png`.

Лучший результат в quick-режиме показал `greedy` при 12 каналах: `accuracy = 0.992268`, `macro_f1 = 0.992277`, `feature_vector_size = 48`. Для сравнения, базовая конфигурация `full` на 24 каналах дала `accuracy = 0.887887` и `macro_f1 = 0.889097`. Такой отрыв выглядит многообещающим, но пока не должен интерпретироваться как окончательный научный вывод, поскольку быстрый режим использует малое число участников и может переоценивать эффективность greedy-подбора.

Ограничение текущего этапа состоит в неоднозначности исходной разметки каналов GRABMyo. При чтении WFDB-записей обнаружены 32 сигнала (`F*`, `W*`, `U*`), тогда как проектная постановка и baseline ориентированы на 24 канала. Для согласования всех baseline-like экспериментов временно зафиксирован project subset на 24 канала через `resolve_project_channel_indices()`. Перед включением итоговых чисел в диплом необходимо отдельно подтвердить физический смысл исключенных каналов и, при необходимости, повторить Stage 2 на окончательно утвержденной 24-канальной конфигурации.
