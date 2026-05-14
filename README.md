# emg-sampling

## Краткое описание

`emg-sampling` - исследовательский проект для обработки многоканальных ЭМГ-сигналов из датасета GRABMyo.

Текущая версия проекта содержит baseline-пайплайн распознавания жестов кисти:

`raw EMG -> preprocessing -> windowing -> time-domain features -> LDA classifier -> metrics`

Проект используется как основа для дальнейших экспериментов по сокращению числа каналов ЭМГ-сигнала с 24 до 3-12 каналов.

## Что делает проект

Проект позволяет:

- скачать и подготовить данные GRABMyo;
- собрать индекс записей для выбранных классов;
- извлекать оконные признаки из ЭМГ-сигнала;
- обучать baseline LDA-классификатор;
- сравнивать качество при разных размерах окна;
- сравнивать обработку raw и filtered сигнала;
- проводить эксперименты с синтетическим шумом;
- сохранять результаты в CSV для дальнейшего анализа.

## Текущий статус

Сейчас проект находится в стадии исследовательского прототипа.

Уже реализовано:

- чтение записей GRABMyo через `wfdb`;
- построение индекса записей;
- baseline на LDA;
- time-domain признаки MAV, RMS, WL, ZC;
- sweep по размерам окна;
- фильтрация notch 50 Hz + bandpass 20-450 Hz;
- сравнение raw / filtered;
- эксперименты с синтетическими шумами.

Следующий этап:

- сокращение числа каналов;
- выбор информативных каналов;
- сравнение 3, 4, 6, 8, 12 и 24 каналов;
- расширение набора признаков;
- подготовка GUI-демонстратора.

## Используемый датасет

Используется открытый датасет GRABMyo.

Данные скачиваются в:

`data/raw/grabmyo/`

Команда:

```bash
wget -r -N -c -np -nH --cut-dirs=2 -P data/raw/grabmyo https://physionet.org/files/grabmyo/1.1.0/
```

## Распознаваемые классы

В текущем baseline используются 4 класса:

| Код | Жест |
|---|---|
| WF | wrist flexion |
| WE | wrist extension |
| HO | hand open |
| HC | hand close |

Используемые id жестов в GRABMyo:

- WF - gesture 11
- WE - gesture 12
- HO - gesture 15
- HC - gesture 16

## Общий пайплайн обработки

```text
raw GRABMyo record
        |
        v
optional filtering
notch 50 Hz + bandpass 20-450 Hz
        |
        v
sliding window
150-250 ms, hop = 25% window
        |
        v
time-domain features
MAV, RMS, WL, ZC
        |
        v
LDA classifier
        |
        v
Accuracy, Macro-F1, processing time
```

## Структура проекта

```text
src/emg_sampling/
├── data/              # загрузка GRABMyo, индексы, train/test split
├── preprocessing/     # фильтрация ЭМГ-сигнала
├── features/          # оконная обработка и признаки
├── models/            # baseline-модели
├── experiments/       # воспроизводимые эксперименты
├── visualization/     # графики и визуализация результатов
└── utils/             # общие утилиты

scripts/
├── make_index_4classes.py
├── run_baseline.py
├── run_window_sweep.py
├── run_filtered_vs_raw.py
├── run_noise_experiment.py
└── legacy/            # numbered-скрипты ранних экспериментов
```

Подробнее см. [docs/pipeline.md](docs/pipeline.md) и [docs/experiments.md](docs/experiments.md).

## Установка

### Вариант 1. Через uv

```bash
uv sync
```

### Вариант 2. Через pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Для Windows:

```bash
.venv\Scripts\activate
```

## Загрузка данных

1. Скачать GRABMyo:

```bash
wget -r -N -c -np -nH --cut-dirs=2 -P data/raw/grabmyo https://physionet.org/files/grabmyo/1.1.0/
```

2. Построить индекс 4 классов:

```bash
python scripts/make_index_4classes.py
```

После этого должен появиться файл:

`data/processed/grabmyo_index_4classes.csv`

## Подготовка индекса

Индекс формируется на основе имен WFDB-записей вида:

`session*_participant*_gesture*_trial*`

В индекс попадают только 4 целевых класса. В таблице сохраняются:

- путь к записи (`record_path`);
- участник (`participant`);
- сессия (`session`);
- номер жеста (`gesture`);
- буквенный код класса (`class`);
- численная метка (`y`);
- trial (`trial`).

## Запуск baseline

```bash
python scripts/run_baseline.py
```

Результат:

`results/tables/baseline_results.csv`

## Запуск window sweep

```bash
python scripts/run_window_sweep.py
```

Результат:

`results/tables/window_sweep_results.csv`

## Запуск сравнения raw и filtered

```bash
python scripts/run_filtered_vs_raw.py
```

Результат:

`results/tables/filtered_vs_raw_results.csv`

## Запуск эксперимента с шумом

```bash
python scripts/run_noise_experiment.py
```

Результат:

`results/tables/noise_experiment_results.csv`

## Где смотреть результаты

Основные CSV-таблицы:

- `results/tables/baseline_results.csv`
- `results/tables/window_sweep_results.csv`
- `results/tables/filtered_vs_raw_results.csv`
- `results/tables/noise_experiment_results.csv`

Графики можно сохранять в `results/plots/`.

## Описание основных модулей

| Файл / модуль | Назначение |
|---|---|
| `src/emg_sampling/paths.py` | Единые пути проекта и создание директорий |
| `src/emg_sampling/config.py` | Общие параметры экспериментов |
| `src/emg_sampling/data/grabmyo_loader.py` | Чтение записей GRABMyo и загрузка index CSV |
| `src/emg_sampling/data/index_records.py` | Построение full index и 4-class index |
| `src/emg_sampling/data/split.py` | Train/test split по участникам |
| `src/emg_sampling/preprocessing/filters.py` | Notch и bandpass фильтрация |
| `src/emg_sampling/features/time_domain.py` | Sliding windows и TD-features |
| `src/emg_sampling/models/lda_baseline.py` | Обучение и оценка LDA |
| `src/emg_sampling/experiments/baseline.py` | Baseline experiment |
| `src/emg_sampling/experiments/window_sweep.py` | Sweep по размерам окна |
| `src/emg_sampling/experiments/filtered_vs_raw.py` | Сравнение raw и filtered |
| `src/emg_sampling/experiments/noise_experiment.py` | Эксперимент с синтетическим шумом |
| `scripts/` | Короткие entrypoint-скрипты |
| `scripts/legacy/` | Старые экспериментальные скрипты |

## Time-domain признаки

В baseline используются 4 признака на каждый канал:

| Признак | Описание |
|---|---|
| MAV | Среднее абсолютное значение сигнала |
| RMS | Среднеквадратичное значение |
| WL | Длина waveform, сумма абсолютных разностей соседних отсчетов |
| ZC | Число пересечений нуля |

Если используется 24 канала, размер feature vector:

```text
24 channels * 4 features = 96 features
```

Если в будущем используется 6 каналов:

```text
6 channels * 4 features = 24 features
```

## Train/test split

Разделение выполняется по участникам, а не по отдельным окнам.

Это важно, потому что окна из одной и той же записи сильно похожи. Если случайно делить окна на train и test, качество будет завышено.

В текущей версии последние 2 участника используются как test set, остальные - как train set.

## Описание старых экспериментов

Старые numbered-скрипты сохранены в `scripts/legacy/` и не удалены.

Они полезны как история прототипирования:

- чтение одной записи;
- проверка заголовков и жестов;
- ранние версии baseline;
- отдельные визуализации окон и PSD;
- сравнение фильтрации;
- эксперименты с шумом;
- ранние GPU/Q-learning/bandit прототипы.

## Следующий этап работы

Следующий этап проекта - исследовать, насколько можно сократить число каналов ЭМГ-сигнала без существенной потери качества классификации.

Планируемые эксперименты:

- baseline на 24 каналах;
- сравнение 3, 4, 6, 8, 12, 16 и 24 каналов;
- random channel selection;
- ranking каналов по информативности;
- greedy channel selection;
- расширение набора признаков;
- сравнение качества, задержки и размера feature vector.

Цель следующего этапа - найти минимальный набор каналов, который сохраняет приемлемое качество распознавания.

## Возможные проблемы

- Ошибка `Index CSV not found`: сначала запустите `python scripts/make_index_4classes.py`.
- Ошибка `GRABMyo data not found`: проверьте путь `data/raw/grabmyo/1.1.0/`.
- Долгий запуск экспериментов: это нормально для полного window sweep на всем индексе.
- Ошибки импорта `emg_sampling`: запустите `pip install -e .` или `uv sync`.
