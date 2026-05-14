# Pipeline

## Цель

Этот документ фиксирует воспроизводимый baseline pipeline для GRABMyo.

## Шаги

1. Сбор индекса целевых классов (`scripts/make_index_4classes.py`).
2. Разделение train/test по участникам.
3. Опциональная фильтрация (`notch 50 Hz`, `bandpass 20-450 Hz`).
4. Sliding windows с заданными `win_ms` и `hop_fraction`.
5. Извлечение TD-features (`MAV`, `RMS`, `WL`, `ZC`).
6. Обучение `LinearDiscriminantAnalysis`.
7. Оценка `accuracy` и `macro_f1`.
8. Сохранение таблиц в `results/tables/`.

## Входные и выходные файлы

- Вход: `data/raw/grabmyo/1.1.0/**.hea|dat`.
- Индекс: `data/processed/grabmyo_index_4classes.csv`.
- Результаты: `results/tables/*.csv`.

## Почему split по участникам

Окна из одного участника сильно коррелированы между собой. Split по окнам приводит к утечке и завышенным метрикам. Split по участникам лучше отражает переносимость между людьми.
