# Results Log - emg-sampling

Дата последнего обновления: `2026-05-14`

## 1. Current Status

| Field | Value |
|---|---|
| Current stage | `Stage 2-5` |
| Main objective | Проверить надежность channel reduction experiments, убрать leakage, получить leakage-safe medium results |
| Completed | Leakage audit and fix, quick rerun, medium channel sweep, medium feature sweep, raw-vs-filtered comparison, stability analysis, offline bandit, POC visualization from previous session |
| In progress | Подготовка следующего полного прогона на большем объеме данных |
| Blocked | Полный большой прогон на всем датасете еще не выполнен |
| Main plan | `goal.md` |

Закрытые стадии:

- [x] Stage 1 - baseline pipeline
- [x] Stage 2 - channel reduction quick and medium
- [x] Stage 3 - feature sweep quick and medium
- [x] Stage 4 - offline bandit proof of concept
- [x] Stage 5 - simple hand visualization proof of concept
- [ ] Stage 6 - final analysis on larger/full runs

## 2. Methodology

### 2.1 Leakage-safe protocol

Правильная схема после исправления:

```text
selection_train participants
  -> ranking / greedy channel selection
selection_val participants
  -> selection metric for channel choice
final_train = selection_train + selection_val
  -> final LDA training with fixed selected channels
final_test participants
  -> final evaluation only
```

Главный принцип:

- `selection_val` используется для выбора каналов.
- `final_test` используется только для финальной оценки.
- Во всех новых CSV поле `selection_uses_test` должно быть `false`.

### 2.2 Split modes

| Mode | selection_train | selection_val | final_test | Notes |
|---|---|---|---|---|
| `quick` | `1-2` | `3-4` | `42-43` | 2 записи на `participant x class` |
| `medium` | `1-12` | `13-16` | `40-43` | leakage-safe medium subset |
| `full` | `1-32` | `33-39` | `40-43` | код готов, но в этой сессии не запускался |

### 2.3 Fixed project channel layout

В исходных WFDB-записях GRABMyo были обнаружены 32 сигнала, тогда как проектная постановка ориентирована на 24 канала. Для согласования baseline-like экспериментов используется фиксированный project subset на 24 канала через `resolve_project_channel_indices()`.

## 3. Leakage Audit

### 3.1 What was wrong

До исправления:

- `channel_ranking.py` оценивал каналы на `X_test, y_test`.
- `greedy_selection.py` на каждом шаге выбирал канал по метрике на `X_test, y_test`.
- `channel_sweep.py` передавал в ranking/greedy именно финальный test split.

Это означало прямое использование test set для выбора каналов и, следовательно, завышение quick-результатов.

### 3.2 What was fixed

После исправления:

- добавлен leakage-safe split builder в `src/emg_sampling/data/split.py`;
- `channel_sweep.py` использует `selection_train` и `selection_val` для ranking/greedy;
- финальная оценка выполняется только на `final_test`;
- `feature_sweep.py` теперь читает leakage-safe channel selections и оценивает их только на final test;
- в CSV добавлены поля:
  - `selection_train_participants`
  - `selection_val_participants`
  - `final_test_participants`
  - `selection_metric`
  - `final_metric`
  - `selection_uses_test`

### 3.3 Leakage conclusion

Вывод:

- test leakage был в предыдущей quick-реализации;
- он исправлен;
- новые quick/medium результаты можно рассматривать как методически корректные относительно текущего split protocol.

## 4. Executed Commands

Основные воспроизводимые команды текущего состояния:

```bash
.\.venv\Scripts\python.exe scripts/make_index_4classes.py
.\.venv\Scripts\python.exe scripts/run_channel_sweep.py --quick --methods full random ranking greedy --channel-counts 3 4 6 8 12 16 24 --random-repeats 3
.\.venv\Scripts\python.exe scripts/run_feature_sweep.py --quick --channel-method greedy --channel-counts 3 6 8 12 24 --feature-sets basic extended_td
.\.venv\Scripts\python.exe scripts/run_channel_sweep.py --medium --methods full random ranking greedy --channel-counts 3 4 6 8 12 16 24 --random-repeats 3
.\.venv\Scripts\python.exe scripts/run_feature_sweep.py --medium --channel-method greedy --channel-counts 3 6 8 12 24 --feature-sets basic extended_td
.\.venv\Scripts\python.exe scripts/run_channel_sweep_filtered_comparison.py --channel-counts 3 6 8 12 24 --methods full ranking greedy random --random-repeats 3
.\.venv\Scripts\python.exe scripts/run_channel_stability.py --channel-counts 3 6 8 12 --methods ranking greedy
.\.venv\Scripts\python.exe scripts/run_bandit_simulation.py
```

## 5. Main Artifacts

### 5.1 CSV

- `data/processed/grabmyo_index_4classes.csv`
- `results/tables/channel_sweep_results.csv`
- `results/tables/feature_sweep_results.csv`
- `results/tables/channel_sweep_medium_results.csv`
- `results/tables/feature_sweep_medium_results.csv`
- `results/tables/channel_sweep_filtered_comparison.csv`
- `results/tables/channel_stability_results.csv`
- `results/tables/bandit_simulation_results.csv`

### 5.2 Plots

- `results/plots/channel_sweep_macro_f1.png`
- `results/plots/channel_sweep_accuracy.png`
- `results/plots/channel_sweep_medium_macro_f1.png`
- `results/plots/channel_sweep_medium_accuracy.png`
- `results/plots/channel_sweep_filtered_comparison_macro_f1.png`
- `results/plots/feature_sweep_macro_f1.png`
- `results/plots/feature_sweep_medium_macro_f1.png`
- `results/plots/channel_stability_jaccard.png`
- `results/plots/bandit_reward_by_profile.png`
- `results/plots/bandit_selected_actions.png`
- `results/plots/poc_hand_frame_001.png` - `results/plots/poc_hand_frame_004.png`
- `results/plots/poc_hand_demo.gif`

## 6. Experiment Log

| Date | Command | Stage | Config | Key metrics | Artifacts | Comment |
|---|---|---|---|---|---|---|
| `2026-05-14 11:16` | `run_channel_sweep.py --quick ...` | Stage 2 | leakage-safe quick, raw, `3/4/6/8/12/16/24`, `full/random/ranking/greedy` | best quick channel result after fix: `random 8ch`, `macro_f1=0.925106` | `results/tables/channel_sweep_results.csv` | quick rerun after leakage fix |
| `2026-05-14 11:26` | `run_feature_sweep.py --quick ...` | Stage 3 | leakage-safe quick, greedy source, `basic/extended_td` | best quick feature result after fix: `greedy 6ch + extended_td`, `macro_f1=0.950219` | `results/tables/feature_sweep_results.csv` | quick rerun after leakage fix |
| `2026-05-14 12:03` | `run_channel_sweep.py --medium ...` | Stage 2 | medium split, raw, `3/4/6/8/12/16/24`, `full/random/ranking/greedy` | best medium channel result: `full 24ch`, `macro_f1=0.934522` | `results/tables/channel_sweep_medium_results.csv` | medium results are much more conservative than old quick POC |
| `2026-05-14 12:07` | `run_feature_sweep.py --medium ...` | Stage 3 | medium split, raw, greedy source, `basic/extended_td` | best medium feature result: `full 24ch + extended_td`, `macro_f1=0.957528` | `results/tables/feature_sweep_medium_results.csv` | extended features remain useful on medium split |
| `2026-05-14 12:12` | `run_channel_sweep_filtered_comparison.py ...` | Stage 2 | medium split, raw vs filtered | filtering helps mostly for `greedy 3ch` and `greedy 6ch` | `results/tables/channel_sweep_filtered_comparison.csv` | full summary in Section 9 |
| `2026-05-14 12:15` | `run_channel_stability.py ...` | Stage 2 | 4 different train/val folds, `ranking/greedy`, `3/6/8/12` | ranking much more stable than greedy | `results/tables/channel_stability_results.csv` | full summary in Section 10 |
| `2026-05-14 12:17` | `run_bandit_simulation.py` | Stage 4 | offline profiles over medium CSVs | profile-dependent action selection | `results/tables/bandit_simulation_results.csv` | full summary in Section 11 |

## 7. Channel Sweep Results

### 7.1 Quick results after leakage fix

| Method | Channels | Selected channels | Accuracy | Macro-F1 | Feature size | Comment |
|---|---:|---|---:|---:|---:|---|
| `random` | 8 | `[5, 10, 11, 12, 14, 15, 16, 17]` | 0.924613 | 0.925106 | 32 | лучший quick result после leakage fix |
| `ranking` | 16 | `[14, 13, 7, 5, 12, 15, 6, 9, 4, 8, 19, 17, 0, 20, 16, 11]` | 0.923325 | 0.923127 | 64 | лучшая ranking quick config |
| `full` | 24 | `all` | 0.908505 | 0.909619 | 96 | 24-channel baseline |
| `greedy` | 6 | `[14, 21, 11, 15, 4, 7]` | 0.878866 | 0.881929 | 24 | уже не доминирует после leakage fix |

### 7.2 Medium results

| Method | Channels | Selected channels | Accuracy | Macro-F1 | Feature size | Comment |
|---|---:|---|---:|---:|---:|---|
| `full` | 24 | `all` | 0.934094 | 0.934522 | 96 | лучший medium channel result |
| `ranking` | 24 | ranked all 24 | 0.934094 | 0.934522 | 96 | эквивалентно full при 24 каналах |
| `random` | 16 | `[1, 3, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22]` | 0.931149 | 0.931738 | 64 | surprisingly strong medium random draw |
| `greedy` | 16 | `[6, 8, 5, 16, 9, 23, 22, 4, 13, 12, 1, 0, 20, 3, 15, 17]` | 0.917833 | 0.918365 | 64 | лучший medium greedy result |
| `greedy` | 6 | `[6, 8, 5, 16, 9, 23]` | 0.816121 | 0.816413 | 24 | сильное падение относительно 24ch |

### 7.3 Quick vs medium interpretation

Главное изменение после medium run:

- старый quick proof-of-concept создавал впечатление, что aggressive channel reduction почти не портит качество;
- leakage-safe medium run это не подтвердил;
- на medium split лучший результат дает не reduced subset, а `full 24 channels`.

То есть quick-выводы про почти безболезненное сокращение каналов оказались слишком оптимистичными.

## 8. Feature Sweep Results

### 8.1 Quick feature sweep after leakage fix

| Channels | Feature set | Accuracy | Macro-F1 | Feature size |
|---|---:|---:|---:|---:|
| 3 | `basic` | 0.847294 | 0.849543 | 12 |
| 3 | `extended_td` | 0.905284 | 0.903389 | 33 |
| 6 | `basic` | 0.903995 | 0.905493 | 24 |
| 6 | `extended_td` | 0.950387 | 0.950219 | 66 |
| 24 | `basic` | 0.882088 | 0.883468 | 96 |
| 24 | `extended_td` | 0.896907 | 0.894291 | 264 |

### 8.2 Medium feature sweep

| Channels | Feature set | Accuracy | Macro-F1 | Feature size |
|---|---:|---:|---:|---:|
| 3 | `basic` | 0.741225 | 0.744827 | 12 |
| 3 | `extended_td` | 0.779026 | 0.781798 | 33 |
| 6 | `basic` | 0.813451 | 0.814763 | 24 |
| 6 | `extended_td` | 0.835236 | 0.836649 | 66 |
| 8 | `basic` | 0.809278 | 0.810108 | 32 |
| 8 | `extended_td` | 0.866624 | 0.866629 | 88 |
| 12 | `basic` | 0.818606 | 0.819423 | 48 |
| 12 | `extended_td` | 0.878651 | 0.878847 | 132 |
| 24 | `basic` | 0.927774 | 0.928406 | 96 |
| 24 | `extended_td` | 0.957444 | 0.957528 | 264 |

### 8.3 Feature sweep interpretation

Вывод:

- `extended_td` стабильно помогает и на quick, и на medium;
- однако выигрыш от `extended_td` не компенсирует потери от сильного channel reduction;
- наиболее надежный medium result сейчас - `24 channels + extended_td`.

## 9. Raw vs Filtered Comparison

Средний `macro_f1(filtered) - macro_f1(raw)` на medium split:

| Method | Channels | Delta |
|---|---:|---:|
| `greedy` | 3 | `+0.018059` |
| `greedy` | 6 | `+0.024126` |
| `greedy` | 8 | `-0.001802` |
| `greedy` | 12 | `-0.000993` |
| `full` | 24 | `-0.002050` |
| `ranking` | 3 | `-0.005928` |
| `ranking` | 6 | `-0.007793` |
| `ranking` | 8 | `-0.003220` |
| `ranking` | 12 | `-0.005247` |
| `ranking` | 24 | `-0.002050` |

Интерпретация:

- фильтрация помогает прежде всего для маленьких greedy subsets, особенно `3ch` и `6ch`;
- для `ranking` и для больших наборов каналов эффект чаще нейтральный или слегка отрицательный;
- для полного 24-канального baseline фильтрация в этом medium experiment не дала прироста.

Практический вывод:

- если тестировать очень компактные greedy-конфиги, filtered mode стоит сохранять в рассмотрении;
- для устойчивых более крупных конфигураций raw mode пока выглядит не хуже.

## 10. Stability Analysis

Mean Jaccard between selected channel sets across 4 different train/val folds:

| Method | Channels | Mean Jaccard |
|---|---:|---:|
| `ranking` | 3 | 0.216667 |
| `ranking` | 6 | 0.579365 |
| `ranking` | 8 | 0.696296 |
| `ranking` | 12 | 0.742857 |
| `greedy` | 3 | 0.033333 |
| `greedy` | 6 | 0.174747 |
| `greedy` | 8 | 0.218559 |
| `greedy` | 12 | 0.375559 |

Часто повторяющиеся каналы:

- `ranking 6ch`: `6, 7, 14` встречаются во всех 4 folds
- `ranking 12ch`: `13, 5, 15, 6, 7, 14` встречаются во всех 4 folds
- `greedy 12ch`: наиболее частые `23, 8, 16`

Вывод:

- `ranking` существенно стабильнее `greedy`;
- `greedy` сильно зависит от конкретного train/val split, особенно на `3ch` и `6ch`;
- найденным greedy subsets нельзя доверять как единственно правильным без дополнительной проверки на разных splits.

## 11. Offline Bandit Simulation

Лучшее действие для каждого reward profile:

| Profile | Method | Channels | Feature set | Filtering | Macro-F1 | Processing time ms | Feature size | Reward |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `quality_first` | `full` | 24 | `extended_td` | off | 0.957528 | 0.214653 | 264 | 0.916981 |
| `balanced` | `random` | 16 | `basic` | off | 0.931738 | 0.175192 | 64 | 0.868882 |
| `compression_first` | `random` | 6 | `basic` | off | 0.852585 | 0.133304 | 24 | 0.785119 |
| `low_latency` | `random` | 16 | `basic` | off | 0.931738 | 0.175192 | 64 | 0.874522 |

Интерпретация:

- при приоритете качества bandit выбирает `24ch + extended_td`;
- при сильном штрафе за channels count он переходит к `6ch basic`;
- balanced и low-latency profiles в текущей таблице выбирают `16ch random basic`, потому что этот action сохраняет высокое качество при умеренном размере конфигурации.

Это пока proof of concept, потому что bandit работает offline по уже рассчитанным CSV, а не в online signal-adaptive loop.

## 12. What Can Be Used in Diploma

### 12.1 Cautiously usable

Можно осторожно использовать:

- leakage-safe medium channel sweep;
- leakage-safe medium feature sweep;
- вывод, что после устранения leakage сильное сокращение каналов больше не выглядит почти бесплатным;
- вывод, что `extended_td` помогает, но не отменяет потери от сильного уменьшения числа каналов;
- вывод, что `ranking` стабильнее `greedy`;
- вывод, что filtering полезнее для very low-channel greedy setups, чем для full baseline.

### 12.2 Proof of concept only

Пока только proof of concept:

- quick results;
- POC hand visualization;
- offline bandit simulation;
- точный выбор "лучшего" reduced subset на основе greedy.

## 13. Problems and Resolutions

| Date | Problem | Cause | Resolution | Status |
|---|---|---|---|---|
| `2026-05-14` | `uv sync` was unreliable in current environment | cache and network restrictions | used local `.venv` and editable install | resolved |
| `2026-05-14` | test leakage in channel selection | ranking and greedy used final test split | introduced leakage-safe selection protocol | resolved |
| `2026-05-14` | mismatch between raw 32 signals and project 24-channel setup | dataset layout differs from project assumption | fixed project subset through `resolve_project_channel_indices()` | resolved |

## 14. Commits

Коммиты, относящиеся к этой линии экспериментов:

- `chore: finalize project layout and docs`
- `fix: prevent test leakage in channel selection`
- `feat: add medium-scale channel and feature sweeps`
- `feat: compare raw and filtered channel sweeps`
- `feat: add channel selection stability analysis`
- `feat: add offline bandit simulation`
- `feat: add simple hand movement proof of concept`
- `docs: format results log for diploma use`

## 15. Final Summary

### 15.1 Leakage checks performed

Выполненные проверки:

1. Прочитаны `channel_ranking.py`, `greedy_selection.py`, `channel_sweep.py`, `feature_sweep.py`.
2. Подтверждено, что старая версия использовала final test split для выбора каналов.
3. Внедрены leakage-safe participant splits.
4. Повторно сгенерированы quick CSV и plots после исправления.
5. Сгенерированы medium CSV и plots после исправления.

### 15.2 Was there test leakage?

Да. В предыдущей реализации ranking и greedy использовали test set для выбора каналов. В текущей версии это исправлено, и `selection_uses_test=false` сохранено в новых CSV.

### 15.3 How did medium results change the old quick conclusion?

Изменение принципиальное:

- старый quick POC намекал, что 6-12 каналов почти не хуже 24;
- medium leakage-safe results это не подтвердили;
- наиболее надежная конфигурация сейчас - `24 channels`, особенно с `extended_td`.

### 15.4 Stable channels

Наиболее устойчиво повторяются каналы из ranking-based selection. Особенно стабильно появляются каналы `6`, `7`, `14`, `15`, а на больших наборах также `5` и `13`.

### 15.5 Next run

Следующим запуском стоит:

1. Запустить `full` split mode на большей части датасета, если укладывается по времени.
2. Проверить `extended_td` не только для greedy source, но и для ranking/full.
3. Отдельно перепроверить 24-channel layout against GRABMyo documentation.
4. Добавить confidence intervals или повторные medium splits для финальных таблиц диплома.
