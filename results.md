# Results Log - emg-sampling

Дата последнего обновления: `2026-05-15`

## 1. Current Status

| Field | Value |
|---|---|
| Current stage | `Stage 2-6` |
| Main objective | Проверить надежность channel reduction experiments, убрать leakage, получить leakage-safe medium and larger-scale results |
| Completed | Leakage audit and fix, quick rerun, medium sweeps, full channel sweep, expanded feature sweep for ranking/full, repeated medium-like evaluation with CI, raw-vs-filtered comparison, stability analysis, offline bandit, corrected POC visualization |
| In progress | Финальная интерпретация и упаковка результатов для дипломного текста |
| Blocked | В репозитории нет отдельного PDF с внешней схемой GRABMyo, поэтому audit 24-channel layout опирается на WFDB header labels и project subset mapping |
| Main plan | `goal.md` |

Закрытые стадии:

- [x] Stage 1 - baseline pipeline
- [x] Stage 2 - channel reduction quick and medium
- [x] Stage 3 - feature sweep quick and medium
- [x] Stage 4 - offline bandit proof of concept
- [x] Stage 5 - simple hand visualization proof of concept
- [x] Stage 6 - final analysis on larger/full runs

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
| `full` | `1-32` | `33-39` | `40-43` | leakage-safe full-scale split, прогнан в этой сессии |

### 2.3 Fixed project channel layout

В исходных WFDB-записях GRABMyo были обнаружены 32 сигнала, тогда как проектная постановка ориентирована на 24 канала. Для согласования baseline-like экспериментов используется фиксированный project subset на 24 канала через `resolve_project_channel_indices()`.

WFDB header одного raw record подтверждает набор сигналов:

```text
F1-F16, U1, W1-W6, U2, U3, W7-W12, U4
```

Текущий project subset сохраняет:

```text
F1-F16, W1-W8
```

То есть индексы:

```text
(0..15, 17..22, 25, 26)
```

Из 32 сигналов исключаются:

```text
U1, U2, U3, U4, W9, W10, W11, W12
```

Вывод audit:

- 24-channel setup в проекте не является "первыми 24 каналами" raw записи;
- это осознанный fixed subset из 32 исходных сигналов;
- новая POC-визуализация теперь показывает layout как `4 rings x 6 electrodes`, но это все еще schematic representation именно project subset, а не официальная анатомическая карта из внешней документации.

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
.\.venv\Scripts\python.exe scripts/run_channel_sweep.py --methods full ranking greedy --channel-counts 3 6 8 12 16 24 --max-greedy-channels 16 --output results/tables/channel_sweep_full_results.csv --plot-prefix channel_sweep_full
.\.venv\Scripts\python.exe scripts/run_feature_sweep.py --medium --channel-method ranking --channel-counts 3 6 8 12 24 --feature-sets basic extended_td --output results/tables/feature_sweep_medium_ranking_results.csv --plot-prefix feature_sweep_medium_ranking
.\.venv\Scripts\python.exe scripts/run_feature_sweep.py --medium --channel-method full --channel-counts 24 --feature-sets basic extended_td --output results/tables/feature_sweep_medium_full_results.csv --plot-prefix feature_sweep_medium_full
.\.venv\Scripts\python.exe scripts/run_repeated_medium_evaluation.py --methods ranking full --channel-counts 12 24 --feature-sets basic extended_td --output results/tables/_tmp_repeated_check.csv
.\.venv\Scripts\python.exe scripts/run_repeated_medium_evaluation.py --methods greedy --channel-counts 6 12 --feature-sets basic extended_td --output results/tables/_tmp_repeated_greedy_check.csv
.\.venv\Scripts\python.exe scripts/run_channel_sweep_filtered_comparison.py --channel-counts 3 6 8 12 24 --methods full ranking greedy random --random-repeats 3
.\.venv\Scripts\python.exe scripts/run_channel_stability.py --channel-counts 3 6 8 12 --methods ranking greedy
.\.venv\Scripts\python.exe scripts/run_bandit_simulation.py
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --synthetic --all
```

## 5. Main Artifacts

### 5.1 CSV

- `data/processed/grabmyo_index_4classes.csv`
- `results/tables/channel_sweep_results.csv`
- `results/tables/feature_sweep_results.csv`
- `results/tables/channel_sweep_medium_results.csv`
- `results/tables/channel_sweep_full_results.csv`
- `results/tables/feature_sweep_medium_results.csv`
- `results/tables/feature_sweep_medium_ranking_results.csv`
- `results/tables/feature_sweep_medium_full_results.csv`
- `results/tables/channel_sweep_filtered_comparison.csv`
- `results/tables/channel_stability_results.csv`
- `results/tables/bandit_simulation_results.csv`
- `results/tables/repeated_medium_evaluation_results.csv`

### 5.2 Plots

- `results/plots/channel_sweep_macro_f1.png`
- `results/plots/channel_sweep_accuracy.png`
- `results/plots/channel_sweep_medium_macro_f1.png`
- `results/plots/channel_sweep_medium_accuracy.png`
- `results/plots/channel_sweep_full_macro_f1.png`
- `results/plots/channel_sweep_full_accuracy.png`
- `results/plots/channel_sweep_filtered_comparison_macro_f1.png`
- `results/plots/feature_sweep_macro_f1.png`
- `results/plots/feature_sweep_medium_macro_f1.png`
- `results/plots/feature_sweep_medium_ranking_macro_f1.png`
- `results/plots/feature_sweep_medium_full_macro_f1.png`
- `results/plots/channel_stability_jaccard.png`
- `results/plots/bandit_reward_by_profile.png`
- `results/plots/bandit_selected_actions.png`
- `results/plots/repeated_medium_macro_f1_ci.png`
- `results/plots/poc_hand_WF.png`
- `results/plots/poc_hand_WE.png`
- `results/plots/poc_hand_HO.png`
- `results/plots/poc_hand_HC.png`
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
| `2026-05-15` | `run_channel_sweep.py --methods full ranking greedy ...` | Stage 2 | leakage-safe full split, raw, `3/6/8/12/16/24` | best full result: `full 24ch`, `macro_f1=0.948098`; best reduced result: `greedy 16ch`, `macro_f1=0.944289` | `results/tables/channel_sweep_full_results.csv` | full split turned out feasible on CPU |
| `2026-05-15` | `run_feature_sweep.py --medium --channel-method ranking ...` | Stage 3 | medium split, ranking source, `basic/extended_td` | best ranking result: `12ch + extended_td`, `macro_f1=0.917802` | `results/tables/feature_sweep_medium_ranking_results.csv` | extended features help ranking too |
| `2026-05-15` | `run_feature_sweep.py --medium --channel-method full ...` | Stage 3 | medium split, full source, `24ch basic/extended_td` | `24ch + extended_td`, `macro_f1=0.957528` | `results/tables/feature_sweep_medium_full_results.csv` | confirms Stage 3 best baseline |
| `2026-05-15` | `run_repeated_medium_evaluation.py ...` | Stage 6 | 5 repeated medium-like participant splits | CI confirms `24ch + extended_td` as most reliable config | `results/tables/repeated_medium_evaluation_results.csv` | summary in Section 13 |
| `2026-05-15` | `run_poc_hand_visualization.py --synthetic --all` | Stage 5 | corrected `4 rings x 6 electrodes` schematic | updated PNG set and GIF | `results/plots/poc_hand_*.png`, `results/plots/poc_hand_demo.gif` | channel numbering now uses display channels consistently |

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

### 7.3 Full results

| Method | Channels | Selected channels | Accuracy | Macro-F1 | Comment |
|---|---:|---|---:|---:|---|
| `full` | 24 | `all` | 0.947901 | 0.948098 | лучший full result |
| `ranking` | 24 | ranked all 24 | 0.947901 | 0.948098 | эквивалентно full при 24 каналах |
| `greedy` | 16 | `[8, 5, 1, 6, 9, 17, 15, 22, 20, 18, 4, 3, 21, 2, 11, 10]` | 0.944066 | 0.944289 | лучший reduced full-split result |
| `greedy` | 12 | `[8, 5, 1, 6, 9, 17, 15, 22, 20, 18, 4, 3]` | 0.937592 | 0.937933 | surprisingly close to 24ch |
| `ranking` | 12 | `[8, 0, 13, 5, 15, 14, 7, 6, 9, 1, 11, 3]` | 0.892151 | 0.892246 | заметно слабее greedy на full split |

### 7.4 Quick vs medium vs full interpretation

Главное изменение после medium/full run:

- старый quick proof-of-concept создавал впечатление, что aggressive channel reduction почти не портит качество;
- leakage-safe medium run это не подтвердил;
- full split подтвердил тот же общий тренд: лучший результат дает `24 channels`, но `greedy 12-16ch` уже выглядит как более серьезный, а не совсем провальный компромисс;
- `ranking` на full split оказался слабее `greedy` для reduced subsets.

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

### 8.2 Medium feature sweep from greedy source

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

### 8.3 Medium feature sweep from ranking/full sources

| Method source | Channels | Feature set | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|
| `ranking` | 12 | `basic` | 0.873374 | 0.872662 |
| `ranking` | 12 | `extended_td` | 0.918047 | 0.917802 |
| `ranking` | 24 | `basic` | 0.927774 | 0.928406 |
| `ranking` | 24 | `extended_td` | 0.957444 | 0.957528 |
| `full` | 24 | `basic` | 0.927774 | 0.928406 |
| `full` | 24 | `extended_td` | 0.957444 | 0.957528 |

### 8.4 Feature sweep interpretation

Вывод:

- `extended_td` стабильно помогает и на quick, и на medium;
- это верно не только для `greedy`, но и для `ranking/full`;
- на `ranking 12ch` переход `basic -> extended_td` поднимает `macro_f1` с `0.872662` до `0.917802`;
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

## 12. POC Gesture Visualization

Была переработана демонстрационная визуализация распознавания жестов. В новой версии figure показывает полный путь:

```text
EMG window by channels -> channel layout -> predicted gesture -> hand pose
```

Что теперь есть на одной figure:

- заголовок `Predicted gesture: ...`
- confidence, если он доступен в dataset mode
- панель `EMG window by channels` с раздельными каналами и подписями `CH xx`
- простая схема расположения каналов на предплечье в виде `4 rings x 6 electrodes`
- заметка `Approximate 4-ring x 6-electrode layout`
- понятная схема руки для `WF`, `WE`, `HO`, `HC`
- поясняющий текст по жесту внизу

Synthetic mode поддерживается отдельно и явно помечается:

```text
Synthetic EMG window for visualization only
```

Это важно, потому что synthetic signal не должен восприниматься как реальная запись GRABMyo.

Созданные файлы:

- `results/plots/poc_hand_WF.png`
- `results/plots/poc_hand_WE.png`
- `results/plots/poc_hand_HO.png`
- `results/plots/poc_hand_HC.png`
- `results/plots/poc_hand_demo.gif`

Проверенные команды:

```bash
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --synthetic --gesture WF
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --synthetic --gesture WE
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --synthetic --gesture HO
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --synthetic --gesture HC
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --synthetic --all
.\.venv\Scripts\python.exe scripts/run_poc_hand_visualization.py --from-dataset --gesture WF --medium
```

Ограничения:

- channel layout теперь согласован с проектной схемой `4 x 6`, но остается schematic и не претендует на анатомически точное расположение электродов;
- hand drawing остается 2D proof of concept, а не биомеханической моделью;
- dataset mode сейчас ориентирован на поиск корректно предсказанного окна, а не на потоковую интерактивную анимацию.

## 13. Repeated Medium-like Evaluation with CI

Для более строгих таблиц был добавлен repeated evaluation по 5 leakage-safe medium-like participant splits.

Сводка по `macro_f1 mean [95% CI]`:

| Method | Channels | Feature set | Mean Macro-F1 | 95% CI |
|---|---:|---|---:|---|
| `greedy` | 6 | `basic` | 0.804420 | `[0.757651, 0.851189]` |
| `greedy` | 6 | `extended_td` | 0.859498 | `[0.826815, 0.892181]` |
| `greedy` | 12 | `basic` | 0.868784 | `[0.836166, 0.901402]` |
| `greedy` | 12 | `extended_td` | 0.913643 | `[0.884903, 0.942382]` |
| `ranking` | 12 | `basic` | 0.877378 | `[0.851179, 0.903578]` |
| `ranking` | 12 | `extended_td` | 0.912277 | `[0.867551, 0.957002]` |
| `full` | 24 | `basic` | 0.922004 | `[0.911428, 0.932579]` |
| `full` | 24 | `extended_td` | 0.954330 | `[0.946750, 0.961910]` |

Интерпретация:

- `24ch + extended_td` остается самой надежной конфигурацией и по one-shot medium, и по repeated splits;
- `greedy 12ch + extended_td` выглядит как рабочий компромисс, но его CI заметно ниже полного baseline;
- `ranking 12ch + extended_td` близок к `greedy 12ch + extended_td`, но разброс у него шире;
- `greedy 6ch` даже с `extended_td` уже слишком нестабилен для сильных формулировок в дипломе.

## 14. What Can Be Used in Diploma

### 14.1 Cautiously usable

Можно осторожно использовать:

- leakage-safe full channel sweep;
- leakage-safe medium channel sweep;
- leakage-safe medium feature sweep;
- repeated medium-like evaluation with CI;
- вывод, что после устранения leakage сильное сокращение каналов больше не выглядит почти бесплатным;
- вывод, что `extended_td` помогает, но не отменяет потери от сильного уменьшения числа каналов;
- вывод, что `ranking` стабильнее `greedy`;
- вывод, что filtering полезнее для very low-channel greedy setups, чем для full baseline.

### 14.2 Proof of concept only

Пока только proof of concept:

- quick results;
- POC hand visualization;
- offline bandit simulation;
- точный выбор "лучшего" reduced subset на основе greedy.

## 15. Problems and Resolutions

| Date | Problem | Cause | Resolution | Status |
|---|---|---|---|---|
| `2026-05-14` | `uv sync` was unreliable in current environment | cache and network restrictions | used local `.venv` and editable install | resolved |
| `2026-05-14` | test leakage in channel selection | ranking and greedy used final test split | introduced leakage-safe selection protocol | resolved |
| `2026-05-14` | mismatch between raw 32 signals and project 24-channel setup | dataset layout differs from project assumption | fixed project subset through `resolve_project_channel_indices()` | resolved |
| `2026-05-15` | `feature_sweep.py --medium` with custom `--output` used wrong reference CSV | medium mode remapped output but not the channel_sweep source in one branch | fixed CLI selection of `CHANNEL_SWEEP_MEDIUM_CSV` | resolved |
| `2026-05-15` | POC electrode schematic did not match requested `4 rings x 6` arrangement | initial layout was a rectangular placeholder | layout redrawn and old frame artifacts removed | resolved |

## 16. Commits

Коммиты, относящиеся к этой линии экспериментов:

- `chore: finalize project layout and docs`
- `fix: prevent test leakage in channel selection`
- `feat: add medium-scale channel and feature sweeps`
- `feat: compare raw and filtered channel sweeps`
- `feat: add channel selection stability analysis`
- `feat: add offline bandit simulation`
- `feat: add simple hand movement proof of concept`
- `feat: improve hand gesture poc visualization`
- `docs: format results log for diploma use`

## 17. Final Summary

### 17.1 Leakage checks performed

Выполненные проверки:

1. Прочитаны `channel_ranking.py`, `greedy_selection.py`, `channel_sweep.py`, `feature_sweep.py`.
2. Подтверждено, что старая версия использовала final test split для выбора каналов.
3. Внедрены leakage-safe participant splits.
4. Повторно сгенерированы quick CSV и plots после исправления.
5. Сгенерированы medium CSV и plots после исправления.
6. Сгенерированы full channel sweep CSV and plots.
7. Сгенерированы repeated medium-like CI tables.

### 17.2 Was there test leakage?

Да. В предыдущей реализации ranking и greedy использовали test set для выбора каналов. В текущей версии это исправлено, и `selection_uses_test=false` сохранено в новых CSV.

### 17.3 How did medium results change the old quick conclusion?

Изменение принципиальное:

- старый quick POC намекал, что 6-12 каналов почти не хуже 24;
- medium leakage-safe results это не подтвердили;
- full и repeated runs это подтвердили;
- наиболее надежная конфигурация сейчас - `24 channels`, особенно с `extended_td`.

### 17.4 Stable channels

Наиболее устойчиво повторяются каналы из ranking-based selection. Особенно стабильно появляются каналы `6`, `7`, `14`, `15`, а на больших наборах также `5` и `13`.

### 17.5 What changed after larger runs?

- `full 24ch` на leakage-safe full split дал `macro_f1=0.948098`;
- лучший reduced result на full split - `greedy 16ch`, `macro_f1=0.944289`;
- repeated CI закрепил `24ch + extended_td` как strongest baseline;
- `greedy 12ch + extended_td` можно рассматривать как осторожный компромисс, но не как замену full baseline.

### 17.6 Next run

Следующим запуском стоит:

1. Если нужен еще более строгий результат, сделать полный repeated evaluation с большим числом participant splits и без ручного ограничения конфигураций.
2. При желании проверить filtered variants не только для Stage 2, но и для `extended_td` configs.
3. Если будет доступна внешняя документация или figure из статьи GRABMyo, отдельно сверить schematic visualization с официальной схемой электродов.
4. Если понадобится ускорение, переносить на GPU имеет смысл только feature extraction, но не текущий LDA baseline как таковой.
