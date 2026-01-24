# %% Cell
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Результаты sweep ты уже сохраняешь сюда. :contentReference[oaicite:0]{index=0}
RESULTS_CSV = Path("data/processed/window_sweep_results.csv")

# Что считаем "задержкой"
# 1) algorithmic_delay_ms = win_ms (минимальная задержка из-за накопления окна)
# 2) total_delay_ms = win_ms + feat_time_mean_ms (окно + вычисление фич на CPU)
USE_TOTAL_DELAY = True

# Хотим подчеркнуть "желательную зону" окна
TARGET_MIN_MS = 150.0
TARGET_MAX_MS = 200.0

df = pd.read_csv(RESULTS_CSV)
df = df[df["status"] == "ok"].copy()
df = df.sort_values("win_ms").reset_index(drop=True)

df["algorithmic_delay_ms"] = df["win_ms"]
df["total_delay_ms"] = df["win_ms"] + df["feat_time_mean_ms"].fillna(0.0)

delay_col = "total_delay_ms" if USE_TOTAL_DELAY else "algorithmic_delay_ms"

# Оптимум "по качеству" (макс macro_f1)
best = df.sort_values(["macro_f1", "accuracy", "win_ms"], ascending=[False, False, True]).iloc[0]

# Компромисс: минимальное окно, которое в пределах delta по macro_f1 от лучшего
delta = 0.01
near_best = df[df["macro_f1"] >= float(best["macro_f1"]) - delta]
tradeoff = near_best.sort_values(["win_ms", "accuracy"], ascending=[True, False]).iloc[0]

print("Loaded:", RESULTS_CSV)
print(df[["win_ms", "hop_ms", "accuracy", "macro_f1", "feat_time_mean_ms", "feat_time_p95_ms", delay_col]].to_string(index=False))

print("\nBest by quality:")
print(f"  win={best['win_ms']:.0f} ms | macro_f1={best['macro_f1']:.4f} | acc={best['accuracy']:.4f} | delay={best[delay_col]:.3f} ms")

print(f"\nTrade-off (smallest win within {delta:.2f} macro-F1 of best):")
print(f"  win={tradeoff['win_ms']:.0f} ms | macro_f1={tradeoff['macro_f1']:.4f} | acc={tradeoff['accuracy']:.4f} | delay={tradeoff[delay_col]:.3f} ms")


# %% Cell
# График 1: macro-F1 и accuracy vs delay
plt.figure()

plt.plot(df[delay_col], df["macro_f1"], marker="o")
plt.plot(df[delay_col], df["accuracy"], marker="o")

# Подсветка целевой зоны по окну (150–200 мс)
# Рисуем как вертикальную "полосу" по delay, приблизительно соответствующую win_ms диапазону.
# Если USE_TOTAL_DELAY=True, то delay чуть больше win_ms на доли мс, но для визуализации это неважно.
zmin = float(df.loc[df["win_ms"] == TARGET_MIN_MS, delay_col].iloc[0]) if (df["win_ms"] == TARGET_MIN_MS).any() else TARGET_MIN_MS
zmax = float(df.loc[df["win_ms"] == TARGET_MAX_MS, delay_col].iloc[0]) if (df["win_ms"] == TARGET_MAX_MS).any() else TARGET_MAX_MS
plt.axvspan(zmin, zmax, alpha=0.2)

# Маркеры оптимумов
plt.scatter([best[delay_col]], [best["macro_f1"]], s=120)
plt.scatter([tradeoff[delay_col]], [tradeoff["macro_f1"]], s=120)

plt.title("Quality vs delay (window sweep)")
plt.xlabel("Delay (ms)  [window (+ feature time if enabled)]")
plt.ylabel("Score")
plt.legend(["macro-F1", "accuracy", "target 150–200 ms", "best macro-F1", f"trade-off (Δ={delta})"], loc="lower right")
plt.tight_layout()
plt.show()


# %% Cell
# График 2: Pareto-like scatter (delay vs macro-F1) + подписи окон
plt.figure()
plt.scatter(df[delay_col], df["macro_f1"])

for r in df.itertuples(index=False):
    plt.annotate(f"{int(r.win_ms)}", (getattr(r, delay_col), r.macro_f1), textcoords="offset points", xytext=(5, 4))

plt.axvspan(zmin, zmax, alpha=0.2)
plt.scatter([best[delay_col]], [best["macro_f1"]], s=120)
plt.scatter([tradeoff[delay_col]], [tradeoff["macro_f1"]], s=120)

plt.title("Delay vs macro-F1 (annotated by window size)")
plt.xlabel("Delay (ms)")
plt.ylabel("Macro-F1")
plt.tight_layout()
plt.show()
