# %% Cell
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = Path("data/processed/window_sweep_filtered_vs_raw.csv")

# Подсветка зоны, которую хочешь "как оптимум"
TARGET_MIN_MS = 150.0
TARGET_MAX_MS = 200.0

# Что считать задержкой:
# - algorithmic_delay_ms = win_ms
# - total_delay_ms = win_ms + feat_time_mean_ms (и при желании + preprocess, но preprocess у тебя на record)
USE_TOTAL_DELAY = True

df = pd.read_csv(RESULTS_CSV)
df = df[df["status"] == "ok"].copy()
df = df.sort_values(["filtered", "win_ms"]).reset_index(drop=True)

df["algorithmic_delay_ms"] = df["win_ms"]
df["total_delay_ms"] = df["win_ms"] + df["feat_time_mean_ms"].fillna(0.0)

delay_col = "total_delay_ms" if USE_TOTAL_DELAY else "algorithmic_delay_ms"

raw_df = df[df["filtered"] == False].copy()
flt_df = df[df["filtered"] == True].copy()

# Лучшие точки по macro-F1 для каждого режима
best_raw = raw_df.sort_values(["macro_f1", "accuracy", "win_ms"], ascending=[False, False, True]).iloc[0]
best_flt = flt_df.sort_values(["macro_f1", "accuracy", "win_ms"], ascending=[False, False, True]).iloc[0]

print("Loaded:", RESULTS_CSV)
print("\nRAW best:", f"win={best_raw['win_ms']:.0f} ms", f"macro_f1={best_raw['macro_f1']:.4f}", f"acc={best_raw['accuracy']:.4f}")
print("FLT best:", f"win={best_flt['win_ms']:.0f} ms", f"macro_f1={best_flt['macro_f1']:.4f}", f"acc={best_flt['accuracy']:.4f}")


# %% Cell
# График: Delay vs Macro-F1 (две кривые)
plt.figure()
plt.plot(raw_df[delay_col], raw_df["macro_f1"], marker="o")
plt.plot(flt_df[delay_col], flt_df["macro_f1"], marker="o")

# Подсветка целевой зоны по задержке
plt.axvspan(TARGET_MIN_MS, TARGET_MAX_MS, alpha=0.2)

# Отметим лучшие точки
plt.scatter([best_raw[delay_col]], [best_raw["macro_f1"]], s=120)
plt.scatter([best_flt[delay_col]], [best_flt["macro_f1"]], s=120)

plt.title("Delay vs Macro-F1: RAW vs FILTERED")
plt.xlabel("Delay (ms)  [window (+ feature time if enabled)]")
plt.ylabel("Macro-F1")
plt.legend(["RAW", "FILTERED", "target 150–200 ms", "best RAW", "best FILTERED"], loc="lower right")
plt.tight_layout()
plt.show()


# %% Cell
# Доп. график: Delay vs Accuracy (две кривые)
plt.figure()
plt.plot(raw_df[delay_col], raw_df["accuracy"], marker="o")
plt.plot(flt_df[delay_col], flt_df["accuracy"], marker="o")
plt.axvspan(TARGET_MIN_MS, TARGET_MAX_MS, alpha=0.2)

plt.scatter([best_raw[delay_col]], [best_raw["accuracy"]], s=120)
plt.scatter([best_flt[delay_col]], [best_flt["accuracy"]], s=120)

plt.title("Delay vs Accuracy: RAW vs FILTERED")
plt.xlabel("Delay (ms)  [window (+ feature time if enabled)]")
plt.ylabel("Accuracy")
plt.legend(["RAW", "FILTERED", "target 150–200 ms", "best RAW", "best FILTERED"], loc="lower right")
plt.tight_layout()
plt.show()


# %% Cell
# Таблица для отчёта (в консоль)
cols = [
    "filtered",
    "win_ms",
    "hop_ms",
    "accuracy",
    "macro_f1",
    "feat_time_mean_ms",
    "feat_time_p95_ms",
    "preprocess_record_mean_ms",
    "preprocess_record_p95_ms",
    "algorithmic_delay_ms",
    "total_delay_ms",
]
print(df[cols].to_string(index=False))
