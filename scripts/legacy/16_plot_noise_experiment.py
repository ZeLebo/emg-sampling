# %% Cell
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = Path("data/processed/noise_filtered_vs_raw_3windows.csv")

df = pd.read_csv(RESULTS_CSV).sort_values(["win_ms", "mode"]).reset_index(drop=True)

print(df.to_string(index=False))

# %% Cell
# График: Macro-F1 vs Delay для 3 режимов
plt.figure()

for mode in ["clean", "noisy_raw", "noisy_filtered"]:
    sub = df[df["mode"] == mode]
    plt.plot(sub["algorithmic_delay_ms"], sub["macro_f1"], marker="o", label=mode)

plt.title("Macro-F1 vs window (clean vs noisy vs noisy+filtered)")
plt.xlabel("Window (ms)  [algorithmic delay]")
plt.ylabel("Macro-F1")
plt.legend()
plt.tight_layout()
plt.show()

# %% Cell
# График: Accuracy vs Delay для 3 режимов
plt.figure()

for mode in ["clean", "noisy_raw", "noisy_filtered"]:
    sub = df[df["mode"] == mode]
    plt.plot(sub["algorithmic_delay_ms"], sub["accuracy"], marker="o", label=mode)

plt.title("Accuracy vs window (clean vs noisy vs noisy+filtered)")
plt.xlabel("Window (ms)  [algorithmic delay]")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
 