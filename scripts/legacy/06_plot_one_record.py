# %% Cell
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import welch

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")

df = pd.read_csv(INDEX_CSV)

# Возьмём один пример, например WF (y=0)
row = df[df["y"] == 0].iloc[0]
record_path = row["record_path"]

sig, fields = wfdb.rdsamp(record_path)
fs = float(fields["fs"])
x = sig.astype(np.float32)

# Для графика возьмём 5 секунд и 2 канала
secs = 5
n = int(secs * fs)
t = np.arange(n) / fs
ch0, ch1 = 0, 1

plt.figure()
plt.plot(t, x[:n, ch0])
plt.title(f"Raw EMG (channel {ch0}) | fs={fs} Hz | class={row['class']}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# %% Cell
plt.figure()
plt.plot(t, x[:n, ch1])
plt.title(f"Raw EMG (channel {ch1}) | fs={fs} Hz | class={row['class']}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# %% Cell
# Спектр мощности (PSD) по Welch для одного канала
f, pxx = welch(x[:n, ch0], fs=fs, nperseg=2048)
plt.figure()
plt.semilogy(f, pxx)
plt.title(f"PSD Welch (channel {ch0}) | class={row['class']}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 600)
plt.tight_layout()
plt.show()
