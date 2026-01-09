# %% Cell
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

# === Конфигурация ===
INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")
CLASS_TO_VISUALIZE = 0   # 0=WF, 1=WE, 2=HO, 3=HC
SECONDS_TO_PLOT = 5
CHANNEL = 0

# === Загрузка одного примера ===
df = pd.read_csv(INDEX_CSV)
row = df[df["y"] == CLASS_TO_VISUALIZE].iloc[0]
record_path = row["record_path"]

sig, fields = wfdb.rdsamp(record_path)
fs = float(fields["fs"])
x = sig.astype(np.float32)

n = int(SECONDS_TO_PLOT * fs)

# === Фильтрация ===
def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    # Notch 50 Hz
    w0 = 50.0 / (0.5 * fs)
    b, a = iirnotch(w0, Q=30)
    x = filtfilt(b, a, x, axis=0)

    # Band-pass 20–450 Hz
    low = 20.0 / (0.5 * fs)
    high_hz = min(450.0, 0.45 * fs)
    high = high_hz / (0.5 * fs)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0)

    return x

x_filt = preprocess_signal(x, fs)

# === PSD до/после ===
f_raw, pxx_raw = welch(x[:n, CHANNEL], fs=fs, nperseg=2048)
f_filt, pxx_filt = welch(x_filt[:n, CHANNEL], fs=fs, nperseg=2048)

plt.figure()
plt.semilogy(f_raw, pxx_raw)
plt.semilogy(f_filt, pxx_filt)
plt.title(f"PSD before vs after filtering | ch={CHANNEL} | class={row['class']}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 200)
plt.legend(["Raw", "Filtered"])
plt.tight_layout()
plt.show()
