# %% Cell
from __future__ import annotations

from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import welch, butter, filtfilt, iirnotch, sosfiltfilt

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")

# Что визуализировать
CLASS_TO_VISUALIZE = 0   # 0=WF, 1=WE, 2=HO, 3=HC
CHANNEL = 0
SECONDS = 3.0

# Шум (можно крутить)
BASE_SEED = 123
ADD_HUM_50HZ = True
HUM_AMPL = 0.15          # амплитуда 50 Гц синуса (в долях типичной амплитуды EMG)
ADD_DRIFT = True
DRIFT_AMPL = 0.20        # амплитуда дрейфа (низкочастотный синус)
DRIFT_HZ = 1.0           # частота дрейфа, Гц
ADD_WHITE = True
WHITE_STD = 0.05         # std белого шума
ADD_MOTION_BURST = True
BURST_STD = 0.25         # std всплесков (артефакты движения)
BURST_PROB = 0.02        # вероятность всплеска на сэмпл (примерно)

# Фильтр
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
BP_ORDER = 4


# %% Cell
def _stable_int_seed(s: str, base_seed: int) -> int:
    h = hashlib.sha256((s + str(base_seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit


def add_synthetic_noise(x: np.ndarray, fs: float, seed_key: str) -> np.ndarray:
    """
    x: (T, C) float32
    Возвращает x_noisy того же размера.
    Шум детерминированный на запись (по seed_key), чтобы было воспроизводимо.
    """
    rng = np.random.RandomState(_stable_int_seed(seed_key, BASE_SEED))
    T, C = x.shape
    t = (np.arange(T, dtype=np.float32) / fs).reshape(-1, 1)

    y = x.copy()

    if ADD_HUM_50HZ:
        hum = HUM_AMPL * np.sin(2 * np.pi * 50.0 * t)
        y = y + hum.astype(np.float32)

    if ADD_DRIFT:
        drift = DRIFT_AMPL * np.sin(2 * np.pi * DRIFT_HZ * t + rng.uniform(0, 2*np.pi))
        y = y + drift.astype(np.float32)

    if ADD_WHITE:
        y = y + (rng.normal(0.0, WHITE_STD, size=(T, C)).astype(np.float32))

    if ADD_MOTION_BURST:
        # редкие большие импульсы (артефакты движения)
        mask = rng.rand(T, 1) < BURST_PROB
        bursts = rng.normal(0.0, BURST_STD, size=(T, C)).astype(np.float32)
        y = y + (mask.astype(np.float32) * bursts)

    return y


def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    # Notch 50 Hz
    w0 = NOTCH_HZ / (0.5 * fs)
    bN, aN = iirnotch(w0, Q=NOTCH_Q)
    x = filtfilt(bN, aN, x, axis=0)

    # Band-pass
    low = BANDPASS_LOW / (0.5 * fs)
    high_hz = min(BANDPASS_HIGH, 0.45 * fs)
    high = high_hz / (0.5 * fs)
    sos = butter(BP_ORDER, [low, high], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0)

    return x


# %% Cell
df = pd.read_csv(INDEX_CSV)
row = df[df["y"] == CLASS_TO_VISUALIZE].iloc[0]
record_path = row["record_path"]

sig, fields = wfdb.rdsamp(record_path)
fs = float(fields["fs"])
x = sig.astype(np.float32)

n = int(SECONDS * fs)
t = np.arange(n) / fs

x_clean = x[:n].copy()
x_noisy = add_synthetic_noise(x_clean, fs, seed_key=record_path)
x_noisy_filt = preprocess_signal(x_noisy, fs)

x_clean_ch = x_clean[:, CHANNEL]
x_noisy_ch = x_noisy[:, CHANNEL]
x_noisy_filt_ch = x_noisy_filt[:, CHANNEL]


# %% Cell
# Время: clean vs noisy vs noisy+filtered
plt.figure()
plt.plot(t, x_clean_ch)
plt.plot(t, x_noisy_ch)
plt.plot(t, x_noisy_filt_ch)
plt.title(f"Time domain | class={row['class']} | ch={CHANNEL} | fs={fs:.0f} Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["Clean", "Noisy (synthetic)", "Noisy + Filtered"])
plt.tight_layout()
plt.show()


# %% Cell
# PSD: clean vs noisy vs noisy+filtered
f1, p1 = welch(x_clean_ch, fs=fs, nperseg=2048)
f2, p2 = welch(x_noisy_ch, fs=fs, nperseg=2048)
f3, p3 = welch(x_noisy_filt_ch, fs=fs, nperseg=2048)

plt.figure()
plt.semilogy(f1, p1)
plt.semilogy(f2, p2)
plt.semilogy(f3, p3)
plt.title("PSD Welch (0–200 Hz): clean vs noisy vs noisy+filtered")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 200)
plt.legend(["Clean", "Noisy", "Noisy+Filtered"])
plt.tight_layout()
plt.show()
