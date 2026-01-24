# %% Cell
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")

# Что визуализируем
CLASS_TO_VISUALIZE = 0   # 0=WF, 1=WE, 2=HO, 3=HC
CHANNEL = 0
SECONDS = 3.0            # сколько секунд показать (в начале записи)

# Параметры фильтрации
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
BP_ORDER = 4

# Параметры огибающей (не обязательно, но наглядно)
RECTIFY = True
ENVELOPE_LP_HZ = 10.0    # 5–10 Гц обычно достаточно

df = pd.read_csv(INDEX_CSV)
row = df[df["y"] == CLASS_TO_VISUALIZE].iloc[0]
record_path = row["record_path"]

sig, fields = wfdb.rdsamp(record_path)
fs = float(fields["fs"])
x = sig.astype(np.float32)

n = int(SECONDS * fs)
t = np.arange(n) / fs

x_raw = x[:n, CHANNEL]


# %% Cell
def preprocess_pipeline(x_1d: np.ndarray, fs: float):
    """
    Возвращает:
      notch_only, bandpass_only, notch_then_bandpass, rectified, envelope
    """
    # Notch
    w0 = NOTCH_HZ / (0.5 * fs)
    bN, aN = iirnotch(w0, Q=NOTCH_Q)
    x_notch = filtfilt(bN, aN, x_1d)

    # Band-pass (через SOS)
    low = BANDPASS_LOW / (0.5 * fs)
    high_hz = min(BANDPASS_HIGH, 0.45 * fs)
    high = high_hz / (0.5 * fs)
    sos = butter(BP_ORDER, [low, high], btype="bandpass", output="sos")
    x_bp = sosfiltfilt(sos, x_1d)

    # Notch -> Band-pass (типичный порядок)
    x_nb = sosfiltfilt(sos, x_notch)

    # Rectify (модуль)
    x_rect = np.abs(x_nb) if RECTIFY else x_nb

    # Envelope (LPF)
    if ENVELOPE_LP_HZ is not None and ENVELOPE_LP_HZ > 0:
        lp = ENVELOPE_LP_HZ / (0.5 * fs)
        sos_lp = butter(4, lp, btype="lowpass", output="sos")
        x_env = sosfiltfilt(sos_lp, x_rect)
    else:
        x_env = None

    return x_notch, x_bp, x_nb, x_rect, x_env


x_notch, x_bp, x_nb, x_rect, x_env = preprocess_pipeline(x_raw, fs)


# %% Cell
# 1) Временной сигнал: raw vs filtered (bandpass+notch)
plt.figure()
plt.plot(t, x_raw)
plt.plot(t, x_nb)
plt.title(f"Time domain: raw vs notch+bandpass | ch={CHANNEL} | class={row['class']} | fs={fs:.0f} Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["Raw", "Notch+Bandpass"])
plt.tight_layout()
plt.show()


# %% Cell
# 2) Показать, "что отрезали": residual = raw - filtered
residual = x_raw - x_nb

plt.figure()
plt.plot(t, residual * 10)
plt.title("Removed component (residual = raw - filtered)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()


# %% Cell
# 3) Наглядно про 50 Гц: notch-only vs raw (время)
plt.figure()
plt.plot(t, x_raw)
plt.plot(t, x_notch)
plt.title("Time domain: raw vs notch-only (50 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["Raw", "Notch only"])
plt.tight_layout()
plt.show()


# %% Cell
# 4) PSD: raw vs filtered + отдельно residual
f1, p1 = welch(x_raw, fs=fs, nperseg=2048)
f2, p2 = welch(x_nb, fs=fs, nperseg=2048)
f3, p3 = welch(residual, fs=fs, nperseg=2048)

plt.figure()
plt.semilogy(f1, p1)
plt.semilogy(f2, p2)
plt.semilogy(f3, p3)
plt.title("PSD Welch: raw vs filtered vs removed(residual)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 200)
plt.legend(["Raw", "Filtered", "Removed"])
plt.tight_layout()
plt.show()


# %% Cell
# 5) Rectified + envelope (то, что часто используют как "амплитудная огибающая")
plt.figure()
plt.plot(t, x_nb)
plt.plot(t, x_rect)
if x_env is not None:
    plt.plot(t, x_env)
    plt.legend(["Filtered", "Rectified |x|", f"Envelope LP {ENVELOPE_LP_HZ} Hz"])
else:
    plt.legend(["Filtered", "Rectified |x|"])

plt.title("Conditioning: filtered -> rectified -> envelope")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
