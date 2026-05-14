# %% Cell
from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")
OUT_CSV = Path("data/processed/noise_filtered_vs_raw_3windows.csv")

# Окна, которые просил (150–250 мс)
WIN_MS_LIST = [150.0, 200.0, 250.0]
HOP_FRACTION = 0.25

# Режимы для сравнения:
# 1) clean (baseline) 2) noisy no filter 3) noisy + filter
MODES = ["clean", "noisy_raw", "noisy_filtered"]

# Шум (те же параметры, что в demo; при желании меняй)
BASE_SEED = 123
ADD_HUM_50HZ = True
HUM_AMPL = 0.15
ADD_DRIFT = True
DRIFT_AMPL = 0.20
DRIFT_HZ = 1.0
ADD_WHITE = True
WHITE_STD = 0.05
ADD_MOTION_BURST = True
BURST_STD = 0.25
BURST_PROB = 0.02

# Фильтр
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
BP_ORDER = 4


# %% Cell
def _stable_int_seed(s: str, base_seed: int) -> int:
    h = hashlib.sha256((s + str(base_seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def add_synthetic_noise(x: np.ndarray, fs: float, seed_key: str) -> np.ndarray:
    rng = np.random.RandomState(_stable_int_seed(seed_key, BASE_SEED))
    T, C = x.shape
    t = (np.arange(T, dtype=np.float32) / fs).reshape(-1, 1)
    y = x.copy()

    if ADD_HUM_50HZ:
        y = y + (HUM_AMPL * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)

    if ADD_DRIFT:
        phase = rng.uniform(0, 2*np.pi)
        y = y + (DRIFT_AMPL * np.sin(2 * np.pi * DRIFT_HZ * t + phase)).astype(np.float32)

    if ADD_WHITE:
        y = y + rng.normal(0.0, WHITE_STD, size=(T, C)).astype(np.float32)

    if ADD_MOTION_BURST:
        mask = (rng.rand(T, 1) < BURST_PROB).astype(np.float32)
        bursts = rng.normal(0.0, BURST_STD, size=(T, C)).astype(np.float32)
        y = y + mask * bursts

    return y


def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    w0 = NOTCH_HZ / (0.5 * fs)
    bN, aN = iirnotch(w0, Q=NOTCH_Q)
    x = filtfilt(bN, aN, x, axis=0)

    low = BANDPASS_LOW / (0.5 * fs)
    high_hz = min(BANDPASS_HIGH, 0.45 * fs)
    high = high_hz / (0.5 * fs)
    sos = butter(BP_ORDER, [low, high], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0)
    return x


def sliding_windows(x: np.ndarray, fs: float, win_ms: float, hop_ms: float):
    win = int(round(win_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))
    T = x.shape[0]
    for start in range(0, T - win + 1, hop):
        yield x[start : start + win]


def extract_td_features(w: np.ndarray) -> np.ndarray:
    eps = 1e-8
    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w**2, axis=0) + eps)
    wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
    s1 = w[:-1]
    s2 = w[1:]
    zc = np.sum((s1 * s2) < 0, axis=0)
    return np.concatenate([mav, rms, wl, zc], axis=0)


@lru_cache(maxsize=None)
def read_record(record_path: str) -> tuple[np.ndarray, float]:
    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    return sig.astype(np.float32), fs


def get_signal_for_mode(record_path: str, mode: str) -> tuple[np.ndarray, float, float]:
    """
    Возвращает (x, fs, preprocess_time_ms).
    preprocess_time_ms = время (noise+filter) на весь record (для справки).
    """
    x, fs = read_record(record_path)
    t0 = time.perf_counter()

    if mode == "clean":
        out = x
    elif mode == "noisy_raw":
        out = add_synthetic_noise(x, fs, seed_key=record_path)
    elif mode == "noisy_filtered":
        out = add_synthetic_noise(x, fs, seed_key=record_path)
        out = preprocess_signal(out, fs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t1 = time.perf_counter()
    return out, fs, (t1 - t0) * 1000.0


def build_xy(sub_df: pd.DataFrame, win_ms: float, hop_ms: float, mode: str):
    X_list, y_list = [], []
    feat_times = []
    preprocess_times = []

    for row in sub_df.itertuples(index=False):
        x, fs, pre_ms = get_signal_for_mode(row.record_path, mode)
        preprocess_times.append(pre_ms)

        for w in sliding_windows(x, fs, win_ms, hop_ms):
            t0 = time.perf_counter()
            feat = extract_td_features(w)
            t1 = time.perf_counter()

            X_list.append(feat)
            y_list.append(int(row.y))
            feat_times.append(t1 - t0)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y, np.array(feat_times, dtype=np.float64), np.array(preprocess_times, dtype=np.float64)


def eval_one(df: pd.DataFrame, win_ms: float, mode: str) -> dict:
    participants = sorted(df["participant"].unique())
    test_participants = set(participants[-2:])
    train_participants = set(participants[:-2])

    train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
    test_df = df[df["participant"].isin(test_participants)].reset_index(drop=True)

    hop_ms = win_ms * HOP_FRACTION

    X_train, y_train, t_train, pre_train = build_xy(train_df, win_ms, hop_ms, mode)
    X_test, y_test, t_test, pre_test = build_xy(test_df, win_ms, hop_ms, mode)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro"))

    feat_mean_ms = float(np.mean(t_test) * 1000.0)
    feat_p95_ms = float(np.quantile(t_test, 0.95) * 1000.0)

    pre_mean_ms = float(np.mean(pre_test)) if len(pre_test) else np.nan
    pre_p95_ms = float(np.quantile(pre_test, 0.95)) if len(pre_test) else np.nan

    return {
        "mode": mode,
        "win_ms": float(win_ms),
        "hop_ms": float(hop_ms),
        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "accuracy": acc,
        "macro_f1": f1,
        "feat_time_mean_ms": feat_mean_ms,
        "feat_time_p95_ms": feat_p95_ms,
        "preprocess_record_mean_ms": pre_mean_ms,
        "preprocess_record_p95_ms": pre_p95_ms,
        "algorithmic_delay_ms": float(win_ms),
        "total_delay_ms": float(win_ms) + feat_mean_ms,
    }


# %% Cell
df = pd.read_csv(INDEX_CSV)

rows = []
for win_ms in WIN_MS_LIST:
    for mode in MODES:
        print(f"Running: win={win_ms:.0f} ms | mode={mode}")
        rows.append(eval_one(df, win_ms, mode))

res = pd.DataFrame(rows).sort_values(["win_ms", "mode"]).reset_index(drop=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT_CSV, index=False)

print("\nSaved:", OUT_CSV)
print(res.to_string(index=False))
