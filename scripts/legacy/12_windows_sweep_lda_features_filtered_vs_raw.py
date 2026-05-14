# %% Cell
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, f1_score

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")
OUT_CSV = Path("data/processed/window_sweep_filtered_vs_raw.csv")

# Sweep по размерам окна (мс)
WIN_MS_CANDIDATES = [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
HOP_FRACTION = 0.25

# Фильтрация (standard sEMG)
USE_FILTERING_MODES = [False, True]  # raw vs filtered
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
BP_ORDER = 4


# %% Cell
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


def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """
    x: (T, C)
    """
    # Notch 50 Hz
    w0 = NOTCH_HZ / (0.5 * fs)
    bN, aN = iirnotch(w0, Q=NOTCH_Q)
    x = filtfilt(bN, aN, x, axis=0)

    # Band-pass 20–450 Hz (upper bound protected by fs)
    low = BANDPASS_LOW / (0.5 * fs)
    high_hz = min(BANDPASS_HIGH, 0.45 * fs)
    high = high_hz / (0.5 * fs)
    sos = butter(BP_ORDER, [low, high], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0)

    return x


@lru_cache(maxsize=None)
def read_record(record_path: str) -> tuple[np.ndarray, float]:
    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    return sig.astype(np.float32), fs


@lru_cache(maxsize=None)
def read_record_filtered(record_path: str) -> tuple[np.ndarray, float, float]:
    """
    Возвращает:
      x_filt, fs, preprocess_time_ms (время фильтрации на весь record)
    """
    x, fs = read_record(record_path)
    t0 = time.perf_counter()
    x_f = preprocess_signal(x, fs)
    t1 = time.perf_counter()
    return x_f, fs, (t1 - t0) * 1000.0


def evaluate_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    win_ms: float,
    hop_ms: float,
    filtered: bool,
) -> dict[str, float | int | str | bool]:
    def build_xy(sub_df: pd.DataFrame):
        X_list = []
        y_list = []
        feat_times = []
        preprocess_times_ms = []

        for row in sub_df.itertuples(index=False):
            if filtered:
                x, fs, pre_ms = read_record_filtered(row.record_path)
                preprocess_times_ms.append(pre_ms)
            else:
                x, fs = read_record(row.record_path)

            for w in sliding_windows(x, fs, win_ms, hop_ms):
                t0 = time.perf_counter()
                feat = extract_td_features(w)
                t1 = time.perf_counter()
                X_list.append(feat)
                y_list.append(int(row.y))
                feat_times.append(t1 - t0)

        X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y, np.array(feat_times, dtype=np.float64), np.array(preprocess_times_ms, dtype=np.float64)

    X_train, y_train, t_train, pre_train = build_xy(train_df)
    X_test, y_test, t_test, pre_test = build_xy(test_df)

    if X_train.size == 0 or X_test.size == 0:
        return {
            "filtered": filtered,
            "win_ms": float(win_ms),
            "hop_ms": float(hop_ms),
            "status": "skipped_no_windows",
            "n_train_windows": int(X_train.shape[0]),
            "n_test_windows": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "feat_time_mean_ms": np.nan,
            "feat_time_p95_ms": np.nan,
            "preprocess_record_mean_ms": float(np.mean(pre_test)) if len(pre_test) else np.nan,
            "preprocess_record_p95_ms": float(np.quantile(pre_test, 0.95)) if len(pre_test) else np.nan,
        }

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro"))

    print(f"\n=== Mode={'FILTERED' if filtered else 'RAW'} | Window {win_ms:.1f} ms | Hop {hop_ms:.1f} ms ===")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("Report:\n", classification_report(y_test, pred, digits=4))

    feat_mean_ms = float(np.mean(t_test) * 1000.0) if len(t_test) else np.nan
    feat_p95_ms = float(np.quantile(t_test, 0.95) * 1000.0) if len(t_test) else np.nan

    pre_mean_ms = float(np.mean(pre_test)) if len(pre_test) else np.nan
    pre_p95_ms = float(np.quantile(pre_test, 0.95)) if len(pre_test) else np.nan

    if len(t_test):
        print("Feature extraction per window (test) mean ms:", feat_mean_ms)
        print("Feature extraction per window (test) p95  ms:", feat_p95_ms)
    if filtered and len(pre_test):
        print("Preprocess per record (test) mean ms:", pre_mean_ms)
        print("Preprocess per record (test) p95  ms:", pre_p95_ms)

    return {
        "filtered": filtered,
        "win_ms": float(win_ms),
        "hop_ms": float(hop_ms),
        "status": "ok",
        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "accuracy": acc,
        "macro_f1": f1,
        "feat_time_mean_ms": feat_mean_ms,
        "feat_time_p95_ms": feat_p95_ms,
        "preprocess_record_mean_ms": pre_mean_ms,
        "preprocess_record_p95_ms": pre_p95_ms,
        # минимальная алгоритмическая задержка = длина окна
        "algorithmic_delay_ms": float(win_ms),
    }


def main():
    print("Using index:", INDEX_CSV)
    df = pd.read_csv(INDEX_CSV)

    participants = sorted(df["participant"].unique())
    if len(participants) < 3:
        raise SystemExit("Слишком мало участников для нормального split. Нужны хотя бы 3.")

    test_participants = set(participants[-2:])
    train_participants = set(participants[:-2])

    print("Participants total:", len(participants))
    print("Train participants:", sorted(train_participants))
    print("Test participants:", sorted(test_participants))

    train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
    test_df = df[df["participant"].isin(test_participants)].reset_index(drop=True)

    print("Train records:", len(train_df), "Test records:", len(test_df))

    # Прогреваем кэш чтения (и фильтрации тоже, если включено)
    unique_paths = sorted(set(df["record_path"].tolist()))
    print("Unique records to cache:", len(unique_paths))
    for path in unique_paths:
        read_record(path)
    for path in unique_paths:
        # прогреем фильтр-кэш, чтобы sweep меньше страдал от повторной фильтрации
        # (это по сути "offline preprocessing")
        read_record_filtered(path)

    rows = []
    for filtered in USE_FILTERING_MODES:
        for win_ms in WIN_MS_CANDIDATES:
            hop_ms = win_ms * HOP_FRACTION
            rows.append(evaluate_window(train_df, test_df, win_ms, hop_ms, filtered))

    res_df = pd.DataFrame(rows).sort_values(["filtered", "win_ms"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_CSV, index=False)

    print("\n=== Summary ===")
    print(
        res_df[
            [
                "filtered",
                "win_ms",
                "hop_ms",
                "status",
                "n_train_windows",
                "n_test_windows",
                "accuracy",
                "macro_f1",
                "feat_time_mean_ms",
                "feat_time_p95_ms",
                "preprocess_record_mean_ms",
                "preprocess_record_p95_ms",
                "algorithmic_delay_ms",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
