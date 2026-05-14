"""Noise robustness experiment for baseline pipeline."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from emg_sampling.config import HOP_FRACTION, N_TEST_PARTICIPANTS, NOISE_BASE_SEED, NOISE_WINDOW_MS_CANDIDATES
from emg_sampling.data.grabmyo_loader import load_index, read_record
from emg_sampling.data.split import split_by_participants
from emg_sampling.features.time_domain import extract_td_features, sliding_windows
from emg_sampling.models.lda_baseline import evaluate_classifier, train_lda
from emg_sampling.paths import NOISE_EXPERIMENT_CSV, ensure_project_dirs
from emg_sampling.preprocessing.filters import preprocess_signal


def _stable_int_seed(seed_key: str, base_seed: int = NOISE_BASE_SEED) -> int:
    digest = hashlib.sha256((seed_key + str(base_seed)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def add_synthetic_noise(x: np.ndarray, fs: float, seed_key: str) -> np.ndarray:
    """Adds deterministic synthetic noise to a signal for robustness checks."""
    rng = np.random.RandomState(_stable_int_seed(seed_key))
    signal = x.astype(np.float32, copy=True)
    samples, channels = signal.shape
    t = (np.arange(samples, dtype=np.float32) / fs).reshape(-1, 1)

    signal += (0.15 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)

    drift_phase = rng.uniform(0.0, 2.0 * np.pi)
    signal += (0.20 * np.sin(2 * np.pi * 1.0 * t + drift_phase)).astype(np.float32)

    signal += rng.normal(0.0, 0.05, size=(samples, channels)).astype(np.float32)

    burst_mask = (rng.rand(samples, 1) < 0.02).astype(np.float32)
    burst_values = rng.normal(0.0, 0.25, size=(samples, channels)).astype(np.float32)
    signal += burst_mask * burst_values

    return signal


def _signal_for_mode(record_path: str, mode: str) -> tuple[np.ndarray, float, float]:
    """Returns signal, sampling rate and preprocessing time in ms."""
    signal, fs = read_record(record_path)
    t0 = time.perf_counter()

    if mode == "clean":
        out = signal
    elif mode == "noisy_raw":
        out = add_synthetic_noise(signal, fs=fs, seed_key=record_path)
    elif mode == "noisy_filtered":
        out = add_synthetic_noise(signal, fs=fs, seed_key=record_path)
        out = preprocess_signal(out, fs=fs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t1 = time.perf_counter()
    return out, fs, (t1 - t0) * 1000.0


def _build_xy(
    records_df: pd.DataFrame,
    win_ms: float,
    hop_fraction: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    feat_times: list[float] = []
    preprocess_times_ms: list[float] = []

    hop_ms = win_ms * hop_fraction
    for row in records_df.itertuples(index=False):
        signal, fs, pre_ms = _signal_for_mode(row.record_path, mode)
        preprocess_times_ms.append(pre_ms)

        for window in sliding_windows(signal, fs=fs, win_ms=win_ms, hop_ms=hop_ms):
            t0 = time.perf_counter()
            features = extract_td_features(window)
            t1 = time.perf_counter()

            X_list.append(features)
            y_list.append(int(row.y))
            feat_times.append(t1 - t0)

    X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return (
        X,
        y,
        np.array(feat_times, dtype=np.float64),
        np.array(preprocess_times_ms, dtype=np.float64),
    )


def _summarize_times(times: np.ndarray, scale: float = 1.0) -> tuple[float, float]:
    if len(times) == 0:
        return np.nan, np.nan
    return float(np.mean(times) * scale), float(np.quantile(times, 0.95) * scale)


def run_noise_experiment(
    index_csv: Path,
    win_ms_candidates: list[float] | None = None,
    hop_fraction: float = HOP_FRACTION,
) -> pd.DataFrame:
    """Runs clean/noisy/noisy+filtered comparison over selected windows."""
    ensure_project_dirs()

    if win_ms_candidates is None:
        win_ms_candidates = NOISE_WINDOW_MS_CANDIDATES

    df = load_index(index_csv)
    train_df, test_df = split_by_participants(df, n_test_participants=N_TEST_PARTICIPANTS)

    rows: list[dict[str, float | int | str]] = []
    for win_ms in win_ms_candidates:
        for mode in ("clean", "noisy_raw", "noisy_filtered"):
            X_train, y_train, _, _ = _build_xy(train_df, win_ms, hop_fraction, mode)
            X_test, y_test, feat_times, preprocess_times_ms = _build_xy(
                test_df,
                win_ms,
                hop_fraction,
                mode,
            )

            row: dict[str, float | int | str] = {
                "mode": mode,
                "win_ms": float(win_ms),
                "hop_ms": float(win_ms * hop_fraction),
                "n_train_windows": int(X_train.shape[0]),
                "n_test_windows": int(X_test.shape[0]),
                "algorithmic_delay_ms": float(win_ms),
            }

            feat_mean_ms, feat_p95_ms = _summarize_times(feat_times, scale=1000.0)
            row["feat_time_mean_ms"] = feat_mean_ms
            row["feat_time_p95_ms"] = feat_p95_ms

            pre_mean_ms, pre_p95_ms = _summarize_times(preprocess_times_ms, scale=1.0)
            row["preprocess_record_mean_ms"] = pre_mean_ms
            row["preprocess_record_p95_ms"] = pre_p95_ms

            if X_train.size == 0 or X_test.size == 0:
                row["status"] = "skipped_no_windows"
                row["accuracy"] = np.nan
                row["macro_f1"] = np.nan
            else:
                clf = train_lda(X_train, y_train)
                metrics = evaluate_classifier(clf, X_test, y_test)
                row["status"] = "ok"
                row.update(metrics)

            rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["win_ms", "mode"]).reset_index(drop=True)
    out_df.to_csv(NOISE_EXPERIMENT_CSV, index=False)
    return out_df
