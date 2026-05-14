"""Shared helpers for baseline-like experiments."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from emg_sampling.config import resolve_project_channel_indices
from emg_sampling.features.time_domain import extract_td_features, sliding_windows
from emg_sampling.models.lda_baseline import evaluate_classifier, train_lda
from emg_sampling.preprocessing.filters import preprocess_signal
from emg_sampling.data.grabmyo_loader import read_record


def build_feature_matrix(
    records_df: pd.DataFrame,
    win_ms: float,
    hop_fraction: float,
    filtered: bool = False,
    channel_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds feature matrix and labels from record index.

    Args:
        records_df: DataFrame with at least record_path and y.
        win_ms: Window length in milliseconds.
        hop_fraction: Fraction of window used as hop.
        filtered: If True applies preprocessing before windowing.
        channel_indices: Optional channel subset to keep before feature extraction.

    Returns:
        X: Feature matrix.
        y: Labels.
        feat_times_sec: Per-window feature extraction times.
        preprocess_times_ms: Per-record preprocess times (empty for raw mode).
    """
    if records_df.empty:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    feat_times: list[float] = []
    preprocess_times_ms: list[float] = []

    for row in records_df.itertuples(index=False):
        signal, fs = read_record(row.record_path)
        active_channels = (
            tuple(channel_indices)
            if channel_indices is not None
            else resolve_project_channel_indices(signal.shape[1])
        )
        signal = signal[:, active_channels]

        if filtered:
            t0 = time.perf_counter()
            signal = preprocess_signal(signal, fs=fs)
            t1 = time.perf_counter()
            preprocess_times_ms.append((t1 - t0) * 1000.0)

        hop_ms = win_ms * hop_fraction
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


def summarize_times(times: np.ndarray, unit_scale: float = 1.0) -> tuple[float, float]:
    """Returns mean and p95 for timing arrays."""
    if len(times) == 0:
        return np.nan, np.nan
    mean = float(np.mean(times) * unit_scale)
    p95 = float(np.quantile(times, 0.95) * unit_scale)
    return mean, p95


def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    win_ms: float,
    hop_fraction: float,
    filtered: bool,
    channel_indices: Sequence[int] | None = None,
) -> dict[str, float | int | str | bool]:
    """Runs one train/test evaluation for selected parameters."""
    X_train, y_train, _, _ = build_feature_matrix(
        train_df,
        win_ms=win_ms,
        hop_fraction=hop_fraction,
        filtered=filtered,
        channel_indices=channel_indices,
    )
    X_test, y_test, feat_times, preprocess_times_ms = build_feature_matrix(
        test_df,
        win_ms=win_ms,
        hop_fraction=hop_fraction,
        filtered=filtered,
        channel_indices=channel_indices,
    )

    row: dict[str, float | int | str | bool] = {
        "filtered": filtered,
        "win_ms": float(win_ms),
        "hop_ms": float(win_ms * hop_fraction),
        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
        "algorithmic_delay_ms": float(win_ms),
    }

    feat_mean_ms, feat_p95_ms = summarize_times(feat_times, unit_scale=1000.0)
    pre_mean_ms, pre_p95_ms = summarize_times(preprocess_times_ms, unit_scale=1.0)
    row["feat_time_mean_ms"] = feat_mean_ms
    row["feat_time_p95_ms"] = feat_p95_ms
    row["preprocess_record_mean_ms"] = pre_mean_ms
    row["preprocess_record_p95_ms"] = pre_p95_ms

    if X_train.size == 0 or X_test.size == 0:
        row["status"] = "skipped_no_windows"
        row["accuracy"] = np.nan
        row["macro_f1"] = np.nan
        return row

    clf = train_lda(X_train, y_train)
    metrics = evaluate_classifier(clf, X_test, y_test)

    row["status"] = "ok"
    row.update(metrics)
    return row
