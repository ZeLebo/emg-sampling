"""Channel-count sweep for GRABMyo baseline experiments."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emg_sampling.config import DEFAULT_WINDOW_MS, HOP_FRACTION, N_TEST_PARTICIPANTS, resolve_project_channel_indices
from emg_sampling.data.grabmyo_loader import load_index, read_record
from emg_sampling.data.split import split_by_participants
from emg_sampling.experiments.common import build_feature_matrix
from emg_sampling.features.time_domain import BASIC_TD_FEATURE_NAMES, extract_td_features, get_feature_column_indices, sliding_windows
from emg_sampling.models.lda_baseline import evaluate_classifier, train_lda
from emg_sampling.paths import CHANNEL_SWEEP_CSV, PLOTS_DIR, ensure_project_dirs
from emg_sampling.preprocessing.filters import preprocess_signal
from emg_sampling.selection import greedy_channel_order, rank_channels_by_macro_f1, sample_random_channels

DEFAULT_CHANNEL_COUNTS = (3, 4, 6, 8, 12, 16, 24)
DEFAULT_METHODS = ("full", "random", "ranking", "greedy")
DEFAULT_RANDOM_REPEATS = 5
DEFAULT_MAX_GREEDY_CHANNELS = 16


def _participants_string(df: pd.DataFrame) -> str:
    return ",".join(str(participant) for participant in sorted(df["participant"].unique()))


def _make_quick_subset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a small but class-balanced subset for quick experiments."""
    quick_train = (
        train_df[train_df["participant"].isin(sorted(train_df["participant"].unique())[:4])]
        .groupby(["participant", "class"], as_index=False, sort=True)
        .head(2)
        .reset_index(drop=True)
    )
    quick_test = (
        test_df.groupby(["participant", "class"], as_index=False, sort=True)
        .head(2)
        .reset_index(drop=True)
    )
    return quick_train, quick_test


def _evaluate_feature_subset(
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_test_full: np.ndarray,
    y_test: np.ndarray,
    total_channels: int,
    selected_channels: Sequence[int],
) -> dict[str, float]:
    columns = get_feature_column_indices(
        list(selected_channels),
        total_channels=total_channels,
        feature_names=BASIC_TD_FEATURE_NAMES,
    )
    clf = train_lda(X_train_full[:, columns], y_train)
    return evaluate_classifier(clf, X_test_full[:, columns], y_test)


def _estimate_processing_time_ms(
    records_df: pd.DataFrame,
    win_ms: float,
    hop_fraction: float,
    filtered: bool,
    selected_channels: Sequence[int],
    max_records: int = 2,
    max_windows_per_record: int = 12,
) -> float:
    """Measures mean per-window feature extraction time on a small calibration subset."""
    samples: list[float] = []
    hop_ms = win_ms * hop_fraction

    for row in records_df.head(max_records).itertuples(index=False):
        signal, fs = read_record(row.record_path)
        if filtered:
            signal = preprocess_signal(signal, fs=fs)
        signal = signal[:, resolve_project_channel_indices(signal.shape[1])]
        signal = signal[:, selected_channels]

        for window_idx, window in enumerate(sliding_windows(signal, fs=fs, win_ms=win_ms, hop_ms=hop_ms)):
            t0 = time.perf_counter()
            extract_td_features(window)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)
            if window_idx + 1 >= max_windows_per_record:
                break

    if not samples:
        return math.nan
    return float(np.mean(samples))


def _plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path) -> None:
    plot_df = (
        df.groupby(["method", "channels_count"], as_index=False)[metric]
        .mean(numeric_only=True)
        .sort_values(["method", "channels_count"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in plot_df["method"].unique():
        method_df = plot_df[plot_df["method"] == method]
        ax.plot(method_df["channels_count"], method_df[metric], marker="o", linewidth=2, label=method)

    ax.set_title(title)
    ax.set_xlabel("Channels count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_channel_sweep_plots(df: pd.DataFrame, plots_dir: Path = PLOTS_DIR) -> list[Path]:
    """Saves the standard channel-sweep summary plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (plots_dir / "channel_sweep_macro_f1.png", "macro_f1", "Channel sweep - Macro-F1", "Macro-F1"),
        (plots_dir / "channel_sweep_accuracy.png", "accuracy", "Channel sweep - Accuracy", "Accuracy"),
        (
            plots_dir / "channel_sweep_feature_vector_size.png",
            "feature_vector_size",
            "Channel sweep - Feature vector size",
            "Features",
        ),
        (
            plots_dir / "channel_sweep_processing_time.png",
            "processing_time_ms",
            "Channel sweep - Processing time",
            "Processing time per window (ms)",
        ),
    ]
    for output_path, metric, title, ylabel in outputs:
        _plot_metric(df=df, metric=metric, title=title, ylabel=ylabel, output_path=output_path)
    return [output_path for output_path, *_rest in outputs]


def run_channel_sweep(
    index_csv: Path,
    win_ms: float = DEFAULT_WINDOW_MS,
    hop_fraction: float = HOP_FRACTION,
    filtered: bool = False,
    methods: Sequence[str] = DEFAULT_METHODS,
    channel_counts: Sequence[int] = DEFAULT_CHANNEL_COUNTS,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
    max_greedy_channels: int = DEFAULT_MAX_GREEDY_CHANNELS,
    quick: bool = False,
    output_csv: Path = CHANNEL_SWEEP_CSV,
) -> pd.DataFrame:
    """Runs channel-reduction sweep and saves the results table."""
    ensure_project_dirs()

    df = load_index(index_csv)
    train_df, test_df = split_by_participants(df, n_test_participants=N_TEST_PARTICIPANTS)
    if quick:
        train_df, test_df = _make_quick_subset(train_df, test_df)

    X_train_full, y_train, _, _ = build_feature_matrix(
        train_df,
        win_ms=win_ms,
        hop_fraction=hop_fraction,
        filtered=filtered,
    )
    X_test_full, y_test, _, _ = build_feature_matrix(
        test_df,
        win_ms=win_ms,
        hop_fraction=hop_fraction,
        filtered=filtered,
    )

    if X_train_full.size == 0 or X_test_full.size == 0:
        raise ValueError("Feature matrices are empty. Check the index or quick subset settings.")

    features_per_channel = len(BASIC_TD_FEATURE_NAMES)
    total_channels = X_train_full.shape[1] // features_per_channel
    if total_channels <= 0:
        raise ValueError("Could not infer total number of channels from feature matrix.")

    requested_counts = sorted({int(count) for count in channel_counts if 0 < int(count) <= total_channels})
    if not requested_counts:
        raise ValueError("No valid channel counts requested.")

    methods = tuple(dict.fromkeys(method.lower() for method in methods))
    valid_methods = set(DEFAULT_METHODS)
    unknown_methods = set(methods).difference(valid_methods)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    ranking_df = rank_channels_by_macro_f1(
        X_train=X_train_full,
        y_train=y_train,
        X_test=X_test_full,
        y_test=y_test,
        total_channels=total_channels,
    )
    ranking_order = ranking_df["channel_idx"].astype(int).tolist()

    greedy_df = greedy_channel_order(
        X_train=X_train_full,
        y_train=y_train,
        X_test=X_test_full,
        y_test=y_test,
        total_channels=total_channels,
        max_channels=min(max_greedy_channels, total_channels),
    )
    greedy_order = [int(channel) for channel in greedy_df["added_channel"].tolist()]

    records_count = int(len(train_df) + len(test_df))
    train_participants = _participants_string(train_df)
    test_participants = _participants_string(test_df)

    rows: list[dict[str, float | int | str | bool]] = []

    def append_row(method: str, channels_count: int, selected_channels: list[int], random_repeat: int | None = None) -> None:
        metrics = _evaluate_feature_subset(
            X_train_full=X_train_full,
            y_train=y_train,
            X_test_full=X_test_full,
            y_test=y_test,
            total_channels=total_channels,
            selected_channels=selected_channels,
        )
        processing_time_ms = _estimate_processing_time_ms(
            records_df=test_df,
            win_ms=win_ms,
            hop_fraction=hop_fraction,
            filtered=filtered,
            selected_channels=selected_channels,
        )
        row: dict[str, float | int | str | bool] = {
            "method": method,
            "channels_count": channels_count,
            "selected_channels": str(selected_channels),
            "window_ms": float(win_ms),
            "filtering": bool(filtered),
            "feature_set": "basic",
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "processing_time_ms": processing_time_ms,
            "feature_vector_size": int(channels_count * features_per_channel),
            "records_count": records_count,
            "train_participants": train_participants,
            "test_participants": test_participants,
            "quick_mode": bool(quick),
            "random_repeat": random_repeat if random_repeat is not None else -1,
        }
        rows.append(row)

    for channels_count in requested_counts:
        if "full" in methods and channels_count == total_channels:
            append_row("full", channels_count, list(range(total_channels)))

        if "random" in methods:
            repeats = random_repeats if channels_count < total_channels else 1
            for repeat_idx in range(repeats):
                selected = (
                    list(range(total_channels))
                    if channels_count == total_channels
                    else sample_random_channels(
                        total_channels=total_channels,
                        channels_count=channels_count,
                        random_state=1000 + channels_count * 10 + repeat_idx,
                    )
                )
                append_row("random", channels_count, selected, random_repeat=repeat_idx)

        if "ranking" in methods:
            append_row("ranking", channels_count, ranking_order[:channels_count])

        if "greedy" in methods and channels_count <= len(greedy_order):
            append_row("greedy", channels_count, greedy_order[:channels_count])

    out_df = pd.DataFrame(rows).sort_values(
        ["channels_count", "method", "random_repeat"],
        ascending=[True, True, True],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    save_channel_sweep_plots(out_df, plots_dir=PLOTS_DIR)
    return out_df.reset_index(drop=True)
