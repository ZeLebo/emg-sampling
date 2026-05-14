"""Repeated medium-like participant splits for uncertainty estimates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emg_sampling.config import DEFAULT_WINDOW_MS, HOP_FRACTION
from emg_sampling.data.grabmyo_loader import load_index
from emg_sampling.experiments.common import evaluate_split
from emg_sampling.experiments.channel_sweep import _evaluate_feature_subset
from emg_sampling.experiments.common import build_feature_matrix
from emg_sampling.features.time_domain import BASIC_TD_FEATURE_NAMES
from emg_sampling.paths import PLOTS_DIR, REPEATED_MEDIUM_EVALUATION_CSV, ensure_project_dirs
from emg_sampling.selection import greedy_channel_order, rank_channels_by_macro_f1


@dataclass(frozen=True)
class RepeatedSplit:
    name: str
    selection_train: tuple[int, ...]
    selection_val: tuple[int, ...]
    final_test: tuple[int, ...]


DEFAULT_SPLITS = (
    RepeatedSplit("medium_ref", tuple(range(1, 13)), tuple(range(13, 17)), tuple(range(40, 44))),
    RepeatedSplit("fold_b", tuple(range(5, 17)), tuple(range(17, 21)), tuple(range(21, 25))),
    RepeatedSplit("fold_c", tuple(range(9, 21)), tuple(range(21, 25)), tuple(range(25, 29))),
    RepeatedSplit("fold_d", tuple(range(13, 25)), tuple(range(25, 29)), tuple(range(29, 33))),
    RepeatedSplit("fold_e", tuple(range(17, 29)), tuple(range(29, 33)), tuple(range(33, 37))),
)
DEFAULT_METHODS = ("greedy", "ranking", "full")
DEFAULT_CHANNEL_COUNTS = (6, 12, 24)
DEFAULT_FEATURE_SETS = ("basic", "extended_td")


def _participants_string(participants: Sequence[int]) -> str:
    return ",".join(str(participant) for participant in participants)


def _subset_records(df: pd.DataFrame, participants: Sequence[int]) -> pd.DataFrame:
    return df[df["participant"].isin(participants)].reset_index(drop=True)


def _z95_interval(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    if arr.size == 1:
        return (mean, mean)
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    delta = 1.96 * sem
    return (mean - delta, mean + delta)


def _summarize_config(group_df: pd.DataFrame) -> dict[str, object]:
    macro_values = group_df["macro_f1"].astype(float).to_numpy()
    accuracy_values = group_df["accuracy"].astype(float).to_numpy()
    macro_ci_low, macro_ci_high = _z95_interval(macro_values)
    accuracy_ci_low, accuracy_ci_high = _z95_interval(accuracy_values)
    return {
        "row_type": "summary",
        "split_name": "all",
        "method": group_df["method"].iloc[0],
        "channels_count": int(group_df["channels_count"].iloc[0]),
        "feature_set": group_df["feature_set"].iloc[0],
        "n_repeats": int(len(group_df)),
        "macro_f1_mean": float(np.mean(macro_values)),
        "macro_f1_std": float(np.std(macro_values, ddof=1)) if len(group_df) > 1 else 0.0,
        "macro_f1_ci95_low": macro_ci_low,
        "macro_f1_ci95_high": macro_ci_high,
        "accuracy_mean": float(np.mean(accuracy_values)),
        "accuracy_std": float(np.std(accuracy_values, ddof=1)) if len(group_df) > 1 else 0.0,
        "accuracy_ci95_low": accuracy_ci_low,
        "accuracy_ci95_high": accuracy_ci_high,
        "selected_channels": "",
        "selection_train_participants": "",
        "selection_val_participants": "",
        "final_test_participants": "",
    }


def save_repeated_medium_plot(df: pd.DataFrame, output_path: Path) -> None:
    summary_df = df[df["row_type"] == "summary"].copy()
    summary_df["label"] = (
        summary_df["method"]
        + "-"
        + summary_df["channels_count"].astype(str)
        + "ch-"
        + summary_df["feature_set"]
    )
    summary_df = summary_df.sort_values(["channels_count", "method", "feature_set"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(summary_df), dtype=np.float64)
    y = summary_df["macro_f1_mean"].to_numpy(dtype=np.float64)
    yerr = np.vstack(
        [
            y - summary_df["macro_f1_ci95_low"].to_numpy(dtype=np.float64),
            summary_df["macro_f1_ci95_high"].to_numpy(dtype=np.float64) - y,
        ]
    )
    ax.bar(x, y, color="#5c7cfa", alpha=0.85)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#1c2541", elinewidth=1.4, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["label"], rotation=35, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Repeated medium-like evaluation - Macro-F1 mean with 95% CI")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_repeated_medium_evaluation(
    index_csv: Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    channel_counts: Sequence[int] = DEFAULT_CHANNEL_COUNTS,
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
    win_ms: float = DEFAULT_WINDOW_MS,
    hop_fraction: float = HOP_FRACTION,
    filtered: bool = False,
    splits: Sequence[RepeatedSplit] = DEFAULT_SPLITS,
    output_csv: Path = REPEATED_MEDIUM_EVALUATION_CSV,
) -> pd.DataFrame:
    ensure_project_dirs()
    df = load_index(index_csv)
    methods = tuple(dict.fromkeys(methods))
    channel_counts = tuple(sorted({int(count) for count in channel_counts}))
    rows: list[dict[str, object]] = []

    for split in splits:
        selection_train_df = _subset_records(df, split.selection_train)
        selection_val_df = _subset_records(df, split.selection_val)
        final_train_df = pd.concat([selection_train_df, selection_val_df], ignore_index=True)
        final_test_df = _subset_records(df, split.final_test)

        X_selection_train, y_selection_train, _, _ = build_feature_matrix(
            selection_train_df,
            win_ms=win_ms,
            hop_fraction=hop_fraction,
            filtered=filtered,
            feature_set="basic",
        )
        X_selection_val, y_selection_val, _, _ = build_feature_matrix(
            selection_val_df,
            win_ms=win_ms,
            hop_fraction=hop_fraction,
            filtered=filtered,
            feature_set="basic",
        )

        total_channels = X_selection_train.shape[1] // len(BASIC_TD_FEATURE_NAMES)
        ranking_df = rank_channels_by_macro_f1(
            X_train=X_selection_train,
            y_train=y_selection_train,
            X_test=X_selection_val,
            y_test=y_selection_val,
            total_channels=total_channels,
        )
        greedy_df = greedy_channel_order(
            X_train=X_selection_train,
            y_train=y_selection_train,
            X_test=X_selection_val,
            y_test=y_selection_val,
            total_channels=total_channels,
            max_channels=max(channel_counts),
        )
        ranking_order = ranking_df["channel_idx"].astype(int).tolist()
        greedy_order = greedy_df["added_channel"].astype(int).tolist()

        for method in methods:
            for channels_count in channel_counts:
                if channels_count > total_channels:
                    continue
                if method == "full":
                    if channels_count != total_channels:
                        continue
                    selected_channels = list(range(total_channels))
                elif method == "ranking":
                    selected_channels = ranking_order[:channels_count]
                elif method == "greedy":
                    if channels_count > len(greedy_order):
                        continue
                    selected_channels = greedy_order[:channels_count]
                else:
                    raise ValueError(f"Unknown method: {method}")

                selection_metrics = _evaluate_feature_subset(
                    X_train_full=X_selection_train,
                    y_train=y_selection_train,
                    X_test_full=X_selection_val,
                    y_test=y_selection_val,
                    total_channels=total_channels,
                    selected_channels=selected_channels,
                )

                for feature_set in feature_sets:
                    eval_row = evaluate_split(
                        train_df=final_train_df,
                        test_df=final_test_df,
                        win_ms=win_ms,
                        hop_fraction=hop_fraction,
                        filtered=filtered,
                        channel_indices=selected_channels,
                        feature_set=feature_set,
                    )
                    rows.append(
                        {
                            "row_type": "detail",
                            "split_name": split.name,
                            "method": method,
                            "channels_count": channels_count,
                            "feature_set": feature_set,
                            "accuracy": float(eval_row["accuracy"]),
                            "macro_f1": float(eval_row["macro_f1"]),
                            "selected_channels": str(selected_channels),
                            "selection_score_accuracy": float(selection_metrics["accuracy"]),
                            "selection_score_macro_f1": float(selection_metrics["macro_f1"]),
                            "selection_train_participants": _participants_string(split.selection_train),
                            "selection_val_participants": _participants_string(split.selection_val),
                            "final_test_participants": _participants_string(split.final_test),
                        }
                    )

    detail_df = pd.DataFrame(rows)
    summary_rows = [
        _summarize_config(group_df)
        for _, group_df in detail_df.groupby(["method", "channels_count", "feature_set"], sort=True)
    ]
    out_df = pd.concat([detail_df, pd.DataFrame(summary_rows)], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    save_repeated_medium_plot(out_df, PLOTS_DIR / "repeated_medium_macro_f1_ci.png")
    return out_df
