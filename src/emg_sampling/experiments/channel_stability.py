"""Stability analysis for channel selection methods."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emg_sampling.data.grabmyo_loader import load_index
from emg_sampling.experiments.common import build_feature_matrix
from emg_sampling.features.time_domain import BASIC_TD_FEATURE_NAMES, get_feature_column_indices
from emg_sampling.models.lda_baseline import evaluate_classifier, train_lda
from emg_sampling.selection import greedy_channel_order, rank_channels_by_macro_f1
from emg_sampling.paths import CHANNEL_STABILITY_CSV, PLOTS_DIR, ensure_project_dirs

DEFAULT_STABILITY_COUNTS = (3, 6, 8, 12)
DEFAULT_STABILITY_METHODS = ("ranking", "greedy")


def _fold_definitions() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [
        (tuple(range(5, 17)), (1, 2, 3, 4)),
        ((1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16), (5, 6, 7, 8)),
        ((1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16), (9, 10, 11, 12)),
        (tuple(range(1, 13)), (13, 14, 15, 16)),
    ]


def _participants_string(participants: Sequence[int]) -> str:
    return ",".join(str(participant) for participant in participants)


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / len(set_a | set_b)


def save_channel_stability_plot(df: pd.DataFrame, output_path: Path) -> None:
    pairwise_df = df[df["row_type"] == "pairwise"].copy()
    summary_df = (
        pairwise_df.groupby(["method", "channels_count"], as_index=False)["jaccard"]
        .mean(numeric_only=True)
        .sort_values(["method", "channels_count"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary_df["method"].unique():
        method_df = summary_df[summary_df["method"] == method]
        ax.plot(method_df["channels_count"], method_df["jaccard"], marker="o", linewidth=2, label=method)

    ax.set_title("Channel selection stability - mean Jaccard")
    ax.set_xlabel("Channels count")
    ax.set_ylabel("Mean Jaccard")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _score_selected_channels(
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    selected_channels: Sequence[int],
) -> dict[str, float]:
    total_channels = X_train.shape[1] // len(BASIC_TD_FEATURE_NAMES)
    columns = get_feature_column_indices(
        list(selected_channels),
        total_channels=total_channels,
        feature_names=BASIC_TD_FEATURE_NAMES,
    )
    clf = train_lda(X_train[:, columns], y_train)
    return evaluate_classifier(clf, X_val[:, columns], y_val)


def run_channel_stability(
    index_csv: Path,
    channel_counts: Sequence[int] = DEFAULT_STABILITY_COUNTS,
    methods: Sequence[str] = DEFAULT_STABILITY_METHODS,
    win_ms: float = 200.0,
    hop_fraction: float = 0.25,
    filtered: bool = False,
    output_csv: Path = CHANNEL_STABILITY_CSV,
) -> pd.DataFrame:
    ensure_project_dirs()
    df = load_index(index_csv)

    folds = _fold_definitions()
    selection_rows: list[dict[str, object]] = []

    for seed_idx, (train_participants, val_participants) in enumerate(folds):
        train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
        val_df = df[df["participant"].isin(val_participants)].reset_index(drop=True)

        X_train, y_train, _, _ = build_feature_matrix(
            train_df,
            win_ms=win_ms,
            hop_fraction=hop_fraction,
            filtered=filtered,
        )
        X_val, y_val, _, _ = build_feature_matrix(
            val_df,
            win_ms=win_ms,
            hop_fraction=hop_fraction,
            filtered=filtered,
        )

        total_channels = X_train.shape[1] // 4
        ranking_df = rank_channels_by_macro_f1(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            total_channels=total_channels,
        )
        greedy_df = greedy_channel_order(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            total_channels=total_channels,
            max_channels=max(channel_counts),
        )

        ranking_order = ranking_df["channel_idx"].astype(int).tolist()
        greedy_order = greedy_df["added_channel"].astype(int).tolist()

        for method in methods:
            for channels_count in channel_counts:
                if method == "ranking":
                    selected_channels = ranking_order[:channels_count]
                elif method == "greedy":
                    selected_channels = greedy_order[:channels_count]
                else:
                    raise ValueError(f"Unknown method: {method}")

                frequency_counter = Counter(selected_channels)
                selection_metrics = _score_selected_channels(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    selected_channels=selected_channels,
                )
                selection_rows.append(
                    {
                        "row_type": "selection",
                        "seed": seed_idx,
                        "method": method,
                        "channels_count": int(channels_count),
                        "selected_channels": str(selected_channels),
                        "train_participants": _participants_string(train_participants),
                        "val_participants": _participants_string(val_participants),
                        "selection_score_accuracy": float(selection_metrics["accuracy"]),
                        "selection_score_macro_f1": float(selection_metrics["macro_f1"]),
                        "top_channel_frequency": max(frequency_counter.values()) if frequency_counter else 0,
                    }
                )

    pairwise_rows: list[dict[str, object]] = []
    selection_df = pd.DataFrame(selection_rows)
    for method in methods:
        for channels_count in channel_counts:
            subset_df = selection_df[
                (selection_df["method"] == method) & (selection_df["channels_count"] == int(channels_count))
            ]
            selections = {
                int(row.seed): ast.literal_eval(row.selected_channels)
                for row in subset_df.itertuples(index=False)
            }
            for seed_a, seed_b in combinations(sorted(selections), 2):
                pairwise_rows.append(
                    {
                        "row_type": "pairwise",
                        "seed": -1,
                        "method": method,
                        "channels_count": int(channels_count),
                        "selected_channels": "",
                        "train_participants": "",
                        "val_participants": "",
                        "selection_score_accuracy": None,
                        "selection_score_macro_f1": None,
                        "top_channel_frequency": None,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "jaccard": _jaccard(selections[seed_a], selections[seed_b]),
                    }
                )

    out_df = pd.concat([selection_df, pd.DataFrame(pairwise_rows)], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    save_channel_stability_plot(out_df, output_path=PLOTS_DIR / "channel_stability_jaccard.png")
    return out_df
