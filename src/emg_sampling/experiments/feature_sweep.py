"""Feature-set sweep for selected channel configurations."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from emg_sampling.config import DEFAULT_WINDOW_MS, HOP_FRACTION, SPLIT_MODES
from emg_sampling.data.grabmyo_loader import load_index
from emg_sampling.data.split import make_leakage_safe_split
from emg_sampling.experiments.common import evaluate_split
from emg_sampling.features.feature_sets import get_feature_names
from emg_sampling.paths import CHANNEL_SWEEP_CSV, FEATURE_SWEEP_CSV, PLOTS_DIR, ensure_project_dirs

DEFAULT_FEATURE_SWEEP_COUNTS = (3, 6, 8, 12, 24)
DEFAULT_FEATURE_SETS = ("basic", "extended_td")


def _participants_string(participants: Sequence[int]) -> str:
    return ",".join(str(participant) for participant in participants)


def _load_reference_channels(
    channel_sweep_csv: Path,
    method: str,
    counts: Sequence[int],
    win_ms: float,
    filtered: bool,
    split_mode: str,
) -> dict[int, list[int]]:
    sweep_df = pd.read_csv(channel_sweep_csv)
    filtered_df = sweep_df[
        (sweep_df["method"] == method)
        & (sweep_df["window_ms"] == float(win_ms))
        & (sweep_df["filtering"] == bool(filtered))
        & (sweep_df["split_mode"] == split_mode)
    ].copy()
    full_df = sweep_df[
        (sweep_df["method"] == "full")
        & (sweep_df["window_ms"] == float(win_ms))
        & (sweep_df["filtering"] == bool(filtered))
        & (sweep_df["split_mode"] == split_mode)
    ].copy()

    if filtered_df.empty and full_df.empty:
        raise ValueError(
            f"No channel_sweep rows found for method={method}/full, win_ms={win_ms}, "
            f"filtered={filtered}, split_mode={split_mode} in {channel_sweep_csv}"
        )

    channel_map: dict[int, list[int]] = {}
    for channels_count in counts:
        matches = filtered_df[filtered_df["channels_count"] == int(channels_count)]
        if matches.empty and int(channels_count) == 24:
            matches = full_df[full_df["channels_count"] == 24]
        if matches.empty:
            continue
        best_row = matches.sort_values("macro_f1", ascending=False).iloc[0]
        channel_map[int(channels_count)] = list(ast.literal_eval(best_row["selected_channels"]))
    return channel_map


def _plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for feature_set in df["feature_set"].unique():
        feature_df = df[df["feature_set"] == feature_set].sort_values("channels_count")
        ax.plot(feature_df["channels_count"], feature_df[metric], marker="o", linewidth=2, label=feature_set)

    ax.set_title(title)
    ax.set_xlabel("Channels count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_sweep_plots(
    df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    prefix: str = "feature_sweep",
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (plots_dir / f"{prefix}_macro_f1.png", "macro_f1", "Feature sweep - Macro-F1", "Macro-F1"),
        (
            plots_dir / f"{prefix}_processing_time.png",
            "processing_time_ms",
            "Feature sweep - Processing time",
            "Processing time per window (ms)",
        ),
        (
            plots_dir / f"{prefix}_feature_vector_size.png",
            "feature_vector_size",
            "Feature sweep - Feature vector size",
            "Features",
        ),
    ]
    for output_path, metric, title, ylabel in outputs:
        _plot_metric(df=df, metric=metric, title=title, ylabel=ylabel, output_path=output_path)
    return [output_path for output_path, *_rest in outputs]


def run_feature_sweep(
    index_csv: Path,
    channel_sweep_csv: Path = CHANNEL_SWEEP_CSV,
    win_ms: float = DEFAULT_WINDOW_MS,
    hop_fraction: float = HOP_FRACTION,
    filtered: bool = False,
    channel_counts: Sequence[int] = DEFAULT_FEATURE_SWEEP_COUNTS,
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
    channel_method: str = "greedy",
    quick: bool = False,
    medium: bool = False,
    output_csv: Path = FEATURE_SWEEP_CSV,
    plot_prefix: str = "feature_sweep",
) -> pd.DataFrame:
    ensure_project_dirs()

    if quick and medium:
        raise ValueError("Use only one of quick or medium modes.")

    split_mode = "quick" if quick else "medium" if medium else "full"
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode: {split_mode}")

    df = load_index(index_csv)
    split = make_leakage_safe_split(df, mode=split_mode)

    selected_channels_map = _load_reference_channels(
        channel_sweep_csv=channel_sweep_csv,
        method=channel_method,
        counts=channel_counts,
        win_ms=win_ms,
        filtered=filtered,
        split_mode=split_mode,
    )

    rows: list[dict[str, float | int | str | bool]] = []
    records_count = int(
        len(split.selection_train_df) + len(split.selection_val_df) + len(split.final_test_df)
    )
    selection_train_participants = _participants_string(split.selection_train_participants)
    selection_val_participants = _participants_string(split.selection_val_participants)
    final_test_participants = _participants_string(split.final_test_participants)
    train_participants = _participants_string(sorted(split.final_train_df["participant"].unique()))

    for channels_count in channel_counts:
        channels_count = int(channels_count)
        if channels_count not in selected_channels_map:
            continue
        selected_channels = selected_channels_map[channels_count]

        for feature_set in feature_sets:
            row = evaluate_split(
                train_df=split.final_train_df,
                test_df=split.final_test_df,
                win_ms=win_ms,
                hop_fraction=hop_fraction,
                filtered=filtered,
                channel_indices=selected_channels,
                feature_set=feature_set,
            )
            row.update(
                {
                    "split_mode": split_mode,
                    "channel_method": "full" if channels_count == 24 else channel_method,
                    "channels_count": channels_count,
                    "selected_channels": str(selected_channels),
                    "feature_vector_size": int(channels_count * len(get_feature_names(feature_set))),
                    "processing_time_ms": row["feat_time_mean_ms"],
                    "records_count": records_count,
                    "train_participants": train_participants,
                    "test_participants": final_test_participants,
                    "selection_train_participants": selection_train_participants,
                    "selection_val_participants": selection_val_participants,
                    "final_test_participants": final_test_participants,
                    "selection_metric": "macro_f1",
                    "final_metric": "macro_f1",
                    "selection_uses_test": False,
                    "quick_mode": split_mode == "quick",
                    "medium_mode": split_mode == "medium",
                }
            )
            rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["channels_count", "feature_set"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    save_feature_sweep_plots(out_df, plots_dir=PLOTS_DIR, prefix=plot_prefix)
    return out_df.reset_index(drop=True)
