"""Filtered vs raw comparison for leakage-safe channel sweeps."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from emg_sampling.experiments.channel_sweep import run_channel_sweep
from emg_sampling.paths import CHANNEL_SWEEP_FILTERED_COMPARISON_CSV, PLOTS_DIR, ensure_project_dirs


def save_filtered_comparison_plot(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = (
        df.groupby(["method", "filtering", "channels_count"], as_index=False)["macro_f1"]
        .mean(numeric_only=True)
        .sort_values(["method", "filtering", "channels_count"])
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for (method, filtering), subset_df in plot_df.groupby(["method", "filtering"]):
        label = f"{method}-{'filtered' if filtering else 'raw'}"
        ax.plot(
            subset_df["channels_count"],
            subset_df["macro_f1"],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_title("Channel sweep filtered vs raw - Macro-F1")
    ax.set_xlabel("Channels count")
    ax.set_ylabel("Macro-F1")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_channel_filtered_comparison(
    index_csv: Path,
    channel_counts: Sequence[int],
    methods: Sequence[str],
    random_repeats: int = 3,
    medium: bool = True,
    output_csv: Path = CHANNEL_SWEEP_FILTERED_COMPARISON_CSV,
) -> pd.DataFrame:
    ensure_project_dirs()

    raw_df = run_channel_sweep(
        index_csv=index_csv,
        filtered=False,
        methods=methods,
        channel_counts=channel_counts,
        random_repeats=random_repeats,
        medium=medium,
        output_csv=output_csv,
        plot_prefix="channel_sweep_filtered_comparison_raw_tmp",
        save_plots_flag=False,
    )
    filtered_df = run_channel_sweep(
        index_csv=index_csv,
        filtered=True,
        methods=methods,
        channel_counts=channel_counts,
        random_repeats=random_repeats,
        medium=medium,
        output_csv=output_csv,
        plot_prefix="channel_sweep_filtered_comparison_filtered_tmp",
        save_plots_flag=False,
    )

    out_df = (
        pd.concat([raw_df, filtered_df], ignore_index=True)
        .sort_values(["filtering", "channels_count", "method", "random_repeat"])
        .reset_index(drop=True)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    save_filtered_comparison_plot(
        out_df,
        output_path=PLOTS_DIR / "channel_sweep_filtered_comparison_macro_f1.png",
    )
    return out_df
