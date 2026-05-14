"""Plots experiment result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_vs_window(csv_path: Path, metric: str = "macro_f1") -> None:
    """Plots selected metric by window size from result CSV."""
    df = pd.read_csv(csv_path)
    plt.figure()
    if "filtered" in df.columns:
        for filtered in sorted(df["filtered"].dropna().unique()):
            subset = df[df["filtered"] == filtered]
            label = "filtered" if bool(filtered) else "raw"
            plt.plot(subset["win_ms"], subset[metric], marker="o", label=label)
        plt.legend()
    else:
        plt.plot(df["win_ms"], df[metric], marker="o")
    plt.xlabel("Window (ms)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs window size")
    plt.tight_layout()
