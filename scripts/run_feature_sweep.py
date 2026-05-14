from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.experiments.feature_sweep import run_feature_sweep
from emg_sampling.paths import CHANNEL_SWEEP_CSV, CHANNEL_SWEEP_MEDIUM_CSV, FEATURE_SWEEP_CSV, FEATURE_SWEEP_MEDIUM_CSV, INDEX_4CLASSES_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRABMyo feature-set sweep.")
    parser.add_argument("--quick", action="store_true", help="Use a small subset of records for fast verification.")
    parser.add_argument("--medium", action="store_true", help="Use a medium-scale fixed participant split.")
    parser.add_argument("--window-ms", type=float, default=200.0, help="Sliding window size in milliseconds.")
    parser.add_argument("--filtered", action="store_true", help="Apply notch + bandpass preprocessing before features.")
    parser.add_argument(
        "--channel-counts",
        nargs="+",
        type=int,
        default=[3, 6, 8, 12, 24],
        help="Channel counts to evaluate.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["basic", "extended_td"],
        help="Feature sets to evaluate.",
    )
    parser.add_argument(
        "--channel-method",
        default="greedy",
        choices=["greedy", "ranking", "random"],
        help="Channel selection method source taken from channel_sweep_results.csv",
    )
    parser.add_argument(
        "--channel-sweep-csv",
        type=Path,
        default=CHANNEL_SWEEP_CSV,
        help="Reference channel_sweep CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    channel_sweep_csv = args.channel_sweep_csv
    plot_prefix = "feature_sweep"
    if output is None:
        if args.medium:
            output = FEATURE_SWEEP_MEDIUM_CSV
            plot_prefix = "feature_sweep_medium"
            if channel_sweep_csv == CHANNEL_SWEEP_CSV:
                channel_sweep_csv = CHANNEL_SWEEP_MEDIUM_CSV
        else:
            output = FEATURE_SWEEP_CSV
    result = run_feature_sweep(
        index_csv=INDEX_4CLASSES_CSV,
        channel_sweep_csv=channel_sweep_csv,
        win_ms=args.window_ms,
        filtered=args.filtered,
        channel_counts=args.channel_counts,
        feature_sets=args.feature_sets,
        channel_method=args.channel_method,
        quick=args.quick,
        medium=args.medium,
        output_csv=output,
        plot_prefix=plot_prefix,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
