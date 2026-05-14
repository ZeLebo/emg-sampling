from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.experiments.channel_sweep import run_channel_sweep
from emg_sampling.paths import CHANNEL_SWEEP_CSV, INDEX_4CLASSES_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRABMyo channel-reduction sweep.")
    parser.add_argument("--quick", action="store_true", help="Use a small subset of records for fast verification.")
    parser.add_argument("--medium", action="store_true", help="Use a medium-scale fixed participant split.")
    parser.add_argument("--window-ms", type=float, default=200.0, help="Sliding window size in milliseconds.")
    parser.add_argument("--filtered", action="store_true", help="Apply notch + bandpass preprocessing before features.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "random", "ranking", "greedy"],
        help="Selection methods to evaluate.",
    )
    parser.add_argument(
        "--channel-counts",
        nargs="+",
        type=int,
        default=[3, 4, 6, 8, 12, 16, 24],
        help="Channel counts to evaluate.",
    )
    parser.add_argument("--random-repeats", type=int, default=5, help="Number of random subsets per channel count.")
    parser.add_argument(
        "--max-greedy-channels",
        type=int,
        default=16,
        help="Maximum channel count for greedy forward selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHANNEL_SWEEP_CSV,
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_channel_sweep(
        index_csv=INDEX_4CLASSES_CSV,
        win_ms=args.window_ms,
        filtered=args.filtered,
        methods=args.methods,
        channel_counts=args.channel_counts,
        random_repeats=args.random_repeats,
        max_greedy_channels=args.max_greedy_channels,
        quick=args.quick,
        medium=args.medium,
        output_csv=args.output,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
