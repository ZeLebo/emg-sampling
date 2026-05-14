from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.experiments.channel_filtered_comparison import run_channel_filtered_comparison
from emg_sampling.paths import CHANNEL_SWEEP_FILTERED_COMPARISON_CSV, INDEX_4CLASSES_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and filtered leakage-safe channel sweeps.")
    parser.add_argument(
        "--channel-counts",
        nargs="+",
        type=int,
        default=[3, 6, 8, 12, 24],
        help="Channel counts to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "ranking", "greedy", "random"],
        help="Selection methods to evaluate.",
    )
    parser.add_argument("--random-repeats", type=int, default=3, help="Number of random repeats.")
    parser.add_argument(
        "--output",
        type=Path,
        default=CHANNEL_SWEEP_FILTERED_COMPARISON_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_channel_filtered_comparison(
        index_csv=INDEX_4CLASSES_CSV,
        channel_counts=args.channel_counts,
        methods=args.methods,
        random_repeats=args.random_repeats,
        output_csv=args.output,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
