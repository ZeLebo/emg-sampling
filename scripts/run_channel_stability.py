from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.experiments.channel_stability import run_channel_stability
from emg_sampling.paths import CHANNEL_STABILITY_CSV, INDEX_4CLASSES_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run channel selection stability analysis.")
    parser.add_argument(
        "--channel-counts",
        nargs="+",
        type=int,
        default=[3, 6, 8, 12],
        help="Channel counts to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ranking", "greedy"],
        help="Selection methods to evaluate.",
    )
    parser.add_argument("--filtered", action="store_true", help="Apply filtering before feature extraction.")
    parser.add_argument(
        "--output",
        type=Path,
        default=CHANNEL_STABILITY_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_channel_stability(
        index_csv=INDEX_4CLASSES_CSV,
        channel_counts=args.channel_counts,
        methods=args.methods,
        filtered=args.filtered,
        output_csv=args.output,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
