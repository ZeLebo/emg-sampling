from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.experiments.repeated_medium_evaluation import run_repeated_medium_evaluation
from emg_sampling.paths import INDEX_4CLASSES_CSV, REPEATED_MEDIUM_EVALUATION_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated medium-like evaluations with confidence intervals.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["greedy", "ranking", "full"],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--channel-counts",
        nargs="+",
        type=int,
        default=[6, 12, 24],
        help="Channel counts to evaluate.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["basic", "extended_td"],
        help="Feature sets to evaluate.",
    )
    parser.add_argument("--filtered", action="store_true", help="Apply preprocessing before feature extraction.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPEATED_MEDIUM_EVALUATION_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_repeated_medium_evaluation(
        index_csv=INDEX_4CLASSES_CSV,
        methods=args.methods,
        channel_counts=args.channel_counts,
        feature_sets=args.feature_sets,
        filtered=args.filtered,
        output_csv=args.output,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
