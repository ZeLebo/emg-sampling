from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.experiments.bandit_simulation import run_bandit_simulation
from emg_sampling.paths import BANDIT_SIMULATION_CSV, CHANNEL_SWEEP_MEDIUM_CSV, FEATURE_SWEEP_MEDIUM_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline bandit simulation over saved experiment CSVs.")
    parser.add_argument(
        "--channel-sweep-csv",
        type=Path,
        default=CHANNEL_SWEEP_MEDIUM_CSV,
        help="Channel sweep CSV path.",
    )
    parser.add_argument(
        "--feature-sweep-csv",
        type=Path,
        default=FEATURE_SWEEP_MEDIUM_CSV,
        help="Feature sweep CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BANDIT_SIMULATION_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bandit_simulation(
        channel_sweep_csv=args.channel_sweep_csv,
        feature_sweep_csv=args.feature_sweep_csv,
        output_csv=args.output,
    )
    print(result.groupby("profile", as_index=False).first().to_string(index=False))


if __name__ == "__main__":
    main()
