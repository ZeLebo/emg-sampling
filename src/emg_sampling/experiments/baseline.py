"""Baseline experiment: one window size and one signal mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emg_sampling.config import DEFAULT_WINDOW_MS, HOP_FRACTION, N_TEST_PARTICIPANTS
from emg_sampling.data.grabmyo_loader import load_index
from emg_sampling.data.split import split_by_participants
from emg_sampling.experiments.common import evaluate_split
from emg_sampling.paths import BASELINE_RESULTS_CSV, ensure_project_dirs


def run_baseline(
    index_csv: Path,
    win_ms: float = DEFAULT_WINDOW_MS,
    hop_fraction: float = HOP_FRACTION,
    filtered: bool = False,
) -> pd.DataFrame:
    """Runs baseline LDA pipeline and saves one-row metrics table."""
    ensure_project_dirs()

    df = load_index(index_csv)
    train_df, test_df = split_by_participants(df, n_test_participants=N_TEST_PARTICIPANTS)
    row = evaluate_split(
        train_df=train_df,
        test_df=test_df,
        win_ms=win_ms,
        hop_fraction=hop_fraction,
        filtered=filtered,
    )

    row["experiment"] = "baseline"
    out_df = pd.DataFrame([row])
    out_df.to_csv(BASELINE_RESULTS_CSV, index=False)
    return out_df
