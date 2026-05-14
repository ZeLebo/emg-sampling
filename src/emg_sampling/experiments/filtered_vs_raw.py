"""Comparison experiment for raw and filtered signals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emg_sampling.config import HOP_FRACTION, N_TEST_PARTICIPANTS, WINDOW_MS_CANDIDATES
from emg_sampling.data.grabmyo_loader import load_index
from emg_sampling.data.split import split_by_participants
from emg_sampling.experiments.common import evaluate_split
from emg_sampling.paths import FILTERED_VS_RAW_CSV, ensure_project_dirs


def run_filtered_vs_raw_sweep(
    index_csv: Path,
    window_ms_candidates: list[float] | None = None,
    hop_fraction: float = HOP_FRACTION,
) -> pd.DataFrame:
    """Runs window-size sweep for raw and filtered signal variants."""
    ensure_project_dirs()

    if window_ms_candidates is None:
        window_ms_candidates = WINDOW_MS_CANDIDATES

    df = load_index(index_csv)
    train_df, test_df = split_by_participants(df, n_test_participants=N_TEST_PARTICIPANTS)

    rows: list[dict[str, float | int | str | bool]] = []
    for filtered in (False, True):
        for win_ms in window_ms_candidates:
            rows.append(
                evaluate_split(
                    train_df=train_df,
                    test_df=test_df,
                    win_ms=win_ms,
                    hop_fraction=hop_fraction,
                    filtered=filtered,
                )
            )

    out_df = pd.DataFrame(rows).sort_values(["filtered", "win_ms"]).reset_index(drop=True)
    out_df.to_csv(FILTERED_VS_RAW_CSV, index=False)
    return out_df
