"""Dataset split helpers."""

from __future__ import annotations

import pandas as pd


def split_by_participants(
    df: pd.DataFrame,
    n_test_participants: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits records by participant.

    The last N participants are used for test split.
    This avoids train-test leakage between windows from one person.
    """
    if "participant" not in df.columns:
        raise ValueError("Expected 'participant' column in dataframe")

    participants = sorted(df["participant"].unique())
    if len(participants) <= n_test_participants:
        raise ValueError(
            "Not enough participants for split. "
            f"Got {len(participants)}, requested {n_test_participants} for test."
        )

    test_participants = set(participants[-n_test_participants:])
    train_participants = set(participants[:-n_test_participants])

    train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
    test_df = df[df["participant"].isin(test_participants)].reset_index(drop=True)

    return train_df, test_df
