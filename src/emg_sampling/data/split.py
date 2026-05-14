"""Dataset split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ParticipantSplit:
    """Participant-level split for leakage-free channel selection."""

    selection_train_df: pd.DataFrame
    selection_val_df: pd.DataFrame
    final_train_df: pd.DataFrame
    final_test_df: pd.DataFrame
    selection_train_participants: tuple[int, ...]
    selection_val_participants: tuple[int, ...]
    final_test_participants: tuple[int, ...]
    mode: str


def split_by_participants(
    df: pd.DataFrame,
    n_test_participants: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy train/test split by participant."""
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


def _subset_records(
    df: pd.DataFrame,
    participants: tuple[int, ...],
    max_records_per_class: int | None = None,
) -> pd.DataFrame:
    subset_df = df[df["participant"].isin(participants)].copy()
    if max_records_per_class is not None:
        subset_df = (
            subset_df.groupby(["participant", "class"], as_index=False, sort=True)
            .head(max_records_per_class)
            .reset_index(drop=True)
        )
    else:
        subset_df = subset_df.reset_index(drop=True)
    return subset_df


def make_leakage_safe_split(df: pd.DataFrame, mode: str = "full") -> ParticipantSplit:
    """Returns fixed participant splits for quick, medium, and full modes."""
    if "participant" not in df.columns:
        raise ValueError("Expected 'participant' column in dataframe")

    participants = tuple(sorted(int(participant) for participant in df["participant"].unique()))
    participants_set = set(participants)

    if mode == "quick":
        selection_train = (1, 2)
        selection_val = (3, 4)
        final_test = (42, 43)
        max_records_per_class = 2
    elif mode == "medium":
        selection_train = tuple(range(1, 13))
        selection_val = tuple(range(13, 17))
        final_test = tuple(range(40, 44))
        max_records_per_class = None
    elif mode == "full":
        selection_train = tuple(range(1, 33))
        selection_val = tuple(range(33, 40))
        final_test = tuple(range(40, 44))
        max_records_per_class = None
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    for name, split_participants in (
        ("selection_train", selection_train),
        ("selection_val", selection_val),
        ("final_test", final_test),
    ):
        missing = set(split_participants).difference(participants_set)
        if missing:
            raise ValueError(f"Missing participants for {name} split: {sorted(missing)}")

    if set(selection_train) & set(selection_val):
        raise ValueError("selection_train and selection_val overlap")
    if set(selection_train) & set(final_test):
        raise ValueError("selection_train and final_test overlap")
    if set(selection_val) & set(final_test):
        raise ValueError("selection_val and final_test overlap")

    selection_train_df = _subset_records(
        df,
        participants=selection_train,
        max_records_per_class=max_records_per_class,
    )
    selection_val_df = _subset_records(
        df,
        participants=selection_val,
        max_records_per_class=max_records_per_class,
    )
    final_train_df = pd.concat([selection_train_df, selection_val_df], ignore_index=True)
    final_test_df = _subset_records(
        df,
        participants=final_test,
        max_records_per_class=max_records_per_class,
    )

    return ParticipantSplit(
        selection_train_df=selection_train_df,
        selection_val_df=selection_val_df,
        final_train_df=final_train_df,
        final_test_df=final_test_df,
        selection_train_participants=selection_train,
        selection_val_participants=selection_val,
        final_test_participants=final_test,
        mode=mode,
    )
