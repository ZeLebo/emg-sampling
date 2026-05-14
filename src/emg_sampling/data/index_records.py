"""Builds record indexes from raw GRABMyo files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from emg_sampling.config import LABELS, TARGET_GESTURES

_RECORD_PATTERN = re.compile(
    r"session(?P<session>\d+)_participant(?P<participant>\d+)_gesture(?P<gesture>\d+)_trial(?P<trial>\d+)"
)


def build_full_index(raw_root: Path) -> pd.DataFrame:
    """Builds a full index of GRABMyo WFDB records."""
    rows: list[dict[str, int | str]] = []
    for header_path in raw_root.rglob("*.hea"):
        match = _RECORD_PATTERN.match(header_path.stem)
        if not match:
            continue

        rows.append(
            {
                "record_path": str(header_path.with_suffix("")),
                "session": int(match.group("session")),
                "participant": int(match.group("participant")),
                "gesture": int(match.group("gesture")),
                "trial": int(match.group("trial")),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["record_path", "session", "participant", "gesture", "trial"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["participant", "session", "gesture", "trial"])
        .reset_index(drop=True)
    )


def build_4class_index(raw_root: Path) -> pd.DataFrame:
    """Builds an index for 4 target gestures used by the baseline."""
    full_df = build_full_index(raw_root)
    if full_df.empty:
        return pd.DataFrame(
            columns=[
                "record_path",
                "session",
                "participant",
                "gesture",
                "class",
                "y",
                "trial",
            ]
        )

    rows: list[dict[str, int | str]] = []
    for row in full_df.itertuples(index=False):
        gesture = int(row.gesture)
        if gesture not in TARGET_GESTURES:
            continue
        class_code = TARGET_GESTURES[gesture]
        rows.append(
            {
                "record_path": row.record_path,
                "session": int(row.session),
                "participant": int(row.participant),
                "gesture": gesture,
                "class": class_code,
                "y": LABELS[class_code],
                "trial": int(row.trial),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["participant", "session", "gesture", "trial"])
        .reset_index(drop=True)
    )
