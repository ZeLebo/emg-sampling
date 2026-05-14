"""Helpers for selecting best experiment rows."""

from __future__ import annotations

import pandas as pd


def select_best_by_quality(df: pd.DataFrame) -> pd.Series:
    """Returns row with highest macro_f1, then accuracy."""
    valid = df[df["status"] == "ok"].copy()
    if valid.empty:
        raise ValueError("No rows with status='ok'")
    return valid.sort_values(["macro_f1", "accuracy", "win_ms"], ascending=[False, False, True]).iloc[0]


def select_smallest_near_best(
    df: pd.DataFrame,
    delta_macro_f1: float = 0.01,
) -> pd.Series:
    """Returns smallest-window row that is near best quality."""
    best = select_best_by_quality(df)
    valid = df[df["status"] == "ok"].copy()
    near_best = valid[valid["macro_f1"] >= float(best["macro_f1"]) - delta_macro_f1]
    if near_best.empty:
        return best
    return near_best.sort_values(["win_ms", "accuracy"], ascending=[True, False]).iloc[0]
