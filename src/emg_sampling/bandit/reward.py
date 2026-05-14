"""Reward functions for offline bandit profiles."""

from __future__ import annotations

import pandas as pd


def compute_reward(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.Series:
    """Computes scalar reward from quality, latency, channels, and feature size."""
    return (
        df["macro_f1"]
        - alpha * df["processing_time_ms"]
        - beta * df["channels_count"]
        - gamma * df["feature_vector_size"]
    )
