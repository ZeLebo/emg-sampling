"""Profile-based offline action selection."""

from __future__ import annotations

import pandas as pd

from emg_sampling.bandit.reward import compute_reward

PROFILE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "quality_first": (0.01, 0.0005, 0.0001),
    "balanced": (0.03, 0.0020, 0.0004),
    "compression_first": (0.02, 0.0060, 0.0012),
    "low_latency": (0.08, 0.0015, 0.0003),
}


def evaluate_profiles(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Scores all actions under each reward profile and returns sorted table."""
    rows: list[pd.DataFrame] = []
    for profile, (alpha, beta, gamma) in PROFILE_WEIGHTS.items():
        profile_df = actions_df.copy()
        profile_df["profile"] = profile
        profile_df["alpha"] = alpha
        profile_df["beta"] = beta
        profile_df["gamma"] = gamma
        profile_df["reward"] = compute_reward(profile_df, alpha=alpha, beta=beta, gamma=gamma)
        rows.append(profile_df)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["profile", "reward"], ascending=[True, False])
        .reset_index(drop=True)
    )
