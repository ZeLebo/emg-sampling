"""Offline bandit simulation over saved experiment tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from emg_sampling.bandit import evaluate_profiles, load_action_table
from emg_sampling.paths import BANDIT_SIMULATION_CSV, CHANNEL_SWEEP_MEDIUM_CSV, FEATURE_SWEEP_MEDIUM_CSV, PLOTS_DIR, ensure_project_dirs


def save_bandit_plots(results_df: pd.DataFrame) -> None:
    best_df = results_df.groupby("profile", as_index=False).first()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(best_df["profile"], best_df["reward"], color="tab:blue")
    ax.set_title("Offline bandit - reward by profile")
    ax.set_ylabel("Reward")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bandit_reward_by_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(best_df))
    ax.bar(x, best_df["channels_count"], width=0.35, label="channels_count")
    ax.bar(x, best_df["feature_vector_size"] / 10.0, width=0.35, label="feature_vector_size / 10", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(best_df["profile"], rotation=15)
    ax.set_title("Offline bandit - selected actions")
    ax.set_ylabel("Configuration size")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bandit_selected_actions.png", dpi=150)
    plt.close(fig)


def run_bandit_simulation(
    channel_sweep_csv: Path = CHANNEL_SWEEP_MEDIUM_CSV,
    feature_sweep_csv: Path = FEATURE_SWEEP_MEDIUM_CSV,
    output_csv: Path = BANDIT_SIMULATION_CSV,
) -> pd.DataFrame:
    ensure_project_dirs()
    actions_df = load_action_table(
        channel_sweep_csv=str(channel_sweep_csv),
        feature_sweep_csv=str(feature_sweep_csv),
    )
    actions_df = actions_df[actions_df["split_mode"] == "medium"].reset_index(drop=True)
    results_df = evaluate_profiles(actions_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    save_bandit_plots(results_df)
    return results_df
