"""Builds action tables from saved experiment CSVs."""

from __future__ import annotations

import pandas as pd


def load_action_table(channel_sweep_csv: str, feature_sweep_csv: str) -> pd.DataFrame:
    """Loads unique candidate actions from channel and feature sweeps."""
    channel_df = pd.read_csv(channel_sweep_csv).copy()
    feature_df = pd.read_csv(feature_sweep_csv).copy()

    channel_df = channel_df.assign(
        action_source="channel_sweep",
        feature_set="basic",
        method=channel_df["method"],
    )[
        [
            "split_mode",
            "action_source",
            "method",
            "channels_count",
            "selected_channels",
            "feature_set",
            "window_ms",
            "filtering",
            "macro_f1",
            "accuracy",
            "processing_time_ms",
            "feature_vector_size",
        ]
    ]

    feature_df = feature_df.assign(
        action_source="feature_sweep",
        method=feature_df["channel_method"],
    )[
        [
            "split_mode",
            "action_source",
            "method",
            "channels_count",
            "selected_channels",
            "feature_set",
            "win_ms",
            "filtered",
            "macro_f1",
            "accuracy",
            "processing_time_ms",
            "feature_vector_size",
        ]
    ].rename(columns={"win_ms": "window_ms", "filtered": "filtering"})

    action_df = (
        pd.concat([channel_df, feature_df], ignore_index=True)
        .drop_duplicates(
            subset=[
                "split_mode",
                "method",
                "channels_count",
                "selected_channels",
                "feature_set",
                "window_ms",
                "filtering",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )
    return action_df
