"""Single-channel ranking for fast EMG channel selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from emg_sampling.features.time_domain import BASIC_TD_FEATURE_NAMES, get_feature_column_indices
from emg_sampling.models.lda_baseline import evaluate_classifier, train_lda


def rank_channels_by_macro_f1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    total_channels: int,
) -> pd.DataFrame:
    """Ranks channels by single-channel LDA Macro-F1."""
    rows: list[dict[str, float | int]] = []
    for channel_idx in range(total_channels):
        columns = get_feature_column_indices(
            [channel_idx],
            total_channels=total_channels,
            feature_names=BASIC_TD_FEATURE_NAMES,
        )
        clf = train_lda(X_train[:, columns], y_train)
        metrics = evaluate_classifier(clf, X_test[:, columns], y_test)
        rows.append(
            {
                "channel_idx": channel_idx,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "accuracy", "channel_idx"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
