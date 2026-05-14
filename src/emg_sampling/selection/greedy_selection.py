"""Greedy forward channel selection for EMG experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from emg_sampling.features.time_domain import BASIC_TD_FEATURE_NAMES, get_feature_column_indices
from emg_sampling.models.lda_baseline import evaluate_classifier, train_lda


def greedy_channel_order(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    total_channels: int,
    max_channels: int,
) -> pd.DataFrame:
    """Builds greedy channel order by maximizing validation Macro-F1 at each step."""
    if max_channels <= 0:
        raise ValueError("max_channels must be positive")
    if max_channels > total_channels:
        raise ValueError("max_channels cannot exceed total_channels")

    selected: list[int] = []
    available = set(range(total_channels))
    rows: list[dict[str, float | int | str]] = []

    for step in range(1, max_channels + 1):
        best_channel: int | None = None
        best_accuracy = -np.inf
        best_macro_f1 = -np.inf

        for candidate in sorted(available):
            candidate_channels = selected + [candidate]
            columns = get_feature_column_indices(
                candidate_channels,
                total_channels=total_channels,
                feature_names=BASIC_TD_FEATURE_NAMES,
            )
            clf = train_lda(X_train[:, columns], y_train)
            metrics = evaluate_classifier(clf, X_test[:, columns], y_test)
            if metrics["macro_f1"] > best_macro_f1 or (
                metrics["macro_f1"] == best_macro_f1 and metrics["accuracy"] > best_accuracy
            ):
                best_channel = candidate
                best_accuracy = metrics["accuracy"]
                best_macro_f1 = metrics["macro_f1"]

        if best_channel is None:
            break

        selected.append(best_channel)
        available.remove(best_channel)
        rows.append(
            {
                "step": step,
                "added_channel": best_channel,
                "selected_channels": str(selected),
                "accuracy": float(best_accuracy),
                "macro_f1": float(best_macro_f1),
            }
        )

    return pd.DataFrame(rows)
