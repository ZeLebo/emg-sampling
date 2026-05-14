"""Feature-set registry used by experiments."""

from __future__ import annotations

from emg_sampling.features.time_domain import (
    BASIC_TD_FEATURE_NAMES,
    EXTENDED_TD_FEATURE_NAMES,
    extract_td_feature_set,
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "basic": BASIC_TD_FEATURE_NAMES,
    "extended_td": EXTENDED_TD_FEATURE_NAMES,
}


def get_feature_names(feature_set: str) -> tuple[str, ...]:
    """Returns ordered feature names for a named feature set."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(f"Unknown feature set: {feature_set}") from exc


def extract_feature_set(window, feature_set: str):
    """Extracts one named feature set from a signal window."""
    return extract_td_feature_set(window, get_feature_names(feature_set))
