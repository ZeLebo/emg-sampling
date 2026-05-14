"""Random channel-selection baselines."""

from __future__ import annotations

import numpy as np


def sample_random_channels(
    total_channels: int,
    channels_count: int,
    random_state: int | None = None,
) -> list[int]:
    """Returns a sorted random subset of channels."""
    if channels_count <= 0:
        raise ValueError("channels_count must be positive")
    if channels_count > total_channels:
        raise ValueError("channels_count cannot exceed total_channels")

    rng = np.random.default_rng(random_state)
    selected = rng.choice(total_channels, size=channels_count, replace=False)
    return sorted(int(channel_idx) for channel_idx in selected)
