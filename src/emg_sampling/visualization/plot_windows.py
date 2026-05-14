"""Window visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_window(window: np.ndarray, fs: float, channel: int = 0) -> None:
    """Plots one extracted window for selected channel."""
    if window.ndim != 2:
        raise ValueError(f"Expected (window_samples, channels), got {window.shape}")
    t = np.arange(window.shape[0], dtype=np.float32) / fs
    plt.figure()
    plt.plot(t, window[:, channel])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Window view, channel {channel}")
    plt.tight_layout()
