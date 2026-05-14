"""Signal-level visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_channel(signal: np.ndarray, fs: float, channel: int = 0, seconds: float = 3.0) -> None:
    """Plots one channel fragment from multi-channel signal."""
    if signal.ndim != 2:
        raise ValueError(f"Expected (samples, channels), got {signal.shape}")
    n = min(signal.shape[0], int(seconds * fs))
    t = np.arange(n, dtype=np.float32) / fs
    plt.figure()
    plt.plot(t, signal[:n, channel])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"EMG channel {channel}")
    plt.tight_layout()
