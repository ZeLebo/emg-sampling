"""Windowing and time-domain feature extraction."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def sliding_windows(
    x: np.ndarray,
    fs: float,
    win_ms: float,
    hop_ms: float,
) -> Iterator[np.ndarray]:
    """Generates sliding windows from signal.

    Args:
        x: Signal with shape (samples, channels).
        fs: Sampling frequency.
        win_ms: Window length in milliseconds.
        hop_ms: Hop length in milliseconds.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (samples, channels), got shape {x.shape}")

    win = int(round(win_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))
    if win <= 1:
        raise ValueError(f"Window size is too small: {win} samples")
    if hop <= 0:
        raise ValueError(f"Hop size is too small: {hop} samples")

    total = x.shape[0]
    for start in range(0, total - win + 1, hop):
        yield x[start : start + win]


def extract_td_features(window: np.ndarray) -> np.ndarray:
    """Extracts basic time-domain EMG features.

    Features by channel: MAV, RMS, WL, ZC.

    Args:
        window: Array with shape (window_samples, channels).

    Returns:
        Feature vector with shape (4 * channels,).
    """
    if window.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {window.shape}")

    eps = 1e-8
    mav = np.mean(np.abs(window), axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0) + eps)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

    s1 = window[:-1]
    s2 = window[1:]
    zc = np.sum((s1 * s2) < 0, axis=0)

    return np.concatenate([mav, rms, wl, zc], axis=0).astype(np.float32, copy=False)
