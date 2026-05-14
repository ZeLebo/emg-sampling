"""Signal preprocessing for sEMG."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from emg_sampling.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BANDPASS_ORDER,
    NOTCH_HZ,
    NOTCH_Q,
)


def _to_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    if x.ndim == 1:
        return x[:, None], True
    if x.ndim == 2:
        return x, False
    raise ValueError(f"Expected 1D or 2D signal, got shape {x.shape}")


def notch_filter(
    x: np.ndarray,
    fs: float,
    notch_hz: float = NOTCH_HZ,
    q: float = NOTCH_Q,
) -> np.ndarray:
    """Applies notch filter to remove powerline interference."""
    x2d, was_1d = _to_2d(x)
    w0 = notch_hz / (0.5 * fs)
    b_notch, a_notch = iirnotch(w0, Q=q)
    y = filtfilt(b_notch, a_notch, x2d, axis=0)
    return y[:, 0] if was_1d else y


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float = BANDPASS_LOW,
    high_hz: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Applies Butterworth band-pass filter for sEMG range."""
    x2d, was_1d = _to_2d(x)

    low = low_hz / (0.5 * fs)
    high_hz_guarded = min(high_hz, 0.45 * fs)
    high = high_hz_guarded / (0.5 * fs)
    if not 0 < low < high < 1:
        raise ValueError(
            "Invalid filter boundaries. "
            f"Calculated low={low:.4f}, high={high:.4f} for fs={fs}."
        )

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x2d, axis=0)
    return y[:, 0] if was_1d else y


def preprocess_signal(
    x: np.ndarray,
    fs: float,
    use_notch: bool = True,
    use_bandpass: bool = True,
) -> np.ndarray:
    """Applies standard sEMG preprocessing: notch 50 Hz and bandpass 20-450 Hz."""
    y = x.astype(np.float32, copy=True)
    if use_notch:
        y = notch_filter(y, fs=fs)
    if use_bandpass:
        y = bandpass_filter(y, fs=fs)
    return y.astype(np.float32, copy=False)
