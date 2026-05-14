"""Loading helpers for GRABMyo records and prepared indexes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


@lru_cache(maxsize=None)
def read_record(record_path: str | Path) -> tuple[np.ndarray, float]:
    """Reads one GRABMyo record.

    Args:
        record_path: Path to WFDB record without extension.

    Returns:
        signal: Array with shape (samples, channels).
        fs: Sampling frequency in Hz.
    """
    path = Path(record_path)
    header_path = path.with_suffix(".hea")
    if not header_path.exists():
        raise FileNotFoundError(f"WFDB header file not found: {header_path}")

    signal, fields = wfdb.rdsamp(str(path))
    return signal.astype(np.float32), float(fields["fs"])


def load_index(index_csv: Path) -> pd.DataFrame:
    """Loads prepared GRABMyo index CSV."""
    if not index_csv.exists():
        raise FileNotFoundError(
            f"Index CSV not found: {index_csv}. Run scripts/make_index_4classes.py first."
        )

    df = pd.read_csv(index_csv)
    required = {"record_path", "participant", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Index CSV is missing required columns: {sorted(missing)}")

    return df
