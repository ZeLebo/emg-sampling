# %% cell
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")
WIN_MS = 200.0

def extract_td_features(w: np.ndarray) -> np.ndarray:
    eps = 1e-8
    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w ** 2, axis=0) + eps)
    wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
    s1 = w[:-1]
    s2 = w[1:]
    zc = np.sum((s1 * s2) < 0, axis=0)
    return np.concatenate([mav, rms, wl, zc], axis=0)

def main():
    df = pd.read_csv(INDEX_CSV)
    row = df[df["y"] == 0].iloc[0]  # WF
    sig, fields = wfdb.rdsamp(row["record_path"])
    fs = float(fields["fs"])
    x = sig.astype(np.float32)

    win = int(round(WIN_MS * fs / 1000.0))
    w = x[:win]  # первое окно

    feat = extract_td_features(w)
    print("Feature vector length:", feat.shape[0])
    print("First 20 values:", feat[:20])

    # Объяснение структуры: по 32 канала
    C = x.shape[1]
    print("Channels:", C)
    print("Features per channel = 4 (MAV,RMS,WL,ZC)")
    print("Total = 4*C =", 4*C)

if __name__ == "__main__":
    main()
