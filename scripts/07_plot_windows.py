# %% cell
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")

WIN_MS = 200.0
HOP_MS = 50.0

def main():
    df = pd.read_csv(INDEX_CSV)
    row = df[df["y"] == 0].iloc[0]  # WF
    record_path = row["record_path"]

    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    x = sig.astype(np.float32)

    win = int(round(WIN_MS * fs / 1000.0))
    hop = int(round(HOP_MS * fs / 1000.0))

    # Рисуем 3 секунды одного канала
    secs = 3
    n = int(secs * fs)
    t = np.arange(n) / fs
    ch = 0

    plt.figure()
    plt.plot(t, x[:n, ch])
    plt.title(f"EMG with windows | class={row['class']} | ch={ch}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Нарисуем первые 10 окон (их границы)
    count = 0
    for start in range(0, n - win + 1, hop):
        plt.axvspan(start/fs, (start+win)/fs, alpha=0.2)
        count += 1
        if count >= 10:
            break

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
