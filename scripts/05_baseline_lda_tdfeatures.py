# %% Cell
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, f1_score

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")

# Параметры окна
WIN_MS = 200.0
HOP_MS = 50.0

# Фильтрацию на этом шаге НЕ делаем (пока).
# Сейчас задача: пройти end-to-end и получить первую метрику.


def sliding_windows(x: np.ndarray, fs: float, win_ms: float, hop_ms: float):
    win = int(round(win_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))
    T = x.shape[0]
    for start in range(0, T - win + 1, hop):
        yield x[start:start + win]


def extract_td_features(w: np.ndarray) -> np.ndarray:
    """
    w: (Twin, C)
    Возвращает (F,) — TD признаки по каналам, склеенные в один вектор.
    """
    eps = 1e-8

    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w ** 2, axis=0) + eps)
    wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)

    # zero-crossings (простая версия)
    s1 = w[:-1]
    s2 = w[1:]
    zc = np.sum((s1 * s2) < 0, axis=0)

    return np.concatenate([mav, rms, wl, zc], axis=0)


def read_record(record_path: str) -> tuple[np.ndarray, float]:
    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    return sig.astype(np.float32), fs


def main():
    df = pd.read_csv(INDEX_CSV)

    # Деление по участникам: последние 2 участника в test (для стабильности на малом N)
    participants = sorted(df["participant"].unique())
    if len(participants) < 3:
        raise SystemExit("Слишком мало участников для нормального split. Нужны хотя бы 3.")

    test_participants = set(participants[-2:])
    train_participants = set(participants[:-2])

    print("Participants total:", len(participants))
    print("Train participants:", sorted(train_participants))
    print("Test participants:", sorted(test_participants))

    def build_xy(sub_df: pd.DataFrame):
        X_list = []
        y_list = []
        per_window_times = []

        for row in sub_df.itertuples(index=False):
            x, fs = read_record(row.record_path)

            for w in sliding_windows(x, fs, WIN_MS, HOP_MS):
                t0 = time.perf_counter()
                feat = extract_td_features(w)
                t1 = time.perf_counter()

                X_list.append(feat)
                y_list.append(int(row.y))
                per_window_times.append(t1 - t0)

        X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y, np.array(per_window_times, dtype=np.float64)

    train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
    test_df = df[df["participant"].isin(test_participants)].reset_index(drop=True)

    print("Train records:", len(train_df), "Test records:", len(test_df))

    X_train, y_train, t_train = build_xy(train_df)
    X_test, y_test, t_test = build_xy(test_df)

    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print("\nAccuracy:", acc)
    print("Macro-F1:", f1)
    print("\nReport:\n", classification_report(y_test, pred, digits=4))

    # Время на извлечение фич (без учёта чтения файла)
    if len(t_test):
        print("\nFeature extraction time per window (test):")
        print("  mean ms:", float(np.mean(t_test) * 1000.0))
        print("  p95  ms:", float(np.quantile(t_test, 0.95) * 1000.0))

    # Алгоритмическая задержка от окна (минимум)
    print("\nAlgorithmic delay (window) ms:", WIN_MS)
    print("Hop ms:", HOP_MS)


if __name__ == "__main__":
    main()
