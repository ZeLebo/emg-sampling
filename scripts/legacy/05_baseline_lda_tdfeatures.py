# %% Cell
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, f1_score

INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")
OUT_CSV = Path("data/processed/window_sweep_results.csv")

# Sweep по размерам окна (мс)
WIN_MS_CANDIDATES = [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]

# Держим тот же overlap, что был в baseline: для 200 мс это hop=50 мс (25% окна).
HOP_FRACTION = 0.25

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


@lru_cache(maxsize=None)
def read_record(record_path: str) -> tuple[np.ndarray, float]:
    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    return sig.astype(np.float32), fs


def evaluate_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    win_ms: float,
    hop_ms: float,
) -> dict[str, float | int | str]:
    def build_xy(sub_df: pd.DataFrame):
        X_list = []
        y_list = []
        per_window_times = []

        for row in sub_df.itertuples(index=False):
            x, fs = read_record(row.record_path)

            for w in sliding_windows(x, fs, win_ms, hop_ms):
                t0 = time.perf_counter()
                feat = extract_td_features(w)
                t1 = time.perf_counter()

                X_list.append(feat)
                y_list.append(int(row.y))
                per_window_times.append(t1 - t0)

        X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y, np.array(per_window_times, dtype=np.float64)

    X_train, y_train, t_train = build_xy(train_df)
    X_test, y_test, t_test = build_xy(test_df)

    if X_train.size == 0 or X_test.size == 0:
        return {
            "win_ms": float(win_ms),
            "hop_ms": float(hop_ms),
            "status": "skipped_no_windows",
            "n_train_windows": int(X_train.shape[0]),
            "n_test_windows": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "feat_time_mean_ms": np.nan,
            "feat_time_p95_ms": np.nan,
        }

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"\n=== Window {win_ms:.1f} ms | Hop {hop_ms:.1f} ms ===")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("Report:\n", classification_report(y_test, pred, digits=4))

    feat_mean_ms = float(np.mean(t_test) * 1000.0) if len(t_test) else np.nan
    feat_p95_ms = float(np.quantile(t_test, 0.95) * 1000.0) if len(t_test) else np.nan
    if len(t_test):
        print("Feature extraction per window (test) mean ms:", feat_mean_ms)
        print("Feature extraction per window (test) p95  ms:", feat_p95_ms)

    return {
        "win_ms": float(win_ms),
        "hop_ms": float(hop_ms),
        "status": "ok",
        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "feat_time_mean_ms": feat_mean_ms,
        "feat_time_p95_ms": feat_p95_ms,
    }


def main():
    print("Using index:", INDEX_CSV)
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

    train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
    test_df = df[df["participant"].isin(test_participants)].reset_index(drop=True)

    print("Train records:", len(train_df), "Test records:", len(test_df))
    # Прогреваем кэш чтения записей один раз, чтобы sweep не тратил время на повторный I/O.
    unique_paths = sorted(set(df["record_path"].tolist()))
    print("Unique records to cache:", len(unique_paths))
    for path in unique_paths:
        read_record(path)

    rows = []
    for win_ms in WIN_MS_CANDIDATES:
        hop_ms = win_ms * HOP_FRACTION
        result = evaluate_window(train_df, test_df, win_ms, hop_ms)
        rows.append(result)

    res_df = pd.DataFrame(rows).sort_values("win_ms").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_CSV, index=False)

    print("\n=== Sweep summary ===")
    print(res_df[[
        "win_ms",
        "hop_ms",
        "status",
        "n_train_windows",
        "n_test_windows",
        "accuracy",
        "macro_f1",
        "feat_time_mean_ms",
        "feat_time_p95_ms",
    ]].to_string(index=False))
    print("\nSaved sweep table:", OUT_CSV)

    valid = res_df[res_df["status"] == "ok"].copy()
    if valid.empty:
        print("\nNo valid window settings produced train/test windows.")
        return

    # 1) Максимум качества.
    best = valid.sort_values(["macro_f1", "accuracy", "win_ms"], ascending=[False, False, True]).iloc[0]

    # 2) Компромисс: минимальное окно среди тех, кто не хуже best_macro_f1 - delta.
    delta = 0.01
    near_best = valid[valid["macro_f1"] >= float(best["macro_f1"]) - delta]
    tradeoff = near_best.sort_values(["win_ms", "accuracy"], ascending=[True, False]).iloc[0]

    print("\n=== Best by quality ===")
    print(
        f"win={best['win_ms']:.1f} ms, hop={best['hop_ms']:.1f} ms, "
        f"macro_f1={best['macro_f1']:.4f}, acc={best['accuracy']:.4f}"
    )

    print("\n=== Trade-off optimum (smallest window within 0.01 macro-F1 of best) ===")
    print(
        f"win={tradeoff['win_ms']:.1f} ms, hop={tradeoff['hop_ms']:.1f} ms, "
        f"macro_f1={tradeoff['macro_f1']:.4f}, acc={tradeoff['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
