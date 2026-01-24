# %% Cell
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score

# GPU (опционально). Если torch не установлен: uv add torch
try:
    import torch
except Exception as e:
    torch = None


# =========================
# Конфигурация эксперимента
# =========================
INDEX_CSV = Path("data/processed/grabmyo_index_4classes.csv")
OUT_REWARDS_CSV = Path("data/processed/qlearning_window_filter_rewards.csv")

# Состояния (условия среды)
STATES = ["clean", "noisy"]  # noisy = synthetic noise

# Действия (конфигурация пайплайна)
WINDOW_MS_LIST = [150.0, 200.0, 250.0]
FILTER_OPTIONS = [0, 1]  # 0=raw, 1=notch+bandpass
ACTIONS = [(w, f) for w in WINDOW_MS_LIST for f in FILTER_OPTIONS]

HOP_FRACTION = 0.25

# Reward weights
LAMBDA_DELAY = 0.15   # штраф за задержку
MU_CPU = 0.00         # штраф за время фич (можно включить позже)

# Ограничение данных для скорости (можно = None чтобы брать всё)
MAX_TRAIN_RECORDS = None  # например 200
MAX_TEST_RECORDS = None   # например 50

# Synthetic noise params (как у тебя)
BASE_SEED = 123
ADD_HUM_50HZ = True
HUM_AMPL = 0.15
ADD_DRIFT = True
DRIFT_AMPL = 0.20
DRIFT_HZ = 1.0
ADD_WHITE = True
WHITE_STD = 0.05
ADD_MOTION_BURST = True
BURST_STD = 0.25
BURST_PROB = 0.02

# Фильтр (standard sEMG)
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
BP_ORDER = 4

# Q-learning
EPISODES = 60
EPSILON = 0.15
ALPHA = 0.35
GAMMA = 0.0  # здесь задача фактически bandit (один шаг), поэтому gamma=0


# =========================
# Утилиты
# =========================
def stable_int_seed(s: str, base_seed: int) -> int:
    h = hashlib.sha256((s + str(base_seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def pick_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()


# =========================
# Чтение данных
# =========================
def read_record(record_path: str) -> Tuple[np.ndarray, float]:
    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    return sig.astype(np.float32), fs


# =========================
# GPU: шум + фильтр + фичи
# =========================
def add_synthetic_noise_torch(x: torch.Tensor, fs: float, seed_key: str) -> torch.Tensor:
    """
    x: (T, C) float32 on DEVICE
    """
    # RNG детерминированный на запись
    seed = stable_int_seed(seed_key, BASE_SEED)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    T, C = x.shape
    t = (torch.arange(T, device=DEVICE, dtype=torch.float32) / float(fs)).view(-1, 1)

    y = x

    if ADD_HUM_50HZ:
        y = y + (HUM_AMPL * torch.sin(2.0 * torch.pi * 50.0 * t))

    if ADD_DRIFT:
        phase = torch.rand(1, generator=g, device=DEVICE).item() * 2.0 * np.pi
        y = y + (DRIFT_AMPL * torch.sin(2.0 * torch.pi * DRIFT_HZ * t + phase))

    if ADD_WHITE:
        y = y + torch.randn((T, C), generator=g, device=DEVICE, dtype=torch.float32) * WHITE_STD

    if ADD_MOTION_BURST:
        mask = (torch.rand((T, 1), generator=g, device=DEVICE) < BURST_PROB).to(torch.float32)
        bursts = torch.randn((T, C), generator=g, device=DEVICE, dtype=torch.float32) * BURST_STD
        y = y + mask * bursts

    return y


def bandpass_notch_numpy(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Фильтрация на CPU (scipy). Это не самое быстрое, но:
    - воспроизводимо
    - совпадает с тем, что ты уже использовал
    Если захочешь полностью на GPU — можно отдельно сделать FIR+conv1d.
    """
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

    # Notch 50 Hz
    w0 = NOTCH_HZ / (0.5 * fs)
    bN, aN = iirnotch(w0, Q=NOTCH_Q)
    x = filtfilt(bN, aN, x, axis=0)

    # Band-pass
    low = BANDPASS_LOW / (0.5 * fs)
    high_hz = min(BANDPASS_HIGH, 0.45 * fs)
    high = high_hz / (0.5 * fs)
    sos = butter(BP_ORDER, [low, high], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0)

    return x.astype(np.float32)


def extract_td_features_torch(x: torch.Tensor, fs: float, win_ms: float, hop_ms: float) -> torch.Tensor:
    """
    x: (T, C) float32 on DEVICE
    return: (Nwin, 4*C) on DEVICE
    """
    win = int(round(win_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))
    if x.shape[0] < win:
        return torch.empty((0, 4 * x.shape[1]), device=x.device, dtype=torch.float32)

    # torch.unfold(0, ...) on (T, C) -> (Nwin, C, win)
    w = x.unfold(0, win, hop).transpose(1, 2)  # -> (Nwin, win, C)

    # MAV over time
    mav = w.abs().mean(dim=1)

    # RMS over time
    rms = torch.sqrt((w * w).mean(dim=1) + 1e-8)

    # WL: sum of abs(diff) over time
    wl = (w[:, 1:, :] - w[:, :-1, :]).abs().sum(dim=1)

    # ZC: count sign changes over time
    s1 = w[:, :-1, :]
    s2 = w[:, 1:, :]
    zc = ((s1 * s2) < 0).sum(dim=1).to(torch.float32)

    feats = torch.cat([mav, rms, wl, zc], dim=1)
    return feats

# =========================
# Оценка action -> reward
# =========================
@dataclass(frozen=True)
class EvalKey:
    state: str
    win_ms: float
    filtered: int


_eval_cache: Dict[EvalKey, Dict[str, float]] = {}


def evaluate_action(
    df: pd.DataFrame,
    state: str,
    win_ms: float,
    filtered: int,
) -> Dict[str, float]:
    """
    Возвращает метрики и reward для (state, action).
    state: clean/noisy
    filtered: 0/1
    """
    key = EvalKey(state=state, win_ms=win_ms, filtered=filtered)
    if key in _eval_cache:
        return _eval_cache[key]

    participants = sorted(df["participant"].unique())
    test_participants = set(participants[-2:])
    train_participants = set(participants[:-2])

    train_df = df[df["participant"].isin(train_participants)].reset_index(drop=True)
    test_df = df[df["participant"].isin(test_participants)].reset_index(drop=True)

    if MAX_TRAIN_RECORDS is not None:
        train_df = train_df.iloc[:MAX_TRAIN_RECORDS].copy()
    if MAX_TEST_RECORDS is not None:
        test_df = test_df.iloc[:MAX_TEST_RECORDS].copy()

    hop_ms = win_ms * HOP_FRACTION

    X_tr, y_tr, feat_ms_tr = build_xy(df_part=train_df, state=state, win_ms=win_ms, hop_ms=hop_ms, filtered=filtered)
    X_te, y_te, feat_ms_te = build_xy(df_part=test_df, state=state, win_ms=win_ms, hop_ms=hop_ms, filtered=filtered)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    acc = float(accuracy_score(y_te, pred))
    mf1 = float(f1_score(y_te, pred, average="macro"))

    # Delay: минимум = win_ms
    delay_norm = float(win_ms) / float(max(WINDOW_MS_LIST))

    # Compute cost: mean ms per window (feature only)
    feat_mean_ms = float(np.mean(feat_ms_te)) if len(feat_ms_te) else 0.0
    feat_norm = feat_mean_ms / 1.0  # ~1ms scale (можно менять)

    reward = mf1 - LAMBDA_DELAY * delay_norm - MU_CPU * feat_norm

    out = {
        "state": state,
        "win_ms": float(win_ms),
        "hop_ms": float(hop_ms),
        "filtered": int(filtered),
        "accuracy": acc,
        "macro_f1": mf1,
        "feat_time_mean_ms": float(feat_mean_ms),
        "reward": float(reward),
        "n_train_windows": int(X_tr.shape[0]),
        "n_test_windows": int(X_te.shape[0]),
    }
    _eval_cache[key] = out
    return out


def build_xy(df_part: pd.DataFrame, state: str, win_ms: float, hop_ms: float, filtered: int):
    X_list = []
    y_list = []
    feat_times_ms = []

    for row in df_part.itertuples(index=False):
        # read (CPU)
        x_np, fs = read_record(row.record_path)

        # state: add synthetic noise (GPU if available)
        if torch is not None and DEVICE == "cuda":
            x = torch.from_numpy(x_np).to(DEVICE, non_blocking=True)
            if state == "noisy":
                x = add_synthetic_noise_torch(x, fs, seed_key=row.record_path)
            # filtering: scipy CPU (чтобы совпадало с предыдущими результатами)
            if filtered == 1:
                # выгружаем на CPU -> фильтруем -> обратно на GPU
                x_cpu = x.detach().cpu().numpy()
                x_cpu = bandpass_notch_numpy(x_cpu, fs)
                x = torch.from_numpy(x_cpu).to(DEVICE, non_blocking=True)

            # features on GPU
            t0 = time.perf_counter()
            feats = extract_td_features_torch(x, fs, win_ms, hop_ms)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # в sklearn нужно на CPU numpy
            feats_np = feats.detach().cpu().numpy()

            X_list.append(feats_np)
            y_list.append(np.full((feats_np.shape[0],), int(row.y), dtype=np.int64))
            feat_times_ms.append((t1 - t0) * 1000.0 / max(feats_np.shape[0], 1))

        else:
            # CPU-only path
            x = x_np
            if state == "noisy":
                # numpy noise (для cpu)
                rng = np.random.RandomState(stable_int_seed(row.record_path, BASE_SEED))
                T, C = x.shape
                t = (np.arange(T, dtype=np.float32) / fs).reshape(-1, 1)
                y = x.copy()
                if ADD_HUM_50HZ:
                    y += (HUM_AMPL * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
                if ADD_DRIFT:
                    phase = rng.uniform(0, 2*np.pi)
                    y += (DRIFT_AMPL * np.sin(2 * np.pi * DRIFT_HZ * t + phase)).astype(np.float32)
                if ADD_WHITE:
                    y += rng.normal(0.0, WHITE_STD, size=(T, C)).astype(np.float32)
                if ADD_MOTION_BURST:
                    mask = (rng.rand(T, 1) < BURST_PROB).astype(np.float32)
                    bursts = rng.normal(0.0, BURST_STD, size=(T, C)).astype(np.float32)
                    y += mask * bursts
                x = y

            if filtered == 1:
                x = bandpass_notch_numpy(x, fs)

            # features cpu (без torch)
            t0 = time.perf_counter()
            feats_np = extract_td_features_cpu(x, fs, win_ms, hop_ms)
            t1 = time.perf_counter()

            X_list.append(feats_np)
            y_list.append(np.full((feats_np.shape[0],), int(row.y), dtype=np.int64))
            feat_times_ms.append((t1 - t0) * 1000.0 / max(feats_np.shape[0], 1))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    feat_times_ms = np.array(feat_times_ms, dtype=np.float64)
    return X, y, feat_times_ms


def extract_td_features_cpu(x: np.ndarray, fs: float, win_ms: float, hop_ms: float) -> np.ndarray:
    win = int(round(win_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))
    T, C = x.shape
    if T < win:
        return np.empty((0, 4 * C), dtype=np.float32)

    feats = []
    for start in range(0, T - win + 1, hop):
        w = x[start : start + win]

        mav = np.mean(np.abs(w), axis=0)
        rms = np.sqrt(np.mean(w ** 2, axis=0) + 1e-8)
        wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
        s1 = w[:-1]
        s2 = w[1:]
        zc = np.sum((s1 * s2) < 0, axis=0).astype(np.float32)

        feats.append(np.concatenate([mav, rms, wl, zc], axis=0))

    return np.vstack(feats).astype(np.float32)


# =========================
# Q-learning (bandit)
# =========================
def epsilon_greedy(Q: np.ndarray, s_idx: int, eps: float) -> int:
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[s_idx]))


def main():
    print("Device:", DEVICE)
    if torch is None:
        print("torch: not installed (CPU path)")
    else:
        print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

    df = pd.read_csv(INDEX_CSV)

    # 1) Предварительно посчитаем rewards для всех (state, action) и сохраним таблицу
    rows = []
    for state in STATES:
        for (win_ms, filtered) in ACTIONS:
            m = evaluate_action(df, state=state, win_ms=win_ms, filtered=filtered)
            rows.append(m)
            print(
                f"state={state:5s} | win={win_ms:3.0f} | filt={filtered} | "
                f"F1={m['macro_f1']:.4f} | acc={m['accuracy']:.4f} | reward={m['reward']:.4f}"
            )

    res = pd.DataFrame(rows).sort_values(["state", "win_ms", "filtered"]).reset_index(drop=True)
    OUT_REWARDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_REWARDS_CSV, index=False)
    print("Saved:", OUT_REWARDS_CSV)

    # 2) Q-learning (один шаг): Q(s,a) -> reward(s,a)
    nS = len(STATES)
    nA = len(ACTIONS)
    Q = np.zeros((nS, nA), dtype=np.float64)

    # Быстрый доступ reward по (s,a)
    reward_lookup: Dict[Tuple[int, int], float] = {}
    for s_idx, state in enumerate(STATES):
        for a_idx, (win_ms, filtered) in enumerate(ACTIONS):
            r = float(res[(res["state"] == state) & (res["win_ms"] == win_ms) & (res["filtered"] == filtered)]["reward"].iloc[0])
            reward_lookup[(s_idx, a_idx)] = r

    for ep in range(EPISODES):
        s_idx = np.random.randint(nS)
        a_idx = epsilon_greedy(Q, s_idx, EPSILON)
        r = reward_lookup[(s_idx, a_idx)]

        # gamma=0 => one-step
        Q[s_idx, a_idx] = (1 - ALPHA) * Q[s_idx, a_idx] + ALPHA * (r)

    # 3) Итоговая политика
    print("\n=== Learned policy (argmax_a Q[s,a]) ===")
    for s_idx, state in enumerate(STATES):
        best_a = int(np.argmax(Q[s_idx]))
        win_ms, filtered = ACTIONS[best_a]
        print(f"state={state:5s} -> action: win={win_ms:.0f} ms, filtered={filtered}")

    print("\n=== Q-table ===")
    for s_idx, state in enumerate(STATES):
        print(f"\nstate={state}")
        for a_idx, (win_ms, filtered) in enumerate(ACTIONS):
            print(f"  a{a_idx}: win={win_ms:3.0f}, filt={filtered} | Q={Q[s_idx, a_idx]:.4f}")

    # 4) Для отчёта: “лучшее” по reward напрямую (то же, что выучит Q-learning)
    print("\n=== Direct best by reward ===")
    for state in STATES:
        sub = res[res["state"] == state].sort_values(["reward"], ascending=False).iloc[0]
        print(f"state={state:5s} -> win={sub['win_ms']:.0f} ms, filtered={int(sub['filtered'])}, reward={sub['reward']:.4f}, F1={sub['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
