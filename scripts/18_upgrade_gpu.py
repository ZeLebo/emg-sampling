# %% Cell
"""
NOISY-ONLY bandit experiment for GRABMyo (PhysioNet) sEMG (WFDB)

Pipeline:
WFDB read -> (optional) filtering notch+bandpass -> (optional) synthetic noise ->
GPU windowing -> TD features (MAV/RMS/WL/ZC) -> LDA -> Accuracy/Macro-F1
Then contextual bandit (single state: "noisy") with forced exploration and alpha_lr sweep.

Outputs (in data/processed/):
- bandit_noisy_action_stats.csv          (mean/std over multiple noise seeds per action)
- bandit_noisy_alpha_sweep.csv           (bandit learned policy vs alpha_lr)
- bandit_noisy_delay_vs_f1.png
- bandit_noisy_delay_vs_reward.png
- bandit_noisy_alpha_policy.png
- bandit_noisy_alpha_convergence.png

Assumptions:
- You have already created: data/processed/grabmyo_index_4classes.csv
- That CSV contains at least: a WFDB record path + participant + label (flex-detected).
"""

import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INDEX_CSV = os.path.join("data", "processed", "grabmyo_index_4classes.csv")
OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Windowing
WIN_LIST_MS = [150, 200]
HOP_FRAC = 0.25  # hop = 0.25 * win
FEATURES = ["MAV", "RMS", "WL", "ZC"]  # per channel

# Filtering (same as your baseline)
FS_DEFAULT = 2048.0  # GRABMyo often 2048 Hz; will be overwritten if WFDB provides fs
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BP_LO = 20.0
BP_HI = 450.0
FILTER_ORDER = 4

# Bandit reward
LAMBDA_DELAY = 0.6  # reward = F1 - lambda * delay_seconds (matches your earlier logs)

# Actions (win_ms, filtered)
ACTIONS: List[Tuple[int, int]] = [(w, 1) for w in WIN_LIST_MS]
NOISE_SEEDS = list(range(0, 5))

# Bandit training
ALPHA_SWEEP = [0.05, 0.1, 0.2, 0.5, 1.0]  # learning rate sweep (bandit update)
EPISODES = 400
EPSILON = 0.15

# Reproducibility base
SEED_BASE = 12345

# Compute device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IO cache
WFDB_CACHE_MAXSIZE = 512


# =========================
# Utilities
# =========================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alg_delay_sec(win_ms: int) -> float:
    return float(win_ms) / 1000.0


def reward_from_f1(f1: float, win_ms: int) -> float:
    return float(f1) - float(LAMBDA_DELAY) * alg_delay_sec(win_ms)


def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_index(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rec_col = _detect_col(df, ["record", "record_path", "wfdb_record", "path", "rec"])
    part_col = _detect_col(df, ["participant", "subject", "subj", "participant_id", "subject_id"])
    label_col = _detect_col(df, ["label", "class", "gesture", "y"])

    missing = []
    if rec_col is None:
        missing.append("record path column (e.g., record/record_path/path)")
    if part_col is None:
        missing.append("participant column (e.g., participant/subject)")
    if label_col is None:
        missing.append("label column (e.g., label/class/gesture)")

    if missing:
        raise ValueError(
            f"Index CSV missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Fix your index CSV or update column detection."
        )

    df = df.rename(columns={rec_col: "record_path", part_col: "participant", label_col: "label"})

    # Normalize participant to something sortable
    df["participant"] = df["participant"].astype(str)

    # Normalize label to int classes
    if df["label"].dtype == object:
        # if it's strings like WF/WE/HO/HC or gesture ids, map consistently
        uniq = sorted(df["label"].unique().tolist())
        mapping = {k: i for i, k in enumerate(uniq)}
        df["label_id"] = df["label"].map(mapping).astype(int)
        df.attrs["label_mapping"] = mapping
    else:
        df["label_id"] = df["label"].astype(int)
        df.attrs["label_mapping"] = None

    return df[["record_path", "participant", "label", "label_id"]]


# =========================
# Signal processing
# =========================
@lru_cache(maxsize=16)
def design_filters(fs: float):
    # Notch
    b_notch, a_notch = signal.iirnotch(w0=NOTCH_HZ, Q=NOTCH_Q, fs=fs)
    # Bandpass
    sos_bp = signal.butter(FILTER_ORDER, [BP_LO, BP_HI], btype="bandpass", fs=fs, output="sos")
    return (b_notch, a_notch), sos_bp


def apply_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """
    x: (T, C) float32
    """
    (b_notch, a_notch), sos_bp = design_filters(fs)
    # filtfilt to avoid phase distortion
    y = signal.filtfilt(b_notch, a_notch, x, axis=0)
    y = signal.sosfiltfilt(sos_bp, y, axis=0)
    return y.astype(np.float32, copy=False)


def add_synthetic_noise(
    x: np.ndarray,
    fs: float,
    seed: int,
    hum_amp: float = 0.03,
    drift_amp: float = 0.02,
    white_sigma: float = 0.02,
    burst_amp: float = 0.12,
    burst_prob_per_sec: float = 0.6,
    burst_dur_ms: Tuple[int, int] = (10, 60),
) -> np.ndarray:
    """
    Adds: 50 Hz hum + 1 Hz drift + white noise + sparse bursts.
    Parameters are relative to signal std scale (roughly); tune if needed.
    x: (T, C)
    """
    rng = np.random.default_rng(seed)
    T, C = x.shape
    t = np.arange(T, dtype=np.float32) / fs

    y = x.astype(np.float32, copy=True)

    # Use per-channel scale for stability
    scale = np.std(y, axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)

    # hum (50 Hz)
    hum = np.sin(2 * np.pi * NOTCH_HZ * t, dtype=np.float32)[:, None] * (hum_amp * scale[None, :])
    # drift (1 Hz)
    drift = np.sin(2 * np.pi * 1.0 * t, dtype=np.float32)[:, None] * (drift_amp * scale[None, :])
    # white
    white = rng.normal(0.0, 1.0, size=(T, C)).astype(np.float32) * (white_sigma * scale[None, :])

    y += hum + drift + white

    # bursts
    total_sec = T / fs
    expected_bursts = burst_prob_per_sec * total_sec
    n_bursts = rng.poisson(expected_bursts)
    for _ in range(int(n_bursts)):
        dur_ms = int(rng.integers(burst_dur_ms[0], burst_dur_ms[1] + 1))
        dur = int(max(1, round(dur_ms * fs / 1000.0)))
        start = int(rng.integers(0, max(1, T - dur)))
        # burst as band-limited noise-ish spike train
        burst = rng.normal(0.0, 1.0, size=(dur, C)).astype(np.float32) * (burst_amp * scale[None, :])
        y[start : start + dur, :] += burst

    return y


# =========================
# GPU feature extraction
# =========================
def extract_td_features_torch(x: torch.Tensor, fs: float, win_ms: int, hop_ms: int) -> torch.Tensor:
    """
    x: (T, C) float32 on DEVICE
    returns: (Nwin, 4*C)
    """
    win = int(round(win_ms * fs / 1000.0))
    hop = int(round(hop_ms * fs / 1000.0))
    if x.shape[0] < win:
        return torch.empty((0, 4 * x.shape[1]), device=x.device, dtype=torch.float32)

    # (T,C).unfold(0, win, hop) -> (Nwin, C, win)
    w = x.unfold(0, win, hop).transpose(1, 2)  # -> (Nwin, win, C)

    mav = w.abs().mean(dim=1)
    rms = torch.sqrt((w * w).mean(dim=1) + 1e-8)
    wl = (w[:, 1:, :] - w[:, :-1, :]).abs().sum(dim=1)
    zc = ((w[:, :-1, :] * w[:, 1:, :]) < 0).sum(dim=1).to(torch.float32)

    feats = torch.cat([mav, rms, wl, zc], dim=1)
    return feats


# =========================
# Dataset IO + evaluation
# =========================
@dataclass
class EvalResult:
    state: str
    win_ms: int
    filtered: int
    noise_seed: int
    macro_f1: float
    acc: float
    delay_s: float
    reward: float
    n_train: int
    n_test: int
    seconds: float


@lru_cache(maxsize=WFDB_CACHE_MAXSIZE)
def _wfdb_read_record_cached(record_path: str) -> Tuple[np.ndarray, float]:
    """
    Returns:
      x: (T, C) float32
      fs: float
    """
    rec = wfdb.rdrecord(record_path)
    x = rec.p_signal
    if x is None:
        raise RuntimeError(f"WFDB record has no p_signal: {record_path}")
    x = np.asarray(x, dtype=np.float32)
    x.setflags(write=False)
    fs = float(getattr(rec, "fs", FS_DEFAULT) or FS_DEFAULT)
    return x, fs


def wfdb_read_record(record_path: str) -> Tuple[np.ndarray, float]:
    norm_path = os.path.normpath(record_path)
    return _wfdb_read_record_cached(norm_path)


def split_by_participants(index_df: pd.DataFrame, n_test_participants: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = sorted(index_df["participant"].unique().tolist())
    if len(parts) <= n_test_participants:
        raise ValueError(f"Not enough participants for split: have {len(parts)}, need > {n_test_participants}")
    test_parts = parts[-n_test_participants:]
    train_df = index_df[~index_df["participant"].isin(test_parts)].reset_index(drop=True)
    test_df = index_df[index_df["participant"].isin(test_parts)].reset_index(drop=True)
    return train_df, test_df


def build_Xy_for_df(
    df: pd.DataFrame,
    fs_override: Optional[float],
    win_ms: int,
    filtered: int,
    noisy: bool,
    noise_seed: int,
    progress_bar=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build window-level dataset for a list of records.
    Returns:
      X: (N, 4*C)
      y: (N,)
      fs_used: float
    """
    X_list = []
    y_list = []
    fs_used = None

    hop_ms = int(round(win_ms * HOP_FRAC))
    for _, row in df.iterrows():
        record_path = str(row["record_path"])
        label_id = int(row["label_id"])

        x, fs = wfdb_read_record(record_path)
        if fs_override is not None:
            fs = float(fs_override)

        fs_used = fs if fs_used is None else fs_used

        # preprocess
        if noisy:
            # make noise deterministic per record too (so different records differ)
            record_hash = (abs(hash(record_path)) % 1_000_000)
            ns = int(noise_seed * 10_000 + record_hash)
            x = add_synthetic_noise(x, fs, seed=ns)

        if filtered == 1:
            x = apply_filter(x, fs)

        # to GPU
        xt = torch.from_numpy(x).to(device=DEVICE, dtype=torch.float32)

        feats = extract_td_features_torch(xt, fs=fs, win_ms=win_ms, hop_ms=hop_ms)
        if feats.shape[0] == 0:
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        X_list.append(feats.detach().cpu().numpy())
        y_list.append(np.full((feats.shape[0],), label_id, dtype=np.int64))
        if progress_bar is not None:
            progress_bar.update(1)

    if not X_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), float(fs_used or FS_DEFAULT)

    X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)
    return X, y, float(fs_used or FS_DEFAULT)


def evaluate_action_noisy(
    index_df: pd.DataFrame,
    win_ms: int,
    filtered: int,
    noise_seed: int,
    fs_override: Optional[float] = None,
    progress_bar=None,
) -> EvalResult:
    """
    Single evaluation for the "noisy" state (with given noise_seed).
    """
    t0 = time.perf_counter()
    train_df, test_df = split_by_participants(index_df, n_test_participants=2)

    Xtr, ytr, fs_used = build_Xy_for_df(
        train_df,
        fs_override=fs_override,
        win_ms=win_ms,
        filtered=filtered,
        noisy=True,
        noise_seed=noise_seed,
        progress_bar=progress_bar,
    )
    Xte, yte, _ = build_Xy_for_df(
        test_df,
        fs_override=fs_override,
        win_ms=win_ms,
        filtered=filtered,
        noisy=True,
        noise_seed=noise_seed,
        progress_bar=progress_bar,
    )

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        raise RuntimeError(f"Empty dataset after windowing. Train={Xtr.shape}, Test={Xte.shape}")

    clf = LinearDiscriminantAnalysis()
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    macro_f1 = float(f1_score(yte, yhat, average="macro"))
    acc = float(accuracy_score(yte, yhat))

    delay_s = alg_delay_sec(win_ms)
    reward = reward_from_f1(macro_f1, win_ms)
    seconds = float(time.perf_counter() - t0)

    return EvalResult(
        state="noisy",
        win_ms=win_ms,
        filtered=filtered,
        noise_seed=noise_seed,
        macro_f1=macro_f1,
        acc=acc,
        delay_s=delay_s,
        reward=reward,
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        seconds=seconds,
    )


# =========================
# Bandit
# =========================
def forced_exploration_eps_greedy(Q: np.ndarray, N: np.ndarray, eps: float) -> int:
    untried = np.where(N == 0)[0]
    if untried.size > 0:
        return int(np.random.choice(untried))
    if np.random.rand() < eps:
        return int(np.random.randint(Q.size))
    return int(np.argmax(Q))


def run_bandit_noisy(
    index_df: pd.DataFrame,
    alpha_lr: float,
    episodes: int,
    eps: float,
    noise_seeds: List[int],
    eval_cache: Optional[Dict[Tuple[int, int, int], EvalResult]] = None,
    show_progress: bool = False,
) -> Dict:
    Q = np.zeros(len(ACTIONS), dtype=np.float64)
    N = np.zeros(len(ACTIONS), dtype=np.int64)

    best_a_hist = []
    pick_hist = []

    it = range(episodes)
    if show_progress:
        it = tqdm(it, total=episodes, desc=f"bandit alpha={alpha_lr:g}", leave=False, dynamic_ncols=True, unit="ep")

    for _ in it:
        a = forced_exploration_eps_greedy(Q, N, eps)
        win_ms, filt = ACTIONS[a]

        ns = int(np.random.choice(noise_seeds))
        if eval_cache is not None:
            res = eval_cache[(win_ms, filt, ns)]
        else:
            res = evaluate_action_noisy(index_df, win_ms=win_ms, filtered=filt, noise_seed=ns)
        r = float(res.reward)

        N[a] += 1
        Q[a] = (1.0 - alpha_lr) * Q[a] + alpha_lr * r

        pick_hist.append(a)
        best_a_hist.append(int(np.argmax(Q)))

    return {
        "alpha_lr": float(alpha_lr),
        "Q": Q,
        "N": N,
        "best_action_idx": int(np.argmax(Q)),
        "best_action": ACTIONS[int(np.argmax(Q))],
        "pick_hist": np.array(pick_hist, dtype=np.int64),
        "best_a_hist": np.array(best_a_hist, dtype=np.int64),
    }


# =========================
# Reporting / plots
# =========================
def precompute_eval_cache(
    index_df: pd.DataFrame,
    noise_seeds: List[int],
    fs_override: Optional[float] = None,
) -> Dict[Tuple[int, int, int], EvalResult]:
    cache: Dict[Tuple[int, int, int], EvalResult] = {}
    total = len(index_df) * len(ACTIONS) * len(noise_seeds)

    with tqdm(total=total, desc="precompute action/seed evals", dynamic_ncols=True, unit="rec") as pbar:
        for (win_ms, filt) in ACTIONS:
            for ns in noise_seeds:
                pbar.set_postfix({"win_ms": win_ms, "filt": filt, "seed": ns})
                res = evaluate_action_noisy(
                    index_df,
                    win_ms=win_ms,
                    filtered=filt,
                    noise_seed=ns,
                    fs_override=fs_override,
                    progress_bar=pbar,
                )
                cache[(win_ms, filt, ns)] = res

    return cache


def compute_action_stats(
    index_df: pd.DataFrame,
    noise_seeds: List[int],
    eval_cache: Optional[Dict[Tuple[int, int, int], EvalResult]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (win_ms, filt) in ACTIONS:
        for ns in noise_seeds:
            if eval_cache is not None:
                res = eval_cache[(win_ms, filt, ns)]
            else:
                res = evaluate_action_noisy(index_df, win_ms=win_ms, filtered=filt, noise_seed=ns)
            rows.append(res.__dict__)
            print(
                f"state=noisy | win={win_ms} | filt={filt} | seed={ns} | "
                f"F1={res.macro_f1:.4f} | acc={res.acc:.4f} | reward={res.reward:.4f}"
            )

    trials = pd.DataFrame(rows)

    stats = (
        trials.groupby(["win_ms", "filtered"], as_index=False)
        .agg(
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            delay_s=("delay_s", "first"),
            n=("macro_f1", "count"),
        )
    )
    return stats, trials


def plot_delay_vs_metric(stats: pd.DataFrame, metric: str, metric_std: str, outpath: str, title: str):
    plt.figure()
    for filt in [0, 1]:
        sub = stats[stats["filtered"] == filt].sort_values("delay_s")
        x = sub["delay_s"].to_numpy()
        y = sub[metric].to_numpy()
        yerr = sub[metric_std].fillna(0.0).to_numpy()
        plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", label=f"filtered={filt}")
    plt.xlabel("Algorithmic delay (s)")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_alpha_policy(sweep_df: pd.DataFrame, outpath: str):
    plt.figure()
    xs = sweep_df["alpha_lr"].to_numpy()
    ys = sweep_df["best_reward_est"].to_numpy()
    plt.plot(xs, ys, marker="o", linestyle="-")
    for _, r in sweep_df.iterrows():
        plt.annotate(
            f"w{int(r.win_ms)} f{int(r.filtered)}",
            (r.alpha_lr, r.best_reward_est),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    plt.xlabel("alpha_lr")
    plt.ylabel("Best estimated Q (reward)")
    plt.title("Noisy bandit: learned best action vs alpha_lr")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_alpha_convergence(sweep_runs: List[Dict], outpath: str):
    plt.figure()
    for run in sweep_runs:
        plt.plot(run["best_a_hist"], label=f"alpha={run['alpha_lr']}")
    plt.xlabel("Episode")
    plt.ylabel("argmax(Q) action index")
    plt.title("Noisy bandit convergence (best action index over time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print(f"Device: {DEVICE}")
    print(f"torch: {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    print(f"Index: {INDEX_CSV}")
    print(f"lambda_delay={LAMBDA_DELAY} | episodes={EPISODES} | eps={EPSILON} | noise_seeds={len(NOISE_SEEDS)}")
    print(f"Actions: {ACTIONS}")

    set_all_seeds(SEED_BASE)

    index_df = load_index(INDEX_CSV)
    eval_cache = precompute_eval_cache(index_df, NOISE_SEEDS)

    # 1) Action stats over multiple noise realizations
    stats, trials = compute_action_stats(index_df, NOISE_SEEDS, eval_cache=eval_cache)

    csv_action_stats = os.path.join(OUT_DIR, "bandit_noisy_action_stats.csv")
    stats.to_csv(csv_action_stats, index=False)
    print(f"Saved: {csv_action_stats}")

    # 2) Graphs: Delay vs F1 / Reward
    png_f1 = os.path.join(OUT_DIR, "bandit_noisy_delay_vs_f1.png")
    png_rw = os.path.join(OUT_DIR, "bandit_noisy_delay_vs_reward.png")

    plot_delay_vs_metric(
        stats,
        metric="f1_mean",
        metric_std="f1_std",
        outpath=png_f1,
        title=f"Noisy: Delay vs Macro-F1 (mean±std), λ={LAMBDA_DELAY}",
    )
    plot_delay_vs_metric(
        stats,
        metric="reward_mean",
        metric_std="reward_std",
        outpath=png_rw,
        title=f"Noisy: Delay vs Reward (mean±std), λ={LAMBDA_DELAY}",
    )
    print(f"Saved: {png_f1}")
    print(f"Saved: {png_rw}")

    # Oracle best by mean reward
    oracle = stats.sort_values("reward_mean", ascending=False).iloc[0]
    print("\n=== Oracle (best mean reward across noise seeds) ===")
    print(
        f"win={int(oracle.win_ms)} ms | filtered={int(oracle.filtered)} | "
        f"reward_mean={oracle.reward_mean:.4f}±{oracle.reward_std:.4f} | "
        f"f1_mean={oracle.f1_mean:.4f}±{oracle.f1_std:.4f}"
    )

    # 3) Bandit sweep over alpha_lr
    sweep_rows = []
    sweep_runs = []

    for alpha_lr in tqdm(ALPHA_SWEEP, desc="alpha sweep", dynamic_ncols=True, unit="alpha"):
        set_all_seeds(SEED_BASE + int(alpha_lr * 1000))

        run = run_bandit_noisy(
            index_df,
            alpha_lr=alpha_lr,
            episodes=EPISODES,
            eps=EPSILON,
            noise_seeds=NOISE_SEEDS,
            eval_cache=eval_cache,
            show_progress=True,
        )
        sweep_runs.append(run)

        best_idx = run["best_action_idx"]
        best_win, best_filt = run["best_action"]
        sweep_rows.append(
            {
                "alpha_lr": alpha_lr,
                "best_action_idx": best_idx,
                "win_ms": best_win,
                "filtered": best_filt,
                "best_reward_est": float(run["Q"][best_idx]),
                "n_pulls_best": int(run["N"][best_idx]),
                "pulls_total": int(run["N"].sum()),
            }
        )

        print(
            f"alpha={alpha_lr:.3f} -> best action: win={best_win} ms, filt={best_filt} | "
            f"Q={run['Q'][best_idx]:.4f} | pulls_best={run['N'][best_idx]}"
        )

    sweep_df = pd.DataFrame(sweep_rows).sort_values("alpha_lr")
    csv_sweep = os.path.join(OUT_DIR, "bandit_noisy_alpha_sweep.csv")
    sweep_df.to_csv(csv_sweep, index=False)
    print(f"\nSaved: {csv_sweep}")

    # 4) Bandit sweep plots
    png_policy = os.path.join(OUT_DIR, "bandit_noisy_alpha_policy.png")
    png_conv = os.path.join(OUT_DIR, "bandit_noisy_alpha_convergence.png")

    plot_alpha_policy(sweep_df, png_policy)
    plot_alpha_convergence(sweep_runs, png_conv)
    print(f"Saved: {png_policy}")
    print(f"Saved: {png_conv}")
    cache_info = _wfdb_read_record_cached.cache_info()
    print(
        f"WFDB cache | hits={cache_info.hits} misses={cache_info.misses} "
        f"currsize={cache_info.currsize}/{WFDB_CACHE_MAXSIZE}"
    )


if __name__ == "__main__":
    main()
