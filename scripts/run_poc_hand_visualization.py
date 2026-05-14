from __future__ import annotations

import argparse
import ast
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

from emg_sampling.config import HOP_FRACTION, LABELS
from emg_sampling.data.grabmyo_loader import load_index, read_record
from emg_sampling.data.split import split_by_participants
from emg_sampling.experiments.common import build_feature_matrix
from emg_sampling.features.feature_sets import extract_feature_set
from emg_sampling.features.time_domain import sliding_windows
from emg_sampling.models.lda_baseline import train_lda
from emg_sampling.paths import CHANNEL_SWEEP_CSV, INDEX_4CLASSES_CSV, PLOTS_DIR, ensure_project_dirs

LABEL_TO_CLASS = {value: key for key, value in LABELS.items()}


def _make_quick_subset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quick_train = (
        train_df[train_df["participant"].isin(sorted(train_df["participant"].unique())[:4])]
        .groupby(["participant", "class"], as_index=False, sort=True)
        .head(2)
        .reset_index(drop=True)
    )
    quick_test = (
        test_df.groupby(["participant", "class"], as_index=False, sort=True)
        .head(2)
        .reset_index(drop=True)
    )
    return quick_train, quick_test


def _load_channels(channel_sweep_csv: Path, quick: bool, channels_count: int = 12) -> list[int]:
    sweep_df = pd.read_csv(channel_sweep_csv)
    matches = sweep_df[
        (sweep_df["method"] == "greedy")
        & (sweep_df["channels_count"] == channels_count)
        & (sweep_df["quick_mode"] == bool(quick))
    ].sort_values("macro_f1", ascending=False)
    if matches.empty:
        raise ValueError(f"No greedy {channels_count}-channel configuration found in {channel_sweep_csv}")
    return list(ast.literal_eval(matches.iloc[0]["selected_channels"]))


def _draw_hand(ax, class_code: str) -> None:
    palm = Circle((0.5, 0.35), 0.16, fill=False, linewidth=3)
    ax.add_patch(palm)

    if class_code == "HO":
        finger_heights = [0.78, 0.85, 0.88, 0.82, 0.72]
        x_positions = [0.28, 0.40, 0.52, 0.64, 0.76]
    elif class_code == "HC":
        finger_heights = [0.50, 0.54, 0.56, 0.53, 0.48]
        x_positions = [0.28, 0.40, 0.52, 0.64, 0.76]
    elif class_code == "WF":
        finger_heights = [0.72, 0.80, 0.83, 0.77, 0.68]
        x_positions = [0.22, 0.35, 0.48, 0.61, 0.74]
    else:
        finger_heights = [0.72, 0.80, 0.83, 0.77, 0.68]
        x_positions = [0.32, 0.45, 0.58, 0.71, 0.84]

    for x, y in zip(x_positions, finger_heights, strict=True):
        ax.plot([x, x], [0.50, y], color="black", linewidth=3)

    ax.plot([0.34, 0.18], [0.32, 0.46], color="black", linewidth=3)
    ax.plot([0.34, 0.22], [0.28, 0.18], color="black", linewidth=3)

    if class_code == "WF":
        ax.plot([0.5, 0.28], [0.18, 0.08], color="tab:blue", linewidth=4)
    elif class_code == "WE":
        ax.plot([0.5, 0.72], [0.18, 0.08], color="tab:orange", linewidth=4)
    else:
        ax.plot([0.5, 0.5], [0.18, 0.02], color="black", linewidth=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Predicted gesture: {class_code}")


def _find_predicted_examples(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_channels: list[int],
    win_ms: float,
) -> list[dict[str, object]]:
    X_train, y_train, _, _ = build_feature_matrix(
        train_df,
        win_ms=win_ms,
        hop_fraction=HOP_FRACTION,
        filtered=False,
        channel_indices=selected_channels,
        feature_set="basic",
    )
    clf = train_lda(X_train, y_train)

    examples: dict[str, dict[str, object]] = {}
    hop_ms = win_ms * HOP_FRACTION

    for row in test_df.to_dict(orient="records"):
        signal, fs = read_record(row["record_path"])
        signal = signal[:, selected_channels]
        for window in sliding_windows(signal, fs=fs, win_ms=win_ms, hop_ms=hop_ms):
            features = extract_feature_set(window, feature_set="basic").reshape(1, -1)
            pred_label = int(clf.predict(features)[0])
            pred_class = LABEL_TO_CLASS[pred_label]
            if pred_class not in examples:
                examples[pred_class] = {
                    "predicted_class": pred_class,
                    "true_class": row["class"],
                    "window": window,
                    "fs": fs,
                }
            if len(examples) == len(LABEL_TO_CLASS):
                return [examples[class_code] for class_code in sorted(examples)]

    return [examples[class_code] for class_code in sorted(examples)]


def _save_frame(example: dict[str, object], output_path: Path) -> None:
    window = example["window"]
    fs = float(example["fs"])
    class_code = str(example["predicted_class"])
    true_class = str(example["true_class"])
    samples_to_show = min(int(fs * 0.2), window.shape[0])
    time_axis = [sample_idx / fs for sample_idx in range(samples_to_show)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(time_axis, window[:samples_to_show, 0], linewidth=1.5, color="tab:green")
    axes[0].set_title(f"Signal window - true {true_class}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    _draw_hand(axes[1], class_code)

    fig.suptitle(f"LDA proof of concept - predicted {class_code}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _try_save_gif(frame_paths: list[Path], output_path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    images = [Image.open(frame_path) for frame_path in frame_paths]
    if not images:
        return False
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=700,
        loop=0,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple hand-movement proof of concept from GRABMyo windows.")
    parser.add_argument("--quick", action="store_true", help="Use the quick participant subset used by channel_sweep.")
    parser.add_argument("--window-ms", type=float, default=200.0, help="Sliding window size in milliseconds.")
    parser.add_argument("--channel-sweep-csv", type=Path, default=CHANNEL_SWEEP_CSV, help="Reference channel_sweep CSV.")
    return parser.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    df = load_index(INDEX_4CLASSES_CSV)
    train_df, test_df = split_by_participants(df, n_test_participants=2)
    if args.quick:
        train_df, test_df = _make_quick_subset(train_df, test_df)

    selected_channels = _load_channels(args.channel_sweep_csv, quick=args.quick, channels_count=12)
    examples = _find_predicted_examples(
        train_df=train_df,
        test_df=test_df,
        selected_channels=selected_channels,
        win_ms=args.window_ms,
    )
    if not examples:
        raise SystemExit("No predicted examples found for visualization.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for frame_idx, example in enumerate(examples, start=1):
        output_path = PLOTS_DIR / f"poc_hand_frame_{frame_idx:03d}.png"
        _save_frame(example, output_path)
        frame_paths.append(output_path)
        print(f"Saved: {output_path}")

    gif_path = PLOTS_DIR / "poc_hand_demo.gif"
    if _try_save_gif(frame_paths, gif_path):
        print(f"Saved: {gif_path}")
    else:
        print("GIF was not created because Pillow is unavailable.")


if __name__ == "__main__":
    main()
