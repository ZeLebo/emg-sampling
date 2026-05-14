"""Readable proof-of-concept gesture visualization."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D

from emg_sampling.config import HOP_FRACTION, LABELS
from emg_sampling.data.grabmyo_loader import load_index, read_record
from emg_sampling.data.split import make_leakage_safe_split
from emg_sampling.experiments.common import build_feature_matrix
from emg_sampling.features.feature_sets import extract_feature_set
from emg_sampling.features.time_domain import sliding_windows
from emg_sampling.models.lda_baseline import train_lda
from emg_sampling.paths import INDEX_4CLASSES_CSV

GESTURE_CODES = ("WF", "WE", "HO", "HC")
GESTURE_NAME_MAP = {
    "WF": "Wrist Flexion",
    "WE": "Wrist Extension",
    "HO": "Hand Open",
    "HC": "Hand Close",
}
GESTURE_DESCRIPTION_MAP = {
    "WF": "Wrist Flexion (WF): bending the hand toward the palm side at the wrist joint.",
    "WE": "Wrist Extension (WE): bending the hand backward at the wrist joint.",
    "HO": "Hand Open (HO): fingers are extended and the hand is open.",
    "HC": "Hand Close (HC): fingers are flexed and the hand is closed.",
}
DEFAULT_SYNTHETIC_CHANNELS = {
    "WF": [3, 7, 12, 18, 20, 21],
    "WE": [4, 8, 13, 17, 19, 22],
    "HO": [1, 5, 9, 14, 16, 23],
    "HC": [2, 6, 10, 11, 15, 24],
}
LABEL_TO_CLASS = {value: key for key, value in LABELS.items()}


def gesture_name(gesture_code: str) -> str:
    """Returns human-readable gesture name."""
    if gesture_code not in GESTURE_NAME_MAP:
        raise ValueError(f"Unknown gesture code: {gesture_code}")
    return GESTURE_NAME_MAP[gesture_code]


def gesture_description(gesture_code: str) -> str:
    """Returns explanatory sentence for gesture."""
    if gesture_code not in GESTURE_DESCRIPTION_MAP:
        raise ValueError(f"Unknown gesture code: {gesture_code}")
    return GESTURE_DESCRIPTION_MAP[gesture_code]


def default_channels_for_gesture(gesture_code: str) -> list[int]:
    """Returns default synthetic channels for one gesture."""
    return list(DEFAULT_SYNTHETIC_CHANNELS[gesture_code])


def generate_synthetic_emg(
    gesture_code: str,
    fs: float,
    duration_s: float,
    channel_indices: list[int],
    random_state: int = 42,
) -> np.ndarray:
    """Generates pseudo-EMG for demonstration only."""
    if gesture_code not in GESTURE_CODES:
        raise ValueError(f"Unknown gesture code: {gesture_code}")

    rng = np.random.default_rng(random_state + hash(gesture_code) % 1000)
    samples = int(round(fs * duration_s))
    t = np.arange(samples, dtype=np.float32) / fs
    signal = np.zeros((samples, len(channel_indices)), dtype=np.float32)

    active_channels = set(default_channels_for_gesture(gesture_code))
    gesture_phase = {"WF": 0.15, "WE": 0.65, "HO": 1.10, "HC": 1.55}[gesture_code]

    for idx, channel in enumerate(channel_indices):
        base_noise = rng.normal(0.0, 0.035, size=samples).astype(np.float32)
        low_component = 0.03 * np.sin(2 * np.pi * (6.0 + idx * 0.4) * t + gesture_phase)
        mid_component = 0.02 * np.sin(2 * np.pi * (18.0 + idx) * t + 0.3 * idx)
        burst_envelope = 0.25 + 0.75 * np.sin(2 * np.pi * 2.2 * t + gesture_phase) ** 2
        activation = 1.0 if channel in active_channels else 0.45
        burst = activation * burst_envelope * rng.normal(0.0, 0.06, size=samples)
        signal[:, idx] = base_noise + low_component + mid_component + burst.astype(np.float32)

    return signal


def _display_channel_labels(channel_indices: list[int]) -> list[str]:
    return [f"CH {channel:02d}" for channel in channel_indices]


def plot_emg_channels(
    ax,
    signal: np.ndarray,
    fs: float,
    channel_indices: list[int],
    title: str = "EMG window by channels",
) -> None:
    """Plots vertically offset EMG traces by channel."""
    shown_signal = signal[:, :12]
    shown_channels = channel_indices[: shown_signal.shape[1]]
    if signal.shape[1] > 12:
        title = f"{title}\nShowing first 12 selected channels"

    time_ms = (np.arange(shown_signal.shape[0], dtype=np.float32) / fs) * 1000.0
    spacing = max(0.25, float(np.max(np.std(shown_signal, axis=0)) * 8.0))
    offsets = np.arange(len(shown_channels))[::-1] * spacing

    for idx, (channel, offset) in enumerate(zip(shown_channels, offsets, strict=True)):
        trace = shown_signal[:, idx] + offset
        ax.plot(time_ms, trace, linewidth=1.2, color="#157a6e")
        ax.text(
            time_ms[0] - 10.0,
            offset,
            f"CH {channel:02d}",
            va="center",
            ha="right",
            fontsize=9,
            color="#153243",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Time, ms")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.03)


def plot_channel_layout(
    ax,
    selected_channels: list[int],
    total_channels: int = 24,
) -> None:
    """Plots a schematic 4-ring by 6-electrode forearm layout."""
    ax.set_title("Channel layout", fontsize=12, loc="left")
    ax.text(
        0.02,
        0.98,
        "Approximate 4-ring x 6-electrode layout",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5c6770",
    )

    forearm_outline = patches.Polygon(
        [
            (0.30, 0.06),
            (0.68, 0.06),
            (0.80, 0.20),
            (0.84, 0.50),
            (0.80, 0.80),
            (0.68, 0.94),
            (0.30, 0.94),
            (0.18, 0.80),
            (0.14, 0.50),
            (0.18, 0.20),
        ],
        closed=True,
        linewidth=2.0,
        edgecolor="#495057",
        facecolor="#f1f3f5",
        joinstyle="round",
    )
    ax.add_patch(forearm_outline)

    ring_ys = np.linspace(0.20, 0.80, 4)
    ring_angles = np.deg2rad([205, 235, 270, 305, 335, 25])
    positions: dict[int, tuple[float, float]] = {}
    channel_id = 1
    for ring_idx, y in enumerate(ring_ys):
        ellipse_width = 0.56 + 0.06 * np.cos((ring_idx - 1.5) / 1.5)
        ellipse_height = 0.13 + 0.01 * ring_idx
        ax.add_patch(
            patches.Ellipse(
                (0.49, y),
                width=ellipse_width,
                height=ellipse_height,
                linewidth=1.0,
                linestyle="--",
                edgecolor="#c0c7cf",
                facecolor="none",
            )
        )
        for angle in ring_angles:
            if channel_id > total_channels:
                break
            x = 0.49 + 0.5 * ellipse_width * np.cos(angle)
            y_pos = y + 0.5 * ellipse_height * np.sin(angle)
            positions[channel_id] = (float(x), float(y_pos))
            channel_id += 1

    selected_set = set(selected_channels)
    for channel_id, (x, y) in positions.items():
        selected = channel_id in selected_set
        circle = patches.Circle(
            (x, y),
            radius=0.028 if selected else 0.023,
            facecolor="#d62828" if selected else "#adb5bd",
            edgecolor="#212529",
            linewidth=1.0,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            f"{channel_id:02d}",
            va="center",
            ha="center",
            fontsize=7,
            color="white" if selected else "#212529",
            fontweight="bold" if selected else None,
        )

    ax.annotate(
        "Wrist",
        xy=(0.49, 0.05),
        xytext=(0.82, 0.08),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#495057"},
        fontsize=8,
        color="#495057",
    )
    ax.annotate(
        "4 rings",
        xy=(0.86, 0.67),
        xytext=(0.83, 0.90),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#495057"},
        fontsize=8,
        color="#495057",
    )
    ax.annotate(
        "6 electrodes per ring",
        xy=(0.72, 0.23),
        xytext=(0.76, 0.34),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#495057"},
        fontsize=8,
        color="#495057",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_neutral_hand(ax) -> None:
    wrist = patches.Circle((0.38, 0.48), 0.02, facecolor="#adb5bd", edgecolor="none", alpha=0.8)
    palm = patches.FancyBboxPatch(
        (0.42, 0.38),
        0.18,
        0.18,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.8,
        linestyle="--",
        edgecolor="#adb5bd",
        facecolor="none",
    )
    ax.add_patch(wrist)
    ax.add_patch(palm)
    for x in [0.45, 0.49, 0.53, 0.57]:
        ax.plot([x, x], [0.56, 0.78], linestyle="--", color="#adb5bd", linewidth=1.5)
    ax.plot([0.44, 0.36], [0.42, 0.55], linestyle="--", color="#adb5bd", linewidth=1.5)


def _draw_open_hand(ax, angle_deg: float = 0.0, color: str = "#1d3557", annotate: str | None = None) -> None:
    base = Affine2D().rotate_deg_around(0.38, 0.48, angle_deg) + ax.transData
    forearm = patches.FancyBboxPatch(
        (0.06, 0.42),
        0.32,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=2.4,
        edgecolor=color,
        facecolor="#d9e7f5",
        transform=base,
    )
    wrist = patches.Circle((0.38, 0.48), 0.028, facecolor="#edf2f4", edgecolor=color, linewidth=2.0, transform=base)
    palm = patches.FancyBboxPatch(
        (0.40, 0.37),
        0.20,
        0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2.6,
        edgecolor=color,
        facecolor="#f8edeb",
        transform=base,
    )
    ax.add_patch(forearm)
    ax.add_patch(wrist)
    ax.add_patch(palm)

    for idx, x in enumerate([0.43, 0.48, 0.53, 0.58]):
        finger = patches.FancyBboxPatch(
            (x, 0.60),
            0.035,
            0.22 + idx * 0.01,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=2.0,
            edgecolor=color,
            facecolor="#f8edeb",
            transform=base,
        )
        ax.add_patch(finger)

    thumb = patches.FancyBboxPatch(
        (0.34, 0.42),
        0.13,
        0.04,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=2.0,
        edgecolor=color,
        facecolor="#f8edeb",
        transform=Affine2D().rotate_deg_around(0.38, 0.44, -30 + angle_deg) + ax.transData,
    )
    ax.add_patch(thumb)

    if annotate is not None:
        ax.annotate(
            annotate,
            xy=(0.63, 0.73),
            xytext=(0.76, 0.86),
            arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": color},
            fontsize=11,
            color=color,
            fontweight="bold",
        )


def _draw_closed_hand(ax, color: str = "#7f5539") -> None:
    forearm = patches.FancyBboxPatch(
        (0.06, 0.42),
        0.32,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=2.4,
        edgecolor=color,
        facecolor="#e8d8c4",
    )
    wrist = patches.Circle((0.38, 0.48), 0.028, facecolor="#f1f3f5", edgecolor=color, linewidth=2.0)
    fist = patches.Ellipse((0.55, 0.49), 0.25, 0.20, edgecolor=color, facecolor="#f8edeb", linewidth=2.6)
    thumb = patches.FancyBboxPatch(
        (0.44, 0.38),
        0.11,
        0.05,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=2.0,
        edgecolor=color,
        facecolor="#f8edeb",
        transform=Affine2D().rotate_deg_around(0.44, 0.40, -28) + ax.transData,
    )
    ax.add_patch(forearm)
    ax.add_patch(wrist)
    ax.add_patch(fist)
    ax.add_patch(thumb)
    for x in [0.50, 0.55, 0.60, 0.65]:
        ax.add_patch(
            patches.Arc((x, 0.58), 0.06, 0.06, theta1=180, theta2=360, color=color, linewidth=2.0)
        )
    ax.text(0.72, 0.80, "Hand Close", fontsize=11, color=color, fontweight="bold")


def plot_hand_gesture(ax, gesture_code: str) -> None:
    """Plots readable forearm-hand gesture schematic."""
    ax.set_title("Gesture visualization", fontsize=12, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if gesture_code in {"WF", "WE"}:
        _draw_neutral_hand(ax)
        angle = -24 if gesture_code == "WF" else 24
        annotation = "Flexion" if gesture_code == "WF" else "Extension"
        _draw_open_hand(ax, angle_deg=angle, color="#0b6e4f", annotate=annotation)
    elif gesture_code == "HO":
        _draw_open_hand(ax, angle_deg=0.0, color="#0b6e4f")
        ax.text(0.76, 0.84, "Hand Open", fontsize=11, color="#0b6e4f", fontweight="bold")
    elif gesture_code == "HC":
        _draw_closed_hand(ax)
    else:
        raise ValueError(f"Unknown gesture code: {gesture_code}")


def create_poc_figure(
    signal: np.ndarray,
    fs: float,
    selected_channels: list[int],
    predicted_gesture: str,
    confidence: float | None = None,
    synthetic: bool = False,
    output_path: Path | None = None,
) -> None:
    """Creates multi-panel demonstration figure."""
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = GridSpec(
        nrows=6,
        ncols=6,
        figure=fig,
        height_ratios=[0.8, 2.5, 2.5, 0.2, 2.6, 0.8],
        width_ratios=[1.4, 1.4, 1.4, 1.1, 1.1, 1.1],
    )

    title_ax = fig.add_subplot(gs[0, :])
    signal_ax = fig.add_subplot(gs[1:3, :4])
    layout_ax = fig.add_subplot(gs[1:3, 4:])
    hand_ax = fig.add_subplot(gs[4, :])
    footer_ax = fig.add_subplot(gs[5, :])

    title_ax.axis("off")
    footer_ax.axis("off")

    full_name = gesture_name(predicted_gesture)
    title_ax.text(
        0.0,
        0.72,
        f"Predicted gesture: {full_name} ({predicted_gesture})",
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="center",
        color="#102a43",
    )
    confidence_text = f"{confidence:.2f}" if confidence is not None else "n/a"
    title_ax.text(
        0.0,
        0.22,
        f"Confidence: {confidence_text}",
        fontsize=12,
        ha="left",
        va="center",
        color="#334e68",
    )
    if synthetic:
        title_ax.text(
            0.52,
            0.22,
            "Synthetic EMG window for visualization only",
            fontsize=11,
            ha="left",
            va="center",
            color="#9c6644",
            style="italic",
        )

    plot_emg_channels(signal_ax, signal=signal, fs=fs, channel_indices=selected_channels)
    plot_channel_layout(layout_ax, selected_channels=selected_channels)
    plot_hand_gesture(hand_ax, predicted_gesture)
    footer_ax.text(
        0.0,
        0.5,
        gesture_description(predicted_gesture),
        fontsize=12,
        ha="left",
        va="center",
        color="#243b53",
    )

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_selected_channels(channel_sweep_csv: Path, split_mode: str, channels_count: int = 12) -> list[int]:
    sweep_df = pd.read_csv(channel_sweep_csv)
    matches = sweep_df[
        (sweep_df["method"] == "greedy")
        & (sweep_df["channels_count"] == channels_count)
        & (sweep_df["split_mode"] == split_mode)
    ].sort_values("macro_f1", ascending=False)
    if matches.empty:
        raise ValueError(
            f"No greedy {channels_count}-channel configuration found for split_mode={split_mode} in {channel_sweep_csv}"
        )
    return list(ast.literal_eval(matches.iloc[0]["selected_channels"]))


def _find_dataset_example(
    gesture_code: str,
    channel_sweep_csv: Path,
    split_mode: str,
    win_ms: float,
) -> dict[str, object]:
    df = load_index(INDEX_4CLASSES_CSV)
    split = make_leakage_safe_split(df, mode=split_mode)
    selected_channel_indices = _load_selected_channels(
        channel_sweep_csv,
        split_mode=split_mode,
        channels_count=12,
    )

    X_train, y_train, _, _ = build_feature_matrix(
        split.final_train_df,
        win_ms=win_ms,
        hop_fraction=HOP_FRACTION,
        filtered=False,
        channel_indices=selected_channel_indices,
        feature_set="basic",
    )
    clf = train_lda(X_train, y_train)
    hop_ms = win_ms * HOP_FRACTION
    target_label = LABELS[gesture_code]

    for row in split.final_test_df.to_dict(orient="records"):
        if row["y"] != target_label:
            continue
        signal, fs = read_record(row["record_path"])
        signal = signal[:, selected_channel_indices]
        for window in sliding_windows(signal, fs=fs, win_ms=win_ms, hop_ms=hop_ms):
            features = extract_feature_set(window, feature_set="basic").reshape(1, -1)
            predicted_label = int(clf.predict(features)[0])
            if predicted_label != target_label:
                continue
            confidence = None
            if hasattr(clf, "predict_proba"):
                confidence = float(np.max(clf.predict_proba(features)[0]))
            return {
                "signal": window,
                "fs": fs,
                "selected_channels": [channel_idx + 1 for channel_idx in selected_channel_indices],
                "predicted_gesture": gesture_code,
                "confidence": confidence,
                "synthetic": False,
            }

    raise RuntimeError(f"Could not find a dataset window predicted as {gesture_code}")


def generate_visualization_payload(
    gesture_code: str,
    synthetic: bool,
    channel_sweep_csv: Path,
    split_mode: str,
    win_ms: float,
    selected_channels: list[int] | None = None,
) -> dict[str, object]:
    """Returns data needed for create_poc_figure."""
    if synthetic:
        channels = selected_channels or default_channels_for_gesture(gesture_code)
        signal = generate_synthetic_emg(
            gesture_code=gesture_code,
            fs=2000.0,
            duration_s=0.22,
            channel_indices=channels,
        )
        return {
            "signal": signal,
            "fs": 2000.0,
            "selected_channels": channels,
            "predicted_gesture": gesture_code,
            "confidence": None,
            "synthetic": True,
        }

    dataset_payload = _find_dataset_example(
        gesture_code=gesture_code,
        channel_sweep_csv=channel_sweep_csv,
        split_mode=split_mode,
        win_ms=win_ms,
    )
    return dataset_payload


def try_save_gif(frame_paths: list[Path], output_path: Path) -> bool:
    """Tries to compose saved PNG files into a GIF."""
    try:
        from PIL import Image
    except ImportError:
        return False

    images = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            images.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
    if not images:
        return False
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=850,
        loop=0,
    )
    return True
