from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from emg_sampling.paths import CHANNEL_SWEEP_CSV, CHANNEL_SWEEP_MEDIUM_CSV, PLOTS_DIR, ensure_project_dirs
from emg_sampling.visualization.poc_hand import (
    GESTURE_CODES,
    create_poc_figure,
    default_channels_for_gesture,
    generate_visualization_payload,
    gesture_name,
    try_save_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate readable EMG gesture proof-of-concept figures.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--synthetic", action="store_true", help="Use synthetic EMG for visualization.")
    mode_group.add_argument("--from-dataset", action="store_true", help="Use dataset-derived EMG windows.")

    gesture_group = parser.add_mutually_exclusive_group(required=True)
    gesture_group.add_argument("--gesture", choices=list(GESTURE_CODES), help="Generate one gesture figure.")
    gesture_group.add_argument("--all", action="store_true", help="Generate one figure per gesture.")

    parser.add_argument("--quick", action="store_true", help="Use quick split when reading dataset examples.")
    parser.add_argument("--medium", action="store_true", help="Use medium split when reading dataset examples.")
    parser.add_argument("--window-ms", type=float, default=200.0, help="Sliding window size in milliseconds.")
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=None,
        help="Explicit channels to visualize, for example --channels 3 7 12 18",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for saved PNG/GIF files.",
    )
    parser.add_argument(
        "--channel-sweep-csv",
        type=Path,
        default=None,
        help="Reference channel_sweep CSV for dataset mode.",
    )
    return parser.parse_args()


def _split_mode(args: argparse.Namespace) -> str:
    if args.quick and args.medium:
        raise SystemExit("Use only one of --quick or --medium.")
    if args.quick:
        return "quick"
    if args.medium:
        return "medium"
    return "medium"


def _channel_sweep_csv(args: argparse.Namespace) -> Path:
    if args.channel_sweep_csv is not None:
        return args.channel_sweep_csv
    if args.quick:
        return CHANNEL_SWEEP_CSV
    return CHANNEL_SWEEP_MEDIUM_CSV


def _output_path(output_dir: Path, gesture_code: str) -> Path:
    return output_dir / f"poc_hand_{gesture_code}.png"


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_mode = _split_mode(args)
    synthetic = args.synthetic or not args.from_dataset
    gestures = list(GESTURE_CODES) if args.all else [args.gesture]
    channel_sweep_csv = _channel_sweep_csv(args)

    frame_paths: list[Path] = []
    for gesture_code in gestures:
        selected_channels = args.channels or default_channels_for_gesture(gesture_code)
        payload = generate_visualization_payload(
            gesture_code=gesture_code,
            synthetic=synthetic,
            channel_sweep_csv=channel_sweep_csv,
            split_mode=split_mode,
            win_ms=args.window_ms,
            selected_channels=selected_channels if synthetic else None,
        )
        output_path = _output_path(output_dir, gesture_code)
        create_poc_figure(
            signal=payload["signal"],
            fs=float(payload["fs"]),
            selected_channels=list(payload["selected_channels"]),
            predicted_gesture=str(payload["predicted_gesture"]),
            confidence=payload["confidence"],
            synthetic=bool(payload["synthetic"]),
            output_path=output_path,
        )
        frame_paths.append(output_path)
        print(f"Saved: {output_path}")
        print(f"Gesture: {gesture_name(gesture_code)} ({gesture_code})")

    if len(frame_paths) > 1:
        gif_path = output_dir / "poc_hand_demo.gif"
        if try_save_gif(frame_paths, gif_path):
            print(f"Saved: {gif_path}")
        else:
            print("GIF was not created because Pillow is unavailable.")


if __name__ == "__main__":
    main()
