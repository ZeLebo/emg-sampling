from __future__ import annotations

import _bootstrap  # noqa: F401

from emg_sampling.data.index_records import build_4class_index
from emg_sampling.paths import GRABMYO_RAW_DIR, INDEX_4CLASSES_CSV, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    dataset_root = GRABMYO_RAW_DIR / "1.1.0"
    if not dataset_root.exists():
        raise SystemExit(
            "GRABMyo data not found. Expected directory: "
            f"{dataset_root}. Download dataset first."
        )

    df = build_4class_index(dataset_root)
    if df.empty:
        raise SystemExit(f"No target records found under {dataset_root}")

    df.to_csv(INDEX_4CLASSES_CSV, index=False)
    print(f"Saved: {INDEX_4CLASSES_CSV}")
    print(f"Rows: {len(df)}")
    print(df["class"].value_counts().sort_index().to_string())
    print(f"Participants: {df['participant'].nunique()}")


if __name__ == "__main__":
    main()
