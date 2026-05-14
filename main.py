from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emg_sampling.paths import ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    print("emg-sampling project is ready.")
    print("Use scripts/run_baseline.py or scripts/run_window_sweep.py to run experiments.")


if __name__ == "__main__":
    main()
