from __future__ import annotations

import _bootstrap  # noqa: F401

from emg_sampling.experiments.window_sweep import run_window_sweep
from emg_sampling.paths import INDEX_4CLASSES_CSV


def main() -> None:
    result = run_window_sweep(index_csv=INDEX_4CLASSES_CSV)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
