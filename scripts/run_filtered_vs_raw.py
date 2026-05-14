from __future__ import annotations

import _bootstrap  # noqa: F401

from emg_sampling.experiments.filtered_vs_raw import run_filtered_vs_raw_sweep
from emg_sampling.paths import INDEX_4CLASSES_CSV


def main() -> None:
    result = run_filtered_vs_raw_sweep(index_csv=INDEX_4CLASSES_CSV)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
