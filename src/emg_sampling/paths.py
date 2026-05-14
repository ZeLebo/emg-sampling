"""Filesystem paths used by the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GRABMYO_RAW_DIR = RAW_DATA_DIR / "grabmyo"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

INDEX_4CLASSES_CSV = PROCESSED_DATA_DIR / "grabmyo_index_4classes.csv"
BASELINE_RESULTS_CSV = TABLES_DIR / "baseline_results.csv"
WINDOW_SWEEP_CSV = TABLES_DIR / "window_sweep_results.csv"
FILTERED_VS_RAW_CSV = TABLES_DIR / "filtered_vs_raw_results.csv"
NOISE_EXPERIMENT_CSV = TABLES_DIR / "noise_experiment_results.csv"
CHANNEL_SWEEP_CSV = TABLES_DIR / "channel_sweep_results.csv"
FEATURE_SWEEP_CSV = TABLES_DIR / "feature_sweep_results.csv"
CHANNEL_SWEEP_MEDIUM_CSV = TABLES_DIR / "channel_sweep_medium_results.csv"
FEATURE_SWEEP_MEDIUM_CSV = TABLES_DIR / "feature_sweep_medium_results.csv"
CHANNEL_SWEEP_FILTERED_COMPARISON_CSV = TABLES_DIR / "channel_sweep_filtered_comparison.csv"
CHANNEL_STABILITY_CSV = TABLES_DIR / "channel_stability_results.csv"
BANDIT_SIMULATION_CSV = TABLES_DIR / "bandit_simulation_results.csv"
REPEATED_MEDIUM_EVALUATION_CSV = TABLES_DIR / "repeated_medium_evaluation_results.csv"


def ensure_project_dirs() -> None:
    """Creates required local directories for processed data and results."""
    for path in (PROCESSED_DATA_DIR, TABLES_DIR, PLOTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
