from emg_sampling.paths import ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    print("emg-sampling project is ready.")
    print("Use scripts/run_baseline.py or scripts/run_window_sweep.py to run experiments.")


if __name__ == "__main__":
    main()
