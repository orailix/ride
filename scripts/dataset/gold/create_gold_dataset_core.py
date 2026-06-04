"""Build the shared Gold core used by all model-specific benchmark datasets."""

import argparse
from pathlib import Path

from src.dataset.gold.core_data import build_core_dataset


def main() -> None:
    """Parse CLI arguments and launch Gold core construction."""
    parser = argparse.ArgumentParser(
        description="Generate train/test snapshot splits and gold evaluation reference tables."
    )
    parser.add_argument("--start-train-day", required=True, help="Train split start day (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end-train-day", required=True, help="Train split end day (YYYY-MM-DD), inclusive.")
    parser.add_argument("--start-test-day", required=True, help="Test split start day (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end-test-day", required=True, help="Test split end day (YYYY-MM-DD), inclusive.")
    parser.add_argument("--n-train", required=True, type=int, help="Number of train snapshot timestamps.")
    parser.add_argument("--n-test", required=True, type=int, help="Number of test snapshot timestamps.")
    parser.add_argument("--n-future", required=True, type=int, help="Number of future events to predict.")
    parser.add_argument("--idle-time-beg", required=True, type=int, help="Minutes before first planned event to include.")
    parser.add_argument("--idle-time-end", required=True, type=int, help="Minutes after last observed event to include.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--events-dir", type=Path, default=Path("data/silver/events"), help="Directory containing silver events_<YYYYMM>.parquet.")
    parser.add_argument("--journeys-dir", type=Path, default=Path("data/silver/journeys"), help="Directory containing silver journeys_<YYYYMM>.parquet.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory containing dataset_core_spec.yaml, eval parquet tables, and metadata.yaml.")
    parser.add_argument("--missing-event-placeholder", type=int, default=-1, help="Placeholder value used for missing future events in eval table generation.")
    parser.add_argument("--build-train-eval-table", action="store_true", help="Also generate and export train_eval_table.parquet.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    build_core_dataset(
        start_train_day=args.start_train_day,
        end_train_day=args.end_train_day,
        start_test_day=args.start_test_day,
        end_test_day=args.end_test_day,
        n_train=args.n_train,
        n_test=args.n_test,
        n_future=args.n_future,
        idle_time_beg=args.idle_time_beg,
        idle_time_end=args.idle_time_end,
        output_root=args.output_root,
        seed=args.seed,
        events_dir=args.events_dir,
        journeys_dir=args.journeys_dir,
        missing_event_placeholder=args.missing_event_placeholder,
        build_train_eval_table=args.build_train_eval_table,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
