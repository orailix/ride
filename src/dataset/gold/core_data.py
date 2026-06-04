"""Gold benchmark core construction utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset.pipeline.helpers import write_yaml
from src.dataset.gold.helpers import (
    TAIL_SAFETY_BUFFER_MIN,
    build_gold_eval_table,
    sample_train_test_snapshots,
)


def _month_range(start_day: str, end_day: str) -> list[str]:
    """Return YYYYMM month ids covering an inclusive day range."""
    start = pd.Timestamp(start_day).to_period("M")
    end = pd.Timestamp(end_day).to_period("M")
    return [p.strftime("%Y%m") for p in pd.period_range(start=start, end=end, freq="M")]


def build_core_dataset(
    *,
    start_train_day: str,
    end_train_day: str,
    start_test_day: str,
    end_test_day: str,
    n_train: int,
    n_test: int,
    n_future: int,
    idle_time_beg: int,
    idle_time_end: int,
    output_root: Path,
    seed: int = 42,
    events_dir: Path = Path("data/silver/events"),
    journeys_dir: Path = Path("data/silver/journeys"),
    missing_event_placeholder: int = -1,
    build_train_eval_table: bool = False,
    show_progress: bool = True,
) -> None:
    """Build a Gold benchmark core and write its specification and eval tables.

    Args:
        start_train_day: First calendar day eligible for training snapshots.
        end_train_day: Last calendar day eligible for training snapshots.
        start_test_day: First calendar day eligible for test snapshots.
        end_test_day: Last calendar day eligible for test snapshots.
        n_train: Number of training snapshot timestamps to sample.
        n_test: Number of test snapshot timestamps to sample.
        n_future: Number of future events predicted for each active train.
        idle_time_beg: Minutes before journey start included in activity
            windows.
        idle_time_end: Minutes after journey end included in activity windows.
        output_root: Directory where the core specification, evaluation tables,
            and metadata are written.
        seed: Random seed used for snapshot sampling.
        events_dir: Directory containing monthly Silver event parquet files.
        journeys_dir: Directory containing monthly Silver journey parquet files.
        missing_event_placeholder: Operational point id used to mark padded
            future event slots.
        build_train_eval_table: Whether to also write a train evaluation table.
        show_progress: Whether to display progress bars while building tables.

    Returns:
        None. The function writes files under `output_root`.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_core_spec_output = output_root / "dataset_core_spec.yaml"
    test_eval_output = output_root / "test_eval_table.parquet"
    train_eval_output = output_root / "train_eval_table.parquet"
    metadata_output = output_root / "metadata.yaml"

    # Load journey boundaries for the months needed to sample active train/test
    # snapshot timestamps.
    months = sorted(
        set(
            _month_range(start_train_day, end_train_day)
            + _month_range(start_test_day, end_test_day)
        )
    )
    journeys = pd.concat(
        [
                pd.read_parquet(
                    journeys_dir / f"journeys_{month}.parquet",
                    columns=["service_date", "start_observed_ts", "end_observed_ts", "start_planned_ts"],
                )
            for month in months
        ],
        ignore_index=True,
    )

    # Sample fixed snapshot timestamps and persist the core specification used
    # by all model-specific Gold builders.
    splits = sample_train_test_snapshots(
        start_train_day=start_train_day,
        end_train_day=end_train_day,
        start_test_day=start_test_day,
        end_test_day=end_test_day,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        journeys=journeys,
        idle_time_beg=idle_time_beg,
        idle_time_end=idle_time_end,
    )
    payload = {
        "start_train_day": start_train_day,
        "end_train_day": end_train_day,
        "start_test_day": start_test_day,
        "end_test_day": end_test_day,
        "n_train": n_train,
        "n_test": n_test,
        "n_future": n_future,
        "idle_time_beg": idle_time_beg,
        "idle_time_end": idle_time_end,
        "tail_safety_buffer_min": TAIL_SAFETY_BUFFER_MIN,
        "seed": seed,
        **splits,
    }
    write_yaml(dataset_core_spec_output, payload)
    print(
        f"[gold_core] Wrote dataset core spec to {dataset_core_spec_output} "
        f"(train={len(splits['train_snapshots'])}, test={len(splits['test_snapshots'])})"
    )

    # Materialize the standardized evaluation reference table for the test split
    # and optionally for the train split.
    gold_eval = build_gold_eval_table(
        events_dir=events_dir,
        journeys_dir=journeys_dir,
        snapshot_config=payload,
        missing_event_placeholder=missing_event_placeholder,
        build_train_eval_table=build_train_eval_table,
        show_progress=show_progress,
    )
    gold_eval["test_eval_table"].to_parquet(test_eval_output, index=False)
    if build_train_eval_table:
        gold_eval["train_eval_table"].to_parquet(train_eval_output, index=False)
    print(f"[gold_core] Wrote gold eval tables to {output_root}")

    # Record the inputs, parameters, outputs, and row counts for reproducibility.
    metadata = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "inputs": {
            "events_dir": str(events_dir),
            "journeys_dir": str(journeys_dir),
        },
        "parameters": {
            "start_train_day": start_train_day,
            "end_train_day": end_train_day,
            "start_test_day": start_test_day,
            "end_test_day": end_test_day,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_future": int(n_future),
            "idle_time_beg": int(idle_time_beg),
            "idle_time_end": int(idle_time_end),
            "tail_safety_buffer_min": int(TAIL_SAFETY_BUFFER_MIN),
            "missing_event_placeholder": int(missing_event_placeholder),
            "seed": int(seed),
            "build_train_eval_table": bool(build_train_eval_table),
        },
        "outputs": {
            "dataset_core_spec_path": str(dataset_core_spec_output),
            "test_eval_table_path": str(test_eval_output),
            "train_eval_table_path": str(train_eval_output) if build_train_eval_table else None,
        },
        "counts": {
            "n_train_snapshots": int(len(splits["train_snapshots"])),
            "n_test_snapshots": int(len(splits["test_snapshots"])),
            "n_test_eval_rows": int(len(gold_eval["test_eval_table"])),
            "n_train_eval_rows": int(len(gold_eval["train_eval_table"])) if build_train_eval_table else 0,
        },
    }
    write_yaml(metadata_output, metadata)
    print(f"[gold_core] Wrote metadata to {metadata_output}")
