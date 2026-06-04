"""Evaluate the last-known-delay translation baseline.

The baseline copies the delay known at the snapshot time into each requested
future horizon and evaluates those predictions with the shared Gold core
evaluation table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmark.models.translation import build_last_known_baseline_predictions
from src.dataset.pipeline.helpers import write_yaml
from src.benchmark.utils.evaluation import evaluate_delay_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate last-known-delay baseline on eval tables.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where model metrics are written.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_eval = pd.read_parquet(args.test_eval_table)

    test_pred = build_last_known_baseline_predictions(test_eval)

    metrics = {
        "test": evaluate_delay_predictions(eval_table=test_eval, predictions=test_pred),
    }

    output_path = args.output_dir / "eval_metrics.yaml"
    write_yaml(output_path, metrics)
    print(f"[translation] Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
