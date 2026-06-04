"""Run the graph-event model and write a test-eval summary.

The graph-event model is deterministic in this benchmark setup, so this wrapper
runs the evaluation entry point once and stores the result in the same
``test_eval_summary.yaml`` format used by repeated-seed models.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.benchmark.utils.test_eval import aggregate_test_metrics, load_test_metrics_from_eval_yaml
from src.dataset.pipeline.helpers import write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-run test-eval wrapper for the graph-event model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing the graph-event evaluation tables.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where metrics and the summary are written.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[test-eval] running graph-event model in {args.output_dir}")

    command = [
        sys.executable,
        "-m",
        "scripts.benchmark.eval.graph_event",
        "--data-dir",
        str(args.data_dir),
        "--test-eval-table",
        str(args.test_eval_table),
        "--output-dir",
        str(args.output_dir),
    ]
    subprocess.run(command, check=True)

    summary = aggregate_test_metrics(
        per_seed_metrics=[load_test_metrics_from_eval_yaml(args.output_dir)]
    )
    write_yaml(args.output_dir / "test_eval_summary.yaml", summary)
    print(f"[test-eval] wrote summary to {args.output_dir / 'test_eval_summary.yaml'}")


if __name__ == "__main__":
    main()
