"""Evaluate the graph-event model on the Gold test split.

This model predicts directly from the graph-event tables and shared travel time
statistics. The produced prediction table is evaluated with the same Gold core
evaluator used by the other benchmark models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark.models.graph_event import load_graph_event_tables, run_graph_events
from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.dataset.pipeline.helpers import write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the graph-event regression model on the test split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing graph-event tables under test/ plus shared node_links/travel_time_samples.",
    )
    parser.add_argument(
        "--test-eval-table",
        type=Path,
        required=True,
        help="Path to test_eval_table.parquet used for final evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where model outputs/metrics are written.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_graph_event_tables(
        data_dir=args.data_dir,
        test_eval_table=args.test_eval_table,
    )

    pred_table = run_graph_events(
        journeys=data["journeys"],
        events=data["events"],
        node_links=data["node_links"],
        travel_time_stats=data["travel_time_stats"],
        eval_table=data["eval_table"],
    )
    metrics = evaluate_delay_predictions(eval_table=data["eval_table"], predictions=pred_table)
    metrics_path = args.output_dir / "eval_metrics.yaml"
    write_yaml(metrics_path, {"test": metrics})


if __name__ == "__main__":
    main()
