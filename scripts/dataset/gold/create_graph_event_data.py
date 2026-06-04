"""Build the model-specific Gold graph-event dataset from Silver and Gold core inputs."""

import argparse
from pathlib import Path

from src.dataset.gold.graph_event_data import build_graph_event_dataset


def main() -> None:
    """Parse CLI arguments and launch Gold graph-event dataset construction."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the model-specific Gold graph-event dataset from Silver sources and a Gold core spec."
        )
    )
    parser.add_argument("--dataset-core-spec", type=Path, required=True, help="Path to dataset core spec YAML (train/test snapshots + core gold parameters).")
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver"), help="Root silver directory containing events/, journeys/, and static/ files.")
    parser.add_argument("--missing-event-placeholder", type=int, default=-1, help="Placeholder value used when a past/future event id is missing.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for graph-event outputs.")
    args = parser.parse_args()
    build_graph_event_dataset(
        dataset_core_spec=args.dataset_core_spec,
        silver_dir=args.silver_dir,
        missing_event_placeholder=args.missing_event_placeholder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
