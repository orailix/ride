"""Build the model-specific Gold GNN dataset from a Silver release and Gold core."""

import argparse
from pathlib import Path

from src.dataset.gold.gnn_data import build_gnn_dataset


def main() -> None:
    """Parse CLI arguments and launch Gold GNN dataset construction."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the model-specific Gold GNN dataset. "
            "Outputs are train/test PyTorch Geometric graph chunks plus metadata."
        )
    )
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver"), help="Root silver directory containing events/, journeys/, and static/ files.")
    parser.add_argument("--station-embedding-dim", type=int, default=8, help="Number of station embedding components.")
    parser.add_argument("--nb-past-events", type=int, default=5, help="Number of closest past events used to derive train state.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for GNN node/edge parquet tables and metadata.")
    parser.add_argument("--dataset-core-spec", type=Path, required=True, help="Path to dataset core spec YAML (train/test snapshots + core gold parameters).")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--missing-event-placeholder", type=int, default=-1, help="Placeholder value used when a past/future event id is missing.")
    parser.add_argument("--graph-chunk-size", type=int, default=10000, help="Maximum number of snapshot graphs per saved chunk file.")
    args = parser.parse_args()

    build_gnn_dataset(
        silver_dir=args.silver_dir,
        station_embedding_dim=args.station_embedding_dim,
        nb_past_events=args.nb_past_events,
        output_dir=args.output_dir,
        dataset_core_spec=args.dataset_core_spec,
        missing_event_placeholder=args.missing_event_placeholder,
        show_progress=not args.no_progress,
        graph_chunk_size=args.graph_chunk_size,
    )


if __name__ == "__main__":
    main()
