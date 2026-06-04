"""Build the model-specific Gold sequential dataset from a Silver release and Gold core."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.gold.sequential_data import build_sequential_dataset


def main() -> None:
    """Parse CLI arguments and launch Gold sequential dataset construction."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the model-specific Gold sequential dataset. "
            "Outputs are train/test sequential arrays plus scheme and normalization metadata."
        )
    )
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver"), help="Root silver directory containing events/, journeys/, and static/ files.")
    parser.add_argument("--nb-past-events", type=int, default=5, help="Number of closest past events (ids) to include as columns.")
    parser.add_argument("--n-next-links", type=int, default=10, help="Number of next path-link tokens to export per row.")
    parser.add_argument("--station-embedding-dim", type=int, default=8, help="Number of station embedding components.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for train/test sequential arrays, scheme.yaml, and normalization.yaml.")
    parser.add_argument("--dataset-core-spec", type=Path, required=True, help="Path to dataset core spec YAML (train/test snapshots + core gold parameters).")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--missing-event-placeholder", type=int, default=-1, help="Placeholder value used when a past/future event id is missing.")
    args = parser.parse_args()

    build_sequential_dataset(
        silver_dir=args.silver_dir,
        nb_past_events=args.nb_past_events,
        n_next_links=args.n_next_links,
        station_embedding_dim=args.station_embedding_dim,
        output_dir=args.output_dir,
        dataset_core_spec=args.dataset_core_spec,
        missing_event_placeholder=args.missing_event_placeholder,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
