"""Build Silver datasets from Bronze inputs using manifests from the pipeline config."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from src.dataset.pipeline.helpers import (
    execute_manifest,
    read_yaml,
    load_manifest,
    write_yaml,
)


def main() -> None:
    """Parse CLI arguments and execute the configured Silver pipeline manifests."""
    parser = argparse.ArgumentParser(description="Run the silver pipeline.")
    parser.add_argument("config", help="Path to the pipeline configuration file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    enriched_config = dict(config)
    enriched_config["run_timestamp"] = run_timestamp

    silver_metadata: Dict[str, object] = {
        "config_path": str(config_path),
        "config": enriched_config,
        "datasets": [],
    }

    bronze_metadata = read_yaml(Path("data/bronze") / "metadata.yaml")
    if bronze_metadata:
        silver_metadata["bronze_run"] = bronze_metadata

    manifest_root = Path(config.get("silver_manifest_dir", "manifests/silver"))
    op_nodes_manifest_path = manifest_root / "op_nodes.yaml"
    line_sections_manifest_path = manifest_root / "line_sections.yaml"
    node_links_manifest_path = manifest_root / "node_links.yaml"
    events_manifest_path = manifest_root / "events.yaml"
    journeys_manifest_path = manifest_root / "journeys.yaml"

    execute_manifest(
        manifest=load_manifest(op_nodes_manifest_path),
        manifest_path=op_nodes_manifest_path,
        metadata=silver_metadata,
    )

    execute_manifest(
        manifest=load_manifest(line_sections_manifest_path),
        manifest_path=line_sections_manifest_path,
        metadata=silver_metadata,
    )

    execute_manifest(
        manifest=load_manifest(node_links_manifest_path),
        manifest_path=node_links_manifest_path,
        metadata=silver_metadata,
    )

    events_months = list(config.get("events", {}).get("months", []) or [])
    if events_months:
        events_manifest = load_manifest(events_manifest_path)
        for month in events_months:
            execute_manifest(
                manifest=events_manifest,
                manifest_path=events_manifest_path,
                metadata=silver_metadata,
                wildcards={"main": month, "output": month},
            )
    elif events_manifest_path.exists():
        print(f"Skipping events manifest '{events_manifest_path}' (no months configured).")

    # Journeys per month (requires both bronze and silver events for that month)
    if events_months:
        journeys_manifest = load_manifest(journeys_manifest_path)
        for month in events_months:
            execute_manifest(
                manifest=journeys_manifest,
                manifest_path=journeys_manifest_path,
                metadata=silver_metadata,
                wildcards={
                    "main": month,
                    "output": month,
                    "bronze": month,
                    "silver_events": month,
                },
            )
    elif journeys_manifest_path.exists():
        print(f"Skipping journeys manifest '{journeys_manifest_path}' (no months configured).")

    write_yaml(Path("data/silver") / "metadata.yaml", silver_metadata)


if __name__ == "__main__":
    main()
