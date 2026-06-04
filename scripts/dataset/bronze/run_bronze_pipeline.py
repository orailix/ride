"""Build Bronze datasets from raw inputs using manifests from the pipeline config."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.dataset.pipeline.helpers import (
    execute_manifest,
    read_yaml,
    load_manifest,
    write_yaml,
)


def main() -> None:
    """Parse CLI arguments and execute the configured Bronze pipeline manifests."""
    parser = argparse.ArgumentParser(description="Run the bronze pipeline.")
    parser.add_argument("config", help="Path to the pipeline configuration file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    enriched_config = dict(config)
    enriched_config["run_timestamp"] = run_timestamp


    raw_dir = Path("data/raw")
    bronze_dir = Path("data/bronze")

    raw_meta_path = raw_dir / "metadata.yaml"
    if raw_meta_path.exists():
        raw_metadata = read_yaml(raw_meta_path)
    else:
        raw_metadata = {
            "sources": [],
        }
    raw_metadata["config_path"] = str(config_path)
    raw_metadata["config"] = enriched_config

    bronze_meta_path = bronze_dir / "metadata.yaml"
    if bronze_meta_path.exists():
        bronze_metadata = read_yaml(bronze_meta_path)
    else:
        bronze_metadata = {
            "config_path": str(config_path),
            "config": enriched_config,
            "sources": raw_metadata.get("sources", []),
            "datasets": [],
        }

    bronze_metadata.setdefault("sources", raw_metadata.get("sources", []))
    bronze_metadata.setdefault("datasets", [])
    bronze_metadata["config_path"] = str(config_path)
    bronze_metadata["config"] = enriched_config

    events_months = list(config.get("events", {}).get("months", []) or [])

    # Apply manifests
    manifest_root = Path(config.get("bronze_manifest_dir", "manifests/bronze"))
    op_nodes_manifest_path = manifest_root / "op_nodes.yaml"
    line_sections_manifest_path = manifest_root / "line_sections.yaml"
    events_manifest_path = manifest_root / "events.yaml"

    execute_manifest(
        manifest=load_manifest(op_nodes_manifest_path),
        manifest_path=op_nodes_manifest_path,
        metadata=bronze_metadata,
    )

    execute_manifest(
        manifest=load_manifest(line_sections_manifest_path),
        manifest_path=line_sections_manifest_path,
        metadata=bronze_metadata,
    )
    if events_months:
        events_manifest = load_manifest(events_manifest_path)
        for month in events_months:
            execute_manifest(
                manifest=events_manifest,
                manifest_path=events_manifest_path,
                metadata=bronze_metadata,
                wildcards={"main": month, "output": month},
            )
    elif events_manifest_path.exists():
        print(f"Skipping events manifest '{events_manifest_path}' (no event downloads).")

    write_yaml(bronze_meta_path, bronze_metadata)
    write_yaml(raw_meta_path, raw_metadata)


if __name__ == "__main__":
    main()
