"""Download raw source files using source settings from the pipeline config."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from src.dataset.bronze.data_download import download_sources
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    """Parse CLI arguments, download configured sources, and write raw metadata."""
    parser = argparse.ArgumentParser(description="Download bronze-layer source files.")
    parser.add_argument("config", help="Path to the pipeline configuration file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    enriched_config = dict(config)
    enriched_config["run_timestamp"] = run_timestamp

    sources, _ = download_sources(config)

    metadata_payload: Dict[str, object] = {
        "config_path": str(config_path),
        "config": enriched_config,
        "sources": sources,
    }

    bronze_metadata = dict(metadata_payload)
    bronze_metadata["datasets"] = []

    raw_dir = Path("data/raw")
    bronze_dir = Path("data/bronze")
    write_yaml(raw_dir / "metadata.yaml", metadata_payload)
    write_yaml(bronze_dir / "metadata.yaml", bronze_metadata)


if __name__ == "__main__":
    main()
