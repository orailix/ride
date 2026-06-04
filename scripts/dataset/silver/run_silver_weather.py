"""Build the Silver weather table from downloaded raw weather chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.pipeline.helpers import execute_manifest, load_manifest, read_yaml, write_yaml


def main() -> None:
    """Parse CLI arguments, run the Silver weather manifest, and update metadata."""
    parser = argparse.ArgumentParser(description="Run the silver weather manifest (concatenate raw weather files).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/silver/weather.yaml"),
        help="Path to the silver weather manifest.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    silver_meta_path = Path("data/silver/metadata.yaml")
    if silver_meta_path.exists():
        silver_metadata = read_yaml(silver_meta_path)
    else:
        silver_metadata = {"datasets": []}
    silver_metadata.setdefault("datasets", [])

    raw_meta_path = Path("data/raw/metadata.yaml")
    if raw_meta_path.exists():
        raw_metadata = read_yaml(raw_meta_path)
        weather_sources = [
            source
            for source in raw_metadata.get("sources", [])
            if source.get("dataset") == "weather_raw"
        ]
        if weather_sources:
            weather_raw_metadata = {"sources": weather_sources}
            if "config_path" in raw_metadata:
                weather_raw_metadata["config_path"] = raw_metadata["config_path"]
            if "config" in raw_metadata:
                weather_raw_metadata["config"] = raw_metadata["config"]
            silver_metadata["raw_weather_run"] = weather_raw_metadata

    execute_manifest(
        manifest=manifest,
        manifest_path=args.manifest,
        metadata=silver_metadata,
    )
    write_yaml(silver_meta_path, silver_metadata)


if __name__ == "__main__":
    main()
