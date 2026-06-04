"""Download raw Open-Meteo weather data for configured event months and Silver nodes."""

from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.dataset.pipeline.helpers import read_yaml, write_yaml
from src.dataset.silver.weather_utils import ARCHIVE_URL, REQUIRED_VARS, fetch_weather_batches


def main() -> None:
    """Parse CLI arguments, download one weather chunk, and append raw metadata."""
    parser = argparse.ArgumentParser(
        description="Download one batch of Open-Meteo archive data for operational points."
    )
    parser.add_argument("config", help="Path to the dataset pipeline configuration file.")
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="Starting index of the op_nodes slice to fetch (0-based).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of nodes to include in this batch (max 100).",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Brussels",
        help="Timezone passed to the Open-Meteo API.",
    )
    parser.add_argument(
        "--nodes-path",
        type=Path,
        default=Path("data/silver/static/op_nodes.parquet"),
        help="Path to silver op_nodes parquet with op_id/lat/lon.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/weather"),
        help="Directory where the raw weather parquet will be written.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)
    months = sorted(str(month) for month in config["events"]["months"])
    first_year, first_month = int(months[0][:4]), int(months[0][4:])
    last_year, last_month = int(months[-1][:4]), int(months[-1][4:])
    start_date = (date(first_year, first_month, 1) - timedelta(days=1)).isoformat()
    end_day = calendar.monthrange(last_year, last_month)[1]
    end_date = (date(last_year, last_month, end_day) + timedelta(days=1)).isoformat()

    output_path = fetch_weather_batches(
        start_date=start_date,
        end_date=end_date,
        chunk_start=args.chunk_start,
        chunk_size=args.chunk_size,
        timezone=args.timezone,
        nodes_path=args.nodes_path,
        raw_dir=args.raw_dir,
    )

    raw_meta_path = Path("data/raw/metadata.yaml")
    if raw_meta_path.exists():
        raw_metadata = read_yaml(raw_meta_path)
    else:
        raw_metadata = {"sources": []}
    raw_metadata.setdefault("sources", [])

    entry = {
        "dataset": "weather_raw",
        "url": ARCHIVE_URL,
        "path": str(output_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "extra": {
            "config_path": str(config_path),
            "start_date": start_date,
            "end_date": end_date,
            "chunk_start": args.chunk_start,
            "chunk_size": args.chunk_size,
            "timezone": args.timezone,
            "variables": list(REQUIRED_VARS),
        },
    }
    raw_metadata["sources"].append(entry)
    write_yaml(raw_meta_path, raw_metadata)


if __name__ == "__main__":
    main()
