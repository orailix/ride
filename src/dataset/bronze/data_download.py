"""Bronze raw-source download utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


EVENTS_URL_TEMPLATE = (
    "https://fr.ftp.opendatasoft.com/infrabel/PunctualityHistory/"
    "Data_raw_punctuality_{month}.csv"
)
OP_NODES_URL = (
    "https://opendata.infrabel.be/api/explore/v2.1/catalog/datasets/"
    "operationele-punten-van-het-netwerk/exports/csv?lang=en&use_labels=true&delimiter=%3B"
)
LINE_SECTIONS_URL = (
    "https://opendata.infrabel.be/api/explore/v2.1/catalog/datasets/"
    "lijnsecties/exports/csv?lang=en&use_labels=true&delimiter=%3B"
)

MONTH_PATTERN = re.compile(r"^\d{6}$")


@dataclass(frozen=True)
class DownloadResult:
    """Metadata describing one downloaded raw source file."""
    dataset: str
    path: Path
    url: str
    downloaded_at: datetime
    size_bytes: int
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """Serialize the download metadata for manifests or logs."""
        payload = {
            "dataset": self.dataset,
            "url": self.url,
            "path": str(self.path),
            "downloaded_at": self.downloaded_at.isoformat(),
            "size_bytes": self.size_bytes,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


def _ensure_month(month: str) -> str:
    """Validate an event month in the YYYYMM format used by Infrabel files."""
    if month is None or not MONTH_PATTERN.match(month):
        raise ValueError(f"Month must be in YYYYMM format, got '{month}'")
    return month


def _write_bytes(destination: Path, content: bytes) -> int:
    """Write downloaded bytes and return the resulting file size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination.stat().st_size


def _download_csv(
    *,
    url: str,
    dataset: str,
    destination: Path,
    timeout: int = 60,
    extra: Optional[Dict[str, object]] = None,
) -> DownloadResult:
    """Download one CSV URL to disk and return its metadata."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    size = _write_bytes(destination, response.content)
    return DownloadResult(
        dataset=dataset,
        path=destination,
        url=url,
        downloaded_at=datetime.now(timezone.utc),
        size_bytes=size,
        extra=extra or {},
    )


def download_op_nodes(
    *,
    data_dir: Optional[Path] = None,
    timeout: int = 60,
) -> DownloadResult:
    """Download the raw operational nodes CSV into the raw static folder."""
    base_dir = Path(data_dir) if data_dir is not None else Path("data")
    destination = base_dir / "raw" / "static" / "operationele-punten-van-het-netwerk.csv"
    return _download_csv(
        url=OP_NODES_URL,
        dataset="op_nodes_raw",
        destination=destination,
        timeout=timeout,
    )


def download_line_sections(
    *,
    data_dir: Optional[Path] = None,
    timeout: int = 60,
) -> DownloadResult:
    """Download the raw line sections CSV into the raw static folder."""
    base_dir = Path(data_dir) if data_dir is not None else Path("data")
    destination = base_dir / "raw" / "static" / "lijnsecties.csv"
    return _download_csv(
        url=LINE_SECTIONS_URL,
        dataset="line_sections_raw",
        destination=destination,
        timeout=timeout,
    )


def download_events_month(
    month: str,
    *,
    data_dir: Optional[Path] = None,
    timeout: int = 60,
) -> DownloadResult:
    """Download one monthly raw punctuality events CSV."""
    month = _ensure_month(month)
    base_dir = Path(data_dir) if data_dir is not None else Path("data")
    destination = (
        base_dir
        / "raw"
        / "events"
        / f"Data_raw_punctuality_{month}.csv"
    )
    url = EVENTS_URL_TEMPLATE.format(month=month)
    return _download_csv(
        url=url,
        dataset="events_raw",
        destination=destination,
        timeout=timeout,
        extra={"month": month},
    )


def download_sources(
    config: Dict[str, Any],
    *,
    data_dir: Optional[Path] = None,
    timeout: int = 60,
) -> Tuple[List[Dict[str, object]], List[DownloadResult]]:
    """Download all raw sources requested by a Bronze download configuration."""
    sources: List[Dict[str, object]] = []
    event_downloads: List[DownloadResult] = []

    for downloader in (
        download_op_nodes,
        download_line_sections,
    ):
        result = downloader(data_dir=data_dir, timeout=timeout)
        print(f"Downloaded {result.dataset} to {result.path} ({result.size_bytes} bytes)")
        sources.append(result.as_dict())

    months = list(config.get("events", {}).get("months", []) or [])
    for month in months:
        result = download_events_month(month, data_dir=data_dir, timeout=timeout)
        print(f"Downloaded events {month} to {result.path} ({result.size_bytes} bytes)")
        sources.append(result.as_dict())
        event_downloads.append(result)

    return sources, event_downloads
