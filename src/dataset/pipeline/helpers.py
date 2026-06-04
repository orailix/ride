"""Dataset pipeline helper utilities."""

from __future__ import annotations

import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from src.dataset.pipeline.manifest_runner import apply_manifest


def _strip_bom(text: str) -> str:
    """Remove a leading byte-order mark from a decoded string."""
    return text.lstrip("\ufeff") if text.startswith("\ufeff") else text


def _sanitize(obj: Any) -> Any:
    """Recursively remove byte-order marks from YAML keys and string values."""
    if isinstance(obj, dict):
        return {(_strip_bom(k) if isinstance(k, str) else k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    if isinstance(obj, str):
        return _strip_bom(obj)
    return obj


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file into a sanitized dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8")
    return _sanitize(yaml.safe_load(text) or {})


def load_manifest(path: Path) -> Dict[str, Any]:
    """Read a pipeline manifest YAML file into a sanitized dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8")
    return _sanitize(yaml.safe_load(text) or {})


def manifest_paths(root: Path) -> List[Path]:
    """Resolve a manifest file or directory into sorted YAML manifest paths."""
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.glob("*.yaml"))
    raise FileNotFoundError(f"No manifest files found under {root}")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dictionary as YAML, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def resolve_output_path(io_cfg: Dict[str, Any], wildcards: Optional[Dict[str, str]] = None) -> Path:
    """Resolve a manifest output path, applying wildcard substitution when provided."""
    output_cfg = io_cfg.get("output") or {}
    template = output_cfg.get("path")
    if not template:
        raise ValueError("Manifest io.output must include a path.")
    path = template
    if wildcards:
        token = wildcards.get("output") or wildcards.get("main")
        if token:
            path = path.replace("*", token)
    return Path(path)


def execute_manifest(
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
    metadata: Dict[str, Any],
    wildcards: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Run one manifest and append its output metadata entry."""
    df = apply_manifest(manifest, wildcards=wildcards)
    output_path = resolve_output_path(manifest["io"], wildcards)
    entry: Dict[str, Any] = {
        "dataset": manifest.get("id"),
        "manifest_path": str(manifest_path),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "path": str(output_path),
        "rows": len(df),
    }
    if wildcards:
        entry["wildcards"] = dict(wildcards)
    metadata.setdefault("datasets", []).append(entry)
    return df
