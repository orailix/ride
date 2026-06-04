"""Silver operational-node transform utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Dict, Iterable

import numpy as np
import pandas as pd

def build_op_nodes(
    *,
    sources: Mapping[str, Any],
    wildcards: Optional[Dict[str, str]] = None,
    params: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Build Silver operational nodes from raw nodes, line sections, and events."""
    params = dict(params or {})
    verbose = bool(params.pop("verbose", verbose))

    # Validate and copy the source tables used for node completion.
    op_nodes_source = sources["op_nodes"]
    line_sections_source = sources["line_sections"]

    if not isinstance(op_nodes_source, pd.DataFrame):
        raise TypeError("op_nodes source must be an eagerly loaded DataFrame.")
    if not isinstance(line_sections_source, pd.DataFrame):
        raise TypeError("line_sections source must be an eagerly loaded DataFrame.")

    op_nodes = op_nodes_source.copy()
    line_sections = line_sections_source.copy()
    if verbose:
        print(
            "[build_op_nodes] "
            f"Loaded op_nodes={len(op_nodes)}, line_sections={len(line_sections)}"
        )

    # Resolve optional event files used to find operational-node ids that are
    # referenced in events but absent from the raw node table.
    event_paths: list[Path] = []
    events_source = sources.get("events")
    if isinstance(events_source, (str, Path)):
        events_path = Path(events_source)
        if events_path.is_dir():
            event_paths = sorted(events_path.glob("*.parquet"))
        elif "*" in events_path.name:
            event_paths = sorted(events_path.parent.glob(events_path.name))
        else:
            event_paths = [events_path]
        if verbose:
            print(
                "[build_op_nodes] "
                f"Resolved {len(event_paths)} event parquet files from '{events_path}'"
            )
    elif events_source is not None:
        raise TypeError("events source must be a path or glob when provided.")
    elif verbose:
        print("[build_op_nodes] No events source provided")

    # Track ids already present and ids with complete coordinates separately:
    # line-section geometry can complete existing nodes with missing coordinates.
    present_ids_series = pd.to_numeric(op_nodes["op_id"], errors="coerce").dropna().astype(np.int64)
    present_ids = set(present_ids_series.tolist())
    complete_ids_series = pd.to_numeric(
        op_nodes.loc[op_nodes["lon"].notna() & op_nodes["lat"].notna(), "op_id"],
        errors="coerce",
    ).dropna().astype(np.int64)
    complete_ids = set(complete_ids_series.tolist())

    def _extract_endpoint(coords: list, choose_start: bool) -> tuple[float, float]:
        """Return the first or last coordinate from a possibly nested geometry."""
        node = coords[0 if choose_start else -1]
        while isinstance(node, (list, tuple)) and node and isinstance(node[0], (list, tuple)):
            node = node[0 if choose_start else -1]
        lon, lat = node[0], node[1]
        return float(lon), float(lat)

    inferred_rows: list[dict[str, object]] = []
    event_placeholder_rows: list[dict[str, object]] = []
    inferred_from_sections = 0
    placeholders_from_events = 0

    manual_overrides = params.pop("manual_overrides", None) or []
    raw_delete_ids = params.pop("manual_delete_op_ids", None) or []
    manual_delete_ids: list[int] = []
    for value in raw_delete_ids if isinstance(raw_delete_ids, (list, tuple, set)) else [raw_delete_ids]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            manual_delete_ids.append(int(value))
        except (TypeError, ValueError):
            print(f"[build_op_nodes] Skipped manual delete value '{value}' (invalid)")

    # Infer missing node coordinates from line-section endpoints when the raw
    # operational node table has an id but no usable geometry.
    for _, row in line_sections.iterrows():
        geoshape_raw = row.get("geoshape")
        if pd.isna(geoshape_raw):
            continue
        coords = json.loads(geoshape_raw)["coordinates"]
        endpoints = (
            (row.get("op_begin_id"), row.get("op_begin_name"), True),
            (row.get("op_end_id"), row.get("op_end_name"), False),
        )
        for op_key, op_name, is_begin in endpoints:
            if pd.isna(op_key):
                continue
            try:
                op_id = int(op_key)
            except (TypeError, ValueError):
                continue
            if op_id in complete_ids:
                continue

            lon, lat = _extract_endpoint(coords, is_begin)
            inferred_rows.append(
                {
                    "op_id": op_id,
                    "op_name": op_name,
                    "op_type": "Unknown",
                    "lat": lat,
                    "lon": lon,
                }
            )
            inferred_from_sections += 1
            present_ids.add(op_id)
            complete_ids.add(op_id)

    known_inferred_ids = {row["op_id"] for row in inferred_rows}

    # Add placeholder nodes for event references that are absent from the raw
    # node table and cannot be inferred from line sections.
    for events_path in event_paths:
        try:
            events = pd.read_parquet(events_path, columns=["op_id", "op_name"])
        except Exception:
            events = pd.read_parquet(events_path, columns=["op_id"])
            events["op_name"] = None
        if events.empty:
            continue

        events["op_id_int"] = pd.to_numeric(events["op_id"], errors="coerce").astype("Int64")
        valid_events = events.dropna(subset=["op_id_int"])
        if valid_events.empty:
            continue

        op_ids = valid_events["op_id_int"].dropna().astype(np.int64).unique()

        new_ids = [
            int(op_id)
            for op_id in op_ids
            if int(op_id) not in present_ids and int(op_id) not in known_inferred_ids
        ]
        if not new_ids:
            continue

        for op_id in new_ids:
            name_series = (
                valid_events.loc[valid_events["op_id_int"] == op_id, "op_name"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            op_name = next((val for val in name_series if val), None)
            event_placeholder_rows.append(
                {
                    "op_id": op_id,
                    "op_name": op_name,
                    "op_type": "Unknown",
                    "lat": np.nan,
                    "lon": np.nan,
                }
            )
            placeholders_from_events += 1
            present_ids.add(op_id)

        if verbose:
            print(
                "[build_op_nodes] "
                f"Added {len(new_ids)} placeholder nodes from {events_path.name}"
            )

    # Prefer raw names and types for nodes whose coordinates were inferred from
    # line-section geometry.
    for row in inferred_rows:
        op_id = row["op_id"]
        if op_id in present_ids:
            match = op_nodes.loc[op_nodes["op_id"] == op_id]
            if not match.empty:
                row["op_name"] = match.iloc[0]["op_name"]
                row["op_type"] = match.iloc[0]["op_type"]

    inferred_df = pd.DataFrame(inferred_rows).drop_duplicates(subset="op_id", keep="first")
    placeholder_df = pd.DataFrame(event_placeholder_rows).drop_duplicates(subset="op_id", keep="first")

    combined = (
        pd.concat([inferred_df, placeholder_df, op_nodes], ignore_index=True)
        .drop_duplicates(subset="op_id", keep="first")
        .reset_index(drop=True)
    )
    combined["op_id"] = pd.to_numeric(combined["op_id"], errors="coerce").astype("Int64")

    # Apply explicit manual edits after automated inference so they take
    # precedence over source and inferred coordinates.
    if manual_delete_ids:
        delete_mask = combined["op_id"].isin(manual_delete_ids)
        removed_ids = combined.loc[delete_mask, "op_id"].tolist()
        combined = combined.loc[~delete_mask].reset_index(drop=True)
        if removed_ids:
            print(f"[build_op_nodes] Removed op_ids: {removed_ids}")
        missing_delete = sorted(set(manual_delete_ids) - set(removed_ids))
        if missing_delete:
            print(
                "[build_op_nodes] Manual delete op_ids not present in dataset: "
                f"{missing_delete}"
            )

    for entry in manual_overrides:
        try:
            op_id = int(entry.get("op_id"))
            lat = float(entry.get("lat"))
            lon = float(entry.get("lon"))
        except (TypeError, ValueError):
            print(f"[build_op_nodes] Skipped manual override {entry} (invalid data)")
            continue

        mask = combined["op_id"] == op_id
        if mask.any():
            combined.loc[mask, "lat"] = lat
            combined.loc[mask, "lon"] = lon
            print(f"[build_op_nodes] Applied manual coordinates for op_id {op_id}")
        else:
            print(f"[build_op_nodes] Manual override ignored for op_id {op_id} (not found)")

    combined = combined.sort_values(by="op_id").reset_index(drop=True)

    # Make operation names unique while preserving their original order after
    # sorting by operation id.
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in combined["op_name"]:
        key = name if pd.notna(name) else None
        counts[key] = counts.get(key, 0) + 1
        unique_names.append(name if counts[key] == 1 else f"{name}_{counts[key]}")

    combined["op_name"] = unique_names
    combined["op_id"] = combined["op_id"].astype("Int64")

    missing_coords_mask = combined["lat"].isna() | combined["lon"].isna()
    if missing_coords_mask.any():
        missing_ids = combined.loc[missing_coords_mask, "op_id"].tolist()
        print(
            "[build_op_nodes] Remaining op_ids without coordinates: "
            f"{missing_ids}"
        )
    else:
        print("[build_op_nodes] All op_nodes have coordinates.")

    if verbose:
        print(
            "[build_op_nodes] "
            f"inferred_from_sections={inferred_from_sections}, "
            f"event_placeholders={placeholders_from_events}, "
            f"final_count={len(combined)}"
        )

    return combined
