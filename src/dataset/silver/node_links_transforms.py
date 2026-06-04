"""Silver node-link transform utilities."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from shapely.geometry import shape

EARTH_RADIUS_M = 6_371_000.0


def _extract_coordinates(geom) -> np.ndarray:
    """Extract coordinate pairs from a line geometry."""
    return np.asarray(geom.coords, dtype="float64")


def _haversine_distance(coords: np.ndarray) -> float:
    """Compute the cumulative great-circle distance along coordinate pairs."""
    if coords.shape[0] < 2:
        return 0.0
    lats = np.radians(coords[:, 1])
    lons = np.radians(coords[:, 0])
    dlat = np.diff(lats)
    dlon = np.diff(lons)
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(lats[:-1]) * np.cos(lats[1:]) * sin_dlon**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_M * c.sum())


def _iter_section_edges(row: pd.Series) -> list[dict[str, Any]]:
    """Yield node-link rows between consecutive matched nodes in one line section."""
    geoshape_val = row.get("geoshape")
    matched = row.get("matched_op_node_ids")

    geometry = shape(json.loads(geoshape_val))
    coords = _extract_coordinates(geometry)

    matched = list(matched)

    valid_indices = [idx for idx, value in enumerate(matched) if pd.notna(value)]
    if len(valid_indices) < 2:
        return []

    edges: list[dict[str, Any]] = []
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        u = matched[left]
        v = matched[right]
        if pd.isna(u) or pd.isna(v):
            continue
        try:
            u_id = int(u)
            v_id = int(v)
        except (TypeError, ValueError):
            continue

        segment_coords = coords[left : right + 1]
        distance = _haversine_distance(segment_coords)

        edges.append(
            {
                "line_section_id": row.get("line_section_id"),
                "u_node_id": u_id,
                "v_node_id": v_id,
                "distance_m": distance,
            }
        )

    return edges


def build_node_links(
    *,
    sources: Mapping[str, Any],
    wildcards: Optional[dict[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Build Silver node links from matched line-section geometries."""
    sections_source = sources.get("main")
    if not isinstance(sections_source, pd.DataFrame):
        raise TypeError("Expected 'main' source to be an eagerly loaded DataFrame.")

    sections = sections_source.copy()

    rows: list[dict[str, Any]] = []
    # Each enriched line section contributes one edge per consecutive pair of
    # matched operational nodes along its geometry.
    for _, section in sections.iterrows():
        rows.extend(_iter_section_edges(section))

    if not rows:
        return pd.DataFrame(columns=["u_node_id", "v_node_id", "line_section_id", "distance_m"])

    edges_df = pd.DataFrame(rows)
    edges_df["distance_m"] = edges_df["distance_m"].astype("float64")
    edges_df["u_node_id"] = edges_df["u_node_id"].astype("Int64")
    edges_df["v_node_id"] = edges_df["v_node_id"].astype("Int64")

    edges_df = edges_df.reset_index(drop=True)
    edges_df["link_id"] = (edges_df.index + 1).astype("Int64")  # deterministic identifier
    return edges_df
