"""Silver line-section geometry transform utilities."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point


DEFAULT_EPSILON_M = 1.0
POSITION_EPS = 1e-6
MIN_BOUNDING_BOX_DEG = 1e-4


def _to_scalar(value: Any) -> Any:
    """Convert NumPy scalar values to their Python equivalents."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coord_key(lon: float, lat: float, precision: int = 9) -> tuple[float, float]:
    """Build a rounded coordinate key for exact node-to-vertex matching."""
    return (round(float(lon), precision), round(float(lat), precision))


def _load_linestring(geoshape: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse a GeoJSON-like line-section geometry string."""
    data = json.loads(geoshape)
    coords = np.asarray(data.get("coordinates", []), dtype="float64")
    return coords, data


def _compute_bbox_padding(coords: np.ndarray, eps_m: float) -> float:
    """Convert a meter tolerance into a conservative longitude/latitude padding."""
    if coords.size == 0:
        return MIN_BOUNDING_BOX_DEG
    avg_lat = float(np.mean(coords[:, 1]))
    cos_lat = max(abs(np.cos(np.radians(avg_lat))), 0.2)
    padding = eps_m / (111_320.0 * cos_lat)
    return max(padding, MIN_BOUNDING_BOX_DEG)


def _build_local_transformers(coords: np.ndarray) -> tuple[Transformer, Transformer]:
    """Build local projection transformers around one line-section geometry."""
    lon0 = float(np.mean(coords[:, 0]))
    lat0 = float(np.mean(coords[:, 1]))
    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)
    to_geo = Transformer.from_crs(crs_local, "EPSG:4326", always_xy=True)
    return to_local, to_geo


def _add_assignment(entries: List[Dict[str, Any]], used_ids: set[Any], node_id: Any, position: float, lon: float, lat: float) -> bool:
    """Insert or update one matched operational node along a line section."""
    node_scalar = _to_scalar(node_id)
    if node_scalar in used_ids:
        return False

    for entry in entries:
        if abs(entry["pos"] - position) <= POSITION_EPS and pd.isna(entry["op_id"]):
            entry["op_id"] = node_scalar
            used_ids.add(node_scalar)
            return True

    entries.append(
        {
            "pos": float(position),
            "lon": float(lon),
            "lat": float(lat),
            "op_id": node_scalar,
        }
    )
    entries.sort(key=lambda e: e["pos"])
    used_ids.add(node_scalar)
    return True


def _check_missing_op_refs(sections: pd.DataFrame, op_nodes: pd.DataFrame) -> None:
    """Print line-section references to operational nodes missing from the node table."""
    op_id_set = {
        int(val)
        for val in op_nodes.get("op_id", [])
        if pd.notna(val)
    }
    missing_refs: list[tuple[Any, int]] = []
    for _, row in sections.iterrows():
        for key in ("op_begin_id", "op_end_id"):
            try:
                val_int = int(row.get(key))
            except (TypeError, ValueError):
                continue
            if val_int not in op_id_set:
                missing_refs.append((row.get("line_section_id"), val_int))
        matched = row.get("matched_op_node_ids")
        if isinstance(matched, (list, tuple)):
            for val in matched:
                if pd.isna(val):
                    continue
                try:
                    val_int = int(val)
                except (TypeError, ValueError):
                    continue
                if val_int not in op_id_set:
                    missing_refs.append((row.get("line_section_id"), val_int))

    if missing_refs:
        print("[build_line_sections] Missing op_ids referenced by line_sections (likely due to manual deletions).")
        print("  Format: (line_section_id, missing_op_id)")
        for pair in missing_refs:
            print(f"  - {pair}")


def build_line_sections(
    *,
    sources: Mapping[str, Any],
    wildcards: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Attach matched operational nodes to Silver line-section geometries."""
    params = dict(params or {})
    sections_source = sources.get('main')
    if not isinstance(sections_source, pd.DataFrame):
        raise TypeError("Expected 'main' source to be an eagerly loaded DataFrame.")

    nodes_source = sources.get('op_nodes')
    if not isinstance(nodes_source, pd.DataFrame):
        raise TypeError("Expected 'op_nodes' source to be an eagerly loaded DataFrame.")

    sections = sections_source.copy()
    op_nodes = nodes_source.copy()

    eps_m = float(params.get('matching_epsilon_m', DEFAULT_EPSILON_M))

    # Apply manual section removals before matching so downstream geometry
    # enrichment only considers retained source sections.
    raw_delete_ids = params.pop("manual_delete_line_section_ids", None) or []
    manual_delete_ids: list[str] = []
    if not isinstance(raw_delete_ids, (list, tuple, set)):
        raw_delete_ids = [raw_delete_ids]
    for value in raw_delete_ids:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        manual_delete_ids.append(str(value))

    if manual_delete_ids:
        id_series = sections["line_section_id"].astype(str)
        delete_mask = id_series.isin(manual_delete_ids)
        removed = id_series.loc[delete_mask].tolist()
        sections = sections.loc[~delete_mask].reset_index(drop=True)
        if removed:
            print(f"[build_line_sections] Removed line_section_ids: {removed}")
        missing = sorted(set(manual_delete_ids) - set(removed))
        if missing:
            print(
                "[build_line_sections] Manual delete line_section_ids not present: "
                f"{missing}"
            )

    required_cols = {'op_id', 'lon', 'lat'}
    if not required_cols.issubset(op_nodes.columns):
        missing = required_cols - set(op_nodes.columns)
        raise KeyError(f"op_nodes source is missing columns: {sorted(missing)}")

    op_nodes = op_nodes.copy()
    op_nodes['op_id'] = op_nodes['op_id'].apply(_to_scalar)

    # Index operational nodes by their original coordinates for exact vertex
    # matches before using distance-based snapping.
    coord_index: dict[tuple[float, float], list[Any]] = {}
    for _, node_row in op_nodes.iterrows():
        key = _coord_key(node_row['lon'], node_row['lat'])
        coord_index.setdefault(key, []).append(node_row['op_id'])

    out_geoshapes: list[str] = []
    out_matches: list[Any] = []

    # For each line-section geometry, preserve existing vertices, assign known
    # endpoint nodes, and insert snapped intermediate operational nodes.
    for _, row in sections.iterrows():
        geoshape_value = row.get('geoshape')
        if not isinstance(geoshape_value, str) or not geoshape_value.strip():
            out_geoshapes.append(geoshape_value)
            out_matches.append(np.nan)
            continue

        coords, geo_dict = _load_linestring(geoshape_value)
        if coords.size == 0:
            out_geoshapes.append(geoshape_value)
            out_matches.append(np.nan)
            continue

        to_local, to_geo = _build_local_transformers(coords)
        xs, ys = to_local.transform(coords[:, 0], coords[:, 1])
        line_xy = np.column_stack([xs, ys])
        line_geom = LineString(line_xy)
        positions = np.array([line_geom.project(Point(pair)) for pair in line_xy], dtype='float64')

        entries: list[dict[str, Any]] = []
        used_ids: set[Any] = set()
        for pos, (lon, lat) in zip(positions, coords):
            entries.append({'pos': float(pos), 'lon': float(lon), 'lat': float(lat), 'op_id': np.nan})

        begin_id = row.get('op_begin_id')
        if begin_id is not None and not pd.isna(begin_id) and entries:
            begin_scalar = _to_scalar(begin_id)
            entries[0]['op_id'] = begin_scalar
            used_ids.add(begin_scalar)

        end_id = row.get('op_end_id')
        if end_id is not None and not pd.isna(end_id) and entries:
            end_scalar = _to_scalar(end_id)
            entries[-1]['op_id'] = end_scalar
            used_ids.add(end_scalar)

        for entry in entries[1:-1]:
            if pd.notna(entry['op_id']):
                continue
            key = _coord_key(entry['lon'], entry['lat'])
            for candidate in coord_index.get(key, []):
                if candidate not in used_ids:
                    entry['op_id'] = candidate
                    used_ids.add(candidate)
                    break

        length = float(line_geom.length)
        guard = 0.0
        if length > 0.0:
            guard_raw = min(2.0, 0.01 * length)
            guard = min(guard_raw, max(0.0, (length / 2.0) - 1e-6))

        lon_bounds = (float(coords[:, 0].min()), float(coords[:, 0].max()))
        lat_bounds = (float(coords[:, 1].min()), float(coords[:, 1].max()))
        delta = _compute_bbox_padding(coords, eps_m)

        subset_mask = (
            ~op_nodes['op_id'].isin(used_ids)
            & (op_nodes['lon'] >= lon_bounds[0] - delta)
            & (op_nodes['lon'] <= lon_bounds[1] + delta)
            & (op_nodes['lat'] >= lat_bounds[0] - delta)
            & (op_nodes['lat'] <= lat_bounds[1] + delta)
        )
        subset = op_nodes.loc[subset_mask]
        if not subset.empty:
            node_xs, node_ys = to_local.transform(
                subset['lon'].to_numpy(dtype='float64'),
                subset['lat'].to_numpy(dtype='float64'),
            )
            for (op_id, lon, lat), x, y in zip(
                subset[['op_id', 'lon', 'lat']].itertuples(index=False, name=None),
                node_xs,
                node_ys,
            ):
                if op_id in used_ids:
                    continue
                point = Point(x, y)
                proj = float(line_geom.project(point))
                if proj <= guard or (length - proj) <= guard:
                    continue
                snapped_point = line_geom.interpolate(proj)
                if point.distance(snapped_point) > eps_m:
                    continue
                snap_lon, snap_lat = to_geo.transform(snapped_point.x, snapped_point.y)
                _add_assignment(entries, used_ids, op_id, proj, snap_lon, snap_lat)

        coords_out = [[entry['lon'], entry['lat']] for entry in entries]
        matched = [entry['op_id'] if pd.notna(entry['op_id']) else np.nan for entry in entries]

        geo_copy = dict(geo_dict) if geo_dict is not None else {}
        geo_copy['coordinates'] = coords_out
        out_geoshapes.append(json.dumps(geo_copy))
        out_matches.append(matched)

    sections = sections.assign(
        geoshape=out_geoshapes,
        matched_op_node_ids=out_matches,
    )
    _check_missing_op_refs(sections, op_nodes)

    return sections
