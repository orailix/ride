"""Gold GNN dataset construction utilities."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData
from src.dataset.pipeline.helpers import read_yaml, write_yaml
from src.dataset.gold.helpers import (
    build_indexes,
    get_active_mask_for_snapshot,
    get_padded_arrays_from_components,
)
from src.dataset.gold.station_embeddings import create_station_embeddings_from_silver


def _to_ns_int64(values: pd.Series | pd.DatetimeIndex | np.ndarray | list) -> np.ndarray:
    """Convert datetime-like values to int64 nanoseconds."""
    dt = pd.to_datetime(values, errors="coerce")
    return dt.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def _next_non_empty_link_id(path: object, start_transition_idx: int) -> int | None:
    """Return the next non-empty raw link id from a transition index."""
    if path is None:
        return None
    for t in range(start_transition_idx, len(path)):
        sub = path[t]
        if len(sub) > 0:
            return int(sub[0])
    return None


def _orient_subpath_tokens(
    start_node_id: int,
    subpath: object,
    *,
    link_endpoints: dict[int, tuple[int, int]],
) -> list[str]:
    """Annotate each link id with inferred direction (uv/vu)."""
    cur = int(start_node_id)
    tokens: list[str] = []
    for link_id_raw in subpath:
        link_id = int(link_id_raw)
        u, v = link_endpoints[link_id]
        if u == cur:
            tokens.append(f"{link_id}uv")
            cur = v
        elif v == cur:
            tokens.append(f"{link_id}vu")
            cur = u
        else:
            raise ValueError(
                f"Subpath orientation mismatch: link_id={link_id}, start_node={start_node_id}, current_node={cur}, endpoints=({u},{v})"
            )
    return tokens


def _extract_current_link_token(
    *,
    train_key: tuple[str, str],
    event_idx: int,
    snap_ns: int,
    last_known_delay: float,
    path_by_key: dict[tuple[str, str], object],
    events_index: dict[str, object],
    link_distance_m: dict[int, float],
    link_endpoints: dict[int, tuple[int, int]],
) -> str:
    """Build the current oriented link token for one train at one snapshot."""
    path = path_by_key.get(train_key)
    if path is None:
        return ""

    key_slices = events_index["key_slices"]
    if not isinstance(key_slices, dict):
        raise TypeError("events_index['key_slices'] must be a dict.")
    start, end = key_slices[train_key]
    op_ids = events_index["event_op_ids"][start:end]
    planned_ns = events_index["event_planned_ns"][start:end]
    # Before first event: current link unknown.
    if event_idx < 0:
        return ""
    elif event_idx >= len(path):
        return ""
    else:
        sub = path[event_idx]
        if len(sub) == 0:
            op_id = int(op_ids[min(event_idx, len(op_ids) - 1)])
            nxt = _next_non_empty_link_id(path, event_idx + 1)
            nxt_text = str(nxt) if nxt is not None else "END"
            return f"STOPPED@op{op_id}to{nxt_text}"
        else:
            planned_delta_sec = float((int(planned_ns[event_idx + 1]) - int(planned_ns[event_idx])) / 1e9)
            pred_start_s = float(int(planned_ns[event_idx]) / 1e9) + float(last_known_delay)
            elapsed_sec = max(0.0, float(snap_ns / 1e9) - pred_start_s)
            sub_int = [int(x) for x in sub]
            oriented = _orient_subpath_tokens(
                int(op_ids[event_idx]),
                sub_int,
                link_endpoints=link_endpoints,
            )
            if len(sub_int) == 1:
                cur_sub_idx = 0
            else:
                distances = np.asarray([float(link_distance_m[x]) for x in sub_int], dtype=np.float64)
                total_d = float(distances.sum())
                if total_d <= 0:
                    cur_sub_idx = 0
                else:
                    per_link = planned_delta_sec * (distances / total_d)
                    cumulative = np.cumsum(per_link)
                    cur_sub_idx = int(np.searchsorted(cumulative, elapsed_sec, side="left"))
                    cur_sub_idx = max(0, min(cur_sub_idx, len(sub_int) - 1))
            return oriented[cur_sub_idx]


def build_weather_lookup(
    *,
    weather_path: Path,
    snapshots: pd.DatetimeIndex,
    event_op_ids: np.ndarray,
) -> dict[str, object]:
    """Build dense weather lookup arrays for requested snapshots and stations."""
    weather = pd.read_parquet(
        weather_path,
        columns=["op_id", "time", "temperature_2m", "rain", "snowfall", "relative_humidity_2m", "wind_speed_10m"],
    ).copy()
    weather["op_id"] = pd.to_numeric(weather["op_id"]).astype("Int64")
    weather["time"] = pd.to_datetime(weather["time"])
    snapshot_hours = snapshots.floor("h").unique()
    weather = weather.loc[weather["time"].isin(snapshot_hours)]
    weather = weather.loc[weather["op_id"].isin(event_op_ids)]
    weather = weather.sort_values(["time", "op_id"]).reset_index(drop=True)

    weather_times = pd.Index(weather["time"].unique(), name="time")
    weather_op_ids = pd.Index(
        np.asarray(np.unique(event_op_ids), dtype=np.int64),
        name="op_id",
    )
    time_pos = weather_times.get_indexer(weather["time"])
    op_pos = weather_op_ids.get_indexer(weather["op_id"].astype("int64"))
    weather_temperature = np.full((len(weather_times), len(weather_op_ids)), np.nan, dtype="float64")
    weather_rain = np.full_like(weather_temperature, np.nan)
    weather_snowfall = np.full_like(weather_temperature, np.nan)
    weather_rh = np.full_like(weather_temperature, np.nan)
    weather_wind = np.full_like(weather_temperature, np.nan)
    weather_temperature[time_pos, op_pos] = weather["temperature_2m"].to_numpy(dtype="float64", copy=False)
    weather_rain[time_pos, op_pos] = weather["rain"].to_numpy(dtype="float64", copy=False)
    weather_snowfall[time_pos, op_pos] = weather["snowfall"].to_numpy(dtype="float64", copy=False)
    weather_rh[time_pos, op_pos] = weather["relative_humidity_2m"].to_numpy(dtype="float64", copy=False)
    weather_wind[time_pos, op_pos] = weather["wind_speed_10m"].to_numpy(dtype="float64", copy=False)

    return {
        "weather_times": weather_times,
        "weather_op_ids": weather_op_ids,
        "weather_temperature": weather_temperature,
        "weather_rain": weather_rain,
        "weather_snowfall": weather_snowfall,
        "weather_rh": weather_rh,
        "weather_wind": weather_wind,
    }


def build_link_angle_lookup(
    *,
    node_links: pd.DataFrame,
    op_nodes_path: Path,
) -> dict[int, tuple[float, float, float, float]]:
    """Compute directional angle features for each infrastructure link."""
    op_nodes = pd.read_parquet(op_nodes_path, columns=["op_id", "lat", "lon"])
    op_nodes["op_id"] = pd.to_numeric(op_nodes["op_id"], errors="raise").astype(np.int64)
    op_nodes["lat"] = pd.to_numeric(op_nodes["lat"], errors="raise").astype(np.float64)
    op_nodes["lon"] = pd.to_numeric(op_nodes["lon"], errors="raise").astype(np.float64)
    pos_by_op_id = {
        int(r.op_id): (float(r.lat), float(r.lon))
        for r in op_nodes.itertuples(index=False)
    }
    link_angle_lookup: dict[int, tuple[float, float, float, float]] = {}
    for r in node_links.itertuples(index=False):
        link_id = int(r.link_id)
        u = int(r.u_node_id)
        v = int(r.v_node_id)
        if u not in pos_by_op_id:
            raise KeyError(f"Missing coordinates for u_node_id={u} (link_id={link_id})")
        if v not in pos_by_op_id:
            raise KeyError(f"Missing coordinates for v_node_id={v} (link_id={link_id})")
        lat_u, lon_u = pos_by_op_id[u]
        lat_v, lon_v = pos_by_op_id[v]
        uv_angle = float(np.arctan2(lat_v - lat_u, lon_v - lon_u))
        vu_angle = float(np.arctan2(lat_u - lat_v, lon_u - lon_v))
        uv_sin = float(np.sin(uv_angle))
        uv_cos = float(np.cos(uv_angle))
        vu_sin = float(np.sin(vu_angle))
        vu_cos = float(np.cos(vu_angle))
        link_angle_lookup[link_id] = (uv_sin, uv_cos, vu_sin, vu_cos)
    return link_angle_lookup


def build_train_features(
    *,
    index: dict,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    nb_past_events: int,
    nb_future_events: int,
    idle_time_beg: int,
    idle_time_end: int,
    missing_event_placeholder: int,
    show_progress: bool,
) -> tuple[pd.DataFrame, dict[tuple[int, str, str], float]]:
    """Build train-node features and last-known-delay lookup values."""
    events_index = index["events"]
    journeys_index = index["journeys"]

    # Keep relation/operator in the same raw style as tabular context features.
    relation_values = journeys_index["train_relation"]
    operator_values = journeys_index["operator"]
    key_slices = events_index["key_slices"]
    idle_time_beg_ns = idle_time_beg * 60 * 1_000_000_000
    idle_time_end_ns = idle_time_end * 60 * 1_000_000_000
    padded_cache: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    columns = [
        "snapshot_ts",
        "train_node_id",
        "train_id",
        "service_date",
        "train_relation",
        "operator",
    ]
    feature_rows: list[tuple[object, ...]] = []
    last_known_delay_lookup: dict[tuple[int, str, str], float] = {}

    # Snapshot expansion loop: one output row per active train key/train node.
    for snap_ts, snap_ns in tqdm(
        zip(snapshots, sampled_ns),
        total=len(snapshots),
        desc="Building train nodes",
        disable=not show_progress,
    ):
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snap_ns,
        )
        active_indices = active_mask.nonzero()[0]
        rows_snapshot: list[tuple[object, ...]] = []

        # Row construction for each active journey key at this snapshot.
        for train_node_id, idx in enumerate(active_indices):
            train_id = str(journeys_index["train_ids"][idx])
            service_date = str(journeys_index["service_dates"][idx])
            train_key = (train_id, service_date)
            cached = padded_cache.get(train_key)
            if cached is None:
                start, end = events_index["key_slices"][train_key]
                cached = get_padded_arrays_from_components(
                    ts=events_index["event_ts_ns"][start:end],
                    op_ids=events_index["event_op_ids"][start:end],
                    planned=events_index["event_planned_ns"][start:end],
                    delays=events_index["event_delay"][start:end],
                    types=events_index["event_type"][start:end],
                    nb_past_events=nb_past_events,
                    nb_future_events=nb_future_events,
                    idle_time_beg_ns=idle_time_beg_ns,
                    idle_time_end_ns=idle_time_end_ns,
                    missing_event_placeholder=missing_event_placeholder,
                )
                padded_cache[train_key] = cached
            padded_ts, padded_op_ids, padded_planned, padded_delay, _padded_type = cached

            start, end = key_slices[train_key]
            ts_slice = events_index["event_ts_ns"][start:end]
            if len(ts_slice) == 0:
                raise ValueError(
                    f"No events found for train_id={train_id}, service_date={service_date}"
                )
            split_idx = int(np.searchsorted(padded_ts, snap_ns, side="right"))
            past_slots = np.arange(split_idx - nb_past_events, split_idx, dtype=np.int64)[::-1]
            future_slots = np.arange(split_idx, split_idx + nb_future_events, dtype=np.int64)
            prev_op_id = int(padded_op_ids[past_slots[0]])
            next_op_id = int(padded_op_ids[future_slots[0]])
            prev_is_placeholder = prev_op_id == missing_event_placeholder
            next_is_placeholder = next_op_id == missing_event_placeholder
            if prev_is_placeholder and next_is_placeholder:
                raise ValueError(
                    f"Both stations are placeholders for train_id={train_id}, service_date={service_date}, ts={snap_ts}"
                )
            past_delay_1 = float(padded_delay[past_slots[0]])
            future_planned_1 = (padded_planned[future_slots[0]] - snap_ns) / 1e9
            last_known_delay = past_delay_1 - (((future_planned_1 + past_delay_1) < 0) * (future_planned_1 + past_delay_1))

            train_relation = str(relation_values[idx])
            operator = str(operator_values[idx])
            rows_snapshot.append(
                (
                    np.int64(snap_ns),
                    np.int32(train_node_id),
                    train_id,
                    service_date,
                    train_relation,
                    operator,
                )
            )
            lk_key = (int(snap_ns), train_id, service_date)
            if lk_key in last_known_delay_lookup:
                raise ValueError(
                    f"Duplicate last_known_delay key for train_id={train_id}, service_date={service_date}, ts={snap_ts}"
                )
            last_known_delay_lookup[lk_key] = float(last_known_delay)

        feature_rows.extend(rows_snapshot)

    train_features = pd.DataFrame.from_records(feature_rows, columns=columns)
    return train_features, last_known_delay_lookup


def build_station_features(
    *,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    station_feature_cache: dict[str, object],
    show_progress: bool,
) -> pd.DataFrame:
    """Build station-node features for each snapshot."""
    station_embedding_dim = int(station_feature_cache["station_embedding_dim"])
    lap_cols = [f"lap_emb_{i}" for i in range(station_embedding_dim)]
    columns = [
        "snapshot_ts",
        "station_id",
        "station_lat",
        "station_lon",
        "weather_temperature_2m",
        "weather_rain",
        "weather_snowfall",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ] + lap_cols
    rows: list[tuple[object, ...]] = []

    weather_times = station_feature_cache["weather_times"]
    weather_temperature = station_feature_cache["weather_temperature"]
    weather_rain = station_feature_cache["weather_rain"]
    weather_snowfall = station_feature_cache["weather_snowfall"]
    weather_rh = station_feature_cache["weather_rh"]
    weather_wind = station_feature_cache["weather_wind"]
    station_lat = station_feature_cache["station_lat"]
    station_lon = station_feature_cache["station_lon"]
    station_embeddings = station_feature_cache["station_embeddings"]
    n_stations = len(station_feature_cache["station_op_ids"])

    weather_time_index_cache: dict[pd.Timestamp, int] = {}

    for snap_ts, snap_ns in tqdm(
        zip(snapshots, sampled_ns),
        total=len(snapshots),
        desc="Building station nodes",
        disable=not show_progress,
    ):
        weather_time = snap_ts.floor("h")
        time_idx = weather_time_index_cache.get(weather_time)
        if time_idx is None:
            time_idx = int(weather_times.get_indexer([weather_time])[0])
            if time_idx < 0:
                raise KeyError(f"Missing weather time {weather_time}")
            weather_time_index_cache[weather_time] = time_idx

        for station_id in range(n_stations):
            row = [
                np.int64(snap_ns),
                np.int32(station_id),
                np.float32(station_lat[station_id]),
                np.float32(station_lon[station_id]),
                np.float32(weather_temperature[time_idx, station_id]),
                np.float32(weather_rain[time_idx, station_id]),
                np.float32(weather_snowfall[time_idx, station_id]),
                np.float32(weather_rh[time_idx, station_id]),
                np.float32(weather_wind[time_idx, station_id]),
            ]
            row.extend(np.asarray(station_embeddings[station_id], dtype=np.float32).tolist())
            rows.append(tuple(row))

    return pd.DataFrame.from_records(rows, columns=columns)


def build_train_to_past_and_future_station_edges(
    *,
    index: dict,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    nb_past_events: int,
    nb_future_events: int,
    station_op_ids: np.ndarray,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train-to-station edges for past observed and future scheduled events."""
    events_index = index["events"]
    journeys_index = index["journeys"]
    key_slices = events_index["key_slices"]
    event_ts_ns = events_index["event_ts_ns"]
    event_op_ids = events_index["event_op_ids"]
    event_planned_ns = events_index["event_planned_ns"]
    event_delay = events_index["event_delay"]
    event_type = events_index["event_type"]
    op_to_station = {int(op_id): int(i) for i, op_id in enumerate(station_op_ids)}

    past_columns = [
        "snapshot_ts",
        "train_node_id",
        "station_id",
        "past_planned_delta",
        "past_delay_sec",
        "past_event_type",
        "past_idx",
    ]
    future_columns = [
        "snapshot_ts",
        "train_node_id",
        "station_id",
        "future_planned_delta",
        "future_event_type",
        "future_idx",
        "future_delay_delta",
    ]
    past_rows: list[tuple[object, ...]] = []
    future_rows: list[tuple[object, ...]] = []

    for snap_ts, snap_ns in tqdm(
        zip(snapshots, sampled_ns),
        total=len(snapshots),
        desc="Building train->station edges",
        disable=not show_progress,
    ):
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snap_ns,
        )
        active_indices = active_mask.nonzero()[0]
        for train_node_id, idx in enumerate(active_indices):
            train_id = str(journeys_index["train_ids"][idx])
            service_date = str(journeys_index["service_dates"][idx])
            train_key = (train_id, service_date)
            start, end = key_slices[train_key]
            ts_slice = event_ts_ns[start:end]
            if len(ts_slice) == 0:
                raise ValueError(
                    f"No events found for train_id={train_id}, service_date={service_date}"
                )
            split_idx = int(np.searchsorted(ts_slice, snap_ns, side="right"))

            first_past = int(split_idx - 1)
            last_past = int(max(split_idx - nb_past_events, 0))
            for past_idx, local_event_idx in enumerate(range(first_past, last_past - 1, -1), start=1):
                global_event_idx = int(start + local_event_idx)
                op_id = int(event_op_ids[global_event_idx])
                if op_id not in op_to_station:
                    raise KeyError(
                        f"Missing station_id mapping for op_id={op_id} (train_id={train_id}, service_date={service_date}, ts={snap_ts})"
                    )
                past_rows.append(
                    (
                        np.int64(snap_ns),
                        np.int32(train_node_id),
                        np.int32(op_to_station[op_id]),
                        np.int64(event_planned_ns[global_event_idx]),
                        np.float32(event_delay[global_event_idx]),
                        str(event_type[global_event_idx]),
                        np.int16(past_idx),
                    )
                )

            last_future = int(min(split_idx + nb_future_events, len(ts_slice)))
            for future_idx, local_event_idx in enumerate(range(split_idx, last_future), start=1):
                global_event_idx = int(start + local_event_idx)
                op_id = int(event_op_ids[global_event_idx])
                if op_id not in op_to_station:
                    raise KeyError(
                        f"Missing station_id mapping for op_id={op_id} (train_id={train_id}, service_date={service_date}, ts={snap_ts})"
                    )
                future_rows.append(
                    (
                        np.int64(snap_ns),
                        np.int32(train_node_id),
                        np.int32(op_to_station[op_id]),
                        np.int64(event_planned_ns[global_event_idx]),
                        str(event_type[global_event_idx]),
                        np.int16(future_idx),
                        np.float32(event_delay[global_event_idx]),
                    )
                )

    train_to_past_station_features = pd.DataFrame.from_records(past_rows, columns=past_columns)
    train_to_future_station_features = pd.DataFrame.from_records(future_rows, columns=future_columns)
    return train_to_past_station_features, train_to_future_station_features


def apply_future_delay_delta(
    *,
    train_nodes: pd.DataFrame,
    train_to_future_station_edges: pd.DataFrame,
    last_known_delay_lookup: dict[tuple[int, str, str], float],
) -> pd.DataFrame:
    """Convert future-edge absolute delay targets to deltas from last known delay."""
    key_cols = ["snapshot_ts", "train_node_id", "train_id", "service_date"]
    missing_train = [c for c in key_cols if c not in train_nodes.columns]
    if missing_train:
        raise KeyError(f"Missing train node columns for delta mapping: {missing_train}")
    missing_future = [c for c in ["snapshot_ts", "train_node_id", "future_delay_delta"] if c not in train_to_future_station_edges.columns]
    if missing_future:
        raise KeyError(f"Missing future edge columns for delta mapping: {missing_future}")

    train_keys = train_nodes.loc[:, key_cols].copy()
    train_keys["snapshot_ts"] = pd.to_numeric(train_keys["snapshot_ts"], errors="raise").astype(np.int64)
    train_keys["train_node_id"] = pd.to_numeric(train_keys["train_node_id"], errors="raise").astype(np.int32)
    merged = train_to_future_station_edges.merge(
        train_keys,
        on=["snapshot_ts", "train_node_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["train_id"].isna().any() or merged["service_date"].isna().any():
        raise ValueError("Missing train_id/service_date after future-edge -> train-node merge.")

    lk_vals: list[float] = []
    snap_vals = pd.to_numeric(merged["snapshot_ts"], errors="raise").to_numpy(dtype=np.int64, copy=False)
    train_ids = merged["train_id"].astype(str).to_numpy(dtype=object, copy=False)
    service_dates = merged["service_date"].astype(str).to_numpy(dtype=object, copy=False)
    for i in range(len(merged)):
        lk_key = (int(snap_vals[i]), str(train_ids[i]), str(service_dates[i]))
        if lk_key not in last_known_delay_lookup:
            raise KeyError(
                f"Missing last_known_delay for train_id={train_ids[i]}, service_date={service_dates[i]}, snapshot_ts={snap_vals[i]}"
            )
        lk_vals.append(float(last_known_delay_lookup[lk_key]))
    lk_arr = np.asarray(lk_vals, dtype=np.float64)
    future_delay = pd.to_numeric(merged["future_delay_delta"], errors="raise").to_numpy(dtype=np.float64, copy=False)
    merged["future_delay_delta"] = (future_delay - lk_arr).astype(np.float32)
    return merged.drop(columns=["train_id", "service_date"])


def build_station_to_station_and_stop_features(
    *,
    index: dict,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    link_distance_m: dict[int, float],
    link_angle_lookup: dict[int, tuple[float, float, float, float]],
    link_endpoints: dict[int, tuple[int, int]],
    station_op_ids: np.ndarray,
    last_known_delay_lookup: dict[tuple[int, str, str], float],
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build station-link traffic features and stopped-train station features."""
    events_index = index["events"]
    journeys_index = index["journeys"]
    key_slices = events_index["key_slices"]
    event_ts_ns = events_index["event_ts_ns"]
    path_by_key: dict[tuple[str, str], object] = {
        (str(journeys_index["train_ids"][i]), str(journeys_index["service_dates"][i])): journeys_index["deduced_path"][i]
        for i in range(len(journeys_index["train_ids"]))
    }

    op_to_station = {int(op_id): int(i) for i, op_id in enumerate(station_op_ids)}
    node_links: list[tuple[int, int, int, float, float, float, float, float]] = []
    directed_tokens: set[str] = set()
    for link_id in sorted(link_endpoints):
        u, v = link_endpoints[int(link_id)]
        if int(u) not in op_to_station or int(v) not in op_to_station:
            raise KeyError(
                f"Missing station_id mapping for link_id={int(link_id)} with endpoints ({int(u)}, {int(v)})"
            )
        dist = float(link_distance_m[int(link_id)])
        if int(link_id) not in link_angle_lookup:
            raise KeyError(f"Missing angle lookup for link_id={int(link_id)}")
        uv_sin, uv_cos, vu_sin, vu_cos = link_angle_lookup[int(link_id)]
        link_id_int = int(link_id)
        u_station = op_to_station[int(u)]
        v_station = op_to_station[int(v)]
        tok_uv = f"{link_id_int}uv"
        tok_vu = f"{link_id_int}vu"
        node_links.append(
            (
                link_id_int,
                u_station,
                v_station,
                dist,
                float(uv_sin),
                float(uv_cos),
                float(vu_sin),
                float(vu_cos),
            )
        )
        directed_tokens.add(tok_uv)
        directed_tokens.add(tok_vu)

    edge_columns = [
        "snapshot_ts",
        "link_id",
        "u_station_id",
        "v_station_id",
        "link_distance",
        "uv_angle_sin",
        "uv_angle_cos",
        "vu_angle_sin",
        "vu_angle_cos",
        "uv_nb_of_trains",
        "uv_avg_delay",
        "vu_nb_of_trains",
        "vu_avg_delay",
    ]
    station_stop_columns = [
        "snapshot_ts",
        "station_id",
        "stopped_nb_of_trains",
        "stopped_avg_delay",
    ]
    edge_rows: list[tuple[object, ...]] = []
    station_stop_rows: list[tuple[object, ...]] = []

    for snap_ts, snap_ns in tqdm(
        zip(snapshots, sampled_ns),
        total=len(snapshots),
        desc="Building station-link features",
        disable=not show_progress,
    ):
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snap_ns,
        )
        active_indices = active_mask.nonzero()[0]
        link_counts: dict[str, int] = {}
        link_delay_sums: dict[str, float] = {}
        stop_counts: dict[int, int] = {}
        stop_delay_sums: dict[int, float] = {}

        for idx in active_indices:
            train_id = str(journeys_index["train_ids"][idx])
            service_date = str(journeys_index["service_dates"][idx])
            train_key = (train_id, service_date)
            lk_key = (int(snap_ns), train_id, service_date)
            if lk_key not in last_known_delay_lookup:
                raise KeyError(
                    f"Missing last_known_delay for train_id={train_id}, service_date={service_date}, ts={snap_ts}"
                )
            last_known_delay = float(last_known_delay_lookup[lk_key])
            start, end = key_slices[train_key]
            ts_slice = event_ts_ns[start:end]
            if len(ts_slice) == 0:
                raise ValueError(
                    f"No events found for train_id={train_id}, service_date={service_date}"
                )
            current_event_idx = int(np.searchsorted(ts_slice, snap_ns, side="right")) - 1
            tok0 = _extract_current_link_token(
                train_key=train_key,
                event_idx=current_event_idx,
                snap_ns=int(snap_ns),
                last_known_delay=last_known_delay,
                path_by_key=path_by_key,
                events_index=events_index,
                link_distance_m=link_distance_m,
                link_endpoints=link_endpoints,
            )
            if tok0 == "":
                continue
            if tok0.startswith("STOPPED@"):
                m = re.match(r"^STOPPED@op(\d+)to.*$", tok0)
                if m is None:
                    raise ValueError(f"Unexpected STOPPED token format: {tok0!r}")
                op_id = int(m.group(1))
                if op_id not in op_to_station:
                    raise KeyError(f"Missing station_id mapping for stopped op_id={op_id}")
                stop_counts[op_id] = int(stop_counts.get(op_id, 0)) + 1
                stop_delay_sums[op_id] = float(stop_delay_sums.get(op_id, 0.0)) + float(last_known_delay)
                continue
            if tok0 not in directed_tokens:
                raise ValueError(f"Unexpected directed link token: {tok0!r}")
            link_counts[tok0] = int(link_counts.get(tok0, 0)) + 1
            link_delay_sums[tok0] = float(link_delay_sums.get(tok0, 0.0)) + float(last_known_delay)

        for link_id, u_station_id, v_station_id, dist, uv_sin, uv_cos, vu_sin, vu_cos in node_links:
            tok_uv = f"{int(link_id)}uv"
            tok_vu = f"{int(link_id)}vu"
            uv_nb = int(link_counts.get(tok_uv, 0))
            vu_nb = int(link_counts.get(tok_vu, 0))
            uv_avg = np.nan if uv_nb == 0 else float(link_delay_sums[tok_uv] / uv_nb)
            vu_avg = np.nan if vu_nb == 0 else float(link_delay_sums[tok_vu] / vu_nb)
            edge_rows.append(
                (
                    np.int64(snap_ns),
                    np.int32(link_id),
                    np.int32(u_station_id),
                    np.int32(v_station_id),
                    np.float32(dist),
                    np.float32(uv_sin),
                    np.float32(uv_cos),
                    np.float32(vu_sin),
                    np.float32(vu_cos),
                    np.int16(uv_nb),
                    np.float32(uv_avg),
                    np.int16(vu_nb),
                    np.float32(vu_avg),
                )
            )

        for station_id, op_id in enumerate(station_op_ids):
            nb = int(stop_counts.get(int(op_id), 0))
            avg = np.nan if nb == 0 else float(stop_delay_sums[int(op_id)] / nb)
            station_stop_rows.append(
                (
                    np.int64(snap_ns),
                    np.int32(station_id),
                    np.int16(nb),
                    np.float32(avg),
                )
            )

    station_to_station_features = pd.DataFrame.from_records(edge_rows, columns=edge_columns)
    station_stop_features = pd.DataFrame.from_records(station_stop_rows, columns=station_stop_columns)
    return station_to_station_features, station_stop_features


def _iter_snapshot_chunks(
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    *,
    chunk_size: int,
):
    """Yield snapshot and timestamp chunks for streaming graph construction."""
    n = len(snapshots)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield snapshots[start:end], sampled_ns[start:end]


def merge_station_features(
    *,
    station_features: pd.DataFrame,
    station_stop_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach stopped-train aggregates to station-node features."""
    merged = station_features.merge(
        station_stop_features,
        on=["snapshot_ts", "station_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["stopped_nb_of_trains"].isna().any():
        raise ValueError("Unexpected missing stopped station features after merge.")
    merged["stopped_nb_of_trains"] = merged["stopped_nb_of_trains"].astype(np.int16)
    merged["stopped_avg_delay"] = pd.to_numeric(
        merged["stopped_avg_delay"], errors="coerce"
    ).astype(np.float32)
    return merged


def build_nodes_and_edges_features(
    *,
    index: dict,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    nb_past_events: int,
    nb_future_events: int,
    idle_time_beg: int,
    idle_time_end: int,
    missing_event_placeholder: int,
    station_feature_cache: dict[str, object],
    link_distance_m: dict[int, float],
    link_angle_lookup: dict[int, tuple[float, float, float, float]],
    link_endpoints: dict[int, tuple[int, int]],
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build raw node and edge feature tables for a set of snapshots."""
    train_features, last_known_delay_lookup = build_train_features(
        index=index,
        snapshots=snapshots,
        sampled_ns=sampled_ns,
        nb_past_events=nb_past_events,
        nb_future_events=nb_future_events,
        idle_time_beg=idle_time_beg,
        idle_time_end=idle_time_end,
        missing_event_placeholder=missing_event_placeholder,
        show_progress=show_progress,
    )

    station_features = build_station_features(
        snapshots=snapshots,
        sampled_ns=sampled_ns,
        station_feature_cache=station_feature_cache,
        show_progress=show_progress,
    )

    station_to_station_features, station_stop_features = build_station_to_station_and_stop_features(
        index=index,
        snapshots=snapshots,
        sampled_ns=sampled_ns,
        link_distance_m=link_distance_m,
        link_angle_lookup=link_angle_lookup,
        link_endpoints=link_endpoints,
        station_op_ids=station_feature_cache["station_op_ids"],
        last_known_delay_lookup=last_known_delay_lookup,
        show_progress=show_progress,
    )
    station_features = merge_station_features(
        station_features=station_features,
        station_stop_features=station_stop_features,
    )
    train_to_past_station_features, train_to_future_station_features = build_train_to_past_and_future_station_edges(
        index=index,
        snapshots=snapshots,
        sampled_ns=sampled_ns,
        nb_past_events=nb_past_events,
        nb_future_events=nb_future_events,
        station_op_ids=station_feature_cache["station_op_ids"],
        show_progress=show_progress,
    )
    train_to_future_station_features = apply_future_delay_delta(
        train_nodes=train_features,
        train_to_future_station_edges=train_to_future_station_features,
        last_known_delay_lookup=last_known_delay_lookup,
    )
    return (
        train_features,
        station_features,
        station_to_station_features,
        train_to_past_station_features,
        train_to_future_station_features,
    )


def build_ml_train_nodes(*, train_nodes: pd.DataFrame) -> pd.DataFrame:
    """Convert raw train-node features into pre-normalized ML columns."""
    relation_categories = ["EURST", "EXTRA", "IC", "ICE", "INT", "L", "P", "TGV", "THAL", "S", "CHARTER", "nan"]
    rel = train_nodes["train_relation"].astype("string").fillna("nan").str.replace(r"^(S).*", r"\1", regex=True)
    train_nodes["train_relation_type"] = rel.str.split(" ").str[0]
    train_nodes["train_relation_type"] = pd.Categorical(
        train_nodes["train_relation_type"],
        categories=relation_categories,
    )
    train_nodes["operator_is_sncb_nmbs"] = (
        train_nodes["operator"].astype("string") == "SNCB/NMBS"
    ).astype(int)

    snapshot_dt = pd.to_datetime(train_nodes["snapshot_ts"])
    train_nodes["snapshot_day_of_year"] = snapshot_dt.dt.dayofyear
    train_nodes["snapshot_day_of_week"] = pd.Categorical(
        snapshot_dt.dt.day_name(),
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    train_nodes["snapshot_time_of_day"] = (
        snapshot_dt.dt.hour + snapshot_dt.dt.minute / 60.0 + snapshot_dt.dt.second / 3600.0
    )
    frequencies = [1, 2, 4]
    for freq in frequencies:
        train_nodes[f"snapshot_hour_sin_{freq}"] = np.sin(
            freq * 2 * np.pi * train_nodes["snapshot_time_of_day"] / 24.0
        )
        train_nodes[f"snapshot_hour_cos_{freq}"] = np.cos(
            freq * 2 * np.pi * train_nodes["snapshot_time_of_day"] / 24.0
        )
    for freq in frequencies:
        train_nodes[f"snapshot_year_sin_{freq}"] = np.sin(
            freq * 2 * np.pi * train_nodes["snapshot_day_of_year"] / 365.0
        )
        train_nodes[f"snapshot_year_cos_{freq}"] = np.cos(
            freq * 2 * np.pi * train_nodes["snapshot_day_of_year"] / 365.0
        )

    cat_cols = ["train_relation_type", "snapshot_day_of_week"]
    cat_encoded = pd.get_dummies(train_nodes[cat_cols], prefix=cat_cols, dummy_na=False, dtype=int)
    train_nodes = pd.concat([train_nodes, cat_encoded], axis=1)
    return train_nodes


def build_ml_station_nodes(*, station_nodes: pd.DataFrame) -> pd.DataFrame:
    """Convert raw station-node features into pre-normalized ML columns."""
    for col in ["stopped_avg_delay", "weather_rain", "weather_snowfall"]:
        station_nodes[col] = pd.to_numeric(station_nodes[col], errors="coerce")
    station_nodes["stopped_avg_delay"] = station_nodes["stopped_avg_delay"].fillna(0.0)
    for col in ["stopped_avg_delay", "weather_rain", "weather_snowfall"]:
        values = station_nodes[col].to_numpy(dtype=np.float64, copy=False)
        station_nodes[col] = np.sign(values) * np.sqrt(np.abs(values))
    return station_nodes


def build_ml_station_to_station_edges(*, station_to_station_edges: pd.DataFrame) -> pd.DataFrame:
    """Convert raw station-to-station edge features into pre-normalized ML columns."""
    for col in ["uv_avg_delay", "vu_avg_delay"]:
        station_to_station_edges[col] = pd.to_numeric(
            station_to_station_edges[col], errors="coerce"
        ).fillna(0.0)
        values = pd.to_numeric(station_to_station_edges[col], errors="coerce").to_numpy(dtype=np.float64)
        station_to_station_edges[col] = np.sign(values) * np.sqrt(np.abs(values))

    for col in [
        "link_distance",
        "uv_nb_of_trains",
        "vu_nb_of_trains",
        "uv_angle_sin",
        "uv_angle_cos",
        "vu_angle_sin",
        "vu_angle_cos",
    ]:
        station_to_station_edges[col] = pd.to_numeric(station_to_station_edges[col], errors="coerce")
    return station_to_station_edges


def build_ml_train_to_station_edges(
    *,
    train_to_past_station_edges: pd.DataFrame,
    train_to_future_station_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw train-to-station edge features into pre-normalized ML columns."""
    past_snapshot_dt = pd.to_datetime(train_to_past_station_edges["snapshot_ts"])
    past_planned_dt = pd.to_datetime(
        pd.to_numeric(train_to_past_station_edges["past_planned_delta"], errors="coerce"),
        unit="ns",
    )
    train_to_past_station_edges["past_planned_delta"] = (past_planned_dt - past_snapshot_dt).dt.total_seconds()
    train_to_past_station_edges["past_delay_sec"] = pd.to_numeric(
        train_to_past_station_edges["past_delay_sec"], errors="coerce"
    )
    past_planned_vals = pd.to_numeric(
        train_to_past_station_edges["past_planned_delta"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    past_delay_vals = pd.to_numeric(
        train_to_past_station_edges["past_delay_sec"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    train_to_past_station_edges["past_planned_delta"] = np.sign(past_planned_vals) * np.sqrt(np.abs(past_planned_vals))
    train_to_past_station_edges["past_delay_sec"] = np.sign(past_delay_vals) * np.sqrt(np.abs(past_delay_vals))
    past_event_type = train_to_past_station_edges["past_event_type"].astype("string")
    train_to_past_station_edges["past_event_type_A"] = (past_event_type == "A").astype(int)
    train_to_past_station_edges["past_event_type_D"] = (past_event_type == "D").astype(int)
    train_to_past_station_edges["past_event_type_P"] = (past_event_type == "P").astype(int)
    train_to_past_station_edges["past_idx"] = pd.to_numeric(
        train_to_past_station_edges["past_idx"], errors="coerce"
    )

    future_snapshot_dt = pd.to_datetime(train_to_future_station_edges["snapshot_ts"])
    future_planned_dt = pd.to_datetime(
        pd.to_numeric(train_to_future_station_edges["future_planned_delta"], errors="coerce"),
        unit="ns",
    )
    train_to_future_station_edges["future_planned_delta"] = (future_planned_dt - future_snapshot_dt).dt.total_seconds()
    train_to_future_station_edges["future_delay_delta"] = pd.to_numeric(
        train_to_future_station_edges["future_delay_delta"], errors="coerce"
    )
    future_planned_vals = pd.to_numeric(
        train_to_future_station_edges["future_planned_delta"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    future_delay_vals = pd.to_numeric(
        train_to_future_station_edges["future_delay_delta"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    train_to_future_station_edges["future_planned_delta"] = np.sign(future_planned_vals) * np.sqrt(np.abs(future_planned_vals))
    train_to_future_station_edges["future_delay_delta"] = np.sign(future_delay_vals) * np.sqrt(np.abs(future_delay_vals))
    future_event_type = train_to_future_station_edges["future_event_type"].astype("string")
    train_to_future_station_edges["future_event_type_A"] = (future_event_type == "A").astype(int)
    train_to_future_station_edges["future_event_type_D"] = (future_event_type == "D").astype(int)
    train_to_future_station_edges["future_event_type_P"] = (future_event_type == "P").astype(int)
    train_to_future_station_edges["future_idx"] = pd.to_numeric(
        train_to_future_station_edges["future_idx"], errors="coerce"
    )
    return train_to_past_station_edges, train_to_future_station_edges


def _update_running_stats(
    *,
    running: dict[str, dict[str, float | int | bool]],
    features: pd.DataFrame,
    train_mask: np.ndarray,
    signed_sqrt_cols: list[str],
    zscore_cols: list[str],
    minmax_cols: list[str],
) -> None:
    """Update running train-split normalization statistics from one feature table."""
    # Z-score accumulators (sum/sum_sq) for transformed numeric columns.
    for col in signed_sqrt_cols + zscore_cols:
        vals = pd.to_numeric(features.loc[train_mask, col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        v = vals[finite]
        acc = running.setdefault(
            col,
            {"count": 0, "sum": 0.0, "sum_sq": 0.0, "sqrt": bool(col in signed_sqrt_cols)},
        )
        acc["count"] = int(acc["count"]) + int(v.size)
        acc["sum"] = float(acc["sum"]) + float(v.sum())
        acc["sum_sq"] = float(acc["sum_sq"]) + float(np.square(v).sum())

    # Min-max accumulators for bounded local-link columns.
    for col in minmax_cols:
        vals = pd.to_numeric(features.loc[train_mask, col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        v = vals[finite]
        acc = running.setdefault(
            col,
            {"norm": "minmax", "min": float("inf"), "max": float("-inf")},
        )
        acc["min"] = min(float(acc["min"]), float(np.min(v)))
        acc["max"] = max(float(acc["max"]), float(np.max(v)))


def _finalize_running_stats(
    running: dict[str, dict[str, float | int | bool]]
) -> dict[str, dict[str, float | bool]]:
    """Convert running normalization accumulators into final statistics."""
    out: dict[str, dict[str, float | bool]] = {}
    for col, acc in running.items():
        if str(acc.get("norm", "")) == "minmax":
            out[col] = {"norm": "minmax", "min": float(acc["min"]), "max": float(acc["max"])}
            continue
        count = int(acc["count"])
        s = float(acc["sum"])
        ss = float(acc["sum_sq"])
        if count <= 0:
            mean = np.nan
            std = np.nan
        elif count == 1:
            mean = s
            std = 0.0
        else:
            mean = s / count
            var = (ss - (s * s) / count) / (count - 1)
            std = float(np.sqrt(max(var, 0.0)))
        out[col] = {"norm": "zscore", "mean": float(mean), "std": float(std), "sqrt": bool(acc["sqrt"])}
    return out


def _apply_normalization_stats(
    *,
    features: pd.DataFrame,
    normalization_stats: dict[str, dict[str, float | bool]],
) -> pd.DataFrame:
    """Apply precomputed normalization statistics to numeric feature columns."""
    for col, stats in normalization_stats.items():
        norm = str(stats.get("norm", "zscore"))
        values = pd.to_numeric(features[col], errors="coerce")

        if norm == "minmax":
            vmin = float(stats["min"])
            vmax = float(stats["max"])
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
                features[col] = np.zeros(len(values), dtype=np.float64)
            else:
                scaled = (values - vmin) / (vmax - vmin)
                features[col] = np.clip(scaled, 0.0, 1.0)
            continue

        std = float(stats["std"])
        if std == 0.0 or np.isnan(std):
            features[col] = values - float(stats["mean"])
        else:
            features[col] = (values - float(stats["mean"])) / std
    return features


def split_future_edge_targets(
    *,
    train_to_future_station_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split future-edge input features from future-delay targets."""
    y_key_cols = ["snapshot_ts", "train_node_id", "station_id", "future_idx"]
    y_cols = y_key_cols + ["future_delay_delta"]
    missing = [c for c in y_cols if c not in train_to_future_station_edges.columns]
    if missing:
        raise KeyError(f"Missing future-edge target columns: {missing}")
    y_edges = train_to_future_station_edges.loc[:, y_cols].copy()
    x_edges = train_to_future_station_edges.drop(columns=["future_delay_delta"]).copy()
    return x_edges, y_edges


def restore_future_idx_for_target_table(
    *,
    target_table: pd.DataFrame,
    future_idx_stats: dict[str, float | bool],
) -> pd.DataFrame:
    """Undo future-index normalization for target table metadata."""
    values = pd.to_numeric(target_table["future_idx"], errors="raise").to_numpy(dtype=np.float64, copy=False)
    norm = str(future_idx_stats.get("norm", "zscore"))
    if norm == "minmax":
        vmin = float(future_idx_stats["min"])
        vmax = float(future_idx_stats["max"])
        restored = values * (vmax - vmin) + vmin
    else:
        std = float(future_idx_stats["std"])
        mean = float(future_idx_stats["mean"])
        if std == 0.0 or np.isnan(std):
            restored = values + mean
        else:
            restored = values * std + mean
    target_table = target_table.copy()
    target_table["future_idx"] = np.rint(restored).astype(np.int16)
    return target_table


def _split_table_by_snapshot_ts(
    *,
    table_name: str,
    features: pd.DataFrame,
    train_snapshot_ns: np.ndarray,
    test_snapshot_ns: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one feature table by train and test snapshot timestamps."""
    if "snapshot_ts" not in features.columns:
        raise KeyError(f"Missing snapshot_ts column in table '{table_name}'")
    snapshot_ns = pd.to_numeric(features["snapshot_ts"], errors="raise").to_numpy(dtype=np.int64, copy=False)
    train_mask = np.isin(snapshot_ns, train_snapshot_ns)
    test_mask = np.isin(snapshot_ns, test_snapshot_ns)
    if np.any(train_mask & test_mask):
        raise ValueError(f"Overlapping train/test snapshots detected in table '{table_name}'")
    covered = train_mask | test_mask
    if not np.all(covered):
        raise ValueError(f"Found rows outside train/test snapshots in table '{table_name}'")
    train_df = features.loc[train_mask].reset_index(drop=True)
    test_df = features.loc[test_mask].reset_index(drop=True)
    return train_df, test_df


def to_ml_features(
    *,
    train_nodes: pd.DataFrame,
    station_nodes: pd.DataFrame,
    station_to_station_edges: pd.DataFrame,
    train_to_past_station_edges: pd.DataFrame,
    train_to_future_station_edges: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, list[str]]],
]:
    """Convert raw graph feature tables into pre-normalized ML feature tables."""
    train_nodes = build_ml_train_nodes(train_nodes=train_nodes)
    station_nodes = build_ml_station_nodes(station_nodes=station_nodes)
    station_to_station_edges = build_ml_station_to_station_edges(
        station_to_station_edges=station_to_station_edges
    )
    train_to_past_station_edges, train_to_future_station_edges = build_ml_train_to_station_edges(
        train_to_past_station_edges=train_to_past_station_edges,
        train_to_future_station_edges=train_to_future_station_edges,
    )
    ml_groups = {
        "train_nodes": {"signed_sqrt_cols": [], "zscore_cols": [], "minmax_cols": []},
        "station_nodes": {
            "signed_sqrt_cols": ["stopped_avg_delay", "weather_rain", "weather_snowfall"],
            "zscore_cols": [
                "weather_temperature_2m",
                "weather_relative_humidity_2m",
                "weather_wind_speed_10m",
            ],
            "minmax_cols": ["stopped_nb_of_trains", "station_lat", "station_lon"],
        },
        "station_to_station_edges": {
            "signed_sqrt_cols": ["uv_avg_delay", "vu_avg_delay"],
            "zscore_cols": [],
            "minmax_cols": ["link_distance", "uv_nb_of_trains", "vu_nb_of_trains"],
        },
        "train_to_past_station_edges": {
            "signed_sqrt_cols": ["past_planned_delta", "past_delay_sec"],
            "zscore_cols": [],
            "minmax_cols": ["past_idx"],
        },
        "train_to_future_station_edges": {
            "signed_sqrt_cols": ["future_planned_delta", "future_delay_delta"],
            "zscore_cols": [],
            "minmax_cols": ["future_idx"],
        },
    }
    return (
        train_nodes,
        station_nodes,
        station_to_station_edges,
        train_to_past_station_edges,
        train_to_future_station_edges,
        ml_groups,
    )


def _require_columns(*, features: pd.DataFrame, expected: list[str], table_name: str) -> None:
    """Raise if an ordered export table is missing required columns."""
    missing = [c for c in expected if c not in features.columns]
    if missing:
        raise KeyError(f"Missing ordered ML columns in '{table_name}': {missing}")


def build_ordered_train_nodes(*, train_nodes: pd.DataFrame) -> pd.DataFrame:
    """Order train-node columns according to the exported GNN schema."""
    ordered = [
        "snapshot_ts",
        "train_node_id",
        "train_id",
        "service_date",
        "operator_is_sncb_nmbs",
        "snapshot_hour_sin_1",
        "snapshot_hour_sin_2",
        "snapshot_hour_sin_4",
        "snapshot_hour_cos_1",
        "snapshot_hour_cos_2",
        "snapshot_hour_cos_4",
        "snapshot_year_sin_1",
        "snapshot_year_sin_2",
        "snapshot_year_sin_4",
        "snapshot_year_cos_1",
        "snapshot_year_cos_2",
        "snapshot_year_cos_4",
    ]
    ordered += sorted([c for c in train_nodes.columns if c.startswith("train_relation_type_")])
    ordered += sorted([c for c in train_nodes.columns if c.startswith("snapshot_day_of_week_")])
    _require_columns(features=train_nodes, expected=ordered, table_name="train_nodes")
    return train_nodes.loc[:, ordered]


def build_ordered_station_nodes(*, station_nodes: pd.DataFrame) -> pd.DataFrame:
    """Order station-node columns according to the exported GNN schema."""
    lap_emb_cols = sorted(
        [c for c in station_nodes.columns if c.startswith("lap_emb_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    ordered = [
        "snapshot_ts",
        "station_id",
        "station_lat",
        "station_lon",
        "weather_temperature_2m",
        "weather_rain",
        "weather_snowfall",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ]
    ordered += lap_emb_cols
    ordered += ["stopped_nb_of_trains", "stopped_avg_delay"]
    _require_columns(features=station_nodes, expected=ordered, table_name="station_nodes")
    return station_nodes.loc[:, ordered]


def build_ordered_station_to_station_edges(
    *, station_to_station_edges: pd.DataFrame
) -> pd.DataFrame:
    """Order station-to-station edge columns according to the exported GNN schema."""
    ordered = [
        "snapshot_ts",
        "link_id",
        "u_station_id",
        "v_station_id",
        "link_distance",
        "uv_angle_sin",
        "uv_angle_cos",
        "vu_angle_sin",
        "vu_angle_cos",
        "uv_nb_of_trains",
        "uv_avg_delay",
        "vu_nb_of_trains",
        "vu_avg_delay",
    ]
    _require_columns(
        features=station_to_station_edges,
        expected=ordered,
        table_name="station_to_station_edges",
    )
    return station_to_station_edges.loc[:, ordered]


def build_ordered_train_to_past_station_edges(
    *, train_to_past_station_edges: pd.DataFrame
) -> pd.DataFrame:
    """Order train-to-past-station edge columns according to the exported GNN schema."""
    ordered = [
        "snapshot_ts",
        "train_node_id",
        "station_id",
        "past_planned_delta",
        "past_delay_sec",
        "past_idx",
        "past_event_type_A",
        "past_event_type_D",
        "past_event_type_P",
    ]
    _require_columns(
        features=train_to_past_station_edges,
        expected=ordered,
        table_name="train_to_past_station_edges",
    )
    return train_to_past_station_edges.loc[:, ordered]


def build_ordered_train_to_future_station_edges(
    *, train_to_future_station_edges: pd.DataFrame
) -> pd.DataFrame:
    """Order train-to-future-station edge columns according to the exported GNN schema."""
    ordered = [
        "snapshot_ts",
        "train_node_id",
        "station_id",
        "future_planned_delta",
        "future_idx",
        "future_delay_delta",
        "future_event_type_A",
        "future_event_type_D",
        "future_event_type_P",
    ]
    _require_columns(
        features=train_to_future_station_edges,
        expected=ordered,
        table_name="train_to_future_station_edges",
    )
    return train_to_future_station_edges.loc[:, ordered]


def build_ordered_feature_tables(
    *,
    train_nodes: pd.DataFrame,
    station_nodes: pd.DataFrame,
    station_to_station_edges: pd.DataFrame,
    train_to_past_station_edges: pd.DataFrame,
    train_to_future_station_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Order all graph feature tables according to the exported GNN schema."""
    return (
        build_ordered_train_nodes(train_nodes=train_nodes),
        build_ordered_station_nodes(station_nodes=station_nodes),
        build_ordered_station_to_station_edges(station_to_station_edges=station_to_station_edges),
        build_ordered_train_to_past_station_edges(train_to_past_station_edges=train_to_past_station_edges),
        build_ordered_train_to_future_station_edges(train_to_future_station_edges=train_to_future_station_edges),
    )


def _build_uv_vu_swap_indices(cols: list[str]) -> np.ndarray:
    """Build feature-column indices that swap uv and vu directional fields."""
    pos = {c: i for i, c in enumerate(cols)}
    idx: list[int] = []
    for c in cols:
        if c.startswith("uv_"):
            twin = "vu_" + c[3:]
            idx.append(pos[twin] if twin in pos else pos[c])
        elif c.startswith("vu_"):
            twin = "uv_" + c[3:]
            idx.append(pos[twin] if twin in pos else pos[c])
        else:
            idx.append(pos[c])
    return np.asarray(idx, dtype=np.int64)


def _map_ids_to_local(ids: pd.Series, lookup: pd.Series) -> np.ndarray:
    """Map stable table ids to local graph node indices."""
    return ids.map(lookup).to_numpy(dtype=np.int64, copy=False)


def build_split_graphs(
    *,
    split_tables: dict[str, pd.DataFrame],
    feature_spec: dict[str, list[str]],
) -> list[HeteroData]:
    """Convert one split of feature tables into PyG heterogeneous graphs."""
    grouped: dict[str, dict[int, pd.DataFrame]] = {}
    empty: dict[str, pd.DataFrame] = {}
    for name, df in split_tables.items():
        grouped[name] = {
            int(ts): g.reset_index(drop=True)
            for ts, g in df.groupby("snapshot_ts", sort=False)
        }
        empty[name] = df.iloc[0:0].copy()

    snapshots = np.sort(
        split_tables["train_nodes"]["snapshot_ts"].drop_duplicates().to_numpy(dtype=np.int64, copy=False)
    )
    sts_swap_idx = _build_uv_vu_swap_indices(feature_spec["sts_edge_cols"])

    graphs: list[HeteroData] = []

    for snapshot_ts in snapshots:
        tn = grouped["train_nodes"].get(int(snapshot_ts), empty["train_nodes"]).sort_values(
            "train_node_id", kind="mergesort"
        ).reset_index(drop=True)
        sn = grouped["station_nodes"].get(int(snapshot_ts), empty["station_nodes"]).sort_values(
            "station_id", kind="mergesort"
        ).reset_index(drop=True)
        sts = grouped["station_to_station_edges"].get(int(snapshot_ts), empty["station_to_station_edges"]).reset_index(drop=True)
        past = grouped["train_to_past_station_edges"].get(int(snapshot_ts), empty["train_to_past_station_edges"]).reset_index(drop=True)
        future = grouped["train_to_future_station_edges"].get(int(snapshot_ts), empty["train_to_future_station_edges"]).reset_index(drop=True)
        y_future = grouped["train_to_future_station_edges_y"].get(int(snapshot_ts), empty["train_to_future_station_edges_y"]).reset_index(drop=True)

        train_lookup = pd.Series(
            np.arange(len(tn), dtype=np.int64),
            index=pd.to_numeric(tn["train_node_id"]).to_numpy(dtype=np.int64, copy=False),
        )
        station_lookup = pd.Series(
            np.arange(len(sn), dtype=np.int64),
            index=pd.to_numeric(sn["station_id"]).to_numpy(dtype=np.int64, copy=False),
        )

        data = HeteroData()
        data.snapshot_ts = int(snapshot_ts)
        data.train_ids = tn["train_id"].astype(str).tolist()
        data.service_dates = tn["service_date"].astype(str).tolist()
        data["train"].x = torch.from_numpy(tn.loc[:, feature_spec["train_node_cols"]].to_numpy(dtype=np.float32, copy=True))
        data["station"].x = torch.from_numpy(sn.loc[:, feature_spec["station_node_cols"]].to_numpy(dtype=np.float32, copy=True))

        sts_u = _map_ids_to_local(sts["u_station_id"], station_lookup)
        sts_v = _map_ids_to_local(sts["v_station_id"], station_lookup)
        sts_edge_index = np.vstack([sts_u, sts_v]).astype(np.int64, copy=False)
        sts_edge_attr = sts.loc[:, feature_spec["sts_edge_cols"]].to_numpy(dtype=np.float32, copy=True)
        data[("station", "to", "station")].edge_index = torch.from_numpy(sts_edge_index)
        data[("station", "to", "station")].edge_attr = torch.from_numpy(sts_edge_attr)
        data[("station", "to_rev", "station")].edge_index = torch.from_numpy(np.vstack([sts_v, sts_u]).astype(np.int64, copy=False))
        data[("station", "to_rev", "station")].edge_attr = torch.from_numpy(sts_edge_attr[:, sts_swap_idx].astype(np.float32, copy=False))

        past_t = _map_ids_to_local(past["train_node_id"], train_lookup)
        past_s = _map_ids_to_local(past["station_id"], station_lookup)
        past_edge_index = np.vstack([past_t, past_s]).astype(np.int64, copy=False)
        past_edge_attr = past.loc[:, feature_spec["past_edge_cols"]].to_numpy(dtype=np.float32, copy=True)
        data[("train", "past", "station")].edge_index = torch.from_numpy(past_edge_index)
        data[("train", "past", "station")].edge_attr = torch.from_numpy(past_edge_attr)
        data[("station", "past_rev", "train")].edge_index = torch.from_numpy(np.vstack([past_s, past_t]).astype(np.int64, copy=False))
        data[("station", "past_rev", "train")].edge_attr = torch.from_numpy(past_edge_attr)

        fut_t = _map_ids_to_local(future["train_node_id"], train_lookup)
        fut_s = _map_ids_to_local(future["station_id"], station_lookup)
        fut_edge_index = np.vstack([fut_t, fut_s]).astype(np.int64, copy=False)
        fut_edge_attr = future.loc[:, feature_spec["future_edge_cols"]].to_numpy(dtype=np.float32, copy=True)
        data[("train", "future", "station")].edge_index = torch.from_numpy(fut_edge_index)
        data[("train", "future", "station")].edge_attr = torch.from_numpy(fut_edge_attr)
        data[("station", "future_rev", "train")].edge_index = torch.from_numpy(np.vstack([fut_s, fut_t]).astype(np.int64, copy=False))
        data[("station", "future_rev", "train")].edge_attr = torch.from_numpy(fut_edge_attr)
        data[("train", "future", "station")].y = torch.from_numpy(pd.to_numeric(y_future["future_delay_delta"]).to_numpy(dtype=np.float32, copy=False))
        data[("train", "future", "station")].future_rank = torch.from_numpy(
            np.rint(pd.to_numeric(y_future["future_idx"]).to_numpy(dtype=np.float64, copy=False)).astype(np.int16)
        )
        graphs.append(data)
    return graphs


def _write_temp_chunk_tables(
    *,
    tmp_dir: Path,
    chunk_id: int,
    train_features: pd.DataFrame,
    station_features: pd.DataFrame,
    station_to_station_features: pd.DataFrame,
    train_to_past_station_features: pd.DataFrame,
    train_to_future_station_features: pd.DataFrame,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write intermediate graph feature tables for one snapshot chunk."""
    train_chunk_path = tmp_dir / f"train_nodes_part_{chunk_id:05d}.parquet"
    station_chunk_path = tmp_dir / f"station_nodes_part_{chunk_id:05d}.parquet"
    station_to_station_chunk_path = tmp_dir / f"station_to_station_edges_part_{chunk_id:05d}.parquet"
    train_to_past_station_chunk_path = tmp_dir / f"train_to_past_station_edges_part_{chunk_id:05d}.parquet"
    train_to_future_station_chunk_path = tmp_dir / f"train_to_future_station_edges_part_{chunk_id:05d}.parquet"
    train_features.to_parquet(train_chunk_path, index=False)
    station_features.to_parquet(station_chunk_path, index=False)
    station_to_station_features.to_parquet(station_to_station_chunk_path, index=False)
    train_to_past_station_features.to_parquet(train_to_past_station_chunk_path, index=False)
    train_to_future_station_features.to_parquet(train_to_future_station_chunk_path, index=False)
    return (
        train_chunk_path,
        station_chunk_path,
        station_to_station_chunk_path,
        train_to_past_station_chunk_path,
        train_to_future_station_chunk_path,
    )


def normalize_and_export(
    *,
    train_chunk_paths: list[Path],
    station_chunk_paths: list[Path],
    station_to_station_chunk_paths: list[Path],
    train_to_past_station_chunk_paths: list[Path],
    train_to_future_station_chunk_paths: list[Path],
    output_dir: Path,
    train_snapshots: pd.DatetimeIndex,
    test_snapshots: pd.DatetimeIndex,
    normalization_stats: dict[str, dict[str, dict[str, float | bool]]],
) -> None:
    """Normalize chunk tables and export train/test parquet feature tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_snapshot_ns = _to_ns_int64(train_snapshots)
    test_snapshot_ns = _to_ns_int64(test_snapshots)
    for split_name in ("train", "test"):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

    def _load_normalized_table(
        *,
        table_name: str,
        chunk_paths: list[Path],
    ) -> pd.DataFrame:
        """Load and normalize all chunks for one graph table."""
        return pd.concat(
            [
                _apply_normalization_stats(
                    features=pd.read_parquet(p),
                    normalization_stats=normalization_stats[table_name],
                )
                for p in chunk_paths
            ],
            ignore_index=True,
        )

    def _write_split_table(
        *,
        table_name: str,
        table_df: pd.DataFrame,
    ) -> None:
        """Write one normalized table split into train and test partitions."""
        split_train, split_test = _split_table_by_snapshot_ts(
            table_name=table_name,
            features=table_df,
            train_snapshot_ns=train_snapshot_ns,
            test_snapshot_ns=test_snapshot_ns,
        )
        split_train.to_parquet(output_dir / "train" / f"{table_name}.parquet", index=False)
        split_test.to_parquet(output_dir / "test" / f"{table_name}.parquet", index=False)

    train_nodes = build_ordered_train_nodes(
        train_nodes=_load_normalized_table(
            table_name="train_nodes",
            chunk_paths=train_chunk_paths,
        )
    )
    _write_split_table(table_name="train_nodes", table_df=train_nodes)
    del train_nodes

    station_nodes = build_ordered_station_nodes(
        station_nodes=_load_normalized_table(
            table_name="station_nodes",
            chunk_paths=station_chunk_paths,
        )
    )
    _write_split_table(table_name="station_nodes", table_df=station_nodes)
    del station_nodes

    station_to_station_edges = build_ordered_station_to_station_edges(
        station_to_station_edges=_load_normalized_table(
            table_name="station_to_station_edges",
            chunk_paths=station_to_station_chunk_paths,
        )
    )
    _write_split_table(table_name="station_to_station_edges", table_df=station_to_station_edges)
    del station_to_station_edges

    train_to_past_station_edges = build_ordered_train_to_past_station_edges(
        train_to_past_station_edges=_load_normalized_table(
            table_name="train_to_past_station_edges",
            chunk_paths=train_to_past_station_chunk_paths,
        )
    )
    _write_split_table(
        table_name="train_to_past_station_edges",
        table_df=train_to_past_station_edges,
    )
    del train_to_past_station_edges

    train_to_future_station_edges = build_ordered_train_to_future_station_edges(
        train_to_future_station_edges=_load_normalized_table(
            table_name="train_to_future_station_edges",
            chunk_paths=train_to_future_station_chunk_paths,
        )
    )
    train_to_future_station_edges, train_to_future_station_edges_y = split_future_edge_targets(
        train_to_future_station_edges=train_to_future_station_edges
    )
    train_to_future_station_edges_y = restore_future_idx_for_target_table(
        target_table=train_to_future_station_edges_y,
        future_idx_stats=normalization_stats["train_to_future_station_edges"]["future_idx"],
    )
    _write_split_table(
        table_name="train_to_future_station_edges",
        table_df=train_to_future_station_edges,
    )
    _write_split_table(
        table_name="train_to_future_station_edges_y",
        table_df=train_to_future_station_edges_y,
    )


def _get_graph_feature_spec(split_tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Derive PyG feature-column groups from ordered graph tables."""
    return {
        "train_node_cols": [c for c in split_tables["train_nodes"].columns if c not in {"snapshot_ts", "train_node_id", "train_id", "service_date"}],
        "station_node_cols": [c for c in split_tables["station_nodes"].columns if c not in {"snapshot_ts", "station_id"}],
        "sts_edge_cols": [c for c in split_tables["station_to_station_edges"].columns if c not in {"snapshot_ts", "link_id", "u_station_id", "v_station_id"}],
        "past_edge_cols": [c for c in split_tables["train_to_past_station_edges"].columns if c not in {"snapshot_ts", "train_node_id", "station_id"}],
        "future_edge_cols": [c for c in split_tables["train_to_future_station_edges"].columns if c not in {"snapshot_ts", "train_node_id", "station_id"}],
    }


def export_processed_graph_chunks(
    *,
    train_chunk_paths: list[Path],
    station_chunk_paths: list[Path],
    station_to_station_chunk_paths: list[Path],
    train_to_past_station_chunk_paths: list[Path],
    train_to_future_station_chunk_paths: list[Path],
    output_dir: Path,
    train_snapshots: pd.DatetimeIndex,
    test_snapshots: pd.DatetimeIndex,
    normalization_stats: dict[str, dict[str, dict[str, float | bool]]],
    graph_chunk_size: int,
    show_progress: bool = True,
) -> None:
    """Normalize graph chunks and export PyG graph part files."""
    processed_dir = output_dir
    for split_name in ("train", "test"):
        split_dir = processed_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for p in split_dir.glob("graphs_part_*.pt"):
            p.unlink(missing_ok=True)
        for p in split_dir.glob("edge_meta_part_*.pt"):
            p.unlink(missing_ok=True)

    train_snapshot_ns = _to_ns_int64(train_snapshots)
    test_snapshot_ns = _to_ns_int64(test_snapshots)
    feature_spec: dict[str, list[str]] | None = None
    graphs_by_split = {"train": [], "test": []}
    part_idx_by_split = {"train": 0, "test": 0}

    def _flush_split(split_name: str) -> None:
        """Write the buffered graph list for one split."""
        if not graphs_by_split[split_name]:
            return
        split_dir = processed_dir / split_name
        part_idx = part_idx_by_split[split_name]
        torch.save(graphs_by_split[split_name], split_dir / f"graphs_part_{part_idx:05d}.pt")
        graphs_by_split[split_name].clear()
        part_idx_by_split[split_name] += 1
        gc.collect()
    for chunk_id, chunk_paths in enumerate(
        tqdm(
            zip(
                train_chunk_paths,
                station_chunk_paths,
                station_to_station_chunk_paths,
                train_to_past_station_chunk_paths,
                train_to_future_station_chunk_paths,
                strict=True,
            ),
            total=len(train_chunk_paths),
            desc="Export GNN graph chunks",
            disable=not show_progress,
        )
    ):
        train_nodes = build_ordered_train_nodes(train_nodes=_apply_normalization_stats(features=pd.read_parquet(chunk_paths[0]), normalization_stats=normalization_stats["train_nodes"]))
        station_nodes = build_ordered_station_nodes(station_nodes=_apply_normalization_stats(features=pd.read_parquet(chunk_paths[1]), normalization_stats=normalization_stats["station_nodes"]))
        station_to_station_edges = build_ordered_station_to_station_edges(station_to_station_edges=_apply_normalization_stats(features=pd.read_parquet(chunk_paths[2]), normalization_stats=normalization_stats["station_to_station_edges"]))
        train_to_past_station_edges = build_ordered_train_to_past_station_edges(train_to_past_station_edges=_apply_normalization_stats(features=pd.read_parquet(chunk_paths[3]), normalization_stats=normalization_stats["train_to_past_station_edges"]))
        train_to_future_station_edges = build_ordered_train_to_future_station_edges(train_to_future_station_edges=_apply_normalization_stats(features=pd.read_parquet(chunk_paths[4]), normalization_stats=normalization_stats["train_to_future_station_edges"]))
        train_to_future_station_edges, train_to_future_station_edges_y = split_future_edge_targets(train_to_future_station_edges=train_to_future_station_edges)
        train_to_future_station_edges_y = restore_future_idx_for_target_table(target_table=train_to_future_station_edges_y, future_idx_stats=normalization_stats["train_to_future_station_edges"]["future_idx"])
        chunk_tables = {
            "train_nodes": train_nodes,
            "station_nodes": station_nodes,
            "station_to_station_edges": station_to_station_edges,
            "train_to_past_station_edges": train_to_past_station_edges,
            "train_to_future_station_edges": train_to_future_station_edges,
            "train_to_future_station_edges_y": train_to_future_station_edges_y,
        }
        split_tables = {
            split_name: {
                name: _split_table_by_snapshot_ts(
                    table_name=name,
                    features=table_df,
                    train_snapshot_ns=train_snapshot_ns,
                    test_snapshot_ns=test_snapshot_ns,
                )[0 if split_name == "train" else 1]
                for name, table_df in chunk_tables.items()
            }
            for split_name in ("train", "test")
        }
        for split_name in ("train", "test"):
            if len(split_tables[split_name]["train_nodes"]) == 0:
                continue
            if feature_spec is None:
                feature_spec = _get_graph_feature_spec(split_tables[split_name])
                write_yaml(output_dir / "feature_spec.yaml", feature_spec)
            graphs_by_split[split_name].extend(build_split_graphs(split_tables=split_tables[split_name], feature_spec=feature_spec))
            if len(graphs_by_split[split_name]) >= graph_chunk_size:
                _flush_split(split_name)
        del train_nodes, station_nodes, station_to_station_edges, train_to_past_station_edges, train_to_future_station_edges, train_to_future_station_edges_y, chunk_tables, split_tables
        gc.collect()
    for split_name in ("train", "test"):
        _flush_split(split_name)


def prepare_station_feature_cache(
    *,
    weather_lookup: dict[str, object],
    station_embedding_lookup: dict[int, np.ndarray],
    station_position_lookup: dict[int, tuple[float, float]],
    station_embedding_dim: int,
) -> dict[str, object]:
    """Combine station weather, position, and embedding arrays for graph building."""
    station_op_ids = weather_lookup["weather_op_ids"].to_numpy(dtype=np.int64, copy=False)
    station_embeddings = np.empty((len(station_op_ids), station_embedding_dim), dtype=np.float32)
    station_lat = np.empty(len(station_op_ids), dtype=np.float32)
    station_lon = np.empty(len(station_op_ids), dtype=np.float32)
    for station_id, op_id in enumerate(station_op_ids):
        emb = station_embedding_lookup.get(int(op_id))
        if emb is None:
            raise KeyError(f"Missing station embedding for op_id={int(op_id)}")
        emb_arr = np.asarray(emb, dtype=np.float32)
        if emb_arr.shape[0] != station_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for op_id={int(op_id)}: "
                f"got {emb_arr.shape[0]}, expected {station_embedding_dim}"
            )
        station_embeddings[station_id] = emb_arr
        pos = station_position_lookup.get(int(op_id))
        if pos is None:
            raise KeyError(f"Missing station position for op_id={int(op_id)}")
        station_lat[station_id] = np.float32(pos[0])
        station_lon[station_id] = np.float32(pos[1])

    return {
        "weather_times": weather_lookup["weather_times"],
        "weather_temperature": weather_lookup["weather_temperature"],
        "weather_rain": weather_lookup["weather_rain"],
        "weather_snowfall": weather_lookup["weather_snowfall"],
        "weather_rh": weather_lookup["weather_rh"],
        "weather_wind": weather_lookup["weather_wind"],
        "station_op_ids": station_op_ids,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "station_embeddings": station_embeddings,
        "station_embedding_dim": station_embedding_dim,
    }


def prepare_gnn_build_context(
    *,
    events_dir: Path,
    journeys_dir: Path,
    weather_path: Path,
    node_links_path: Path,
    op_nodes_path: Path,
    station_embedding_dim: int,
    dataset_core_spec: Path,
    show_progress: bool,
) -> dict[str, object]:
    """Load shared inputs and indexes needed for the GNN build."""
    # Snapshot plan and base train/test timeline.
    cfg = read_yaml(dataset_core_spec)
    train_snapshots = pd.DatetimeIndex(pd.to_datetime(cfg["train_snapshots"]))
    test_snapshots = pd.DatetimeIndex(pd.to_datetime(cfg["test_snapshots"]))
    all_snapshots = pd.DatetimeIndex(train_snapshots.tolist() + test_snapshots.tolist()).sort_values()
    sampled_ns = _to_ns_int64(all_snapshots)

    # Build index from silver events/journeys.
    index = build_indexes(
        events_dir=events_dir,
        journeys_dir=journeys_dir,
        snapshot_config=cfg,
        splits_to_build=["train", "test"],
        index_events_optional_get=[],
        index_journeys_optional_get=["train_relation", "operator", "deduced_path"],
        show_progress=show_progress,
    )

    node_links = pd.read_parquet(node_links_path, columns=["link_id", "u_node_id", "v_node_id", "distance_m"])
    station_op_ids = np.unique(
        pd.concat([node_links["u_node_id"], node_links["v_node_id"]], ignore_index=True).to_numpy(dtype=np.int64)
    )
    # Build weather tensor slices on the full station set used by node_links.
    weather_lookup = build_weather_lookup(
        weather_path=weather_path,
        snapshots=all_snapshots,
        event_op_ids=station_op_ids,
    )
    link_distance_m = {
        int(r.link_id): float(r.distance_m)
        for r in node_links.itertuples(index=False)
    }
    link_angle_lookup = build_link_angle_lookup(
        node_links=node_links,
        op_nodes_path=op_nodes_path,
    )
    op_nodes = pd.read_parquet(op_nodes_path, columns=["op_id", "lat", "lon"])
    op_nodes["op_id"] = pd.to_numeric(op_nodes["op_id"], errors="raise").astype(np.int64)
    op_nodes["lat"] = pd.to_numeric(op_nodes["lat"], errors="raise").astype(np.float64)
    op_nodes["lon"] = pd.to_numeric(op_nodes["lon"], errors="raise").astype(np.float64)
    station_position_lookup = {
        int(r.op_id): (float(r.lat), float(r.lon))
        for r in op_nodes.itertuples(index=False)
    }
    link_endpoints = {
        int(r.link_id): (int(r.u_node_id), int(r.v_node_id))
        for r in node_links.itertuples(index=False)
    }
    station_embeddings, node_order, _, _, _ = create_station_embeddings_from_silver(
        node_links_path=node_links_path,
        op_nodes_path=op_nodes_path,
        embedding_dim=station_embedding_dim,
    )
    station_embedding_lookup = {int(op_id): station_embeddings[idx] for idx, op_id in enumerate(node_order)}

    return {
        "cfg": cfg,
        "train_snapshots": train_snapshots,
        "test_snapshots": test_snapshots,
        "all_snapshots": all_snapshots,
        "sampled_ns": sampled_ns,
        "index": index,
        "weather_lookup": weather_lookup,
        "link_distance_m": link_distance_m,
        "link_angle_lookup": link_angle_lookup,
        "station_position_lookup": station_position_lookup,
        "link_endpoints": link_endpoints,
        "station_embedding_lookup": station_embedding_lookup,
        "station_embedding_dim": station_embedding_dim,
    }


def build_and_export_streaming(
    *,
    input_dir: Path,
    journeys_dir: Path,
    weather_path: Path,
    node_links_path: Path,
    op_nodes_path: Path,
    station_embedding_dim: int,
    nb_past_events: int,
    missing_event_placeholder: int,
    dataset_core_spec: Path,
    output_dir: Path,
    show_progress: bool = True,
    chunk_size_snapshots: int = 1000,
    graph_chunk_size: int = 1000,
) -> None:
    """Build and export the GNN dataset with chunked graph construction."""
    # Stage 1: load and pre-index all silver-derived inputs.
    stage = prepare_gnn_build_context(
        events_dir=input_dir,
        journeys_dir=journeys_dir,
        weather_path=weather_path,
        node_links_path=node_links_path,
        op_nodes_path=op_nodes_path,
        station_embedding_dim=station_embedding_dim,
        dataset_core_spec=dataset_core_spec,
        show_progress=show_progress,
    )
    cfg = stage["cfg"]
    train_snapshots = stage["train_snapshots"]
    test_snapshots = stage["test_snapshots"]
    all_snapshots = stage["all_snapshots"]
    sampled_ns = stage["sampled_ns"]
    index = stage["index"]
    weather_lookup = stage["weather_lookup"]
    link_distance_m = stage["link_distance_m"]
    link_angle_lookup = stage["link_angle_lookup"]
    link_endpoints = stage["link_endpoints"]
    station_embedding_lookup = stage["station_embedding_lookup"]
    station_position_lookup = stage["station_position_lookup"]
    station_feature_cache = prepare_station_feature_cache(
        weather_lookup=weather_lookup,
        station_embedding_lookup=station_embedding_lookup,
        station_position_lookup=station_position_lookup,
        station_embedding_dim=station_embedding_dim,
    )

    tmp_dir = output_dir / "_tmp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_chunk_paths: list[Path] = []
    station_chunk_paths: list[Path] = []
    station_to_station_chunk_paths: list[Path] = []
    train_to_past_station_chunk_paths: list[Path] = []
    train_to_future_station_chunk_paths: list[Path] = []

    # Stage 2: pass 1/2 - build ML-ready chunk tables and collect normalization stats.
    running_stats: dict[str, dict[str, dict[str, float | int | bool]]] = {
        "train_nodes": {},
        "station_nodes": {},
        "station_to_station_edges": {},
        "train_to_past_station_edges": {},
        "train_to_future_station_edges": {},
    }
    n_chunks = (len(all_snapshots) + chunk_size_snapshots - 1) // chunk_size_snapshots
    chunk_iter = tqdm(
        _iter_snapshot_chunks(
            all_snapshots,
            sampled_ns,
            chunk_size=chunk_size_snapshots,
        ),
        total=n_chunks,
        desc="Build train node chunks",
        disable=not show_progress,
    )
    for chunk_id, (chunk_snapshots, chunk_ns) in enumerate(chunk_iter):
        (
            train_features,
            station_features,
            station_to_station_features,
            train_to_past_station_features,
            train_to_future_station_features,
        ) = build_nodes_and_edges_features(
            index=index,
            snapshots=chunk_snapshots,
            sampled_ns=chunk_ns,
            nb_past_events=nb_past_events,
            nb_future_events=int(cfg["n_future"]),
            idle_time_beg=int(cfg["idle_time_beg"]),
            idle_time_end=int(cfg["idle_time_end"]),
            missing_event_placeholder=missing_event_placeholder,
            station_feature_cache=station_feature_cache,
            link_distance_m=link_distance_m,
            link_angle_lookup=link_angle_lookup,
            link_endpoints=link_endpoints,
            show_progress=False,
        )
        (
            train_features,
            station_features,
            station_to_station_features,
            train_to_past_station_features,
            train_to_future_station_features,
            ml_groups,
        ) = to_ml_features(
            train_nodes=train_features,
            station_nodes=station_features,
            station_to_station_edges=station_to_station_features,
            train_to_past_station_edges=train_to_past_station_features,
            train_to_future_station_edges=train_to_future_station_features,
        )
        table_map = {
            "train_nodes": train_features,
            "station_nodes": station_features,
            "station_to_station_edges": station_to_station_features,
            "train_to_past_station_edges": train_to_past_station_features,
            "train_to_future_station_edges": train_to_future_station_features,
        }
        for table_name, table_df in table_map.items():
            train_mask = pd.to_datetime(table_df["snapshot_ts"]).isin(train_snapshots).to_numpy()
            _update_running_stats(
                running=running_stats[table_name],
                features=table_df,
                train_mask=train_mask,
                signed_sqrt_cols=ml_groups[table_name]["signed_sqrt_cols"],
                zscore_cols=ml_groups[table_name]["zscore_cols"],
                minmax_cols=ml_groups[table_name]["minmax_cols"],
            )
        (
            train_chunk_path,
            station_chunk_path,
            station_to_station_chunk_path,
            train_to_past_station_chunk_path,
            train_to_future_station_chunk_path,
        ) = _write_temp_chunk_tables(
            tmp_dir=tmp_dir,
            chunk_id=chunk_id,
            train_features=train_features,
            station_features=station_features,
            station_to_station_features=station_to_station_features,
            train_to_past_station_features=train_to_past_station_features,
            train_to_future_station_features=train_to_future_station_features,
        )
        train_chunk_paths.append(train_chunk_path)
        station_chunk_paths.append(station_chunk_path)
        station_to_station_chunk_paths.append(station_to_station_chunk_path)
        train_to_past_station_chunk_paths.append(train_to_past_station_chunk_path)
        train_to_future_station_chunk_paths.append(train_to_future_station_chunk_path)

        del train_features
        del station_to_station_features
        del station_features
        del train_to_past_station_features
        del train_to_future_station_features
        gc.collect()

    # Stage 3: finalize normalization parameters from train snapshots.
    normalization_stats = {
        table_name: _finalize_running_stats(table_running)
        for table_name, table_running in running_stats.items()
    }
    del stage, sampled_ns, index, weather_lookup, link_distance_m, link_angle_lookup, link_endpoints
    del station_embedding_lookup, station_position_lookup, station_feature_cache, running_stats
    gc.collect()

    # Stage 4: pass 2/2 - apply normalization and export processed graph chunks.
    export_processed_graph_chunks(
        train_chunk_paths=train_chunk_paths,
        station_chunk_paths=station_chunk_paths,
        station_to_station_chunk_paths=station_to_station_chunk_paths,
        train_to_past_station_chunk_paths=train_to_past_station_chunk_paths,
        train_to_future_station_chunk_paths=train_to_future_station_chunk_paths,
        output_dir=output_dir,
        train_snapshots=train_snapshots,
        test_snapshots=test_snapshots,
        normalization_stats=normalization_stats,
        graph_chunk_size=graph_chunk_size,
        show_progress=show_progress,
    )
    normalization_path = output_dir / "normalization.yaml"

    write_yaml(normalization_path, normalization_stats)

    # Stage 5: cleanup temporary chunk files.
    for p in train_chunk_paths:
        p.unlink(missing_ok=True)
    for p in station_chunk_paths:
        p.unlink(missing_ok=True)
    for p in station_to_station_chunk_paths:
        p.unlink(missing_ok=True)
    for p in train_to_past_station_chunk_paths:
        p.unlink(missing_ok=True)
    for p in train_to_future_station_chunk_paths:
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()


def build_gnn_dataset(
    *,
    silver_dir: Path,
    station_embedding_dim: int,
    nb_past_events: int,
    output_dir: Path,
    dataset_core_spec: Path,
    missing_event_placeholder: int = -1,
    show_progress: bool = True,
    graph_chunk_size: int = 10000,
) -> None:
    """Build the Gold GNN dataset from a Silver dataset and Gold core.

    Args:
        silver_dir: Root directory of the Silver dataset.
        station_embedding_dim: Number of station embedding dimensions to add.
        nb_past_events: Number of past event slots encoded per active train.
        output_dir: Directory where graph chunks, schema, and normalization
            metadata are written.
        dataset_core_spec: Path to the Gold core specification YAML.
        missing_event_placeholder: Operational point id used for padded event
            slots.
        show_progress: Whether to display progress bars.
        graph_chunk_size: Maximum number of snapshot graphs written per graph
            part file.

    Returns:
        None. The function writes the GNN dataset under `output_dir`.
    """
    input_dir = silver_dir / "events"
    journeys_dir = silver_dir / "journeys"
    weather_path = silver_dir / "static" / "weather.parquet"
    node_links_path = silver_dir / "static" / "node_links.parquet"
    op_nodes_path = silver_dir / "static" / "op_nodes.parquet"

    build_and_export_streaming(
        input_dir=input_dir,
        journeys_dir=journeys_dir,
        weather_path=weather_path,
        node_links_path=node_links_path,
        op_nodes_path=op_nodes_path,
        station_embedding_dim=station_embedding_dim,
        nb_past_events=nb_past_events,
        missing_event_placeholder=missing_event_placeholder,
        dataset_core_spec=dataset_core_spec,
        output_dir=output_dir,
        show_progress=show_progress,
        graph_chunk_size=graph_chunk_size,
    )
    print(f"[gold_gnn_data] Wrote GNN node/edge tables to {output_dir}")
