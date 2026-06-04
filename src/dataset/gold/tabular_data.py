"""Gold tabular dataset construction utilities."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def _add_station_embedding_features(
    features: pd.DataFrame,
    *,
    station_embedding_lookup: dict[int, np.ndarray],
    station_embedding_dim: int,
) -> pd.DataFrame:
    """Append station embedding components for each past and future event slot."""
    zero_vec = np.zeros(station_embedding_dim, dtype=np.float64)
    new_cols: dict[str, np.ndarray] = {}
    for side in ("past", "future"):
        i = 1
        while f"{side}_event_id_{i}" in features.columns:
            ids = pd.to_numeric(features[f"{side}_event_id_{i}"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            emb = np.vstack([station_embedding_lookup.get(int(op_id), zero_vec) for op_id in ids])
            for comp in range(station_embedding_dim):
                new_cols[f"{side}_event_emb_{i}_{comp}"] = emb[:, comp]
            i += 1
    return pd.concat([features, pd.DataFrame(new_cols, index=features.index)], axis=1)


def _add_ml_features_no_norm(
    features: pd.DataFrame,
    *,
    station_embedding_lookup: dict[int, np.ndarray],
    station_embedding_dim: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Add engineered features up to pre-normalization stage."""
    # Journey-level categorical metadata.
    relation_categories = ["EURST", "EXTRA", "IC", "ICE", "INT", "L", "P", "TGV", "THAL", "S", "CHARTER", "nan"]
    rel = features["train_relation"].astype("string").fillna("nan").str.replace(r"^(S).*", r"\1", regex=True)
    features["train_relation_type"] = rel.str.split(" ").str[0]
    features["train_relation_type"] = pd.Categorical(
        features["train_relation_type"],
        categories=relation_categories,
    )
    features["operator_is_sncb_nmbs"] = (
        features["operator"].astype("string") == "SNCB/NMBS"
    ).astype(int)

    # Convert planned timestamps to seconds relative to snapshot time.
    snapshot_dt = pd.to_datetime(features["snapshot_ts"])
    planned_cols = [c for c in features.columns if "planned_delta" in c]
    for col in planned_cols:
        planned_dt = pd.to_datetime(features[col])
        features[col] = (planned_dt - snapshot_dt).dt.total_seconds()

    # Calendar/cyclical time features.
    features["snapshot_day_of_year"] = snapshot_dt.dt.dayofyear
    features["snapshot_day_of_week"] = pd.Categorical(
        snapshot_dt.dt.day_name(),
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    features["snapshot_time_of_day"] = (
        snapshot_dt.dt.hour + snapshot_dt.dt.minute / 60.0 + snapshot_dt.dt.second / 3600.0
    )
    frequencies = [1, 2, 4]
    for freq in frequencies:
        features[f"snapshot_hour_sin_{freq}"] = np.sin(
            freq * 2 * np.pi * features["snapshot_time_of_day"] / 24.0
        )
        features[f"snapshot_hour_cos_{freq}"] = np.cos(
            freq * 2 * np.pi * features["snapshot_time_of_day"] / 24.0
        )
    for freq in frequencies:
        features[f"snapshot_year_sin_{freq}"] = np.sin(
            freq * 2 * np.pi * features["snapshot_day_of_year"] / 365.0
        )
        features[f"snapshot_year_cos_{freq}"] = np.cos(
            freq * 2 * np.pi * features["snapshot_day_of_year"] / 365.0
        )

    # One-hot categorical features.
    cat_cols = ["train_relation_type", "snapshot_day_of_week"]
    cat_encoded = pd.get_dummies(features[cat_cols], prefix=cat_cols, dummy_na=False, dtype=int)
    features = pd.concat([features, cat_encoded], axis=1)

    # Targets become residuals w.r.t. last known delay.
    future_delay_cols = [c for c in features.columns if c.startswith("future_delay_delta_")]
    for col in future_delay_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce") - pd.to_numeric(
            features["last_known_delay"], errors="coerce"
        )

    # Event-type one-hot expansion for each past/future slot.
    event_type_ohe_cols: dict[str, np.ndarray] = {}
    for side in ("past", "future"):
        i = 1
        while f"{side}_event_type_{i}" in features.columns:
            type_col = f"{side}_event_type_{i}"
            type_values = features[type_col].astype("string")
            event_type_ohe_cols[f"{type_col}_A"] = (type_values == "A").astype(int).to_numpy(copy=False)
            event_type_ohe_cols[f"{type_col}_D"] = (type_values == "D").astype(int).to_numpy(copy=False)
            event_type_ohe_cols[f"{type_col}_P"] = (type_values == "P").astype(int).to_numpy(copy=False)
            i += 1
    if event_type_ohe_cols:
        features = pd.concat([features, pd.DataFrame(event_type_ohe_cols, index=features.index)], axis=1)

    # Missing local-link delay defaults to 0 for downstream numeric transforms.
    link_avg_delay_cols = [c for c in features.columns if c.startswith("link") and c.endswith("_avg_delay")]
    for col in link_avg_delay_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    # Declare normalization groups for pass-1 stats collection.
    delay_cols = [
        c
        for c in features.columns
        if c.startswith("past_delay_sec_")
        or c.startswith("future_delay_delta_")
        or (c.startswith("link") and c.endswith("_avg_delay"))
    ]
    minmax_cols = [
        c
        for c in features.columns
        if (c.startswith("link") and c.endswith("_distance"))
        or (c.startswith("link") and c.endswith("_nb_of_trains"))
    ]
    zscore_cols = [
        "weather_temperature_2m",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ]
    signed_sqrt_cols = delay_cols + planned_cols + ["weather_rain", "weather_snowfall"]

    # Signed sqrt transform on delay/time-like columns.
    for col in signed_sqrt_cols:
        values = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=np.float64)
        features[col] = np.sign(values) * np.sqrt(np.abs(values))

    # Append station embedding vectors for each event slot.
    features = _add_station_embedding_features(
        features,
        station_embedding_lookup=station_embedding_lookup,
        station_embedding_dim=station_embedding_dim,
    )
    return features, signed_sqrt_cols, zscore_cols, minmax_cols


def _apply_normalization_stats(
    *,
    features: pd.DataFrame,
    normalization_stats: dict[str, dict[str, float | bool]],
) -> pd.DataFrame:
    """Apply precomputed normalization statistics to numeric feature columns."""
    for col, stats in normalization_stats.items():
        norm = str(stats.get("norm", "zscore"))
        values = pd.to_numeric(features[col], errors="coerce")

        # Min-max branch (with clipping to [0,1]).
        if norm == "minmax":
            vmin = float(stats["min"])
            vmax = float(stats["max"])
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
                features[col] = np.zeros(len(values), dtype=np.float64)
            else:
                scaled = (values - vmin) / (vmax - vmin)
                features[col] = np.clip(scaled, 0.0, 1.0)
            continue

        # Default z-score branch.
        std = float(stats["std"])
        if std == 0.0 or np.isnan(std):
            features[col] = values - float(stats["mean"])
        else:
            features[col] = (values - float(stats["mean"])) / std
    return features


def _token_distance(tok: str, *, link_distance_m: dict[int, float]) -> float:
    """Return numeric distance for one path token."""
    if tok == "" or tok.startswith("STOPPED@"):
        return 0.0
    if tok.endswith("uv") or tok.endswith("vu"):
        link_id = int(tok[:-2])
        return float(link_distance_m[link_id])
    raise ValueError(f"Unexpected link token format: {tok!r}")


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


def _extract_link_window_tokens(
    *,
    train_key: tuple[str, str],
    event_idx: int,
    snap_ns: int,
    last_known_delay: float,
    path_by_key: dict[tuple[str, str], object],
    events_index: dict[str, object],
    n_next_links: int,
    link_distance_m: dict[int, float],
    link_endpoints: dict[int, tuple[int, int]],
) -> list[str]:
    """Build current+next oriented link tokens for one train at one snapshot."""
    path = path_by_key.get(train_key)
    if path is None:
        return [""] * n_next_links
    if n_next_links <= 0:
        return []

    key_slices = events_index["key_slices"]
    if not isinstance(key_slices, dict):
        raise TypeError("events_index['key_slices'] must be a dict.")
    start, end = key_slices[train_key]
    op_ids = events_index["event_op_ids"][start:end]
    planned_ns = events_index["event_planned_ns"][start:end]
    tokens: list[str] = []

    # Before first event: current link unknown, then follow from first transition.
    if event_idx < 0:
        tokens.append("")
        trans_idx = 0
        start_sub_idx = 0
    elif event_idx >= len(path):
        return [""] * n_next_links
    else:
        sub = path[event_idx]
        if len(sub) == 0:
            op_id = int(op_ids[min(event_idx, len(op_ids) - 1)])
            nxt = _next_non_empty_link_id(path, event_idx + 1)
            nxt_text = str(nxt) if nxt is not None else "END"
            tokens.append(f"STOPPED@op{op_id}to{nxt_text}")
            trans_idx = event_idx + 1
            start_sub_idx = 0
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
            tokens.append(oriented[cur_sub_idx])
            trans_idx = event_idx
            start_sub_idx = cur_sub_idx + 1

    while trans_idx < len(path) and len(tokens) < n_next_links:
        sub = path[trans_idx]
        start_node = int(op_ids[min(trans_idx, len(op_ids) - 1)])
        if len(sub) == 0:
            op_id = start_node
            nxt = _next_non_empty_link_id(path, trans_idx + 1)
            nxt_text = str(nxt) if nxt is not None else "END"
            tokens.append(f"STOPPED@op{op_id}to{nxt_text}")
            trans_idx += 1
            start_sub_idx = 0
            continue
        oriented = _orient_subpath_tokens(
            start_node,
            sub,
            link_endpoints=link_endpoints,
        )
        begin = max(start_sub_idx, 0) if trans_idx == max(event_idx, 0) else 0
        for tok in oriented[begin:]:
            if len(tokens) >= n_next_links:
                break
            tokens.append(tok)
        trans_idx += 1
        start_sub_idx = 0

    if len(tokens) < n_next_links:
        tokens.extend([""] * (n_next_links - len(tokens)))
    return tokens


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
    weather_op_ids = pd.Index(weather["op_id"].astype("int64").unique(), name="op_id")
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


def build_feature_rows(
    *,
    index: dict,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    nb_past_events: int,
    nb_future_events: int,
    idle_time_beg: int,
    idle_time_end: int,
    missing_event_placeholder: int,
    n_next_links: int,
    link_distance_m: dict[int, float],
    link_endpoints: dict[int, tuple[int, int]],
    weather_lookup: dict[str, object],
    show_progress: bool,
) -> tuple[list[tuple[object, ...]], list[str]]:
    """Expand active snapshots into raw tabular feature rows."""
    events_index = index["events"]
    journeys_index = index["journeys"]

    # Reusable per-journey/event caches for fast row generation.
    padded_cache: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    weather_time_index_cache: dict[pd.Timestamp, int] = {}
    path_by_key: dict[tuple[str, str], object] = {
        (str(journeys_index["train_ids"][i]), str(journeys_index["service_dates"][i])): journeys_index["deduced_path"][i]
        for i in range(len(journeys_index["train_ids"]))
    }

    # Weather lookup tensors and scalar constants used during snapshot expansion.
    weather_times = weather_lookup["weather_times"]
    weather_op_ids = weather_lookup["weather_op_ids"]
    weather_temperature = weather_lookup["weather_temperature"]
    weather_rain = weather_lookup["weather_rain"]
    weather_snowfall = weather_lookup["weather_snowfall"]
    weather_rh = weather_lookup["weather_rh"]
    weather_wind = weather_lookup["weather_wind"]
    idle_time_beg_ns = idle_time_beg * 60 * 1_000_000_000
    idle_time_end_ns = idle_time_end * 60 * 1_000_000_000

    # Flat tabular schema (base + context windows + local-link + tail/weather).
    base_cols = ["snapshot_ts", "train_id", "service_date"]
    past_cols = []
    for i in range(1, nb_past_events + 1):
        past_cols.extend(
            [
                f"past_event_id_{i}",
                f"past_planned_delta_{i}",
                f"past_delay_sec_{i}",
                f"past_event_type_{i}",
            ]
        )
    future_cols = []
    for i in range(1, nb_future_events + 1):
        future_cols.extend(
            [
                f"future_event_id_{i}",
                f"future_planned_delta_{i}",
                f"future_delay_delta_{i}",
                f"future_event_type_{i}",
            ]
        )
    tail_cols = [
        "last_known_delay",
        "weather_temperature_2m",
        "weather_rain",
        "weather_snowfall",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ]
    local_link_cols: list[str] = []
    for i in range(n_next_links):
        local_link_cols.extend(
            [
                f"link{i}_distance",
                f"link{i}_nb_of_trains",
                f"link{i}_avg_delay",
                f"link{i}_is_placeholder",
            ]
        )
    columns = base_cols + past_cols + future_cols + local_link_cols + tail_cols
    weather_start_idx = len(columns) - len(tail_cols) + 1
    feature_rows: list[tuple[object, ...]] = []
    local_link_start_idx = len(base_cols) + len(past_cols) + len(future_cols)
    tail_start_idx = local_link_start_idx + len(local_link_cols)
    future_planned_1_idx = len(base_cols) + len(past_cols) + 1
    link_slot_width = 4

    # Snapshot expansion loop: one output row per active train key.
    for snap_ts, snap_ns in tqdm(
        zip(snapshots, sampled_ns),
        total=len(snapshots),
        desc="Building snapshots",
        disable=not show_progress,
    ):
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snap_ns,
        )
        active_indices = active_mask.nonzero()[0]
        weather_time = snap_ts.floor("h")
        time_idx = weather_time_index_cache.get(weather_time)
        if time_idx is None:
            time_idx = int(weather_times.get_indexer([weather_time])[0])
            if time_idx < 0:
                raise KeyError(f"Missing weather time {weather_time}")
            weather_time_index_cache[weather_time] = time_idx
        rows_snapshot: list[list[object]] = []
        token_rows: list[list[str]] = []
        prev_ops: list[int] = []
        next_ops: list[int] = []

        # Row construction for each active journey key at this snapshot.
        for idx in active_indices:
            train_id = journeys_index["train_ids"][idx]
            service_date = journeys_index["service_dates"][idx]
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
            padded_ts, padded_op_ids, padded_planned, padded_delay, padded_type = cached
            split_idx = int(np.searchsorted(padded_ts, snap_ns, side="right"))
            past_slots = np.arange(split_idx - nb_past_events, split_idx, dtype=np.int64)[::-1]
            future_slots = np.arange(split_idx, split_idx + nb_future_events, dtype=np.int64)

            row: list[object] = [snap_ts, train_id, service_date]
            prev_op_id = int(padded_op_ids[past_slots[0]])
            next_op_id = int(padded_op_ids[future_slots[0]])
            prev_is_placeholder = prev_op_id == missing_event_placeholder
            next_is_placeholder = next_op_id == missing_event_placeholder
            if prev_is_placeholder and next_is_placeholder:
                raise ValueError(
                    f"Both stations are placeholders for train_id={train_id}, service_date={service_date}, ts={snap_ts}"
                )
            for slot_pos, slot_label in enumerate(range(1, nb_past_events + 1)):
                slot = past_slots[slot_pos]
                row.extend(
                    [
                        int(padded_op_ids[slot]),
                        int(padded_planned[slot]),
                        padded_delay[slot],
                        str(padded_type[slot]),
                    ]
                )
            for col_idx in range(1, nb_future_events + 1):
                slot = future_slots[col_idx - 1]
                row.extend(
                    [
                        int(padded_op_ids[slot]),
                        int(padded_planned[slot]),
                        padded_delay[slot],
                        str(padded_type[slot]),
                    ]
                )
            current_event_idx = int(split_idx - nb_past_events - 1)
            last_known_delay = float(padded_delay[past_slots[0]]) - (
                ((padded_planned[future_slots[0]] - snap_ns) / 1e9 + float(padded_delay[past_slots[0]])) < 0
            ) * (((padded_planned[future_slots[0]] - snap_ns) / 1e9) + float(padded_delay[past_slots[0]]))
            next_tokens = _extract_link_window_tokens(
                train_key=train_key,
                event_idx=current_event_idx,
                snap_ns=int(snap_ns),
                last_known_delay=last_known_delay,
                path_by_key=path_by_key,
                events_index=events_index,
                n_next_links=n_next_links,
                link_distance_m=link_distance_m,
                link_endpoints=link_endpoints,
            )
            for _ in range(n_next_links):
                row.extend([0.0, 0, np.nan, 1])
            for i, tok in enumerate(next_tokens):
                base = local_link_start_idx + link_slot_width * i
                row[base] = float(_token_distance(tok, link_distance_m=link_distance_m))
                row[base + 3] = int(tok == "")
            row.append(last_known_delay)
            row.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
            rows_snapshot.append(row)
            token_rows.append(next_tokens)
            prev_ops.append(prev_op_id)
            next_ops.append(next_op_id)

        # Local-link crowding features and weather interpolation for this snapshot.
        if rows_snapshot:
            lk_delay = np.asarray([float(r[tail_start_idx]) for r in rows_snapshot], dtype=np.float64)
            planned1 = np.asarray([int(r[future_planned_1_idx]) for r in rows_snapshot], dtype=np.int64)
            current_buckets: dict[str, list[int]] = {}
            for i, tokens in enumerate(token_rows):
                tok0 = str(tokens[0]) if len(tokens) > 0 else ""
                if tok0:
                    current_buckets.setdefault(tok0, []).append(i)
            for link_slot in range(n_next_links):
                base = local_link_start_idx + link_slot_width * link_slot
                link_vals = np.asarray([str(tokens[link_slot]) for tokens in token_rows], dtype=object)
                nb = np.zeros(len(rows_snapshot), dtype=np.int64)
                avg = np.full(len(rows_snapshot), np.nan, dtype=np.float64)
                if link_slot == 0:
                    for ids_list in current_buckets.values():
                        ids = np.asarray(ids_list, dtype=np.int64)
                        m = int(len(ids))
                        ord_ids = ids[np.argsort(planned1[ids], kind="mergesort")]
                        pos = np.arange(m, dtype=np.int64)
                        nb[ord_ids] = pos
                        if m > 1:
                            csum = np.cumsum(lk_delay[ord_ids], dtype=np.float64)
                            mask = pos > 0
                            avg[ord_ids[mask]] = csum[pos[mask] - 1] / pos[mask]
                else:
                    token_stats: dict[str, tuple[int, float]] = {}
                    for tok, ids_list in current_buckets.items():
                        ids = np.asarray(ids_list, dtype=np.int64)
                        token_stats[tok] = (int(len(ids)), float(np.nanmean(lk_delay[ids])))
                    for i, tok in enumerate(link_vals):
                        if not tok:
                            continue
                        stats = token_stats.get(str(tok))
                        if stats is None:
                            continue
                        nb[i] = stats[0]
                        avg[i] = stats[1]
                for i, row in enumerate(rows_snapshot):
                    row[base + 1] = int(nb[i])
                    row[base + 2] = float(avg[i]) if np.isfinite(avg[i]) else np.nan

            prev_ops_arr = np.asarray(prev_ops, dtype=np.int64)
            next_ops_arr = np.asarray(next_ops, dtype=np.int64)
            prev_placeholder = prev_ops_arr == missing_event_placeholder
            next_placeholder = next_ops_arr == missing_event_placeholder
            if np.any(prev_placeholder & next_placeholder):
                raise ValueError(f"Both stations are placeholders for snapshot {snap_ts}")
            prev_idx = weather_op_ids.get_indexer(prev_ops_arr)
            next_idx = weather_op_ids.get_indexer(next_ops_arr)
            if np.any((prev_idx < 0) & ~prev_placeholder):
                bad = prev_ops_arr[(prev_idx < 0) & ~prev_placeholder][0]
                raise KeyError(f"Missing weather for op_id={bad} at {weather_time}")
            if np.any((next_idx < 0) & ~next_placeholder):
                bad = next_ops_arr[(next_idx < 0) & ~next_placeholder][0]
                raise KeyError(f"Missing weather for op_id={bad} at {weather_time}")

            prev_temp = weather_temperature[time_idx, prev_idx]
            next_temp = weather_temperature[time_idx, next_idx]
            prev_rain = weather_rain[time_idx, prev_idx]
            next_rain = weather_rain[time_idx, next_idx]
            prev_snow = weather_snowfall[time_idx, prev_idx]
            next_snow = weather_snowfall[time_idx, next_idx]
            prev_rh = weather_rh[time_idx, prev_idx]
            next_rh = weather_rh[time_idx, next_idx]
            prev_wind = weather_wind[time_idx, prev_idx]
            next_wind = weather_wind[time_idx, next_idx]

            w_temp = np.where(prev_placeholder, next_temp, np.where(next_placeholder, prev_temp, (prev_temp + next_temp) / 2.0))
            w_rain = np.where(prev_placeholder, next_rain, np.where(next_placeholder, prev_rain, (prev_rain + next_rain) / 2.0))
            w_snow = np.where(prev_placeholder, next_snow, np.where(next_placeholder, prev_snow, (prev_snow + next_snow) / 2.0))
            w_rh = np.where(prev_placeholder, next_rh, np.where(next_placeholder, prev_rh, (prev_rh + next_rh) / 2.0))
            w_wind = np.where(prev_placeholder, next_wind, np.where(next_placeholder, prev_wind, (prev_wind + next_wind) / 2.0))
            for i, row in enumerate(rows_snapshot):
                row[weather_start_idx + 0] = float(w_temp[i])
                row[weather_start_idx + 1] = float(w_rain[i])
                row[weather_start_idx + 2] = float(w_snow[i])
                row[weather_start_idx + 3] = float(w_rh[i])
                row[weather_start_idx + 4] = float(w_wind[i])
            feature_rows.extend(tuple(r) for r in rows_snapshot)

    return feature_rows, columns


def build_ordered_feature_table(
    *,
    features: pd.DataFrame,
    nb_past_events: int,
    nb_future_events: int,
    station_embedding_dim: int,
) -> pd.DataFrame:
    """Order the feature table columns according to the exported tabular schema."""
    metric_order = {
        "distance": 0,
        "nb_of_trains": 1,
        "avg_delay": 2,
        "is_placeholder": 3,
    }

    def _link_col_sort_key(col: str) -> tuple[int, int, str]:
        """Sort link-context columns by slot and metric order."""
        m = re.match(r"^link(\d+)_(distance|nb_of_trains|avg_delay|is_placeholder)$", col)
        return (int(m.group(1)), metric_order[m.group(2)], col)

    link_feature_cols = sorted(
        [
            c
            for c in features.columns
            if c.startswith("link") and (
                c.endswith("_distance")
                or c.endswith("_nb_of_trains")
                or c.endswith("_avg_delay")
                or c.endswith("_is_placeholder")
            )
        ],
        key=_link_col_sort_key,
    )
    ordered_cols = [
        "snapshot_ts",
        "train_id",
        "service_date",
        "operator_is_sncb_nmbs",
        *link_feature_cols,
        "weather_temperature_2m",
        "weather_rain",
        "weather_snowfall",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ]
    ordered_cols += [f"snapshot_hour_sin_{freq}" for freq in [1, 2, 4]]
    ordered_cols += [f"snapshot_hour_cos_{freq}" for freq in [1, 2, 4]]
    ordered_cols += [f"snapshot_year_sin_{freq}" for freq in [1, 2, 4]]
    ordered_cols += [f"snapshot_year_cos_{freq}" for freq in [1, 2, 4]]
    ordered_cols += sorted([c for c in features.columns if c.startswith("train_relation_type_")])
    ordered_cols += sorted([c for c in features.columns if c.startswith("snapshot_day_of_week_")])
    ordered_cols += [f"past_planned_delta_{i}" for i in range(1, nb_past_events + 1)]
    ordered_cols += [f"past_delay_sec_{i}" for i in range(1, nb_past_events + 1)]
    ordered_cols += [f"future_planned_delta_{i}" for i in range(1, nb_future_events + 1)]
    ordered_cols += [f"future_delay_delta_{i}" for i in range(1, nb_future_events + 1)]
    for i in range(1, nb_past_events + 1):
        ordered_cols += [f"past_event_type_{i}_A", f"past_event_type_{i}_D", f"past_event_type_{i}_P"]
    for i in range(1, nb_future_events + 1):
        ordered_cols += [f"future_event_type_{i}_A", f"future_event_type_{i}_D", f"future_event_type_{i}_P"]
    for i in range(1, nb_past_events + 1):
        ordered_cols += [f"past_event_emb_{i}_{comp}" for comp in range(station_embedding_dim)]
    for i in range(1, nb_future_events + 1):
        ordered_cols += [f"future_event_emb_{i}_{comp}" for comp in range(station_embedding_dim)]

    missing_cols = [c for c in ordered_cols if c not in features.columns]
    if missing_cols:
        raise KeyError(f"Missing ordered ML columns: {missing_cols}")
    return features.loc[:, ordered_cols]


def _iter_snapshot_chunks(
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    *,
    chunk_size: int,
):
    """Yield snapshot and timestamp chunks for streaming feature construction."""
    n = len(snapshots)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield snapshots[start:end], sampled_ns[start:end]


def _update_running_stats(
    *,
    running: dict[str, dict[str, float | int | bool]],
    features: pd.DataFrame,
    train_mask: np.ndarray,
    signed_sqrt_cols: list[str],
    zscore_cols: list[str],
    minmax_cols: list[str],
) -> None:
    """Update running train-split normalization statistics from one feature chunk."""
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


def _assemble_numpy_parts(
    *,
    part_paths: list[Path],
    output_path: Path,
    dtype: np.dtype | type,
) -> None:
    """Merge temporary numpy chunk files into one output array."""
    if not part_paths:
        np.save(output_path, np.empty((0, 0), dtype=dtype))
        return
    if dtype == object:
        first = np.load(part_paths[0], allow_pickle=True)
    else:
        first = np.load(part_paths[0], allow_pickle=False, mmap_mode="r")
    ncols = first.shape[1] if first.ndim > 1 else 1
    total_rows = 0
    for p in part_paths:
        if dtype == object:
            arr = np.load(p, allow_pickle=True)
        else:
            arr = np.load(p, allow_pickle=False, mmap_mode="r")
        total_rows += int(arr.shape[0])

    if dtype == object:
        out = np.empty((total_rows, ncols), dtype=object)
        pos = 0
        for p in part_paths:
            arr = np.load(p, allow_pickle=True)
            n = arr.shape[0]
            out[pos : pos + n] = arr
            pos += n
        np.save(output_path, out)
        return

    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(total_rows, ncols))
    pos = 0
    for p in part_paths:
        arr = np.load(p, mmap_mode="r")
        n = arr.shape[0]
        out[pos : pos + n] = arr
        pos += n
    del out


def prepare_tabular_build_context(
    *,
    events_dir: Path,
    journeys_dir: Path,
    weather_path: Path,
    node_links_path: Path,
    op_nodes_path: Path,
    station_embedding_dim: int,
    missing_event_placeholder: int,
    dataset_core_spec: Path,
    show_progress: bool,
) -> dict[str, object]:
    """Load shared inputs and indexes needed for the tabular build."""
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

    # Build weather tensor slices only for needed ops/timestamps.
    event_op_ids = np.unique(index["events"]["event_op_ids"])
    weather_lookup = build_weather_lookup(
        weather_path=weather_path,
        snapshots=all_snapshots,
        event_op_ids=event_op_ids,
    )

    # Graph metadata used for oriented path-link features.
    node_links = pd.read_parquet(node_links_path, columns=["link_id", "u_node_id", "v_node_id", "distance_m"])
    link_distance_m = {
        int(r.link_id): float(r.distance_m)
        for r in node_links.itertuples(index=False)
    }
    link_endpoints = {
        int(r.link_id): (int(r.u_node_id), int(r.v_node_id))
        for r in node_links.itertuples(index=False)
    }

    # Station embeddings and journey-level metadata lookup table.
    station_embeddings, node_order, _, _, _ = create_station_embeddings_from_silver(
        node_links_path=node_links_path,
        op_nodes_path=op_nodes_path,
        embedding_dim=station_embedding_dim,
    )
    station_embedding_lookup = {int(op_id): station_embeddings[idx] for idx, op_id in enumerate(node_order)}

    journeys_index = index["journeys"]
    journeys_meta = pd.DataFrame(
        {
            "train_id": journeys_index["train_ids"].astype(str, copy=False),
            "service_date": journeys_index["service_dates"].astype(str, copy=False),
            "train_relation": journeys_index["train_relation"].astype(str, copy=False),
            "operator": journeys_index["operator"].astype(str, copy=False),
        }
    )
    return {
        "cfg": cfg,
        "train_snapshots": train_snapshots,
        "test_snapshots": test_snapshots,
        "all_snapshots": all_snapshots,
        "sampled_ns": sampled_ns,
        "index": index,
        "weather_lookup": weather_lookup,
        "link_distance_m": link_distance_m,
        "link_endpoints": link_endpoints,
        "station_embedding_lookup": station_embedding_lookup,
        "journeys_meta": journeys_meta,
    }


def build_snapshot_rows(
    *,
    index: dict,
    snapshots: pd.DatetimeIndex,
    sampled_ns: np.ndarray,
    nb_past_events: int,
    nb_future_events: int,
    idle_time_beg: int,
    idle_time_end: int,
    missing_event_placeholder: int,
    n_next_links: int,
    link_distance_m: dict[int, float],
    link_endpoints: dict[int, tuple[int, int]],
    weather_lookup: dict[str, object],
) -> pd.DataFrame:
    """Build raw snapshot rows before journey metadata and ML transforms."""
    # Expand active snapshots into raw per-train rows using only silver-index data.
    feature_rows, feature_columns = build_feature_rows(
        index=index,
        snapshots=snapshots,
        sampled_ns=sampled_ns,
        nb_past_events=nb_past_events,
        nb_future_events=nb_future_events,
        idle_time_beg=idle_time_beg,
        idle_time_end=idle_time_end,
        missing_event_placeholder=missing_event_placeholder,
        n_next_links=n_next_links,
        link_distance_m=link_distance_m,
        link_endpoints=link_endpoints,
        weather_lookup=weather_lookup,
        show_progress=False,
    )
    features = pd.DataFrame.from_records(feature_rows, columns=feature_columns)
    planned_cols = [c for c in features.columns if "planned_ts" in c]
    for col in planned_cols:
        vals = pd.to_numeric(features[col], errors="coerce")
        features[col] = pd.to_datetime(vals, unit="ns")
    return features


def add_context_features(
    *,
    features: pd.DataFrame,
    journeys_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Attach journey-level categorical metadata to raw feature rows."""
    return features.merge(journeys_meta, on=["train_id", "service_date"], how="left")


def to_ml_features(
    *,
    features: pd.DataFrame,
    station_embedding_lookup: dict[int, np.ndarray],
    station_embedding_dim: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Convert raw feature rows into pre-normalized ML feature columns."""
    return _add_ml_features_no_norm(
        features,
        station_embedding_lookup=station_embedding_lookup,
        station_embedding_dim=station_embedding_dim,
    )


def normalize_and_export(
    *,
    feature_chunk_paths: list[Path],
    tmp_dir: Path,
    output_dir: Path,
    n_chunks: int,
    show_progress: bool,
    train_snapshots: pd.DatetimeIndex,
    test_snapshots: pd.DatetimeIndex,
    nb_past_events: int,
    nb_future_events: int,
    station_embedding_dim: int,
    missing_event_placeholder: int,
    normalization_stats: dict[str, dict[str, float | bool]],
) -> None:
    """Apply normalization and export final train/test numpy arrays."""
    # Pass 2: apply learned normalization and materialize split numpy arrays.
    part_paths: dict[str, dict[str, list[Path]]] = {
        "train": {"x": [], "y": [], "y_mask": [], "md": []},
        "test": {"x": [], "y": [], "y_mask": [], "md": []},
    }
    part_idx = 0
    x_cols: list[str] | None = None
    y_cols: list[str] = [f"future_delay_delta_{i}" for i in range(1, nb_future_events + 1)]
    md_cols = ["snapshot_ts", "train_id", "service_date"]

    chunk_iter_2 = tqdm(
        feature_chunk_paths,
        total=n_chunks,
        desc="Pass 2/2 (build normalized arrays)",
        disable=not show_progress,
    )
    for chunk_path in chunk_iter_2:
        features = pd.read_parquet(chunk_path)
        features = _apply_normalization_stats(features=features, normalization_stats=normalization_stats)
        y_horizons = [int(c.split("_")[-1]) for c in y_cols]
        future_id_cols = [f"future_event_id_{h}" for h in y_horizons]
        y_mask_full = np.column_stack(
            [
                pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=np.float64)
                != float(missing_event_placeholder)
                for col in future_id_cols
            ]
        ).astype(bool, copy=False)
        features = build_ordered_feature_table(
            features=features,
            nb_past_events=nb_past_events,
            nb_future_events=nb_future_events,
            station_embedding_dim=station_embedding_dim,
        )

        if x_cols is None:
            x_cols = [c for c in features.columns if c not in set(md_cols + y_cols)]

        split_masks = {
            "train": pd.to_datetime(features["snapshot_ts"]).isin(train_snapshots).to_numpy(),
            "test": pd.to_datetime(features["snapshot_ts"]).isin(test_snapshots).to_numpy(),
        }

        for split, mask in split_masks.items():
            split_df = features.loc[mask]
            md = split_df[md_cols].to_numpy(dtype=object, copy=True)
            y = split_df[y_cols].to_numpy(dtype=np.float32, copy=True)
            y_mask = y_mask_full[mask].astype(bool, copy=True)
            x = split_df[x_cols].to_numpy(dtype=np.float32, copy=True)
            md_path = tmp_dir / f"{split}_md_part_{part_idx:05d}.npy"
            y_path = tmp_dir / f"{split}_y_part_{part_idx:05d}.npy"
            y_mask_path = tmp_dir / f"{split}_y_mask_part_{part_idx:05d}.npy"
            x_path = tmp_dir / f"{split}_x_part_{part_idx:05d}.npy"
            np.save(md_path, md)
            np.save(y_path, y)
            np.save(y_mask_path, y_mask)
            np.save(x_path, x)
            part_paths[split]["md"].append(md_path)
            part_paths[split]["y"].append(y_path)
            part_paths[split]["y_mask"].append(y_mask_path)
            part_paths[split]["x"].append(x_path)
        part_idx += 1
        del features
        gc.collect()

    # Merge chunk parts into final train/test arrays.
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        _assemble_numpy_parts(
            part_paths=part_paths[split]["md"],
            output_path=split_dir / "md.npy",
            dtype=object,
        )
        _assemble_numpy_parts(
            part_paths=part_paths[split]["y"],
            output_path=split_dir / "y.npy",
            dtype=np.float32,
        )
        _assemble_numpy_parts(
            part_paths=part_paths[split]["y_mask"],
            output_path=split_dir / "y_mask.npy",
            dtype=np.bool_,
        )
        _assemble_numpy_parts(
            part_paths=part_paths[split]["x"],
            output_path=split_dir / "x.npy",
            dtype=np.float32,
        )

    # Persist schema + normalization contract.
    assert x_cols is not None
    write_yaml(
        output_dir / "scheme.yaml",
        {
            "md_columns": md_cols,
            "y_columns": y_cols,
            "y_mask_columns": [f"is_target_valid_{i}" for i in range(1, nb_future_events + 1)],
            "x_columns": x_cols,
        },
    )
    write_yaml(output_dir / "normalization.yaml", normalization_stats)

    for split in ("train", "test"):
        for kind in ("md", "y", "y_mask", "x"):
            for p in part_paths[split][kind]:
                p.unlink(missing_ok=True)


def build_and_export_streaming(
    *,
    input_dir: Path,
    journeys_dir: Path,
    weather_path: Path,
    node_links_path: Path,
    op_nodes_path: Path,
    nb_past_events: int,
    n_next_links: int,
    station_embedding_dim: int,
    missing_event_placeholder: int,
    dataset_core_spec: Path,
    output_dir: Path,
    show_progress: bool = True,
    chunk_size_snapshots: int = 1000,
) -> None:
    """Build and export the tabular dataset with two-pass streaming."""
    # Stage 1: load and pre-index all silver-derived inputs.
    stage = prepare_tabular_build_context(
        events_dir=input_dir,
        journeys_dir=journeys_dir,
        weather_path=weather_path,
        node_links_path=node_links_path,
        op_nodes_path=op_nodes_path,
        station_embedding_dim=station_embedding_dim,
        missing_event_placeholder=missing_event_placeholder,
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
    link_endpoints = stage["link_endpoints"]
    station_embedding_lookup = stage["station_embedding_lookup"]
    journeys_meta = stage["journeys_meta"]

    tmp_dir = output_dir / "_tmp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    feature_chunk_paths: list[Path] = []

    # Stage 2: pass 1/2 - build raw ML chunks and collect normalization stats.
    running_stats: dict[str, dict[str, float | int | bool]] = {}
    n_chunks = (len(all_snapshots) + chunk_size_snapshots - 1) // chunk_size_snapshots
    chunk_iter_1 = tqdm(
        _iter_snapshot_chunks(
            all_snapshots,
            sampled_ns,
            chunk_size=chunk_size_snapshots,
        ),
        total=n_chunks,
        desc="Pass 1/2 (build features)",
        disable=not show_progress,
    )
    for chunk_id, (chunk_snapshots, chunk_ns) in enumerate(chunk_iter_1):
        features = build_snapshot_rows(
            index=index,
            snapshots=chunk_snapshots,
            sampled_ns=chunk_ns,
            nb_past_events=nb_past_events,
            nb_future_events=int(cfg["n_future"]),
            idle_time_beg=int(cfg["idle_time_beg"]),
            idle_time_end=int(cfg["idle_time_end"]),
            missing_event_placeholder=missing_event_placeholder,
            n_next_links=n_next_links,
            link_distance_m=link_distance_m,
            link_endpoints=link_endpoints,
            weather_lookup=weather_lookup,
        )
        features = add_context_features(features=features, journeys_meta=journeys_meta)
        features, signed_sqrt_cols, zscore_cols, minmax_cols = to_ml_features(
            features=features,
            station_embedding_lookup=station_embedding_lookup,
            station_embedding_dim=station_embedding_dim,
        )
        train_mask = pd.to_datetime(features["snapshot_ts"]).isin(train_snapshots).to_numpy()
        _update_running_stats(
            running=running_stats,
            features=features,
            train_mask=train_mask,
            signed_sqrt_cols=signed_sqrt_cols,
            zscore_cols=zscore_cols,
            minmax_cols=minmax_cols,
        )
        chunk_path = tmp_dir / f"features_part_{chunk_id:05d}.parquet"
        features.to_parquet(chunk_path, index=False)
        feature_chunk_paths.append(chunk_path)
        del features
        gc.collect()

    # Stage 3: finalize normalization params from train snapshots.
    normalization_stats = _finalize_running_stats(running_stats)

    # Stage 4: pass 2/2 - apply normalization and export final arrays.
    normalize_and_export(
        feature_chunk_paths=feature_chunk_paths,
        tmp_dir=tmp_dir,
        output_dir=output_dir,
        n_chunks=n_chunks,
        show_progress=show_progress,
        train_snapshots=train_snapshots,
        test_snapshots=test_snapshots,
        nb_past_events=nb_past_events,
        nb_future_events=int(cfg["n_future"]),
        station_embedding_dim=station_embedding_dim,
        missing_event_placeholder=missing_event_placeholder,
        normalization_stats=normalization_stats,
    )

    # Stage 5: cleanup temporary chunk files.
    for p in feature_chunk_paths:
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()


def build_tabular_dataset(
    *,
    silver_dir: Path,
    nb_past_events: int,
    n_next_links: int,
    station_embedding_dim: int,
    output_dir: Path,
    dataset_core_spec: Path,
    missing_event_placeholder: int = -1,
    show_progress: bool = True,
) -> None:
    """Build the Gold tabular dataset from a Silver dataset and Gold core.

    Args:
        silver_dir: Root directory of the Silver dataset.
        nb_past_events: Number of past event slots encoded per active train.
        n_next_links: Number of current/future link-context slots to encode.
        station_embedding_dim: Number of station embedding dimensions to add.
        output_dir: Directory where train/test arrays, schema, and
            normalization metadata are written.
        dataset_core_spec: Path to the Gold core specification YAML.
        missing_event_placeholder: Operational point id used for padded event
            slots.
        show_progress: Whether to display progress bars.

    Returns:
        None. The function writes the tabular dataset under `output_dir`.
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
        nb_past_events=nb_past_events,
        n_next_links=n_next_links,
        station_embedding_dim=station_embedding_dim,
        missing_event_placeholder=missing_event_placeholder,
        dataset_core_spec=dataset_core_spec,
        output_dir=output_dir,
        show_progress=show_progress,
    )
    print(f"[gold_tabular_snapshot] Wrote ML arrays/scheme to {output_dir}")
