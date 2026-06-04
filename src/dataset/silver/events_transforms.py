"""Silver event and journey transform utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import heapq
import math

import numpy as np
import pandas as pd


DEFAULT_INTRA_STOP_CONFLICT_THRESHOLD_SEC = 120
DEFAULT_PLANNED_MONOTONIC_THRESHOLD_SEC = 120
DEFAULT_OBSERVED_MONOTONIC_THRESHOLD_SEC = 120
DEFAULT_MAX_LATE_SEC = 4 * 3600
DEFAULT_MAX_EARLY_SEC = 3600


def _build_timestamp(series_date: pd.Series, series_time: pd.Series) -> pd.Series:
    """Combine Bronze date and time string columns into timestamps."""
    return pd.to_datetime(
        series_date.fillna("").astype(str) + " " + series_time.fillna("").astype(str),
        format="%d%b%Y %H:%M:%S",
        errors="coerce",
    )

def merge_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge raw schedule and observation columns into planned/observed timestamps."""
    df["planned_arr_ts"] = _build_timestamp(df["sched_arrival_date"], df["sched_arrival_time"])
    df["planned_dep_ts"] = _build_timestamp(df["sched_departure_date"], df["sched_departure_time"])
    df["obs_arr_ts"] = _build_timestamp(df["observed_arrival_date"], df["observed_arrival_time"])
    df["obs_dep_ts"] = _build_timestamp(df["observed_departure_date"], df["observed_departure_time"])

    df = df.drop(
        columns=[
            "sched_arrival_date",
            "sched_arrival_time",
            "sched_departure_date",
            "sched_departure_time",
            "observed_arrival_date",
            "observed_arrival_time",
            "observed_departure_date",
            "observed_departure_time",
        ]
    )

    return df


def clean_intra_stop_schedule_conflicts(events: pd.DataFrame, threshold_sec: int) -> tuple[pd.DataFrame, int, int]:
    """Resolve same-stop arrival/departure conflicts or drop affected journeys."""
    df = events.copy()

    issue_mask = (
        df["planned_dep_ts"].notna()
        & df["planned_arr_ts"].notna()
        & (df["planned_dep_ts"] < df["planned_arr_ts"])
    )

    diff_planned = (df["planned_arr_ts"] - df["planned_dep_ts"]).dt.total_seconds()
    diff_observed = (df["obs_arr_ts"] - df["obs_dep_ts"]).dt.total_seconds()

    drop_mask = issue_mask & (
        (diff_planned.fillna(0) > threshold_sec)
        | (diff_observed.notna() & (diff_observed > threshold_sec))
    )

    drop_keys = df.loc[drop_mask, ["train_id", "service_date"]]
    drop_count = int(drop_keys.drop_duplicates().shape[0])

    if drop_count:
        journey_index = pd.MultiIndex.from_frame(df[["train_id", "service_date"]])
        drop_index = pd.MultiIndex.from_frame(drop_keys.drop_duplicates())
        keep_mask = ~journey_index.isin(drop_index)
        df = df[keep_mask].copy()

    adjust_mask = issue_mask & ~drop_mask
    if adjust_mask.any():
        df.loc[adjust_mask, "planned_arr_ts"] = df.loc[adjust_mask, "planned_dep_ts"]
        df.loc[adjust_mask, "obs_arr_ts"] = df.loc[adjust_mask, "obs_dep_ts"]

    return df, drop_count, int(adjust_mask.sum())

def separate_event_types(events: pd.DataFrame) -> pd.DataFrame:
    """Explode stops into event rows while retaining original sequence order.

    If the planned arrival and departure timestamps are equal, collapse into
    a single passage event.
    """
    df = events.copy()
    df["_row_idx"] = np.arange(len(df), dtype=np.int64)

    passthrough = [
        col for col in ["train_relation", "operator", "relation_direction"]
        if col in df.columns
    ]
    line_cols = [col for col in ["arr_line_id", "dep_line_id"] if col in df.columns]
    base_cols = ["train_id", "service_date", "op_id", "op_name", *line_cols, *passthrough]

    frames = []

    passing_mask = (
        df["planned_arr_ts"].notna()
        & df["planned_dep_ts"].notna()
        & (df["planned_arr_ts"] == df["planned_dep_ts"])
    )
    if passing_mask.any():
        passings = df.loc[passing_mask].copy()
        passings["event_type"] = "P"
        passings["planned_ts"] = passings["planned_arr_ts"]
        passings["observed_ts"] = passings["obs_arr_ts"]
        passings["_order"] = passings["_row_idx"] * 3 + 0
        frames.append(passings)

    arrival_mask = df["planned_arr_ts"].notna() & ~passing_mask
    if arrival_mask.any():
        arrivals = df.loc[arrival_mask].copy()
        arrivals["event_type"] = "A"
        arrivals["planned_ts"] = arrivals["planned_arr_ts"]
        arrivals["observed_ts"] = arrivals["obs_arr_ts"]
        arrivals["dep_line_id"] = pd.NA
        arrivals["_order"] = arrivals["_row_idx"] * 3 + 1
        frames.append(arrivals)

    departure_mask = df["planned_dep_ts"].notna() & ~passing_mask
    if departure_mask.any():
        departures = df.loc[departure_mask].copy()
        departures["event_type"] = "D"
        departures["planned_ts"] = departures["planned_dep_ts"]
        departures["observed_ts"] = departures["obs_dep_ts"]
        departures["arr_line_id"] = pd.NA
        departures["_order"] = departures["_row_idx"] * 3 + 2
        frames.append(departures)

    if not frames:
        return pd.DataFrame(columns=[*base_cols, "event_type", "planned_ts", "observed_ts"])

    result = (
        pd.concat(
            [frame[base_cols + ["event_type", "planned_ts", "observed_ts", "_order"]] for frame in frames],
            ignore_index=True,
        )
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return result

def _enforce_monotonicity(
    events: pd.DataFrame,
    column: str,
    threshold_sec: int,
) -> tuple[pd.DataFrame, int, int]:
    """Clamp backward steps in-place for the target timestamp column.

    The function assumes the source-provided row order is the correct spatial
    stop order for each journey. It never resorts rows; instead it forward-fills
    backward timestamp steps within the tolerated window and drops whole journeys
    if a backward step exceeds the threshold.
    """
    df = events.copy()

    prev_col = f"_prev_{column}"
    delta_col = f"_delta_{column}"

    df[prev_col] = df.groupby(["train_id", "service_date"])[column].shift()
    df[delta_col] = (df[column] - df[prev_col]).dt.total_seconds()

    threshold_sec = max(int(threshold_sec), 0)

    # Track cumulative backward drift per streak of negative deltas. Any run of
    # regressions that pushes past the threshold causes the journey to be dropped.
    neg_deltas = df[delta_col].where(df[delta_col] < 0, 0).fillna(0)
    reset_groups = (df[delta_col] >= 0).cumsum()
    cumulative_neg = neg_deltas.groupby(
        [df["train_id"], df["service_date"], reset_groups]
    ).cumsum()
    drop_condition = cumulative_neg < -threshold_sec

    drop_mask = drop_condition.groupby([df["train_id"], df["service_date"]]).transform("any")
    drop_count = int(
        df.loc[drop_mask, ["train_id", "service_date"]].drop_duplicates().shape[0]
    )

    df_kept = df[~drop_mask].copy()
    adjustments = int((df_kept[delta_col] < 0).sum())

    negative_mask = df_kept[delta_col] < 0
    if negative_mask.any():
        df_kept.loc[negative_mask, column] = df_kept.loc[negative_mask, prev_col]

    df_kept[column] = (
        df_kept.groupby(["train_id", "service_date"])[column]
        .transform("cummax")
    )

    df_kept = df_kept.drop(columns=[prev_col, delta_col])
    return df_kept, drop_count, adjustments


def enforce_planned_monotonicity(events: pd.DataFrame, threshold_sec: int) -> tuple[pd.DataFrame, int, int]:
    """Clamp or drop planned timestamps that move backward within each journey."""
    return _enforce_monotonicity(events, "planned_ts", threshold_sec)

def enforce_observed_monotonicity(events: pd.DataFrame, threshold_sec: int) -> tuple[pd.DataFrame, int, int]:
    """Clamp or drop observed timestamps that move backward within each journey."""
    return _enforce_monotonicity(events, "observed_ts", threshold_sec)


def add_delay_seconds(events: pd.DataFrame) -> pd.DataFrame:
    """Add observed-minus-planned delay in seconds to each event row."""
    df = events.copy()
    df["delay_sec"] = (df["observed_ts"] - df["planned_ts"]).dt.total_seconds()
    return df


def filter_journeys_by_delay(events: pd.DataFrame, max_late_sec: int, max_early_sec: int) -> pd.DataFrame:
    """Drop whole journeys whose event delays exceed configured bounds."""
    group_keys = ["train_id", "service_date"]
    max_delay = events.groupby(group_keys)["delay_sec"].transform("max")
    min_delay = events.groupby(group_keys)["delay_sec"].transform("min")
    mask = (max_delay <= max_late_sec) & (min_delay >= -max_early_sec)
    return events.loc[mask].copy()


def drop_rows_without_timestamps(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove event rows without planned or observed timestamps and report affected journeys."""
    mask_has_ts = events["planned_ts"].notna() | events["observed_ts"].notna()
    dropped = (
        events.loc[~mask_has_ts, ["train_id", "service_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return events.loc[mask_has_ts].copy(), dropped


def drop_empty_journeys(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove journeys that contain no remaining events."""
    counts = events.groupby(["train_id", "service_date"])["op_id"].transform("size")
    mask = counts > 0
    dropped = (
        events.loc[~mask, ["train_id", "service_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return events.loc[mask].copy(), dropped


def deduplicate_events(events: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Keep the first event per journey, stop, and event type."""
    key_cols = ["train_id", "service_date", "event_type", "op_id"]
    before = len(events)
    result = events.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return result, before - len(result)


def _resolve_params(params: Optional[Mapping[str, Any]]) -> Tuple[int, int, int, int, int]:
    """Resolve Silver event-cleaning parameters and backward-compatible aliases."""
    params = dict(params or {})
    intra_stop_conflict_threshold = int(
        params.get(
            "intra_stop_conflict_threshold_sec",
            params.get(
                "stop_arrival_departure_conflict_threshold_sec",
                params.get(
                    "schedule_skew_threshold_sec",
                    params.get("skew_threshold_sec", DEFAULT_INTRA_STOP_CONFLICT_THRESHOLD_SEC),
                ),
            ),
        )
    )
    apply_intra_stop_conflicts = _to_bool(params.get("apply_intra_stop_conflicts"), default=True)
    planned_monotonic_threshold = int(
        params.get(
            "planned_monotonic_threshold_sec",
            params.get(
                "schedule_monotonic_threshold_sec",
                params.get("monotonic_threshold_sec", DEFAULT_PLANNED_MONOTONIC_THRESHOLD_SEC),
            ),
        )
    )
    apply_planned_monotonic = _to_bool(params.get("apply_planned_monotonic"), default=True)
    observed_monotonic_threshold = int(
        params.get(
            "observed_monotonic_threshold_sec",
            params.get("monotonic_threshold_sec", DEFAULT_OBSERVED_MONOTONIC_THRESHOLD_SEC),
        )
    )
    apply_observed_monotonic = _to_bool(params.get("apply_observed_monotonic"), default=True)
    max_late_sec = int(params.get("max_late_sec", DEFAULT_MAX_LATE_SEC))
    max_early_sec = int(params.get("max_early_sec", DEFAULT_MAX_EARLY_SEC))
    return (
        intra_stop_conflict_threshold,
        apply_intra_stop_conflicts,
        planned_monotonic_threshold,
        apply_planned_monotonic,
        observed_monotonic_threshold,
        apply_observed_monotonic,
        max_late_sec,
        max_early_sec,
    )


def _to_bool(value: object, default: bool = False) -> bool:
    """Parse common manifest boolean representations."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_events(
    *,
    sources: Mapping[str, Any],
    wildcards: Optional[Dict[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Build cleaned Silver event rows while preserving source stop order.

    The source event rows are assumed to follow the correct spatial station
    sequence for each journey, so the transform splits and adjusts rows without
    sorting by timestamp.
    """
    params = dict(params or {})
    verbose = _to_bool(params.get("verbose"), default=False)
    (
        intra_stop_conflict_threshold,
        apply_intra_stop_conflicts,
        planned_monotonic_threshold,
        apply_planned_monotonic,
        observed_monotonic_threshold,
        apply_observed_monotonic,
        max_late_sec,
        max_early_sec,
    ) = _resolve_params(params)

    raw_source = sources.get("main")
    if not isinstance(raw_source, pd.DataFrame):
        raise TypeError("Expected 'main' source to be a pandas DataFrame.")
    bronze_events = raw_source.copy()

    # Prepare event-level rows from the source stop records while preserving
    # the source-provided spatial order within each journey.
    events = merge_timestamp_columns(bronze_events)

    intra_drop_count = 0
    intra_adjusted_pairs = 0
    planned_drop_count = 0
    planned_adjusted = 0
    observed_drop_count = 0
    observed_adjusted = 0

    if apply_intra_stop_conflicts:
        events, intra_drop_count, intra_adjusted_pairs = clean_intra_stop_schedule_conflicts(
            events, intra_stop_conflict_threshold
        )

    events = separate_event_types(events)

    # Remove journeys that cannot support valid event timelines, then enforce
    # monotonic planned and observed timestamps without sorting the rows.
    missing_observed = events["observed_ts"].isna()
    missing_planned = events["planned_ts"].isna()
    drop_mask = missing_observed | missing_planned
    if drop_mask.any():
        bad_keys = events.loc[drop_mask, ["train_id", "service_date"]].drop_duplicates()
        bad_index = pd.MultiIndex.from_frame(bad_keys)
        journey_index = pd.MultiIndex.from_frame(events[["train_id", "service_date"]])
        keep_mask = ~journey_index.isin(bad_index)
        if verbose:
            print(f"[events] dropped {bad_keys.shape[0]} journeys (missing planned/observed timestamps after split)")
        events = events[keep_mask].copy()

    if apply_planned_monotonic:
        events, planned_drop_count, planned_adjusted = enforce_planned_monotonicity(
            events, planned_monotonic_threshold
        )

    if apply_observed_monotonic:
        events, observed_drop_count, observed_adjusted = enforce_observed_monotonicity(
            events, observed_monotonic_threshold
        )

    if verbose:
        if apply_intra_stop_conflicts and intra_drop_count:
            print(f"[events] dropped {intra_drop_count} journeys due to intra-stop conflicts")
        if apply_intra_stop_conflicts and intra_adjusted_pairs:
            print(f"[events] adjusted {intra_adjusted_pairs} intra-stop arrival/departure pairs")
        if apply_planned_monotonic and planned_drop_count:
            print(
                "[events] dropped "
                f"{planned_drop_count} journeys after planned monotonicity check "
                f"(threshold={planned_monotonic_threshold}s)"
            )
        if apply_planned_monotonic and planned_adjusted:
            print(f"[events] adjusted {planned_adjusted} planned timestamps to enforce monotonicity")
        if apply_observed_monotonic and observed_drop_count:
            print(
                "[events] dropped "
                f"{observed_drop_count} journeys after observed monotonicity check "
                f"(threshold={observed_monotonic_threshold}s)"
            )
        if apply_observed_monotonic and observed_adjusted:
            print(f"[events] adjusted {observed_adjusted} observed timestamps to enforce monotonicity")

    # Finalize cleaned event rows by adding delays and removing invalid or
    # duplicate rows.
    events = add_delay_seconds(events)
    journeys_before_delay = events[["train_id", "service_date"]].drop_duplicates()
    events_filtered = filter_journeys_by_delay(events, max_late_sec, max_early_sec)

    if verbose:
        journeys_after_delay = events_filtered[["train_id", "service_date"]].drop_duplicates()
        removed_delay = len(journeys_before_delay) - len(journeys_after_delay)
        if removed_delay > 0:
            print(f"[events] dropped {removed_delay} journeys (delay bounds)")

    events_filtered, dropped_ts = drop_rows_without_timestamps(events_filtered)
    if verbose and len(dropped_ts):
        print(f"[events] dropped {len(dropped_ts)} journeys (missing timestamps)")

    events_filtered, dropped_empty = drop_empty_journeys(events_filtered)
    if verbose and len(dropped_empty):
        print(f"[events] dropped {len(dropped_empty)} journeys (empty after cleaning)")

    events_filtered, drop_duplicates_count = deduplicate_events(events_filtered)
    if verbose and drop_duplicates_count:
        print(f"[events] removed {drop_duplicates_count} duplicate event rows")

    cleaned = events_filtered.reset_index(drop=True)
    cleaned["_event_row_order"] = np.arange(len(cleaned), dtype=np.int64)
    cleaned = (
        cleaned.sort_values(["train_id", "service_date", "_event_row_order"], kind="mergesort")
        .drop(columns="_event_row_order")
        .reset_index(drop=True)
    )

    columns = [
        "train_id",
        "service_date",
        "op_id",
        "event_type",
        "arr_line_id",
        "dep_line_id",
        "planned_ts",
        "observed_ts",
        "delay_sec",
    ]

    if verbose:
        unique_journeys = cleaned[["train_id", "service_date"]].drop_duplicates()
        print(
            f"[events] output rows: {len(cleaned):,} "
            f"({len(unique_journeys):,} journeys)"
        )

    return cleaned[columns].copy()


def _build_graph(
    line_sections: pd.DataFrame,
    node_links: pd.DataFrame,
) -> Mapping[int, list[tuple[int, str | None, float, int]]]:
    """Build an undirected node-link graph annotated with railway line IDs."""
    section_to_line = (
        line_sections[["line_section_id", "line_id"]]
        .dropna()
        .astype(str)
        .set_index("line_section_id")["line_id"]
        .to_dict()
    )

    full_graph = {}

    for row in node_links.itertuples(index=False):
        link_id = int(row.link_id)
        src = int(row.u_node_id)
        dst = int(row.v_node_id)
        length = float(getattr(row, "distance_m"))
        line_id = section_to_line.get(str(getattr(row, "line_section_id", None)))

        full_graph.setdefault(src, []).append((dst, line_id, length, link_id))
        full_graph.setdefault(dst, []).append((src, line_id, length, link_id))

    return full_graph


def _dijkstra_custom(
    graph: Mapping[int, Iterable[tuple[int, str | None, float, int]]],
    src: int,
    dst: int,
    dep_line: str | None,
    arr_line: str | None,
) -> list[int] | None:
    """
    Standard Dijkstra, but we bias edge weights so the path starts on
    `dep_line`, ends on `arr_line`, and mildly prefers those lines along the
    way. Biasing is purely multiplicative on the physical edge length:

    - Edges touching the source favour dep_line first, then arr_line.
    - Edges touching the destination favour arr_line first, then dep_line.
    - All other edges get a small bonus if their line_id matches either input.
    """

    def edge_bias(node: int, tgt: int, line_id: str | None) -> float:
        """Return the line-preference multiplier for one graph edge."""
        if not line_id:
            return 1.0

        def pick(order: list[str | None]) -> float:
            """Choose the first matching preferred line multiplier."""
            for idx, candidate in enumerate(order):
                if candidate and line_id == candidate:
                    return 0.55 if idx == 0 else 0.75
            return 1.0

        if node == src or tgt == src:
            return pick([dep_line, arr_line])
        if node == dst or tgt == dst:
            return pick([arr_line, dep_line])

        preferred = []
        if dep_line:
            preferred.append(dep_line)
        if arr_line and arr_line != dep_line:
            preferred.append(arr_line)

        if preferred and line_id in preferred:
            return 0.65
        return 1.0

    heap = [(0.0, src, None, None)]
    best = {src: 0.0}
    parent = {}

    while heap:
        cost, node, prev, link_id = heapq.heappop(heap)
        if node in parent:
            continue
        if prev is not None:
            parent[node] = (prev, link_id)
        if node == dst:
            break

        for tgt, line_id, length, edge_link in graph.get(node, ()):
            bias = edge_bias(node, tgt, line_id)
            new_cost = cost + bias * length
            if new_cost < best.get(tgt, float("inf")):
                best[tgt] = new_cost
                heapq.heappush(heap, (new_cost, tgt, node, edge_link))

    if dst not in parent:
        return None

    path = []
    node = dst
    while node != src:
        prev, link_id = parent[node]
        if link_id is None:
            return None
        path.append(link_id)
        node = prev
    path.reverse()
    return path


def resolve_pair(
    src: int,
    dst: int,
    dep_line: str | None,
    arr_line: str | None,
    full_graph: Mapping[int, list[tuple[int, str | None, float, int]]],
) -> list[int]:
    """Resolve the node-link path between two operational points."""
    if src == dst:
        return []

    path = _dijkstra_custom(full_graph, src, dst, dep_line, arr_line)
    if path is None:
        raise ValueError(f"Unable to resolve path between {src} and {dst}")
    return path


def annotate_journey_paths(
    events: pd.DataFrame,
    line_sections: pd.DataFrame,
    node_links: pd.DataFrame,
) -> pd.DataFrame:
    """Attach deduced node-link paths between consecutive events for each journey."""
    full_graph = _build_graph(line_sections, node_links)
    cache_path = Path("temp/cache.pkl")
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    def resolve(src: int, dst: int, dep_line: str | None, arr_line: str | None) -> list[int]:
        """Resolve one segment path while reusing the on-disk path cache."""
        key = (src, dst, dep_line or None, arr_line or None)
        if key not in cache:
            cache[key] = resolve_pair(src, dst, dep_line, arr_line, full_graph)
        return cache[key]

    rows = []
    for (train_id, service_date), group in events.groupby(["train_id", "service_date"], sort=False):
        group = group.reset_index(drop=True)
        segments = []
        for idx in range(len(group) - 1):
            a = group.loc[idx]
            b = group.loc[idx + 1]
            if int(a["op_id"]) == int(b["op_id"]):
                segments.append([])
            else:
                segments.append(
                    resolve(
                        int(a["op_id"]),
                        int(b["op_id"]),
                        a.get("dep_line_id"),
                        b.get("arr_line_id"),
                    )
                )

        rows.append(
            {
                "train_id": train_id,
                "service_date": service_date,
                "deduced_paths": segments,
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return pd.DataFrame(rows)


def build_journeys(
    *,
    sources: Mapping[str, Any],
    wildcards: Optional[Dict[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Build Silver journey summaries from cleaned event rows."""
    bronze_events = sources["bronze_events"]
    silver_events = sources["silver_events"]
    node_links = sources["node_links"]
    line_sections = sources["line_sections"]

    key_cols = ["train_id", "service_date"]
    valid_keys = silver_events[key_cols].drop_duplicates()

    # Keep only Bronze journeys that survived event cleaning so journey-level
    # metadata stays aligned with the Silver event table.
    filtered_bronze = bronze_events.merge(valid_keys, on=key_cols, how="inner")

    # Resolve network paths between consecutive Silver events.
    events_for_paths = silver_events[
        key_cols + ["op_id", "arr_line_id", "dep_line_id", "observed_ts", "planned_ts"]
    ].copy()
    events_for_paths = events_for_paths.sort_values(
        key_cols + ["observed_ts", "planned_ts"],
        kind="mergesort",
    ).reset_index(drop=True)

    journey_paths = annotate_journey_paths(
        events_for_paths,
        line_sections,
        node_links,
    )

    # Aggregate journey-level descriptors and attach the deduced path sequence.
    journeys = (
        silver_events.merge(
            filtered_bronze[
                key_cols
                + ["train_relation", "operator", "relation_direction"]
            ].drop_duplicates(key_cols),
            on=key_cols,
            how="left",
        )
        .groupby(key_cols)
        .agg(
            train_relation=("train_relation", "first"),
            operator=("operator", "first"),
            relation_direction=("relation_direction", "first"),
            start_op_id=("op_id", "first"),
            end_op_id=("op_id", "last"),
            start_planned_ts=("planned_ts", "first"),
            end_planned_ts=("planned_ts", "last"),
            start_observed_ts=("observed_ts", "first"),
            end_observed_ts=("observed_ts", "last"),
            events_count=("event_type", "size"),
            max_delay_sec=("delay_sec", "max"),
            min_delay_sec=("delay_sec", "min"),
        )
        .reset_index()
    )

    journeys = (
        journeys.merge(journey_paths, on=key_cols, how="left")
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    journeys["events_count"] = journeys["events_count"].astype("Int64")
    return journeys
