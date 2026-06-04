"""Shared helpers for Gold core and model-specific dataset builders."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from tqdm import tqdm


REQUIRED_SEQUENCE_COLUMNS: Final[tuple[str, ...]] = (
    "train_id",
    "service_date",
    "event_ts",
    "op_id",
    "planned_ts",
    "delay_sec",
    "event_type",
)
TAIL_SAFETY_BUFFER_MIN: Final[int] = 1
REQUIRED_ACTIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "train_id",
    "service_date",
    "start_planned_ts",
    "start_observed_ts",
    "end_observed_ts",
)

def _to_ns_int64(values: pd.Series | np.ndarray | list) -> np.ndarray:
    """Convert datetime-like values to int64 nanoseconds regardless of source unit."""
    dt = pd.to_datetime(values, errors="coerce")
    return dt.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def compute_journey_activity_windows(
    journeys: pd.DataFrame,
    idle_time_beg: int,
    idle_time_end: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute snapshot activity-window bounds for Silver journeys.

    A journey is considered active from:
    - `min(start_planned_ts - idle_time_beg, start_observed_ts)`
    until:
    - `end_observed_ts + idle_time_end`

    Args:
        journeys: Silver journey table with planned/observed boundary
            timestamps.
        idle_time_beg: Minutes before the planned start at which a journey may
            become active.
        idle_time_end: Minutes after the observed end at which a journey stops
            being active.

    Returns:
        Pair of `appearance_start` and `disappearance_end` timestamp series
        aligned to the input journey index.
    """
    missing = [col for col in REQUIRED_ACTIVITY_COLUMNS if col not in journeys.columns]
    if missing:
        raise KeyError(f"Missing required journey column(s): {missing}")

    start_planned = pd.to_datetime(journeys["start_planned_ts"], errors="coerce")
    start_observed = pd.to_datetime(journeys["start_observed_ts"], errors="coerce")
    end_observed = pd.to_datetime(journeys["end_observed_ts"], errors="coerce")

    invalid_mask = start_planned.isna() | start_observed.isna() | end_observed.isna()
    if bool(invalid_mask.any()):
        raise ValueError(
            "Journey activity windows require valid datetimes in "
            "`start_planned_ts`, `start_observed_ts`, and `end_observed_ts`."
        )

    appearance_start = np.minimum(
        (start_planned - pd.Timedelta(minutes=int(idle_time_beg))).to_numpy(dtype="datetime64[ns]"),
        start_observed.to_numpy(dtype="datetime64[ns]"),
    )
    disappearance_end = (
        end_observed + pd.Timedelta(minutes=int(idle_time_end))
    ).to_numpy(dtype="datetime64[ns]")

    return (
        pd.Series(pd.to_datetime(appearance_start), index=journeys.index, name="appearance_start"),
        pd.Series(pd.to_datetime(disappearance_end), index=journeys.index, name="disappearance_end"),
    )


def get_active_mask_for_snapshot(
    appearance_start: pd.Series | np.ndarray,
    disappearance_end: pd.Series | np.ndarray,
    snapshot_ts: object,
) -> np.ndarray:
    """Return the journeys active at one snapshot timestamp.

    Args:
        appearance_start: Per-journey inclusive activity-window start times.
        disappearance_end: Per-journey exclusive activity-window end times.
        snapshot_ts: Snapshot timestamp to test against the activity windows.

    Returns:
        Boolean array whose true entries identify journeys active at
        `snapshot_ts`.
    """
    return (
        appearance_start <= snapshot_ts
    ) & (
        disappearance_end > snapshot_ts
    )


def sample_train_test_snapshots(
    *,
    start_train_day: str | pd.Timestamp,
    end_train_day: str | pd.Timestamp,
    start_test_day: str | pd.Timestamp,
    end_test_day: str | pd.Timestamp,
    n_train: int,
    n_test: int,
    seed: int,
    journeys: pd.DataFrame,
    idle_time_beg: int = 0,
    idle_time_end: int = 0,
) -> dict[str, list[str]]:
    """Sample fixed train/test snapshot timestamp values.

    Args:
        start_train_day: First calendar day eligible for training snapshots.
        end_train_day: Last calendar day eligible for training snapshots.
        start_test_day: First calendar day eligible for test snapshots.
        end_test_day: Last calendar day eligible for test snapshots.
        n_train: Number of training snapshot timestamps to sample.
        n_test: Number of test snapshot timestamps to sample.
        seed: Random seed used for timestamp sampling.
        journeys: Silver journey table used to identify seconds with at least
            one active journey.
        idle_time_beg: Minutes before journey start included in activity
            windows.
        idle_time_end: Minutes after journey end included in activity windows.

    Returns:
        Dictionary with `train_snapshots` and `test_snapshots` as ISO-8601
        timestamp strings.
    """

    train_start_ts = pd.Timestamp(start_train_day).normalize()
    train_end_day_ts = pd.Timestamp(end_train_day).normalize()
    train_end_ts = train_end_day_ts + pd.Timedelta(days=1)
    base_test_start_ts = pd.Timestamp(start_test_day).normalize()
    test_end_ts = pd.Timestamp(end_test_day).normalize() + pd.Timedelta(days=1)

    if train_end_ts <= train_start_ts:
        raise ValueError("`end_train_day` must be >= `start_train_day`.")
    if test_end_ts <= base_test_start_ts:
        raise ValueError("`end_test_day` must be >= `start_test_day`.")

    # Adjust test start to avoid overlap with trains from the last train service day.
    if "service_date" not in journeys.columns:
        raise KeyError("Missing required journeys column: service_date")
    service_date_ts = pd.to_datetime(journeys["service_date"], format="%d%b%Y", errors="coerce").dt.normalize()
    last_train_day_end_obs = pd.to_datetime(
        journeys.loc[service_date_ts == train_end_day_ts, "end_observed_ts"],
        errors="coerce",
    ).max()
    train_tail_cutoff_ts = base_test_start_ts
    if pd.notna(last_train_day_end_obs):
        train_tail_cutoff_ts = last_train_day_end_obs + pd.Timedelta(
            minutes=idle_time_end + TAIL_SAFETY_BUFFER_MIN
        )
    effective_test_start_ts = max(base_test_start_ts, train_tail_cutoff_ts)
    if effective_test_start_ts >= test_end_ts:
        raise ValueError(
            "Effective test start is outside test window: "
            f"effective_test_start_ts={effective_test_start_ts}, test_end_ts={test_end_ts}"
        )

    rng = np.random.default_rng(seed)
    one_sec_ns = 1_000_000_000

    def _sample_active_seconds(low_ns: int, high_ns: int, n: int) -> list[int]:
        """Sample active seconds within one split interval."""
        if n <= 0:
            return []

        low_sec = low_ns // one_sec_ns
        high_sec = high_ns // one_sec_ns
        if high_sec <= low_sec:
            raise ValueError("Invalid split bounds after second conversion.")

        start_obs_ns = _to_ns_int64(journeys["start_observed_ts"])
        end_obs_ns = _to_ns_int64(journeys["end_observed_ts"])
        start_planned_ns = _to_ns_int64(journeys["start_planned_ts"])

        appearance_ns = np.minimum(
            start_planned_ns - int(pd.Timedelta(minutes=idle_time_beg).value),
            start_obs_ns,
        )
        disappearance_ns = end_obs_ns + int(pd.Timedelta(minutes=idle_time_end).value)

        start_sec = np.floor_divide(appearance_ns + one_sec_ns - 1, one_sec_ns)
        end_sec = np.floor_divide(disappearance_ns + one_sec_ns - 1, one_sec_ns)

        start_sec = np.clip(start_sec, low_sec, high_sec)
        end_sec = np.clip(end_sec, low_sec, high_sec)
        valid_interval = start_sec < end_sec
        start_sec = start_sec[valid_interval]
        end_sec = end_sec[valid_interval]

        diff = np.zeros((high_sec - low_sec) + 1, dtype=np.int32)
        np.add.at(diff, start_sec - low_sec, 1)
        np.add.at(diff, end_sec - low_sec, -1)
        active_counts = np.cumsum(diff[:-1])
        valid_seconds = np.flatnonzero(active_counts > 0) + low_sec

        if len(valid_seconds) < n:
            raise ValueError(
                f"Requested {n} snapshots but only {len(valid_seconds)} active seconds are available."
            )
        sampled_seconds = rng.choice(valid_seconds, size=n, replace=False)
        sampled_ns = (np.sort(sampled_seconds).astype(np.int64) * one_sec_ns).tolist()
        return sampled_ns

    train_vals = _sample_active_seconds(int(train_start_ts.value), int(train_end_ts.value), int(n_train))
    test_vals = _sample_active_seconds(int(effective_test_start_ts.value), int(test_end_ts.value), int(n_test))
    
    return {
        "train_snapshots": [pd.Timestamp(v).isoformat(timespec="nanoseconds") for v in train_vals],
        "test_snapshots": [pd.Timestamp(v).isoformat(timespec="nanoseconds") for v in test_vals],
    }




def get_padded_arrays_from_components(
    *,
    ts: np.ndarray,
    op_ids: np.ndarray,
    planned: np.ndarray,
    delays: np.ndarray,
    types: np.ndarray,
    nb_past_events: int,
    nb_future_events: int,
    idle_time_beg_ns: int,
    idle_time_end_ns: int,
    missing_event_placeholder: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad one journey before and after its observed events.

    Args:
        ts: Observed event timestamps as int64 nanoseconds.
        op_ids: Operational point ids for the observed events.
        planned: Planned event timestamps as int64 nanoseconds.
        delays: Event delays in seconds.
        types: Event type labels for the observed events.
        nb_past_events: Number of placeholder events to add before the journey.
        nb_future_events: Number of placeholder events to add after the journey.
        idle_time_beg_ns: Offset in nanoseconds used for the pre-journey
            placeholder timestamp.
        idle_time_end_ns: Offset in nanoseconds used for the post-journey
            placeholder timestamp.
        missing_event_placeholder: Operational point id used for placeholder
            events.

    Returns:
        Tuple of padded arrays: observed timestamps, operational point ids,
        planned timestamps, delays, and event types.
    """
    n = len(ts)
    past = int(nb_past_events)
    future = int(nb_future_events)
    out_n = past + n + future

    ts_i64 = ts.astype(np.int64, copy=False)
    planned_i64 = planned.astype(np.int64, copy=False)
    op_i64 = op_ids.astype(np.int64, copy=False)

    past_delta_ns = int(idle_time_beg_ns)
    future_delta_ns = int(idle_time_end_ns)
    appearance_ts = min(int(ts_i64[0]), int(planned_i64[0]) - past_delta_ns)
    disappearance_ts = int(ts_i64[-1]) + future_delta_ns

    padded_ts = np.empty(out_n, dtype=np.int64)
    padded_ts[:past] = appearance_ts
    padded_ts[past : past + n] = ts_i64
    padded_ts[past + n :] = disappearance_ts

    padded_op_ids = np.empty(out_n, dtype=np.int64)
    padded_op_ids[:past] = int(missing_event_placeholder)
    padded_op_ids[past : past + n] = op_i64
    padded_op_ids[past + n :] = int(missing_event_placeholder)

    planned_past = int(planned_i64[0]) - past_delta_ns
    planned_future = int(planned_i64[-1]) + future_delta_ns
    padded_planned = np.empty(out_n, dtype=np.int64)
    padded_planned[:past] = planned_past
    padded_planned[past : past + n] = planned_i64
    padded_planned[past + n :] = planned_future

    padded_delay = np.empty(out_n, dtype=delays.dtype)
    padded_delay[:past] = 0
    padded_delay[past : past + n] = delays
    padded_delay[past + n :] = delays[-1]

    types_obj = types.astype(object, copy=False)
    padded_type = np.empty(out_n, dtype=object)
    padded_type[:past] = "D"
    padded_type[past : past + n] = types_obj
    padded_type[past + n :] = "A"

    return padded_ts, padded_op_ids, padded_planned, padded_delay, padded_type


def build_indexes(
    *,
    events_dir: Path,
    journeys_dir: Path,
    snapshot_config: dict,
    splits_to_build: list[str] | tuple[str, ...] = ("train", "test"),
    index_events_optional_get: list[str] | None = None,
    index_journeys_optional_get: list[str] | None = None,
    show_progress: bool = True,
) -> dict[str, dict[str, np.ndarray | dict[tuple[str, str], tuple[int, int]]]]:
    """Build compact Silver event and journey indexes for Gold snapshots.

    Args:
        events_dir: Directory containing monthly Silver event parquet files.
        journeys_dir: Directory containing monthly Silver journey parquet files.
        snapshot_config: Gold core configuration containing train/test
            snapshots and activity-window parameters.
        splits_to_build: Split names whose snapshots should be indexed.
        index_events_optional_get: Optional event fields to include in the
            event index.
        index_journeys_optional_get: Optional journey fields to include in the
            journey index.
        show_progress: Whether to display month-level progress bars.

    Returns:
        Dictionary with `events` and `journeys` entries. The event index stores
        event arrays plus `key_slices`, which maps `(train_id, service_date)` to
        the slice for that journey. The journey index stores active journey
        arrays and their activity-window bounds.
    """
    idle_time_beg = int(snapshot_config["idle_time_beg"])
    idle_time_end = int(snapshot_config["idle_time_end"])
    requested_splits = tuple(dict.fromkeys(str(s).lower() for s in splits_to_build))
    allowed_splits = {"train", "test"}
    unknown_splits = [s for s in requested_splits if s not in allowed_splits]
    if unknown_splits:
        raise ValueError(f"Unknown split(s) for index build: {unknown_splits}")
    if not requested_splits:
        raise ValueError("splits_to_build must contain at least one split among ['train', 'test'].")

    snapshot_parts: list[pd.DatetimeIndex] = []
    if "train" in requested_splits:
        snapshot_parts.append(pd.DatetimeIndex(pd.to_datetime(snapshot_config["train_snapshots"])))
    if "test" in requested_splits:
        snapshot_parts.append(pd.DatetimeIndex(pd.to_datetime(snapshot_config["test_snapshots"])))
    snapshot_values: list[pd.Timestamp] = []
    for part in snapshot_parts:
        snapshot_values.extend(part.tolist())
    index_snapshots = pd.DatetimeIndex(snapshot_values)
    snapshot_ns_all = np.sort(_to_ns_int64(index_snapshots))
    months = sorted(index_snapshots.strftime("%Y%m").unique().tolist())

    past_delta_ns = idle_time_beg * 60 * 1_000_000_000
    future_delta_ns = idle_time_end * 60 * 1_000_000_000

    optional_events_allowed = {"event_arr_line", "event_dep_line"}
    optional_journeys_allowed = {"train_relation", "operator", "deduced_path"}
    optional_events = set(index_events_optional_get or [])
    optional_journeys = set(index_journeys_optional_get or [])
    unknown_events = optional_events - optional_events_allowed
    unknown_journeys = optional_journeys - optional_journeys_allowed
    if unknown_events:
        raise ValueError(f"Unknown optional event index fields requested: {sorted(unknown_events)}")
    if unknown_journeys:
        raise ValueError(f"Unknown optional journey index fields requested: {sorted(unknown_journeys)}")

    events_parts: dict[str, list[np.ndarray]] = {
        "train_ids": [],
        "service_dates": [],
        "event_ts_ns": [],
        "event_op_ids": [],
        "event_planned_ns": [],
        "event_delay": [],
        "event_type": [],
    }
    if "event_arr_line" in optional_events:
        events_parts["event_arr_line"] = []
    if "event_dep_line" in optional_events:
        events_parts["event_dep_line"] = []
    journeys_parts: dict[str, list[np.ndarray]] = {
        "train_ids": [],
        "service_dates": [],
        "appearance_start": [],
        "disappearance_end": [],
    }
    if "train_relation" in optional_journeys:
        journeys_parts["train_relation"] = []
    if "operator" in optional_journeys:
        journeys_parts["operator"] = []
    if "deduced_path" in optional_journeys:
        journeys_parts["deduced_path"] = []

    month_iter = tqdm(months, desc="Building index", disable=not show_progress)
    for month in month_iter:
        journey_columns = [
            "train_id",
            "service_date",
            "start_observed_ts",
            "end_observed_ts",
            "start_planned_ts",
        ]
        if "train_relation" in optional_journeys:
            journey_columns.append("train_relation")
        if "operator" in optional_journeys:
            journey_columns.append("operator")
        if "deduced_path" in optional_journeys:
            journey_columns.append("deduced_paths")
        month_journeys = pd.read_parquet(
            journeys_dir / f"journeys_{month}.parquet",
            columns=journey_columns,
        )
        month_journeys["train_id"] = month_journeys["train_id"].astype(str)
        month_journeys["service_date"] = month_journeys["service_date"].astype(str)

        appearance_start_month, disappearance_end_month = compute_journey_activity_windows(
            month_journeys,
            idle_time_beg=idle_time_beg,
            idle_time_end=idle_time_end,
        )
        appearance_start_month_ns = _to_ns_int64(appearance_start_month)
        disappearance_end_month_ns = _to_ns_int64(disappearance_end_month)

        first_idx = np.searchsorted(snapshot_ns_all, appearance_start_month_ns, side="left")
        next_snapshot = np.full_like(appearance_start_month_ns, np.iinfo(np.int64).max)
        in_bounds = first_idx < len(snapshot_ns_all)
        next_snapshot[in_bounds] = snapshot_ns_all[first_idx[in_bounds]]
        active_mask = in_bounds & (next_snapshot < disappearance_end_month_ns)
        if not np.any(active_mask):
            continue

        active_cols = ["train_id", "service_date"]
        if "train_relation" in optional_journeys:
            active_cols.append("train_relation")
        if "operator" in optional_journeys:
            active_cols.append("operator")
        if "deduced_path" in optional_journeys:
            active_cols.append("deduced_paths")
        active_j = month_journeys.loc[active_mask, active_cols].copy()
        active_j["appearance_start"] = appearance_start_month_ns[active_mask]
        active_j["disappearance_end"] = disappearance_end_month_ns[active_mask]

        journeys_parts["train_ids"].append(active_j["train_id"].to_numpy(dtype=object, copy=False))
        journeys_parts["service_dates"].append(active_j["service_date"].to_numpy(dtype=object, copy=False))
        journeys_parts["appearance_start"].append(active_j["appearance_start"].to_numpy(dtype=np.int64, copy=False))
        journeys_parts["disappearance_end"].append(active_j["disappearance_end"].to_numpy(dtype=np.int64, copy=False))
        if "train_relation" in optional_journeys:
            journeys_parts["train_relation"].append(
                active_j["train_relation"].astype("string").fillna("nan").to_numpy(dtype=object, copy=False)
            )
        if "operator" in optional_journeys:
            journeys_parts["operator"].append(
                active_j["operator"].astype("string").fillna("nan").to_numpy(dtype=object, copy=False)
            )
        if "deduced_path" in optional_journeys:
            journeys_parts["deduced_path"].append(
                active_j["deduced_paths"].to_numpy(dtype=object, copy=False)
            )

        month_events = pd.read_parquet(
            events_dir / f"events_{month}.parquet",
            columns=[
                "train_id",
                "service_date",
                "op_id",
                "event_type",
                "observed_ts",
                "planned_ts",
                "delay_sec",
            ]
            + (["arr_line_id"] if "event_arr_line" in optional_events else [])
            + (["dep_line_id"] if "event_dep_line" in optional_events else []),
        )
        month_events["train_id"] = month_events["train_id"].astype(str)
        month_events["service_date"] = month_events["service_date"].astype(str)
        month_events = month_events.merge(
            active_j[["train_id", "service_date"]],
            on=["train_id", "service_date"],
            how="inner",
            sort=False,
        )
        if month_events.empty:
            continue
        month_events["event_ts"] = pd.to_datetime(month_events["observed_ts"])
        month_events["planned_ts"] = pd.to_datetime(month_events["planned_ts"])
        month_events["op_id"] = pd.to_numeric(month_events["op_id"]).astype("Int64")

        events_parts["train_ids"].append(month_events["train_id"].to_numpy(dtype=object, copy=False))
        events_parts["service_dates"].append(month_events["service_date"].to_numpy(dtype=object, copy=False))
        events_parts["event_ts_ns"].append(_to_ns_int64(month_events["event_ts"]))
        events_parts["event_op_ids"].append(month_events["op_id"].astype("int64").to_numpy())
        events_parts["event_planned_ns"].append(_to_ns_int64(month_events["planned_ts"]))
        events_parts["event_delay"].append(month_events["delay_sec"].to_numpy())
        events_parts["event_type"].append(month_events["event_type"].astype(str).to_numpy())
        if "event_arr_line" in optional_events:
            events_parts["event_arr_line"].append(
                month_events["arr_line_id"].astype("string").fillna("").to_numpy()
            )
        if "event_dep_line" in optional_events:
            events_parts["event_dep_line"].append(
                month_events["dep_line_id"].astype("string").fillna("").to_numpy()
            )

    if not journeys_parts["train_ids"]:
        raise ValueError("No active journeys overlap provided snapshots.")
    if not events_parts["train_ids"]:
        raise ValueError("No relevant events found for active journeys.")

    journeys_parts["train_ids"] = np.concatenate(journeys_parts["train_ids"]).astype(object, copy=False)
    journeys_parts["service_dates"] = np.concatenate(journeys_parts["service_dates"]).astype(object, copy=False)
    journeys_parts["appearance_start"] = np.concatenate(journeys_parts["appearance_start"]).astype(np.int64, copy=False)
    journeys_parts["disappearance_end"] = np.concatenate(journeys_parts["disappearance_end"]).astype(np.int64, copy=False)
    if "train_relation" in optional_journeys:
        journeys_parts["train_relation"] = np.concatenate(journeys_parts["train_relation"]).astype(object, copy=False)
    if "operator" in optional_journeys:
        journeys_parts["operator"] = np.concatenate(journeys_parts["operator"]).astype(object, copy=False)
    if "deduced_path" in optional_journeys:
        journeys_parts["deduced_path"] = np.concatenate(journeys_parts["deduced_path"]).astype(object, copy=False)

    key_slices: dict[tuple[str, str], tuple[int, int]] = {}
    events_parts["train_ids"] = np.concatenate(events_parts["train_ids"]).astype(object, copy=False)
    events_parts["service_dates"] = np.concatenate(events_parts["service_dates"]).astype(object, copy=False)
    key_train = events_parts["train_ids"]
    key_service = events_parts["service_dates"]
    prev_key = (key_train[0], key_service[0])
    start_idx = 0
    for i in range(1, len(key_train)):
        cur_key = (key_train[i], key_service[i])
        if cur_key != prev_key:
            key_slices[prev_key] = (start_idx, i)
            prev_key = cur_key
            start_idx = i
    key_slices[prev_key] = (start_idx, len(key_train))
    events_parts.pop("train_ids", None)
    events_parts.pop("service_dates", None)

    events_parts["event_ts_ns"] = np.concatenate(events_parts["event_ts_ns"]).astype(np.int64, copy=False)
    events_parts["event_op_ids"] = np.concatenate(events_parts["event_op_ids"]).astype(np.int64, copy=False)
    events_parts["event_planned_ns"] = np.concatenate(events_parts["event_planned_ns"]).astype(np.int64, copy=False)
    events_parts["event_delay"] = np.concatenate(events_parts["event_delay"])
    events_parts["event_type"] = np.concatenate(events_parts["event_type"]).astype(str, copy=False)
    if "event_arr_line" in optional_events:
        events_parts["event_arr_line"] = np.concatenate(events_parts["event_arr_line"]).astype(str, copy=False)
    if "event_dep_line" in optional_events:
        events_parts["event_dep_line"] = np.concatenate(events_parts["event_dep_line"]).astype(str, copy=False)
    events_parts["key_slices"] = key_slices

    return {
        "events": events_parts,
        "journeys": journeys_parts,
    }


def _get_padded_for_key(
    *,
    train_key: tuple[str, str],
    events_index: dict[str, np.ndarray | dict[tuple[str, str], tuple[int, int]]],
    padded_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_future: int,
    past_delta_ns: int,
    future_delta_ns: int,
    missing_event_placeholder: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cached padded arrays for one train/service-date key."""
    cached = padded_cache.get(train_key)
    if cached is not None:
        return cached
    key_slices = events_index["key_slices"]
    if not isinstance(key_slices, dict):
        raise TypeError("events_index['key_slices'] must be a dict.")
    start, end = key_slices[train_key]
    cached = get_padded_arrays_from_components(
        ts=events_index["event_ts_ns"][start:end],  # type: ignore[index]
        op_ids=events_index["event_op_ids"][start:end],  # type: ignore[index]
        planned=events_index["event_planned_ns"][start:end],  # type: ignore[index]
        delays=events_index["event_delay"][start:end],  # type: ignore[index]
        types=events_index["event_type"][start:end],  # type: ignore[index]
        nb_past_events=1,
        nb_future_events=n_future,
        idle_time_beg_ns=past_delta_ns,
        idle_time_end_ns=future_delta_ns,
        missing_event_placeholder=missing_event_placeholder,
    )
    padded_cache[train_key] = cached
    return cached


def _build_eval_table_for_snapshots(
    *,
    snapshot_list: list[str],
    split_name: str,
    journeys_index: dict[str, np.ndarray],
    events_index: dict[str, np.ndarray | dict[tuple[str, str], tuple[int, int]]],
    n_future: int,
    past_delta_ns: int,
    future_delta_ns: int,
    missing_event_placeholder: int,
    padded_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    show_progress: bool,
) -> pd.DataFrame:
    """Build the standardized evaluation table for one snapshot split."""
    future_obs_cols = [f"future_obs_ts_{i}" for i in range(1, n_future + 1)]
    future_op_cols = [f"future_op_id_{i}" for i in range(1, n_future + 1)]
    future_type_cols = [f"future_event_type_{i}" for i in range(1, n_future + 1)]
    future_delay_cols = [f"future_delay_{i}" for i in range(1, n_future + 1)]
    columns = (
        ["ts", "train_id", "service_date", "last_known_delay"]
        + future_obs_cols
        + future_op_cols
        + future_type_cols
        + future_delay_cols
    )
    rows: list[tuple[object, ...]] = []

    snapshots = pd.to_datetime(snapshot_list)
    snapshot_ns = _to_ns_int64(snapshots)
    snapshot_iter = tqdm(
        zip(snapshots, snapshot_ns),
        total=len(snapshots),
        desc=f"Building {split_name} eval tables",
        disable=not show_progress,
    )
    for snap_ts, snap_ns in snapshot_iter:
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snap_ns,
        )
        active_indices = active_mask.nonzero()[0]
        for idx in active_indices:
            train_id = journeys_index["train_ids"][idx]
            service_date = journeys_index["service_dates"][idx]
            padded_ts, padded_op_ids, padded_planned, padded_delay, padded_type = _get_padded_for_key(
                train_key=(train_id, service_date),
                events_index=events_index,
                padded_cache=padded_cache,
                n_future=n_future,
                past_delta_ns=past_delta_ns,
                future_delta_ns=future_delta_ns,
                missing_event_placeholder=missing_event_placeholder,
            )
            split_idx = int(np.searchsorted(padded_ts, snap_ns, side="right"))
            past_slot = split_idx - 1
            future_slots = np.arange(split_idx, split_idx + n_future, dtype=np.int64)

            past_delay_1 = float(padded_delay[past_slot])
            future_planned_1 = (padded_planned[future_slots[0]] - snap_ns) / 1e9
            last_known_delay = past_delay_1 - (
                (future_planned_1 + past_delay_1) < 0
            ) * (future_planned_1 + past_delay_1)

            row_items: list[object] = [snap_ts, train_id, service_date, last_known_delay]
            future_mask = ~(padded_op_ids == missing_event_placeholder)[future_slots]

            future_ts = np.where(future_mask, padded_ts[future_slots], -1).astype(np.int64, copy=False)
            future_op = np.where(future_mask, padded_op_ids[future_slots], -1).astype(np.int16, copy=False)
            future_type = np.full(n_future, -1, dtype=np.int8)
            valid_types = padded_type[future_slots][future_mask]
            future_type[future_mask] = np.where(
                valid_types == "A",
                np.int8(0),
                np.where(valid_types == "D", np.int8(1), np.int8(2)),
            ).astype(np.int8, copy=False)
            future_delay = np.where(future_mask, padded_delay[future_slots], np.nan).astype(np.float32, copy=False)

            row_items.extend(future_ts.tolist())
            row_items.extend(future_op.tolist())
            row_items.extend(future_type.tolist())
            row_items.extend(future_delay.tolist())
            rows.append(tuple(row_items))

    eval_table = pd.DataFrame.from_records(rows, columns=columns)
    eval_table["last_known_delay"] = pd.to_numeric(
        eval_table["last_known_delay"], errors="coerce"
    ).astype(np.float32, copy=False)
    for col in future_op_cols:
        eval_table[col] = pd.to_numeric(eval_table[col], errors="coerce").fillna(-1).astype(np.int16, copy=False)
    for col in future_type_cols:
        eval_table[col] = pd.to_numeric(eval_table[col], errors="coerce").fillna(-1).astype(np.int8, copy=False)
    for col in future_delay_cols:
        eval_table[col] = pd.to_numeric(eval_table[col], errors="coerce").astype(np.float32, copy=False)
    for col in future_obs_cols:
        vals = pd.to_numeric(eval_table[col], errors="coerce")
        eval_table[col] = pd.to_datetime(vals.where(vals >= 0), unit="ns")

    return eval_table


def build_gold_eval_table(
    *,
    events_dir: Path,
    journeys_dir: Path,
    snapshot_config: dict,
    missing_event_placeholder: int = -1,
    index_events_optional_get: list[str] | None = None,
    index_journeys_optional_get: list[str] | None = None,
    build_train_eval_table: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Build mandatory standardized metadata/label tables for fair evaluation.

    Args:
        events_dir: Directory containing monthly Silver event parquet files.
        journeys_dir: Directory containing monthly Silver journey parquet files.
        snapshot_config: Gold core configuration containing train/test
            snapshots, future horizon, and activity-window parameters.
        missing_event_placeholder: Operational point id used to mark padded
            future event slots.
        index_events_optional_get: Optional event fields to include while
            building the shared index.
        index_journeys_optional_get: Optional journey fields to include while
            building the shared index.
        build_train_eval_table: Whether to also materialize the train split
            evaluation table.
        show_progress: Whether to display progress bars.

    Returns:
        Dictionary containing `test_eval_table` and, when requested,
        `train_eval_table`.

    Note:
    - `build_train_eval_table=True` can consume a lot of RAM when the number of
      train snapshots is large, because all train rows are materialized in a DataFrame.
    """
    print("Building mandatory standardized metadata/label tables for fair evaluation...")

    n_future = int(snapshot_config["n_future"])
    idle_time_beg = int(snapshot_config["idle_time_beg"])
    idle_time_end = int(snapshot_config["idle_time_end"])

    past_delta_ns = idle_time_beg * 60 * 1_000_000_000
    future_delta_ns = idle_time_end * 60 * 1_000_000_000

    index = build_indexes(
        events_dir=events_dir,
        journeys_dir=journeys_dir,
        snapshot_config=snapshot_config,
        splits_to_build=["train", "test"] if build_train_eval_table else ["test"],
        index_events_optional_get=index_events_optional_get,
        index_journeys_optional_get=index_journeys_optional_get,
        show_progress=show_progress,
    )
    events_parts = index["events"]
    journeys_parts = index["journeys"]

    padded_cache: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    test_eval_table = _build_eval_table_for_snapshots(
        snapshot_list=snapshot_config["test_snapshots"],
        split_name="test",
        journeys_index=journeys_parts,  # type: ignore[arg-type]
        events_index=events_parts,  # type: ignore[arg-type]
        n_future=n_future,
        past_delta_ns=past_delta_ns,
        future_delta_ns=future_delta_ns,
        missing_event_placeholder=missing_event_placeholder,
        padded_cache=padded_cache,
        show_progress=show_progress,
    )
    result = {"test_eval_table": test_eval_table}
    if build_train_eval_table:
        result["train_eval_table"] = _build_eval_table_for_snapshots(
            snapshot_list=snapshot_config["train_snapshots"],
            split_name="train",
            journeys_index=journeys_parts,  # type: ignore[arg-type]
            events_index=events_parts,  # type: ignore[arg-type]
            n_future=n_future,
            past_delta_ns=past_delta_ns,
            future_delta_ns=future_delta_ns,
            missing_event_placeholder=missing_event_placeholder,
            padded_cache=padded_cache,
            show_progress=show_progress,
        )
    return result
