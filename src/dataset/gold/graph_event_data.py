"""Gold graph-event dataset construction utilities."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.pipeline.helpers import read_yaml
from src.dataset.gold.helpers import (
    build_indexes,
    get_active_mask_for_snapshot,
    get_padded_arrays_from_components,
)


def journeys_meta_from_index(index: dict) -> pd.DataFrame:
    """Build journey metadata used by the graph-event dataset."""
    journeys = index["journeys"]
    journeys_meta = pd.DataFrame(
        {
            "train_id": journeys["train_ids"].astype(str, copy=False),
            "service_date": journeys["service_dates"].astype(str, copy=False),
            "train_relation": journeys["train_relation"].astype(str, copy=False),
            "deduced_paths": journeys["deduced_path"],
        }
    )
    relation_categories = {"EURST", "EXTRA", "IC", "ICE", "INT", "L", "P", "TGV", "THAL", "S", "CHARTER", "nan"}
    rel = journeys_meta["train_relation"].astype("string").fillna("nan").str.replace(r"^(S).*", r"\1", regex=True)
    rel = rel.str.split(" ").str[0]
    rel = rel.where(rel.isin(relation_categories), "nan")
    journeys_meta["train_relation"] = rel
    return journeys_meta[["train_id", "service_date", "train_relation", "deduced_paths"]]


def build_travel_time_samples(
    *,
    index: dict,
    journeys_meta: pd.DataFrame,
    train_snapshots: list[str],
    show_progress: bool = True,
) -> dict[tuple[int, str, str, int, str, str, str], list[float]]:
    """Collect train-split travel-time samples by event-transition signature."""
    events_index = index["events"]
    relation_by_key = {
        (str(r.train_id), str(r.service_date)): str(r.train_relation)
        for r in journeys_meta.itertuples(index=False)
    }
    out: dict[tuple[int, str, str, int, str, str, str], list[float]] = defaultdict(list)

    key_slices: dict[tuple[str, str], tuple[int, int]] = events_index["key_slices"]
    event_ts_ns = events_index["event_ts_ns"]
    event_op_ids = events_index["event_op_ids"]
    event_type = events_index["event_type"]
    event_dep_line = events_index["event_dep_line"]
    event_arr_line = events_index["event_arr_line"]
    journey_index = index["journeys"]
    train_keys: set[tuple[str, str]] = set()
    # Filter for train journeys to avoid data leakage
    for snap_ns in pd.to_datetime(train_snapshots, errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64):
        active_mask = get_active_mask_for_snapshot(
            journey_index["appearance_start"],
            journey_index["disappearance_end"],
            snap_ns,
        )
        active_idx = np.nonzero(active_mask)[0]
        for i in active_idx:
            train_keys.add((str(journey_index["train_ids"][i]), str(journey_index["service_dates"][i])))

    key_iter = tqdm(
        key_slices.items(),
        total=len(key_slices),
        desc="Building travel-time samples",
        disable=not show_progress,
    )
    for key, (start, end) in key_iter:
        if (str(key[0]), str(key[1])) not in train_keys:
            continue
        relation = relation_by_key.get(key, "nan")
        if end - start < 2:
            continue
        ts = event_ts_ns[start:end]
        op = event_op_ids[start:end]
        typ = event_type[start:end]
        dep = event_dep_line[start:end]
        arr = event_arr_line[start:end]
        for i in range(len(ts) - 1):
            tpl = (
                int(op[i]),
                str(typ[i]),
                str(dep[i]),
                int(op[i + 1]),
                str(typ[i + 1]),
                str(arr[i + 1]),
                relation,
            )
            dt_sec = (int(ts[i + 1]) - int(ts[i])) / 1e9
            out[tpl].append(float(dt_sec))

    return dict(out)


def summarize_travel_time_samples(
    samples: dict[tuple[int, str, str, int, str, str, str], list[float]]
) -> dict[tuple[int, str, str, int, str, str, str], dict[str, float | int]]:
    """Summarize travel-time sample distributions for graph-event simulation."""
    out: dict[tuple[int, str, str, int, str, str, str], dict[str, float | int]] = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q05": float(np.quantile(arr, 0.05)),
            "q10": float(np.quantile(arr, 0.10)),
            "q15": float(np.quantile(arr, 0.15)),
            "q20": float(np.quantile(arr, 0.20)),
            "q25": float(np.quantile(arr, 0.25)),
            "q30": float(np.quantile(arr, 0.30)),
            "q35": float(np.quantile(arr, 0.35)),
            "q40": float(np.quantile(arr, 0.40)),
            "q45": float(np.quantile(arr, 0.45)),
        }
    return out


def build_journeys_table(
    *,
    index: dict,
    journeys_meta: pd.DataFrame,
    snapshot_cfg: dict,
    missing_event_placeholder: int = -1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build test-snapshot journey rows used by the graph-event model."""
    events_index = index["events"]
    journeys_index = index["journeys"]
    relation_by_key = {
        (str(r.train_id), str(r.service_date)): str(r.train_relation)
        for r in journeys_meta.itertuples(index=False)
    }
    paths_by_key = {
        (str(r.train_id), str(r.service_date)): r.deduced_paths
        for r in journeys_meta.itertuples(index=False)
    }
    key_slices: dict[tuple[str, str], tuple[int, int]] = events_index["key_slices"]
    event_ts_ns: np.ndarray = events_index["event_ts_ns"]
    idle_time_beg_ns = int(snapshot_cfg["idle_time_beg"]) * 60 * 1_000_000_000
    idle_time_end_ns = int(snapshot_cfg["idle_time_end"]) * 60 * 1_000_000_000
    padded_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    snapshots = pd.to_datetime(snapshot_cfg["test_snapshots"], errors="coerce")
    snapshot_ns = snapshots.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    rows: list[tuple[pd.Timestamp, str, str, int, float, str, object]] = []
    snapshot_iter = tqdm(
        zip(snapshots, snapshot_ns),
        total=len(snapshots),
        desc="Building test journeys table",
        disable=not show_progress,
    )
    for snap_ts, snap_ns in snapshot_iter:
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snap_ns,
        )
        active_indices = np.nonzero(active_mask)[0]
        for idx in active_indices:
            train_id = str(journeys_index["train_ids"][idx])
            service_date = str(journeys_index["service_dates"][idx])
            key = (train_id, service_date)
            start, end = key_slices[key]
            local_idx = int(np.searchsorted(event_ts_ns[start:end], snap_ns, side="right")) - 1

            padded = padded_cache.get(key)
            if padded is None:
                padded = get_padded_arrays_from_components(
                    ts=events_index["event_ts_ns"][start:end],
                    op_ids=events_index["event_op_ids"][start:end],
                    planned=events_index["event_planned_ns"][start:end],
                    delays=events_index["event_delay"][start:end],
                    types=events_index["event_type"][start:end],
                    nb_past_events=1,
                    nb_future_events=1,
                    idle_time_beg_ns=idle_time_beg_ns,
                    idle_time_end_ns=idle_time_end_ns,
                    missing_event_placeholder=missing_event_placeholder,
                )
                padded_cache[key] = padded
            padded_ts, _padded_op_ids, padded_planned, padded_delay, _padded_type = padded
            split_idx = int(np.searchsorted(padded_ts, snap_ns, side="right"))
            past_slot = split_idx - 1
            future_slot = split_idx
            past_delay_1 = float(padded_delay[past_slot])
            future_planned_1 = (padded_planned[future_slot] - snap_ns) / 1e9
            last_known_delay = past_delay_1 - (((future_planned_1 + past_delay_1) < 0) * (future_planned_1 + past_delay_1))

            rows.append(
                (
                    pd.Timestamp(snap_ts),
                    train_id,
                    service_date,
                    local_idx,
                    float(last_known_delay),
                    relation_by_key.get(key, "nan"),
                    paths_by_key.get(key, []),
                )
            )

    return pd.DataFrame(
        rows,
        columns=["ts", "train_id", "service_date", "current_event_idx", "last_known_delay", "train_relation", "path"],
    )


def build_events_table_from_index(
    *,
    index: dict,
    keys_filter: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build the event table for the journey keys used by graph-event evaluation."""
    events_index = index["events"]
    key_slices: dict[tuple[str, str], tuple[int, int]] = events_index["key_slices"]
    event_op_ids: np.ndarray = events_index["event_op_ids"]
    event_type: np.ndarray = events_index["event_type"]
    event_planned_ns: np.ndarray = events_index["event_planned_ns"]
    event_dep_line: np.ndarray = events_index["event_dep_line"]
    event_arr_line: np.ndarray = events_index["event_arr_line"]

    rows: list[tuple[str, str, int, str, pd.Timestamp, str, str]] = []
    key_iter = tqdm(
        key_slices.items(),
        total=len(key_slices),
        desc="Building events table",
        disable=False,
    )
    for (train_id, service_date), (start, end) in key_iter:
        if keys_filter is not None and (str(train_id), str(service_date)) not in keys_filter:
            continue
        op = event_op_ids[start:end]
        typ = event_type[start:end]
        planned = pd.to_datetime(event_planned_ns[start:end], unit="ns")
        dep = event_dep_line[start:end]
        arr = event_arr_line[start:end]
        rows.extend(
            (str(train_id), str(service_date), int(op[i]), str(typ[i]), planned[i], str(dep[i]), str(arr[i]))
            for i in range(end - start)
        )

    return pd.DataFrame(
        rows,
        columns=["train_id", "service_date", "op_id", "event_type", "planned_ts", "line_dep", "line_arr"],
    )


def build_graph_event_dataset(
    *,
    dataset_core_spec: Path,
    silver_dir: Path = Path("data/silver"),
    missing_event_placeholder: int = -1,
    output_dir: Path,
) -> None:
    """Build the Gold graph-event dataset from a Silver dataset and Gold core.

    Args:
        dataset_core_spec: Path to the Gold core specification YAML.
        silver_dir: Root directory of the Silver dataset.
        missing_event_placeholder: Operational point id used for padded event
            slots.
        output_dir: Directory where graph-event files are written.

    Returns:
        None. The function writes the graph-event dataset under `output_dir`.
    """
    events_dir = silver_dir / "events"
    journeys_dir = silver_dir / "journeys"
    node_links_path = silver_dir / "static" / "node_links.parquet"

    snapshot_cfg = read_yaml(dataset_core_spec)
    index = build_indexes(
        events_dir=events_dir,
        journeys_dir=journeys_dir,
        snapshot_config=snapshot_cfg,
        splits_to_build=["train", "test"],
        index_events_optional_get=["event_arr_line", "event_dep_line"],
        index_journeys_optional_get=["train_relation", "deduced_path"],
        show_progress=True,
    )
    journeys_meta = journeys_meta_from_index(index)

    travel_time_samples = build_travel_time_samples(
        index=index,
        journeys_meta=journeys_meta,
        train_snapshots=snapshot_cfg["train_snapshots"],
        show_progress=True,
    )
    travel_time_stats = summarize_travel_time_samples(travel_time_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = output_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "travel_time_samples.pkl"
    with output_path.open("wb") as f:
        pickle.dump(travel_time_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    test_journeys = build_journeys_table(
        index=index,
        journeys_meta=journeys_meta,
        snapshot_cfg=snapshot_cfg,
        missing_event_placeholder=missing_event_placeholder,
        show_progress=True,
    )
    test_journeys.to_parquet(test_dir / "journeys.parquet", index=False)
    test_keys = set(zip(test_journeys["train_id"].astype(str), test_journeys["service_date"].astype(str)))
    test_events = build_events_table_from_index(index=index, keys_filter=test_keys)
    test_events.to_parquet(test_dir / "events.parquet", index=False)
    shutil.copy2(node_links_path, output_dir / "node_links.parquet")
