"""Graph-event benchmark model utilities."""

from __future__ import annotations

import heapq
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PRECEDENCE_GAP_S = 40


def load_graph_event_tables(
    *,
    data_dir: Path,
    test_eval_table: Path,
) -> dict[str, object]:
    """Load graph-event test tables, static link data, evaluation rows, and travel-time samples."""
    test_dir = data_dir / "test"
    journeys = pd.read_parquet(test_dir / "journeys.parquet")
    events = pd.read_parquet(test_dir / "events.parquet")
    node_links = pd.read_parquet(data_dir / "node_links.parquet")
    eval_table = pd.read_parquet(test_eval_table)

    with (data_dir / "travel_time_samples.pkl").open("rb") as f:
        travel_time_stats = pickle.load(f)

    return {
        "journeys": journeys,
        "events": events,
        "node_links": node_links,
        "eval_table": eval_table,
        "travel_time_stats": travel_time_stats,
    }


def compute_subpath_link_times(
    *,
    u_op_id: int,
    u_event_type: str,
    u_line_dep: str,
    v_op_id: int,
    v_event_type: str,
    v_line_arr: str,
    planned_delta_sec: float,
    subpath: list[object] | np.ndarray,
    current_delay_sec: float,
    relation: str,
    travel_time_stats: dict,
    link_distance_m: dict[int, float],
) -> list[float]:
    """Allocate expected event-pair travel time over the links in a deduced subpath."""
    total_sec = compute_event_pair_expected_time(
        u_op_id=u_op_id,
        u_event_type=u_event_type,
        u_line_dep=u_line_dep,
        v_op_id=v_op_id,
        v_event_type=v_event_type,
        v_line_arr=v_line_arr,
        planned_delta_sec=planned_delta_sec,
        relation=relation,
        travel_time_stats=travel_time_stats,
    )
    planned_sec = float(planned_delta_sec)
    if not np.isfinite(planned_sec):
        raise ValueError(
            "Invalid planned event-pair travel time (non-finite) for "
            f"a=({u_op_id},{u_event_type}), b=({v_op_id},{v_event_type}), planned_sec={planned_sec}"
        )
    min_total_sec = planned_sec - float(current_delay_sec)
    # Do not let sampled travel times imply arrival earlier than the currently observed delay allows.
    total_sec = max(total_sec, min_total_sec)
    if len(subpath) == 0:
        raise ValueError("Subpath is empty.")
    if len(subpath) == 1:
        return [total_sec]
    if any(not isinstance(x, (int, np.integer)) for x in subpath):
        raise ValueError(f"Invalid non-integer link id in subpath={list(subpath)}")

    subpath_int = [int(x) for x in subpath]
    distances = np.asarray(
        [float(link_distance_m[int(link_id)]) for link_id in subpath_int],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(distances)):
        raise ValueError(f"Non-finite link distance in subpath={list(subpath)}")
    if np.any(distances <= 0):
        raise ValueError(f"Non-positive link distance in subpath={list(subpath)} values={distances.tolist()}")
    dist_sum = float(distances.sum())
    per_link = total_sec * (distances / dist_sum)
    return per_link.astype(np.float64, copy=False).tolist()


def compute_event_pair_expected_time(
    *,
    u_op_id: int,
    u_event_type: str,
    u_line_dep: str,
    v_op_id: int,
    v_event_type: str,
    v_line_arr: str,
    planned_delta_sec: float,
    relation: str,
    travel_time_stats: dict,
) -> float:
    """Estimate travel time between two consecutive train events from samples or schedule."""
    key = (
        int(u_op_id),
        str(u_event_type),
        str(u_line_dep),
        int(v_op_id),
        str(v_event_type),
        str(v_line_arr),
        str(relation),
    )
    stats = travel_time_stats.get(key)
    if stats is None:
        total_sec = float(planned_delta_sec)
    else:
        total_sec = np.nan
        # Prefer lower empirical quantiles when available; otherwise fall back to schedule time.
        for q in ("q40", "q45", "median"):
            v = float(stats.get(q, np.nan))
            if np.isfinite(v) and v > 0:
                total_sec = v
                break
        if not np.isfinite(total_sec) or total_sec <= 0:
            total_sec = float(planned_delta_sec)
    if not np.isfinite(total_sec) or total_sec < 0:
        raise ValueError(
            "Invalid subpath travel time (non-finite or < 0) for "
            f"a=({u_op_id},{u_event_type}), b=({v_op_id},{v_event_type}), relation={relation}, total_sec={total_sec}"
        )
    return total_sec


def locate_subpath_idx_from_elapsed(
    *,
    subpath_expected_times: list[float],
    elapsed_sec: float,
) -> int:
    """Locate the current subpath element from elapsed seconds since event-pair departure."""
    if len(subpath_expected_times) == 0:
        raise ValueError("Empty subpath_expected_times.")
    if elapsed_sec <= 0:
        return 0
    cumulative = np.cumsum(np.asarray(subpath_expected_times, dtype=np.float64))
    idx = int(np.searchsorted(cumulative, elapsed_sec, side="left"))
    return max(0, min(idx, len(subpath_expected_times) - 1))


def annotate_subpath_direction(
    *,
    subpath: list[int],
    start_op_id: int,
    link_endpoints: dict[int, tuple[int, int]],
) -> list[tuple[int, str]]:
    """Annotate each raw link id with the direction followed by the train."""
    directed: list[tuple[int, str]] = []
    current_node = int(start_op_id)
    for link_id in subpath:
        u, v = link_endpoints[int(link_id)]
        if current_node == u:
            directed.append((int(link_id), "uv"))
            current_node = v
        elif current_node == v:
            directed.append((int(link_id), "vu"))
            current_node = u
        else:
            raise ValueError(
                f"Subpath direction inference failed at link_id={link_id}: "
                f"current_node={current_node}, endpoints=({u},{v})"
            )
    return directed


def initialize_snapshot_state(
    *,
    snapshot_ts: pd.Timestamp,
    journeys_filtered: pd.DataFrame,
    events_filtered: pd.DataFrame,
    link_distance_m: dict[int, float],
    link_endpoints: dict[int, tuple[int, int]],
    travel_time_stats: dict,
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[object, list[tuple[str, str]]],
    dict[tuple[str, str], dict[str, np.ndarray]],
]:
    """Create train and network occupancy state at one prediction snapshot."""
    train_state: dict[tuple[str, str], dict[str, object]] = {}
    state_entries: list[tuple[object, int, str, str]] = []
    snapshot_ts_s = int(pd.Timestamp(snapshot_ts).value // 1_000_000_000)

    events_by_key: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, grp in events_filtered.groupby(["train_id", "service_date"], sort=False):
        g = grp.reset_index(drop=True)
        planned_s = (
            pd.to_datetime(g["planned_ts"], errors="coerce").astype("int64") // 1_000_000_000
        ).to_numpy(dtype=np.int64, copy=False)
        events_by_key[key] = {
            "op_id": pd.to_numeric(g["op_id"], errors="coerce").to_numpy(dtype=np.int64, copy=False),
            "event_type": g["event_type"].astype(str).to_numpy(dtype=object, copy=False),
            "line_dep": g["line_dep"].astype(str).to_numpy(dtype=object, copy=False),
            "line_arr": g["line_arr"].astype(str).to_numpy(dtype=object, copy=False),
            "planned_s": planned_s,
        }

    for row in journeys_filtered.itertuples(index=False):
        key = (str(row.train_id), str(row.service_date))
        k = int(row.current_event_idx)
        path = row.path
        relation = str(row.train_relation)
        events_group = events_by_key.get(key)
        if events_group is None:
            raise ValueError(f"Missing events group for journey key={key}")
        delay = float(row.last_known_delay)
        if np.isnan(delay):
            delay = 0.0

        if k < 0:
            # Before the first observed event, keep the train in a synthetic departure state.
            first_planned_s = int(events_group["planned_s"][0])
            subpath_expected_times = [5 * 60.0]
            current_link_expected_finish_ts = (
                first_planned_s if first_planned_s > snapshot_ts_s else int(snapshot_ts_s + 1)
            )
            subpath = ["DEPARTING"]
            subpath_current_idx = 0
            entry_time_s = snapshot_ts_s
        elif k >= len(path):
            # After the deduced path ends, keep the train in a synthetic arrival state.
            last_planned_s = int(events_group["planned_s"][-1])
            subpath_expected_times = [5 * 60.0]
            current_link_expected_finish_ts = int(np.ceil(last_planned_s + delay + 300))
            subpath = ["ARRIVED"]
            subpath_current_idx = 0
            entry_time_s = current_link_expected_finish_ts
        else:
            planned_s = int(events_group["planned_s"][k])
            pred_start_s = planned_s + delay
            elapsed_sec = float(snapshot_ts_s - pred_start_s)
            raw_subpath = [int(link_id) for link_id in path[k]]
            if len(raw_subpath) == 0:
                # Empty subpaths represent dwell time at the current operational point.
                stop_op_id = int(events_group["op_id"][k])
                subpath = [f"STOPPED@op{stop_op_id}"]
            planned_delta_sec = float(events_group["planned_s"][k + 1] - events_group["planned_s"][k])
            subpath_expected_times = compute_subpath_link_times(
                u_op_id=int(events_group["op_id"][k]),
                u_event_type=str(events_group["event_type"][k]),
                u_line_dep=str(events_group["line_dep"][k]),
                v_op_id=int(events_group["op_id"][k + 1]),
                v_event_type=str(events_group["event_type"][k + 1]),
                v_line_arr=str(events_group["line_arr"][k + 1]),
                planned_delta_sec=planned_delta_sec,
                subpath=raw_subpath if len(raw_subpath) > 0 else [f"STOPPED@op{int(events_group['op_id'][k])}"],
                current_delay_sec=delay,
                relation=relation,
                travel_time_stats=travel_time_stats,
                link_distance_m=link_distance_m,
            )
            if len(raw_subpath) > 0:
                subpath = annotate_subpath_direction(
                    subpath=raw_subpath,
                    start_op_id=int(events_group["op_id"][k]),
                    link_endpoints=link_endpoints,
                )
            subpath_current_idx = locate_subpath_idx_from_elapsed(
                subpath_expected_times=subpath_expected_times,
                elapsed_sec=elapsed_sec,
            )
            elapsed_until_link_end_sec = float(np.sum(subpath_expected_times[: subpath_current_idx + 1]))
            current_link_expected_finish_ts = int(np.ceil(pred_start_s + elapsed_until_link_end_sec))
            elapsed_until_link_start_sec = float(np.sum(subpath_expected_times[:subpath_current_idx]))
            entry_time_s = int(np.floor(pred_start_s + elapsed_until_link_start_sec))
        state_entries.append((subpath[subpath_current_idx], entry_time_s, key[0], key[1]))
        train_state[key] = {
            "event_idx": k,
            "subpath": subpath,
            "subpath_expected_times": subpath_expected_times,
            "subpath_current_idx": subpath_current_idx,
            "current_link_expected_finish_ts": current_link_expected_finish_ts,
            "delay": delay,
        }
    state_entries.sort(
        key=lambda x: (
            f"0:{int(x[0][0]):010d}:{x[0][1]}"
            if isinstance(x[0], tuple) and len(x[0]) == 2 and isinstance(x[0][0], int)
            else (f"0:{int(x[0]):010d}" if isinstance(x[0], int) else f"1:{str(x[0])}"),
            x[1],
        )
    )
    network_state: dict[object, list[tuple[str, str]]] = {}
    for link_id, _begin_ts_s, train_id, service_date in state_entries:
        network_state.setdefault(link_id, []).append((train_id, service_date))

    return train_state, network_state, events_by_key


def simulate_snapshot(
    *,
    snapshot_ts: pd.Timestamp,
    journeys: pd.DataFrame,
    events: pd.DataFrame,
    link_distance_m: dict[int, float],
    link_endpoints: dict[int, tuple[int, int]],
    travel_time_stats: dict,
) -> dict[tuple[str, str], list[float]]:
    """Simulate future train-event progression from one snapshot and return delay forecasts."""
    snapshot_ts = pd.to_datetime(snapshot_ts, errors="coerce")
    snapshot_rows = journeys.copy()
    required_cols = {"line_dep", "line_arr"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"Missing required event columns for transition keys: {sorted(missing)}")
    train_state, network_state, events_by_key = initialize_snapshot_state(
        snapshot_ts=snapshot_ts,
        journeys_filtered=snapshot_rows,
        events_filtered=events,
        link_distance_m=link_distance_m,
        link_endpoints=link_endpoints,
        travel_time_stats=travel_time_stats,
    )
    path_by_key = {
        (str(r.train_id), str(r.service_date)): r.path
        for r in snapshot_rows.itertuples(index=False)
    }
    relation_by_key = {
        (str(r.train_id), str(r.service_date)): str(r.train_relation)
        for r in snapshot_rows.itertuples(index=False)
    }

    predictions_by_key: dict[tuple[str, str], list[float]] = {k: [] for k in train_state}

    def remove_from_network(train_key: tuple[str, str], occ_key: object) -> None:
        """Remove one train from its current network occupancy queue."""
        lst = network_state[occ_key]
        lst.remove(train_key)
        if not lst:
            del network_state[occ_key]

    def apply_precedence_on_link_entry(train_key: tuple[str, str], state: dict[str, object]) -> None:
        """Delay a train if it enters a directed link behind another train."""
        occ = state["subpath"][state["subpath_current_idx"]]
        if not (isinstance(occ, tuple) and len(occ) == 2):
            return
        queue = network_state[occ]
        k = queue.index(train_key)
        if k == 0:
            return
        prev_train = queue[k - 1]
        t_prev = int(train_state[prev_train]["current_link_expected_finish_ts"])
        # Trains on the same directed link are kept at least PRECEDENCE_GAP_S apart.
        constrained_ts = int(t_prev + PRECEDENCE_GAP_S)
        state["current_link_expected_finish_ts"] = int(
            max(int(state["current_link_expected_finish_ts"]), constrained_ts)
        )

    def build_state_for_event_idx(train_key: tuple[str, str], event_idx: int, now_s: int) -> dict[str, object]:
        """Build the simulated state for a train starting a new event-pair transition."""
        events_group = events_by_key[train_key]
        path = path_by_key[train_key]
        relation = relation_by_key[train_key]

        delay = float(train_state[train_key]["delay"])
        n_events = len(events_group["op_id"])
        if event_idx < n_events:
            planned_event_s = int(events_group["planned_s"][event_idx])
            delay = float(now_s - planned_event_s)

        if event_idx >= len(path):
            last_planned_s = int(events_group["planned_s"][-1])
            finish_s = int(np.ceil(last_planned_s + delay + 300))
            return {
                "event_idx": event_idx,
                "subpath": ["ARRIVED"],
                "subpath_expected_times": [300.0],
                "subpath_current_idx": 0,
                "current_link_expected_finish_ts": finish_s,
                "delay": delay,
            }

        raw_subpath = [int(link_id) for link_id in path[event_idx]]
        if len(raw_subpath) == 0:
            stop_op_id = int(events_group["op_id"][event_idx])
            subpath = [f"STOPPED@op{stop_op_id}"]
            planned_delta_sec = float(events_group["planned_s"][event_idx + 1] - events_group["planned_s"][event_idx])
            subpath_expected_times = compute_subpath_link_times(
                u_op_id=int(events_group["op_id"][event_idx]),
                u_event_type=str(events_group["event_type"][event_idx]),
                u_line_dep=str(events_group["line_dep"][event_idx]),
                v_op_id=int(events_group["op_id"][event_idx + 1]),
                v_event_type=str(events_group["event_type"][event_idx + 1]),
                v_line_arr=str(events_group["line_arr"][event_idx + 1]),
                planned_delta_sec=planned_delta_sec,
                subpath=subpath,
                current_delay_sec=delay,
                relation=relation,
                travel_time_stats=travel_time_stats,
                link_distance_m=link_distance_m,
            )
            return {
                "event_idx": event_idx,
                "subpath": subpath,
                "subpath_expected_times": subpath_expected_times,
                "subpath_current_idx": 0,
                "current_link_expected_finish_ts": int(np.ceil(now_s + subpath_expected_times[0])),
                "delay": delay,
            }

        subpath = annotate_subpath_direction(
            subpath=raw_subpath,
            start_op_id=int(events_group["op_id"][event_idx]),
            link_endpoints=link_endpoints,
        )
        planned_delta_sec = float(events_group["planned_s"][event_idx + 1] - events_group["planned_s"][event_idx])
        subpath_expected_times = compute_subpath_link_times(
            u_op_id=int(events_group["op_id"][event_idx]),
            u_event_type=str(events_group["event_type"][event_idx]),
            u_line_dep=str(events_group["line_dep"][event_idx]),
            v_op_id=int(events_group["op_id"][event_idx + 1]),
            v_event_type=str(events_group["event_type"][event_idx + 1]),
            v_line_arr=str(events_group["line_arr"][event_idx + 1]),
            planned_delta_sec=planned_delta_sec,
            subpath=raw_subpath,
            current_delay_sec=delay,
            relation=relation,
            travel_time_stats=travel_time_stats,
            link_distance_m=link_distance_m,
        )
        return {
            "event_idx": event_idx,
            "subpath": subpath,
            "subpath_expected_times": subpath_expected_times,
            "subpath_current_idx": 0,
            "current_link_expected_finish_ts": int(np.ceil(now_s + subpath_expected_times[0])),
            "delay": delay,
        }

    def consume_event(train_key: tuple[str, str], state: dict[str, object], now_s: int) -> dict[str, object]:
        """Advance one train from its current occupancy to the next simulated state."""
        occ = state["subpath"][state["subpath_current_idx"]]
        if occ == "ARRIVED":
            return state
        if occ == "DEPARTING":
            return build_state_for_event_idx(train_key, 0, now_s)
        if isinstance(occ, str) and occ.startswith("STOPPED@op"):
            return build_state_for_event_idx(train_key, int(state["event_idx"]) + 1, now_s)

        cur_idx = int(state["subpath_current_idx"])
        if cur_idx + 1 < len(state["subpath"]):
            new_state = dict(state)
            new_state["subpath_current_idx"] = cur_idx + 1
            new_state["current_link_expected_finish_ts"] = int(
                np.ceil(now_s + float(state["subpath_expected_times"][cur_idx + 1]))
            )
            return new_state
        return build_state_for_event_idx(train_key, int(state["event_idx"]) + 1, now_s)

    for occ, queue in network_state.items():
        if not (isinstance(occ, tuple) and len(occ) == 2):
            continue
        if not queue:
            continue
        for k, train_key in enumerate(queue[1:], start=1):
            prev_train = queue[k - 1]
            t_prev = int(train_state[prev_train]["current_link_expected_finish_ts"])
            constrained_ts = int(t_prev + PRECEDENCE_GAP_S)
            st = train_state[train_key]
            st["current_link_expected_finish_ts"] = int(
                max(int(st["current_link_expected_finish_ts"]), constrained_ts)
            )

    heap: list[tuple[int, tuple[str, str]]] = []
    for key in train_state:
        t = int(train_state[key]["current_link_expected_finish_ts"])
        heapq.heappush(heap, (t, key))

    while heap:
        t, train_key = heapq.heappop(heap)
        st = train_state[train_key]

        occ = st["subpath"][st["subpath_current_idx"]]
        prev_event_idx = int(st["event_idx"])
        remove_from_network(train_key, occ)
        new_state = consume_event(train_key, st, t)
        train_state[train_key] = new_state
        new_event_idx = int(new_state["event_idx"])
        if new_event_idx > prev_event_idx:
            # A prediction is recorded whenever the simulated train reaches the next event.
            predictions_by_key[train_key].append(float(new_state["delay"]))

        new_occ = new_state["subpath"][new_state["subpath_current_idx"]]
        network_state.setdefault(new_occ, []).append(train_key)
        apply_precedence_on_link_entry(train_key, new_state)
        if new_occ == "ARRIVED":
            continue

        heapq.heappush(heap, (int(new_state["current_link_expected_finish_ts"]), train_key))

    return predictions_by_key


def create_key_slices(events: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    """Map each train key to its contiguous event slice in a sorted events table."""
    pairs = list(zip(events["train_id"].astype(str), events["service_date"].astype(str)))
    key_slices = {}
    s = 0
    prev = pairs[0]
    for i in range(1, len(pairs)):
        if pairs[i] != prev:
            key_slices[prev] = (s, i)
            s, prev = i, pairs[i]
    key_slices[prev] = (s, len(pairs))
    return key_slices


def build_prediction_table(
    *,
    prediction_rows: list[tuple[pd.Timestamp, str, str, list[float]]],
    eval_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert per-train prediction lists into the benchmark wide prediction table."""
    if eval_table is not None:
        target_cols = sorted(
            [c for c in eval_table.columns if c.startswith("future_delay_")],
            key=lambda c: int(c.split("_")[-1]),
        )
    else:
        k = max((len(preds) for _, _, _, preds in prediction_rows), default=0)
        target_cols = [f"future_delay_{i}" for i in range(1, k + 1)]

    records: list[dict[str, object]] = []
    for ts, train_id, service_date, preds in prediction_rows:
        row: dict[str, object] = {
            "ts": ts,
            "train_id": train_id,
            "service_date": service_date,
        }
        for i, col in enumerate(target_cols):
            row[col] = float(preds[i]) if i < len(preds) else np.nan
        records.append(row)
    return pd.DataFrame.from_records(records)


def run_graph_events(
    *,
    journeys: pd.DataFrame,
    events: pd.DataFrame,
    node_links: pd.DataFrame,
    travel_time_stats: dict,
    eval_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run graph-event simulation for each snapshot and return benchmark predictions."""
    journeys = journeys.copy()
    events = events.copy()
    journeys["ts"] = pd.to_datetime(journeys["ts"], errors="coerce")
    journeys["train_id"] = journeys["train_id"].astype(str)
    journeys["service_date"] = journeys["service_date"].astype(str)
    events["train_id"] = events["train_id"].astype(str)
    events["service_date"] = events["service_date"].astype(str)
    journeys = journeys.dropna(subset=["ts"])
    link_distance_m: dict[int, float] = {
        int(r.link_id): float(r.distance_m)
        for r in node_links[["link_id", "distance_m"]].itertuples(index=False)
    }
    link_endpoints: dict[int, tuple[int, int]] = {
        int(r.link_id): (int(r.u_node_id), int(r.v_node_id))
        for r in node_links[["link_id", "u_node_id", "v_node_id"]].itertuples(index=False)
    }

    key_slices = create_key_slices(events)
    prediction_rows: list[tuple[pd.Timestamp, str, str, list[float]]] = []
    groups = journeys.groupby("ts", sort=True)
    for snapshot_ts, journeys_filtered in tqdm(
        groups,
        total=int(journeys["ts"].nunique()),
        desc="Simulating snapshots",
    ):
        snapshot_ts = pd.Timestamp(snapshot_ts)
        keys = list(set(zip(journeys_filtered["train_id"].astype(str), journeys_filtered["service_date"].astype(str))))
        events_filtered = pd.concat(
            [events.iloc[s:e] for (s, e) in (key_slices[k] for k in keys)],
            ignore_index=True,
        )

        predictions_by_key = simulate_snapshot(
            snapshot_ts=snapshot_ts,
            journeys=journeys_filtered,
            events=events_filtered,
            link_distance_m=link_distance_m,
            link_endpoints=link_endpoints,
            travel_time_stats=travel_time_stats,
        )
        for row in journeys_filtered.itertuples(index=False):
            key = (str(row.train_id), str(row.service_date))
            prediction_rows.append((snapshot_ts, key[0], key[1], predictions_by_key[key]))
    return build_prediction_table(prediction_rows=prediction_rows, eval_table=eval_table)
