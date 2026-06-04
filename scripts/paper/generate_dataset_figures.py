"""Generate dataset figures used by the paper.

It writes figures to ``tables_figures/figures/dataset`` by default.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import colormaps, gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

try:
    import contextily as ctx
except Exception:  # pragma: no cover - figure generation still works offline.
    ctx = None


PAPER_WIDTH = 6.8
GRID_COLOR = "0.88"
FIGURE_MATPLOTLIB_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
}

REL_STS = ("station", "to", "station")
REL_PAST = ("train", "past", "station")
REL_FUTURE = ("train", "future", "station")

N_FUTURE = 15
HORIZON_EDGES_MIN = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, np.inf], dtype=np.float64)
HORIZON_LABELS_SHORT = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45+"]
HORIZON_LABELS_BOX = ["0:5", "5:10", "10:15", "15:20", "20:25", "25:30", "30:35", "35:40", "40:45", "45+"]
DELAY_DELTA_EDGES_MIN = np.array([-np.inf, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, np.inf], dtype=np.float64)
DELAY_DELTA_LABELS_SHORT = ["<-5", "-5:-2", "-2:-1", "-1:-0.5", "-0.5:0", "0:0.5", "0.5:1", "1:2", "2:5", "5:10", "10+"]

WEATHER_GROUPS = {
    "clear": {"codes": {0, 1, 2}, "color": "#F4D35E"},
    "cloud": {"codes": {3}, "color": "#9AA5B1"},
    "fog": {"codes": {45, 48}, "color": "#7B8EA3"},
    "drizzle": {"codes": {51, 53, 55, 56, 57}, "color": "#66C2D7"},
    "rain": {"codes": {61, 63, 65, 66, 67, 80, 81, 82}, "color": "#2C7FB8"},
    "snow": {"codes": {71, 73, 75, 77, 85, 86}, "color": "#A5C8FF"},
    "thunder": {"codes": {95, 96, 99}, "color": "#7A3DB8"},
}


@dataclass(frozen=True)
class FigureSpec:
    label: str
    outputs: tuple[str, ...]
    description: str
    generator: str


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("fig:snapshot-fig", ("network_snapshot.svg",), "Paper Figure 2: Visualization of the railway network for the snapshot at 2024-01-15 08:00:00.", "plot_network_snapshot"),
    FigureSpec("fig:full-network-visualization", ("network_visualization.svg",), "Paper Figure 3: Visualization of the railway network.", "plot_network_visualization"),
    FigureSpec("fig:silver_illus_example", ("silver_illus_example_map.pdf",), "Paper Figure 4: Illustrative example of silver tables for a single journey, with a map view of the corresponding sub-network.", "plot_silver_illus_example_map"),
    FigureSpec("fig:journey-time-station-plot", ("journey_time_station_plot.svg",), "Paper Figure 5: Visualization of the event sequence for one train on a given service date.", "plot_journey_time_station"),
    FigureSpec("fig:op-nodes-descriptive-stats", ("op_nodes_descriptive_stats.svg",), "Paper Figure 6: Descriptive statistics for the silver op_nodes table.", "plot_op_nodes_descriptive_stats"),
    FigureSpec("fig:line-sections-descriptive-stats", ("line_sections_descriptive_stats.svg",), "Paper Figure 7: Descriptive statistics for the silver line_sections table.", "plot_line_sections_descriptive_stats"),
    FigureSpec("fig:compare-event-vs-infra-graph", ("compare_event_vs_infra_graph.svg",), "Paper Figure 8: Comparison between an event-derived railway graph and the infrastructure-derived graph used in RIDE.", "plot_compare_event_vs_infra_graph"),
    FigureSpec("fig:node-links-descriptive-stats", ("node_links_descriptive_stats.svg",), "Paper Figure 9: Descriptive statistics for the silver node_links table.", "plot_node_links_descriptive_stats"),
    FigureSpec("fig:node-links-overlap-zoom", ("node_links_overlap_zoom.svg",), "Paper Figure 10: Local zoom on the operational-point pair with the highest node-link multiplicity.", "plot_node_links_overlap_zoom"),
    FigureSpec("fig:events-descriptive-stats", ("events_descriptive_stats.svg",), "Paper Figure 13: Descriptive statistics for the silver events table.", "plot_events_descriptive_stats"),
    FigureSpec("fig:journey-illustration", ("journey_illustration.svg",), "Paper Figure 15: Illustrative example of a journey-level inferred path.", "plot_journey_illustration"),
    FigureSpec("fig:journeys-descriptive-stats", ("journeys_descriptive_stats.svg",), "Paper Figure 16: Descriptive statistics for the silver journeys table.", "plot_journeys_descriptive_stats"),
    FigureSpec("fig:silver-weather-snapshot", ("weather_snapshot_2024-01-01_02-00-00.svg",), "Paper Figure 17: Illustrative weather snapshot from the silver weather table.", "plot_weather_snapshot"),
    FigureSpec("fig:weather-descriptive-stats", ("weather_descriptive_stats.svg",), "Paper Figure 18: Descriptive statistics for the silver weather table.", "plot_weather_descriptive_stats"),
    FigureSpec("fig:gold_station_embeddings", ("gold_station_embeddings.svg",), "Paper Figure 19: The 8 normalized-Laplacian embedding components used as operational-point features in gold datasets.", "plot_gold_station_embeddings"),
    FigureSpec("fig:gnn_snapshot_country", ("gnn_full_graph_illus.svg",), "Paper Figure 21: Visualization of a full-network GNN snapshot.", "plot_gnn_full_graph_illus"),
    FigureSpec("fig:overall_horizon_bin_counts", ("standard_overall_horizon_bin_counts.svg",), "Paper Figure 22: Overall distribution of valid targets across horizon bins on the standard tier.", "plot_standard_eval_bin_distributions"),
    FigureSpec("fig:horizon_bin_counts_by_future_event", ("standard_horizon_bin_counts_by_future_event.svg",), "Paper Figure 23: Distribution of valid targets across horizon bins broken down by future-event slot on the standard tier.", "plot_standard_eval_bin_distributions"),
    FigureSpec("fig:standard_delay_delta_by_horizon_boxplot", ("standard_delay_delta_by_horizon_boxplot.svg",), "Paper Figure 24: Delay-delta distribution by horizon bin on the standard tier.", "plot_standard_delay_delta_by_horizon_boxplot"),
    FigureSpec("fig:overall_delay_delta_bin_counts", ("standard_overall_delay_delta_bin_counts.svg",), "Paper Figure 25: Overall distribution of valid targets across delay-delta bins on the standard tier.", "plot_standard_eval_bin_distributions"),
    FigureSpec("fig:delay_delta_bin_counts_by_future_event", ("standard_delay_delta_bin_counts_by_future_event.svg",), "Paper Figure 26: Distribution of valid targets across delay-delta bins broken down by future-event slot on the standard tier.", "plot_standard_eval_bin_distributions"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver"), help="Silver dataset root.")
    parser.add_argument("--gold-standard-dir", type=Path, default=Path("data/gold/standard"), help="Gold Standard dataset root.")
    parser.add_argument("--gold-lite-dir", type=Path, default=Path("data/gold/lite"), help="Gold Lite dataset root, used for fallback examples.")
    parser.add_argument("--output-dir", type=Path, default=Path("tables_figures"), help="Root directory where figure artifacts are written.")
    parser.add_argument("--gnn-tier", choices=["standard", "lite"], default="standard", help="Gold tier used for the GNN snapshot figure.")
    parser.add_argument("--list", action="store_true", help="List paper labels, figure numbers, captions, and output filenames without generating figures.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional output stems or paper labels to generate.")
    return parser.parse_args()


def print_figure_mapping() -> None:
    for spec in FIGURES:
        outputs = ", ".join(spec.outputs)
        print(f"{spec.label}: {outputs}")
        print(f"  {spec.description}")


def configure_plots() -> None:
    plt.rcParams.update(
        {
            **FIGURE_MATPLOTLIB_RC,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, filename: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / filename, bbox_inches="tight", facecolor="white")


def safe_add_basemap(ax: plt.Axes, *, labels: bool = True, labels_alpha: float = 0.7, zorder: int = 0) -> None:
    if ctx is None:
        return
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.PositronNoLabels, zorder=zorder)
        if labels:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.PositronOnlyLabels, alpha=labels_alpha, zorder=zorder)
    except Exception as exc:
        print(f"Basemap unavailable: {exc}")


def parse_coords(geoshape: object) -> np.ndarray | None:
    coords = json.loads(geoshape).get("coordinates", []) if isinstance(geoshape, str) else []
    if len(coords) < 2:
        return None
    return np.asarray(coords, dtype="float64")


def matched_positions(x: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, np.ndarray):
        arr = x.astype("float64", copy=False)
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype="float64")
    elif isinstance(x, str):
        try:
            arr = np.asarray(ast.literal_eval(x), dtype="float64")
        except Exception:
            return np.array([], dtype=int), np.array([], dtype=int)
    else:
        return np.array([], dtype=int), np.array([], dtype=int)
    finite_pos = np.where(np.isfinite(arr))[0]
    return finite_pos, arr[finite_pos].astype(int, copy=False)


def haversine_line_km(geoshape: object) -> float:
    coords = parse_coords(geoshape)
    if coords is None:
        return np.nan
    lon1 = np.radians(coords[:-1, 0])
    lat1 = np.radians(coords[:-1, 1])
    lon2 = np.radians(coords[1:, 0])
    lat2 = np.radians(coords[1:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float((6371000.0 * (2.0 * np.arcsin(np.sqrt(a)))).sum() / 1000.0)


class DatasetContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.silver_dir = args.silver_dir
        self.gold_standard_dir = args.gold_standard_dir
        self.gold_lite_dir = args.gold_lite_dir
        self.gnn_tier = args.gnn_tier
        self._cache: dict[tuple[str, object], object] = {}

    @property
    def static_dir(self) -> Path:
        return self.silver_dir / "static"

    @property
    def events_dir(self) -> Path:
        return self.silver_dir / "events"

    @property
    def journeys_dir(self) -> Path:
        return self.silver_dir / "journeys"

    def op_nodes(self, columns: list[str] | None = None) -> pd.DataFrame:
        key = ("op_nodes", tuple(columns) if columns else None)
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.static_dir / "op_nodes.parquet", columns=columns)
        return self._cache[key]  # type: ignore[return-value]

    def line_sections(self, columns: list[str] | None = None) -> pd.DataFrame:
        key = ("line_sections", tuple(columns) if columns else None)
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.static_dir / "line_sections.parquet", columns=columns)
        return self._cache[key]  # type: ignore[return-value]

    def node_links(self, columns: list[str] | None = None) -> pd.DataFrame:
        key = ("node_links", tuple(columns) if columns else None)
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.static_dir / "node_links.parquet", columns=columns)
        return self._cache[key]  # type: ignore[return-value]

    def weather(self, columns: list[str]) -> pd.DataFrame:
        key = ("weather", tuple(columns))
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.static_dir / "weather.parquet", columns=columns)
        return self._cache[key]  # type: ignore[return-value]

    def events_month(self, month: str, columns: list[str]) -> pd.DataFrame:
        key = ("events_month", month, tuple(columns))
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.events_dir / f"events_{month}.parquet", columns=columns)
        return self._cache[key]  # type: ignore[return-value]

    def journeys_month(self, month: str, columns: list[str] | None = None) -> pd.DataFrame:
        key = ("journeys_month", month, tuple(columns) if columns else None)
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.journeys_dir / f"journeys_{month}.parquet", columns=columns)
        return self._cache[key]  # type: ignore[return-value]

    def line_geoms(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        key = ("line_geoms", None)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]
        geoms = {}
        for row in self.line_sections(["line_section_id", "geoshape"]).itertuples(index=False):
            coords = parse_coords(row.geoshape)
            if coords is None:
                continue
            diffs = coords[1:] - coords[:-1]
            seg_len = np.linalg.norm(diffs, axis=1)
            total = float(seg_len.sum())
            if total <= 0:
                continue
            geoms[int(row.line_section_id)] = (coords, seg_len, np.concatenate(([0.0], np.cumsum(seg_len))), total)
        self._cache[key] = geoms
        return geoms

    def link_lookup(self) -> dict[frozenset[int], object]:
        key = ("link_lookup", None)
        if key not in self._cache:
            lookup = {}
            for row in self.node_links().itertuples(index=False):
                lookup[frozenset({int(row.u_node_id), int(row.v_node_id)})] = row
            self._cache[key] = lookup
        return self._cache[key]  # type: ignore[return-value]

    def link_segments(self) -> dict[int, np.ndarray]:
        key = ("link_segments", None)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]
        node_links = self.node_links()
        sections = self.line_sections(["line_section_id", "geoshape", "matched_op_node_ids"])
        section_lookup = {int(row.line_section_id): row for row in sections.itertuples(index=False)}
        segments = {}
        for row in node_links.itertuples(index=False):
            sec = section_lookup.get(int(row.line_section_id))
            if sec is None:
                continue
            coords = parse_coords(sec.geoshape)
            if coords is None:
                continue
            finite_pos, finite_ids = matched_positions(sec.matched_op_node_ids)
            if len(finite_ids) < 2:
                continue
            u = int(row.u_node_id)
            v = int(row.v_node_id)
            u_matches = np.where(finite_ids == u)[0]
            v_matches = np.where(finite_ids == v)[0]
            if len(u_matches) == 0 or len(v_matches) == 0:
                continue
            start = min(int(finite_pos[u_matches[0]]), int(finite_pos[v_matches[0]]))
            end = max(int(finite_pos[u_matches[0]]), int(finite_pos[v_matches[0]]))
            if end >= len(coords):
                continue
            seg = coords[start : end + 1]
            if len(seg) >= 2:
                segments[int(row.link_id)] = seg
        self._cache[key] = segments
        return segments

    def standard_eval_table_path(self) -> Path:
        return self.gold_standard_dir / "core" / "test_eval_table.parquet"

    def gnn_dir(self) -> Path:
        root = self.gold_standard_dir if self.gnn_tier == "standard" else self.gold_lite_dir
        return root / "gnn"


def interpolate_on_line(line_geom: tuple[np.ndarray, np.ndarray, np.ndarray, float], fraction: float) -> np.ndarray:
    coords, seg_len, cum, total = line_geom
    target = fraction * total
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = int(np.clip(idx, 0, len(seg_len) - 1))
    denom = seg_len[idx]
    if denom == 0:
        return coords[idx + 1]
    t = (target - cum[idx]) / denom
    return coords[idx] + t * (coords[idx + 1] - coords[idx])


def compute_train_positions(context: DatasetContext, snapshot_ts: pd.Timestamp, window: str = "2h") -> pd.DataFrame:
    events = context.events_month(
        f"{snapshot_ts:%Y%m}",
        ["train_id", "service_date", "op_id", "observed_ts"],
    ).copy()
    events["observed_ts"] = pd.to_datetime(events["observed_ts"])
    window_delta = pd.to_timedelta(window)
    mask = (events["observed_ts"] >= snapshot_ts - window_delta) & (events["observed_ts"] <= snapshot_ts + window_delta)
    subset = events.loc[mask].copy()
    keys = ["train_id", "service_date"]
    before = (
        subset[subset["observed_ts"] <= snapshot_ts]
        .sort_values(keys + ["observed_ts"])
        .groupby(keys)
        .tail(1)
        .rename(columns={"op_id": "op_id_prev", "observed_ts": "ts_prev"})
    )
    after = (
        subset[subset["observed_ts"] >= snapshot_ts]
        .sort_values(keys + ["observed_ts"])
        .groupby(keys)
        .head(1)
        .rename(columns={"op_id": "op_id_next", "observed_ts": "ts_next"})
    )
    pairs = (
        before[keys + ["op_id_prev", "ts_prev"]]
        .merge(after[keys + ["op_id_next", "ts_next"]], on=keys, how="inner")
        .query("ts_prev < @snapshot_ts and ts_next > @snapshot_ts and op_id_prev != op_id_next")
    )
    points = []
    line_geoms = context.line_geoms()
    link_lookup = context.link_lookup()
    for row in pairs.itertuples(index=False):
        u, v = int(row.op_id_prev), int(row.op_id_next)
        link = link_lookup.get(frozenset({u, v}))
        if link is None:
            continue
        geom = line_geoms.get(int(link.line_section_id))
        if geom is None:
            continue
        total = (row.ts_next - row.ts_prev).total_seconds()
        if total <= 0:
            continue
        frac = np.clip((snapshot_ts - row.ts_prev).total_seconds() / total, 0.0, 1.0)
        if u != int(link.u_node_id):
            frac = 1.0 - frac
        lon, lat = interpolate_on_line(geom, float(frac))
        points.append({"train_id": row.train_id, "service_date": row.service_date, "lon": lon, "lat": lat})
    return pd.DataFrame(points)


# Paper Figure 2, fig:snapshot-fig -> network_snapshot.svg.
# Caption: Visualization of the railway network for the snapshot at 2024-01-15 08:00:00.
def plot_network_snapshot(context: DatasetContext, figure_dir: Path) -> None:
    snapshot_ts = pd.Timestamp("2024-01-15 08:00:00")
    op_nodes = context.op_nodes()
    line_sections = context.line_sections(["geoshape"])
    train_positions = compute_train_positions(context, snapshot_ts)
    events = context.events_month(f"{snapshot_ts:%Y%m}", ["op_id"])
    active_op_ids = pd.Series(events["op_id"].dropna().unique(), dtype="int64")
    op_nodes_active = op_nodes[op_nodes["op_id"].isin(active_op_ids)]

    fig, ax = plt.subplots(figsize=(15, 15), dpi=120)
    for geoshape in line_sections["geoshape"].dropna():
        coords = parse_coords(geoshape)
        if coords is not None:
            ax.plot(coords[:, 0], coords[:, 1], color="#023D5B", linewidth=0.5, zorder=1)
    ax.scatter(op_nodes_active["lon"], op_nodes_active["lat"], s=5, c="#023D5B", zorder=2)
    if not train_positions.empty:
        ax.scatter(train_positions["lon"], train_positions["lat"], s=10, marker="s", c="crimson", linewidth=0.5, zorder=3)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(op_nodes_active["lat"].mean()))), adjustable="box")
    safe_add_basemap(ax)
    ax.legend(
        handles=[
            Line2D([], [], color="#023D5B", linewidth=2, label="rail tracks"),
            Line2D([], [], marker="o", markersize=6, linestyle="None", color="#023D5B", label="operational points"),
            Line2D([], [], marker="s", markersize=10, linestyle="None", markerfacecolor="crimson", markeredgecolor="none", label="trains"),
        ],
        loc="upper right",
        fontsize=15,
    )
    ax.axis("off")
    save_figure(fig, figure_dir, "network_snapshot.svg")
    plt.close(fig)


# Paper Figure 3, fig:full-network-visualization -> network_visualization.svg.
# Caption: Visualization of the railway network.
def plot_network_visualization(context: DatasetContext, figure_dir: Path) -> None:
    snapshot_ts = pd.Timestamp("2024-01-15 08:00:00")
    op_nodes = context.op_nodes()
    line_sections = context.line_sections(["geoshape"])
    events = context.events_month(f"{snapshot_ts:%Y%m}", ["op_id"])
    active_op_ids = pd.Series(events["op_id"].dropna().unique(), dtype="int64")
    op_nodes_active = op_nodes[op_nodes["op_id"].isin(active_op_ids)]

    fig, ax = plt.subplots(figsize=(15, 15), dpi=120)
    for geoshape in line_sections["geoshape"].dropna():
        coords = parse_coords(geoshape)
        if coords is not None:
            ax.plot(coords[:, 0], coords[:, 1], color="#023D5B", linewidth=0.5, zorder=1)
    ax.scatter(op_nodes_active["lon"], op_nodes_active["lat"], s=5, c="#023D5B", zorder=2)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(op_nodes_active["lat"].mean()))), adjustable="box")
    safe_add_basemap(ax)
    ax.legend(
        handles=[
            Line2D([], [], color="#023D5B", linewidth=0.5, label="rail tracks"),
            Line2D([], [], marker="o", markersize=6, linestyle="None", color="#023D5B", label="operational points"),
        ],
        loc="upper right",
        fontsize=15,
    )
    ax.axis("off")
    save_figure(fig, figure_dir, "network_visualization.svg")
    plt.close(fig)


# Paper Figure 4, fig:silver_illus_example -> silver_illus_example_map.pdf.
# Caption: Illustrative example of silver tables for a single journey, with a map view of the corresponding sub-network.
def plot_silver_illus_example_map(context: DatasetContext, figure_dir: Path) -> None:
    target_line_section_id = "1297"
    target_op_ids = [818, 1413, 788, 801]
    op_nodes = context.op_nodes()
    line_sections = context.line_sections()
    node_links = context.node_links()
    target_links = node_links.loc[node_links["line_section_id"] == target_line_section_id].copy()
    target_nodes = op_nodes.loc[op_nodes["op_id"].isin(target_op_ids)].set_index("op_id").loc[target_op_ids].reset_index()
    min_lon = target_nodes["lon"].min() - 0.03
    max_lon = target_nodes["lon"].max() + 0.03
    min_lat = target_nodes["lat"].min() - 0.02
    max_lat = target_nodes["lat"].max() + 0.02
    nearby_sections = []
    for row in line_sections.itertuples(index=False):
        coords = parse_coords(row.geoshape)
        if coords is None:
            continue
        if not (coords[:, 0].max() < min_lon or coords[:, 0].min() > max_lon or coords[:, 1].max() < min_lat or coords[:, 1].min() > max_lat):
            nearby_sections.append((row.line_section_id, coords))
    other_nodes = op_nodes.loc[op_nodes["lon"].between(min_lon, max_lon) & op_nodes["lat"].between(min_lat, max_lat)].copy()
    other_nodes = other_nodes.loc[~other_nodes["op_id"].isin(target_op_ids)]
    label_offsets = {818: (-0.0430, 0.0040), 1413: (-0.0750, 0.0040), 788: (0.0025, -0.0045), 801: (0.0025, -0.0045)}
    link_offsets = {352: (0.0080, -0.0025), 353: (0.0010, -0.0015), 354: (0.0010, -0.0035)}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    for section_id, coords in nearby_sections:
        if str(section_id) == target_line_section_id:
            ax.plot(coords[:, 0], coords[:, 1], color="#D1495B", linewidth=3.0, zorder=2)
        else:
            ax.plot(coords[:, 0], coords[:, 1], color="#9AA5B1", linewidth=1.0, alpha=0.7, zorder=1)
    if not other_nodes.empty:
        ax.scatter(other_nodes["lon"], other_nodes["lat"], s=18, c="#A9B6C2", zorder=3)
    ax.scatter(target_nodes["lon"], target_nodes["lat"], s=50, c="#023D5B", edgecolors="white", linewidths=0.8, zorder=4)
    for row in target_nodes.itertuples(index=False):
        dx, dy = label_offsets[int(row.op_id)]
        ax.text(row.lon + dx, row.lat + dy, f"{row.op_name} ({row.op_id})", fontsize=9, color="#023D5B", weight="bold", zorder=5)
    for row in target_links.itertuples(index=False):
        u = target_nodes.loc[target_nodes["op_id"] == int(row.u_node_id)].iloc[0]
        v = target_nodes.loc[target_nodes["op_id"] == int(row.v_node_id)].iloc[0]
        dx, dy = link_offsets[int(row.link_id)]
        ax.text(
            0.5 * (u.lon + v.lon) + dx,
            0.5 * (u.lat + v.lat) + dy,
            f"link {row.link_id}",
            fontsize=10.5,
            color="#000000",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=6,
        )
    ax.set_aspect(1 / np.cos(np.deg2rad(float(target_nodes["lat"].mean()))), adjustable="box")
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    safe_add_basemap(ax)
    ax.legend(
        handles=[
            Line2D([], [], color="#D1495B", linewidth=3.0, label="line section 1297"),
            Line2D([], [], color="#9AA5B1", linewidth=1.0, label="other nearby line sections"),
            Line2D([], [], marker="o", markersize=7, linestyle="None", markerfacecolor="#023D5B", markeredgecolor="white", label="operational points of interest"),
            Line2D([], [], marker="o", markersize=7, linestyle="None", markerfacecolor="#A9B6C2", markeredgecolor="white", label="other nearby operational points"),
        ],
        loc="upper left",
        fontsize=9,
        frameon=True,
    )
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, figure_dir, "silver_illus_example_map.pdf")
    plt.close(fig)


# Paper Figure 5, fig:journey-time-station-plot -> journey_time_station_plot.svg.
# Caption: Visualization of the event sequence for one train on a given service date.
def plot_journey_time_station(context: DatasetContext, figure_dir: Path) -> None:
    month = "202408"
    train_id = "5105"
    service_date = "29AUG2024"
    events = context.events_month(month, ["train_id", "service_date", "op_id", "event_type", "planned_ts", "observed_ts", "delay_sec"])
    op_nodes = context.op_nodes(["op_id", "op_name"])
    journey = events.loc[(events["train_id"].astype(str) == train_id) & (events["service_date"].astype(str) == service_date)].copy()
    if journey.empty:
        raise ValueError(f"No events found for train_id={train_id}, service_date={service_date}.")
    journey = journey.merge(op_nodes, on="op_id", how="left").sort_values("planned_ts").reset_index(drop=True)
    station_order = journey.drop_duplicates("op_id")[["op_id", "op_name"]].reset_index(drop=True)
    station_order["y"] = range(len(station_order))
    journey["y"] = journey["op_id"].map(dict(zip(station_order["op_id"], station_order["y"])))

    height = max(3.2, 0.20 * len(station_order) + 1.1)
    fig, ax = plt.subplots(figsize=(6.5, height), dpi=160)
    ax.plot(journey["planned_ts"], journey["y"], color="#2f6f9f", linewidth=1.9, label="scheduled events times", zorder=3, alpha=0.7)
    ax.plot(journey["observed_ts"], journey["y"], color="#d05a3a", linewidth=1.9, label="observed events times", zorder=4, alpha=0.7)
    for _, row in journey.iterrows():
        ax.plot([row["planned_ts"], row["observed_ts"]], [row["y"], row["y"]], color="0.72", linewidth=0.65, alpha=0.75, zorder=2)
    ax.invert_yaxis()
    ax.set_yticks(station_order["y"])
    ax.set_yticklabels(station_order["op_name"].str.title())
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Operation point")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.grid(axis="y", color="0.93", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", handlelength=2.6)
    ax.margins(x=0.03, y=0.02)
    fig.tight_layout(pad=0.2)
    save_figure(fig, figure_dir, "journey_time_station_plot.svg")
    plt.close(fig)


def stream_event_op_stats(context: DatasetContext) -> pd.DataFrame:
    daily_parts = []
    delay_sum = pd.Series(dtype="float64")
    delay_count = pd.Series(dtype="int64")
    for path in sorted(context.events_dir.glob("events_*.parquet")):
        df = pd.read_parquet(path, columns=["service_date", "op_id", "delay_sec"]).dropna(subset=["service_date", "op_id", "delay_sec"])
        df["op_id"] = df["op_id"].astype(int)
        df["service_date"] = pd.to_datetime(df["service_date"], format="%d%b%Y", errors="coerce")
        df = df.dropna(subset=["service_date"])
        daily_parts.append(df.groupby(["op_id", "service_date"], sort=False).size().reset_index(name="events_in_day"))
        g = df.groupby("op_id", sort=False)["delay_sec"]
        delay_sum = delay_sum.add(g.sum(), fill_value=0)
        delay_count = delay_count.add(g.count(), fill_value=0)
    daily_counts = pd.concat(daily_parts, ignore_index=True)
    avg_events = daily_counts.groupby("op_id", sort=False)["events_in_day"].mean().reset_index(name="avg_events_active_day")
    avg_delay = (delay_sum / delay_count).div(60.0).rename("avg_delay_min").reset_index().rename(columns={"index": "op_id"})
    return avg_events.merge(avg_delay, on="op_id", how="outer")


# Paper Figure 6, fig:op-nodes-descriptive-stats -> op_nodes_descriptive_stats.svg.
# Caption: Descriptive statistics for the silver op_nodes table.
def plot_op_nodes_descriptive_stats(context: DatasetContext, figure_dir: Path) -> None:
    op_nodes = context.op_nodes(["op_id", "lat", "lon", "op_name"])
    line_sections = context.line_sections(["geoshape"])
    op_stats = op_nodes.merge(stream_event_op_stats(context), on="op_id", how="left")
    activity_norm = Normalize(vmin=0, vmax=float(op_stats["avg_events_active_day"].quantile(0.99)))
    delay_norm = Normalize(vmin=float(op_stats["avg_delay_min"].quantile(0.01)), vmax=float(op_stats["avg_delay_min"].quantile(0.99)))
    aspect = 1 / np.cos(np.deg2rad(float(op_stats["lat"].mean())))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=180)
    for ax in axes:
        for geoshape in line_sections["geoshape"].dropna():
            coords = parse_coords(geoshape)
            if coords is not None:
                ax.plot(coords[:, 0], coords[:, 1], color="#B8C4CC", linewidth=0.45, alpha=0.8, zorder=1)
        ax.set_aspect(aspect, adjustable="box")
        ax.axis("off")
        safe_add_basemap(ax, zorder=0)
    sc1 = axes[0].scatter(op_stats["lon"], op_stats["lat"], c=op_stats["avg_events_active_day"], cmap="YlOrRd", norm=activity_norm, s=10, edgecolors="none", zorder=2, alpha=0.8)
    axes[0].set_title("Average Number of Events per Active Day")
    fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.02).set_label("Number of events per active day")
    sc2 = axes[1].scatter(op_stats["lon"], op_stats["lat"], c=op_stats["avg_delay_min"], cmap="YlOrRd", norm=delay_norm, s=10, edgecolors="none", zorder=2, alpha=0.8)
    axes[1].set_title("Average Delay per Operational Point")
    fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.02).set_label("Average delay (minutes)")
    fig.tight_layout()
    save_figure(fig, figure_dir, "op_nodes_descriptive_stats.svg")
    plt.close(fig)


# Paper Figure 7, fig:line-sections-descriptive-stats -> line_sections_descriptive_stats.svg.
# Caption: Descriptive statistics for the silver line_sections table.
def plot_line_sections_descriptive_stats(context: DatasetContext, figure_dir: Path) -> None:
    sections = context.line_sections(["geoshape", "matched_op_node_ids"])
    count_values = sections["matched_op_node_ids"].map(lambda x: len(matched_positions(x)[1])).dropna().astype(int)
    count_share = count_values.value_counts(normalize=True).sort_index()
    distance_values = sections["geoshape"].map(haversine_line_km).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), dpi=180)
    axes[0].hist(distance_values, bins=50, weights=np.ones_like(distance_values, dtype=float) / len(distance_values), color="#4C78A8", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Cumulative Section Distance")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Share of line sections")
    axes[0].grid(axis="y", color="0.9")
    axes[1].bar(count_share.index, count_share.values, color="#72B7B2", edgecolor="white", linewidth=0.8, width=0.8)
    axes[1].set_title("Matched Operational Points per Section")
    axes[1].set_xlabel("Number of matched operational points")
    axes[1].set_ylabel("Share of line sections")
    axes[1].grid(axis="y", color="0.9")
    axes[1].set_xticks(count_share.index)
    fig.tight_layout()
    save_figure(fig, figure_dir, "line_sections_descriptive_stats.svg")
    plt.close(fig)


def build_event_edge_counter(events_dir: Path) -> Counter:
    edge_counter: Counter = Counter()
    for path in sorted(events_dir.glob("events_*.parquet")):
        df = pd.read_parquet(path, columns=["train_id", "service_date", "op_id"]).dropna(subset=["op_id"]).copy()
        df["op_id"] = df["op_id"].astype(int)
        prev_op = df.groupby(["train_id", "service_date"], sort=False)["op_id"].shift()
        mask = prev_op.notna() & (prev_op != df["op_id"])
        for u, v in zip(prev_op.loc[mask].astype(int).to_numpy(), df.loc[mask, "op_id"].to_numpy()):
            edge_counter[tuple(sorted((int(u), int(v))))] += 1
    return edge_counter


# Paper Figure 8, fig:compare-event-vs-infra-graph -> compare_event_vs_infra_graph.svg.
# Caption: Comparison between an event-derived railway graph and the infrastructure-derived graph used in RIDE.
def plot_compare_event_vs_infra_graph(context: DatasetContext, figure_dir: Path) -> None:
    op_nodes = context.op_nodes()
    line_sections = context.line_sections(["geoshape"])
    coord_map = {int(row.op_id): (float(row.lon), float(row.lat)) for row in op_nodes.itertuples(index=False)}
    event_segments = [[coord_map[u], coord_map[v]] for (u, v) in build_event_edge_counter(context.events_dir) if u in coord_map and v in coord_map]
    infra_segments = [coords for coords in (parse_coords(g) for g in line_sections["geoshape"].dropna()) if coords is not None]
    xlim = (op_nodes["lon"].min() - 0.08, op_nodes["lon"].max() + 0.08)
    ylim = (op_nodes["lat"].min() - 0.05, op_nodes["lat"].max() + 0.05)
    aspect = 1 / np.cos(np.deg2rad(float(op_nodes["lat"].mean())))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=180)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(aspect, adjustable="box")
        safe_add_basemap(ax)
        ax.axis("off")
    axes[0].add_collection(LineCollection(event_segments, colors="#123B5D", linewidths=0.35, alpha=0.5, zorder=1))
    axes[0].scatter(op_nodes["lon"], op_nodes["lat"], s=2.5, c="#123B5D", alpha=0.7, zorder=2)
    axes[0].set_title("Successive-event adjacency", fontsize=12)
    for coords in infra_segments:
        axes[1].plot(coords[:, 0], coords[:, 1], color="#123B5D", linewidth=0.35, alpha=0.8, zorder=1)
    axes[1].scatter(op_nodes["lon"], op_nodes["lat"], s=2.5, c="#123B5D", alpha=0.7, zorder=2)
    axes[1].set_title("Infrastructure-derived network", fontsize=12)
    fig.legend(
        handles=[
            Line2D([], [], color="#123B5D", linewidth=1.5, label="rail tracks"),
            Line2D([], [], marker="o", markersize=5, linestyle="None", color="#123B5D", label="operational points"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    save_figure(fig, figure_dir, "compare_event_vs_infra_graph.svg")
    plt.close(fig)


def stream_link_traversal_stats(context: DatasetContext) -> dict[int, float]:
    per_link_day_counts: defaultdict[int, Counter] = defaultdict(Counter)
    for path in sorted(context.journeys_dir.glob("journeys_*.parquet")):
        df = pd.read_parquet(path, columns=["service_date", "deduced_paths"])
        for row in df.itertuples(index=False):
            day_counter: Counter = Counter()
            if row.deduced_paths is not None:
                for seg in row.deduced_paths:
                    if seg is not None and len(seg) > 0:
                        for link_id in np.asarray(seg, dtype=int):
                            day_counter[int(link_id)] += 1
            for link_id, cnt in day_counter.items():
                per_link_day_counts[link_id][str(row.service_date)] += cnt
    return {link_id: float(np.fromiter(day_counter.values(), dtype="float64").mean()) for link_id, day_counter in per_link_day_counts.items() if day_counter}


# Paper Figure 9, fig:node-links-descriptive-stats -> node_links_descriptive_stats.svg.
# Caption: Descriptive statistics for the silver node_links table.
def plot_node_links_descriptive_stats(context: DatasetContext, figure_dir: Path) -> None:
    node_links = context.node_links()
    link_segments = context.link_segments()
    active_day_avg = stream_link_traversal_stats(context)
    link_stats = node_links[["link_id", "distance_m"]].copy()
    link_stats["distance_km"] = link_stats["distance_m"] / 1000.0
    link_stats["avg_traversals_active_day"] = link_stats["link_id"].map(active_day_avg)
    distance_values = link_stats["distance_km"].dropna()
    activity_values = link_stats["avg_traversals_active_day"].dropna()
    activity_positive = activity_values.loc[activity_values > 0]
    activity_norm = Normalize(vmin=float(activity_positive.min()), vmax=float(activity_positive.max()))
    aspect = 1 / np.cos(np.deg2rad(float(np.mean([seg[:, 1].mean() for seg in link_segments.values()]))))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), dpi=180)
    axes[0].hist(distance_values, bins=50, weights=np.ones_like(distance_values, dtype=float) / len(distance_values), color="#4C78A8", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Node-Link Distance")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Share of node links")
    axes[0].grid(axis="y", color="0.9")
    for seg in link_segments.values():
        axes[1].plot(seg[:, 0], seg[:, 1], color="#D1D5DB", linewidth=0.6, alpha=0.45, zorder=1)
    segments, values = [], []
    for row in link_stats.itertuples(index=False):
        seg = link_segments.get(int(row.link_id))
        if seg is not None and not pd.isna(row.avg_traversals_active_day) and row.avg_traversals_active_day > 0:
            segments.append(seg)
            values.append(float(row.avg_traversals_active_day))
    if segments:
        lc = LineCollection(segments, cmap="YlOrRd", norm=activity_norm, linewidths=1.4, alpha=0.75, zorder=2)
        lc.set_array(np.asarray(values, dtype="float64"))
        axes[1].add_collection(lc)
        fig.colorbar(lc, ax=axes[1], fraction=0.046, pad=0.02).set_label("Average traversals per active day")
    axes[1].set_title("Average Traversals per Active Day")
    axes[1].set_aspect(aspect, adjustable="box")
    axes[1].axis("off")
    safe_add_basemap(axes[1], labels_alpha=0.65)
    fig.tight_layout()
    save_figure(fig, figure_dir, "node_links_descriptive_stats.svg")
    plt.close(fig)


# Paper Figure 10, fig:node-links-overlap-zoom -> node_links_overlap_zoom.svg.
# Caption: Local zoom on the operational-point pair with the highest node-link multiplicity.
def plot_node_links_overlap_zoom(context: DatasetContext, figure_dir: Path) -> None:
    node_links = context.node_links()
    op_nodes = context.op_nodes(["op_id", "lon", "lat", "op_name"])
    link_segments = context.link_segments()
    op_lookup = op_nodes.set_index("op_id")[["lon", "lat", "op_name"]].to_dict("index")
    pair_counts: Counter = Counter()
    pair_to_links: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for row in node_links.itertuples(index=False):
        pair = tuple(sorted((int(row.u_node_id), int(row.v_node_id))))
        pair_counts[pair] += 1
        pair_to_links[pair].append(int(row.link_id))
    best_pair, _ = pair_counts.most_common(1)[0]
    best_links = pair_to_links[best_pair]
    u, v = best_pair
    segments = [link_segments[link_id] for link_id in best_links if link_id in link_segments]
    all_coords = np.vstack(segments)
    xmin, ymin = all_coords[:, 0].min(), all_coords[:, 1].min()
    xmax, ymax = all_coords[:, 0].max(), all_coords[:, 1].max()
    nearby_segments = [
        seg
        for seg in link_segments.values()
        if not (seg[:, 0].max() < xmin - 0.02 or seg[:, 0].min() > xmax + 0.02 or seg[:, 1].max() < ymin - 0.02 or seg[:, 1].min() > ymax + 0.02)
    ]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=90)
    for seg in nearby_segments:
        ax.plot(seg[:, 0], seg[:, 1], color="#111111", linewidth=1.0, alpha=0.9, zorder=1)
    u_info = op_lookup[u]
    v_info = op_lookup[v]
    ax.scatter([u_info["lon"], v_info["lon"]], [u_info["lat"], v_info["lat"]], s=36, color="#111111", edgecolors="white", linewidth=0.8, zorder=3)
    aspect = 1 / np.cos(np.deg2rad(float(all_coords[:, 1].mean())))
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    xspan, yspan = xmax - xmin, ymax - ymin
    half_x = max(xspan / 2.0, (yspan * aspect) / 2.0)
    half_y = max(yspan / 2.0, (xspan / aspect) / 2.0)
    ax.set_xlim(xmid - 1.88 * half_x, xmid + 1.88 * half_x)
    ax.set_ylim(ymid - 1.88 * half_y, ymid + 1.88 * half_y)
    ax.set_aspect(aspect, adjustable="box")
    safe_add_basemap(ax)
    ax.legend(
        handles=[
            Line2D([], [], color="#111111", linewidth=1.0, label="node links"),
            Line2D([], [], marker="o", linestyle="None", markersize=6, color="#111111", markerfacecolor="#111111", markeredgecolor="white", label="operational points"),
        ],
        loc="upper left",
        framealpha=0.95,
    )
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, figure_dir, "node_links_overlap_zoom.svg")
    plt.close(fig)


# Paper Figure 13, fig:events-descriptive-stats -> events_descriptive_stats.svg.
# Caption: Descriptive statistics for the silver events table.
def plot_events_descriptive_stats(context: DatasetContext, figure_dir: Path) -> None:
    delay_min_plot, delay_max_plot = -20, 60
    bins = np.linspace(delay_min_plot, delay_max_plot, 80)
    overall_counts = np.zeros(len(bins) - 1, dtype=float)
    yearly_counts: dict[int, np.ndarray] = {}
    monthly_sum: defaultdict[pd.Timestamp, float] = defaultdict(float)
    monthly_count: defaultdict[pd.Timestamp, int] = defaultdict(int)
    event_type_counts: Counter = Counter()
    for path in sorted(context.events_dir.glob("events_*.parquet")):
        df = pd.read_parquet(path, columns=["service_date", "event_type", "delay_sec"]).dropna(subset=["service_date", "delay_sec", "event_type"]).copy()
        df["service_date"] = pd.to_datetime(df["service_date"], format="%d%b%Y", errors="coerce")
        df = df.dropna(subset=["service_date"])
        df["delay_min"] = df["delay_sec"] / 60.0
        values = df.loc[df["delay_min"].between(delay_min_plot, delay_max_plot), "delay_min"].to_numpy(dtype=float)
        overall_counts += np.histogram(values, bins=bins)[0]
        for year, grp in df.groupby(df["service_date"].dt.year):
            yearly_counts.setdefault(int(year), np.zeros(len(bins) - 1, dtype=float))
            vals = grp.loc[grp["delay_min"].between(delay_min_plot, delay_max_plot), "delay_min"].to_numpy(dtype=float)
            yearly_counts[int(year)] += np.histogram(vals, bins=bins)[0]
        month = df["service_date"].dt.to_period("M").dt.to_timestamp()
        for m, s in df.groupby(month)["delay_min"].sum().items():
            monthly_sum[m] += float(s)
        for m, c in df.groupby(month)["delay_min"].count().items():
            monthly_count[m] += int(c)
        event_type_counts.update(df["event_type"].tolist())
    monthly_delay = pd.DataFrame({"month_start": sorted(monthly_sum)})
    monthly_delay["avg_delay_min"] = monthly_delay["month_start"].map(lambda m: monthly_sum[m] / monthly_count[m])
    event_type_share = pd.Series(event_type_counts, dtype=float).div(sum(event_type_counts.values())).reindex(["A", "D", "P"], fill_value=0)
    event_type_colors = {"A": "#3B82F6", "D": "#F97316", "P": "#10B981"}
    year_colors = {2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), dpi=180)
    (ax1, ax2), (ax3, ax4) = axes
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax1.hist(centers, bins=bins, weights=overall_counts / overall_counts.sum(), color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax1.set_title("Overall Delay Distribution")
    ax1.set_xlabel("Delay (minutes)")
    ax1.set_ylabel("Share of events")
    ax1.grid(axis="y", color="0.9")
    for year in sorted(yearly_counts):
        counts = yearly_counts[year]
        ax2.hist(
            centers,
            bins=bins,
            weights=counts / counts.sum(),
            histtype="step",
            linewidth=2.0,
            color=year_colors.get(year),
            label=str(year),
            alpha=0.6,
        )
    ax2.set_title("Delay Distribution by Year")
    ax2.set_xlabel("Delay (minutes)")
    ax2.set_ylabel("Share of events")
    ax2.set_xlim(delay_min_plot, delay_max_plot)
    ax2.grid(axis="y", color="0.9")
    ax2.legend(frameon=False)
    ax3.plot(monthly_delay["month_start"], monthly_delay["avg_delay_min"], color="#8B5CF6", linewidth=2.0)
    ax3.set_ylim(0, 4)
    ax3.set_title("Average Delay by Month")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Average delay (minutes)")
    ax3.grid(axis="y", color="0.9")
    month_ticks = list(monthly_delay["month_start"].iloc[::4])
    if month_ticks and month_ticks[-1] != monthly_delay["month_start"].iloc[-1]:
        month_ticks.append(monthly_delay["month_start"].iloc[-1])
    ax3.set_xticks(month_ticks)
    ax3.set_xticklabels([d.strftime("%Y-%m") for d in month_ticks], rotation=45, ha="right")
    ax3.set_xlim(monthly_delay["month_start"].min(), monthly_delay["month_start"].max())
    ax4.bar(event_type_share.index, event_type_share.values, color=[event_type_colors[k] for k in event_type_share.index], edgecolor="white", linewidth=0.8)
    ax4.set_title("Event Type Frequencies")
    ax4.set_xlabel("Event type")
    ax4.set_ylabel("Share of events")
    ax4.grid(axis="y", color="0.9")
    fig.tight_layout()
    save_figure(fig, figure_dir, "events_descriptive_stats.svg")
    plt.close(fig)


class GradientLegendHandle:
    def __init__(self, cmap, label: str) -> None:
        self.cmap = cmap
        self._label = label

    def get_label(self) -> str:
        return self._label


class HandlerGradient(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n_steps = 16
        step_width = width / n_steps
        return [
            Rectangle(
                (xdescent + i * step_width, ydescent),
                step_width,
                height * 0.5,
                transform=trans,
                facecolor=orig_handle.cmap(0.05 + 0.90 * i / max(1, n_steps - 1)),
                edgecolor="none",
            )
            for i in range(n_steps)
        ]


# Paper Figure 15, fig:journey-illustration -> journey_illustration.svg.
# Caption: Illustrative example of a journey-level inferred path.
def plot_journey_illustration(context: DatasetContext, figure_dir: Path) -> None:
    train_id, service_date = "14401", "13NOV2025"
    month = pd.to_datetime(service_date, format="%d%b%Y").strftime("%Y%m")
    journeys = context.journeys_month(month)
    row = journeys.loc[(journeys["train_id"].astype(str) == train_id) & (journeys["service_date"].astype(str) == service_date)]
    if row.empty:
        raise ValueError(f"Journey not found: train_id={train_id}, service_date={service_date}")
    row = row.iloc[0]
    op_nodes = context.op_nodes(["op_id", "lon", "lat", "op_name"])
    op_lookup = op_nodes.set_index("op_id")[["lon", "lat", "op_name"]].to_dict("index")
    link_segments = context.link_segments()
    subpaths = [[link_segments[int(link_id)] for link_id in np.asarray(seg, dtype=int) if int(link_id) in link_segments] for seg in row["deduced_paths"] if seg is not None and len(seg) > 0]
    all_coords = np.vstack([seg for sub in subpaths for seg in sub])
    xmin, ymin = all_coords[:, 0].min(), all_coords[:, 1].min()
    xmax, ymax = all_coords[:, 0].max(), all_coords[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = max(0.08 * dx, 0.06), max(0.10 * dy, 0.05)
    events = context.events_month(month, ["train_id", "service_date", "op_id", "observed_ts", "planned_ts"])
    event_rows = events.loc[(events["train_id"].astype(str) == train_id) & (events["service_date"].astype(str) == service_date) & events["op_id"].notna()].copy()
    event_rows["op_id"] = event_rows["op_id"].astype(int)
    event_rows = event_rows.sort_values(["observed_ts", "planned_ts"], kind="stable")
    event_points = [(op_id, op_lookup[op_id]["lon"], op_lookup[op_id]["lat"]) for op_id in event_rows["op_id"].tolist() if op_id in op_lookup]
    event_ids = {p[0] for p in event_points}

    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=180)
    for geom in link_segments.values():
        if not (geom[:, 0].max() < xmin - pad_x or geom[:, 0].min() > xmax + pad_x or geom[:, 1].max() < ymin - pad_y or geom[:, 1].min() > ymax + pad_y):
            ax.plot(geom[:, 0], geom[:, 1], color="#94A3B8", linewidth=0.9, alpha=0.9, zorder=1)
    cmap = colormaps["turbo"]
    colors = cmap(np.linspace(0.05, 0.95, max(1, len(subpaths))))
    for idx, sub in enumerate(subpaths):
        if sub:
            ax.add_collection(LineCollection(sub, colors=[colors[idx]], linewidths=2.5, alpha=0.92, zorder=3, capstyle="round", joinstyle="round"))
    nearby_nodes = op_nodes.loc[op_nodes["lon"].between(xmin - pad_x, xmax + pad_x) & op_nodes["lat"].between(ymin - pad_y, ymax + pad_y)].copy()
    other_nodes = nearby_nodes.loc[~nearby_nodes["op_id"].isin(event_ids)]
    if not other_nodes.empty:
        ax.scatter(other_nodes["lon"], other_nodes["lat"], s=6, color="#334155", alpha=0.6, zorder=4)
    if event_points:
        xs, ys = [p[1] for p in event_points], [p[2] for p in event_points]
        ax.scatter(xs, ys, s=20, color="#2563EB", edgecolors="white", linewidth=0.45, zorder=5)
        ax.scatter(xs[0], ys[0], s=32, color="#16A34A", edgecolors="white", linewidth=0.8, zorder=6)
        ax.scatter(xs[-1], ys[-1], s=32, color="#DC2626", edgecolors="white", linewidth=0.8, zorder=6)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(all_coords[:, 1].mean()))), adjustable="box")
    safe_add_basemap(ax)
    legend_handles = [
        GradientLegendHandle(cmap, "deduced paths"),
        Line2D([], [], color="#94A3B8", linewidth=1.4, label="other node links"),
        Line2D([], [], marker="o", linestyle="None", markersize=5, color="#334155", markerfacecolor="#334155", label="other nodes"),
        Line2D([], [], marker="o", linestyle="None", markersize=6, color="#2563EB", markerfacecolor="#2563EB", markeredgecolor="white", label="event nodes"),
        Line2D([], [], marker="o", linestyle="None", markersize=7, color="#16A34A", markerfacecolor="#16A34A", markeredgecolor="white", label="start event node"),
        Line2D([], [], marker="o", linestyle="None", markersize=7, color="#DC2626", markerfacecolor="#DC2626", markeredgecolor="white", label="end event node"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, handler_map={GradientLegendHandle: HandlerGradient()})
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, figure_dir, "journey_illustration.svg")
    plt.close(fig)


def relation_type_from_series(s: pd.Series) -> pd.Series:
    categories = ["EURST", "EXTRA", "IC", "ICE", "INT", "L", "P", "TGV", "THAL", "S", "CHARTER", "nan"]
    rel = s.astype("string").fillna("nan").str.replace(r"^(S).*", "\x01", regex=True)
    rel = rel.str.split().str[0].fillna("nan")
    return rel.where(rel.isin(categories), "nan").astype("string")


# Paper Figure 16, fig:journeys-descriptive-stats -> journeys_descriptive_stats.svg.
# Caption: Descriptive statistics for the silver journeys table.
def plot_journeys_descriptive_stats(context: DatasetContext, figure_dir: Path) -> None:
    categories = ["EURST", "EXTRA", "IC", "ICE", "INT", "L", "P", "TGV", "THAL", "S", "CHARTER", "nan"]
    node_links = context.node_links(["link_id", "distance_m"])
    link_distance = np.zeros(int(node_links["link_id"].max()) + 1, dtype="float64")
    link_distance[node_links["link_id"].to_numpy(dtype=int)] = node_links["distance_m"].to_numpy(dtype="float64")
    operator_counts = pd.Series(dtype="int64")
    relation_counts = pd.Series(0, index=categories, dtype="int64")
    events_count_parts, total_distance_parts, final_delay_parts, final_relation_parts = [], [], [], []
    for path in sorted(context.journeys_dir.glob("journeys_*.parquet")):
        df = pd.read_parquet(path, columns=["operator", "train_relation", "events_count", "end_planned_ts", "end_observed_ts", "deduced_paths"])
        rel_type = relation_type_from_series(df["train_relation"])
        operator_counts = operator_counts.add(df["operator"].fillna("nan").value_counts(), fill_value=0)
        relation_counts = relation_counts.add(rel_type.value_counts().reindex(categories, fill_value=0), fill_value=0)
        events_count_parts.append(df["events_count"].to_numpy(dtype="int32", copy=False))
        totals = np.empty(len(df), dtype="float64")
        for i, paths in enumerate(df["deduced_paths"].to_numpy()):
            total_m = 0.0
            if paths is not None:
                for seg in paths:
                    if seg is not None and len(seg) > 0:
                        total_m += float(link_distance[np.asarray(seg, dtype=int)].sum())
            totals[i] = total_m / 1000.0
        total_distance_parts.append(totals)
        final_delay_parts.append(((pd.to_datetime(df["end_observed_ts"]) - pd.to_datetime(df["end_planned_ts"])).dt.total_seconds() / 60.0).to_numpy(dtype="float64", copy=False))
        final_relation_parts.append(rel_type.to_numpy(dtype=object, copy=False))
    journeys_stats = pd.DataFrame(
        {
            "events_count": np.concatenate(events_count_parts),
            "total_distance_km": np.concatenate(total_distance_parts),
            "final_delay_min": np.concatenate(final_delay_parts),
            "train_relation_type": np.concatenate(final_relation_parts),
        }
    )
    operator_share = (operator_counts / operator_counts.sum()).sort_values(ascending=False)
    relation_share = (relation_counts / relation_counts.sum()).reindex(categories)
    delay_min_plot = float(np.quantile(journeys_stats["final_delay_min"], 0.01))
    delay_max_plot = float(np.quantile(journeys_stats["final_delay_min"], 0.99))
    boxplot_categories = [c for c in categories if (journeys_stats["train_relation_type"] == c).any()]
    boxplot_data = [
        journeys_stats.loc[(journeys_stats["train_relation_type"] == cat) & journeys_stats["final_delay_min"].between(delay_min_plot, delay_max_plot), "final_delay_min"].to_numpy()
        for cat in boxplot_categories
    ]
    relation_colors = {"EURST": "#2563EB", "EXTRA": "#8B5CF6", "IC": "#F97316", "ICE": "#DC2626", "INT": "#0EA5E9", "L": "#10B981", "P": "#64748B", "TGV": "#D97706", "THAL": "#E11D48", "S": "#14B8A6", "CHARTER": "#A855F7", "nan": "#9CA3AF"}

    fig = plt.figure(figsize=(13, 12), dpi=180)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.6, wspace=0.28)
    ax1, ax2, ax3, ax4, ax5 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])
    ax1.barh(operator_share.index[::-1], operator_share.values[::-1], color="#4C78A8", edgecolor="white", linewidth=0.8)
    ax1.set_title("Operator Distribution")
    ax1.set_xlabel("Share of journeys")
    rel_present = relation_share[relation_share > 0]
    ax2.bar(rel_present.index, rel_present.values, color=[relation_colors.get(k, "#9CA3AF") for k in rel_present.index], edgecolor="white", linewidth=0.8)
    ax2.set_title("Train-Relation Distribution")
    ax2.set_xlabel("Train relation type")
    ax2.set_ylabel("Share of journeys")
    ax3.hist(journeys_stats["events_count"].to_numpy(dtype=float), bins=50, weights=np.ones(len(journeys_stats), dtype=float) / len(journeys_stats), color="#72B7B2", edgecolor="white", linewidth=0.4)
    ax3.set_title("Events per Journey")
    ax3.set_xlabel("Number of events")
    ax3.set_ylabel("Share of journeys")
    distance_values = journeys_stats["total_distance_km"].to_numpy(dtype=float)
    ax4.hist(distance_values, bins=60, weights=np.ones_like(distance_values, dtype=float) / len(distance_values), color="#F2CF5B", edgecolor="white", linewidth=0.4)
    ax4.set_title("Total Deduced-Path Distance")
    ax4.set_xlabel("Distance (km)")
    ax4.set_ylabel("Share of journeys")
    bp = ax5.boxplot(boxplot_data, tick_labels=boxplot_categories, patch_artist=True, showfliers=False)
    for patch, cat in zip(bp["boxes"], boxplot_categories):
        patch.set_facecolor(relation_colors.get(cat, "#9CA3AF"))
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    ax5.set_title("Final Journey Delay by Train Relation")
    ax5.set_xlabel("Train relation type")
    ax5.set_ylabel("Final delay (minutes)")
    ax5.set_ylim(-5, 25)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(axis="x" if ax is ax1 else "y", color="0.9")
    for ax in [ax2, ax5]:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, hspace=0.6, wspace=0.28)
    save_figure(fig, figure_dir, "journeys_descriptive_stats.svg")
    plt.close(fig)


def weather_group(code: object) -> str | None:
    if pd.isna(code):
        return None
    code = int(code)
    for group, spec in WEATHER_GROUPS.items():
        if code in spec["codes"]:
            return group
    return f"other ({code})"


def make_white_anchored_cmap(base_name: str, n: int = 256) -> ListedColormap:
    base = colormaps[base_name](np.linspace(0, 1, n))
    base[0] = np.array([1, 1, 1, 1])
    return ListedColormap(base, name=f"white_anchored_{base_name}")


# Paper Figure 17, fig:silver-weather-snapshot -> weather_snapshot_2024-01-01_02-00-00.svg.
# Caption: Illustrative weather snapshot from the silver weather table.
def plot_weather_snapshot(context: DatasetContext, figure_dir: Path) -> None:
    required = ["temperature_2m", "rain", "snowfall", "relative_humidity_2m", "wind_speed_10m", "weather_code"]
    labels = {"temperature_2m": "Temperature 2m", "rain": "Rain", "snowfall": "Snowfall", "relative_humidity_2m": "Relative humidity 2m", "wind_speed_10m": "Wind speed 10m", "weather_code": "Weather code"}
    units = {"temperature_2m": "°C", "rain": "mm", "snowfall": "cm", "relative_humidity_2m": "%", "wind_speed_10m": "km/h", "weather_code": ""}
    scales = {
        "temperature_2m": {"mode": "local", "cmap": "Blues", "vmin": 0, "vmax": 10, "clim_label": "climatological range 3 to 9 °C"},
        "rain": {"mode": "zero_anchored", "vmin": 0, "cmap": "Blues"},
        "snowfall": {"mode": "zero_anchored", "vmin": 0, "cmap": "Purples"},
        "relative_humidity_2m": {"mode": "local", "cmap": "BuGn"},
        "wind_speed_10m": {"mode": "local", "cmap": "viridis"},
    }
    nodes = context.op_nodes(["op_id", "lat", "lon"])
    weather = context.weather(["op_id", "time", *required]).copy()
    weather["time"] = pd.to_datetime(weather["time"])
    df = weather.merge(nodes, on="op_id", how="inner")
    requested = pd.Timestamp("2024-01-01 02:00:00")
    available_times = df["time"].drop_duplicates().sort_values()
    chosen_time = available_times.iloc[int((available_times - requested).abs().argmin())]
    snapshot = df[df["time"] == chosen_time].dropna(subset=["lat", "lon"]).copy()
    extent = (snapshot["lon"].min(), snapshot["lon"].max(), snapshot["lat"].min(), snapshot["lat"].max())
    pad_x, pad_y = (extent[1] - extent[0]) * 0.05, (extent[3] - extent[2]) * 0.05
    aspect = 1.0 / max(1e-6, np.cos(np.deg2rad(float(snapshot["lat"].mean()))))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), dpi=220)
    for ax, var_name in zip(axes.ravel(), required):
        ax.set_xlim(extent[0] - pad_x, extent[1] + pad_x)
        ax.set_ylim(extent[2] - pad_y, extent[3] + pad_y)
        ax.set_aspect(aspect, adjustable="box")
        safe_add_basemap(ax)
        valid = snapshot.dropna(subset=[var_name]).copy()
        ax.set_title(labels[var_name], fontsize=11)
        if var_name == "weather_code":
            valid["weather_group"] = valid[var_name].apply(weather_group)
            colors = valid["weather_group"].map({g: WEATHER_GROUPS[g]["color"] for g in WEATHER_GROUPS})
            ax.scatter(valid["lon"], valid["lat"], s=18, c=colors.tolist(), edgecolors="white", linewidths=0.25, alpha=0.95, zorder=2)
            present_codes = valid.groupby("weather_group")[var_name].apply(lambda s: sorted(set(map(int, s)))).to_dict()
            handles = [
                Line2D([], [], marker="o", linestyle="None", markersize=5.5, markerfacecolor=WEATHER_GROUPS[group]["color"], markeredgecolor="white", label=f"{', '.join(map(str, present_codes[group]))} ({group})")
                for group in WEATHER_GROUPS
                if group in present_codes
            ]
            ax.legend(handles=handles, title="phenomenon", loc="lower left", bbox_to_anchor=(0, 0.08), fontsize=7, title_fontsize=7, framealpha=0.92)
        else:
            scale = scales[var_name]
            cmap = make_white_anchored_cmap(scale["cmap"]) if var_name in {"rain", "snowfall"} else scale["cmap"]
            values = valid[var_name]
            vmin = scale.get("vmin", float(values.min()))
            vmax = scale.get("vmax", max(float(values.max()), float(vmin) + 1e-6))
            sc = ax.scatter(valid["lon"], valid["lat"], c=values, s=18, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.2, alpha=0.95, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            label = units[var_name]
            if var_name == "temperature_2m":
                label = f"{label} ({scale['clim_label']})"
            cbar.set_label(label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    safe_time = str(chosen_time).replace(":", "-").replace(" ", "_")
    save_figure(fig, figure_dir, f"weather_snapshot_{safe_time}.svg")
    plt.close(fig)


# Paper Figure 18, fig:weather-descriptive-stats -> weather_descriptive_stats.svg.
# Caption: Descriptive statistics for the silver weather table.
def plot_weather_descriptive_stats(context: DatasetContext, figure_dir: Path) -> None:
    continuous_specs = [
        ("temperature_2m", "Temperature", "°C", "#4C78A8"),
        ("rain", "Rain", "mm", "#2C7FB8"),
        ("snowfall", "Snowfall", "cm", "#A5C8FF"),
        ("relative_humidity_2m", "Relative Humidity", "%", "#2A9D8F"),
        ("wind_speed_10m", "Wind Speed", "km/h", "#F4A261"),
    ]
    weather = context.weather([c for c, *_ in continuous_specs] + ["weather_code"])
    groups = weather["weather_code"].map(weather_group)
    group_counts = groups.value_counts(normalize=True).sort_values(ascending=False)
    present_codes = {group: sorted(int(c) for c in weather.loc[groups == group, "weather_code"].dropna().unique()) for group in group_counts.index}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=180)
    for ax, (col, title, unit, color) in zip(axes.ravel()[:5], continuous_specs):
        values = weather[col].dropna().to_numpy(dtype="float64")
        weights = np.ones_like(values, dtype=float) / len(values)
        ax.hist(values, bins=60, weights=weights, color=color, edgecolor="white", linewidth=0.35)
        if col in {"rain", "snowfall"}:
            ax.set_xlim(0.0, float(values.max()))
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_ylabel("Share of rows")
        ax.grid(axis="y", color="0.9")
    ax = axes.ravel()[5]
    present_groups = [g for g in WEATHER_GROUPS if g in group_counts.index]
    ax.bar(range(len(present_groups)), group_counts.reindex(present_groups).values, color=[WEATHER_GROUPS[g]["color"] for g in present_groups], edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(present_groups)))
    ax.set_xticklabels(present_groups)
    ax.set_title("Weather Code Categories")
    ax.set_xlabel("Weather phenomenon")
    ax.set_ylabel("Share of rows")
    ax.grid(axis="y", color="0.9")
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    mapping_text = "codes:\n" + "\n".join(f"{g}: {', '.join(map(str, present_codes[g]))}" for g in present_groups)
    ax.text(0.98, 0.98, mapping_text, transform=ax.transAxes, va="top", ha="right", multialignment="left", fontsize=8.5, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9})
    fig.tight_layout()
    save_figure(fig, figure_dir, "weather_descriptive_stats.svg")
    plt.close(fig)


# Paper Figure 19, fig:gold_station_embeddings -> gold_station_embeddings.svg.
# Caption: The 8 normalized-Laplacian embedding components used as operational-point features in gold datasets.
def plot_gold_station_embeddings(context: DatasetContext, figure_dir: Path) -> None:
    from src.dataset.gold.station_embeddings import create_station_embeddings_from_silver

    embeddings, _, graph, positions, _ = create_station_embeddings_from_silver(
        node_links_path=context.static_dir / "node_links.parquet",
        op_nodes_path=context.static_dir / "op_nodes.parquet",
        embedding_dim=8,
    )
    nb_components, nrows, ncols = 8, 2, 4
    nodes_with_pos = list(positions.keys())
    subgraph = graph.subgraph(nodes_with_pos)
    aspect = 1.0 / np.cos(np.deg2rad(float(np.mean([positions[n][1] for n in nodes_with_pos]))))
    vmin, vmax = float(np.min(embeddings[:, :nb_components])), float(np.max(embeddings[:, :nb_components]))
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 6.4), constrained_layout=True)
    for comp_idx, ax in enumerate(axes.reshape(-1)[:nb_components]):
        nx.draw(
            subgraph,
            pos={n: positions[n] for n in nodes_with_pos},
            ax=ax,
            with_labels=False,
            node_size=5,
            node_color=embeddings[nodes_with_pos, comp_idx],
            edge_color="#7a7a7a",
            width=0.18,
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect(aspect, adjustable="box")
        ax.set_title(f"Component {comp_idx + 1}", fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    sm = ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.reshape(-1).tolist(), shrink=0.86, pad=0.012, aspect=28)
    cbar.set_label("Embedding value", rotation=270, labelpad=12)
    save_figure(fig, figure_dir, "gold_station_embeddings.svg")
    plt.close(fig)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def invert_minmax(values: np.ndarray, stats: dict) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return v * (float(stats["max"]) - float(stats["min"])) + float(stats["min"])


def build_train_positions_with_offset(station_pos, past_edge_index, past_idx, future_edge_index, future_rank, train_ids, *, offset_scale=0.01):
    past_station_by_train, past_rank_by_train = {}, {}
    for edge_idx in range(past_edge_index.shape[1]):
        train_idx, station_idx, rank = int(past_edge_index[0, edge_idx]), int(past_edge_index[1, edge_idx]), int(past_idx[edge_idx])
        if train_idx not in past_rank_by_train or rank < past_rank_by_train[train_idx]:
            past_rank_by_train[train_idx] = rank
            past_station_by_train[train_idx] = station_idx
    future_station_by_train, future_rank_by_train = {}, {}
    for edge_idx in range(future_edge_index.shape[1]):
        train_idx, station_idx, rank = int(future_edge_index[0, edge_idx]), int(future_edge_index[1, edge_idx]), int(future_rank[edge_idx])
        if train_idx not in future_rank_by_train or rank < future_rank_by_train[train_idx]:
            future_rank_by_train[train_idx] = rank
            future_station_by_train[train_idx] = station_idx
    train_pos = {}
    for train_idx in train_ids:
        has_prev, has_next = train_idx in past_station_by_train, train_idx in future_station_by_train
        if not has_prev and not has_next:
            continue
        if has_prev and has_next:
            prev_station, next_station = int(past_station_by_train[train_idx]), int(future_station_by_train[train_idx])
            if f"s{prev_station}" not in station_pos or f"s{next_station}" not in station_pos:
                continue
            x1, y1 = station_pos[f"s{prev_station}"]
            x2, y2 = station_pos[f"s{next_station}"]
        elif has_next:
            next_station = int(future_station_by_train[train_idx])
            if f"s{next_station}" not in station_pos:
                continue
            x2, y2 = station_pos[f"s{next_station}"]
            x1, y1 = x2, y2
        else:
            prev_station = int(past_station_by_train[train_idx])
            if f"s{prev_station}" not in station_pos:
                continue
            x1, y1 = station_pos[f"s{prev_station}"]
            x2, y2 = x1, y1
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        ortho_x, ortho_y = (0.0, 1.0) if norm == 0 else (-dy / norm, dx / norm)
        side = -1.0 if (train_idx % 2 == 0) else 1.0
        train_pos[f"t{int(train_idx)}"] = (0.5 * (x1 + x2) + side * offset_scale * ortho_x, 0.5 * (y1 + y2) + side * offset_scale * ortho_y)
    return train_pos


def symmetric_radii(n_edges: int, base_rad: float = 0.10) -> list[float]:
    if n_edges <= 1:
        return [0.0]
    half = n_edges // 2
    if n_edges % 2 == 0:
        mags = [(i + 0.5) * base_rad for i in range(half)]
        return [-m for m in reversed(mags)] + mags
    mags = [(i + 1) * base_rad for i in range(half)]
    return [-m for m in reversed(mags)] + [0.0] + mags


def draw_curved_train_station_edges(ax, pos, edges, *, color, linewidth, alpha=1.0, curve_strength=0.08, min_rad=0.03, max_rad=0.12, zorder=1) -> None:
    grouped: defaultdict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        grouped[edge] += 1
    for (u, v), count in grouped.items():
        if u not in pos or v not in pos:
            continue
        p0, p1 = pos[u], pos[v]
        dist = float(((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5)
        edge_base_rad = max(min_rad, min(max_rad, curve_strength / max(dist, 1e-9)))
        for rad in symmetric_radii(count, base_rad=edge_base_rad):
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-", connectionstyle=f"arc3,rad={rad}", linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, shrinkA=0, shrinkB=0))


# Paper Figure 21, fig:gnn_snapshot_country -> gnn_full_graph_illus.svg.
# Caption: Visualization of a full-network GNN snapshot.
def plot_gnn_full_graph_illus(context: DatasetContext, figure_dir: Path) -> None:
    import torch

    gnn_dir = context.gnn_dir()
    split, chunk_id, graph_local_idx = "train", 0, 9
    graphs = torch.load(gnn_dir / split / f"graphs_part_{chunk_id:05d}.pt", map_location="cpu", weights_only=False)
    feature_spec = read_yaml(gnn_dir / "feature_spec.yaml")
    normalization = read_yaml(gnn_dir / "normalization.yaml")
    graph = graphs[graph_local_idx]
    station_cols = feature_spec["station_node_cols"]
    station_x = graph["station"].x.detach().cpu().numpy()
    station_lat = invert_minmax(station_x[:, station_cols.index("station_lat")], normalization["station_nodes"]["station_lat"])
    station_lon = invert_minmax(station_x[:, station_cols.index("station_lon")], normalization["station_nodes"]["station_lon"])
    station_ids = np.arange(len(station_lat), dtype=np.int64)
    station_pos = {f"s{int(i)}": (float(station_lon[i]), float(station_lat[i])) for i in station_ids}
    station_edge_index = graph[REL_STS].edge_index.detach().cpu().numpy()
    past_edge_index_full = graph[REL_PAST].edge_index.detach().cpu().numpy()
    past_attr_full = graph[REL_PAST].edge_attr.detach().cpu().numpy()
    past_idx_raw = invert_minmax(past_attr_full[:, feature_spec["past_edge_cols"].index("past_idx")], normalization["train_to_past_station_edges"]["past_idx"]) if past_attr_full.shape[0] else np.empty(0)
    past_idx_raw = np.rint(past_idx_raw).astype(np.int16)
    past_mask = past_idx_raw <= 7
    past_edge_index, past_idx = past_edge_index_full[:, past_mask], past_idx_raw[past_mask]
    future_edge_index_full = graph[REL_FUTURE].edge_index.detach().cpu().numpy()
    future_rank_full = graph[REL_FUTURE].future_rank.detach().cpu().numpy().astype(np.int16)
    future_mask = future_rank_full <= 7
    future_edge_index, future_rank = future_edge_index_full[:, future_mask], future_rank_full[future_mask]
    train_ids = np.arange(int(graph["train"].x.shape[0]), dtype=np.int64)

    G = nx.Graph()
    for station_node in station_pos:
        G.add_node(station_node, kind="station")
    for train_id in train_ids:
        G.add_node(f"t{int(train_id)}", kind="train")
    for edge_idx in range(station_edge_index.shape[1]):
        u, v = f"s{int(station_edge_index[0, edge_idx])}", f"s{int(station_edge_index[1, edge_idx])}"
        if u != v:
            G.add_edge(u, v, kind="sts")
    past_edges = [(f"t{int(past_edge_index[0, i])}", f"s{int(past_edge_index[1, i])}") for i in range(past_edge_index.shape[1])]
    future_edges = [(f"t{int(future_edge_index[0, i])}", f"s{int(future_edge_index[1, i])}") for i in range(future_edge_index.shape[1])]
    station_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "station"]
    train_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "train"]
    sts_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "sts"]
    train_pos = build_train_positions_with_offset(station_pos, past_edge_index, past_idx, future_edge_index, future_rank, train_ids, offset_scale=0.03)
    pos = dict(station_pos)
    pos.update({node: train_pos[node] for node in train_nodes if node in train_pos})

    fig, ax = plt.subplots(figsize=(15, 15), dpi=150)
    nx.draw_networkx_edges(G, pos, edgelist=sts_edges, edge_color="#1D3557", width=0.5, alpha=0.35, ax=ax)
    draw_curved_train_station_edges(ax, pos, past_edges, color="#2A9D8F", linewidth=0.5, alpha=0.25, curve_strength=0.03, min_rad=0.03, max_rad=0.15)
    draw_curved_train_station_edges(ax, pos, future_edges, color="#E76F51", linewidth=0.5, alpha=0.25, curve_strength=0.03, min_rad=0.03, max_rad=0.15)
    nx.draw_networkx_nodes(G, pos, nodelist=station_nodes, node_color="#1D3557", node_size=3, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in train_nodes if n in pos], node_color="#D62828", node_shape="s", node_size=5, alpha=0.9, ax=ax)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(station_lat)))), adjustable="box")
    ax.legend(
        handles=[
            Line2D([], [], color="#1D3557", linewidth=1.2, label="station-to-station edges"),
            Line2D([], [], color="#2A9D8F", linewidth=1.2, label="train-to-past-station edges"),
            Line2D([], [], color="#E76F51", linewidth=1.2, label="train-to-future-station edges"),
            Line2D([], [], marker="o", markersize=6, linestyle="None", markerfacecolor="#1D3557", markeredgecolor="none", label="station nodes"),
            Line2D([], [], marker="s", markersize=6, linestyle="None", markerfacecolor="#D62828", markeredgecolor="none", label="train nodes"),
        ],
        loc="upper right",
        frameon=False,
    )
    ax.axis("off")
    save_figure(fig, figure_dir, "gnn_full_graph_illus.svg")
    plt.close(fig)


def _bin_counts(values: np.ndarray, edges: np.ndarray, n_labels: int) -> np.ndarray:
    valid = np.isfinite(values)
    idx = np.searchsorted(edges, values[valid], side="right") - 1
    idx = np.clip(idx, 0, n_labels - 1)
    return np.bincount(idx, minlength=n_labels)


def _to_percent(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return np.zeros_like(counts, dtype=np.float64) if total == 0 else 100.0 * counts / total


def load_standard_eval_table(context: DatasetContext) -> pd.DataFrame:
    cols = ["ts", "last_known_delay"]
    for j in range(1, N_FUTURE + 1):
        cols.extend([f"future_obs_ts_{j}", f"future_delay_{j}"])
    key = ("standard_eval_table", tuple(cols))
    if key not in context._cache:
        context._cache[key] = pd.read_parquet(context.standard_eval_table_path(), columns=cols)
    return context._cache[key]  # type: ignore[return-value]


# Paper Figures 22, 23, 25, and 26:
# - fig:overall_horizon_bin_counts -> standard_overall_horizon_bin_counts.svg.
# - fig:horizon_bin_counts_by_future_event -> standard_horizon_bin_counts_by_future_event.svg.
# - fig:overall_delay_delta_bin_counts -> standard_overall_delay_delta_bin_counts.svg.
# - fig:delay_delta_bin_counts_by_future_event -> standard_delay_delta_bin_counts_by_future_event.svg.
# Captions: Valid-target count distributions by horizon and delay-delta bins on the standard tier.
def plot_standard_eval_bin_distributions(context: DatasetContext, figure_dir: Path) -> None:
    eval_table = load_standard_eval_table(context)
    snapshot_ts = pd.to_datetime(eval_table["ts"], errors="coerce")
    last_known_delay = pd.to_numeric(eval_table["last_known_delay"], errors="coerce").to_numpy(dtype=np.float64)
    horizon_counts, delay_delta_counts = {}, {}
    for j in range(1, N_FUTURE + 1):
        future_obs = pd.to_datetime(eval_table[f"future_obs_ts_{j}"], errors="coerce")
        future_delay = pd.to_numeric(eval_table[f"future_delay_{j}"], errors="coerce").to_numpy(dtype=np.float64)
        horizon_min = (future_obs - snapshot_ts).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
        delay_delta_min = (future_delay - last_known_delay) / 60.0
        horizon_counts[j] = _bin_counts(horizon_min, HORIZON_EDGES_MIN, len(HORIZON_LABELS_SHORT))
        delay_delta_counts[j] = _bin_counts(delay_delta_min, DELAY_DELTA_EDGES_MIN, len(DELAY_DELTA_LABELS_SHORT))

    overall_horizon_counts = np.sum(np.stack([horizon_counts[j] for j in range(1, N_FUTURE + 1)], axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(12.5, 4.6), constrained_layout=True)
    x = np.arange(len(HORIZON_LABELS_SHORT))
    colors = ["#4C78A8"] * len(HORIZON_LABELS_SHORT)
    colors[-1] = "#2F5D8A"
    ax.bar(x, overall_horizon_counts, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_xticks(x, HORIZON_LABELS_SHORT)
    ax.set_xlabel("Horizon bin (min)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    save_figure(fig, figure_dir, "standard_overall_horizon_bin_counts.svg")
    plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(26, 13.5), constrained_layout=True, sharey=True)
    horizon_percents = {j: _to_percent(horizon_counts[j]) for j in range(1, N_FUTURE + 1)}
    max_horizon_percent = max(float(np.max(v)) for v in horizon_percents.values())
    for j, ax in enumerate(axes.ravel(), start=1):
        ax.bar(x, horizon_percents[j], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(f"Future event {j}", fontsize=20, pad=8)
        ax.set_xticks(x)
        if j > 10:
            ax.set_xticklabels(HORIZON_LABELS_SHORT, rotation=45, ha="center", fontsize=15)
            ax.set_xlabel("Horizon bin (min)", fontsize=20)
        else:
            ax.set_xticklabels([])
        if (j - 1) % 5 == 0:
            ax.set_ylabel("Share of valid targets (%)", fontsize=20)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(0, max_horizon_percent * 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
    save_figure(fig, figure_dir, "standard_horizon_bin_counts_by_future_event.svg")
    plt.close(fig)

    overall_delay_counts = np.sum(np.stack([delay_delta_counts[j] for j in range(1, N_FUTURE + 1)], axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(13.5, 4.6), constrained_layout=True)
    x = np.arange(len(DELAY_DELTA_LABELS_SHORT))
    colors = ["#F58518"] * len(DELAY_DELTA_LABELS_SHORT)
    colors[0] = colors[-1] = "#C96A0F"
    ax.bar(x, overall_delay_counts, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_xticks(x, DELAY_DELTA_LABELS_SHORT)
    ax.set_xlabel("Delay-delta bin (min)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    save_figure(fig, figure_dir, "standard_overall_delay_delta_bin_counts.svg")
    plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(28, 13.5), constrained_layout=True, sharey=True)
    delay_percents = {j: _to_percent(delay_delta_counts[j]) for j in range(1, N_FUTURE + 1)}
    max_delay_percent = max(float(np.max(v)) for v in delay_percents.values())
    for j, ax in enumerate(axes.ravel(), start=1):
        ax.bar(x, delay_percents[j], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(f"Future event {j}", fontsize=20, pad=8)
        ax.set_xticks(x)
        if j > 10:
            ax.set_xticklabels(DELAY_DELTA_LABELS_SHORT, rotation=45, ha="center", fontsize=15)
            ax.set_xlabel("Delay-delta bin (min)", fontsize=20)
        else:
            ax.set_xticklabels([])
        if (j - 1) % 5 == 0:
            ax.set_ylabel("Share of valid targets (%)", fontsize=20)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(0, max_delay_percent * 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
    save_figure(fig, figure_dir, "standard_delay_delta_bin_counts_by_future_event.svg")
    plt.close(fig)


# Paper Figure 24, fig:standard_delay_delta_by_horizon_boxplot -> standard_delay_delta_by_horizon_boxplot.svg.
# Caption: Delay-delta distribution by horizon bin on the standard tier.
def plot_standard_delay_delta_by_horizon_boxplot(context: DatasetContext, figure_dir: Path) -> None:
    eval_table = load_standard_eval_table(context)
    eval_table["ts"] = pd.to_datetime(eval_table["ts"])
    rng = np.random.default_rng(42)
    sampled_frames = []
    for j in range(1, N_FUTURE + 1):
        obs = pd.to_datetime(eval_table[f"future_obs_ts_{j}"])
        delay = eval_table[f"future_delay_{j}"].to_numpy(dtype=np.float64, copy=False)
        last_delay = eval_table["last_known_delay"].to_numpy(dtype=np.float64, copy=False)
        valid = obs.notna().to_numpy() & np.isfinite(delay)
        if not valid.any():
            continue
        horizon_min = ((obs[valid] - eval_table.loc[valid, "ts"]).dt.total_seconds().to_numpy(dtype=np.float64, copy=False)) / 60.0
        delay_delta_min = (delay[valid] - last_delay[valid]) / 60.0
        slot_frame = pd.DataFrame({"horizon_min": horizon_min, "delay_delta_min": delay_delta_min})
        slot_frame["horizon_bin"] = pd.cut(slot_frame["horizon_min"], bins=HORIZON_EDGES_MIN, labels=HORIZON_LABELS_BOX, right=False)
        slot_frame = slot_frame.dropna(subset=["horizon_bin"])
        if len(slot_frame) > 50_000:
            take = rng.choice(len(slot_frame), size=50_000, replace=False)
            slot_frame = slot_frame.iloc[np.sort(take)]
        sampled_frames.append(slot_frame)
    sampled_long = pd.concat(sampled_frames, ignore_index=True)
    sampled_long["horizon_bin"] = pd.Categorical(sampled_long["horizon_bin"], categories=HORIZON_LABELS_BOX, ordered=True)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=300)
    sns.boxplot(
        data=sampled_long,
        x="horizon_bin",
        y="delay_delta_min",
        order=HORIZON_LABELS_BOX,
        width=0.6,
        showfliers=False,
        boxprops={"facecolor": "#A8C5E5", "edgecolor": "#1F4E79", "linewidth": 1.0},
        medianprops={"color": "#B22222", "linewidth": 1.4},
        whiskerprops={"color": "#1F4E79", "linewidth": 1.0},
        capprops={"color": "#1F4E79", "linewidth": 1.0},
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.7, linestyle="--")
    ax.set_xlabel("Prediction horizon (minutes)", fontsize=11)
    ax.set_ylabel("Delay change (minutes)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, figure_dir, "standard_delay_delta_by_horizon_boxplot.svg")
    plt.close(fig)


GENERATOR_FUNCTIONS: dict[str, Callable[[DatasetContext, Path], None]] = {
    "plot_network_snapshot": plot_network_snapshot,
    "plot_network_visualization": plot_network_visualization,
    "plot_silver_illus_example_map": plot_silver_illus_example_map,
    "plot_journey_time_station": plot_journey_time_station,
    "plot_op_nodes_descriptive_stats": plot_op_nodes_descriptive_stats,
    "plot_line_sections_descriptive_stats": plot_line_sections_descriptive_stats,
    "plot_compare_event_vs_infra_graph": plot_compare_event_vs_infra_graph,
    "plot_node_links_descriptive_stats": plot_node_links_descriptive_stats,
    "plot_node_links_overlap_zoom": plot_node_links_overlap_zoom,
    "plot_events_descriptive_stats": plot_events_descriptive_stats,
    "plot_journey_illustration": plot_journey_illustration,
    "plot_journeys_descriptive_stats": plot_journeys_descriptive_stats,
    "plot_weather_snapshot": plot_weather_snapshot,
    "plot_weather_descriptive_stats": plot_weather_descriptive_stats,
    "plot_gold_station_embeddings": plot_gold_station_embeddings,
    "plot_gnn_full_graph_illus": plot_gnn_full_graph_illus,
    "plot_standard_eval_bin_distributions": plot_standard_eval_bin_distributions,
    "plot_standard_delay_delta_by_horizon_boxplot": plot_standard_delay_delta_by_horizon_boxplot,
}


def selected_generators(only: list[str] | None) -> list[str]:
    if not only:
        return list(dict.fromkeys(spec.generator for spec in FIGURES))
    wanted = set(only)
    generators = []
    for spec in FIGURES:
        stems = {Path(output).stem for output in spec.outputs}
        filenames = set(spec.outputs)
        if spec.label in wanted or stems & wanted or filenames & wanted:
            generators.append(spec.generator)
    unknown = wanted - {spec.label for spec in FIGURES} - {Path(output).stem for spec in FIGURES for output in spec.outputs} - {output for spec in FIGURES for output in spec.outputs}
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown --only figure label or output stem: {unknown_list}")
    return list(dict.fromkeys(generators))


def main() -> None:
    args = parse_args()
    if args.list:
        print_figure_mapping()
        return

    configure_plots()
    figure_dir = args.output_dir / "figures" / "dataset"
    figure_dir.mkdir(parents=True, exist_ok=True)
    context = DatasetContext(args)

    for generator_name in selected_generators(args.only):
        print(f"Generating {generator_name}...")
        GENERATOR_FUNCTIONS[generator_name](context, figure_dir)


if __name__ == "__main__":
    main()
