"""Generate feature-ablation result tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


ABLATION_ORDER = [
    "train_information",
    "snapshot_time_features",
    "event_planned_timing",
    "past_delay_features",
    "event_type_features",
    "event_node_embeddings",
    "local_network_context",
    "weather_features",
]

ABLATION_LABELS = {
    "train_information": "train information",
    "snapshot_time_features": "snapshot-time features",
    "event_planned_timing": "planned timing",
    "past_delay_features": "past delay features",
    "event_type_features": "event types",
    "event_node_embeddings": "node embeddings",
    "local_rail_topology": "local network context",
    "local_network_context": "local network context",
    "weather_features": "weather features",
}

TABLE_LABELS = {
    "train_information": "train information",
    "snapshot_time_features": "snapshot-time features",
    "event_planned_timing": "planned timing",
    "past_delay_features": "past delay features",
    "event_type_features": "event types",
    "event_node_embeddings": "node embeddings",
    "local_network_context": "local network context",
    "weather_features": "weather features",
}

PAPER_WIDTH = 6.8
GRID_COLOR = "0.88"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("runs/benchmark/lite/mlp/test_eval_summary.yaml"),
        help="Full-feature lite MLP test-evaluation summary.",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("runs/ablation/mlp_lite_test_eval"),
        help="Directory containing one test-evaluation summary per removed feature family.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables_figures"),
        help="Root directory where table and figure artifacts are written.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_yaml(path)


def fmt_metric(summary: dict[str, Any], metric: str) -> str:
    mean = float(summary[metric]["mean"])
    std = float(summary[metric].get("std", 0.0))
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def fmt_delta(delta: float) -> str:
    return f"{delta:+.2f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def load_ablation_summaries(ablation_dir: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    missing = []
    for family in ABLATION_ORDER:
        path = ablation_dir / family / "test_eval_summary.yaml"
        if not path.exists():
            missing.append(path)
            continue
        summaries[family] = load_yaml(path)
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing ablation summary files:\n{missing_list}")
    return summaries


def build_ablation_df(
    baseline_summary: dict[str, Any],
    ablation_summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    base_mae = float(baseline_summary["mae"]["mean"])
    base_rmse = float(baseline_summary["rmse"]["mean"])
    rows = []
    for family in ABLATION_ORDER:
        summary = ablation_summaries[family]
        mae = float(summary["mae"]["mean"])
        rmse = float(summary["rmse"]["mean"])
        rows.append(
            {
                "family_key": family,
                "family": ABLATION_LABELS[family],
                "MAE": mae,
                "MAE std": float(summary["mae"].get("std", 0.0)),
                "RMSE": rmse,
                "RMSE std": float(summary["rmse"].get("std", 0.0)),
                "delta_MAE": mae - base_mae,
                "delta_RMSE": rmse - base_rmse,
            }
        )
    return pd.DataFrame(rows)


def build_ablation_table(
    baseline_summary: dict[str, Any],
    ablation_summaries: dict[str, dict[str, Any]],
) -> str:
    base_mae = float(baseline_summary["mae"]["mean"])
    base_rmse = float(baseline_summary["rmse"]["mean"])
    rows = [
        f"MLP (all features) & {fmt_metric(baseline_summary, 'mae')} & --- & "
        f"{fmt_metric(baseline_summary, 'rmse')} & --- \\\\",
        r"\midrule",
    ]
    for family in ABLATION_ORDER:
        summary = ablation_summaries[family]
        delta_mae = float(summary["mae"]["mean"]) - base_mae
        delta_rmse = float(summary["rmse"]["mean"]) - base_rmse
        rows.append(
            f"w/o {TABLE_LABELS[family]} & {fmt_metric(summary, 'mae')} & "
            f"{fmt_delta(delta_mae)} & {fmt_metric(summary, 'rmse')} & "
            f"{fmt_delta(delta_rmse)} \\\\"
        )
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Setting & MAE & $\Delta$MAE & RMSE & $\Delta$RMSE \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Main results of the lite-tier MLP feature-family ablation study. The first row shows the full-feature model, and each subsequent row removes one feature family. MAE and RMSE are reported as mean $\pm$ standard deviation across runs, with $\Delta$ values measured relative to the base model.}}
\label{{tab:mlp_feature_ablation}}
\end{{table}}
"""


def configure_plots() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        },
    )


def save_current_figure(figure_dir: Path, stem: str) -> None:
    plt.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")


def plot_ablation_deltas(ablation_df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = ablation_df.copy().sort_values("delta_MAE")
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 3.9), dpi=300, sharey=True)

    for ax, value_col, title in zip(
        axes,
        ["delta_MAE", "delta_RMSE"],
        ["MAE delta", "RMSE delta"],
    ):
        values = plot_df[value_col]
        colors = ["#C44E52" if val > 0 else "#55A868" for val in values]
        bars = ax.barh(
            plot_df["family"],
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Delta")
        ax.tick_params(axis="y", pad=12)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)

        span = max(abs(values.min()), abs(values.max()))
        pad = max(span * 0.18, 0.5)
        ax.set_xlim(values.min() - pad, values.max() + pad)

        for bar, val in zip(bars, values):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ha = "left" if val >= 0 else "right"
            offset = 0.08 * span if span else 0.1
            ax.text(x + (offset if val >= 0 else -offset), y, f"{val:+.2f}", va="center", ha=ha, fontsize=7.5)

    axes[0].set_ylabel("Removed feature family", labelpad=18)
    fig.tight_layout(rect=(0.06, 0.02, 1.0, 1.0))
    save_current_figure(figure_dir, "lite_mlp_ablation_deltas_bar")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_summary = load_summary(args.baseline_summary)
    ablation_summaries = load_ablation_summaries(args.ablation_dir)

    table_dir = args.output_dir / "tables" / "ablation"
    figure_dir = args.output_dir / "figures" / "ablation"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        table_dir / "lite_mlp_ablation_table.tex",
        build_ablation_table(baseline_summary, ablation_summaries),
    )

    configure_plots()
    plot_ablation_deltas(build_ablation_df(baseline_summary, ablation_summaries), figure_dir)


if __name__ == "__main__":
    main()
