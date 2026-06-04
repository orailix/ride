"""Generate Gold Standard benchmark result tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


MODEL_ORDER = [
    "Translation",
    "Graph-event",
    "MLP",
    "XGBoost",
    "LSTM",
    "Transformer",
    "GNN",
]

MODEL_RUN_DIRS = {
    "Translation": "translation",
    "Graph-event": "graph_event",
    "MLP": "mlp",
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Transformer": "transformer",
    "GNN": "gnn",
}

MODEL_LABELS = {
    "Translation": "Translation",
    "Graph-event": "Graph-event",
    "MLP": "MLP",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "GNN": "GNN",
}

MODEL_COLORS = {
    "Translation": "#7f7f7f",
    "Graph-event": "#8c564b",
    "MLP": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "LSTM": "#2ca02c",
    "Transformer": "#d62728",
    "GNN": "#9467bd",
}

HORIZON_ORDER = [
    "[0,5)m",
    "[5,10)m",
    "[10,15)m",
    "[15,20)m",
    "[20,25)m",
    "[25,30)m",
    "[30,35)m",
    "[35,40)m",
    "[40,45)m",
    "[45,inf)m",
]

HORIZON_LABELS = {
    "[0,5)m": "0:5",
    "[5,10)m": "5:10",
    "[10,15)m": "10:15",
    "[15,20)m": "15:20",
    "[20,25)m": "20:25",
    "[25,30)m": "25:30",
    "[30,35)m": "30:35",
    "[35,40)m": "35:40",
    "[40,45)m": "40:45",
    "[45,inf)m": "45+",
}

DELAY_DELTA_ORDER = [
    "[-inf,-300)s",
    "[-300,-120)s",
    "[-120,-60)s",
    "[-60,-30)s",
    "[-30,0)s",
    "[0,30)s",
    "[30,60)s",
    "[60,120)s",
    "[120,300)s",
    "[300,600)s",
    "[600,inf)s",
]

DELAY_DELTA_LABELS = {
    "[-inf,-300)s": "<-5",
    "[-300,-120)s": "-5:-2",
    "[-120,-60)s": "-2:-1",
    "[-60,-30)s": "-1:-0.5",
    "[-30,0)s": "-0.5:0",
    "[0,30)s": "0:0.5",
    "[30,60)s": "0.5:1",
    "[60,120)s": "1:2",
    "[120,300)s": "2:5",
    "[300,600)s": "5:10",
    "[600,inf)s": "10+",
}

PAPER_WIDTH = 6.8
HALF_WIDTH = 3.3
GRID_COLOR = "0.88"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("runs/benchmark/standard"),
        help="Directory containing one test-evaluation run directory per model.",
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


def load_summaries(summary_dir: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    missing = []
    for model in MODEL_ORDER:
        path = summary_dir / MODEL_RUN_DIRS[model] / "test_eval_summary.yaml"
        if not path.exists():
            missing.append(path)
            continue
        summaries[model] = load_yaml(path)
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing standard summary files:\n{missing_list}")
    return summaries


def metric_mean(summary: dict[str, Any], metric: str) -> float:
    return float(summary[metric]["mean"])


def metric_std(summary: dict[str, Any], metric: str) -> float:
    return float(summary[metric].get("std", 0.0))


def nested_mae(summary: dict[str, Any], section: str, key: str) -> float:
    return float(summary[section][key]["mae"]["mean"])


def fmt_main_metric(summary: dict[str, Any], metric: str, best: float) -> str:
    mean = metric_mean(summary, metric)
    std = metric_std(summary, metric)
    if std == 0.0:
        text = f"{mean:.2f}"
    else:
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    if mean == best:
        return f"\\textbf{{{text}}}"
    return text


def fmt_breakdown(value: float, best: float) -> str:
    text = f"{value:.1f}"
    if value == best:
        return f"\\textbf{{{text}}}"
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_main_table(summaries: dict[str, dict[str, Any]]) -> str:
    best_mae = min(metric_mean(summaries[model], "mae") for model in MODEL_ORDER)
    best_rmse = min(metric_mean(summaries[model], "rmse") for model in MODEL_ORDER)

    rows = [
        f"{model} & {fmt_main_metric(summaries[model], 'mae', best_mae)} & "
        f"{fmt_main_metric(summaries[model], 'rmse', best_rmse)} \\\\"
        for model in MODEL_ORDER
    ]
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering
\begin{{tabular}}{{lll}}
\toprule
Model & MAE & RMSE \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Test-set performance on the standard tier. MAE and RMSE in seconds.}}
\label{{tab:standard_main_results}}
\end{{table}}
"""


def build_breakdown_table(
    summaries: dict[str, dict[str, Any]],
    *,
    section: str,
    order: list[str],
    labels: dict[str, str],
    tabular_spec: str,
    caption: str,
    label: str,
) -> str:
    best_by_key = {
        key: min(nested_mae(summaries[model], section, key) for model in MODEL_ORDER)
        for key in order
    }
    header = "Model & " + " & ".join(labels[key] for key in order) + r" \\"
    rows = []
    for model in MODEL_ORDER:
        values = [
            fmt_breakdown(nested_mae(summaries[model], section, key), best_by_key[key])
            for key in order
        ]
        rows.append(f"{model} & " + " & ".join(values) + r" \\")
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{{tabular_spec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
"""


def build_horizon_table(summaries: dict[str, dict[str, Any]]) -> str:
    return build_breakdown_table(
        summaries,
        section="per_horizon_minutes",
        order=HORIZON_ORDER,
        labels=HORIZON_LABELS,
        tabular_spec="lllllllllll",
        caption="Test-set MAE in seconds by prediction horizon on the standard tier. Bins are in minutes.",
        label="tab:standard_horizon_mae",
    )


def build_delay_delta_table(summaries: dict[str, dict[str, Any]]) -> str:
    return build_breakdown_table(
        summaries,
        section="per_delay_delta_bin",
        order=DELAY_DELTA_ORDER,
        labels=DELAY_DELTA_LABELS,
        tabular_spec="llllllllllll",
        caption=(
            "Test-set MAE in seconds by delay-delta bin on the standard tier. "
            "Bins are in minutes; negative values correspond to delay recovery "
            "and positive values to delay accumulation."
        ),
        label="tab:standard_delay_delta_mae",
    )


def main_results_df(summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        summary = summaries[model]
        rows.append(
            {
                "Model": model,
                "MAE": float(summary["mae"]["mean"]),
                "MAE std": float(summary["mae"].get("std", 0.0)),
                "RMSE": float(summary["rmse"]["mean"]),
                "RMSE std": float(summary["rmse"].get("std", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def breakdown_df(
    summaries: dict[str, dict[str, Any]],
    *,
    section: str,
    order: list[str],
    labels: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": model,
                **{labels[key]: nested_mae(summaries[model], section, key) for key in order},
            }
            for model in MODEL_ORDER
        ]
    )


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


def plot_main_results(main_df: pd.DataFrame, tier: str, figure_dir: Path) -> None:
    df = main_df.copy()
    df["Model label"] = df["Model"].map(MODEL_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 3.0), dpi=300, sharey=True)
    metrics = [("MAE", "MAE std"), ("RMSE", "RMSE std")]
    y_positions = range(len(df))

    for ax, (metric, err_col) in zip(axes, metrics):
        ax.barh(
            y_positions,
            df[metric],
            color=[MODEL_COLORS[m] for m in df["Model"]],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks(list(y_positions), labels=df["Model label"])
        ax.invert_yaxis()

    axes[0].set_ylabel("Model")
    fig.tight_layout()
    save_current_figure(figure_dir, f"{tier}_main_results_bar")
    plt.close(fig)


def plot_breakdown_competitive_facets(
    df: pd.DataFrame,
    tier: str,
    stem_suffix: str,
    title: str,
    ylabel: str,
    figure_dir: Path,
    relative_to_best: bool = False,
) -> None:
    plot_df = df.copy()
    long_df = plot_df.melt(id_vars="Model", var_name="Bin", value_name="MAE")
    long_df["Model"] = pd.Categorical(long_df["Model"], MODEL_ORDER, ordered=True)
    long_df["Model label"] = long_df["Model"].map(MODEL_LABELS)

    if relative_to_best:
        best = long_df.groupby("Bin", as_index=False)["MAE"].min().rename(columns={"MAE": "best_mae"})
        long_df = long_df.merge(best, on="Bin", how="left")
        long_df["Value"] = long_df["MAE"] - long_df["best_mae"]
    else:
        long_df["Value"] = long_df["MAE"]

    bins = list(df.columns[1:])
    n_bins = len(bins)
    ncols = 5 if n_bins <= 10 else 4
    nrows = (n_bins + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAPER_WIDTH, 1.55 * nrows), dpi=300, sharey=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, bin_label in zip(axes, bins):
        sub = long_df[long_df["Bin"] == bin_label].sort_values("Model")
        ax.barh(
            sub["Model label"],
            sub["Value"],
            color=[MODEL_COLORS[m] for m in sub["Model"]],
            edgecolor="black",
            linewidth=0.4,
        )
        if relative_to_best:
            ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(bin_label, pad=2)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        local_max = float(sub["Value"].max())
        if relative_to_best:
            xmax = max(local_max * 1.12, 1.0)
            ax.set_xlim(-0.02 * xmax, xmax)
        else:
            xmin = float(sub["Value"].min())
            pad = max((local_max - xmin) * 0.08, 2.0)
            ax.set_xlim(max(0.0, xmin - pad), local_max + pad)

    for ax in axes[n_bins:]:
        ax.axis("off")

    fig.text(0.5, 0.005, ylabel, ha="center")
    fig.tight_layout()
    save_current_figure(figure_dir, f"{tier}_{stem_suffix}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.summary_dir)

    table_dir = args.output_dir / "tables" / "standard"
    figure_dir = args.output_dir / "figures" / "standard"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    write_text(table_dir / "standard_main_results_table.tex", build_main_table(summaries))
    write_text(table_dir / "standard_horizon_mae_table.tex", build_horizon_table(summaries))
    write_text(table_dir / "standard_delay_delta_mae_table.tex", build_delay_delta_table(summaries))

    configure_plots()
    plot_main_results(main_results_df(summaries), "standard", figure_dir)
    plot_breakdown_competitive_facets(
        breakdown_df(
            summaries,
            section="per_horizon_minutes",
            order=HORIZON_ORDER,
            labels=HORIZON_LABELS,
        ),
        tier="standard",
        stem_suffix="horizon_mae_plot",
        title="horizon breakdown relative to best model",
        ylabel="MAE gap to best",
        figure_dir=figure_dir,
        relative_to_best=True,
    )
    plot_breakdown_competitive_facets(
        breakdown_df(
            summaries,
            section="per_delay_delta_bin",
            order=DELAY_DELTA_ORDER,
            labels=DELAY_DELTA_LABELS,
        ),
        tier="standard",
        stem_suffix="delay_delta_mae_plot",
        title="delay-delta breakdown relative to best model",
        ylabel="MAE gap to best",
        figure_dir=figure_dir,
        relative_to_best=True,
    )


if __name__ == "__main__":
    main()
