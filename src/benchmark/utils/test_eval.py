"""Run repeated train/evaluate sweeps and aggregate test metrics."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.dataset.pipeline.helpers import read_yaml, write_yaml


COUNT_METRIC_NAMES = {
    "n_rows_matched",
    "n_rows_unmatched",
    "n_values",
    "n_values_evaluated",
}


def seed_run_dir(*, output_dir: Path, seed: int) -> Path:
    """Return the run directory used for one test-evaluation seed."""
    return output_dir / f"seed_{seed:02d}"


def infer_benchmark_tier_from_data_dir(data_dir: Path) -> str:
    """Infer the benchmark tier name from a Gold data directory path."""
    parts = {part.lower() for part in data_dir.parts}
    if "lite" in parts:
        return "lite"
    if "standard" in parts:
        return "standard"
    raise ValueError(
        f"Could not infer benchmark tier from data_dir={data_dir}. "
        "Pass --tier explicitly or override with --config."
    )


def default_test_eval_config_path(*, model_name: str, tier: str) -> Path:
    """Return the default best-model configuration path for a model and tier."""
    return Path("configs/benchmark/best_models") / tier / f"{model_name}.yaml"


def resolve_test_eval_config(
    *,
    model_name: str,
    data_dir: Path,
    tier: str,
    config: Path | None,
) -> Path:
    """Resolve the configuration file used for a test-evaluation run."""
    if config is not None:
        return config
    resolved_tier = infer_benchmark_tier_from_data_dir(data_dir) if tier == "auto" else tier
    return default_test_eval_config_path(model_name=model_name, tier=resolved_tier)


def set_or_append_cli_arg(
    *,
    command: Sequence[str],
    arg_name: str,
    value: str | int | float,
) -> list[str]:
    """Replace a CLI argument value or append the argument if it is absent."""
    updated = list(command)
    arg_value = str(value)
    for idx, token in enumerate(updated):
        if token != arg_name:
            continue
        if idx + 1 >= len(updated):
            raise ValueError(f"Command argument '{arg_name}' is missing a value.")
        updated[idx + 1] = arg_value
        return updated
    updated.extend([arg_name, arg_value])
    return updated


def load_test_metrics_from_eval_yaml(run_dir: Path) -> dict[str, Any]:
    """Load test metrics produced by the evaluation command in one run directory."""
    return read_yaml(run_dir / "eval_metrics.yaml")["test"]


def _collect_numeric_leaves(
    *,
    value: Any,
    path: tuple[str, ...],
    out: dict[tuple[str, ...], float],
) -> None:
    """Collect finite scalar metrics from a nested metrics dictionary."""
    if path and path[-1] in COUNT_METRIC_NAMES:
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _collect_numeric_leaves(value=child, path=path + (str(key),), out=out)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            out[path] = numeric_value


def _set_nested_metric(
    *,
    tree: dict[str, Any],
    path: tuple[str, ...],
    value: dict[str, float | int],
) -> None:
    """Assign an aggregated metric summary at a nested metric path."""
    cursor = tree
    for part in path[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def aggregate_test_metrics(
    *,
    per_seed_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-seed test metrics into mean and population standard deviation."""
    values_by_path: dict[tuple[str, ...], list[float]] = {}
    for metrics in per_seed_metrics:
        flattened: dict[tuple[str, ...], float] = {}
        _collect_numeric_leaves(value=metrics, path=(), out=flattened)
        for path, value in flattened.items():
            values_by_path.setdefault(path, []).append(value)

    summary: dict[str, Any] = {}
    for path, values in values_by_path.items():
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        _set_nested_metric(
            tree=summary,
            path=path,
            value={
                "mean": mean_value,
                "std": math.sqrt(variance),
            },
        )
    return summary


def run_seed_sweep(
    *,
    output_dir: Path,
    data_dir: Path,
    seeds: Sequence[int],
    command_plan_builder: Callable[[Path, int, Path], dict[str, Any]],
) -> dict[str, Any]:
    """Train and evaluate one model across seeds, then write an aggregate summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = seed_run_dir(output_dir=output_dir, seed=seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[test-eval] running seed={seed} in {run_dir}")
        plan = command_plan_builder(data_dir, seed, run_dir)
        plan_seed = plan["seed"]
        plan_run_dir = plan["run_dir"]
        train_command = [str(token) for token in plan["train_command"]]
        eval_command = [str(token) for token in plan["eval_command"]]

        write_yaml(
            run_dir / "test_eval_plan.yaml",
            {
                "seed": plan_seed,
                "run_dir": str(plan_run_dir),
                "train_command": train_command,
                "eval_command": eval_command,
            },
        )

        subprocess.run(train_command, check=True)
        subprocess.run(eval_command, check=True)

        test_metrics = load_test_metrics_from_eval_yaml(run_dir)
        run_results.append(
            {
                "seed": plan_seed,
                "run_dir": plan_run_dir,
                "train_command": train_command,
                "eval_command": eval_command,
                "test_metrics": test_metrics,
            }
        )

    summary = aggregate_test_metrics(
        per_seed_metrics=[result["test_metrics"] for result in run_results]
    )
    write_yaml(output_dir / "test_eval_summary.yaml", summary)
    print(f"[test-eval] wrote summary to {output_dir / 'test_eval_summary.yaml'}")
    return summary
