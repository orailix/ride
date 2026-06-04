"""Shared helpers for Optuna-based benchmark hyperparameter searches."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import optuna

from src.dataset.pipeline.helpers import read_yaml


def _apply_derived_op(*, name: str, base_value: Any, spec: dict[str, Any]) -> str | int | float | bool:
    """Resolve a derived search-space entry from an already suggested value."""
    op = str(spec.get("op", "")).lower()
    if op == "identity":
        return base_value
    if op == "multiply":
        return base_value * spec["by"]
    if op == "floor_divide":
        return base_value // spec["by"]
    raise ValueError(f"Unsupported derived op '{op}' for entry '{name}'.")


def suggest_from_spec(
    *,
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> str | int | float | bool:
    """Suggest or return one hyperparameter value from a YAML search-space entry."""
    param_type = str(spec.get("type", "")).lower()
    if param_type == "categorical":
        choices = list(spec.get("choices") or [])
        if not choices:
            raise ValueError(f"Categorical search-space entry '{name}' must define non-empty choices.")
        return trial.suggest_categorical(name, choices)
    if param_type == "float":
        min_value = float(spec["min"])
        max_value = float(spec["max"])
        step = spec.get("step")
        return trial.suggest_float(
            name,
            min_value,
            max_value,
            log=bool(spec.get("log", False)),
            step=None if step is None else float(step),
        )
    if param_type == "int":
        min_value = int(spec["min"])
        max_value = int(spec["max"])
        step = int(spec.get("step", 1))
        return trial.suggest_int(
            name,
            min_value,
            max_value,
            step=step,
            log=bool(spec.get("log", False)),
        )
    if param_type == "fixed":
        if "value" not in spec:
            raise ValueError(f"Fixed search-space entry '{name}' must define a value.")
        return spec["value"]
    raise ValueError(f"Unsupported search-space type '{param_type}' for entry '{name}'.")


def suggest_hparams_from_search_space(
    *,
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
    required_keys: Sequence[str],
) -> dict[str, str | int | float | bool]:
    """Resolve the ordered hyperparameters required by one benchmark training script."""
    missing = [key for key in required_keys if key not in search_space]
    if missing:
        raise KeyError(f"Missing required search-space entries: {missing}")
    resolved: dict[str, str | int | float | bool] = {}
    for key in required_keys:
        spec = search_space[key]
        param_type = str(spec.get("type", "")).lower()
        if param_type != "derived":
            resolved[key] = suggest_from_spec(trial=trial, name=key, spec=spec)
            continue

        source_key = str(spec.get("from", ""))
        if not source_key:
            raise ValueError(f"Derived search-space entry '{key}' must define a source via 'from'.")
        if source_key not in resolved:
            raise KeyError(
                f"Derived search-space entry '{key}' depends on '{source_key}', "
                "which must appear earlier in required_keys."
            )
        resolved[key] = _apply_derived_op(name=key, base_value=resolved[source_key], spec=spec)
    return resolved


def build_trial_dir(*, base_dir: Path, trial: optuna.Trial) -> Path:
    """Return the output directory assigned to an Optuna trial."""
    return base_dir / f"trial_{trial.number:05d}"


def read_trial_objective(*, trial_dir: Path) -> tuple[float, int]:
    """Read the best validation MAE and epoch from a completed trial directory."""
    history = read_yaml(trial_dir / "train_history.yaml")
    val_curve = [float(v) for v in history["val_mae_seconds_per_epoch"]]
    best_epoch_idx = min(range(len(val_curve)), key=val_curve.__getitem__)
    return float(val_curve[best_epoch_idx]), int(best_epoch_idx + 1)


def wait_for_study(
    *,
    study_name: str,
    storage: str,
    sampler: optuna.samplers.BaseSampler,
    ready_path: Path,
) -> optuna.Study:
    """Wait until the launcher has created the shared Optuna study."""
    while True:
        if not ready_path.exists():
            time.sleep(1.0)
            continue
        try:
            return optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        except Exception:
            time.sleep(1.0)
