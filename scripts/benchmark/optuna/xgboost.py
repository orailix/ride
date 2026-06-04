"""Run Optuna hyperparameter search workers for the XGBoost benchmark model.

Each trial launches the regular XGBoost training script in a per-trial
directory, records the sampled hyperparameters and worker metadata, and reports
validation MAE in seconds back to a shared Optuna study. Multiple workers can
share the same SQLite study during Slurm runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import optuna

from src.benchmark.utils.optuna import (
    build_trial_dir,
    suggest_hparams_from_search_space,
    wait_for_study,
)
from src.dataset.pipeline.helpers import read_yaml, write_yaml


XGBOOST_TRAIN_HPARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "val_fraction",
    "early_stopping_rounds",
)


def suggest_hparams(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, int | float]:
    """Sample and validate the XGBoost training hyperparameters for one trial."""
    return suggest_hparams_from_search_space(
        trial=trial,
        search_space=args.search_space,
        required_keys=XGBOOST_TRAIN_HPARAM_KEYS,
    )

def build_train_command(
    *,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    hparams: dict[str, int | float],
) -> list[str]:
    """Build the CLI command for the regular XGBoost training entry point."""
    return [
        sys.executable,
        "-m",
        "scripts.benchmark.train.xgboost",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--no-verbose",
        "--n-estimators",
        str(hparams["n_estimators"]),
        "--val-fraction",
        str(hparams["val_fraction"]),
        "--early-stopping-rounds",
        str(hparams["early_stopping_rounds"]),
        "--max-depth",
        str(hparams["max_depth"]),
        "--learning-rate",
        str(hparams["learning_rate"]),
        "--subsample",
        str(hparams["subsample"]),
        "--colsample-bytree",
        str(hparams["colsample_bytree"]),
        "--min-child-weight",
        str(hparams["min_child_weight"]),
        "--reg-alpha",
        str(hparams["reg_alpha"]),
        "--reg-lambda",
        str(hparams["reg_lambda"]),
    ]


def read_trial_objective(*, trial_dir: Path) -> float:
    """Read the scalar validation objective written by XGBoost training."""
    history = read_yaml(trial_dir / "train_history.yaml")
    return float(history["val_mae_seconds_mean"])

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Run one Optuna trial and return its validation MAE in seconds."""
    trial_dir = build_trial_dir(base_dir=args.output_dir, trial=trial)
    trial_dir.mkdir(parents=True, exist_ok=True)

    hparams = suggest_hparams(trial, args)
    seed = 1000 + int(trial.number)
    command = build_train_command(
        data_dir=args.data_dir,
        output_dir=trial_dir,
        seed=seed,
        hparams=hparams,
    )

    worker_info = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "slurm_localid": os.environ.get("SLURM_LOCALID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_yaml(
        trial_dir / "trial_config.yaml",
        {
            "trial_number": int(trial.number),
            "seed": seed,
            "hparams": hparams,
            "command": command,
            "worker_info": worker_info,
        },
    )

    subprocess.run(command, check=True, env=os.environ.copy())
    val_mae_s = read_trial_objective(trial_dir=trial_dir)

    trial.set_user_attr("seed", seed)
    trial.set_user_attr("trial_dir", str(trial_dir))
    for key, value in hparams.items():
        trial.set_user_attr(key, value)
    return val_mae_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna worker for XGBoost hyperparameter optimization. "
            "Run multiple copies of this script against the same study/storage for Slurm parallelism."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Gold tabular dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Study output directory containing per-trial folders.")
    parser.add_argument("--search-space-config", type=Path, required=True, help="YAML config describing sampled and fixed Optuna/trainer hyperparameters.")
    parser.add_argument("--study-name", type=str, required=True, help="Optuna study name.")
    parser.add_argument("--time-budget-hours", type=float, required=True, help="Wall-clock budget for this worker in hours.")
    parser.add_argument("--max-trials", type=int, default=None, help="Maximum number of trials for this worker.")
    args = parser.parse_args()
    args.search_space = read_yaml(args.search_space_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (args.output_dir / "optuna_study.db").resolve()
    storage = f"sqlite:///{storage_path}"
    ready_path = args.output_dir / ".study_ready"
    # Slurm workers use distinct sampler seeds but synchronize on one study.
    sampler_seed = int(os.environ.get("SLURM_PROCID", "0"))
    print(
        "[optuna-xgboost] "
        f"procid={os.environ.get('SLURM_PROCID', '')} "
        f"localid={os.environ.get('SLURM_LOCALID', '')} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    sampler = optuna.samplers.TPESampler(seed=sampler_seed, constant_liar=True)
    procid = int(os.environ.get("SLURM_PROCID", "0"))
    if procid == 0:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        ready_path.write_text("ready\n")
    else:
        study = wait_for_study(
            study_name=args.study_name,
            storage=storage,
            sampler=sampler,
            ready_path=ready_path,
        )
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.max_trials,
        timeout=max(1, int(round(args.time_budget_hours * 3600.0))),
    )


if __name__ == "__main__":
    main()
