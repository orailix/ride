"""Run repeated train+eval for the XGBoost benchmark model.

This wrapper resolves the fixed benchmark config, retrains the horizon-specific
XGBoost models once per seed on the full training split, evaluates each run on
the shared Gold core test table, and writes the aggregated test summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.benchmark.utils.test_eval import resolve_test_eval_config, run_seed_sweep
from src.dataset.pipeline.helpers import read_yaml


def _parse_seeds(text: str) -> list[int]:
    """Parse the comma-separated seed list used for repeated test evaluation."""
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def _build_command_plan_builder(
    *,
    train_config: dict[str, Any],
    test_eval_table: Path,
) -> Any:
    """Create the train/eval command pair executed for each seed."""
    def _builder(data_dir: Path, seed: int, run_dir: Path) -> dict[str, Any]:
        train_command = [
            sys.executable,
            "-m",
            "scripts.benchmark.train.xgboost",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(run_dir),
            "--seed",
            str(seed),
            "--n-estimators",
            str(train_config["n_estimators"]),
            # Final test evaluation retrains on the full training split; tuning
            # and early stopping have already been completed before this stage.
            "--val-fraction",
            "0.0",
            "--max-depth",
            str(train_config["max_depth"]),
            "--learning-rate",
            str(train_config["learning_rate"]),
            "--subsample",
            str(train_config["subsample"]),
            "--colsample-bytree",
            str(train_config["colsample_bytree"]),
            "--min-child-weight",
            str(train_config["min_child_weight"]),
            "--reg-alpha",
            str(train_config["reg_alpha"]),
            "--reg-lambda",
            str(train_config["reg_lambda"]),
        ]

        eval_command = [
            sys.executable,
            "-m",
            "scripts.benchmark.eval.xgboost",
            "--data-dir",
            str(data_dir),
            "--test-eval-table",
            str(test_eval_table),
            "--model-dir",
            str(run_dir),
        ]
        return {
            "seed": seed,
            "run_dir": run_dir,
            "train_command": train_command,
            "eval_command": eval_command,
        }

    return _builder


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated train+eval runner for the XGBoost benchmark.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing the gold train/test arrays.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where per-seed runs and the summary are written.")
    parser.add_argument("--tier", choices=("auto", "lite", "standard"), default="auto", help="Benchmark tier used to resolve the default config when --config is not provided.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the fixed XGBoost train/eval config YAML.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated list of integer seeds to run.")
    args = parser.parse_args()

    config_path = resolve_test_eval_config(model_name="xgboost", data_dir=args.data_dir, tier=args.tier, config=args.config)
    config = read_yaml(config_path)
    train_config = config["train"]
    seeds = _parse_seeds(args.seeds)

    run_seed_sweep(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seeds=seeds,
        command_plan_builder=_build_command_plan_builder(
            train_config=train_config,
            test_eval_table=args.test_eval_table,
        ),
    )


if __name__ == "__main__":
    main()
