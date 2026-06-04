"""Run repeated train+eval for the GNN benchmark model.

This wrapper resolves the fixed benchmark config, retrains the GNN once per
seed on the full training split, evaluates each run on the shared Gold core test
table, and writes the aggregated test summary.
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
    eval_config: dict[str, Any],
    test_eval_table: Path,
) -> Any:
    """Create the train/eval command pair executed for each seed."""
    def _builder(data_dir: Path, seed: int, run_dir: Path) -> dict[str, Any]:
        train_command = [
            sys.executable,
            "-m",
            "scripts.benchmark.train.gnn",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(run_dir),
            "--seed",
            str(seed),
            "--epochs",
            str(train_config["epochs"]),
            "--batch-size",
            str(train_config["batch_size"]),
            "--num-workers",
            str(train_config["num_workers"]),
            # Final test evaluation retrains on the full training split; epochs
            # and hyperparameters have already been selected before this stage.
            "--val-fraction",
            "0.0",
            "--precision",
            str(train_config["precision"]),
            "--lr",
            str(train_config["lr"]),
            "--weight-decay",
            str(train_config["weight_decay"]),
            "--hidden-dim",
            str(train_config["hidden_dim"]),
            "--num-layers",
            str(train_config["num_layers"]),
            "--gnn-dropout",
            str(train_config["gnn_dropout"]),
            "--head-dropout",
            str(train_config["head_dropout"]),
            "--edge-head-hidden-dim",
            str(train_config["edge_head_hidden_dim"]),
            "--hetero-aggr",
            str(train_config["hetero_aggr"]),
            "--early-stopping-patience",
            "-1",
        ]
        train_command.append("--use-layer-norm" if train_config["use_layer_norm"] else "--no-use-layer-norm")

        eval_batch_size = int(eval_config.get("batch_size", train_config["batch_size"]))
        eval_command = [
            sys.executable,
            "-m",
            "scripts.benchmark.eval.gnn",
            "--data-dir",
            str(data_dir),
            "--test-eval-table",
            str(test_eval_table),
            "--model-dir",
            str(run_dir),
            "--batch-size",
            str(eval_batch_size),
        ]
        return {
            "seed": seed,
            "run_dir": run_dir,
            "train_command": train_command,
            "eval_command": eval_command,
        }

    return _builder


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated train+eval runner for the GNN benchmark.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing the gold train/test graph chunks.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where per-seed runs and the summary are written.")
    parser.add_argument("--tier", choices=("auto", "lite", "standard"), default="auto", help="Benchmark tier used to resolve the default config when --config is not provided.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the fixed GNN train/eval config YAML.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated list of integer seeds to run.")
    args = parser.parse_args()

    config_path = resolve_test_eval_config(model_name="gnn", data_dir=args.data_dir, tier=args.tier, config=args.config)
    config = read_yaml(config_path)
    train_config = config["train"]
    eval_config = config["eval"]
    seeds = _parse_seeds(args.seeds)

    run_seed_sweep(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seeds=seeds,
        command_plan_builder=_build_command_plan_builder(
            train_config=train_config,
            eval_config=eval_config,
            test_eval_table=args.test_eval_table,
        ),
    )


if __name__ == "__main__":
    main()
