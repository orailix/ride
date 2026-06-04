"""Run final repeated test evaluation for MLP feature-family ablations.

Feature ablations use a two-stage protocol: epochs are first selected on a
validation split, then each ablated model is retrained for the selected number
of epochs on the full training split and evaluated over multiple seeds. This
script implements the second stage and writes one summary per ablated family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.benchmark.utils.feature_ablation import feature_family_choices
from src.benchmark.utils.test_eval import run_seed_sweep
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def _parse_seeds(text: str) -> list[int]:
    """Parse the comma-separated seed list used for repeated test evaluation."""
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def _selected_epochs(selection_summary: dict[str, Any]) -> dict[str, int]:
    """Read the selected epoch for each ablated feature family."""
    rows = selection_summary.get("families", [])
    out: dict[str, int] = {}
    for row in rows:
        family = str(row["family"])
        out[family] = int(row["selected_epoch"])
    return out


def _format_hidden_dims(hidden_dims: list[int]) -> str:
    """Serialize hidden dimensions for the underlying training script CLI."""
    return ",".join(str(v) for v in hidden_dims)


def _build_command_plan_builder(
    *,
    family: str,
    selected_epoch: int,
    train_config: dict[str, Any],
    eval_batch_size: int,
    test_eval_table: Path,
) -> Any:
    """Create train/eval commands for one ablated family and seed."""
    def _builder(data_dir: Path, seed: int, run_dir: Path) -> dict[str, Any]:
        train_command = [
            sys.executable,
            "-m",
            "scripts.benchmark.ablation.train_mlp",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(run_dir),
            "--ablate-family",
            family,
            "--seed",
            str(seed),
            "--hidden-dims",
            _format_hidden_dims([int(v) for v in train_config["hidden_dims"]]),
            "--dropout",
            str(train_config["dropout"]),
            "--epochs",
            str(selected_epoch),
            "--batch-size",
            str(train_config["batch_size"]),
            "--num-workers",
            str(train_config["num_workers"]),
            "--val-fraction",
            "0.0",
            "--precision",
            str(train_config["precision"]),
            "--lr",
            str(train_config["lr"]),
            "--weight-decay",
            str(train_config["weight_decay"]),
            "--early-stopping-patience",
            "-1",
        ]
        eval_command = [
            sys.executable,
            "-m",
            "scripts.benchmark.ablation.eval_mlp",
            "--data-dir",
            str(data_dir),
            "--test-eval-table",
            str(test_eval_table),
            "--model-dir",
            str(run_dir),
            "--ablate-family",
            family,
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
    parser = argparse.ArgumentParser(
        description="Run test-set MLP feature-family ablations using epochs preselected on the validation split."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Gold lite tabular dataset directory.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where per-family runs and summaries are written.")
    parser.add_argument("--epoch-selection-summary", type=Path, required=True, help="Path to epoch_selection_summary.yaml from the validation-phase ablation runs.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark/best_models/lite/mlp.yaml"), help="Path to the fixed full-feature lite MLP config.")
    parser.add_argument("--families", type=str, default="all", help="Comma-separated feature families to run, or 'all'.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of integer seeds to run.")
    args = parser.parse_args()

    config = read_yaml(args.config)
    train_config = config["train"]
    eval_batch_size = int(config.get("eval", {}).get("batch_size", train_config["batch_size"]))
    selection_summary = read_yaml(args.epoch_selection_summary)
    family_to_epoch = _selected_epochs(selection_summary)

    if args.families.strip().lower() == "all":
        families = feature_family_choices()
    else:
        requested = [part.strip() for part in args.families.split(",") if part.strip()]
        valid = set(feature_family_choices())
        unknown = [family for family in requested if family not in valid]
        if unknown:
            raise ValueError(f"Unknown feature families: {unknown}. Expected from {sorted(valid)}")
        families = requested

    missing_epochs = [family for family in families if family not in family_to_epoch]
    if missing_epochs:
        raise ValueError(
            f"Missing selected epochs for families {missing_epochs} in {args.epoch_selection_summary}."
        )

    seeds = _parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    family_runs: list[dict[str, Any]] = []
    for family in families:
        family_dir = args.output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        selected_epoch = family_to_epoch[family]
        print(
            f"[mlp-ablation-test-eval] running family='{family}' "
            f"with selected_epoch={selected_epoch} in {family_dir}"
        )
        run_seed_sweep(
            output_dir=family_dir,
            data_dir=args.data_dir,
            seeds=seeds,
            command_plan_builder=_build_command_plan_builder(
                family=family,
                selected_epoch=selected_epoch,
                train_config=train_config,
                eval_batch_size=eval_batch_size,
                test_eval_table=args.test_eval_table,
            ),
        )
        family_runs.append(
            {
                "family": family,
                "selected_epoch": selected_epoch,
                "output_dir": str(family_dir),
                "seeds": list(seeds),
            }
        )

    write_yaml(
        args.output_dir / "ablation_runs.yaml",
        {
            "config_path": str(args.config),
            "epoch_selection_summary": str(args.epoch_selection_summary),
            "test_eval_table": str(args.test_eval_table),
            "families": family_runs,
        },
    )
    print(f"[mlp-ablation-test-eval] wrote run index to {args.output_dir / 'ablation_runs.yaml'}")


if __name__ == "__main__":
    main()
