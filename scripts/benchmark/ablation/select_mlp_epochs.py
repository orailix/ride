"""Select the number of training epochs for MLP feature-family ablations.

This is the validation stage of the ablation protocol. Each feature family is
removed in memory, trained with the fixed Lite MLP hyperparameters, and assigned
the epoch with the best validation MAE in seconds. The selected epochs are then
consumed by ``scripts.benchmark.ablation.test_eval_mlp`` for repeated test
evaluation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.benchmark.utils.feature_ablation import feature_family_choices
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def format_hidden_dims(hidden_dims: list[int]) -> str:
    """Serialize hidden dimensions for the underlying training script CLI."""
    return ",".join(str(v) for v in hidden_dims)


def best_epoch_from_history(history: dict[str, Any]) -> tuple[int, float]:
    """Return the one-indexed epoch with the lowest validation MAE in seconds."""
    val_mae = [float(v) for v in history.get("val_mae_seconds_per_epoch", [])]
    if not val_mae:
        raise ValueError("Missing val_mae_seconds_per_epoch in train_history.yaml.")
    best_idx = min(range(len(val_mae)), key=lambda i: val_mae[i])
    return best_idx + 1, val_mae[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLP feature-family ablations on the validation split and select the best epoch for each family."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Gold lite tabular dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where per-family validation runs and the epoch summary are written.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark/best_models/lite/mlp.yaml"), help="Path to the fixed full-feature lite MLP config.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used during the validation-phase ablation runs.")
    parser.add_argument("--epochs", type=int, default=75, help="Maximum number of epochs for the validation-phase runs.")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience for the validation-phase runs.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction used to select the best epoch.")
    parser.add_argument("--families", type=str, default="all", help="Comma-separated feature families to run, or 'all'.")
    args = parser.parse_args()

    config = read_yaml(args.config)
    train_config = config["train"]

    if args.families.strip().lower() == "all":
        families = feature_family_choices()
    else:
        requested = [part.strip() for part in args.families.split(",") if part.strip()]
        valid = set(feature_family_choices())
        unknown = [family for family in requested if family not in valid]
        if unknown:
            raise ValueError(f"Unknown feature families: {unknown}. Expected from {sorted(valid)}")
        families = requested

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for family in families:
        run_dir = args.output_dir / family
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "scripts.benchmark.ablation.train_mlp",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(run_dir),
            "--ablate-family",
            family,
            "--seed",
            str(args.seed),
            "--hidden-dims",
            format_hidden_dims([int(v) for v in train_config["hidden_dims"]]),
            "--dropout",
            str(train_config["dropout"]),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(train_config["batch_size"]),
            "--num-workers",
            str(train_config["num_workers"]),
            "--val-fraction",
            str(args.val_fraction),
            "--precision",
            str(train_config["precision"]),
            "--lr",
            str(train_config["lr"]),
            "--weight-decay",
            str(train_config["weight_decay"]),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
        ]

        print(f"[mlp-ablation-select] running family='{family}' in {run_dir}")
        subprocess.run(command, check=True)

        history = read_yaml(run_dir / "train_history.yaml")
        selected_epoch, best_val_mae_seconds = best_epoch_from_history(history)
        summary_rows.append(
            {
                "family": family,
                "seed": int(args.seed),
                "selected_epoch": int(selected_epoch),
                "best_val_mae_seconds": float(best_val_mae_seconds),
                "run_dir": str(run_dir),
            }
        )

    write_yaml(
        args.output_dir / "epoch_selection_summary.yaml",
        {
            "config_path": str(args.config),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_stopping_patience),
            "val_fraction": float(args.val_fraction),
            "families": summary_rows,
        },
    )
    print(f"[mlp-ablation-select] wrote summary to {args.output_dir / 'epoch_selection_summary.yaml'}")


if __name__ == "__main__":
    main()
