"""Evaluate one trained MLP feature-family ablation run.

The script applies the same in-memory feature removal used during training,
converts model outputs back to absolute future delays in seconds through the
shared MLP prediction helper, and evaluates against the Gold core
``test_eval_table.parquet`` so ablation runs remain comparable to the benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.benchmark.models.mlp import (
    MLPRegressor,
    build_prediction_eval_table,
    load_split_arrays,
    predict_in_batches,
)
from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.benchmark.utils.feature_ablation import (
    build_ablation_column_index,
    feature_family_choices,
)
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MLP feature-ablation run on the test split."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet used for final evaluation.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Training run folder containing model.pt and where predictions/metrics are written.")
    parser.add_argument("--ablate-family", type=str, required=True, choices=feature_family_choices(), help="Feature family removed before evaluation.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size.")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu")
    checkpoint_family = checkpoint.get("ablate_family")
    if checkpoint_family is not None and checkpoint_family != args.ablate_family:
        raise ValueError(
            f"Checkpoint ablation family '{checkpoint_family}' does not match "
            f"requested family '{args.ablate_family}'."
        )

    # Rebuild the ablated feature view from the dataset scheme. The saved model
    # input dimension then acts as a guard against evaluating with another family.
    scheme = read_yaml(args.data_dir / "scheme.yaml")
    keep_indices, _kept_columns, _removed_columns = build_ablation_column_index(
        x_columns=list(scheme["x_columns"]),
        ablate_family=args.ablate_family,
    )
    model = MLPRegressor(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dims=[int(v) for v in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])

    x_test, _, md_test, _ = load_split_arrays(args.data_dir, "test")
    x_test = x_test[:, keep_indices].astype("float32", copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    y_pred_test = predict_in_batches(model=model, x=x_test, batch_size=args.batch_size, device=device)

    y_cols = [str(c) for c in checkpoint["y_columns"]]
    normalization = read_yaml(args.data_dir / "normalization.yaml")
    test_eval_table = pd.read_parquet(args.test_eval_table)
    # Ablation MLPs are trained on delay deltas, but the benchmark evaluator
    # expects absolute future delays in seconds with the Gold core target names.
    eval_target_cols = [c.replace("future_delay_delta_", "future_delay_") for c in y_cols]
    target_stats = {
        y_col: (
            float(normalization[y_col]["mean"]),
            float(normalization[y_col]["std"]),
            bool(normalization[y_col].get("sqrt", False)),
        )
        for y_col in y_cols
    }

    test_pred_table = build_prediction_eval_table(
        md=md_test,
        y_pred=y_pred_test,
        y_columns=y_cols,
        eval_table=test_eval_table,
        eval_target_columns=eval_target_cols,
        target_stats=target_stats,
    )
    eval_metrics_test = evaluate_delay_predictions(eval_table=test_eval_table, predictions=test_pred_table)

    test_pred_table.to_parquet(args.model_dir / "test_predictions.parquet", index=False)
    write_yaml(
        args.model_dir / "eval_metrics.yaml",
        {"test": eval_metrics_test},
    )


if __name__ == "__main__":
    main()
