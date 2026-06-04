"""Evaluate a trained Transformer benchmark run on the Gold test split.

The script groups tabular test rows by snapshot, predicts with a Transformer
over each snapshot sequence, converts delay-delta outputs back to absolute
future delays in seconds, and evaluates against the shared Gold core
``test_eval_table.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.benchmark.models.transformer import (
    SnapshotSequenceDataset,
    TransformerRegressor,
    build_prediction_eval_table,
    build_snapshot_groups,
    collate_predict,
    load_split_arrays,
    predict_grouped,
)
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer regressor on the test split.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet used for final evaluation.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Training run folder containing model.pt and where predictions/metrics are written.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (in snapshots).")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader workers.")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu")
    model = TransformerRegressor(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        d_model=int(checkpoint["d_model"]),
        nhead=int(checkpoint["nhead"]),
        num_layers=int(checkpoint["num_layers"]),
        dim_feedforward=int(checkpoint["dim_feedforward"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])

    x_test, _, md_test, _ = load_split_arrays(args.data_dir, "test")
    test_groups = build_snapshot_groups(md_test)
    test_ds = SnapshotSequenceDataset(x=x_test, row_groups=test_groups)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_predict,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    y_pred_test = predict_grouped(
        model=model,
        loader=test_loader,
        n_rows=x_test.shape[0],
        n_targets=int(checkpoint["output_dim"]),
        device=device,
    )

    y_cols = [str(c) for c in checkpoint["y_columns"]]
    normalization = read_yaml(args.data_dir / "normalization.yaml")
    test_eval_table = pd.read_parquet(args.test_eval_table)
    # Transformer targets use the same delay-delta normalization as the tabular
    # models, while benchmark metrics use absolute future delays in seconds.
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
    write_yaml(args.model_dir / "eval_metrics.yaml", {"test": eval_metrics_test})


if __name__ == "__main__":
    main()
