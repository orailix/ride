"""Evaluate a trained LSTM benchmark run on the Gold test split.

The script loads a sequence-to-sequence checkpoint, predicts over the sequential
test arrays, converts delay-delta outputs back to absolute future delays in
seconds, and evaluates against the shared Gold core ``test_eval_table.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.benchmark.models.lstm import (
    LSTMSeq2SeqRegressor,
    build_prediction_eval_table,
    load_split_arrays,
    predict_in_batches,
)
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM seq2seq regressor on the sequential test split.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x_static,x_past_seq,x_future_known_seq,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet used for final evaluation.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Training run folder containing model.pt and where predictions/metrics are written.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size.")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu")
    model = LSTMSeq2SeqRegressor(
        past_input_dim=int(checkpoint["past_input_dim"]),
        static_dim=int(checkpoint["static_dim"]),
        future_known_step_dim=int(checkpoint["future_known_step_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        static_hidden_dim=int(checkpoint["static_hidden_dim"]),
        static_out_dim=int(checkpoint["static_out_dim"]),
        head_hidden_dim=int(checkpoint["head_hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])

    x_test_static, x_test_past, x_test_future, _y_test, md_test, _y_test_mask = load_split_arrays(args.data_dir, "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    y_pred_test = predict_in_batches(
        model=model,
        x_static=x_test_static,
        x_past=x_test_past,
        x_future_known=x_test_future,
        batch_size=args.batch_size,
        device=device,
    )

    y_cols = [str(c) for c in checkpoint["y_columns"]]
    normalization = read_yaml(args.data_dir / "normalization.yaml")
    test_eval_table = pd.read_parquet(args.test_eval_table)
    # Sequential models are trained on delay-delta targets; the evaluator
    # expects absolute future-delay columns from the Gold core.
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
