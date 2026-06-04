"""Evaluate trained XGBoost horizon models on the Gold test split.

The script loads one saved XGBoost model per prediction horizon, predicts
delay-delta targets on the tabular test arrays, converts them back to absolute
future delays in seconds, and evaluates against the shared Gold core
``test_eval_table.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.benchmark.models.xgboost import (
    build_prediction_eval_table,
    load_split_arrays,
)
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained XGBoost horizon models on the test split.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet used for final evaluation.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Training run folder containing models/ and where predictions/metrics are written.")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.model_dir / "models"

    scheme = read_yaml(args.data_dir / "scheme.yaml")
    y_cols = list(scheme["y_columns"])
    x_test, _, md_test, _ = load_split_arrays(args.data_dir, "test")

    y_pred_test = np.full((x_test.shape[0], len(y_cols)), np.nan, dtype=np.float32)
    for i, y_col in enumerate(y_cols):
        model_i = XGBRegressor()
        model_i.load_model(models_dir / f"{y_col}.json")
        y_pred_test[:, i] = model_i.predict(x_test).astype(np.float32, copy=False)

    normalization = read_yaml(args.data_dir / "normalization.yaml")
    test_eval_table = pd.read_parquet(args.test_eval_table)
    # Each XGBoost model predicts one normalized delay-delta horizon; the helper
    # restores absolute future-delay columns before metric computation.
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
