"""Train the XGBoost benchmark models on Gold tabular arrays.

The script fits one horizon-specific regressor per target column, optionally
uses a temporal validation split for early stopping, reports validation MAE in
seconds, and saves one model file per prediction horizon.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.benchmark.models.xgboost import (
    compute_mae_seconds,
    load_split_arrays,
    split_train_arrays,
)
from src.dataset.pipeline.helpers import read_yaml, write_yaml

from xgboost import XGBRegressor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost regressor on exported gold numpy arrays.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where model outputs/metrics are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of train snapshots held out for validation.")
    parser.add_argument("--early-stopping-rounds", type=int, default=25, help="Early stopping patience on temporal validation MAE.")
    parser.add_argument("--max-depth", type=int, default=8, help="Tree max depth.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--subsample", type=float, default=1.0, help="Row subsampling ratio.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsampling ratio per tree.")
    parser.add_argument("--min-child-weight", type=float, default=5.0, help="Minimum sum of instance weight needed in a child.")
    parser.add_argument("--reg-alpha", type=float, default=1.0, help="L1 regularization.")
    parser.add_argument("--reg-lambda", type=float, default=5.0, help="L2 regularization.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scheme = read_yaml(args.data_dir / "scheme.yaml")
    normalization = read_yaml(args.data_dir / "normalization.yaml")
    y_cols = list(scheme["y_columns"])
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, md_train, y_train_mask = load_split_arrays(args.data_dir, "train")
    y_train_mask = y_train_mask.astype(bool, copy=False)
    x_train, y_train, y_train_mask, x_val, y_val, y_val_mask = split_train_arrays(
        x=x_train,
        y=y_train,
        mask=y_train_mask,
        md=md_train,
        val_fraction=args.val_fraction,
    )
    has_validation = len(x_val) > 0

    train_history: dict[str, float] = {}
    weighted_abs_err = 0.0
    weighted_count = 0
    horizon_iter = tqdm(
        enumerate(y_cols),
        total=len(y_cols),
        desc="Training XGBoost horizons",
        disable=args.no_verbose,
    )
    for i, y_col in horizon_iter:
        valid_train_i = y_train_mask[:, i]
        valid_val_i = y_val_mask[:, i]
        n_valid_train = int(valid_train_i.sum())
        n_valid_val = int(valid_val_i.sum())
        if n_valid_train <= 0:
            raise ValueError(f"No valid training targets for horizon column {y_col}.")
        if has_validation and n_valid_val <= 0:
            raise ValueError(f"No valid validation targets for horizon column {y_col}.")
        model_i = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            random_state=args.seed,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds if has_validation else None,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            n_jobs=args.n_jobs,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        y_train_i = y_train[:, i]
        y_val_i = y_val[:, i]
        # Sample weights remove placeholder future targets for this horizon
        # while keeping the shared training rows passed to XGBoost.
        w_train_i = valid_train_i.astype(np.float32, copy=False)
        w_val_i = valid_val_i.astype(np.float32, copy=False)
        model_i.fit(
            x_train,
            y_train_i,
            sample_weight=w_train_i,
            eval_set=[(x_val, y_val_i)] if has_validation else None,
            sample_weight_eval_set=[w_val_i] if has_validation else None,
            verbose=False,
        )
        if has_validation:
            stats = normalization[y_col]
            val_mae_seconds, n_valid_values = compute_mae_seconds(
                y_pred=model_i.predict(x_val)[valid_val_i],
                y_true=y_val_i[valid_val_i],
                mean_t=float(stats["mean"]),
                std_t=float(stats["std"]),
                use_sqrt=bool(stats.get("sqrt", False)),
            )
            train_history[y_col] = val_mae_seconds
            weighted_abs_err += val_mae_seconds * n_valid_values
            weighted_count += n_valid_values
        model_i.save_model(models_dir / f"{y_col}.json")
    write_yaml(
        args.output_dir / "train_history.yaml",
        {
            "val_mae_seconds_mean": (weighted_abs_err / weighted_count) if has_validation else None,
            "val_mae_seconds_per_horizon": train_history,
            "val_fraction": float(args.val_fraction),
            "n_train_rows": int(len(x_train)),
            "n_val_rows": int(len(x_val)),
        },
    )


if __name__ == "__main__":
    main()
