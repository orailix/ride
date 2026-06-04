"""XGBoost benchmark model utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_split_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one Gold tabular split from numpy arrays."""
    split_dir = data_dir / split
    x = np.load(split_dir / "x.npy")
    y = np.load(split_dir / "y.npy")
    md = np.load(split_dir / "md.npy", allow_pickle=True)
    y_mask = np.load(split_dir / "y_mask.npy")
    return x, y, md, y_mask


def split_train_arrays(
    *,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    md: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split tabular train arrays by holding out the temporally last snapshots."""
    if val_fraction <= 0.0:
        return x, y, mask, x[:0], y[:0], mask[:0]
    snapshot_ts = pd.DatetimeIndex(pd.to_datetime(md[:, 0]))
    unique_snapshots = np.sort(snapshot_ts.unique())
    n_val = max(1, int(round(len(unique_snapshots) * val_fraction)))
    n_val = min(n_val, len(unique_snapshots) - 1)
    val_snapshots = set(unique_snapshots[-n_val:])
    val_mask = np.asarray(snapshot_ts.isin(val_snapshots), dtype=bool)
    train_mask = ~val_mask

    return (
        x[train_mask],
        y[train_mask],
        mask[train_mask],
        x[val_mask],
        y[val_mask],
        mask[val_mask],
    )


def compute_mae_seconds(
    *,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mean_t: float,
    std_t: float,
    use_sqrt: bool,
) -> tuple[float, int]:
    """Compute MAE in seconds after restoring normalized delay-delta targets."""
    pred_t = y_pred.astype(np.float64, copy=False) * std_t + mean_t
    true_t = y_true.astype(np.float64, copy=False) * std_t + mean_t
    if use_sqrt:
        pred_sec = np.sign(pred_t) * np.square(np.abs(pred_t))
        true_sec = np.sign(true_t) * np.square(np.abs(true_t))
    else:
        pred_sec = pred_t
        true_sec = true_t
    abs_err = np.abs(pred_sec - true_sec)
    return float(np.mean(abs_err)), int(abs_err.size)


def build_prediction_eval_table(
    *,
    md: np.ndarray,
    y_pred: np.ndarray,
    y_columns: list[str],
    eval_table: pd.DataFrame,
    eval_target_columns: list[str],
    target_stats: dict[str, tuple[float, float, bool]],
) -> pd.DataFrame:
    """Convert normalized XGBoost predictions into an eval-table-shaped prediction table."""
    md_df = pd.DataFrame(md, columns=["snapshot_ts", "train_id", "service_date"]).rename(
        columns={"snapshot_ts": "ts"}
    )
    pred_df = pd.DataFrame(y_pred, columns=y_columns)
    baseline = eval_table[["ts", "train_id", "service_date", "last_known_delay"]]
    out = md_df.merge(baseline, on=["ts", "train_id", "service_date"], how="inner")
    pred_seconds: dict[str, np.ndarray] = {}
    for y_col, eval_col in zip(y_columns, eval_target_columns):
        mean_t, std_t, use_sqrt = target_stats[y_col]
        t_hat = pred_df[y_col].to_numpy(dtype=np.float64) * std_t + mean_t
        # Model outputs are delay deltas; benchmark predictions are absolute future delays.
        if use_sqrt:
            delta_hat = np.sign(t_hat) * np.square(np.abs(t_hat))
        else:
            delta_hat = t_hat
        pred_seconds[eval_col] = delta_hat + out["last_known_delay"].to_numpy(dtype=np.float64)
    for eval_col in eval_target_columns:
        out[eval_col] = pred_seconds[eval_col]
    return out[["ts", "train_id", "service_date"] + eval_target_columns]
