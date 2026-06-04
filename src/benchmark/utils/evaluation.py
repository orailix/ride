"""Evaluate benchmark predictions against a Gold evaluation table."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEY_COLUMNS = ["ts", "train_id", "service_date"]


def _sorted_future_delay_cols(df: pd.DataFrame) -> list[str]:
    """Return future-delay columns sorted by their horizon index."""
    cols = [c for c in df.columns if c.startswith("future_delay_")]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _sorted_future_obs_cols(df: pd.DataFrame) -> list[str]:
    """Return future-observation timestamp columns sorted by their horizon index."""
    cols = [c for c in df.columns if c.startswith("future_obs_ts_")]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _mae_rmse(diff: np.ndarray) -> tuple[float, float]:
    """Compute MAE and RMSE from a vector of prediction errors."""
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    return mae, rmse


def _validate_inputs(
    *,
    eval_table: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Validate required key and target columns before evaluation."""
    missing_keys_eval = [c for c in KEY_COLUMNS if c not in eval_table.columns]
    missing_keys_pred = [c for c in KEY_COLUMNS if c not in predictions.columns]
    if missing_keys_eval:
        raise KeyError(f"Missing key columns in eval_table: {missing_keys_eval}")
    if missing_keys_pred:
        raise KeyError(f"Missing key columns in predictions: {missing_keys_pred}")

    target_cols = _sorted_future_delay_cols(eval_table)
    pred_cols = _sorted_future_delay_cols(predictions)
    if target_cols != pred_cols:
        raise KeyError(
            f"Prediction columns must match eval columns. eval={target_cols}, pred={pred_cols}"
        )
    return target_cols, _sorted_future_obs_cols(eval_table)


def _merge_eval_predictions(
    *,
    eval_table: pd.DataFrame,
    predictions: pd.DataFrame,
    target_cols: list[str],
    obs_cols: list[str],
) -> pd.DataFrame:
    """Join evaluation rows and predictions on benchmark key columns."""
    merged = eval_table[KEY_COLUMNS + ["last_known_delay"] + target_cols + obs_cols].merge(
        predictions[KEY_COLUMNS + target_cols],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if merged.empty:
        raise ValueError("No overlapping rows between eval_table and predictions on key columns.")
    return merged


def _validate_prediction_coverage(
    *,
    merged: pd.DataFrame,
    target_cols: list[str],
) -> None:
    """Require predictions wherever the evaluation table has a non-missing target."""
    missing_counts: dict[str, int] = {}
    for col in target_cols:
        y_true = pd.to_numeric(merged[f"{col}_true"], errors="coerce").to_numpy(dtype=np.float64)
        y_pred = pd.to_numeric(merged[f"{col}_pred"], errors="coerce").to_numpy(dtype=np.float64)
        missing_pred = np.isfinite(y_true) & ~np.isfinite(y_pred)
        n_missing = int(missing_pred.sum())
        if n_missing:
            missing_counts[col] = n_missing
    if missing_counts:
        raise ValueError(
            "Predictions contain missing/non-finite values for non-missing eval targets: "
            f"{missing_counts}"
        )


def _key_alignment_counts(
    *,
    eval_table: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[int, int, int]:
    """Count matched and unmatched unique prediction keys."""
    eval_keys = eval_table[KEY_COLUMNS].drop_duplicates()
    pred_keys = predictions[KEY_COLUMNS].drop_duplicates()
    joined = eval_keys.merge(pred_keys, on=KEY_COLUMNS, how="outer", indicator=True)
    matched = int((joined["_merge"] == "both").sum())
    unmatched_eval = int((joined["_merge"] == "left_only").sum())
    unmatched_pred = int((joined["_merge"] == "right_only").sum())
    return matched, unmatched_eval, unmatched_pred


def _validate_key_alignment(
    *,
    eval_table: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[int, int, int]:
    """Require prediction keys to exactly match evaluation keys."""
    matched, unmatched_eval, unmatched_pred = _key_alignment_counts(
        eval_table=eval_table,
        predictions=predictions,
    )
    if unmatched_eval or unmatched_pred:
        raise ValueError(
            "Prediction keys must exactly match eval keys: "
            f"matched={matched}, unmatched_eval={unmatched_eval}, unmatched_pred={unmatched_pred}"
        )
    return matched, unmatched_eval, unmatched_pred


def _global_metrics(
    *,
    merged: pd.DataFrame,
    target_cols: list[str],
) -> tuple[float, float, int]:
    """Compute global metrics over all non-placeholder future-delay targets."""
    y_true = np.column_stack([merged[f"{c}_true"].to_numpy(dtype=np.float64) for c in target_cols])
    y_pred = np.column_stack([merged[f"{c}_pred"].to_numpy(dtype=np.float64) for c in target_cols])
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    diff = (y_pred - y_true)[valid]
    if diff.size == 0:
        raise ValueError("No valid target/prediction pairs after NaN filtering.")
    mae, rmse = _mae_rmse(diff)
    return mae, rmse, int(diff.size)


def _metrics_per_horizon_minutes(
    *,
    merged: pd.DataFrame,
    target_cols: list[str],
    obs_cols: list[str],
    edges_min: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Compute metrics after grouping targets by snapshot-to-event horizon."""
    out: dict[str, dict[str, float | int]] = {}
    if not obs_cols or len(obs_cols) != len(target_cols):
        return out

    ts_vals = pd.to_datetime(merged["ts"], errors="coerce")
    labels = [f"[{int(edges_min[i])},{'inf' if np.isinf(edges_min[i + 1]) else int(edges_min[i + 1])})m" for i in range(len(edges_min) - 1)]
    bucket_diffs: dict[str, list[np.ndarray]] = {k: [] for k in labels}

    for t_col, o_col in zip(target_cols, obs_cols):
        y_t = merged[f"{t_col}_true"].to_numpy(dtype=np.float64)
        y_p = merged[f"{t_col}_pred"].to_numpy(dtype=np.float64)
        o_t = pd.to_datetime(merged[o_col], errors="coerce")
        horizon_min = (o_t - ts_vals).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
        valid = np.isfinite(y_t) & np.isfinite(y_p) & np.isfinite(horizon_min)
        if not np.any(valid):
            continue
        d = y_p[valid] - y_t[valid]
        idx = np.searchsorted(edges_min, horizon_min[valid], side="right") - 1
        idx = np.clip(idx, 0, len(labels) - 1)
        for i, label in enumerate(labels):
            mask = idx == i
            if np.any(mask):
                bucket_diffs[label].append(d[mask])

    for label, parts in bucket_diffs.items():
        if not parts:
            continue
        d = np.concatenate(parts)
        m, r = _mae_rmse(d)
        out[label] = {"mae": m, "rmse": r, "n_values": int(d.size)}
    return out


def _metrics_per_delay_delta_bin(
    *,
    merged: pd.DataFrame,
    target_cols: list[str],
    delta_edges: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Compute metrics after grouping targets by future delay change from snapshot time."""
    out: dict[str, dict[str, float | int]] = {}
    delta_labels = [
        f"[{'-inf' if np.isneginf(delta_edges[i]) else int(delta_edges[i])},"
        f"{'inf' if np.isposinf(delta_edges[i + 1]) else int(delta_edges[i + 1])})s"
        for i in range(len(delta_edges) - 1)
    ]
    delta_parts: dict[str, list[np.ndarray]] = {k: [] for k in delta_labels}
    last_known = pd.to_numeric(merged["last_known_delay"], errors="coerce").to_numpy(dtype=np.float64)
    for col in target_cols:
        y_t = pd.to_numeric(merged[f"{col}_true"], errors="coerce").to_numpy(dtype=np.float64)
        y_p = pd.to_numeric(merged[f"{col}_pred"], errors="coerce").to_numpy(dtype=np.float64)
        delta_t = y_t - last_known
        valid = np.isfinite(y_t) & np.isfinite(y_p) & np.isfinite(delta_t)
        if not np.any(valid):
            continue
        d = y_p[valid] - y_t[valid]
        idx = np.searchsorted(delta_edges, delta_t[valid], side="right") - 1
        idx = np.clip(idx, 0, len(delta_labels) - 1)
        for i, label in enumerate(delta_labels):
            mask = idx == i
            if np.any(mask):
                delta_parts[label].append(d[mask])
    for label, parts in delta_parts.items():
        if not parts:
            continue
        d = np.concatenate(parts)
        m, r = _mae_rmse(d)
        out[label] = {"mae": m, "rmse": r, "n_values": int(d.size)}
    return out


def evaluate_delay_predictions(
    *,
    eval_table: pd.DataFrame,
    predictions: pd.DataFrame,
    delta_edges: np.ndarray = np.array(
        [-np.inf, -300, -120, -60, -30, 0, 30, 60, 120, 300, 600, np.inf],
        dtype=np.float64,
    ),
    edges_min: np.ndarray = np.array(
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, np.inf],
        dtype=np.float64,
    ),
) -> dict[str, float | int]:
    """Evaluate delay predictions against eval_table (global + horizon metrics)."""
    target_cols, obs_cols = _validate_inputs(eval_table=eval_table, predictions=predictions)
    matched_keys, unmatched_eval, _unmatched_pred = _validate_key_alignment(
        eval_table=eval_table,
        predictions=predictions,
    )
    print(f"[evaluation] Key alignment OK: matched={matched_keys}")
    merged = _merge_eval_predictions(
        eval_table=eval_table,
        predictions=predictions,
        target_cols=target_cols,
        obs_cols=obs_cols,
    )
    _validate_prediction_coverage(merged=merged, target_cols=target_cols)
    mae, rmse, n_values = _global_metrics(merged=merged, target_cols=target_cols)
    per_horizon_minutes = _metrics_per_horizon_minutes(
        merged=merged,
        target_cols=target_cols,
        obs_cols=obs_cols,
        edges_min=edges_min,
    )
    per_delay_delta_bin = _metrics_per_delay_delta_bin(
        merged=merged,
        target_cols=target_cols,
        delta_edges=delta_edges,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "n_rows_matched": int(len(merged)),
        "n_rows_unmatched": unmatched_eval,
        "n_values_evaluated": n_values,
        "per_horizon_minutes": per_horizon_minutes,
        "per_delay_delta_bin": per_delay_delta_bin,
    }
