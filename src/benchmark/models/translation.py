"""Translation baseline utilities."""

from __future__ import annotations

import pandas as pd


def build_last_known_baseline_predictions(eval_table: pd.DataFrame) -> pd.DataFrame:
    """Predict every future delay as the delay known at the snapshot time."""
    key_cols = ["ts", "train_id", "service_date"]
    target_cols = sorted(
        [c for c in eval_table.columns if c.startswith("future_delay_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if "last_known_delay" not in eval_table.columns:
        raise KeyError("Missing required column: last_known_delay")
    if not target_cols:
        raise KeyError("No target columns found with prefix future_delay_.")

    pred = eval_table[key_cols].copy()
    last_known = pd.to_numeric(eval_table["last_known_delay"], errors="coerce").to_numpy(dtype=float)
    for col in target_cols:
        pred[col] = last_known
    return pred
