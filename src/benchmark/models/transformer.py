"""Transformer benchmark model utilities."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.benchmark.utils.precision import autocast_context
from src.dataset.pipeline.helpers import read_yaml

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
)



def load_split_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one Gold tabular split from numpy arrays."""
    split_dir = data_dir / split
    x = np.load(split_dir / "x.npy")
    y = np.load(split_dir / "y.npy")
    md = np.load(split_dir / "md.npy", allow_pickle=True)
    y_mask = np.load(split_dir / "y_mask.npy")
    return x, y, md, y_mask


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for one benchmark run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_snapshot_groups(md: np.ndarray) -> list[np.ndarray]:
    """Group row indices by snapshot timestamp in temporal order."""
    md_df = pd.DataFrame(md, columns=["snapshot_ts", "train_id", "service_date"])
    snapshot_ts = pd.DatetimeIndex(pd.to_datetime(md_df["snapshot_ts"]))
    md_df = md_df.assign(snapshot_ts=snapshot_ts)
    groups = md_df.groupby("snapshot_ts", sort=False).indices
    ordered_snapshots = np.sort(snapshot_ts.unique())
    return [np.asarray(groups[ts], dtype=np.int64) for ts in ordered_snapshots]


def split_snapshot_groups(
    *,
    row_groups: list[np.ndarray],
    val_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split snapshot groups by holding out the temporally last snapshots."""
    if val_fraction <= 0.0:
        return row_groups, []
    n_groups = len(row_groups)
    n_val = max(1, int(round(n_groups * val_fraction)))
    n_val = min(n_val, n_groups - 1)
    split_idx = n_groups - n_val
    return row_groups[:split_idx], row_groups[split_idx:]


class SnapshotSequenceDataset(Dataset):
    """Dataset where each item contains all tabular rows for one snapshot."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        row_groups: list[np.ndarray],
        y: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        """Store tabular arrays and snapshot-level row groups for grouped batching."""
        self.x = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32, copy=False)))
        self.y = None if y is None else torch.from_numpy(np.ascontiguousarray(y.astype(np.float32, copy=False)))
        self.valid_mask = None if valid_mask is None else torch.from_numpy(np.ascontiguousarray(valid_mask.astype(bool, copy=True)))
        self.row_groups = row_groups

    def __len__(self) -> int:
        """Return the number of snapshot groups."""
        return len(self.row_groups)

    def __getitem__(self, idx: int):
        """Return all rows for one snapshot, optionally with targets and masks."""
        row_idx = self.row_groups[idx]
        row_idx_t = torch.from_numpy(row_idx)
        x = self.x[row_idx_t]
        if self.y is None or self.valid_mask is None:
            return x, row_idx_t
        y = self.y[row_idx_t]
        valid = self.valid_mask[row_idx_t]
        return x, y, valid, row_idx_t


def collate_train(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Pad snapshot groups for training and build the combined loss mask."""
    x_list = [b[0] for b in batch]
    y_list = [b[1] for b in batch]
    valid_list = [b[2] for b in batch]
    lengths = torch.tensor([x.shape[0] for x in x_list], dtype=torch.long)

    x_pad = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(y_list, batch_first=True, padding_value=0.0)
    valid_pad = pad_sequence(valid_list, batch_first=True, padding_value=False)

    padding_mask = torch.arange(x_pad.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    # The loss mask excludes padded rows and placeholder future targets.
    loss_mask = padding_mask.unsqueeze(-1).expand_as(y_pad) | (~valid_pad)
    return x_pad, y_pad, padding_mask, loss_mask


def collate_predict(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad snapshot groups for prediction while keeping original row indices."""
    x_list = [b[0] for b in batch]
    idx_list = [b[1] for b in batch]
    lengths = torch.tensor([x.shape[0] for x in x_list], dtype=torch.long)

    x_pad = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    idx_pad = pad_sequence(idx_list, batch_first=True, padding_value=-1)
    padding_mask = torch.arange(x_pad.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    return x_pad, padding_mask, idx_pad


class TransformerRegressor(nn.Module):
    """Transformer encoder that predicts delay-delta targets for rows in each snapshot."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        """Initialize input projection, Transformer encoder, and output projection.

        Args:
            input_dim: Number of tabular input features.
            output_dim: Number of future-event target columns.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of encoder layers.
            dim_feedforward: Hidden dimension of the encoder feed-forward blocks.
            dropout: Dropout used inside Transformer encoder layers.
        """
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Predict normalized delay deltas for padded snapshot-row batches."""
        h = self.in_proj(x)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.out_proj(h)


def predict_grouped(
    *,
    model: nn.Module,
    loader: DataLoader,
    n_rows: int,
    n_targets: int,
    device: torch.device,
) -> np.ndarray:
    """Predict grouped snapshot batches and scatter outputs back to row order."""
    model.eval()
    out = np.full((n_rows, n_targets), np.nan, dtype=np.float32)
    with torch.no_grad():
        for xb, pad_mask, idxb in loader:
            xb = xb.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            pred = model(xb, padding_mask=pad_mask).cpu().numpy()
            idx_np = idxb.numpy()
            for i in range(pred.shape[0]):
                keep = idx_np[i] >= 0
                # idx_pad maps non-padding positions back to the original flat arrays.
                out[idx_np[i, keep]] = pred[i, keep]
    return out


@torch.no_grad()
def evaluate_val_set(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_use_sqrt: torch.Tensor,
    precision: str,
) -> tuple[float, float]:
    """Evaluate validation loss and MAE after restoring delay deltas to seconds."""
    model.eval()
    loss_running = 0.0
    n_batches = 0
    abs_err_sum = 0.0
    n_values = 0
    for xb, yb, pad_mask, loss_mask in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)
        with autocast_context(device=device, precision=precision):
            pred = model(xb, padding_mask=pad_mask)
            valid = ~loss_mask
            if not torch.any(valid):
                continue
            loss = criterion(pred[valid], yb[valid])
        loss_running += float(loss.item())
        n_batches += 1
        pred_t = pred * target_std + target_mean
        true_t = yb * target_std + target_mean
        pred_sec = torch.where(
            target_use_sqrt,
            torch.sign(pred_t) * torch.square(torch.abs(pred_t)),
            pred_t,
        )
        true_sec = torch.where(
            target_use_sqrt,
            torch.sign(true_t) * torch.square(torch.abs(true_t)),
            true_t,
        )
        abs_err_sum += float(torch.abs(pred_sec[valid] - true_sec[valid]).sum().item())
        n_values += int(valid.sum().item())
    return loss_running / max(n_batches, 1), abs_err_sum / max(n_values, 1)


def build_prediction_eval_table(
    *,
    md: np.ndarray,
    y_pred: np.ndarray,
    y_columns: list[str],
    eval_table: pd.DataFrame,
    eval_target_columns: list[str],
    target_stats: dict[str, tuple[float, float, bool]],
) -> pd.DataFrame:
    """Convert normalized Transformer predictions into an eval-table-shaped prediction table."""
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
