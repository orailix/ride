"""LSTM benchmark model utilities."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.benchmark.utils.precision import autocast_context
from src.dataset.pipeline.helpers import read_yaml


def load_split_arrays(
    data_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one Gold sequential split from numpy arrays."""
    split_dir = data_dir / split
    x_static = np.load(split_dir / "x_static.npy")
    x_past = np.load(split_dir / "x_past_seq.npy")
    x_future_known = np.load(split_dir / "x_future_known_seq.npy")
    y = np.load(split_dir / "y.npy")
    md = np.load(split_dir / "md.npy", allow_pickle=True)
    y_mask = np.load(split_dir / "y_mask.npy")
    return x_static, x_past, x_future_known, y, md, y_mask


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for one benchmark run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Seq2SeqDataset(Dataset):
    """Torch dataset for static, past-sequence, future-known-sequence, and target arrays."""

    def __init__(
        self,
        *,
        x_static: np.ndarray,
        x_past: np.ndarray,
        x_future_known: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
    ) -> None:
        """Convert numpy arrays to contiguous tensors used by the LSTM trainer."""
        self.x_static = torch.from_numpy(np.ascontiguousarray(x_static.astype(np.float32, copy=False)))
        self.x_past = torch.from_numpy(np.ascontiguousarray(x_past.astype(np.float32, copy=False)))
        self.x_future_known = torch.from_numpy(np.ascontiguousarray(x_future_known.astype(np.float32, copy=False)))
        self.y = torch.from_numpy(np.ascontiguousarray(y.astype(np.float32, copy=False)))
        self.y_mask = torch.from_numpy(np.ascontiguousarray(y_mask.astype(bool, copy=True)))

    def __len__(self) -> int:
        """Return the number of train-snapshot rows."""
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one static/past/future-known/target/mask training item."""
        return self.x_static[idx], self.x_past[idx], self.x_future_known[idx], self.y[idx], self.y_mask[idx]


class LSTMSeq2SeqRegressor(nn.Module):
    """Sequence-to-sequence LSTM that predicts one delay-delta target per future event."""

    def __init__(
        self,
        *,
        past_input_dim: int,
        static_dim: int,
        future_known_step_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        static_hidden_dim: int,
        static_out_dim: int,
        head_hidden_dim: int,
    ) -> None:
        """Initialize the past encoder, static encoder, decoder, and per-step head.

        Args:
            past_input_dim: Feature dimension of each past-event sequence step.
            static_dim: Feature dimension of the per-row static context.
            future_known_step_dim: Feature dimension of each future-known sequence step.
            hidden_dim: Hidden dimension used by encoder and decoder LSTMs.
            num_layers: Number of LSTM layers in encoder and decoder.
            dropout: Dropout used in the static encoder, decoder, and prediction head.
            static_hidden_dim: Hidden dimension of the static context MLP.
            static_out_dim: Output dimension of the static context embedding.
            head_hidden_dim: Hidden dimension of the per-future-event prediction head.
        """
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=past_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.enc_att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim, static_out_dim),
            nn.ReLU(),
        )
        self.h_bridge = nn.Linear(hidden_dim, hidden_dim)
        self.c_bridge = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=future_known_step_dim + hidden_dim + static_out_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x_static: torch.Tensor, x_past: torch.Tensor, x_future_known: torch.Tensor) -> torch.Tensor:
        """Predict normalized delay deltas for all configured future-event slots."""
        B, H, F = x_future_known.shape

        enc_out, (h_n, c_n) = self.encoder(x_past)
        att_w = torch.softmax(self.enc_att(enc_out).squeeze(-1), dim=1)
        # Attention compresses the past-event sequence into one context vector.
        enc_ctx = (enc_out * att_w.unsqueeze(-1)).sum(dim=1)  
        s_emb = self.static_mlp(x_static)

        # The decoder receives future-known features, a normalized step position, and row context.
        ctx = torch.cat([enc_ctx, s_emb], dim=1)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, H, -1).contiguous()
        pos = torch.linspace(0, 1, H, device=x_future_known.device).view(1, H, 1).expand(B, H, 1)
        h0 = torch.tanh(self.h_bridge(h_n))
        c0 = torch.tanh(self.c_bridge(c_n))

        dec_in = torch.cat([x_future_known, pos, ctx_expanded], dim=2)
        dec_out, _ = self.decoder(dec_in,(h0, c0))
        y_steps = self.step_head(dec_out).squeeze(-1)
        return y_steps

@torch.no_grad()
def predict_in_batches(
    *,
    model: nn.Module,
    x_static: np.ndarray,
    x_past: np.ndarray,
    x_future_known: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run sequential arrays through the model in batches."""
    model.eval()
    out: list[np.ndarray] = []
    n = x_static.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = torch.from_numpy(x_static[start:end]).to(device)
        xp = torch.from_numpy(x_past[start:end]).to(device)
        xf = torch.from_numpy(x_future_known[start:end]).to(device)
        pred = model(xs, xp, xf).cpu().numpy()
        out.append(pred)
    return np.vstack(out) if out else np.empty((0, 0), dtype=np.float32)


def split_train_arrays(
    *,
    x_static: np.ndarray,
    x_past: np.ndarray,
    x_future_known: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    md: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequential train arrays by holding out the temporally last snapshots."""
    if val_fraction <= 0.0:
        return (
            x_static,
            x_past,
            x_future_known,
            y,
            y_mask,
            x_static[:0],
            x_past[:0],
            x_future_known[:0],
            y[:0],
            y_mask[:0],
        )
    snapshot_ts = pd.DatetimeIndex(pd.to_datetime(md[:, 0]))
    unique_snapshots = np.sort(snapshot_ts.unique())
    n_val = max(1, int(round(len(unique_snapshots) * val_fraction)))
    n_val = min(n_val, len(unique_snapshots) - 1)
    val_snapshots = set(unique_snapshots[-n_val:])
    val_mask = np.asarray(snapshot_ts.isin(val_snapshots), dtype=bool)
    train_mask = ~val_mask

    return (
        x_static[train_mask],
        x_past[train_mask],
        x_future_known[train_mask],
        y[train_mask],
        y_mask[train_mask],
        x_static[val_mask],
        x_past[val_mask],
        x_future_known[val_mask],
        y[val_mask],
        y_mask[val_mask],
    )


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
    for xs, xp, xf, yb, mb in loader:
        xs = xs.to(device, non_blocking=True)
        xp = xp.to(device, non_blocking=True)
        xf = xf.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        if not torch.any(mb):
            continue
        with autocast_context(device=device, precision=precision):
            pred = model(xs, xp, xf)
            loss = criterion(pred[mb], yb[mb])
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
        abs_err_sum += float(torch.abs(pred_sec[mb] - true_sec[mb]).sum().item())
        n_values += int(mb.sum().item())
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
    """Convert normalized LSTM predictions into an eval-table-shaped prediction table."""
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
