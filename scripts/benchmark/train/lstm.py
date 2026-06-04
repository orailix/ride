"""Train the LSTM sequence-to-sequence benchmark model.

The script trains on Gold sequential arrays, optionally holds out the temporally
last training snapshots for validation, reports validation MAE after converting
normalized delay-delta targets back to seconds, and saves the checkpoint used by
the LSTM evaluation entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.benchmark.models.lstm import (
    LSTMSeq2SeqRegressor,
    Seq2SeqDataset,
    evaluate_val_set,
    load_split_arrays,
    set_seed,
    split_train_arrays,
)
from src.benchmark.utils.precision import SUPPORTED_PRECISIONS, autocast_context, normalize_precision
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM seq2seq regressor on sequential gold arrays.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x_static,x_past_seq,x_future_known_seq,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where model outputs/metrics are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader workers.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of train snapshots held out for validation.")
    parser.add_argument("--precision", choices=list(SUPPORTED_PRECISIONS), default="fp32", help="Training precision policy.")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--static-hidden-dim", type=int, default=128, help="Static encoder hidden size.")
    parser.add_argument("--static-out-dim", type=int, default=64, help="Static encoder output size.")
    parser.add_argument("--head-hidden-dim", type=int, default=256, help="Per-step head hidden size.")
    args = parser.parse_args()
    precision = normalize_precision(args.precision)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    scheme = read_yaml(args.data_dir / "scheme.yaml")
    normalization = read_yaml(args.data_dir / "normalization.yaml")
    y_cols = list(scheme["y_columns"])
    # Training targets are normalized delay deltas; validation reporting uses
    # these statistics to express MAE in seconds.
    target_mean = torch.tensor(
        [float(normalization[y_col]["mean"]) for y_col in y_cols],
        dtype=torch.float32,
    )
    target_std = torch.tensor(
        [float(normalization[y_col]["std"]) for y_col in y_cols],
        dtype=torch.float32,
    )
    target_use_sqrt = torch.tensor(
        [bool(normalization[y_col].get("sqrt", False)) for y_col in y_cols],
        dtype=torch.bool,
    )

    x_train_static, x_train_past, x_train_future, y_train, md_train, y_train_mask = load_split_arrays(args.data_dir, "train")
    (
        x_train_static,
        x_train_past,
        x_train_future,
        y_train,
        y_train_mask,
        x_val_static,
        x_val_past,
        x_val_future,
        y_val,
        y_val_mask,
    ) = split_train_arrays(
        x_static=x_train_static,
        x_past=x_train_past,
        x_future_known=x_train_future,
        y=y_train,
        y_mask=y_train_mask,
        md=md_train,
        val_fraction=args.val_fraction,
    )

    train_ds = Seq2SeqDataset(
        x_static=x_train_static,
        x_past=x_train_past,
        x_future_known=x_train_future,
        y=y_train,
        y_mask=y_train_mask,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    has_validation = len(x_val_static) > 0
    if not has_validation:
        val_loader = None
    else:
        val_ds = Seq2SeqDataset(
            x_static=x_val_static,
            x_past=x_val_past,
            x_future_known=x_val_future,
            y=y_val,
            y_mask=y_val_mask,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)
    target_use_sqrt = target_use_sqrt.to(device)
    model = LSTMSeq2SeqRegressor(
        past_input_dim=x_train_past.shape[2],
        static_dim=x_train_static.shape[1],
        future_known_step_dim=x_train_future.shape[2],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        static_hidden_dim=args.static_hidden_dim,
        static_out_dim=args.static_out_dim,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_mae_seconds: list[float] = []
    best_val_mae_s = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for xs, xp, xf, yb, mb in train_loader:
            xs = xs.to(device, non_blocking=True)
            xp = xp.to(device, non_blocking=True)
            xf = xf.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device=device, precision=precision):
                pred = model(xs, xp, xf)
                # The loss mask removes placeholder future targets while keeping
                # all non-placeholder horizons from the same training row.
                if not torch.any(mb):
                    continue
                loss = criterion(pred[mb], yb[mb])
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        train_losses.append(train_loss)
        if has_validation:
            val_loss, val_mae_s = evaluate_val_set(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                target_mean=target_mean,
                target_std=target_std,
                target_use_sqrt=target_use_sqrt,
                precision=precision,
            )
            val_losses.append(val_loss)
            val_mae_seconds.append(val_mae_s)
        if not args.no_verbose:
            if has_validation:
                print(
                    f"[lstm_seq2seq] epoch={epoch}/{args.epochs} "
                    f"train_l1={train_loss:.6f} val_l1={val_loss:.6f} "
                    f"val_mae_s={val_mae_s:.3f}"
                )
            else:
                print(f"[lstm_seq2seq] epoch={epoch}/{args.epochs} train_l1={train_loss:.6f}")
        if has_validation:
            bad_epochs = 0 if val_mae_s < best_val_mae_s else bad_epochs + 1
            best_val_mae_s = min(best_val_mae_s, val_mae_s)
            if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
                break
    torch.save(
        {
            "state_dict": model.state_dict(),
            "past_input_dim": int(x_train_past.shape[2]),
            "static_dim": int(x_train_static.shape[1]),
            "future_known_step_dim": int(x_train_future.shape[2]),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "static_hidden_dim": int(args.static_hidden_dim),
            "static_out_dim": int(args.static_out_dim),
            "head_hidden_dim": int(args.head_hidden_dim),
            "precision": precision,
            "y_columns": y_cols,
        },
        args.output_dir / "model.pt",
    )
    write_yaml(
        args.output_dir / "train_history.yaml",
        {
            "train_l1_per_epoch": train_losses,
            "val_l1_per_epoch": val_losses,
            "val_mae_seconds_per_epoch": val_mae_seconds,
            "val_fraction": float(args.val_fraction),
            "n_train_rows": int(len(x_train_static)),
            "n_val_rows": int(len(x_val_static)),
        },
    )


if __name__ == "__main__":
    main()
