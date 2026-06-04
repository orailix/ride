"""Train the Transformer benchmark model on Gold tabular arrays.

The script groups train rows by snapshot, optionally holds out the temporally
last training snapshots for validation, reports validation MAE after converting
normalized delay-delta targets back to seconds, and saves the checkpoint used by
the Transformer evaluation entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.benchmark.models.transformer import (
    SnapshotSequenceDataset,
    TransformerRegressor,
    build_snapshot_groups,
    collate_train,
    evaluate_val_set,
    load_split_arrays,
    set_seed,
    split_snapshot_groups,
)
from src.benchmark.utils.precision import SUPPORTED_PRECISIONS, autocast_context, normalize_precision
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer regressor (PyTorch, L1 loss) on exported gold numpy arrays.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/test x,y,y_mask,md npy and scheme.yaml.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where model outputs/metrics are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (in snapshots).")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader workers.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of train snapshots held out for validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer hidden size.")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads.")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer layers.")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="FFN hidden size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--precision", choices=list(SUPPORTED_PRECISIONS), default="fp32", help="Training precision policy.")
    parser.add_argument("--no-verbose", action="store_true", help="Disable per-epoch prints.")
    parser.add_argument("--early-stopping-patience", type=int, default=-1, help="Stop after this many epochs without val_mae_s improvement; disabled if <= 0.")
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

    x_train, y_train, md_train, y_train_mask = load_split_arrays(args.data_dir, "train")
    y_train_mask = y_train_mask.astype(bool, copy=False)

    train_groups = build_snapshot_groups(md_train)
    # Transformer batches contain all rows from a snapshot, so validation
    # splitting happens at the snapshot-group level rather than row by row.
    train_groups, val_groups = split_snapshot_groups(
        row_groups=train_groups,
        val_fraction=args.val_fraction,
    )

    train_ds = SnapshotSequenceDataset(x=x_train, y=y_train, valid_mask=y_train_mask, row_groups=train_groups)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
    )
    has_validation = len(val_groups) > 0
    if not has_validation:
        val_loader = None
    else:
        val_ds = SnapshotSequenceDataset(x=x_train, y=y_train, valid_mask=y_train_mask, row_groups=val_groups)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_train,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)
    target_use_sqrt = target_use_sqrt.to(device)
    model = TransformerRegressor(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_mae_seconds: list[float] = []
    best_val_mae_s, bad_epochs = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb, pad_mask, loss_mask in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device=device, precision=precision):
                pred = model(xb, padding_mask=pad_mask)
                valid = ~loss_mask
                # The loss mask removes placeholder future targets while keeping
                # all non-placeholder horizons from the same snapshot batch.
                if not torch.any(valid):
                    continue
                loss = criterion(pred[valid], yb[valid])
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
                    f"[transformer] epoch={epoch}/{args.epochs} "
                    f"train_l1={train_loss:.6f} val_l1={val_loss:.6f} "
                    f"val_mae_s={val_mae_s:.3f}"
                )
            else:
                print(f"[transformer] epoch={epoch}/{args.epochs} train_l1={train_loss:.6f}")
        if has_validation:
            bad_epochs = 0 if val_mae_s < best_val_mae_s else bad_epochs + 1
            best_val_mae_s = min(best_val_mae_s, val_mae_s)
            if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "output_dim": int(y_train.shape[1]),
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "num_layers": int(args.num_layers),
            "dim_feedforward": int(args.dim_feedforward),
            "dropout": float(args.dropout),
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
            "n_train_snapshots": int(len(train_groups)),
            "n_val_snapshots": int(len(val_groups)),
        },
    )


if __name__ == "__main__":
    main()
