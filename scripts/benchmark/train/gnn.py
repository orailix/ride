"""Train the hetero-GINE model used by the GNN benchmark.

The script trains on Gold GNN graph chunks, optionally holds out the temporally
last training snapshots for validation, reports validation MAE after converting
the normalized graph-edge target back to seconds, and saves the checkpoint plus
the target statistics needed by the evaluation script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from src.benchmark.models.gnn import (
    HeteroGINERegressor,
    REL_FUTURE,
    REL_FUTURE_REV,
    REL_PAST,
    REL_PAST_REV,
    REL_STS,
    REL_STS_REV,
    evaluate_val_set,
    load_split_graphs,
    set_seed,
    split_train_graphs,
    strip_eval_metadata,
)
from src.benchmark.utils.precision import SUPPORTED_PRECISIONS, autocast_context, normalize_precision
from src.dataset.pipeline.helpers import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hetero-GINE regressor on GNN graph chunks."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train and test GNN graph chunks.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where model outputs/metrics are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (in snapshots).")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader workers.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of train snapshots held out for validation.")
    parser.add_argument("--precision", choices=list(SUPPORTED_PRECISIONS), default="fp32", help="Training precision policy.")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Node hidden size.")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of hetero GINE layers.")
    parser.add_argument("--gnn-dropout", type=float, default=0.1, help="Backbone/message-passing dropout rate.")
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Prediction head dropout rate.")
    parser.add_argument("--edge-head-hidden-dim", type=int, default=256, help="Edge MLP head hidden size.")
    parser.add_argument("--hetero-aggr", choices=["sum", "mean"], default="sum", help="Relation aggregation used by HeteroConv.")
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True, help="Enable LayerNorm after each residual node update.")
    args = parser.parse_args()
    precision = normalize_precision(args.precision)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    normalization = read_yaml(args.data_dir / "normalization.yaml")
    train_graphs = load_split_graphs(args.data_dir, "train")
    # Evaluation metadata is not needed during training and should not be part
    # of the tensors seen by the model.
    strip_eval_metadata(train_graphs)
    feature_spec = read_yaml(args.data_dir / "feature_spec.yaml")

    train_graphs, val_graphs = split_train_graphs(
        graphs=train_graphs,
        val_fraction=args.val_fraction,
    )

    target_norm = normalization["train_to_future_station_edges"]
    target_stats = target_norm["future_delay_delta"]

    node_input_dims = {
        "train": len(feature_spec["train_node_cols"]),
        "station": len(feature_spec["station_node_cols"]),
    }
    edge_input_dims = {
        REL_STS: len(feature_spec["sts_edge_cols"]),
        REL_STS_REV: len(feature_spec["sts_edge_cols"]),
        REL_PAST: len(feature_spec["past_edge_cols"]),
        REL_PAST_REV: len(feature_spec["past_edge_cols"]),
        REL_FUTURE: len(feature_spec["future_edge_cols"]),
        REL_FUTURE_REV: len(feature_spec["future_edge_cols"]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mean = torch.tensor(float(target_stats["mean"]), dtype=torch.float32, device=device)
    target_std = torch.tensor(float(target_stats["std"]), dtype=torch.float32, device=device)
    target_use_sqrt = bool(target_stats.get("sqrt", False))
    model = HeteroGINERegressor(
        node_input_dims=node_input_dims,
        edge_input_dims=edge_input_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_dropout=args.gnn_dropout,
        head_dropout=args.head_dropout,
        edge_head_hidden_dim=args.edge_head_hidden_dim,
        hetero_aggr=args.hetero_aggr,
        use_layer_norm=args.use_layer_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    has_validation = len(val_graphs) > 0
    if not has_validation:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_mae_seconds: list[float] = []
    best_val_mae_s = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device=device, precision=precision):
                pred = model(batch)
                target = batch[REL_FUTURE].y
                if target.numel() == 0:
                    continue
                loss = criterion(pred, target)
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
                    f"[gnn-gine] epoch={epoch}/{args.epochs} "
                    f"train_l1={train_loss:.6f} val_l1={val_loss:.6f} "
                    f"val_mae_s={val_mae_s:.3f}"
                )
            else:
                print(f"[gnn-gine] epoch={epoch}/{args.epochs} train_l1={train_loss:.6f}")
        if has_validation:
            bad_epochs = 0 if val_mae_s < best_val_mae_s else bad_epochs + 1
            best_val_mae_s = min(best_val_mae_s, val_mae_s)
            if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "node_input_dims": node_input_dims,
            "edge_input_dims": {str(k): int(v) for k, v in edge_input_dims.items()},
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "gnn_dropout": float(args.gnn_dropout),
            "head_dropout": float(args.head_dropout),
            "edge_head_hidden_dim": int(args.edge_head_hidden_dim),
            "hetero_aggr": str(args.hetero_aggr),
            "use_layer_norm": bool(args.use_layer_norm),
            "precision": precision,
            "feature_spec": feature_spec,
            "target_stats": target_norm["future_delay_delta"],
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
            "n_train_graphs": int(len(train_graphs)),
            "n_val_graphs": int(len(val_graphs)),
        },
    )


if __name__ == "__main__":
    main()
