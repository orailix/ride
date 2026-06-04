"""Evaluate a trained GNN benchmark run on the Gold test split.

The script loads saved hetero-GINE weights, predicts future train-event delays
from test graph chunks, converts normalized edge predictions back to absolute
future delays in seconds, and evaluates against the shared Gold core
``test_eval_table.parquet``.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.benchmark.models.gnn import (
    HeteroGINERegressor,
    build_edge_meta_from_graphs,
    build_prediction_eval_table,
    load_split_graphs,
    predict_in_batches,
)
from src.dataset.pipeline.helpers import write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained hetero-GINE regressor on GNN graph chunks.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train and test GNN graph chunks.")
    parser.add_argument("--test-eval-table", type=Path, required=True, help="Path to test_eval_table.parquet used for final evaluation.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Training run folder containing model.pt and where predictions/metrics are written.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (in snapshots).")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader workers.")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu")
    test_graphs = load_split_graphs(args.data_dir, "test")
    if test_graphs is None:
        raise FileNotFoundError(f"Missing GNN test graphs under {args.data_dir / 'test'}")

    model = HeteroGINERegressor(
        node_input_dims={str(k): int(v) for k, v in checkpoint["node_input_dims"].items()},
        edge_input_dims={
            ast.literal_eval(k): int(v) for k, v in checkpoint["edge_input_dims"].items()
        },
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        gnn_dropout=float(checkpoint["gnn_dropout"]),
        head_dropout=float(checkpoint["head_dropout"]),
        edge_head_hidden_dim=int(checkpoint["edge_head_hidden_dim"]),
        hetero_aggr=str(checkpoint["hetero_aggr"]),
        use_layer_norm=bool(checkpoint["use_layer_norm"]),
    )
    model.load_state_dict(checkpoint["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_pred_test = predict_in_batches(model=model, loader=test_loader, device=device)
    test_meta = build_edge_meta_from_graphs(test_graphs)
    test_eval_table = pd.read_parquet(args.test_eval_table)
    # The model predicts normalized graph-edge targets; the helper restores the
    # benchmark columns and aligns them to the Gold core evaluation rows.
    test_pred_table = build_prediction_eval_table(
        edge_meta=test_meta,
        y_pred_norm=y_pred_test,
        eval_table=test_eval_table,
        target_stats=checkpoint["target_stats"],
    )
    eval_metrics_test = evaluate_delay_predictions(eval_table=test_eval_table, predictions=test_pred_table)

    test_pred_table.to_parquet(args.model_dir / "test_predictions.parquet", index=False)
    write_yaml(args.model_dir / "eval_metrics.yaml", {"test": eval_metrics_test})


if __name__ == "__main__":
    main()
