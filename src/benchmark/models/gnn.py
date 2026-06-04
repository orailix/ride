"""GNN benchmark model utilities."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, HeteroConv

from src.benchmark.utils.precision import autocast_context
from src.dataset.pipeline.helpers import read_yaml

REL_STS = ("station", "to", "station")
REL_STS_REV = ("station", "to_rev", "station")
REL_PAST = ("train", "past", "station")
REL_PAST_REV = ("station", "past_rev", "train")
REL_FUTURE = ("train", "future", "station")
REL_FUTURE_REV = ("station", "future_rev", "train")


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for one benchmark run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_graphs(data_dir: Path, split: str) -> list[HeteroData] | None:
    """Load all PyTorch Geometric graph chunks for a Gold GNN split."""
    split_dir = data_dir / split
    graph_paths = sorted(split_dir.glob("graphs_part_*.pt"))
    if not graph_paths:
        return None
    graphs: list[HeteroData] = []
    for path in graph_paths:
        graphs.extend(torch.load(path, map_location="cpu", weights_only=False))
    return graphs


def strip_eval_metadata(graphs: list[HeteroData]) -> None:
    """Remove evaluation-only identifiers before training the GNN model."""
    for graph in graphs:
        if hasattr(graph, "snapshot_ts"):
            del graph.snapshot_ts
        if hasattr(graph, "train_ids"):
            del graph.train_ids
        if hasattr(graph, "service_dates"):
            del graph.service_dates
        if hasattr(graph[REL_FUTURE], "future_rank"):
            del graph[REL_FUTURE].future_rank


class HeteroGINERegressor(nn.Module):
    """Heterogeneous GINE regressor that predicts one target per future train-station edge."""

    def __init__(
        self,
        *,
        node_input_dims: dict[str, int],
        edge_input_dims: dict[tuple[str, str, str], int],
        hidden_dim: int,
        num_layers: int,
        gnn_dropout: float,
        head_dropout: float,
        edge_head_hidden_dim: int,
        hetero_aggr: str,
        use_layer_norm: bool,
    ) -> None:
        """Initialize node/edge projections, message-passing layers, and the edge head.

        Args:
            node_input_dims: Input feature dimension for each node type.
            edge_input_dims: Input feature dimension for each relation type.
            hidden_dim: Shared hidden dimension for node and edge states.
            num_layers: Number of hetero-GINE message-passing layers.
            gnn_dropout: Dropout applied to node updates inside the GNN.
            head_dropout: Dropout applied in the future-edge prediction head.
            edge_head_hidden_dim: Hidden dimension of the future-edge prediction head.
            hetero_aggr: Aggregation used by ``HeteroConv`` across relations.
            use_layer_norm: Whether to apply LayerNorm after each residual node update.
        """
        super().__init__()
        self._node_proj_key_by_type = {ntype: f"ntype__{ntype}" for ntype in node_input_dims}
        self._edge_proj_key_by_type = {rel: f"rel__{rel[0]}__{rel[1]}__{rel[2]}" for rel in edge_input_dims}
        self._edge_update_key_by_type = {
            rel: f"edge_update__{rel[0]}__{rel[1]}__{rel[2]}" for rel in edge_input_dims
        }
        self._layer_norm_key_by_type = {ntype: f"norm__{ntype}" for ntype in node_input_dims}
        self.node_proj = nn.ModuleDict(
            {
                self._node_proj_key_by_type[ntype]: nn.Linear(in_dim, hidden_dim)
                for ntype, in_dim in node_input_dims.items()
            }
        )
        self.edge_proj = nn.ModuleDict(
            {
                self._edge_proj_key_by_type[rel]: nn.Linear(in_dim, hidden_dim)
                for rel, in_dim in edge_input_dims.items()
            }
        )
        relations = [REL_STS, REL_STS_REV, REL_PAST, REL_PAST_REV, REL_FUTURE, REL_FUTURE_REV]
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.edge_updates = nn.ModuleList()
        self.use_layer_norm = use_layer_norm
        for _ in range(num_layers):
            convs = {
                rel: GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=hidden_dim,
                )
                for rel in relations
            }
            self.layers.append(HeteroConv(convs, aggr=hetero_aggr))
            self.edge_updates.append(
                nn.ModuleDict(
                    {
                        self._edge_update_key_by_type[rel]: nn.Sequential(
                            nn.Linear(hidden_dim * 3, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        )
                        for rel in relations
                    }
                )
            )
            self.layer_norms.append(
                nn.ModuleDict(
                    {
                        self._layer_norm_key_by_type[ntype]: nn.LayerNorm(hidden_dim)
                        for ntype in node_input_dims
                    }
                )
            )
        self.gnn_dropout = nn.Dropout(gnn_dropout)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, edge_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(edge_head_hidden_dim, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Predict normalized future delay deltas for the graph future edges."""
        # Project heterogeneous raw node and edge features into one shared hidden space.
        x_dict = {
            ntype: torch.relu(self.node_proj[self._node_proj_key_by_type[ntype]](data[ntype].x))
                for ntype in self._node_proj_key_by_type
        }
        edge_attr_dict = {
            rel: torch.relu(self.edge_proj[self._edge_proj_key_by_type[rel]](data[rel].edge_attr))
            for rel in self._edge_proj_key_by_type
        }
        for layer_idx, conv in enumerate(self.layers):
            prev_x_dict = x_dict
            # Edge states are updated from source node, destination node, and previous edge state.
            edge_attr_dict = {
                rel: self.edge_updates[layer_idx][self._edge_update_key_by_type[rel]](
                    torch.cat(
                        [
                            prev_x_dict[rel[0]][data[rel].edge_index[0]],
                            prev_x_dict[rel[2]][data[rel].edge_index[1]],
                            edge_attr_dict[rel],
                        ],
                        dim=1,
                    )
                )
                for rel in self._edge_update_key_by_type
            }
            # HeteroConv aggregates messages per destination node type across relation types.
            msg_x_dict = conv(prev_x_dict, data.edge_index_dict, edge_attr_dict)
            x_dict = {}
            for ntype, prev_x in prev_x_dict.items():
                msg_x = msg_x_dict.get(ntype)
                if msg_x is None:
                    msg_x = torch.zeros_like(prev_x)
                x = prev_x + self.gnn_dropout(torch.relu(msg_x))
                x_dict[ntype] = self.layer_norms[layer_idx][self._layer_norm_key_by_type[ntype]](x) if self.use_layer_norm else x

        fut_store = data[REL_FUTURE]
        edge_index = fut_store.edge_index
        edge_attr = edge_attr_dict[REL_FUTURE]
        if edge_index.shape[1] == 0:
            return edge_attr.new_empty((0,))
        # The supervised target is attached only to train -> future station edges.
        src = x_dict["train"][edge_index[0]]
        dst = x_dict["station"][edge_index[1]]
        z = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_head(z).squeeze(-1)


@torch.no_grad()
def predict_in_batches(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run graph batches through the model and concatenate edge predictions."""
    model.eval()
    out: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu().numpy()
        out.append(pred)
    if not out:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)

def split_train_graphs(
    *,
    graphs: list[HeteroData],
    val_fraction: float,
) -> tuple[list[HeteroData], list[HeteroData]]:
    """Split graph chunks temporally by keeping the last training graphs for validation."""
    if val_fraction <= 0.0:
        return graphs, []
    n_graphs = len(graphs)
    n_val = max(1, int(round(n_graphs * val_fraction)))
    n_val = min(n_val, n_graphs - 1)
    split_idx = n_graphs - n_val
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    return train_graphs, val_graphs


@torch.no_grad()
def evaluate_val_set(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_use_sqrt: bool,
    precision: str,
) -> tuple[float, float]:
    """Evaluate validation loss and MAE after restoring delay deltas to seconds."""
    model.eval()
    loss_running = 0.0
    n_batches = 0
    abs_err_sum = 0.0
    n_values = 0
    for batch in loader:
        batch = batch.to(device)
        target = batch[REL_FUTURE].y
        if target.numel() == 0:
            continue
        with autocast_context(device=device, precision=precision):
            pred = model(batch)
            loss = criterion(pred, target)
        loss_running += float(loss.item())
        n_batches += 1
        pred_t = pred * target_std + target_mean
        true_t = target * target_std + target_mean
        if target_use_sqrt:
            pred_sec = torch.sign(pred_t) * torch.square(torch.abs(pred_t))
            true_sec = torch.sign(true_t) * torch.square(torch.abs(true_t))
        else:
            pred_sec = pred_t
            true_sec = true_t
        abs_err_sum += float(torch.abs(pred_sec - true_sec).sum().item())
        n_values += int(target.numel())
    return loss_running / max(n_batches, 1), abs_err_sum / max(n_values, 1)


def _sorted_future_delay_cols(df: pd.DataFrame) -> list[str]:
    """Return future-delay columns ordered by horizon index."""
    cols = [c for c in df.columns if c.startswith("future_delay_")]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def build_edge_meta_from_graphs(graphs: list[HeteroData]) -> pd.DataFrame:
    """Collect evaluation keys for each future edge prediction in graph order."""
    rows: list[dict[str, object]] = []
    for graph in graphs:
        fut = graph[REL_FUTURE]
        if fut.edge_index.shape[1] == 0:
            continue
        train_idx = fut.edge_index[0].detach().cpu().numpy()
        future_rank = fut.future_rank.detach().cpu().numpy()
        rows.extend(
            {
                "snapshot_ts": int(graph.snapshot_ts),
                "future_rank": int(rank),
                "train_id": str(graph.train_ids[int(idx)]),
                "service_date": str(graph.service_dates[int(idx)]),
            }
            for idx, rank in zip(train_idx, future_rank, strict=True)
        )
    return pd.DataFrame(rows, columns=["snapshot_ts", "future_rank", "train_id", "service_date"])


def build_prediction_eval_table(
    *,
    edge_meta: pd.DataFrame,
    y_pred_norm: np.ndarray,
    eval_table: pd.DataFrame,
    target_stats: dict[str, float | bool],
) -> pd.DataFrame:
    """Convert normalized edge predictions into an eval-table-shaped prediction table."""
    mean_t = float(target_stats["mean"])
    std_t = float(target_stats["std"])
    if std_t == 0.0 or np.isnan(std_t):
        t_hat = y_pred_norm.astype(np.float64, copy=False) + mean_t
    else:
        t_hat = y_pred_norm.astype(np.float64, copy=False) * std_t + mean_t
    delta_pred_sec = np.sign(t_hat) * np.square(np.abs(t_hat))

    df = edge_meta.copy()
    df["ts"] = pd.to_datetime(pd.to_numeric(df["snapshot_ts"]), unit="ns")
    df["prediction_delta"] = delta_pred_sec
    df["horizon_col"] = "future_delay_" + df["future_rank"].astype(int).astype(str)

    wide = (
        df.pivot_table(
            index=["ts", "train_id", "service_date"],
            columns="horizon_col",
            values="prediction_delta",
            aggfunc="first",
        )
        .reset_index()
    )
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [str(c) for c in wide.columns.get_level_values(0)]
    target_cols = _sorted_future_delay_cols(eval_table)
    for c in target_cols:
        if c not in wide.columns:
            wide[c] = np.nan
    pred_target_cols = [c for c in wide.columns if c.startswith("future_delay_")]
    extra = [c for c in pred_target_cols if c not in target_cols]
    if extra:
        wide = wide.drop(columns=extra)
    wide = wide.loc[:, ["ts", "train_id", "service_date"] + target_cols]
    eval_keys = eval_table.loc[:, ["ts", "train_id", "service_date", "last_known_delay"]].drop_duplicates()  # defensive
    out = eval_keys.merge(wide, on=["ts", "train_id", "service_date"], how="left", validate="one_to_one")
    last_known = pd.to_numeric(out["last_known_delay"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    for col in target_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        out[col] = vals + last_known
    out = out.drop(columns=["last_known_delay"])
    return out
