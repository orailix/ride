"""Station graph embedding utilities for Gold dataset builders."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def build_station_adjacency_from_node_links(node_links: pd.DataFrame) -> dict[int, set[int]]:
    """Build undirected station adjacency from Silver node links."""
    adjacency: dict[int, set[int]] = {}
    u_nodes = pd.to_numeric(node_links["u_node_id"]).astype(np.int64).to_numpy()
    v_nodes = pd.to_numeric(node_links["v_node_id"]).astype(np.int64).to_numpy()
    for u, v in zip(u_nodes, v_nodes):
        if u not in adjacency:
            adjacency[u] = set()
        if v not in adjacency:
            adjacency[v] = set()
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def compute_laplacian_embeddings(
    adjacency: dict[int, set[int]],
    embedding_dim: int,
) -> tuple[np.ndarray, list[int], nx.Graph]:
    """Compute station embeddings from normalized graph Laplacian eigenvectors."""
    nodes = list(adjacency.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((n, n), dtype=np.float64)
    for u in nodes:
        i = node_to_idx[u]
        for v in adjacency[u]:
            j = node_to_idx[v]
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
    np.fill_diagonal(adj_matrix, 0.0)
    graph = nx.from_numpy_array(adj_matrix)
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    nonzero_mask = np.abs(eigenvalues) >= 1e-12
    nonzero_idx = np.where(nonzero_mask)[0]
    embeddings = eigenvectors[:, nonzero_idx[:embedding_dim]]
    embeddings = embeddings / np.maximum(1e-8, np.linalg.norm(embeddings, axis=1, keepdims=True))
    return embeddings, nodes, graph


def build_positions_from_op_nodes(
    op_nodes: pd.DataFrame,
    node_order: list[int],
) -> dict[int, tuple[float, float]]:
    """Return graph drawing positions by embedding row index as `(lon, lat)`."""
    op_nodes = op_nodes.set_index("op_id")
    positions: dict[int, tuple[float, float]] = {}
    for idx, op_id in enumerate(node_order):
        row = op_nodes.loc[int(op_id)]
        positions[idx] = (float(row["lon"]), float(row["lat"]))
    return positions


def create_station_embeddings_from_silver(
    *,
    node_links_path: Path,
    op_nodes_path: Path,
    embedding_dim: int,
) -> tuple[np.ndarray, list[int], nx.Graph, dict[int, tuple[float, float]], dict[int, set[int]]]:
    """Create station embeddings from Silver infrastructure tables.

    Args:
        node_links_path: Path to the Silver node-link parquet table.
        op_nodes_path: Path to the Silver operational-node parquet table.
        embedding_dim: Number of non-trivial Laplacian components to keep.

    Returns:
        Tuple containing the embedding matrix, operational node id order,
        NetworkX station graph, plotting positions keyed by embedding row index,
        and station adjacency keyed by operational node id.
    """
    node_links = pd.read_parquet(node_links_path, columns=["u_node_id", "v_node_id"])
    adjacency = build_station_adjacency_from_node_links(node_links)
    embeddings, node_order, graph = compute_laplacian_embeddings(adjacency, embedding_dim)
    op_nodes = pd.read_parquet(op_nodes_path, columns=["op_id", "lat", "lon"])
    positions = build_positions_from_op_nodes(op_nodes, node_order)
    return embeddings, node_order, graph, positions, adjacency
