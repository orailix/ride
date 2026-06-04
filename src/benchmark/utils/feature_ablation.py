"""Map benchmark input columns to feature families for ablation experiments."""

from __future__ import annotations

from typing import Iterable


FEATURE_FAMILY_CHOICES = [
    "train_information",
    "snapshot_time_features",
    "event_planned_timing",
    "past_delay_features",
    "event_type_features",
    "event_node_embeddings",
    "local_network_context",
    "weather_features",
]

FEATURE_FAMILY_ALIASES = {
    "local_rail_topology": "local_network_context",
}


def feature_family_choices() -> list[str]:
    """Return the supported feature families in reporting order."""
    return list(FEATURE_FAMILY_CHOICES)


def infer_feature_family_columns(x_columns: Iterable[str]) -> dict[str, list[str]]:
    """Assign each model input column to exactly one ablation feature family."""
    family_to_columns = {family: [] for family in FEATURE_FAMILY_CHOICES}
    unmatched: list[str] = []
    for column in x_columns:
        if column.startswith(("operator_", "train_relation_")):
            family_to_columns["train_information"].append(column)
        elif column.startswith("snapshot_"):
            family_to_columns["snapshot_time_features"].append(column)
        elif "planned_delta" in column and column.startswith(("past_", "future_")):
            family_to_columns["event_planned_timing"].append(column)
        elif column.startswith("past_delay_sec_"):
            family_to_columns["past_delay_features"].append(column)
        elif "event_type" in column and column.startswith(("past_", "future_")):
            family_to_columns["event_type_features"].append(column)
        elif "_emb_" in column and column.startswith(("past_", "future_")):
            family_to_columns["event_node_embeddings"].append(column)
        elif column.startswith(("link", "prev_", "next_")):
            family_to_columns["local_network_context"].append(column)
        elif column.startswith("weather_"):
            family_to_columns["weather_features"].append(column)
        else:
            unmatched.append(column)
    if unmatched:
        raise ValueError(f"Could not assign feature family to columns: {unmatched[:10]}")
    return family_to_columns


def build_ablation_column_index(
    *,
    x_columns: list[str],
    ablate_family: str,
) -> tuple[list[int], list[str], list[str]]:
    """Build the kept-column index and column names for one ablation run."""
    ablate_family = FEATURE_FAMILY_ALIASES.get(ablate_family, ablate_family)
    if ablate_family not in FEATURE_FAMILY_CHOICES:
        raise ValueError(
            f"Unknown feature family '{ablate_family}'. "
            f"Expected one of {feature_family_choices()}."
        )
    family_to_columns = infer_feature_family_columns(x_columns)
    removed_columns = family_to_columns[ablate_family]
    removed_column_set = set(removed_columns)
    keep_indices = [i for i, column in enumerate(x_columns) if column not in removed_column_set]
    kept_columns = [x_columns[i] for i in keep_indices]
    if not removed_columns:
        raise ValueError(f"No columns found for ablated family '{ablate_family}'.")
    if not keep_indices:
        raise ValueError(f"Ablation of '{ablate_family}' would remove every input column.")
    return keep_indices, kept_columns, removed_columns
