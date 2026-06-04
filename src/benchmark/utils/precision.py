"""Precision and autocast helpers for PyTorch benchmark training."""

from __future__ import annotations

from contextlib import nullcontext

import torch


SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16")


def normalize_precision(precision: str) -> str:
    """Validate and normalize a precision string from the CLI."""
    normalized = str(precision).lower()
    if normalized not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{precision}'. Expected one of: {', '.join(SUPPORTED_PRECISIONS)}."
        )
    return normalized


def autocast_context(*, device: torch.device, precision: str):
    """Return the autocast context matching the selected device and precision."""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
