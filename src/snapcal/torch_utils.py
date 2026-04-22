"""Torch runtime helpers."""

from __future__ import annotations


def resolve_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_supports_pin_memory(device) -> bool:
    return getattr(device, "type", "cpu") == "cuda"
