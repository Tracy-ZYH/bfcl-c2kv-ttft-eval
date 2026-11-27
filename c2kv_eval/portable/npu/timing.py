"""Shared timing helpers; device synchronization is injectable for tests."""
from __future__ import annotations

from typing import Any


def synchronize_accelerator(torch_module: Any, device: str) -> None:
    """Synchronize the selected accelerator before a host-wall boundary."""
    kind = device.split(":", 1)[0].lower()
    if kind == "npu":
        backend = getattr(torch_module, "npu", None)
    elif kind == "cuda":
        backend = getattr(torch_module, "cuda", None)
    else:
        return
    synchronize = getattr(backend, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(f"{kind} timing requires an available synchronize()")
    synchronize()
