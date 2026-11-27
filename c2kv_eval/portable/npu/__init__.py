"""Ascend NPU entry points for the frozen BCRO-120 evaluation."""

from .frozen import ConfigError, FrozenBCRO120, load_and_validate

__all__ = ["ConfigError", "FrozenBCRO120", "load_and_validate"]
