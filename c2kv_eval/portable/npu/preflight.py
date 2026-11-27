"""Fail-closed ModelArts preflight for the frozen BCRO-120 NPU run."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .frozen import ConfigError, load_and_validate, validate_static_npu_sources


def _version(module: Any) -> str | None:
    return getattr(module, "__version__", None)


def inspect_npu_environment() -> dict[str, Any]:
    failures: list[str] = []
    modules: dict[str, Any] = {}
    for name in ("torch", "torch_npu", "sgl_kernel_npu"):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as error:  # import errors vary across CANN releases
            failures.append(f"cannot import {name}: {type(error).__name__}: {error}")
    torch = modules.get("torch")
    npu_available = False
    device_count = 0
    if torch is not None:
        npu = getattr(torch, "npu", None)
        if npu is None:
            failures.append("torch has no npu device module")
        else:
            try:
                npu_available = bool(npu.is_available())
                device_count = int(npu.device_count())
            except Exception as error:
                failures.append(f"cannot query torch.npu: {type(error).__name__}: {error}")
            if not callable(getattr(npu, "synchronize", None)):
                failures.append("torch.npu.synchronize is unavailable")
    if not npu_available or device_count < 1:
        failures.append("no usable Ascend NPU is visible")
    report = {
        "schema": "bcro-120-npu-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": _version(modules.get("torch")),
        "torch_npu": _version(modules.get("torch_npu")),
        "sgl_kernel_npu": _version(modules.get("sgl_kernel_npu")),
        "npu_available": npu_available,
        "npu_device_count": device_count,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "cann_package_path": os.environ.get("ASCEND_OPP_PATH"),
        "failures": failures,
    }
    if failures:
        raise ConfigError("NPU environment preflight failed: " + "; ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frozen = load_and_validate(
        root=args.root,
        config_path=args.config,
        model_path=args.model,
        embedding_model_path=args.embedding_model,
        require_external_models=not args.static_only,
        require_bfcl_data=True,
    )
    report: dict[str, Any] = frozen.provenance()
    report["static_npu_capabilities"] = validate_static_npu_sources(frozen.root)
    report["runtime_environment"] = None
    report["validation_level"] = "static-only"
    if not args.static_only:
        report["runtime_environment"] = inspect_npu_environment()
        report["validation_level"] = "npu-environment"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2)
