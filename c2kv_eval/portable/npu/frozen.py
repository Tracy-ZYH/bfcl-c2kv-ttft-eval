"""Validation for the immutable inputs of the formal BCRO-120 run."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """The requested run does not match the frozen BCRO-120 contract."""


EXPECTED_GUARDS = {
    "exact_identifier": True,
    "readonly_repeat_same_turn": True,
    "readonly_repeat_previous_turn": False,
    "readonly_repeat_no_tools_max_tokens": None,
    "post_tweet_literal": True,
    "cancel_booking_full_diversion": True,
    "query_unique_raw_evidence": False,
    "shared_regeneration_budget": 1,
}

EXPECTED_GENERATION = {
    "dtype": "bfloat16",
    "random_seed": 0,
    "max_running_requests": 1,
    "disable_radix_cache": True,
    "disable_cuda_graph": True,
    "page_size": 1,
    "context_length": 262144,
    "max_total_tokens": 262144,
    "chunked_prefill_size": 4096,
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigError(f"missing {label}: {path}")


def _verify_sha(path: Path, expected: str, label: str) -> None:
    _require_file(path, label)
    actual = sha256_file(path)
    if actual != expected:
        raise ConfigError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={path}"
        )


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class FrozenBCRO120:
    root: Path
    config_path: Path
    config: dict[str, Any]
    task_ids: tuple[str, ...]
    model_path: Path
    embedding_model_path: Path
    detector_path: Path

    @property
    def artifact_path(self) -> Path:
        return _resolve(self.root, self.config["bcro_artifact"])

    @property
    def policy_path(self) -> Path:
        return _resolve(self.root, self.config["policy"])

    @property
    def capacity_path(self) -> Path:
        return _resolve(self.root, self.config["capacity"])

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "bcro-120-npu-provenance-v1",
            "repository_commit": _git_revision(self.root),
            "config": str(self.config_path),
            "config_sha256": sha256_file(self.config_path),
            "task_manifest": self.config["task_manifest"],
            "task_manifest_sha256": self.config["task_manifest_sha256"],
            "bcro_artifact": self.config["bcro_artifact"],
            "bcro_artifact_sha256": self.config["bcro_artifact_sha256"],
            "detector_artifact": self.config["detector"]["artifact"],
            "detector_artifact_sha256": self.config["detector"]["artifact_sha256"],
            "detector_threshold": self.config["detector"]["threshold"],
            "model_path": str(self.model_path),
            "model_config_sha256": self.config["model_config_sha256"],
            "embedding_model_path": str(self.embedding_model_path),
            "guards": self.config["guards"],
            "compression": self.config["compression"],
            "generation": self.config["generation"],
        }


def validate_static_npu_sources(root: Path) -> dict[str, bool]:
    """Fail closed if vendored SGLang lacks an exact NPU C2KV path."""
    checks = {
        "npu_backend": (
            root / "third_party/sglang/srt/hardware_backend/npu/utils.py",
            ("import sgl_kernel_npu", "import torch_npu", "args.attention_backend = \"ascend\""),
        ),
        "c2kv_npu_injection": (
            root / "third_party/sglang/srt/mem_cache/c2kv_injection.py",
            ("inject_c2kv_gist", "inject_c2kv_stored_kv", "torch.npu.synchronize"),
        ),
        "npu_paged_pool": (
            root / "third_party/sglang/srt/hardware_backend/npu/memory_pool_npu.py",
            ("page_size", "set_kv_buffer"),
        ),
        "npu_position_bridge": (
            root / "third_party/sglang/srt/managers/scheduler.py",
            ("loc_offset", "inject_c2kv_gist", "inject_c2kv_stored_kv"),
        ),
        "native_c2kv_endpoint": (
            root / "third_party/sglang/srt/entrypoints/http_server.py",
            ("/v1/c2kv/native_generate", "native_ttft_ms", "cache_hit_lookup_load_ms"),
        ),
        "hidden_and_logprobs": (
            root / "third_party/sglang/srt/entrypoints/openai/serving_chat.py",
            ("return_hidden_states", "return_logprob", "hidden_states"),
        ),
    }
    result: dict[str, bool] = {}
    for label, (path, needles) in checks.items():
        _require_file(path, label)
        text = path.read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in text]
        if missing:
            raise ConfigError(f"{label} lacks required NPU contract symbols: {missing}")
        result[label] = True
    return result


def load_and_validate(
    *,
    root: Path | None = None,
    config_path: Path | None = None,
    model_path: Path | None = None,
    embedding_model_path: Path | None = None,
    require_external_models: bool = True,
    require_bfcl_data: bool = True,
) -> FrozenBCRO120:
    root = (root or repository_root()).resolve()
    config_path = (config_path or root / "artifacts/bcro/bcro120_config.json").resolve()
    _require_file(config_path, "BCRO-120 config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "bcro-120-reproduction-config-v1":
        raise ConfigError("not a formal BCRO-120 reproduction config")
    if config.get("guards") != EXPECTED_GUARDS:
        raise ConfigError(f"BCRO-120 guard contract changed: {config.get('guards')!r}")
    if config.get("generation") != EXPECTED_GENERATION:
        raise ConfigError("BCRO-120 generation contract changed")
    compression = config.get("compression") or {}
    if compression != {
        "ratio": 8,
        "history_variant": "H0",
        "history_budget_tokens": 256,
        "selection_protocol": "evidence_sets_v1",
    }:
        raise ConfigError(f"BCRO-120 compression contract changed: {compression!r}")
    detector = config.get("detector") or {}
    if detector.get("kind") != "t02_risk" or detector.get("threshold") != 0.5:
        raise ConfigError("BCRO-120 must use frozen T02 at threshold 0.5")

    hashed = (
        ("task_manifest", "task_manifest_sha256"),
        ("bcro_artifact", "bcro_artifact_sha256"),
        ("split", "split_sha256"),
        ("policy", "policy_sha256"),
        ("capacity", "capacity_sha256"),
    )
    for path_key, sha_key in hashed:
        _verify_sha(_resolve(root, config[path_key]), config[sha_key], path_key)
    detector_path = _resolve(root, detector["artifact"])
    _verify_sha(detector_path, detector["artifact_sha256"], "T02 detector")

    task_path = _resolve(root, config["task_manifest"])
    task_ids = tuple(line.strip() for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(task_ids) != config.get("task_count") or len(task_ids) != 120:
        raise ConfigError(f"expected exactly 120 frozen task IDs, found {len(task_ids)}")
    if len(task_ids) != len(set(task_ids)):
        raise ConfigError("BCRO-120 task manifest contains duplicate IDs")

    split_path = _resolve(root, config["split"])
    with split_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    test117 = {row["task_id"] for row in rows if row["split"] == "test"}
    extras = set(config.get("additional_task_ids") or ())
    if len(test117) != 117 or set(task_ids) != test117 | extras or test117 & extras:
        raise ConfigError("BCRO-120 task set is not the frozen Test117 plus three extras")

    if require_bfcl_data:
        data_path = root / "bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
        _require_file(data_path, "official BFCL Long data")
        data_ids = {
            json.loads(line)["id"]
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not set(task_ids) <= data_ids:
            raise ConfigError("official BFCL data does not cover all frozen task IDs")

    model_path = (model_path or Path(config["model"])).expanduser().resolve()
    embedding_model_path = (
        embedding_model_path or Path(config["embedding_model"])
    ).expanduser().resolve()
    if require_external_models:
        _verify_sha(model_path / "config.json", config["model_config_sha256"], "C2KV model config")
        _require_file(embedding_model_path / "config.json", "embedding model config")

    validate_static_npu_sources(root)
    return FrozenBCRO120(
        root=root,
        config_path=config_path,
        config=config,
        task_ids=task_ids,
        model_path=model_path,
        embedding_model_path=embedding_model_path,
        detector_path=detector_path,
    )
