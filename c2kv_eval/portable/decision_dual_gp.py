"""Frozen five-fold, decision-level latency/risk GP router.

The online contract is deliberately small: only current visible token lengths
and the compressible-history fraction are model inputs. Task IDs select the
preassigned held-out fold; they are never model features or routing rules.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .gp_sjcs import FrozenGaussianProcess, extract_features


FEATURE_NAMES = (
    "log_total_tokens",
    "log_history_tokens",
    "log_tool_tokens",
    "compressible_history_ratio",
)


def feature_vector(observed: Mapping[str, float]) -> np.ndarray:
    total = max(0.0, float(observed["L_total"]))
    history = max(0.0, float(observed["L_hist"]))
    tools = max(0.0, float(observed["L_tool"]))
    return np.asarray([
        math.log1p(total),
        math.log1p(history),
        math.log1p(tools),
        min(1.0, history / max(1.0, total)),
    ], dtype=np.float64)


def task_id_from_payload(payload: Mapping[str, Any]) -> str:
    context = payload.get("c2kv_eval_context") or {}
    candidates = (context.get("task_id"), payload.get("session_id"))
    for value in candidates:
        if isinstance(value, str) and value.startswith("multi_turn_long_context_"):
            return value.split("/", 1)[0]
    raise ValueError("decision-level GP requires an official BFCL Long task ID")


class DecisionDualGPScheduler:
    """Select Full/Racer with a task-held-out latency GP and risk GP."""

    def __init__(self, artifact: Mapping[str, Any]):
        if artifact.get("schema") != "bfcl-decision-dual-gp-v1":
            raise ValueError("a bfcl-decision-dual-gp-v1 artifact is required")
        if artifact.get("frozen") is not True:
            raise ValueError("decision-level GP artifact must be frozen")
        self.artifact = dict(artifact)
        self.task_to_fold = {
            str(task): int(fold)
            for task, fold in artifact["task_to_fold"].items()
        }
        self.models = {}
        for fold, payload in artifact["fold_models"].items():
            self.models[int(fold)] = {
                "gain": FrozenGaussianProcess(payload["gain_gp"]),
                "risk": FrozenGaussianProcess(payload["risk_gp"]),
                "risk_threshold": float(payload["risk_threshold"]),
                "uncertainty_beta": float(payload["uncertainty_beta"]),
                "training_tasks": frozenset(payload["training_tasks"]),
            }
        if set(self.models) != set(range(5)):
            raise ValueError("artifact must contain exactly five held-out models")
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_path(cls, path: str | Path) -> "DecisionDualGPScheduler":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def route_observed(self, task_id: str, observed: Mapping[str, float]) -> dict[str, Any]:
        if task_id not in self.task_to_fold:
            raise ValueError(f"task is absent from frozen fold manifest: {task_id}")
        fold = self.task_to_fold[task_id]
        model = self.models[fold]
        if task_id in model["training_tasks"]:
            raise AssertionError(f"fold leakage detected for {task_id}")
        vector = feature_vector(observed)
        gain_mean, gain_std = model["gain"].predict(vector)
        risk_mean, risk_std = model["risk"].predict(vector)
        beta = model["uncertainty_beta"]
        gain_lower = gain_mean - beta * gain_std
        risk_upper = min(1.0, max(0.0, risk_mean + beta * risk_std))
        history = float(observed["L_hist"])
        compress = (
            history > 0.0
            and gain_lower > 0.0
            and risk_upper < model["risk_threshold"]
        )
        if history <= 0.0:
            reason = "no_compressible_history"
        elif gain_lower <= 0.0:
            reason = "nonpositive_conservative_gain"
        elif risk_upper >= model["risk_threshold"]:
            reason = "action_risk_guard"
        else:
            reason = "positive_gain_low_risk"
        return {
            "schema": "bfcl-decision-dual-gp-route-v1",
            "selected_arm": "c2kv_racer" if compress else "full",
            "compress": compress,
            "reason": reason,
            "held_out_fold": fold,
            "gain_mean_ms": gain_mean,
            "gain_std_ms": gain_std,
            "gain_lower_ms": gain_lower,
            "risk_mean": risk_mean,
            "risk_std": risk_std,
            "risk_upper": risk_upper,
            "risk_threshold": model["risk_threshold"],
            "uncertainty_beta": model["uncertainty_beta"],
            "features": {
                "L_total": float(observed["L_total"]),
                "L_hist": float(observed["L_hist"]),
                "L_tool": float(observed["L_tool"]),
                "compressible_history_ratio": float(vector[-1]),
            },
            "artifact_sha256": self.artifact_sha256,
            "history_compression_ratio": 8 if compress else 1,
            "history_budget_tokens": 256 if compress else None,
            "tool_kv_mode": "full_native_protected",
            "label_semantics": "proxy_action_divergence_not_official_accuracy",
        }

    def route(self, payload: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
        started = time.perf_counter()
        observed = extract_features(payload, tokenizer)
        feature_ms = (time.perf_counter() - started) * 1000.0
        predict_started = time.perf_counter()
        decision = self.route_observed(task_id_from_payload(payload), observed)
        prediction_ms = (time.perf_counter() - predict_started) * 1000.0
        decision["feature_extraction_ms"] = feature_ms
        decision["prediction_ms"] = prediction_ms
        decision["controller_preparation_ms"] = feature_ms + prediction_ms
        return decision
