"""Lightweight GP-assisted Tool/History KV scheduling.

The runtime is intentionally inference-only: artifacts are fitted offline with
scikit-learn, while the SGLang-side proxy needs only NumPy. Tool compression in
this module means persistent raw-KV physical eviction, not C2KV gist extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "log_l_total", "log_l_tool", "log_l_hist", "rho_tool",
    "s_tool", "s_hist", "d_cross",
    "tool_retention", "history_retention", "history_budget_fraction",
)
PROFILE_ARM_PREFIX = "gp_profile_"
DYNAMIC_ARMS = {
    "fixed20k_joint", "gp_length_only", "gp_sjcs",
    *(f"{PROFILE_ARM_PREFIX}a{index}" for index in range(5)),
}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|\d{2,}")
_RESOURCE = re.compile(r"(?:[/._:-][A-Za-z0-9_-]+){1,}|[A-Za-z]+\d+[A-Za-z0-9_-]*")


@dataclass(frozen=True)
class Candidate:
    name: str
    tool_ratio: float
    history_ratio: float
    compress_tools: bool
    compress_history: bool

    def history_target(self, history_tokens: int, budget: int) -> int | None:
        if not self.compress_history or self.name == "A0":
            return None
        nominal = max(1, int(math.ceil(history_tokens * self.history_ratio)))
        return min(int(budget), nominal)

    def history_spec(self, history_tokens: int, budget: int) -> Dict[str, Any]:
        target = self.history_target(history_tokens, budget)
        return {
            "method": "snapkv_persistent",
            "retention_ratio": self.history_ratio,
            "target_tokens": target,
            "tool_retention_ratio": self.tool_ratio,
            "backend": "physical_eviction",
            "persistent_session": True,
            "compress_tools": self.compress_tools,
            "compress_completed_history": self.compress_history,
        }


CANDIDATES = (
    Candidate("A0", 1.0, 1.0, True, True),
    Candidate("A1", 1.0, 0.125, False, True),
    Candidate("A2", 0.25, 0.25, True, True),
    Candidate("A3", 0.125, 0.50, True, True),
    Candidate("A4", 0.50, 0.125, True, True),
)


def is_dynamic_arm(name: str) -> bool:
    return name in DYNAMIC_ARMS


def _words(value: Any) -> set[str]:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {token.lower() for token in _WORD.findall(value)}


def _overlap(query: set[str], candidate: set[str]) -> float:
    if not query or not candidate:
        return 0.0
    return len(query & candidate) / math.sqrt(len(query) * len(candidate))


def _current_start(messages: Sequence[Mapping[str, Any]]) -> int:
    users = [i for i, message in enumerate(messages) if message.get("role") == "user"]
    return users[-1] if users else len(messages)


def _render_len(tokenizer: Any, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None) -> int:
    kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools
    encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ndim") and int(encoded.ndim) > 1:
        encoded = encoded[0]
    elif (isinstance(encoded, (list, tuple)) and encoded
          and isinstance(encoded[0], (list, tuple))):
        encoded = encoded[0]
    return len(encoded)


def extract_features(payload: Mapping[str, Any], tokenizer: Any) -> Dict[str, float]:
    """Extract query-only scheduling features from the visible request."""
    messages = [dict(message) for message in payload.get("messages") or []]
    tools = [dict(tool) for tool in payload.get("tools") or []]
    cutoff = _current_start(messages)
    history = [
        message for message in messages[:cutoff]
        if message.get("role") != "system"
    ]
    current = messages[cutoff:]
    query_text = "\n".join(
        str(message.get("content") or "") for message in current
    )
    query_terms = _words(query_text)

    l_total = _render_len(tokenizer, messages, tools)
    without_tools = _render_len(tokenizer, messages, None)
    l_tool = max(0, l_total - without_tools)
    history_text = "\n".join(
        json.dumps(message, ensure_ascii=False, sort_keys=True)
        for message in history
    )
    l_hist = len(tokenizer.encode(history_text, add_special_tokens=False)) if history_text else 0

    tool_scores = []
    tool_terms: set[str] = set()
    param_terms: set[str] = set()
    for tool in tools:
        function = tool.get("function") or {}
        terms = _words(function)
        tool_terms |= terms
        parameters = function.get("parameters") or {}
        param_terms |= _words(parameters)
        tool_scores.append(_overlap(query_terms, terms))

    history_units = [
        _words(message) for message in history
        if message.get("content") is not None or message.get("tool_calls")
    ]
    history_scores = [_overlap(query_terms, terms) for terms in history_units]
    history_terms = set().union(*history_units) if history_units else set()
    resource_terms = {term.lower() for term in _RESOURCE.findall(query_text)}
    dependency_terms = param_terms | resource_terms
    dependency_hits = dependency_terms & history_terms
    used_tool_terms = tool_terms & history_terms
    denominator = max(1, len(dependency_terms) + len(tool_terms))
    d_cross = min(1.0, (2 * len(dependency_hits) + len(used_tool_terms)) / denominator)

    scope = l_tool + l_hist
    return {
        "L_total": float(l_total),
        "L_tool": float(l_tool),
        "L_hist": float(l_hist),
        "rho_tool": float(l_tool / scope) if scope else 0.0,
        "S_tool": max(tool_scores, default=0.0),
        "S_hist": max(history_scores, default=0.0),
        "D_cross": float(d_cross),
    }


def model_features(observed: Mapping[str, float], candidate: Candidate,
                   history_budget: int, length_only: bool = False) -> np.ndarray:
    hist = max(0.0, float(observed["L_hist"]))
    target = candidate.history_target(int(hist), history_budget)
    hist_fraction = (
        min(candidate.history_ratio, target / hist)
        if target is not None and hist > 0 else 1.0
    )
    semantic = (0.0, 0.0, 0.0) if length_only else (
        float(observed["S_tool"]),
        float(observed["S_hist"]),
        float(observed["D_cross"]),
    )
    return np.asarray([
        math.log1p(float(observed["L_total"])),
        math.log1p(float(observed["L_tool"])),
        math.log1p(hist),
        float(observed["rho_tool"]),
        *semantic,
        candidate.tool_ratio if candidate.compress_tools else 1.0,
        hist_fraction,
        (target / hist) if target is not None and hist > 0 else 1.0,
    ], dtype=np.float64)


def analytic_log_ratio(observed: Mapping[str, float], candidate: Candidate,
                       history_budget: int) -> float:
    total = max(1.0, float(observed["L_total"]))
    tool = max(0.0, float(observed["L_tool"]))
    hist = max(0.0, float(observed["L_hist"]))
    protected = max(0.0, total - tool - hist)
    target = candidate.history_target(int(hist), history_budget)
    active_tool = tool * (candidate.tool_ratio if candidate.compress_tools else 1.0)
    active_hist = hist if target is None else min(hist, float(target))
    active_ratio = min(1.0, max(1e-4, (protected + active_tool + active_hist) / total))
    # Prefill is not perfectly token-linear; this floor is calibrated by the
    # GP residual rather than presented as a measured latency.
    return math.log(0.08 + 0.92 * active_ratio)


class FrozenGaussianProcess:
    def __init__(self, payload: Mapping[str, Any]):
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.x_train = np.asarray(payload["x_train_scaled"], dtype=np.float64)
        self.alpha = np.asarray(payload["alpha"], dtype=np.float64)
        self.cholesky = np.asarray(payload["cholesky"], dtype=np.float64)
        self.constant = float(payload["constant"])
        self.length_scale = np.asarray(payload["length_scale"], dtype=np.float64)
        self.noise = float(payload["noise"])
        self.y_mean = float(payload["y_mean"])
        self.y_std = float(payload["y_std"])

    def predict(self, vector: np.ndarray) -> tuple[float, float]:
        x = (np.asarray(vector, dtype=np.float64) - self.mean) / self.scale
        delta = (self.x_train - x) / self.length_scale
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        root3 = math.sqrt(3.0)
        kernel = self.constant * (1.0 + root3 * distance) * np.exp(-root3 * distance)
        mean_norm = float(kernel @ self.alpha)
        solved = np.linalg.solve(self.cholesky, kernel)
        variance_norm = max(0.0, self.constant + self.noise - float(solved @ solved))
        return (
            self.y_mean + self.y_std * mean_norm,
            abs(self.y_std) * math.sqrt(variance_norm),
        )


class GPSJCSScheduler:
    def __init__(self, artifact: Mapping[str, Any], mode: str,
                 history_budget: int = 256, beta: float = 0.25):
        if artifact.get("schema") != "gp-sjcs-artifact-v1":
            raise ValueError("unsupported GP-SJCS artifact schema")
        self.artifact = dict(artifact)
        self.mode = mode
        self.history_budget = int(history_budget)
        self.beta = float(beta)
        model_name = "length_only" if mode == "gp_length_only" else "semantic"
        self.gp = FrozenGaussianProcess(artifact["models"][model_name])

    @classmethod
    def from_path(cls, path: str | Path, mode: str,
                  history_budget: int = 256, beta: float = 0.25) -> "GPSJCSScheduler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload, mode, history_budget, beta)

    def _quality_allowed(self, candidate: Candidate,
                         features: Mapping[str, float]) -> tuple[bool, str]:
        if candidate.name == "A0":
            return True, "full_fallback"
        if features["S_tool"] >= 0.25 and candidate.tool_ratio < 0.50:
            return False, "query_relevant_tool_requires_native_kv"
        target = candidate.history_target(int(features["L_hist"]), self.history_budget)
        hist_fraction = (
            target / features["L_hist"] if target is not None and features["L_hist"] else 1.0)
        if features["S_hist"] >= 0.25 and hist_fraction < 0.125:
            return False, "query_relevant_history_overcompressed"
        if (features["D_cross"] >= 0.30 and candidate.tool_ratio < 0.50
                and hist_fraction < 0.50):
            return False, "strong_tool_history_dependency"
        return True, "allowed"

    def route(self, payload: Mapping[str, Any], tokenizer: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        observed = extract_features(payload, tokenizer)
        feature_ms = (time.perf_counter() - started) * 1000.0

        if self.mode.startswith(PROFILE_ARM_PREFIX):
            suffix = self.mode.removeprefix(PROFILE_ARM_PREFIX).upper()
            matches = [candidate for candidate in CANDIDATES if candidate.name == suffix]
            if len(matches) != 1:
                raise ValueError(f"unknown fixed GP-SJCS profile arm: {self.mode}")
            chosen = matches[0]
            predictions = {}
            predict_ms = 0.0
            reason = "fixed_candidate_profile"
        elif self.mode == "fixed20k_joint":
            chosen = CANDIDATES[1] if observed["L_total"] >= 20000 else CANDIDATES[0]
            predictions: Dict[str, Any] = {}
            predict_ms = 0.0
            reason = "fixed_20k_threshold"
        else:
            predict_started = time.perf_counter()
            predictions = {}
            length_only = self.mode == "gp_length_only"
            for candidate in CANDIDATES:
                allowed, guard = (
                    (True, "length_only_ablation")
                    if length_only else self._quality_allowed(candidate, observed)
                )
                if candidate.name == "A0":
                    mean, std = 0.0, 0.0
                else:
                    vector = model_features(
                        observed, candidate, self.history_budget, length_only)
                    residual, std = self.gp.predict(vector)
                    mean = analytic_log_ratio(
                        observed, candidate, self.history_budget) + residual
                predictions[candidate.name] = {
                    "predicted_log_ratio": mean,
                    "predicted_ratio": math.exp(mean),
                    "uncertainty": std,
                    "objective": mean + self.beta * std,
                    "allowed": allowed,
                    "guard": guard,
                }
            eligible = [
                candidate for candidate in CANDIDATES
                if predictions[candidate.name]["allowed"]
            ]
            chosen = min(
                eligible,
                key=lambda candidate: predictions[candidate.name]["objective"],
            )
            chosen_prediction = predictions[chosen.name]
            if (chosen.name != "A0"
                    and (chosen_prediction["predicted_ratio"] >= 0.95
                         or chosen_prediction["uncertainty"] > 0.40)):
                chosen = CANDIDATES[0]
                reason = "conservative_full_fallback"
            else:
                reason = "gp_minimum_risk_adjusted_latency"
            predict_ms = (time.perf_counter() - predict_started) * 1000.0

        target = chosen.history_target(int(observed["L_hist"]), self.history_budget)
        decision = {
            "schema": "gp-sjcs-decision-v1",
            "mode": self.mode,
            "selected_config": chosen.name,
            "selection_reason": reason,
            "features": observed,
            "predictions": predictions,
            "feature_extraction_ms": feature_ms,
            "gp_prediction_ms": predict_ms,
            "controller_preparation_ms": feature_ms + predict_ms,
            "requested_history_budget": self.history_budget,
            "requested_history_target_tokens": target,
            "tool_target_ratio": chosen.tool_ratio,
            "history_target_ratio": chosen.history_ratio,
            "materialization": "persistent_physical_raw_kv_eviction",
            "tool_gist_supported": False,
            "history_gist_supported_in_dynamic_joint": False,
            "offline_extraction_applicable": False,
            "history_budget_semantics": "hard_server_token_target_with_alignment_telemetry",
            "artifact_sha256": hashlib.sha256(
                json.dumps(self.artifact, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "history_kv": chosen.history_spec(
                int(observed["L_hist"]), self.history_budget),
        }
        return decision
