"""Frozen binary Full/Racer scheduler for BFCL Long.

The GP predicts the log online-TTFT ratio of the existing 8x C2KV cache-hit
path relative to Full.  Semantic features are deliberately not fitted from
the historical longest-decision file (it does not contain them); they enter
only through the frozen, query-only risk guard at inference time.
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


BCRS_ARMS = frozenset({
    "fixed20k_racer",
    "bcrs_v1_racer",
    "bcrs_racer",
    "bcrs_v2_racer",
})
COST_FEATURE_NAMES = (
    "log_l_total", "log_l_tool", "log_l_hist", "rho_tool",
)


def is_bcrs_arm(name: str) -> bool:
    return name in BCRS_ARMS


def cost_features(observed: Mapping[str, float]) -> np.ndarray:
    return np.asarray([
        math.log1p(max(0.0, float(observed["L_total"]))),
        math.log1p(max(0.0, float(observed["L_tool"]))),
        math.log1p(max(0.0, float(observed["L_hist"]))),
        min(1.0, max(0.0, float(observed["rho_tool"]))),
    ], dtype=np.float64)


def semantic_risk(observed: Mapping[str, float], weights: Mapping[str, float]) -> float:
    values = (
        float(observed.get("S_tool", 0.0)),
        float(observed.get("S_hist", 0.0)),
        float(observed.get("D_cross", 0.0)),
    )
    names = ("s_tool", "s_hist", "d_cross")
    return min(1.0, max(0.0, sum(float(weights[name]) * value
                                 for name, value in zip(names, values))))


def gain_probability(predicted_log_ratio: float, uncertainty: float) -> float:
    """Return P(TTFT_racer < TTFT_full) under the frozen GP.

    The fitted target is log(TTFT_racer / TTFT_full), so a gain corresponds
    to a value below zero. Therefore P(gain) is Phi(-mu / sigma).
    """
    if uncertainty <= 0.0:
        if predicted_log_ratio < 0.0:
            return 1.0
        if predicted_log_ratio > 0.0:
            return 0.0
        return 0.5
    z_score = -float(predicted_log_ratio) / float(uncertainty)
    return 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))


def prior_guided_decision(
    observed: Mapping[str, float],
    predicted_log_ratio: float,
    uncertainty: float,
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen Prior-guided BCRS-v2 routing rule."""
    risk = semantic_risk(observed, rule["semantic_risk_weights"])
    probability = gain_probability(predicted_log_ratio, uncertainty)
    fixed_prior = float(observed["L_total"]) >= float(
        rule.get("history_length_prior_tokens", 20000)
    )
    low = float(rule.get("v2_full_below_tokens", 16000))
    high = float(rule.get("v2_racer_at_or_above_tokens", 24000))
    high_confidence = float(rule.get("v2_high_gain_probability", 0.95))
    low_confidence = float(rule.get("v2_low_gain_probability", 0.05))

    if float(observed["L_hist"]) <= 0.0:
        compress, reason = False, "no_compressible_history"
    elif risk > float(rule["maximum_semantic_risk"]):
        compress, reason = False, "semantic_risk_guard"
    elif float(observed["L_total"]) < low:
        compress, reason = False, "below_16k_full_prior"
    elif float(observed["L_total"]) >= high:
        compress, reason = True, "at_or_above_24k_racer_prior"
    elif probability > high_confidence:
        compress, reason = True, "boundary_gp_high_confidence_gain"
    elif probability < low_confidence:
        compress, reason = False, "boundary_gp_high_confidence_loss"
    else:
        compress = fixed_prior
        reason = "boundary_fixed20k_fallback"
    return {
        "compress": bool(compress),
        "reason": reason,
        "gain_probability": probability,
        "fixed20k_prior": bool(fixed_prior),
        "semantic_risk": risk,
    }


class BCRSScheduler:
    """Inference-only BCRS rule; no online fitting or adaptation."""

    def __init__(self, artifact: Mapping[str, Any], mode: str):
        if artifact.get("schema") != "bcrs-artifact-v1":
            raise ValueError("unsupported BCRS artifact schema")
        if mode not in BCRS_ARMS:
            raise ValueError(f"unsupported BCRS mode: {mode}")
        self.artifact = dict(artifact)
        self.mode = mode
        self.gp = FrozenGaussianProcess(artifact["cost_gp"])
        self.rule = dict(artifact["routing_rule"])
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_path(cls, path: str | Path, mode: str) -> "BCRSScheduler":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), mode)

    def route(self, payload: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
        started = time.perf_counter()
        observed = extract_features(payload, tokenizer)
        feature_ms = (time.perf_counter() - started) * 1000.0
        predict_started = time.perf_counter()
        predicted_log_ratio, uncertainty = self.gp.predict(cost_features(observed))
        predicted_ratio = math.exp(predicted_log_ratio)
        risk = semantic_risk(observed, self.rule["semantic_risk_weights"])
        min_length = int(self.rule["minimum_context_tokens"])

        if self.mode == "fixed20k_racer":
            compress = observed["L_total"] >= 20000
            objective = predicted_log_ratio
            reason = "fixed_20k_prior" if compress else "below_fixed_20k"
            probability = gain_probability(predicted_log_ratio, uncertainty)
            fixed_prior = compress
        elif self.mode == "bcrs_v1_racer":
            objective = (
                predicted_log_ratio
                + float(self.rule["uncertainty_beta"]) * uncertainty
                + float(self.rule["semantic_risk_weight"]) * risk
            )
            compress = (
                observed["L_total"] >= min_length
                and objective <= float(self.rule["compression_objective_threshold"])
                and uncertainty <= float(self.rule["maximum_uncertainty"])
                and risk <= float(self.rule["maximum_semantic_risk"])
            )
            if observed["L_total"] < min_length:
                reason = "below_length_prior"
            elif uncertainty > float(self.rule["maximum_uncertainty"]):
                reason = "uncertainty_guard"
            elif risk > float(self.rule["maximum_semantic_risk"]):
                reason = "semantic_risk_guard"
            elif objective > float(self.rule["compression_objective_threshold"]):
                reason = "insufficient_risk_adjusted_benefit"
            else:
                reason = "risk_adjusted_cost_benefit"
            probability = gain_probability(predicted_log_ratio, uncertainty)
            fixed_prior = observed["L_total"] >= 20000
        else:
            decision = prior_guided_decision(
                observed, predicted_log_ratio, uncertainty, self.rule
            )
            compress = decision["compress"]
            reason = decision["reason"]
            probability = decision["gain_probability"]
            fixed_prior = decision["fixed20k_prior"]
            objective = predicted_log_ratio
            risk = decision["semantic_risk"]

        predict_ms = (time.perf_counter() - predict_started) * 1000.0
        is_v2 = self.mode in {"bcrs_racer", "bcrs_v2_racer"}
        return {
            "schema": "bcrs-decision-v2" if is_v2 else "bcrs-decision-v1",
            "mode": self.mode,
            "selected_arm": "c2kv_racer" if compress else "full",
            "compress": bool(compress),
            "reason": reason,
            "features": observed,
            "predicted_log_ratio": predicted_log_ratio,
            "predicted_ratio": predicted_ratio,
            "uncertainty": uncertainty,
            "gain_probability": probability,
            "fixed20k_prior": bool(fixed_prior),
            "semantic_risk": risk,
            "objective": objective,
            "feature_extraction_ms": feature_ms,
            "gp_prediction_ms": predict_ms,
            "controller_preparation_ms": feature_ms + predict_ms,
            "artifact_sha256": self.artifact_sha256,
            "history_compression_ratio": 8 if compress else 1,
            "history_budget_tokens": 256 if compress else None,
            "tool_kv_mode": "full_native_protected",
            "tool_gist_compression_supported": False,
        }
