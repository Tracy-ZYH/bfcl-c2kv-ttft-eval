"""Frozen Bayesian Cost-Risk Optimization router for Full versus Racer.

BCRO has no context-length threshold.  It predicts absolute Full and Racer
online costs, applies a frozen GP residual correction, then selects Racer only
when the uncertainty- and risk-adjusted utility is positive.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .bcrs import cost_features, semantic_risk
from .gp_sjcs import FrozenGaussianProcess, extract_features


def full_cost_features(observed: Mapping[str, float]) -> np.ndarray:
    log_total = math.log1p(max(0.0, float(observed["L_total"])))
    return np.asarray([log_total, log_total * log_total], dtype=np.float64)


def racer_cost_features(observed: Mapping[str, float]) -> np.ndarray:
    log_total = math.log1p(max(0.0, float(observed["L_total"])))
    rho_tool = min(1.0, max(0.0, float(observed["rho_tool"])))
    rho_hist = 1.0 - rho_tool
    return np.asarray([
        log_total,
        log_total * log_total,
        rho_tool,
        rho_hist,
        log_total * rho_hist,
    ], dtype=np.float64)


class FrozenLogCostModel:
    """Small standardized Ridge model with a positive log-cost output."""

    def __init__(self, payload: Mapping[str, Any]):
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        self.intercept = float(payload["intercept"])
        if not (self.mean.shape == self.scale.shape == self.coefficients.shape):
            raise ValueError("invalid frozen log-cost model dimensions")
        if np.any(self.scale <= 0.0):
            raise ValueError("invalid frozen log-cost model scale")

    def predict(self, vector: np.ndarray) -> float:
        standardized = (np.asarray(vector, dtype=np.float64) - self.mean) / self.scale
        log_cost = self.intercept + float(standardized @ self.coefficients)
        cost = math.exp(log_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("non-finite BCRO cost prediction")
        return cost


def bcro_decision(
    observed: Mapping[str, float],
    predicted_full_ms: float,
    predicted_racer_analytic_ms: float,
    residual_mean: float,
    residual_std: float,
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the threshold-free BCRO decision for already predicted costs."""
    risk = semantic_risk(observed, rule["semantic_risk_weights"])
    values = (
        predicted_full_ms,
        predicted_racer_analytic_ms,
        residual_mean,
        residual_std,
        risk,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return {
            "compress": False,
            "reason": "predictor_invalid_full_fallback",
            "utility_ms": None,
            "semantic_risk": risk,
        }
    if predicted_full_ms <= 0.0 or predicted_racer_analytic_ms <= 0.0 or residual_std < 0.0:
        return {
            "compress": False,
            "reason": "predictor_invalid_full_fallback",
            "utility_ms": None,
            "semantic_risk": risk,
        }

    corrected_racer_ms = predicted_racer_analytic_ms * math.exp(residual_mean)
    uncertainty_penalty_ms = (
        float(rule["uncertainty_beta"])
        * corrected_racer_ms
        * math.expm1(min(50.0, residual_std))
    )
    risk_penalty_ms = (
        float(rule["semantic_risk_lambda"]) * risk * predicted_full_ms
    )
    utility_ms = (
        predicted_full_ms
        - corrected_racer_ms
        - uncertainty_penalty_ms
        - risk_penalty_ms
    )

    if float(observed.get("L_hist", 0.0)) <= 0.0:
        compress, reason = False, "no_compressible_history"
    elif risk > float(rule["maximum_semantic_risk"]):
        compress, reason = False, "semantic_risk_guard"
    elif utility_ms > 0.0:
        compress, reason = True, "positive_cost_risk_utility"
    else:
        compress, reason = False, "nonpositive_cost_risk_utility"
    return {
        "compress": compress,
        "reason": reason,
        "utility_ms": utility_ms,
        "predicted_full_ms": predicted_full_ms,
        "predicted_racer_analytic_ms": predicted_racer_analytic_ms,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "predicted_racer_ms": corrected_racer_ms,
        "uncertainty_penalty_ms": uncertainty_penalty_ms,
        "risk_penalty_ms": risk_penalty_ms,
        "semantic_risk": risk,
    }


class BCROScheduler:
    """Online-callable, frozen Full/Racer BCRO scheduler."""

    def __init__(self, artifact: Mapping[str, Any]):
        if artifact.get("schema") != "bcro-artifact-v1" or not artifact.get("frozen"):
            raise ValueError("a frozen bcro-artifact-v1 is required")
        self.artifact = dict(artifact)
        self.full_model = FrozenLogCostModel(artifact["full_cost_model"])
        self.racer_model = FrozenLogCostModel(artifact["racer_cost_model"])
        self.residual_gp = FrozenGaussianProcess(artifact["residual_gp"])
        self.rule = dict(artifact["routing_rule"])
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_path(cls, path: str | Path) -> "BCROScheduler":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def route_observed(self, observed: Mapping[str, float]) -> dict[str, Any]:
        try:
            full_ms = self.full_model.predict(full_cost_features(observed))
            racer_ms = self.racer_model.predict(racer_cost_features(observed))
            residual_mean, residual_std_raw = self.residual_gp.predict(
                cost_features(observed)
            )
            residual_std = (
                residual_std_raw
                * float(self.rule.get("uncertainty_scale", 1.0))
            )
            decision = bcro_decision(
                observed, full_ms, racer_ms, residual_mean, residual_std, self.rule
            )
            decision["residual_std_raw"] = residual_std_raw
        except (ArithmeticError, KeyError, TypeError, ValueError) as error:
            decision = {
                "compress": False,
                "reason": "predictor_invalid_full_fallback",
                "utility_ms": None,
                "predictor_error": f"{type(error).__name__}: {error}",
                "semantic_risk": None,
            }
        decision.update({
            "schema": "bcro-decision-v1",
            "selected_arm": "c2kv_racer" if decision["compress"] else "full",
            "features": dict(observed),
            "artifact_sha256": self.artifact_sha256,
            "history_compression_ratio": 8 if decision["compress"] else 1,
            "history_budget_tokens": 256 if decision["compress"] else None,
            "tool_kv_mode": "full_native_protected",
            "uses_fixed_length_threshold": False,
        })
        return decision

    def route(self, payload: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
        started = time.perf_counter()
        observed = extract_features(payload, tokenizer)
        feature_ms = (time.perf_counter() - started) * 1000.0
        prediction_started = time.perf_counter()
        decision = self.route_observed(observed)
        decision["feature_extraction_ms"] = feature_ms
        decision["prediction_ms"] = (
            time.perf_counter() - prediction_started
        ) * 1000.0
        decision["controller_preparation_ms"] = (
            feature_ms + decision["prediction_ms"]
        )
        return decision
