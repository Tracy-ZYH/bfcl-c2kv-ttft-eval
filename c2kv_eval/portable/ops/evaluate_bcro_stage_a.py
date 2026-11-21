#!/usr/bin/env python3
"""Fit and evaluate threshold-free BCRO on the frozen Stage-A task split.

Only calibration-task labels are used to fit the analytic cost models and GP
residual.  Held-out timing labels are read after routing is frozen and are used
only for exploratory scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from c2kv_eval.portable.bcro import (
    BCROScheduler,
    full_cost_features,
    racer_cost_features,
)
from c2kv_eval.portable.bcrs import COST_FEATURE_NAMES, cost_features


def finite(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def observed(row: Mapping[str, Any]) -> dict[str, float]:
    total = finite(row, "full_context_tokens")
    history = finite(row, "full_history_tokens")
    # This source has no measured Tool-only column.  Keep the historical proxy
    # explicit instead of presenting it as true Tool token accounting.
    tool_proxy = max(0.0, total - history)
    scope = tool_proxy + history
    return {
        "L_total": total,
        "L_tool": tool_proxy,
        "L_hist": history,
        "rho_tool": tool_proxy / max(1.0, scope),
        "S_tool": 0.0,
        "S_hist": 0.0,
        "D_cross": 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metrics(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    full = [finite(row, "observed_full_online_ttft_ms") for row in rows]
    racer = [finite(row, "observed_c2kv_online_ttft_ms") for row in rows]
    routed = [racer[index] if use else full[index]
              for index, use in enumerate(selected)]
    return {
        "rows": len(rows),
        "compressed": int(sum(selected)),
        "compression_share": float(sum(selected) / len(rows)),
        "full_total_online_ttft_ms": float(sum(full)),
        "routed_total_online_ttft_ms": float(sum(routed)),
        "total_reduction": float(1.0 - sum(routed) / sum(full)),
        "routed_mean_ms": float(statistics.mean(routed)),
        "routed_p50_ms": float(percentile(routed, 0.50)),
        "routed_p95_ms": float(percentile(routed, 0.95)),
    }


def fit_log_cost_model(
    calibration: list[dict[str, Any]],
    feature_fn: Callable[[Mapping[str, float]], np.ndarray],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = np.vstack([feature_fn(observed(row)) for row in calibration])
    y = np.log(np.asarray([finite(row, label) for row in calibration]))
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=1.0, fit_intercept=True).fit(scaler.transform(x), y)
    prediction = model.predict(scaler.transform(x))
    payload = {
        "kind": "standardized_ridge_log_cost",
        "alpha": 1.0,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
    }
    report = {
        "label": label,
        "train_rows": len(calibration),
        "train_log_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "train_log_mae": float(np.mean(np.abs(prediction - y))),
    }
    return payload, report


def predict_log_cost(
    payload: Mapping[str, Any], vector: np.ndarray,
) -> float:
    mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    return float(payload["intercept"] + ((vector - mean) / scale) @ coefficients)


def fit_residual_gp(
    calibration: list[dict[str, Any]],
    full_model: Mapping[str, Any],
    racer_model: Mapping[str, Any],
    parent_gp: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = np.vstack([cost_features(observed(row)) for row in calibration])
    residuals = []
    for row in calibration:
        features = observed(row)
        actual_log_ratio = math.log(
            finite(row, "observed_c2kv_online_ttft_ms")
            / finite(row, "observed_full_online_ttft_ms")
        )
        analytic_log_ratio = (
            predict_log_cost(racer_model, racer_cost_features(features))
            - predict_log_cost(full_model, full_cost_features(features))
        )
        residuals.append(actual_log_ratio - analytic_log_ratio)
    y = np.asarray(residuals, dtype=np.float64)

    # Reuse the original BCRS scaler and fitted kernel hyperparameters.  The
    # residual GP is fitted only on the same 20 calibration tasks and performs
    # no optimization against held-out labels.
    scaler_mean = np.asarray(parent_gp["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(parent_gp["scaler_scale"], dtype=np.float64)
    scaled = (x - scaler_mean) / scaler_scale
    constant = float(parent_gp["constant"])
    length_scale = np.asarray(parent_gp["length_scale"], dtype=np.float64)
    noise = float(parent_gp["noise"])
    kernel = (
        ConstantKernel(constant, constant_value_bounds="fixed")
        * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=1.5)
        + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        alpha=1e-6,
        normalize_y=True,
        random_state=seed,
    ).fit(scaled, y)
    prediction, std = gp.predict(scaled, return_std=True)
    fitted = gp.kernel_
    payload = {
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "x_train_scaled": gp.X_train_.tolist(),
        "alpha": gp.alpha_.tolist(),
        "cholesky": gp.L_.tolist(),
        "constant": float(fitted.k1.k1.constant_value),
        "length_scale": np.asarray(fitted.k1.k2.length_scale).reshape(-1).tolist(),
        "noise": float(fitted.k2.noise_level),
        "y_mean": float(np.asarray(gp._y_train_mean).reshape(-1)[0]),
        "y_std": float(np.asarray(gp._y_train_std).reshape(-1)[0]),
    }
    report = {
        "target": "actual_log_ratio - analytic_log_cost_ratio",
        "train_rows": len(calibration),
        "kernel": str(fitted),
        "kernel_hyperparameters_reused_from_bcrs": True,
        "optimizer": None,
        "train_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "train_mae": float(np.mean(np.abs(prediction - y))),
        "predictive_std_mean": float(np.mean(std)),
    }
    return payload, report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def band(value: float, boundaries: list[tuple[float, str]]) -> str:
    for upper, label in boundaries:
        if value < upper:
            return label
    return boundaries[-1][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcrs-v2-dir", type=Path, required=True)
    parser.add_argument("--historical-paired", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_path = args.bcrs_v2_dir / "bcrs_v2_artifact.json"
    routing_path = args.bcrs_v2_dir / "routing_decisions_v2.csv"
    parent = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not parent.get("frozen") or "cost_gp" not in parent:
        raise ValueError("a frozen BCRS-v2 artifact with cost_gp is required")
    source_sha = hashlib.sha256(args.historical_paired.read_bytes()).hexdigest()
    if source_sha != parent["source_sha256"]:
        raise ValueError("historical paired CSV does not match BCRS-v2 source")

    routes = list(csv.DictReader(routing_path.open(encoding="utf-8")))
    historical = list(csv.DictReader(args.historical_paired.open(encoding="utf-8")))
    historical_by_task = {row["task_id"]: row for row in historical}
    if len(routes) != 197 or len(historical_by_task) != 197:
        raise ValueError("BCRO Stage A expects 197 unique task rows")
    if set(historical_by_task) != {row["task_id"] for row in routes}:
        raise ValueError("routing and historical task sets differ")

    rows: list[dict[str, Any]] = []
    for route in routes:
        source = historical_by_task[route["task_id"]]
        merged = dict(route)
        merged["observed_full_online_ttft_ms"] = source["full_online_ttft_ms"]
        merged["observed_c2kv_online_ttft_ms"] = source["c2kv_online_ttft_ms"]
        rows.append(merged)

    calibration_ids = set(parent["calibration_task_ids"])
    calibration = [row for row in rows if row["task_id"] in calibration_ids]
    heldout_indices = [index for index, row in enumerate(rows)
                       if row["task_id"] not in calibration_ids]
    heldout = [rows[index] for index in heldout_indices]
    if len(calibration) != 20 or len(heldout) != 177:
        raise ValueError("expected the frozen 20/177 task-disjoint split")

    full_model, full_fit = fit_log_cost_model(
        calibration, full_cost_features, "observed_full_online_ttft_ms"
    )
    racer_model, racer_fit = fit_log_cost_model(
        calibration, racer_cost_features, "observed_c2kv_online_ttft_ms"
    )
    residual_gp, residual_fit = fit_residual_gp(
        calibration, full_model, racer_model, parent["cost_gp"], args.seed
    )
    rule = {
        "uncertainty_beta": 1.0,
        "semantic_risk_lambda": 0.35,
        "maximum_semantic_risk": 0.65,
        "semantic_risk_weights": {
            "s_tool": 0.20,
            "s_hist": 0.45,
            "d_cross": 0.35,
        },
        "fallback": "full_with_recorded_predictor_error",
    }
    artifact = {
        "schema": "bcro-artifact-v1",
        "variant": "threshold_free_bayesian_cost_risk_optimization",
        "frozen": True,
        "seed": args.seed,
        "source": str(args.historical_paired.resolve()),
        "source_sha256": source_sha,
        "source_semantics": parent["source_semantics"],
        "calibration_task_ids": sorted(calibration_ids),
        "heldout_task_count": len(heldout),
        "feature_names": list(COST_FEATURE_NAMES),
        "full_cost_model": full_model,
        "racer_cost_model": racer_model,
        "residual_gp": residual_gp,
        "routing_rule": rule,
        "online_feature_contract": parent["online_feature_contract"],
        "selected_backend": "existing_c2kv_racer_8x_fixed256",
        "tool_kv_mode": "full_native_protected",
        "uses_fixed_length_threshold": False,
        "dependency_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parent_bcrs_v2_artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
    }
    scheduler = BCROScheduler(artifact)

    started = time.perf_counter()
    decisions = [scheduler.route_observed(observed(row)) for row in rows]
    route_seconds = time.perf_counter() - started
    bcro = [bool(decision["compress"]) for decision in decisions]
    fixed = [parse_bool(row["fixed20k_selected"]) for row in rows]
    bcrs_v2 = [parse_bool(row["bcrs_v2_selected"]) for row in rows]

    method_specs = (("Fixed 20K", fixed), ("BCRS-v2", bcrs_v2), ("BCRO", bcro))
    comparison_rows: list[dict[str, Any]] = []
    method_metrics: dict[str, Any] = {}
    for name, selected in method_specs:
        all_metrics = metrics(rows, selected)
        heldout_selected = [selected[index] for index in heldout_indices]
        heldout_metrics = metrics(heldout, heldout_selected)
        agreement = sum(left == right for left, right in zip(fixed, selected)) / len(rows)
        comparison_rows.append({
            "method": name,
            "compressed_tasks": all_metrics["compressed"],
            "total_tasks": all_metrics["rows"],
            "compression_share": all_metrics["compression_share"],
            "total_ttft_reduction": all_metrics["total_reduction"],
            "heldout_compressed_tasks": heldout_metrics["compressed"],
            "heldout_tasks": heldout_metrics["rows"],
            "heldout_ttft_reduction": heldout_metrics["total_reduction"],
            "routed_p95_ms": all_metrics["routed_p95_ms"],
            "heldout_routed_p95_ms": heldout_metrics["routed_p95_ms"],
            "agreement_with_fixed20k": agreement,
        })
        method_metrics[name] = {"all197": all_metrics, "heldout177": heldout_metrics}

    length_boundaries = [
        (8000, "<8K"), (16000, "8K-16K"), (24000, "16K-24K"),
        (64000, "24K-64K"), (math.inf, ">=64K"),
    ]
    rho_boundaries = [
        (0.25, "rho_tool<0.25"), (0.50, "rho_tool=0.25-0.50"),
        (0.75, "rho_tool=0.50-0.75"), (math.inf, "rho_tool>=0.75"),
    ]
    distribution_rows: list[dict[str, Any]] = []
    for dimension, boundaries, value_fn in (
        ("length", length_boundaries, lambda row: observed(row)["L_total"]),
        ("tool_ratio", rho_boundaries, lambda row: observed(row)["rho_tool"]),
    ):
        labels = list(dict.fromkeys(label for _, label in boundaries))
        for label in labels:
            indices = [index for index, row in enumerate(rows)
                       if band(value_fn(row), boundaries) == label]
            record: dict[str, Any] = {
                "dimension": dimension, "bucket": label, "rows": len(indices)
            }
            for name, selected in method_specs:
                key = name.lower().replace(" ", "_").replace("-", "")
                count = sum(selected[index] for index in indices)
                record[f"{key}_selected"] = count
                record[f"{key}_share"] = count / len(indices) if indices else 0.0
            distribution_rows.append(record)

    decision_rows: list[dict[str, Any]] = []
    for row, fixed_selected, v2_selected, decision in zip(
        rows, fixed, bcrs_v2, decisions
    ):
        features = observed(row)
        decision_rows.append({
            "task_id": row["task_id"],
            "calibration": row["task_id"] in calibration_ids,
            **features,
            "fixed20k_selected": fixed_selected,
            "bcrs_v2_selected": v2_selected,
            "bcro_selected": decision["compress"],
            "bcro_reason": decision["reason"],
            "predicted_full_ms": decision.get("predicted_full_ms"),
            "predicted_racer_analytic_ms": decision.get("predicted_racer_analytic_ms"),
            "residual_mean": decision.get("residual_mean"),
            "residual_std": decision.get("residual_std"),
            "predicted_racer_ms": decision.get("predicted_racer_ms"),
            "uncertainty_penalty_ms": decision.get("uncertainty_penalty_ms"),
            "risk_penalty_ms": decision.get("risk_penalty_ms"),
            "utility_ms": decision.get("utility_ms"),
            "observed_full_online_ttft_ms": finite(row, "observed_full_online_ttft_ms"),
            "observed_c2kv_online_ttft_ms": finite(row, "observed_c2kv_online_ttft_ms"),
        })

    curve_rows: list[dict[str, Any]] = []
    for rho_tool in (0.25, 0.50, 0.75):
        for total in (4000, 8000, 12000, 16000, 20000, 24000,
                      32000, 48000, 64000, 96000, 128000, 192000):
            features = {
                "L_total": float(total),
                "L_tool": float(total * rho_tool),
                "L_hist": float(total * (1.0 - rho_tool)),
                "rho_tool": rho_tool,
                "S_tool": 0.0, "S_hist": 0.0, "D_cross": 0.0,
            }
            decision = scheduler.route_observed(features)
            curve_rows.append({
                "rho_tool": rho_tool,
                "L_total": total,
                "predicted_full_ms": decision.get("predicted_full_ms"),
                "predicted_racer_ms": decision.get("predicted_racer_ms"),
                "residual_std": decision.get("residual_std"),
                "uncertainty_penalty_ms": decision.get("uncertainty_penalty_ms"),
                "risk_penalty_ms": decision.get("risk_penalty_ms"),
                "utility_ms": decision.get("utility_ms"),
                "selected": decision["selected_arm"],
            })

    crossings = {}
    for rho_tool in (0.25, 0.50, 0.75):
        selected_lengths = [row["L_total"] for row in curve_rows
                            if row["rho_tool"] == rho_tool
                            and row["selected"] == "c2kv_racer"]
        crossings[str(rho_tool)] = {
            "first_sampled_racer_length": min(selected_lengths) if selected_lengths else None,
            "sampled_lengths_are_diagnostic_only": True,
        }

    report = {
        "schema": "bcro-stage-a-report-v1",
        "status": "exploratory_offline_only",
        "data_audit": {
            "source": str(args.historical_paired.resolve()),
            "source_sha256": source_sha,
            "rows": len(rows),
            "unique_tasks": len({row["task_id"] for row in rows}),
            "decisions_per_task": 1,
            "row_semantics": "longest same-input decision per BFCL Long task",
            "calibration_tasks": len(calibration),
            "heldout_tasks": len(heldout),
            "task_overlap": len(calibration_ids & {row["task_id"] for row in heldout}),
            "label": "log(C2KV cache-hit first-draft online TTFT / Full online TTFT)",
            "racer_proxy_limitation": "Recovery/final-action latency is not in this label.",
        },
        "leakage_audit": {
            "direct_task_overlap_between_fit_and_heldout": False,
            "heldout_labels_used_for_fit_or_route": False,
            "development_reuse_risk": (
                "All 197 tasks come from the fixed BFCL Long category and the 177-task "
                "held-out results were inspected during BCRS-v1/v2 development. They are "
                "not a pristine independent test set."
            ),
        },
        "features": {
            "gp": list(COST_FEATURE_NAMES),
            "full_analytic": ["log1p_L_total", "log1p_L_total_squared"],
            "racer_analytic": [
                "log1p_L_total", "log1p_L_total_squared", "rho_tool",
                "rho_hist", "log1p_L_total_x_rho_hist",
            ],
            "semantic": ["S_tool", "S_hist", "D_cross"],
            "historical_semantic_availability": False,
            "tool_length_is_proxy": True,
        },
        "model": {
            "full_fit": full_fit,
            "racer_fit": racer_fit,
            "residual_gp_fit": residual_fit,
            "routing_utility": "T_full - T_racer - beta*sigma_cost - lambda*semantic_risk*T_full",
            "uses_fixed_length_threshold": False,
            "routing_rule": rule,
        },
        "comparison": method_metrics,
        "bcro_reason_counts": dict(sorted(Counter(
            decision["reason"] for decision in decisions
        ).items())),
        "route_distribution": distribution_rows,
        "cost_curve_crossings": crossings,
        "router_cpu_overhead": {
            "calls": len(rows),
            "total_ms": route_seconds * 1000.0,
            "mean_us": route_seconds * 1e6 / len(rows),
        },
        "ttft_target_50pct_exploratory": (
            method_metrics["BCRO"]["all197"]["total_reduction"] >= 0.50
        ),
        "accuracy_available": False,
        "accuracy_drop_target_15pct": None,
        "bcro_minus_bcrs_v2_total_reduction": (
            method_metrics["BCRO"]["all197"]["total_reduction"]
            - method_metrics["BCRS-v2"]["all197"]["total_reduction"]
        ),
        "warnings": [
            "Offline routing uses one longest decision per task, not a closed-loop trajectory.",
            "Observed timing labels score frozen choices only; they are not route inputs.",
            "Historical semantic features are absent and are zero in this replay.",
            "Tool definitions remain Full/native protected in the measured backend.",
            "Accuracy and the <=15% quality target require a later closed-loop experiment.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "bcro_artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "stage_a_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.out_dir / "comparison.csv", comparison_rows)
    write_csv(args.out_dir / "routing_decisions_bcro.csv", decision_rows)
    write_csv(args.out_dir / "route_distribution.csv", distribution_rows)
    write_csv(args.out_dir / "cost_curves.csv", curve_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
