#!/usr/bin/env python3
"""BCRO Stage A2: stratified, task-disjoint CPU-only offline evaluation."""
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
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from c2kv_eval.portable.bcro import BCROScheduler, full_cost_features, racer_cost_features
from c2kv_eval.portable.bcrs import cost_features, prior_guided_decision


SEEDS = (0, 42, 2026)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)


def finite(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def observed(row: Mapping[str, Any]) -> dict[str, float]:
    total = finite(row, "full_context_tokens")
    history = finite(row, "full_history_tokens")
    tool_proxy = max(0.0, total - history)
    return {
        "L_total": total, "L_tool": tool_proxy, "L_hist": history,
        "rho_tool": tool_proxy / max(1.0, tool_proxy + history),
        "S_tool": 0.0, "S_hist": 0.0, "D_cross": 0.0,
    }


def length_bucket(total: float) -> str:
    if total < 8000:
        return "<8K"
    if total < 16000:
        return "8-16K"
    if total < 24000:
        return "16-24K"
    if total < 64000:
        return "24-64K"
    return ">=64K"


def ratio_bucket(rho: float) -> str:
    if rho < 0.25:
        return "<25%"
    if rho < 0.50:
        return "25-50%"
    if rho < 0.75:
        return "50-75%"
    return ">=75%"


def split_strata(rows):
    joint = Counter((length_bucket(observed(row)["L_total"]),
                     ratio_bucket(observed(row)["rho_tool"])) for row in rows)
    sparse_lengths = {length for (length, _), count in joint.items() if count < 5}
    strata = []
    for row in rows:
        features = observed(row)
        length = length_bucket(features["L_total"])
        ratio = ratio_bucket(features["rho_tool"])
        strata.append(f"{length}|marginal-rho" if length in sparse_lengths
                      else f"{length}|{ratio}")
    return strata, {
        "joint_counts": {f"{a}|{b}": n for (a, b), n in sorted(joint.items())},
        "marginalized_length_buckets": sorted(sparse_lengths),
        "effective_strata_counts": dict(sorted(Counter(strata).items())),
        "sparse_rule": "merge ratio cells within a length bucket if any observed joint cell has <5 rows",
    }


def make_split(rows, seed):
    strata, audit = split_strata(rows)
    indices = np.arange(len(rows))
    first = StratifiedShuffleSplit(n_splits=1, train_size=60, random_state=seed)
    train_pos, remaining_pos = next(first.split(indices, strata))
    remaining = indices[remaining_pos]
    remaining_strata = np.asarray(strata)[remaining_pos]
    second = StratifiedShuffleSplit(n_splits=1, train_size=20, random_state=seed + 10000)
    validation_rel, test_rel = next(second.split(remaining, remaining_strata))
    train = list(map(int, indices[train_pos]))
    validation = list(map(int, remaining[validation_rel]))
    test = list(map(int, remaining[test_rel]))
    parts = [set(train), set(validation), set(test)]
    if [len(part) for part in parts] != [60, 20, 117] or any(
        parts[a] & parts[b] for a in range(3) for b in range(a + 1, 3)
    ):
        raise AssertionError("invalid task-disjoint Stage A2 split")
    for name, part in (("train", train), ("test", test)):
        covered = {length_bucket(observed(rows[index])["L_total"]) for index in part}
        if len(covered) != 5:
            raise AssertionError(f"{name} misses a length bucket: {covered}")
    audit["split_counts"] = {"train": 60, "validation": 20, "test": 117}
    return train, validation, test, audit


def ridge_fit(rows, feature_fn: Callable, label: str, alpha: float):
    x = np.vstack([feature_fn(observed(row)) for row in rows])
    y = np.log([finite(row, label) for row in rows])
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return {"scaler": scaler, "model": model, "alpha": alpha}


def ridge_predict(fitted, rows, feature_fn):
    x = np.vstack([feature_fn(observed(row)) for row in rows])
    return fitted["model"].predict(fitted["scaler"].transform(x))


def ridge_payload(fitted):
    scaler, model = fitted["scaler"], fitted["model"]
    return {
        "kind": "standardized_ridge_log_cost", "alpha": float(fitted["alpha"]),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(), "intercept": float(model.intercept_),
    }


def select_alpha(train, validation):
    candidates = []
    for alpha in RIDGE_ALPHAS:
        full = ridge_fit(train, full_cost_features, "full_online_ttft_ms", alpha)
        racer = ridge_fit(train, racer_cost_features, "c2kv_online_ttft_ms", alpha)
        full_error = ridge_predict(full, validation, full_cost_features) - np.log(
            [finite(row, "full_online_ttft_ms") for row in validation])
        racer_error = ridge_predict(racer, validation, racer_cost_features) - np.log(
            [finite(row, "c2kv_online_ttft_ms") for row in validation])
        score = float(np.sqrt(np.mean(full_error ** 2)) + np.sqrt(np.mean(racer_error ** 2)))
        candidates.append({"alpha": alpha, "validation_sum_rmse": score})
    selected = min(candidates, key=lambda row: (row["validation_sum_rmse"], row["alpha"]))
    return float(selected["alpha"]), candidates


def gp_fit(x, y, seed):
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(x.shape[1]),
                       length_scale_bounds=(1e-2, 1e3), nu=1.5)
              + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-6, 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=0, random_state=seed,
        ).fit(scaled, y)
    return {"scaler": scaler, "gp": gp}


def gp_predict(fitted, x):
    return fitted["gp"].predict(fitted["scaler"].transform(x), return_std=True)


def gp_payload(fitted):
    scaler, gp, kernel = fitted["scaler"], fitted["gp"], fitted["gp"].kernel_
    return {
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "x_train_scaled": gp.X_train_.tolist(), "alpha": gp.alpha_.tolist(),
        "cholesky": gp.L_.tolist(), "constant": float(kernel.k1.k1.constant_value),
        "length_scale": np.asarray(kernel.k1.k2.length_scale).reshape(-1).tolist(),
        "noise": float(kernel.k2.noise_level),
        "y_mean": float(np.asarray(gp._y_train_mean).reshape(-1)[0]),
        "y_std": float(np.asarray(gp._y_train_std).reshape(-1)[0]),
    }


def log_ratio(rows):
    return np.log([finite(row, "c2kv_online_ttft_ms") /
                   finite(row, "full_online_ttft_ms") for row in rows])


def analytic_ratio(full_model, racer_model, rows):
    return (ridge_predict(racer_model, rows, racer_cost_features)
            - ridge_predict(full_model, rows, full_cost_features))


def fit_direct_gp(rows, seed):
    return gp_fit(np.vstack([cost_features(observed(row)) for row in rows]),
                  log_ratio(rows), seed)


def fit_residual_gp(rows, full_model, racer_model, seed):
    residual = log_ratio(rows) - analytic_ratio(full_model, racer_model, rows)
    return gp_fit(np.vstack([cost_features(observed(row)) for row in rows]), residual, seed)


def oof_predictions(rows, alpha, seed, kind):
    actual = log_ratio(rows)
    prediction = np.empty(len(rows)); uncertainty = np.empty(len(rows))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    indices = np.arange(len(rows))
    for fold, (fit_pos, predict_pos) in enumerate(splitter.split(indices)):
        fit_rows = [rows[index] for index in fit_pos]
        predict_rows = [rows[index] for index in predict_pos]
        x_predict = np.vstack([cost_features(observed(row)) for row in predict_rows])
        if kind == "direct":
            fitted = fit_direct_gp(fit_rows, seed + fold + 1)
            mean, std = gp_predict(fitted, x_predict)
            prediction[predict_pos] = mean
        else:
            full = ridge_fit(fit_rows, full_cost_features, "full_online_ttft_ms", alpha)
            racer = ridge_fit(fit_rows, racer_cost_features, "c2kv_online_ttft_ms", alpha)
            fitted = fit_residual_gp(fit_rows, full, racer, seed + fold + 101)
            mean, std = gp_predict(fitted, x_predict)
            prediction[predict_pos] = analytic_ratio(full, racer, predict_rows) + mean
        uncertainty[predict_pos] = std
    return actual, prediction, uncertainty


def uncertainty_scale(actual, prediction, std):
    standardized = np.abs(actual - prediction) / np.maximum(std, 1e-12)
    return float(max(1.0, np.quantile(standardized, 0.90) / 1.6448536269514722))


def error_metrics(actual, prediction):
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    return {
        "mae": float(np.mean(np.abs(prediction - actual))),
        "rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
        "direction_accuracy": float(np.mean((prediction < 0) == (actual < 0))),
    }


def uncertainty_metrics(actual, prediction, raw_std, scale):
    error = np.abs(np.asarray(actual) - np.asarray(prediction)); raw = np.asarray(raw_std)
    return {
        "scale": float(scale),
        "raw_68_coverage": float(np.mean(error <= raw)),
        "calibrated_68_coverage": float(np.mean(error <= raw * scale)),
        "raw_90_coverage": float(np.mean(error <= 1.6448536269514722 * raw)),
        "calibrated_90_coverage": float(np.mean(error <= 1.6448536269514722 * raw * scale)),
        "raw_95_coverage": float(np.mean(error <= 1.959963984540054 * raw)),
        "calibrated_95_coverage": float(np.mean(error <= 1.959963984540054 * raw * scale)),
    }


def route_metrics(rows, selected):
    full = [finite(row, "full_online_ttft_ms") for row in rows]
    racer = [finite(row, "c2kv_online_ttft_ms") for row in rows]
    routed = [racer[index] if use else full[index] for index, use in enumerate(selected)]
    return {
        "tasks": len(rows), "compressed": int(sum(selected)),
        "compression_share": float(sum(selected) / len(rows)),
        "total_reduction": float(1 - sum(routed) / sum(full)),
        "routed_mean_ms": float(statistics.mean(routed)),
        "routed_p50_ms": float(np.quantile(routed, .50)),
        "routed_p95_ms": float(np.quantile(routed, .95)),
    }


def distribution(rows):
    return {
        "length": dict(sorted(Counter(length_bucket(observed(row)["L_total"]) for row in rows).items())),
        "tool_ratio": dict(sorted(Counter(ratio_bucket(observed(row)["rho_tool"]) for row in rows).items())),
    }


def bucket_report(rows, selected, predicted):
    output = []
    for dimension, bucket_fn in (
        ("length", lambda row: length_bucket(observed(row)["L_total"])),
        ("tool_ratio", lambda row: ratio_bucket(observed(row)["rho_tool"])),
    ):
        for bucket in sorted({bucket_fn(row) for row in rows}):
            indices = [index for index, row in enumerate(rows) if bucket_fn(row) == bucket]
            chosen = [index for index in indices if selected[index]]
            actual = [1 - finite(rows[index], "c2kv_online_ttft_ms") /
                      finite(rows[index], "full_online_ttft_ms") for index in chosen]
            output.append({
                "dimension": dimension, "bucket": bucket, "tasks": len(indices),
                "selected": len(chosen), "selection_share": len(chosen) / len(indices),
                "mean_predicted_reduction_selected": float(np.mean(1 - np.exp(predicted[chosen]))) if chosen else None,
                "mean_actual_reduction_selected": float(np.mean(actual)) if chosen else None,
                "wrong_compression_count": sum(value <= 0 for value in actual),
                "net_selected_reduction": (1 - sum(finite(rows[index], "c2kv_online_ttft_ms") for index in chosen) /
                                           sum(finite(rows[index], "full_online_ttft_ms") for index in chosen)) if chosen else None,
            })
    return output


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate_seed(rows, seed, output_dir):
    train_idx, validation_idx, test_idx, split_audit = make_split(rows, seed)
    train = [rows[index] for index in train_idx]
    validation = [rows[index] for index in validation_idx]
    test = [rows[index] for index in test_idx]
    alpha, alpha_candidates = select_alpha(train, validation)
    full_model = ridge_fit(train, full_cost_features, "full_online_ttft_ms", alpha)
    racer_model = ridge_fit(train, racer_cost_features, "c2kv_online_ttft_ms", alpha)
    direct_gp = fit_direct_gp(train, seed)
    residual_gp = fit_residual_gp(train, full_model, racer_model, seed + 1000)
    direct_oof = oof_predictions(train, alpha, seed, "direct")
    residual_oof = oof_predictions(train, alpha, seed, "residual")
    direct_scale = uncertainty_scale(*direct_oof)
    residual_scale = uncertainty_scale(*residual_oof)
    artifact = {
        "schema": "bcro-artifact-v1", "variant": "stage_a2_stratified_threshold_free",
        "frozen": True, "seed": seed,
        "calibration_task_ids": sorted(row["task_id"] for row in train),
        "validation_task_ids": sorted(row["task_id"] for row in validation),
        "full_cost_model": ridge_payload(full_model), "racer_cost_model": ridge_payload(racer_model),
        "residual_gp": gp_payload(residual_gp),
        "routing_rule": {
            "uncertainty_beta": 1.0, "uncertainty_scale": residual_scale,
            "semantic_risk_lambda": 0.0, "maximum_semantic_risk": 1.0,
            "semantic_risk_weights": {"s_tool": .20, "s_hist": .45, "d_cross": .35},
            "risk_status": "disabled_missing_historical_features",
            "fallback": "full_with_recorded_predictor_error",
        },
        "selected_backend": "existing_c2kv_racer_8x_fixed256",
        "tool_kv_mode": "full_native_protected",
        "online_feature_contract": ["L_total", "L_tool", "L_hist", "rho_tool", "S_tool", "S_hist", "D_cross"],
    }
    scheduler = BCROScheduler(artifact)
    started = time.perf_counter()
    decisions = [scheduler.route_observed(observed(row)) for row in test]
    route_seconds = time.perf_counter() - started
    bcro_selected = [bool(decision["compress"]) for decision in decisions]
    bcro_prediction = np.asarray([math.log(decision["predicted_racer_ms"] /
                                                decision["predicted_full_ms"]) for decision in decisions])
    x_test = np.vstack([cost_features(observed(row)) for row in test])
    direct_prediction, direct_test_std = gp_predict(direct_gp, x_test)
    bcrs_rule = {
        "history_length_prior_tokens": 20000, "v2_full_below_tokens": 16000,
        "v2_racer_at_or_above_tokens": 24000, "v2_high_gain_probability": .95,
        "v2_low_gain_probability": .05,
        "semantic_risk_weights": {"s_tool": .20, "s_hist": .45, "d_cross": .35},
        "maximum_semantic_risk": 1.0,
    }
    bcrs_selected = [prior_guided_decision(observed(row), float(mean),
                     float(std * direct_scale), bcrs_rule)["compress"]
                     for row, mean, std in zip(test, direct_prediction, direct_test_std)]
    fixed_selected = [observed(row)["L_total"] >= 20000 for row in test]
    actual_test = log_ratio(test)
    analytic_test = analytic_ratio(full_model, racer_model, test)
    residual_test_mean, residual_test_std = gp_predict(residual_gp, x_test)
    residual_test_prediction = analytic_test + residual_test_mean
    x_train = np.vstack([cost_features(observed(row)) for row in train])
    direct_train_prediction, _ = gp_predict(direct_gp, x_train)
    residual_train_mean, _ = gp_predict(residual_gp, x_train)
    residual_train_prediction = analytic_ratio(full_model, racer_model, train) + residual_train_mean
    report = {
        "seed": seed,
        "split": {"audit": split_audit, "train": distribution(train),
                  "validation": distribution(validation), "test": distribution(test), "task_overlap": 0},
        "model_selection": {"ridge_alpha_candidates": alpha_candidates,
                            "selected_ridge_alpha": alpha,
                            "selection_source": "validation20_only",
                            "gp_kernel_source": "optimized_on_train60_only"},
        "prediction": {
            "direct_gp": {
                "train_in_sample": error_metrics(log_ratio(train), direct_train_prediction),
                "train_oof": error_metrics(direct_oof[0], direct_oof[1]),
                "test": error_metrics(actual_test, direct_prediction),
                "uncertainty_oof": uncertainty_metrics(*direct_oof, direct_scale),
                "uncertainty_test": uncertainty_metrics(actual_test, direct_prediction, direct_test_std, direct_scale),
                "kernel": str(direct_gp["gp"].kernel_)},
            "bcro_analytic": {"test": error_metrics(actual_test, analytic_test)},
            "bcro_analytic_plus_residual": {
                "train_in_sample": error_metrics(log_ratio(train), residual_train_prediction),
                "train_oof": error_metrics(residual_oof[0], residual_oof[1]),
                "test": error_metrics(actual_test, residual_test_prediction),
                "uncertainty_oof": uncertainty_metrics(*residual_oof, residual_scale),
                "uncertainty_test": uncertainty_metrics(actual_test, residual_test_prediction,
                                                        residual_test_std, residual_scale),
                "kernel": str(residual_gp["gp"].kernel_)},
        },
        "test_routes": {"Fixed 20K": route_metrics(test, fixed_selected),
                        "BCRS-v2": route_metrics(test, bcrs_selected),
                        "BCRO": route_metrics(test, bcro_selected)},
        "bucket_metrics": bucket_report(test, bcro_selected, bcro_prediction),
        "router_cpu": {"calls": len(test), "total_ms": route_seconds * 1000,
                       "mean_us": route_seconds * 1e6 / len(test)},
        "fallback_count": sum(decision["reason"] == "predictor_invalid_full_fallback" for decision in decisions),
        "semantic_risk_status": "disabled: source has no semantic risk observations",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "bcro_artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    split_rows = []
    for name, indices in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
        for index in indices:
            features = observed(rows[index])
            split_rows.append({"task_id": rows[index]["task_id"], "split": name,
                "length_bucket": length_bucket(features["L_total"]),
                "tool_ratio_bucket": ratio_bucket(features["rho_tool"]),
                "L_total": features["L_total"], "rho_tool": features["rho_tool"]})
    write_csv(output_dir / "split.csv", split_rows)
    predictions = []
    for row, actual, direct, decision, bcrs, fixed in zip(test, actual_test,
            direct_prediction, decisions, bcrs_selected, fixed_selected):
        predictions.append({"task_id": row["task_id"], "actual_log_ratio": actual,
            "direct_gp_log_ratio": direct,
            "bcro_log_ratio": math.log(decision["predicted_racer_ms"] / decision["predicted_full_ms"]),
            "bcro_uncertainty": decision["residual_std"], "fixed20k_selected": fixed,
            "bcrs_v2_selected": bcrs, "bcro_selected": decision["compress"],
            "bcro_reason": decision["reason"]})
    write_csv(output_dir / "test_predictions.csv", predictions)
    write_csv(output_dir / "bucket_metrics.csv", report["bucket_metrics"])
    return report


def aggregate(reports):
    rows = []
    for report in reports:
        for method, metrics in report["test_routes"].items():
            rows.append({"seed": report["seed"], "method": method, **metrics})
    output = {}
    for method in ("Fixed 20K", "BCRS-v2", "BCRO"):
        selected = [row for row in rows if row["method"] == method]
        output[method] = {}
        for key in ("compressed", "compression_share", "total_reduction",
                    "routed_mean_ms", "routed_p50_ms", "routed_p95_ms"):
            values = [float(row[key]) for row in selected]
            output[method][key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                   "min": float(np.min(values)), "max": float(np.max(values))}
    rmse = [r["prediction"]["bcro_analytic_plus_residual"]["test"]["rmse"] for r in reports]
    direction = [r["prediction"]["bcro_analytic_plus_residual"]["test"]["direction_accuracy"] for r in reports]
    output["BCRO_prediction"] = {
        "test_rmse": {"mean": float(np.mean(rmse)), "std": float(np.std(rmse)), "worst": float(max(rmse))},
        "test_direction_accuracy": {"mean": float(np.mean(direction)),
            "std": float(np.std(direction)), "worst": float(min(direction))}}
    return rows, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-paired", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.historical_paired.open(encoding="utf-8")))
    if len(rows) != 197 or len({row["task_id"] for row in rows}) != 197:
        raise ValueError("Stage A2 requires 197 unique Task IDs")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    reports = [evaluate_seed(rows, seed, args.out_dir / f"seed_{seed}") for seed in SEEDS]
    comparison, aggregate_report = aggregate(reports)
    top = {
        "schema": "bcro-stage-a2-report-v1", "status": "exploratory_offline_only",
        "source": str(args.historical_paired.resolve()),
        "source_sha256": hashlib.sha256(args.historical_paired.read_bytes()).hexdigest(),
        "seeds": list(SEEDS), "train_validation_test": [60, 20, 117],
        "split": "joint length/tool-ratio stratification with documented marginal fallback",
        "label_isolation": "split uses only task_id, context length, and tool/history proxy ratio",
        "risk_status": "disabled; historical semantic risk features unavailable",
        "aggregate": aggregate_report, "per_seed": reports,
        "prior_stage_a_reference": {"train_tasks": 20, "heldout_tasks": 177,
            "bcro_test_rmse": 0.13644326786341596,
            "bcro_direction_accuracy": 0.8531073446327684},
        "warnings": [
            "The 197 tasks were previously used during method development; this is not an independent test set.",
            "C2KV cache-hit first-draft TTFT is the Racer cost proxy; recovery latency is absent.",
            "Tool length is full_context minus history, not an independently measured Tool token span.",
            "No closed-loop accuracy conclusion can be drawn from Stage A2."]}
    (args.out_dir / "stage_a2_report.json").write_text(json.dumps(top, indent=2) + "\n")
    write_csv(args.out_dir / "seed_comparison.csv", comparison)
    print(json.dumps({"output": str(args.out_dir), "aggregate": aggregate_report}, indent=2))


if __name__ == "__main__":
    main()
