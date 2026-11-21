#!/usr/bin/env python3
"""Fit and audit the frozen binary BCRS cost model on paired TTFT rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from c2kv_eval.portable.bcrs import COST_FEATURE_NAMES, cost_features


def finite(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def observed(row):
    total = finite(row, "full_context_tokens")
    history = finite(row, "full_history_tokens")
    # The historical longest-decision file has no Tool-only split.  This is
    # an explicitly recorded proxy, never presented as measured Tool tokens.
    tool_proxy = max(0.0, total - history)
    return {
        "L_total": total,
        "L_tool": tool_proxy,
        "L_hist": history,
        "rho_tool": tool_proxy / max(1.0, tool_proxy + history),
        "S_tool": 0.0,
        "S_hist": 0.0,
        "D_cross": 0.0,
    }


def serialize_gp(x, y, seed):
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(x.shape[1]),
                       length_scale_bounds=(1e-2, 1e3), nu=1.5)
              + WhiteKernel(noise_level=0.03,
                            noise_level_bounds=(1e-6, 1.0)))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True,
        n_restarts_optimizer=0, random_state=seed,
    ).fit(scaled, y)
    fitted = gp.kernel_
    prediction, std = gp.predict(scaled, return_std=True)
    payload = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "x_train_scaled": gp.X_train_.tolist(),
        "alpha": gp.alpha_.tolist(),
        "cholesky": gp.L_.tolist(),
        "constant": float(fitted.k1.k1.constant_value),
        "length_scale": np.asarray(fitted.k1.k2.length_scale).reshape(-1).tolist(),
        "noise": float(fitted.k2.noise_level),
        "y_mean": float(np.asarray(gp._y_train_mean).reshape(-1)[0]),
        "y_std": float(np.asarray(gp._y_train_std).reshape(-1)[0]),
    }
    return gp, scaler, payload, {
        "kernel": str(gp.kernel_),
        "train_rows": len(y),
        "train_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "train_mae": float(np.mean(np.abs(prediction - y))),
        "predictive_std_mean": float(np.mean(std)),
    }


def percentile(values, q):
    values = sorted(float(v) for v in values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    alpha = position - lo
    return values[lo] * (1 - alpha) + values[hi] * alpha


def metrics(rows, selected):
    full = [finite(row, "full_online_ttft_ms") for row in rows]
    routed = [finite(row, "c2kv_online_ttft_ms") if use else full[index]
              for index, (row, use) in enumerate(zip(rows, selected))]
    return {
        "rows": len(rows),
        "compressed": sum(bool(value) for value in selected),
        "compression_share": sum(bool(value) for value in selected) / len(rows),
        "full_total_online_ttft_ms": sum(full),
        "routed_total_online_ttft_ms": sum(routed),
        "total_reduction": 1.0 - sum(routed) / sum(full),
        "routed_mean_ms": statistics.mean(routed),
        "routed_p50_ms": percentile(routed, 0.50),
        "routed_p95_ms": percentile(routed, 0.95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-paired", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.historical_paired.open(encoding="utf-8")))
    if len(rows) != 197 or len({row["task_id"] for row in rows}) != 197:
        raise ValueError("Stage A expects one longest-decision row for each of 197 tasks")
    task_ids = sorted(row["task_id"] for row in rows)
    random.Random(args.seed).shuffle(task_ids)
    calibration_ids = set(task_ids[:args.calibration_tasks])
    calibration = [row for row in rows if row["task_id"] in calibration_ids]
    heldout = [row for row in rows if row["task_id"] not in calibration_ids]
    x = np.vstack([cost_features(observed(row)) for row in calibration])
    y = np.asarray([math.log(finite(row, "online_ttft_ratio"))
                    for row in calibration], dtype=np.float64)
    started = time.perf_counter()
    gp, scaler, gp_payload, fit_report = serialize_gp(x, y, args.seed)

    def predictions(target_rows):
        vectors = np.vstack([cost_features(observed(row)) for row in target_rows])
        mean, std = gp.predict(scaler.transform(vectors), return_std=True)
        return list(map(float, mean)), list(map(float, std))

    cal_mean, cal_std = predictions(calibration)
    target_share = 39 / 197
    candidates = []
    for minimum in (18000, 20000, 22000, 24000):
        for beta in (0.0, 0.10, 0.25, 0.50):
            for threshold in (math.log(0.95), math.log(0.90), math.log(0.80), 0.0):
                for max_uncertainty in (percentile(cal_std, 0.75),
                                        percentile(cal_std, 0.90),
                                        max(cal_std) + 1e-12):
                    chosen = [
                        observed(row)["L_total"] >= minimum
                        and mean + beta * std <= threshold
                        and std <= max_uncertainty
                        for row, mean, std in zip(calibration, cal_mean, cal_std)
                    ]
                    report = metrics(calibration, chosen)
                    if report["compression_share"] > 0.25:
                        continue
                    # Calibration objective: real cost reduction with a mild,
                    # predeclared preference for the historical 39/197 long tail.
                    score = (report["total_reduction"]
                             - 0.10 * abs(report["compression_share"] - target_share))
                    candidates.append((score, -minimum, -beta, threshold,
                                       max_uncertainty, report, chosen))
    if not candidates:
        raise RuntimeError("no BCRS calibration rule satisfied the compression-share constraint")
    best = max(candidates, key=lambda item: item[:5])
    _, neg_minimum, neg_beta, threshold, max_uncertainty, cal_metrics, _ = best
    rule = {
        "minimum_context_tokens": int(-neg_minimum),
        "uncertainty_beta": float(-neg_beta),
        "compression_objective_threshold": float(threshold),
        "maximum_uncertainty": float(max_uncertainty),
        # Frozen before online smoke; no benchmark correctness labels used.
        "semantic_risk_weight": 0.35,
        "maximum_semantic_risk": 0.65,
        "semantic_risk_weights": {"s_tool": 0.20, "s_hist": 0.45, "d_cross": 0.35},
        "history_length_prior_tokens": 20000,
        "target_compression_share": target_share,
    }
    all_mean, all_std = predictions(rows)
    selected_bcrs = [
        observed(row)["L_total"] >= rule["minimum_context_tokens"]
        and mean + rule["uncertainty_beta"] * std
            <= rule["compression_objective_threshold"]
        and std <= rule["maximum_uncertainty"]
        for row, mean, std in zip(rows, all_mean, all_std)
    ]
    selected_fixed = [observed(row)["L_total"] >= 20000 for row in rows]
    heldout_indices = [index for index, row in enumerate(rows)
                       if row["task_id"] not in calibration_ids]
    stage_a = {
        "schema": "bcrs-stage-a-report-v1",
        "status": "exploratory_only",
        "fit_seconds": time.perf_counter() - started,
        "source_rows": len(rows),
        "source_semantics": "one longest same-input decision per BFCL Long task",
        "calibration_task_count": len(calibration),
        "heldout_task_count": len(heldout),
        "fixed20k_all197": metrics(rows, selected_fixed),
        "bcrs_all197": metrics(rows, selected_bcrs),
        "fixed20k_heldout": metrics(heldout, [selected_fixed[i] for i in heldout_indices]),
        "bcrs_heldout": metrics(heldout, [selected_bcrs[i] for i in heldout_indices]),
        "bcrs_calibration": cal_metrics,
        "gp_fit": fit_report,
        "routing_rule": rule,
        "accuracy_available": False,
        "warnings": [
            "This offline analysis uses longest-decision rows and cannot establish closed-loop accuracy.",
            "Historical rows lack Tool-only and semantic features; non-history tokens are a Tool-length proxy.",
            "The cost label is C2KV cache-hit first-draft TTFT, used as a Racer first-draft proxy.",
            "Calibration tasks overlap the 200-task category; held-out-177 and all-197 are both reported.",
            "Tool definitions remain Full/native; Tool C2KV gist compression is not supported by this path.",
        ],
    }
    source_sha = hashlib.sha256(args.historical_paired.read_bytes()).hexdigest()
    artifact = {
        "schema": "bcrs-artifact-v1",
        "seed": args.seed,
        "feature_names": list(COST_FEATURE_NAMES),
        "dependency_versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "source": str(args.historical_paired.resolve()),
        "source_sha256": source_sha,
        "source_semantics": stage_a["source_semantics"],
        "calibration_task_ids": sorted(calibration_ids),
        "cost_gp": gp_payload,
        "routing_rule": rule,
        "online_feature_contract": ["L_total", "L_tool", "L_hist", "rho_tool",
                                    "S_tool", "S_hist", "D_cross"],
        "selected_backend": "existing_c2kv_racer_8x_fixed256",
        "tool_kv_mode": "full_native_protected",
        "frozen": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "bcrs_artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "stage_a_report.json").write_text(
        json.dumps(stage_a, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.out_dir / "routing_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["task_id", "full_context_tokens", "full_history_tokens", "calibration",
                  "predicted_log_ratio", "uncertainty", "fixed20k_selected", "bcrs_selected",
                  "observed_full_online_ttft_ms", "observed_c2kv_online_ttft_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, mean, std, fixed, bcrs in zip(rows, all_mean, all_std,
                                               selected_fixed, selected_bcrs):
            writer.writerow({
                "task_id": row["task_id"],
                "full_context_tokens": row["full_context_tokens"],
                "full_history_tokens": row["full_history_tokens"],
                "calibration": row["task_id"] in calibration_ids,
                "predicted_log_ratio": mean,
                "uncertainty": std,
                "fixed20k_selected": fixed,
                "bcrs_selected": bcrs,
                "observed_full_online_ttft_ms": row["full_online_ttft_ms"],
                "observed_c2kv_online_ttft_ms": row["c2kv_online_ttft_ms"],
            })
    print(json.dumps(stage_a, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
