#!/usr/bin/env python3
"""Fit the frozen GP-SJCS residual model from historical paired TTFT rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import sklearn

from c2kv_eval.portable.gp_sjcs import (
    CANDIDATES, FEATURE_NAMES, analytic_log_ratio, model_features,
)


def _finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def _serialize_model(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-6, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True,
        n_restarts_optimizer=0, random_state=seed,
    ).fit(scaled, y)
    fitted = gp.kernel_
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
    prediction, std = gp.predict(scaled, return_std=True)
    report = {
        "kernel": str(gp.kernel_),
        "train_rows": len(y),
        "train_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "train_mae": float(np.mean(np.abs(prediction - y))),
        "predictive_std_mean": float(np.mean(std)),
    }
    return payload, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-paired", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--calibration-tasks", type=int, default=20)
    parser.add_argument("--history-budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.historical_paired.open(encoding="utf-8")))
    task_ids = sorted({row["task_id"] for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(task_ids)
    calibration = sorted(task_ids[:args.calibration_tasks])
    selected = [row for row in rows if row["task_id"] in set(calibration)]
    if not selected:
        raise ValueError("historical paired profile yielded no calibration rows")

    x_semantic: list[np.ndarray] = []
    x_length: list[np.ndarray] = []
    targets: list[float] = []
    a0, a1 = CANDIDATES[0], CANDIDATES[1]
    for row in selected:
        total = _finite(row, "full_context_tokens")
        history = _finite(row, "full_history_tokens")
        nonhistory = max(0.0, total - history)
        observed = {
            "L_total": total,
            # Historical files did not split Tool from protected/current KV.
            # Keep the measured non-history mass and record this limitation.
            "L_tool": nonhistory,
            "L_hist": history,
            "rho_tool": nonhistory / max(1.0, nonhistory + history),
            "S_tool": 0.0,
            "S_hist": 0.0,
            "D_cross": 0.0,
        }
        # Full is an exact zero-residual anchor.
        for length_only, sink in ((False, x_semantic), (True, x_length)):
            sink.append(model_features(
                observed, a0, args.history_budget, length_only=length_only))
        targets.append(0.0)

        ratio = _finite(row, "online_ttft_ratio")
        if ratio <= 0:
            continue
        measured_history_retention = min(
            1.0,
            _finite(row, "c2kv_resident_history_tokens") / max(1.0, history),
        )
        observed_a1 = dict(observed)
        # Preserve the real historical active fraction in the model feature.
        vector_sem = model_features(
            observed_a1, a1, args.history_budget, length_only=False)
        vector_len = model_features(
            observed_a1, a1, args.history_budget, length_only=True)
        vector_sem[-2:] = measured_history_retention
        vector_len[-2:] = measured_history_retention
        x_semantic.append(vector_sem)
        x_length.append(vector_len)
        targets.append(
            math.log(ratio) - analytic_log_ratio(
                observed_a1, a1, args.history_budget)
        )

    x_sem = np.vstack(x_semantic)
    x_len = np.vstack(x_length)
    y = np.asarray(targets, dtype=np.float64)
    if len(x_sem) != len(y) or len(x_len) != len(y):
        raise AssertionError("GP training feature/target length mismatch")

    started = time.perf_counter()
    semantic, semantic_report = _serialize_model(x_sem, y, args.seed)
    length, length_report = _serialize_model(x_len, y, args.seed)
    fit_seconds = time.perf_counter() - started
    source_hash = hashlib.sha256(args.historical_paired.read_bytes()).hexdigest()
    artifact = {
        "schema": "gp-sjcs-artifact-v1",
        "seed": args.seed,
        "dependency_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "feature_names": list(FEATURE_NAMES),
        "history_budget": args.history_budget,
        "calibration_task_ids": calibration,
        "calibration_task_count": len(calibration),
        "calibration_decision_count": len(selected),
        "source": str(args.historical_paired),
        "source_sha256": source_hash,
        "source_overlap_policy": (
            "calibration tasks overlap the 200-task category; report both "
            "held-out tasks and exploratory full-200 results"
        ),
        "training_backend": (
            "historical C2KV history cache-hit residual plus analytic active-token prior; "
            "joint physical-KV candidates are validated by smoke telemetry"
        ),
        "tool_length_limitation": (
            "historical profile lacks a Tool-only split; L_tool uses measured "
            "non-history prompt mass for calibration"
        ),
        "models": {"semantic": semantic, "length_only": length},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "gp-sjcs-fit-log-v1",
        "artifact": str(args.out),
        "fit_seconds": fit_seconds,
        "semantic": semantic_report,
        "length_only": length_report,
        "calibration_task_ids": calibration,
        "selected_historical_rows": len(selected),
        "expanded_train_rows": len(y),
        "warnings": [
            artifact["source_overlap_policy"],
            artifact["tool_length_limitation"],
            "Tool Definition C2KV gist compression is unavailable; dynamic joint arms use physical raw-KV eviction.",
        ],
    }
    args.log.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
