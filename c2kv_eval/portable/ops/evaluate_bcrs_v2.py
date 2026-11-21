#!/usr/bin/env python3
"""Replay Prior-guided BCRS-v2 from frozen Stage-A predictions.

This command does not fit a model or invoke a serving runtime. It consumes the
existing BCRS-v1 artifact/predictions and the paired Full/C2KV timing rows.
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
from typing import Any

from c2kv_eval.portable.bcrs import prior_guided_decision


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def observed(row: dict[str, str]) -> dict[str, float]:
    total = finite(row, "full_context_tokens")
    history = finite(row, "full_history_tokens")
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


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metrics(rows: list[dict[str, str]], selected: list[bool]) -> dict[str, Any]:
    full = [finite(row, "observed_full_online_ttft_ms") for row in rows]
    compressed = [finite(row, "observed_c2kv_online_ttft_ms") for row in rows]
    routed = [compressed[index] if use else full[index]
              for index, use in enumerate(selected)]
    return {
        "rows": len(rows),
        "compressed": sum(selected),
        "compression_share": sum(selected) / len(rows),
        "full_total_online_ttft_ms": sum(full),
        "routed_total_online_ttft_ms": sum(routed),
        "total_reduction": 1.0 - sum(routed) / sum(full),
        "routed_mean_ms": statistics.mean(routed),
        "routed_p50_ms": percentile(routed, 0.50),
        "routed_p95_ms": percentile(routed, 0.95),
    }


def agreement(reference: list[bool], candidate: list[bool]) -> dict[str, Any]:
    count = sum(left == right for left, right in zip(reference, candidate))
    return {"matching_rows": count, "rows": len(reference), "rate": count / len(reference)}


def v1_rejection_reason(row: dict[str, str], rule: dict[str, Any]) -> str:
    total = finite(row, "full_context_tokens")
    mean = finite(row, "predicted_log_ratio")
    uncertainty = finite(row, "uncertainty")
    if total < float(rule["minimum_context_tokens"]):
        return "below_length_prior"
    if uncertainty > float(rule["maximum_uncertainty"]):
        return "uncertainty_guard"
    if mean + float(rule["uncertainty_beta"]) * uncertainty > float(
        rule["compression_objective_threshold"]
    ):
        return "insufficient_predicted_benefit"
    return "other"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--historical-paired", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    artifact_path = args.v1_dir / "bcrs_artifact.json"
    predictions_path = args.v1_dir / "routing_decisions.csv"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema") != "bcrs-artifact-v1" or not artifact.get("frozen"):
        raise ValueError("a frozen bcrs-artifact-v1 is required")
    source_sha = hashlib.sha256(args.historical_paired.read_bytes()).hexdigest()
    if source_sha != artifact["source_sha256"]:
        raise ValueError("historical paired CSV does not match the frozen artifact")

    predictions = list(csv.DictReader(predictions_path.open(encoding="utf-8")))
    historical = list(csv.DictReader(args.historical_paired.open(encoding="utf-8")))
    historical_by_task = {row["task_id"]: row for row in historical}
    if len(predictions) != 197 or set(historical_by_task) != {
        row["task_id"] for row in predictions
    }:
        raise ValueError("Stage A v2 requires the same 197 tasks as Stage A v1")

    rows: list[dict[str, str]] = []
    for prediction in predictions:
        source = historical_by_task[prediction["task_id"]]
        merged = dict(prediction)
        # Keep the measured labels from the frozen paired source authoritative.
        merged["observed_full_online_ttft_ms"] = source["full_online_ttft_ms"]
        merged["observed_c2kv_online_ttft_ms"] = source["c2kv_online_ttft_ms"]
        rows.append(merged)

    rule = dict(artifact["routing_rule"])
    v2_config = {
        "v2_full_below_tokens": 16000,
        "v2_racer_at_or_above_tokens": 24000,
        "v2_high_gain_probability": 0.95,
        "v2_low_gain_probability": 0.05,
        "uncertain_fallback": "fixed20k_prior",
    }
    rule.update(v2_config)

    fixed = [parse_bool(row["fixed20k_selected"]) for row in rows]
    v1 = [parse_bool(row["bcrs_selected"]) for row in rows]
    v2: list[bool] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        decision = prior_guided_decision(
            observed(row), finite(row, "predicted_log_ratio"),
            finite(row, "uncertainty"), rule,
        )
        v2.append(decision["compress"])
        decisions.append(decision)

    calibration_ids = set(artifact["calibration_task_ids"])
    heldout_indices = [index for index, row in enumerate(rows)
                       if row["task_id"] not in calibration_ids]
    heldout_rows = [rows[index] for index in heldout_indices]

    missed_indices = [index for index, (prior, selected) in enumerate(zip(fixed, v1))
                      if prior and not selected]
    rejection_counts = Counter(
        v1_rejection_reason(rows[index], artifact["routing_rule"])
        for index in missed_indices
    )
    rejection_bands = Counter(
        "20K-24K" if finite(rows[index], "full_context_tokens") < 24000 else ">=24K"
        for index in missed_indices
    )

    comparison_rows = []
    method_specs = (
        ("Fixed 20K", fixed),
        ("BCRS-v1", v1),
        ("BCRS-v2", v2),
    )
    for method, selected in method_specs:
        all_metrics = metrics(rows, selected)
        heldout_selected = [selected[index] for index in heldout_indices]
        heldout_metrics = metrics(heldout_rows, heldout_selected)
        comparison_rows.append({
            "method": method,
            "compressed_tasks": all_metrics["compressed"],
            "total_tasks": all_metrics["rows"],
            "compression_share": all_metrics["compression_share"],
            "total_ttft_reduction": all_metrics["total_reduction"],
            "heldout_compressed_tasks": heldout_metrics["compressed"],
            "heldout_tasks": heldout_metrics["rows"],
            "heldout_ttft_reduction": heldout_metrics["total_reduction"],
            "agreement_with_fixed20k": agreement(fixed, selected)["rate"],
        })

    decision_rows = []
    for row, prior, old, new, decision in zip(rows, fixed, v1, v2, decisions):
        decision_rows.append({
            "task_id": row["task_id"],
            "full_context_tokens": int(finite(row, "full_context_tokens")),
            "full_history_tokens": int(finite(row, "full_history_tokens")),
            "calibration": parse_bool(row["calibration"]),
            "predicted_log_ratio": finite(row, "predicted_log_ratio"),
            "uncertainty": finite(row, "uncertainty"),
            "gain_probability": decision["gain_probability"],
            "semantic_risk": decision["semantic_risk"],
            "fixed20k_selected": prior,
            "bcrs_v1_selected": old,
            "bcrs_v2_selected": new,
            "bcrs_v2_reason": decision["reason"],
            "observed_full_online_ttft_ms": finite(row, "observed_full_online_ttft_ms"),
            "observed_c2kv_online_ttft_ms": finite(row, "observed_c2kv_online_ttft_ms"),
        })

    missed_rows = []
    for index in missed_indices:
        row = rows[index]
        missed_rows.append({
            "task_id": row["task_id"],
            "full_context_tokens": int(finite(row, "full_context_tokens")),
            "predicted_log_ratio": finite(row, "predicted_log_ratio"),
            "uncertainty": finite(row, "uncertainty"),
            "v1_rejection_reason": v1_rejection_reason(row, artifact["routing_rule"]),
            "bcrs_v2_selected": v2[index],
            "bcrs_v2_reason": decisions[index]["reason"],
        })

    report = {
        "schema": "bcrs-stage-a-v2-report-v1",
        "status": "exploratory_offline_only",
        "source_rows": len(rows),
        "source_semantics": artifact["source_semantics"],
        "reused_frozen_gp": True,
        "gp_retrained": False,
        "source_sha256": source_sha,
        "v1_artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "v2_rule": v2_config,
        "gain_probability_definition": (
            "Phi(-mu/sigma), because mu predicts log(TTFT_racer/TTFT_full)"
        ),
        "fixed20k_all197": metrics(rows, fixed),
        "bcrs_v1_all197": metrics(rows, v1),
        "bcrs_v2_all197": metrics(rows, v2),
        "fixed20k_heldout": metrics(
            heldout_rows, [fixed[index] for index in heldout_indices]
        ),
        "bcrs_v1_heldout": metrics(
            heldout_rows, [v1[index] for index in heldout_indices]
        ),
        "bcrs_v2_heldout": metrics(
            heldout_rows, [v2[index] for index in heldout_indices]
        ),
        "agreement_with_fixed20k": {
            "fixed20k": agreement(fixed, fixed),
            "bcrs_v1": agreement(fixed, v1),
            "bcrs_v2": agreement(fixed, v2),
        },
        "v1_missed_fixed20k": {
            "count": len(missed_indices),
            "reason_counts": dict(sorted(rejection_counts.items())),
            "length_band_counts": dict(sorted(rejection_bands.items())),
        },
        "v2_reason_counts": dict(sorted(Counter(
            decision["reason"] for decision in decisions
        ).items())),
        "accuracy_available": False,
        "accuracy_assessment": (
            "Pending an online closed-loop run. Per-task accuracy from separately "
            "executed Full/Racer trajectories cannot be post-hoc composed into a "
            "dynamic per-turn BCRS trajectory."
        ),
        "ttft_target_50pct_exploratory": (
            metrics(rows, v2)["total_reduction"] >= 0.50
        ),
        "accuracy_drop_target_15pct": None,
        "warnings": [
            "This is an offline replay over one longest decision per task, not a closed-loop result.",
            "Observed Full/C2KV TTFT labels are used only to score frozen route choices, never as route inputs.",
            "Historical semantic features are unavailable and therefore zero in this replay.",
            "Tool definitions remain Full/native protected; only existing 8x history C2KV fixed-256 is selected.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "stage_a_v2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    v2_artifact = dict(artifact)
    v2_artifact["routing_rule"] = rule
    v2_artifact["variant"] = "prior_guided_bcrs_v2"
    v2_artifact["parent_artifact_sha256"] = report["v1_artifact_sha256"]
    (args.out_dir / "bcrs_v2_artifact.json").write_text(
        json.dumps(v2_artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "bcrs_v2_config.json").write_text(
        json.dumps({"routing_rule": rule}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "comparison.csv", comparison_rows)
    write_csv(args.out_dir / "routing_decisions_v2.csv", decision_rows)
    write_csv(args.out_dir / "v1_missed_fixed20k.csv", missed_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
