#!/usr/bin/env python3
"""Summarize BCRO Stage-B Test117 against the immutable 2026-11-09 Full run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    alpha = position - lo
    return ordered[lo] * (1 - alpha) + ordered[hi] * alpha


def describe(values: list[float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_total_ms": sum(values),
        f"{prefix}_mean_ms": statistics.mean(values) if values else None,
        f"{prefix}_p50_ms": pct(values, 0.50),
        f"{prefix}_p95_ms": pct(values, 0.95),
    }


def selected_ids(split_csv: Path) -> list[str]:
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    ids = [row["task_id"] for row in rows if row["split"] == "test"]
    others = {row["task_id"] for row in rows if row["split"] != "test"}
    if len(rows) != 197 or len(ids) != 117 or len(ids) != len(set(ids)):
        raise ValueError("seed=0 split must contain 60/20/117 unique tasks")
    if set(ids) & others:
        raise ValueError("Test117 overlaps Train/Validation")
    return ids


def summarize_full(root: Path, ids: list[str]) -> dict[str, Any]:
    summary_path = root / "summary_full.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("official_correct_count"), summary.get("official_total_count")) != (63, 200):
        raise ValueError("frozen Full baseline identity changed")
    score_paths = list((root / "score").glob("*/multi_turn/BFCL_v4_multi_turn_long_context_score.json"))
    if len(score_paths) != 1:
        raise ValueError(f"expected one Full score file, found {score_paths}")
    score_rows = jsonl(score_paths[0])
    header = score_rows[0]
    failed = {row["id"] for row in score_rows[1:] if row.get("valid") is False}
    if (header.get("correct_count"), header.get("total_count"), len(failed)) != (63, 200, 137):
        raise ValueError("Full official failure inventory is incomplete")
    unknown = set(ids) - {f"multi_turn_long_context_{i}" for i in range(200)}
    if unknown:
        raise ValueError(f"Full baseline lacks Test tasks: {sorted(unknown)}")
    correct = sum(task not in failed for task in ids)

    logs = list((root / "logs").glob("proxy_full_*.jsonl"))
    if len(logs) != 1:
        raise ValueError(f"expected one Full request log, found {logs}")
    requests = [row for row in jsonl(logs[0])
                if (row.get("eval_context") or {}).get("task_id") in set(ids)]
    ok = [row for row in requests if row.get("status") == "ok"]
    errors = [row for row in requests if row.get("status") != "ok"]
    online = [float(row["server_ttft_ms"]) + 1000.0 * float(row.get("assemble_sec") or 0)
              for row in ok if isinstance(row.get("server_ttft_ms"), (int, float))]
    final_ready = [1000.0 * float(row["wall_sec"]) for row in ok
                   if isinstance(row.get("wall_sec"), (int, float))]
    if len(online) != len(ok):
        raise ValueError("frozen Full selected requests have missing TTFT")
    return {
        "method": "Full",
        "tasks": 117,
        "correct": correct,
        "accuracy": correct / 117,
        "decisions": len(ok),
        **describe(online, "online_ttft"),
        **describe(final_ready, "final_action_ready"),
        "route_full": len(ok),
        "route_racer": 0,
        "compression_decision_ratio": 0.0,
        "cache_hits": None,
        "cache_misses": None,
        "cache_hit_rate": None,
        "offline_extraction_calls": 0,
        "detector_calls": 0,
        "detector_trigger_count": 0,
        "recovery_count": 0,
        "successful_recovery_count": 0,
        "request_errors": len(errors),
        "task_errors": 0,
        "source": str(root.resolve()),
        "source_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    }


def summarize_method(name: str, root: Path, ids: list[str]) -> dict[str, Any]:
    unified_path = root / "unified_summary.json"
    payload = json.loads(unified_path.read_text(encoding="utf-8"))
    unified = payload["rows"]
    observed = {row.get("task_id") for row in unified}
    if len(unified) != 117 or observed != set(ids):
        raise ValueError(f"{name}: output task set differs from frozen Test117")
    incomplete = [row for row in unified if row.get("status") != "completed"]
    if incomplete:
        raise RuntimeError(f"{name}: {len(incomplete)} tasks did not complete")

    online: list[float] = []
    final_ready: list[float] = []
    routes = {"full": 0, "c2kv_racer": 0}
    phase_values = {key: [] for key in (
        "bcrs_route_ms", "controller_prepare_ms",
        "native_plan_ms", "cache_hit_lookup_load_ms",
        "cache_miss_extraction_ms", "native_request_assembly_ms",
        "kv_injection_host_ms", "native_pre_generation_ms", "server_ttft_ms",
    )}
    hits = misses = extraction_calls = request_errors = 0
    prewarm_hits = prewarm_misses = prewarm_tokens = 0
    strict_receipts = strict_passed = strict_misses = strict_model_calls = 0
    for task_id in ids:
        server = root / "task_shards" / task_id / "server"
        records = jsonl(server / "steps.jsonl")
        if not records:
            raise RuntimeError(f"{name}: missing decisions for {task_id}")
        for record in records:
            decision = record.get("bcrs") or {}
            selected = decision.get("selected_arm")
            if selected not in routes:
                raise ValueError(f"{name}: invalid route {selected!r}")
            routes[selected] += 1
            timing = record.get("controller_timing") or {}
            prepare_ms = 1000.0 * float(timing.get("prepare_seconds") or 0)
            route_ms = 1000.0 * float(timing.get("bcrs_route_seconds") or 0)
            phase_values["bcrs_route_ms"].append(route_ms)
            phase_values["controller_prepare_ms"].append(prepare_ms)
            prewarm_ms = 1000.0 * float(
                timing.get("legal_cache_prewarm_seconds") or 0
            )
            phase_values.setdefault("legal_cache_prewarm_ms", []).append(prewarm_ms)
            prewarm = record.get("legal_cache_prewarm") or {}
            if selected == "c2kv_racer":
                if prewarm.get("future_observation_used") is not False:
                    raise ValueError(
                        f"{name}: illegal or missing prewarm visibility receipt for {task_id}"
                    )
                prewarm_hits += int(prewarm.get("cache_hits") or 0)
                prewarm_misses += int(prewarm.get("cache_misses") or 0)
                prewarm_tokens += int(prewarm.get("extracted_source_tokens") or 0)
                strict = record.get("strict_cache_hit")
                if not isinstance(strict, list) or not strict:
                    raise ValueError(f"{name}: missing strict cache-hit receipt for {task_id}")
                strict_receipts += len(strict)
                strict_passed += sum(item.get("passed") is True for item in strict)
                strict_misses += sum(int(item.get("cache_misses") or 0) for item in strict)
                strict_model_calls += sum(int(item.get("model_calls") or 0) for item in strict)
            traces = record.get("generation_trace") or []
            draft = next((trace for trace in traces if trace.get("phase") == "draft"), None)
            if not draft:
                raise RuntimeError(f"{name}: missing draft for {task_id}")
            speed = (((draft.get("generation") or {}).get("stats") or {})
                     .get("speed_timing") or {})
            native = speed.get("native_ttft_ms")
            if not isinstance(native, (int, float)) or not math.isfinite(native):
                raise ValueError(f"{name}: missing native TTFT for {task_id}")
            online.append(float(native) + prepare_ms + route_ms)
            for key in phase_values:
                value = speed.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    phase_values[key].append(float(value))
            runtime = record.get("decision_runtime_seconds")
            if not isinstance(runtime, (int, float)) or not math.isfinite(runtime):
                raise ValueError(f"{name}: missing final-action latency for {task_id}")
            final_ready.append(1000.0 * float(runtime))
            for trace in traces:
                generation = trace.get("generation") or {}
                if trace.get("status") not in {None, "completed"}:
                    request_errors += 1
                extraction = ((generation.get("stats") or {}).get("sglang_extraction") or {})
                hits += int(extraction.get("cache_hits") or 0)
                misses += int(extraction.get("cache_misses") or 0)
                extraction_calls += int(extraction.get("model_calls") or 0)
    compressed = routes["c2kv_racer"]
    cache_total = hits + misses
    correct = sum(row.get("official_score") == 1 for row in unified)
    return {
        "method": name,
        "tasks": 117,
        "correct": correct,
        "accuracy": correct / 117,
        "decisions": len(online),
        **describe(online, "online_ttft"),
        **describe(final_ready, "final_action_ready"),
        "route_full": routes["full"],
        "route_racer": compressed,
        "compression_decision_ratio": compressed / len(online) if online else 0.0,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": hits / cache_total if cache_total else (None if compressed == 0 else 0.0),
        "cache_hit_contract_pass": (
            misses == 0 and extraction_calls == 0
            and strict_receipts == strict_passed
            and strict_misses == 0 and strict_model_calls == 0
        ) if compressed else None,
        "official_result_eligible": (
            misses == 0 and extraction_calls == 0
            and strict_receipts == strict_passed
            and strict_misses == 0 and strict_model_calls == 0
        ) if compressed else True,
        "strict_cache_hit_receipts": strict_receipts,
        "strict_cache_hit_passed": strict_passed,
        "strict_cache_misses": strict_misses,
        "strict_online_extraction_calls": strict_model_calls,
        "legal_prewarm_cache_hits": prewarm_hits,
        "legal_prewarm_cache_misses": prewarm_misses,
        "legal_prewarm_extracted_source_tokens": prewarm_tokens,
        "legal_prewarm_future_observation_used": False if compressed else None,
        "draft_online_extraction_calls": extraction_calls,
        "online_timing_boundary": "route + controller prepare + cache-hit native request entry through first token",
        "legal_prewarm_in_online_ttft": False,
        "end_to_end_decision_latency_source": "decision_runtime_seconds includes legal prewarm and recovery",
        "online_timing_phases_ms": {
            key: {"count": len(values), "total": sum(values),
                  "mean": statistics.mean(values) if values else None}
            for key, values in phase_values.items()
        },
        "offline_extraction_calls": extraction_calls,
        "detector_calls": sum(int(row.get("detector_calls") or 0) for row in unified),
        "detector_trigger_count": sum(int(row.get("detector_trigger_count") or 0) for row in unified),
        "recovery_count": sum(int(row.get("recovery_count") or 0) for row in unified),
        "successful_recovery_count": sum(int(row.get("successful_recovery_count") or 0) for row in unified),
        "request_errors": request_errors,
        "task_errors": len(incomplete),
        "source": str(root.resolve()),
        "source_sha256": hashlib.sha256(unified_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    args = parser.parse_args()
    ids = selected_ids(args.split_csv)
    rows = [
        summarize_full(args.full_root, ids),
        summarize_method("Fixed 20K", args.result_root / "runs/fixed20k", ids),
        summarize_method("BCRO", args.result_root / "runs/bcro", ids),
    ]
    full = rows[0]
    required_correct = math.ceil(0.85 * full["correct"])
    for row in rows:
        row["total_ttft_reduction_vs_full"] = (
            None if row is full else 1.0 - row["online_ttft_total_ms"] / full["online_ttft_total_ms"]
        )
        row["relative_accuracy_drop_vs_full"] = (
            None if row is full or full["correct"] == 0 else
            (full["correct"] - row["correct"]) / full["correct"]
        )
        row["accuracy_required_correct"] = required_correct
        row["ttft_pass"] = None if row is full else row["total_ttft_reduction_vs_full"] >= 0.50
        row["accuracy_pass"] = None if row is full else row["correct"] >= required_correct
        row["joint_pass"] = (
            None if row is full else
            row["ttft_pass"] and row["accuracy_pass"] and row["official_result_eligible"]
        )

    out = args.result_root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "bcro-stage-b-test117-summary-v1",
        "split": "stage_a2_seed0_test117",
        "ttft_scope": "closed_loop_system_workload_not_same_input_paired",
        "full_reused_not_rerun": True,
        "rows": rows,
    }
    (out / "bcro_stage_b_test117.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "bcro_stage_b_test117.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
