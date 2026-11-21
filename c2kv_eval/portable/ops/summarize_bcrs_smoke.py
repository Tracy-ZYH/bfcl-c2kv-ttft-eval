#!/usr/bin/env python3
"""Summarize closed-loop BCRS smoke without calling it paired TTFT."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pct(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    alpha = position - lo
    return values[lo] * (1 - alpha) + values[hi] * alpha


def summarize(name, root):
    unified_path = root / "unified_summary.json"
    if not unified_path.is_file():
        raise FileNotFoundError(unified_path)
    unified = json.loads(unified_path.read_text())["rows"]
    if len(unified) != 20 or any(row.get("status") != "completed" for row in unified):
        raise RuntimeError(f"{name}: expected 20 completed tasks, got {len(unified)}")

    first_draft = []
    final_ready = []
    prepare_ms = []
    route_ms = []
    hits = misses = unique = extraction_calls = 0
    route_counts = {"full": 0, "c2kv_racer": 0}
    budgets = []
    recovery_decisions = 0
    generation_calls = 0
    for task in unified:
        server = root / "task_shards" / task["task_id"] / "server"
        step_rows = rows(server / "steps.jsonl")
        if not step_rows:
            raise RuntimeError(f"{name}: no steps for {task['task_id']}")
        for record in step_rows:
            decision = record.get("bcrs") or {}
            selected = decision.get("selected_arm")
            if selected in route_counts:
                route_counts[selected] += 1
            timing = record.get("controller_timing") or {}
            prepare = float(timing.get("prepare_seconds") or 0.0) * 1000.0
            route = float(timing.get("bcrs_route_seconds") or 0.0) * 1000.0
            prepare_ms.append(prepare)
            route_ms.append(route)
            traces = record.get("generation_trace") or []
            draft = next((trace for trace in traces if trace.get("phase") == "draft"), None)
            if not draft:
                raise RuntimeError(f"{name}: missing draft trace in {server}")
            stats = (draft.get("generation") or {}).get("stats") or {}
            speed = stats.get("speed_timing") or {}
            native = speed.get("native_ttft_ms")
            if not isinstance(native, (int, float)) or not math.isfinite(native):
                raise ValueError(f"{name}: missing native_ttft_ms in {server}")
            first_draft.append(float(native) + prepare + route)
            runtime = record.get("decision_runtime_seconds")
            if not isinstance(runtime, (int, float)) or not math.isfinite(runtime):
                raise ValueError(f"{name}: missing decision_runtime_seconds in {server}")
            final_ready.append(float(runtime) * 1000.0)
            generation_calls += len(traces)
            recovery_decisions += int(len(traces) > 1)
            for trace in traces:
                trace_stats = (trace.get("generation") or {}).get("stats") or {}
                extraction = trace_stats.get("sglang_extraction") or {}
                hits += int(extraction.get("cache_hits") or 0)
                misses += int(extraction.get("cache_misses") or 0)
                unique += int(extraction.get("unique_chunks") or 0)
                extraction_calls += int(extraction.get("model_calls") or 0)
            controller = draft.get("controller") or {}
            budget = controller.get("requested_history_budget_tokens")
            if isinstance(budget, int):
                budgets.append(budget)

    correct = sum(int(row.get("official_score") == 1) for row in unified)
    route_total = sum(route_counts.values())
    cache_requests = hits + misses
    cache_hit_rate = hits / cache_requests if cache_requests else None
    if route_counts["c2kv_racer"]:
        cache_status = "passed" if misses == 0 and cache_requests > 0 else "failed"
    else:
        cache_status = "not_exercised"
    return {
        "method": name,
        "tasks": len(unified),
        "correct": correct,
        "accuracy": correct / len(unified),
        "decisions": len(first_draft),
        "total_online_ttft_ms": sum(first_draft),
        "mean_online_ttft_ms": statistics.mean(first_draft),
        "p50_online_ttft_ms": pct(first_draft, 0.50),
        "p95_online_ttft_ms": pct(first_draft, 0.95),
        "total_final_action_ready_ms": sum(final_ready),
        "mean_final_action_ready_ms": statistics.mean(final_ready),
        "p50_final_action_ready_ms": pct(final_ready, 0.50),
        "p95_final_action_ready_ms": pct(final_ready, 0.95),
        "generation_calls": generation_calls,
        "recovery_decisions": recovery_decisions,
        "route_full": route_counts["full"],
        "route_racer": route_counts["c2kv_racer"],
        "compression_decision_ratio": (
            route_counts["c2kv_racer"] / route_total if route_total else 0.0
        ),
        "gp_route_total_ms": sum(route_ms),
        "gp_route_mean_ms": statistics.mean(route_ms),
        "controller_prepare_total_ms": sum(prepare_ms),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_unique_chunks": unique,
        "cache_extraction_model_calls": extraction_calls,
        "cache_hit_rate": cache_hit_rate,
        "cache_hit_status": cache_status,
        "requested_history_budget_values": sorted(set(budgets)),
        "ttft_comparison_scope": "closed_loop_system_workload_not_same_input_paired",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    runs = args.root / "runs"
    result = [summarize(name, runs / name)
              for name in ("full", "fixed20k", "bcrs")]
    baseline = result[0]
    for row in result:
        row["system_total_reduction_vs_full"] = (
            None if row is baseline else
            1.0 - row["total_online_ttft_ms"] / baseline["total_online_ttft_ms"]
        )
        row["final_action_total_reduction_vs_full"] = (
            None if row is baseline else
            1.0 - row["total_final_action_ready_ms"]
            / baseline["total_final_action_ready_ms"]
        )
        row["relative_accuracy_drop_vs_full"] = (
            None if row is baseline or baseline["correct"] == 0 else
            (baseline["correct"] - row["correct"]) / baseline["correct"]
        )
    failures = [
        f"{row['method']}: measured compressed path was not 100% cache-hit"
        for row in result[1:] if row["cache_hit_status"] == "failed"
    ]
    out = args.root / "summary"
    out.mkdir(exist_ok=True)
    payload = {
        "schema": "bcrs-smoke-summary-v2",
        "rows": result,
        "cache_acceptance_failures": failures,
    }
    (out / "bcrs_smoke.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "bcrs_smoke.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
