"""Summarize official accuracy and latency from one NPU Full/BCRO-120 run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from .frozen import load_and_validate


TIMING_FIELDS = (
    "draft_prefill_path_ms",
    "draft_server_ttft_ms",
    "draft_native_ttft_ms",
    "system_draft_ttft_ms",
    "final_action_ready_ms",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing telemetry: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing series")
    point = (len(ordered) - 1) * q
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    weight = point - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _describe(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timing values must be finite, non-negative and non-empty")
    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing/invalid {name}: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"missing/invalid {name}: {value!r}")
    return result


def _draft(record: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in record.get("generation_trace") or []
        if row.get("phase") == "draft" and row.get("status") == "completed"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one completed draft, found {len(rows)}")
    return rows[0]


def _step_metrics(record: dict[str, Any], method: str) -> dict[str, float]:
    trace = _draft(record)
    stats = ((trace.get("generation") or {}).get("stats") or {})
    speed = stats.get("speed_timing") or {}
    prefill = _number(speed.get("prefill_path_ms"), "prefill_path_ms")
    server = _number(speed.get("server_ttft_ms"), "server_ttft_ms")
    native = _number(speed.get("native_ttft_ms"), "native_ttft_ms")
    timing = record.get("controller_timing") or {}
    route = _number(timing.get("bcrs_route_seconds") or 0.0, "bcrs_route_seconds") * 1000
    prepare = _number(timing.get("prepare_seconds") or 0.0, "prepare_seconds") * 1000
    prewarm = _number(timing.get("legal_cache_prewarm_seconds") or 0.0, "legal_cache_prewarm_seconds") * 1000
    if method == "Full":
        if record.get("ordinary_full_path") is None:
            raise ValueError("Full decision did not use ordinary Full path")
        system = prepare + server
    else:
        if not isinstance(record.get("bcrs"), dict):
            raise ValueError("BCRO decision lacks a frozen route receipt")
        system = route + prepare + prewarm + native
    final = _number(
        record.get("final_action_ready_seconds", record.get("decision_runtime_seconds")),
        "final_action_ready_seconds",
    ) * 1000
    return {
        "draft_prefill_path_ms": prefill,
        "draft_server_ttft_ms": server,
        "draft_native_ttft_ms": native,
        "system_draft_ttft_ms": system,
        "final_action_ready_ms": final,
    }


def summarize_method(root: Path, method: str, task_ids: Iterable[str]) -> dict[str, Any]:
    task_ids = tuple(task_ids)
    unified_path = root / "unified_summary.json"
    payload = json.loads(unified_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    observed = [row.get("task_id") for row in rows]
    if observed != list(task_ids):
        raise ValueError(f"{method}: task IDs/order differ from the requested manifest")
    if any(row.get("status") != "completed" for row in rows):
        raise ValueError(f"{method}: one or more tasks did not complete")

    series = {name: [] for name in TIMING_FIELDS}
    draft_calls = regeneration_calls = decisions = 0
    route_full = route_racer = 0
    cache_hits = cache_misses = online_extraction_calls = 0
    strict_receipts = strict_passed = 0
    for task_id in task_ids:
        steps = _jsonl(root / "task_shards" / task_id / "server" / "steps.jsonl")
        if not steps:
            raise ValueError(f"{method}: no decisions for {task_id}")
        for record in steps:
            decisions += 1
            metrics = _step_metrics(record, method)
            for name, value in metrics.items():
                series[name].append(value)
            traces = record.get("generation_trace") or []
            draft_calls += sum(row.get("phase") == "draft" and row.get("status") == "completed" for row in traces)
            regeneration_calls += sum(row.get("phase") == "regeneration" and row.get("status") == "completed" for row in traces)
            for trace in traces:
                stats = ((trace.get("generation") or {}).get("stats") or {})
                extraction = stats.get("sglang_extraction") or {}
                cache_hits += int(extraction.get("cache_hits") or 0)
                cache_misses += int(extraction.get("cache_misses") or 0)
                online_extraction_calls += int(extraction.get("model_calls") or 0)
            strict = record.get("strict_cache_hit") or []
            strict_receipts += len(strict)
            strict_passed += sum(item.get("passed") is True for item in strict)
            if method == "BCRO":
                selected = (record.get("bcrs") or {}).get("selected_arm")
                if selected == "full":
                    route_full += 1
                elif selected == "c2kv_racer":
                    route_racer += 1
                else:
                    raise ValueError(f"BCRO selected unknown arm: {selected!r}")

    if method == "BCRO" and route_racer:
        if cache_misses or online_extraction_calls or strict_receipts == 0 or strict_receipts != strict_passed:
            raise ValueError(
                "measured BCRO is not cache-hit-only: "
                f"misses={cache_misses}, extraction_calls={online_extraction_calls}, "
                f"strict={strict_passed}/{strict_receipts}"
            )
    correct = sum(row.get("official_score") == 1 for row in rows)
    return {
        "method": method,
        "official_correct_count": correct,
        "official_total_count": len(rows),
        "accuracy": correct / len(rows),
        "timing": {name: _describe(values) for name, values in series.items()},
        "validation": {
            "decisions": decisions,
            "draft_calls": draft_calls,
            "regeneration_calls": regeneration_calls,
            "all_model_calls": draft_calls + regeneration_calls,
            "route_full": route_full,
            "route_racer": route_racer,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "online_extraction_calls": online_extraction_calls,
            "strict_cache_hit_receipts": strict_receipts,
            "strict_cache_hit_passed": strict_passed,
        },
        "source": str(root.resolve()),
    }


def build_summary(full: dict[str, Any], bcro: dict[str, Any]) -> dict[str, Any]:
    if full["official_total_count"] != bcro["official_total_count"]:
        raise ValueError("Full and BCRO task counts differ")
    if full["official_correct_count"] == 0:
        raise ValueError("Full correct count is zero; relative degradation is undefined")
    relative_drop = (
        full["official_correct_count"] - bcro["official_correct_count"]
    ) / full["official_correct_count"]
    reductions: dict[str, dict[str, float]] = {}
    for name in TIMING_FIELDS:
        reductions[name] = {}
        for statistic in ("total_ms", "mean_ms", "p50_ms", "p95_ms"):
            denominator = full["timing"][name][statistic]
            reductions[name][statistic.replace("_ms", "_reduction_vs_full")] = (
                1.0 - bcro["timing"][name][statistic] / denominator
                if denominator else 0.0
            )
    return {
        "schema": "bcro-120-npu-summary-v1",
        "methods": [full, bcro],
        "comparison": {
            "relative_accuracy_drop": relative_drop,
            "timing_reductions": reductions,
        },
    }


def write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = [
        "method", "official_correct_count", "official_total_count", "accuracy",
        "relative_accuracy_drop_vs_full",
    ]
    for name in TIMING_FIELDS:
        for statistic in ("total_ms", "mean_ms", "p50_ms", "p95_ms"):
            fields.append(f"{name}_{statistic}")
            fields.append(f"{name}_{statistic.replace('_ms', '_reduction_vs_full')}")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in summary["methods"]:
            is_full = method["method"] == "Full"
            row = {
                "method": method["method"],
                "official_correct_count": method["official_correct_count"],
                "official_total_count": method["official_total_count"],
                "accuracy": method["accuracy"],
                "relative_accuracy_drop_vs_full": (
                    None if is_full else summary["comparison"]["relative_accuracy_drop"]
                ),
            }
            for name in TIMING_FIELDS:
                for statistic in ("total_ms", "mean_ms", "p50_ms", "p95_ms"):
                    row[f"{name}_{statistic}"] = method["timing"][name][statistic]
                    reduction_key = statistic.replace("_ms", "_reduction_vs_full")
                    row[f"{name}_{reduction_key}"] = (
                        None if is_full else
                        summary["comparison"]["timing_reductions"][name][reduction_key]
                    )
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--bcro-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-tasks", type=int, default=120)
    args = parser.parse_args(argv)
    frozen = load_and_validate(
        config_path=args.config,
        require_external_models=False,
        require_bfcl_data=True,
    )
    if not 1 <= args.max_tasks <= 120:
        parser.error("--max-tasks must be in [1, 120]")
    task_ids = frozen.task_ids[: args.max_tasks]
    full = summarize_method(args.full_root, "Full", task_ids)
    bcro = summarize_method(args.bcro_root, "BCRO", task_ids)
    summary = build_summary(full, bcro)
    write_outputs(summary, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
