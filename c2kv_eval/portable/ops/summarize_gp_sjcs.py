#!/usr/bin/env python3
"""Fail-closed BFCL Long GP-SJCS accuracy and Online-TTFT summary."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = (
    ("full", "Full Recompute"),
    ("fixed20k_joint", "Fixed 20K Router"),
    ("gp_length_only", "GP-Length-only"),
    ("gp_sjcs", "GP-SJCS"),
)
BUCKETS = (
    ("<8K", 0, 8192),
    ("8K-16K", 8192, 16384),
    ("16K-32K", 16384, 32768),
    ("32K-64K", 32768, 65536),
    ("64K+", 65536, 10**12),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def load_summary(root: Path, arm: str) -> dict[str, Any]:
    hits = list((root / "runs" / arm).glob(f"summary_{arm}.json"))
    if len(hits) != 1:
        raise ValueError(f"{arm}: expected one summary, found {hits}")
    return read_json(hits[0])


def score_rows(run_root: Path) -> dict[str, bool]:
    hits = list((run_root / "score").rglob(
        "BFCL_v4_multi_turn_long_context_score.json"))
    if len(hits) != 1:
        raise ValueError(f"{run_root.name}: expected one official score file, found {hits}")
    lines = read_jsonl(hits[0])
    if len(lines) < 2 or "accuracy" not in lines[0]:
        raise ValueError(f"{hits[0]}: missing official header/task rows")
    requested_path = run_root / "test_case_ids_to_generate.json"
    requested_payload = read_json(requested_path)
    requested = requested_payload.get("multi_turn_long_context")
    if not isinstance(requested, list) or any(not isinstance(x, str) for x in requested):
        raise ValueError(f"{requested_path}: invalid fixed task-ID list")
    if len(requested) != len(set(requested)):
        raise ValueError(f"{requested_path}: duplicate task IDs")
    # BFCL writes one header plus failure details; successful tasks may have no row.
    result = {task_id: True for task_id in requested}
    for row in lines[1:]:
        task_id = row.get("id")
        valid = row.get("valid")
        if not isinstance(task_id, str) or not isinstance(valid, bool):
            raise ValueError(f"{hits[0]}: invalid task score row")
        if task_id not in result:
            raise ValueError(f"{hits[0]}: score row outside fixed task-ID list: {task_id}")
        result[task_id] = valid
    header = lines[0]
    total = int(header["total_count"]) if "total_count" in header else -1
    correct = int(header["correct_count"]) if "correct_count" in header else -1
    if total != len(result) or correct != sum(result.values()):
        raise ValueError(
            f"{hits[0]}: header/task consistency failure: "
            f"header={correct}/{total}, reconstructed={sum(result.values())}/{len(result)}")
    return result


def request_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(summary.get("request_log") or ""))
    if not path.is_file():
        raise ValueError(f"missing request log: {path}")
    rows = read_jsonl(path)
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        raise ValueError(
            f"{summary.get('arm')}: {len(errors)} request errors: "
            f"{Counter(str(row.get('error_kind')) for row in errors)}")
    for row in rows:
        if not isinstance(row.get("server_ttft_ms"), (int, float)):
            raise ValueError(f"{summary.get('arm')}: missing server TTFT")
    return rows


def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    context = row.get("eval_context") or {}
    return (
        context.get("task_id"),
        context.get("user_turn"),
        context.get("step"),
        context.get("attempt"),
    )


def online_ttft(row: dict[str, Any]) -> float:
    return float(row["server_ttft_ms"]) + 1000.0 * float(row.get("assemble_sec") or 0.0)


def metric_row(
    arm: str,
    label: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    scores: dict[str, bool],
    full_rows: dict[tuple[Any, ...], dict[str, Any]],
    full_correct: int,
    calibration: set[str],
) -> dict[str, Any]:
    values = [online_ttft(row) for row in rows]
    pairs = [(full_rows[pair_key(row)], row)
             for row in rows if pair_key(row) in full_rows]
    paired_full = [online_ttft(full) for full, _ in pairs]
    paired_method = [online_ttft(method) for _, method in pairs]
    correct = sum(scores.values())
    heldout = {key: value for key, value in scores.items() if key not in calibration}
    rel_drop = ((full_correct - correct) / full_correct) if full_correct else None
    rl = summary.get("request_log_summary") or {}
    if arm != "full":
        if int(rl.get("gp_scheduler_request_count") or 0) != len(rows):
            raise ValueError(f"{arm}: scheduler telemetry missing on requests")
        if any(not isinstance(row.get("gp_sjcs"), dict) for row in rows):
            raise ValueError(f"{arm}: missing per-request GP-SJCS decision")
    return {
        "arm": arm,
        "method": label,
        "tasks": len(scores),
        "correct": correct,
        "accuracy": correct / len(scores) if scores else None,
        "relative_accuracy_drop": rel_drop,
        "heldout_tasks": len(heldout),
        "heldout_correct": sum(heldout.values()),
        "heldout_accuracy": sum(heldout.values()) / len(heldout) if heldout else None,
        "decisions": len(values),
        "total_online_ttft_ms": sum(values),
        "mean_online_ttft_ms": sum(values) / len(values),
        "p50_online_ttft_ms": pct(values, 0.50),
        "p95_online_ttft_ms": pct(values, 0.95),
        "system_workload_total_reduction_vs_full": None,
        "paired_decisions": len(pairs),
        "paired_total_online_ttft_ms": sum(paired_method),
        "paired_full_total_online_ttft_ms": sum(paired_full),
        "paired_total_reduction_vs_full": (
            1.0 - sum(paired_method) / sum(paired_full)
            if paired_full and sum(paired_full) > 0 else None
        ),
        "controller_preparation_ms_mean": rl.get("controller_preparation_ms_mean"),
        "controller_preparation_ms_p95": rl.get("controller_preparation_ms_p95"),
        "selected_config_counts": rl.get("gp_selected_config_counts") or {},
        "tool_retention": rl.get("tool_retention"),
        "history_retention": rl.get("history_retention"),
        "e2e_kv_retention": rl.get("e2e_kv_retention"),
        "e2e_kv_compression_ratio": rl.get("e2e_kv_compression_ratio"),
        "offline_extraction_applicable": False,
        "materialization": (
            "full_raw_recompute" if arm == "full"
            else "persistent_physical_raw_kv_eviction"
        ),
    }


def bucket_rows(
    full_requests: dict[tuple[Any, ...], dict[str, Any]],
    methods: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for arm, rows in methods.items():
        by_key = {pair_key(row): row for row in rows}
        for label, lower, upper in BUCKETS:
            selected = []
            for key, full in full_requests.items():
                prompt = ((full.get("usage") or {}).get("prompt_tokens"))
                if not isinstance(prompt, int) or not lower <= prompt < upper:
                    continue
                method = by_key.get(key)
                if method is not None:
                    selected.append((full, method))
            full_values = [online_ttft(full) for full, _ in selected]
            method_values = [online_ttft(method) for _, method in selected]
            output.append({
                "arm": arm,
                "bucket": label,
                "paired_decisions": len(selected),
                "full_total_online_ttft_ms": sum(full_values),
                "method_total_online_ttft_ms": sum(method_values),
                "total_reduction_vs_full": (
                    1.0 - sum(method_values) / sum(full_values)
                    if full_values and sum(full_values) > 0 else None
                ),
                "full_p50_ms": pct(full_values, 0.50),
                "method_p50_ms": pct(method_values, 0.50),
                "method_p95_ms": pct(method_values, 0.95),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, scale: float = 1.0, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * scale:.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    artifact = read_json(args.artifact)
    calibration = set(map(str, artifact.get("calibration_task_ids") or []))
    summaries = {arm: load_summary(args.root, arm) for arm, _ in ARMS}
    requests = {arm: request_rows(summaries[arm]) for arm, _ in ARMS}
    scores = {arm: score_rows(args.root / "runs" / arm) for arm, _ in ARMS}
    for arm, _ in ARMS:
        total = int(summaries[arm].get("official_total_count") or 0)
        if total != args.expected or len(scores[arm]) != args.expected:
            raise ValueError(
                f"{arm}: expected {args.expected} official tasks, "
                f"summary={total}, per-task={len(scores[arm])}")

    full_request_map = {pair_key(row): row for row in requests["full"]}
    full_correct = sum(scores["full"].values())
    rows = [
        metric_row(
            arm, label, summaries[arm], requests[arm], scores[arm],
            full_request_map, full_correct, calibration,
        )
        for arm, label in ARMS
    ]
    full_total = rows[0]["total_online_ttft_ms"]
    full_heldout_correct = rows[0]["heldout_correct"]
    minimum_correct = math.ceil(0.85 * full_correct)
    for row in rows:
        row["minimum_correct_for_accuracy_pass"] = minimum_correct
        row["heldout_relative_accuracy_drop"] = (
            (full_heldout_correct - row["heldout_correct"]) / full_heldout_correct
            if full_heldout_correct else None
        )
        row["system_workload_total_reduction_vs_full"] = (
            1.0 - row["total_online_ttft_ms"] / full_total
            if full_total > 0 else None
        )
        row["accuracy_pass"] = (
            row["arm"] == "full"
            or (row["relative_accuracy_drop"] is not None
                and row["relative_accuracy_drop"] <= 0.15)
        )
        row["ttft_pass"] = (
            row["arm"] == "full"
            or (row["paired_total_reduction_vs_full"] is not None
                and row["paired_total_reduction_vs_full"] >= 0.50)
        )
        row["joint_pass"] = bool(row["accuracy_pass"] and row["ttft_pass"])

    buckets = bucket_rows(
        full_request_map,
        {arm: requests[arm] for arm, _ in ARMS if arm != "full"},
    )
    out = args.root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "table1_accuracy.csv", rows)
    write_csv(out / "table2_online_ttft.csv", rows)
    write_csv(out / "length_buckets.csv", buckets)
    (out / "gp_sjcs_summary.json").write_text(json.dumps({
        "schema": "bfcl-long-gp-sjcs-summary-v1",
        "artifact": str(args.artifact),
        "calibration_task_ids": sorted(calibration),
        "rows": rows,
        "length_buckets": buckets,
        "caveats": [
            "Tool Definition C2KV gist compression is unavailable in this checkpoint/runtime.",
            "A1-A4 use persistent physical raw-KV eviction and report actual resident KV.",
            "Online TTFT is server first-token latency plus measured proxy preparation/assembly.",
            "Paired total reduction, not closed-loop workload total, is the acceptance latency metric.",
            "Offline gist extraction is not applicable to the physical-KV materialization.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# BFCL Long GP-SJCS",
        "",
        "Tool Definition C2KV gist compression is unavailable; A1-A4 use persistent physical raw-KV eviction. The acceptance latency uses same-key paired requests.",
        "",
        "## Closed-loop Accuracy",
        "",
        "| Method | Tasks | Correct | Accuracy | Relative Drop | Held-out Accuracy | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['tasks']} | {row['correct']} | "
            f"{fmt(row['accuracy'], 100, '%')} | "
            f"{'Baseline' if row['arm'] == 'full' else fmt(row['relative_accuracy_drop'], 100, '%')} | "
            f"{fmt(row['heldout_accuracy'], 100, '%')} | {row['accuracy_pass']} |"
        )
    lines += [
        "",
        "## Online TTFT",
        "",
        "| Method | Decisions | Total (ms) | Mean | P50 | P95 | Paired Reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['decisions']} | "
            f"{fmt(row['total_online_ttft_ms'])} | {fmt(row['mean_online_ttft_ms'])} | "
            f"{fmt(row['p50_online_ttft_ms'])} | {fmt(row['p95_online_ttft_ms'])} | "
            f"{'Baseline' if row['arm'] == 'full' else fmt(row['paired_total_reduction_vs_full'], 100, '%')} |"
        )
    gp = next(row for row in rows if row["arm"] == "gp_sjcs")
    lines += [
        "",
        "## Joint Acceptance",
        "",
        "| Metric | Target | GP-SJCS Result | Status |",
        "|---|---:|---:|---:|",
        f"| Paired Total Online TTFT Reduction | >=50% | {fmt(gp['paired_total_reduction_vs_full'], 100, '%')} | {gp['ttft_pass']} |",
        f"| Relative Accuracy Drop | <=15% | {fmt(gp['relative_accuracy_drop'], 100, '%')} | {gp['accuracy_pass']} |",
        f"| Joint Acceptance | Both PASS |  | {gp['joint_pass']} |",
        "",
        f"Configuration counts: {json.dumps(gp['selected_config_counts'], sort_keys=True)}",
        "",
        f"GP controller mean/P95 overhead: {fmt(gp['controller_preparation_ms_mean'])} / {fmt(gp['controller_preparation_ms_p95'])} ms.",
    ]
    (out / "gp_sjcs_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "length_buckets": buckets}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
