#!/usr/bin/env python3
"""Summarize a targeted BCRO cancel-booking Full-diversion run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _draft_ttft_ms(row: dict) -> float:
    for trace in row.get("generation_trace") or []:
        if trace.get("phase") != "draft":
            continue
        stats = ((trace.get("generation") or {}).get("stats") or {})
        speed = stats.get("speed_timing") or {}
        value = speed.get("native_ttft_ms", speed.get("server_ttft_ms"))
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"missing draft TTFT for {row.get('decision_key')}")


def _task_id(row: dict) -> str:
    session = str(row.get("session_id") or "")
    parts = session.split("/")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"invalid session_id: {session!r}")
    return parts[1]


def summarize(root: Path, expected: set[str]) -> dict:
    run = root / "runs/bcro"
    unified = json.loads(
        (run / "unified_summary.json").read_text(encoding="utf-8")
    )
    official_rows = unified.get("rows") or []
    official_by_task = {str(row["task_id"]): row for row in official_rows}
    if set(official_by_task) != expected:
        raise ValueError(
            f"official task set mismatch: expected={sorted(expected)}, "
            f"actual={sorted(official_by_task)}"
        )

    steps = []
    for path in sorted((run / "task_shards").glob("*/server/steps.jsonl")):
        with path.open(encoding="utf-8") as handle:
            steps.extend(json.loads(line) for line in handle if line.strip())
    if {_task_id(row) for row in steps} != expected:
        raise ValueError("step task set does not match requested tasks")
    failed = [row for row in steps if row.get("status") != "ok"]
    if failed:
        raise ValueError(f"{len(failed)} failed decisions")

    task_rows = []
    all_online_ms = 0.0
    all_final_ms = 0.0
    all_generations = 0
    all_triggers = 0
    for task in sorted(expected):
        selected = [row for row in steps if _task_id(row) == task]
        triggers = [
            row for row in selected
            if (row.get("recovery_diversion") or {}).get("triggered")
        ]
        for row in triggers:
            receipt = row["recovery_diversion"]
            if int(receipt.get("regeneration_count") or 0) != 1:
                raise ValueError(f"{task}: diversion did not use exactly one retry")
            if int(receipt.get("racer_recovery_count") or 0) != 0:
                raise ValueError(f"{task}: Racer recovery was stacked with Full retry")
            traces = row.get("generation_trace") or []
            if len(traces) != 2 or [trace.get("phase") for trace in traces] != [
                "draft", "regeneration"
            ]:
                raise ValueError(f"{task}: unexpected diversion generation trace")
        online_ms = 0.0
        final_ms = 0.0
        generations = 0
        for row in selected:
            timing = row.get("controller_timing") or {}
            route = float((row.get("bcrs") or {}).get(
                "measured_route_seconds"
            ) or 0.0)
            prepare = float(timing.get("prepare_seconds") or 0.0)
            online_ms += 1000.0 * (route + prepare) + _draft_ttft_ms(row)
            final_ms += 1000.0 * float(
                row.get(
                    "final_action_ready_seconds",
                    row.get("decision_runtime_seconds") or 0.0,
                )
            )
            generations += int(row.get("generation_attempts") or 0)
        score = float(official_by_task[task].get("official_score") or 0.0)
        task_rows.append({
            "task_id": task,
            "official_score": score,
            "official_correct": score == 1.0,
            "trigger_count": len(triggers),
            "decision_count": len(selected),
            "generation_count": generations,
            "cumulative_online_ttft_ms": online_ms,
            "cumulative_final_action_ready_ms": final_ms,
        })
        all_triggers += len(triggers)
        all_generations += generations
        all_online_ms += online_ms
        all_final_ms += final_ms

    result = {
        "schema": "bcro-cancel-booking-diversion-summary-v1",
        "tasks": len(task_rows),
        "official_correct": sum(row["official_correct"] for row in task_rows),
        "official_total": len(task_rows),
        "official_accuracy": (
            sum(row["official_correct"] for row in task_rows) / len(task_rows)
        ),
        "trigger_count": all_triggers,
        "decision_count": len(steps),
        "generation_count": all_generations,
        "cumulative_online_ttft_ms": all_online_ms,
        "cumulative_final_action_ready_ms": all_final_ms,
        "task_rows": task_rows,
        "result_root": str(root.resolve()),
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cancel_booking_diversion_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out / "cancel_booking_diversion_tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-task-id", action="append", required=True)
    args = parser.parse_args()
    expected = set(args.expected_task_id)
    if len(expected) != len(args.expected_task_id):
        raise ValueError("duplicate expected task ID")
    print(json.dumps(summarize(args.root, expected), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
