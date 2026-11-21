#!/usr/bin/env python3
"""Summarize the two targeted BCRO precommit-guard runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _native_ttft_ms(row: dict) -> float:
    for trace in row.get("generation_trace") or []:
        if trace.get("phase") != "draft":
            continue
        stats = ((trace.get("generation") or {}).get("stats") or {})
        speed = stats.get("speed_timing") or {}
        value = speed.get("native_ttft_ms", speed.get("server_ttft_ms"))
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"missing draft TTFT for {row.get('decision_key')}")


def summarize(case_root: Path, case: str) -> dict:
    unified = json.loads(
        (case_root / "runs/bcro/unified_summary.json").read_text(encoding="utf-8")
    )
    official_rows = unified.get("rows") or []
    steps = []
    for path in sorted((case_root / "runs/bcro/task_shards").glob("*/server/steps.jsonl")):
        with path.open(encoding="utf-8") as handle:
            steps.extend(json.loads(line) for line in handle if line.strip())
    failed = [row for row in steps if row.get("status") != "ok"]
    if failed:
        raise ValueError(f"{case}: {len(failed)} failed decisions")
    online_ms = 0.0
    final_ms = 0.0
    for row in steps:
        timing = row.get("controller_timing") or {}
        route = float(((row.get("bcrs") or {}).get("measured_route_seconds") or 0.0))
        prepare = float(timing.get("prepare_seconds") or 0.0)
        online_ms += 1000.0 * (route + prepare) + _native_ttft_ms(row)
        final_ms += 1000.0 * float(
            row.get("final_action_ready_seconds", row.get("decision_runtime_seconds") or 0.0)
        )
    triggers = [row for row in steps if (row.get("precommit_guard") or {}).get("triggered")]
    regenerations = sum(
        int((row.get("precommit_guard") or {}).get("regeneration_count") or 0)
        for row in steps
    )
    return {
        "case": case,
        "tasks": len(official_rows),
        "official_correct": sum(float(row.get("official_score") or 0.0) == 1.0 for row in official_rows),
        "official_total": len(official_rows),
        "official_accuracy": (
            sum(float(row.get("official_score") or 0.0) == 1.0 for row in official_rows)
            / len(official_rows) if official_rows else None
        ),
        "guard_trigger_count": len(triggers),
        "full_regeneration_count": regenerations,
        "decision_count": len(steps),
        "cumulative_online_ttft_ms": online_ms,
        "cumulative_final_action_ready_ms": final_ms,
        "result_root": str(case_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        summarize(args.root / "exact_identifier", "exact_identifier"),
        summarize(args.root / "readonly_loop", "readonly_loop"),
    ]
    out = args.root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "guard_smoke_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out / "guard_smoke_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
