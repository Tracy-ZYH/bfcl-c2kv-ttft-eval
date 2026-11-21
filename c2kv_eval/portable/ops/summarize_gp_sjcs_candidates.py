#!/usr/bin/env python3
"""Fail-closed summary for fixed GP-SJCS A0-A4 profiling arms."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from c2kv_eval.portable.ops.summarize_gp_sjcs import (
    online_ttft, pair_key, pct, read_json, request_rows, score_rows,
)

ARMS = tuple((f"gp_profile_a{i}", f"A{i}") for i in range(5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    collected = {}
    for arm, config in ARMS:
        run_root = args.root / "runs" / arm
        summary_path = run_root / f"summary_{arm}.json"
        summary = read_json(summary_path)
        scores = score_rows(run_root)
        rows = request_rows(summary)
        total = int(summary.get("official_total_count") or 0)
        correct = int(summary.get("official_correct_count") or 0)
        if total != args.expected or len(scores) != args.expected:
            raise ValueError(
                f"{arm}: expected {args.expected} tasks, summary={total}, "
                f"reconstructed={len(scores)}")
        if correct != sum(scores.values()):
            raise ValueError(f"{arm}: official score reconstruction mismatch")
        missing = [row for row in rows if not isinstance(row.get("gp_sjcs"), dict)]
        selected = {
            str(row["gp_sjcs"].get("selected_config")) for row in rows
            if isinstance(row.get("gp_sjcs"), dict)
        }
        if missing or selected != {config}:
            raise ValueError(
                f"{arm}: profiling telemetry mismatch: missing={len(missing)}, "
                f"selected={sorted(selected)}, expected={config}")
        collected[arm] = (summary, scores, rows)

    control = {pair_key(row): row for row in collected["gp_profile_a0"][2]}
    output = []
    for arm, config in ARMS:
        summary, scores, rows = collected[arm]
        values = [online_ttft(row) for row in rows]
        pairs = [(control[pair_key(row)], row) for row in rows if pair_key(row) in control]
        base = [online_ttft(a) for a, _ in pairs]
        method = [online_ttft(b) for _, b in pairs]
        rl = summary.get("request_log_summary") or {}
        output.append({
            "arm": arm,
            "config": config,
            "tasks": len(scores),
            "correct": sum(scores.values()),
            "accuracy": sum(scores.values()) / len(scores),
            "decisions": len(values),
            "total_online_ttft_ms": sum(values),
            "mean_online_ttft_ms": sum(values) / len(values),
            "p50_online_ttft_ms": pct(values, 0.50),
            "p95_online_ttft_ms": pct(values, 0.95),
            "paired_decisions_vs_a0": len(pairs),
            "paired_total_reduction_vs_a0": (
                1.0 - sum(method) / sum(base) if base and sum(base) > 0 else None
            ),
            "tool_retention": rl.get("tool_retention"),
            "history_retention": rl.get("history_retention"),
            "e2e_kv_retention": rl.get("e2e_kv_retention"),
            "e2e_kv_compression_ratio": rl.get("e2e_kv_compression_ratio"),
        })

    minimum_correct = math.ceil(0.85 * output[0]["correct"])
    for row in output:
        row["minimum_correct_for_quality_pass"] = minimum_correct
        row["compression_pass"] = (
            row["config"] == "A0"
            or (isinstance(row["e2e_kv_retention"], (int, float))
                and row["e2e_kv_retention"] < 0.999)
        )
        row["quality_pass"] = row["correct"] >= minimum_correct
        row["smoke_pass"] = bool(row["compression_pass"] and row["quality_pass"])
        row["materialization"] = "persistent_physical_raw_kv_eviction"

    overall_pass = all(row["smoke_pass"] for row in output)
    out = args.root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (out / "candidate_profiles.json").write_text(
        json.dumps({
            "schema": "gp-sjcs-candidate-profiles-v1",
            "overall_smoke_pass": overall_pass,
            "rows": output,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"overall_smoke_pass": overall_pass, "rows": output},
                     indent=2, ensure_ascii=False))
    if not overall_pass:
        failed = [row["config"] for row in output if not row["smoke_pass"]]
        raise SystemExit(f"candidate smoke failed quality/compression gate: {failed}")


if __name__ == "__main__":
    main()
