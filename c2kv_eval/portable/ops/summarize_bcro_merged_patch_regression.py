#!/usr/bin/env python3
"""Summarize the three-task merged BCRO patch regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "multi_turn_long_context_151",
    "multi_turn_long_context_191",
    "multi_turn_long_context_193",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def task_id(row: dict) -> str:
    context = row.get("c2kv_eval_context") or {}
    if context.get("task_id"):
        return str(context["task_id"])
    parts = str(row.get("session_id") or "").split("/")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    raise ValueError(f"cannot resolve task ID for {row.get('decision_key')!r}")


def draft_ttft_ms(row: dict) -> float:
    for trace in row.get("generation_trace") or []:
        if trace.get("phase") != "draft":
            continue
        speed = ((trace.get("generation") or {}).get("stats") or {}).get(
            "speed_timing"
        ) or {}
        value = speed.get("native_ttft_ms", speed.get("server_ttft_ms"))
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"missing draft TTFT for {row.get('decision_key')}")


def canonical_action(response: dict) -> dict:
    return {
        "content": response.get("content"),
        "tool_calls": response.get("tool_calls") or [],
    }


def actual_regeneration_traces(row: dict) -> list[dict]:
    """Return model regeneration attempts, not duplicated guard receipts."""
    return [
        trace
        for trace in (row.get("generation_trace") or [])
        if trace.get("phase") == "regeneration"
    ]


def attempt_request_ids(path: Path) -> dict[str, str]:
    mapping = {}
    if not path.exists():
        return mapping
    for row in read_jsonl(path):
        uid = row.get("attempt_uid")
        request_id = row.get("request_id")
        if uid and request_id is not None:
            mapping[str(uid)] = request_id
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run = args.root / "runs/bcro"
    unified = json.loads((run / "unified_summary.json").read_text(encoding="utf-8"))
    official = {str(row["task_id"]): row for row in unified.get("rows") or []}
    if set(official) != EXPECTED:
        raise ValueError(f"official task set mismatch: {sorted(official)}")

    steps = []
    for path in sorted((run / "task_shards").glob("*/server/steps.jsonl")):
        steps.extend(read_jsonl(path))
    if {task_id(row) for row in steps} != EXPECTED:
        raise ValueError("step task set does not match the frozen three-task regression")
    if any(row.get("status") != "ok" for row in steps):
        raise ValueError("one or more decisions did not finish with status=ok")

    rows = []
    total_online = 0.0
    total_final = 0.0
    for task in sorted(EXPECTED):
        selected = [row for row in steps if task_id(row) == task]
        literal = [
            row for row in selected
            if any(
                check.get("guard") == "post_tweet_quoted_content"
                and check.get("triggered")
                for check in (row.get("precommit_guard") or {}).get("checks") or []
            )
        ]
        diversion = [
            row for row in selected
            if (row.get("recovery_diversion") or {}).get("triggered")
        ]
        actual_regenerations = [
            trace
            for row in selected
            for trace in actual_regeneration_traces(row)
        ]
        request_ids = attempt_request_ids(
            run / "task_shards" / task / "server/attempts.jsonl"
        )
        # Guard receipts can be duplicated when the cancel diversion mirrors the
        # precommit receipt. The generation trace is the source of truth.
        regenerations = len(actual_regenerations)
        if any(len(actual_regeneration_traces(row)) > 1 for row in selected):
            raise ValueError(f"{task}: more than one actual regeneration in a decision")
        for row in diversion:
            receipt = row["recovery_diversion"]
            if receipt.get("racer_recovery_count") != 0:
                raise ValueError(f"{task}: diversion stacked with Racer recovery")
            phases = [trace.get("phase") for trace in row.get("generation_trace") or []]
            if phases != ["draft", "regeneration"]:
                raise ValueError(f"{task}: unexpected diversion phases {phases}")
        online = sum(
            1000.0 * (
                float(((row.get("bcrs") or {}).get("measured_route_seconds") or 0.0))
                + float(((row.get("controller_timing") or {}).get("prepare_seconds") or 0.0))
            ) + draft_ttft_ms(row)
            for row in selected
        )
        final = sum(
            1000.0 * float(
                row.get("final_action_ready_seconds", row.get("decision_runtime_seconds") or 0.0)
            )
            for row in selected
        )
        score = float(official[task].get("official_score") or 0.0)
        rows.append({
            "task_id": task,
            "official": "PASS" if score == 1.0 else "FAIL",
            "official_score": score,
            "post_tweet_literal_trigger_count": len(literal),
            "cancel_booking_diversion_trigger_count": len(diversion),
            "regeneration_count": regenerations,
            "actual_regeneration_attempt_uids": [
                trace.get("attempt_uid") for trace in actual_regenerations
            ],
            "actual_regeneration_request_ids": [
                request_ids.get(trace.get("attempt_uid"))
                for trace in actual_regenerations
            ],
            "generation_attempt_count": sum(
                int(row.get("generation_attempts") or 0) for row in selected
            ),
            "cumulative_online_ttft_ms": online,
            "cumulative_final_action_ready_ms": final,
            "final_action": canonical_action(selected[-1].get("response") or {}),
        })
        total_online += online
        total_final += final

    result = {
        "schema": "bcro-merged-patch-regression-v1",
        "official_pass": sum(row["official"] == "PASS" for row in rows),
        "official_total": len(rows),
        "cumulative_online_ttft_ms": total_online,
        "cumulative_final_action_ready_ms": total_final,
        "tasks": rows,
    }
    out = args.root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "merged_patch_regression.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
