#!/usr/bin/env python3
"""Recover only Task 149 rep2 for an interrupted decision dual-GP pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bfcl_long_cache_hit_ttft as bfcl
import toolsandbox_cache_hit_ttft as core


TASK_ID = "multi_turn_long_context_149"
FIELD_NAMES = [
    "method",
    "scenario",
    "decision_key",
    "source_decision_id",
    "source_task_id",
    "source_user_turn",
    "source_step",
    "source_attempt",
    "repeat",
    "timed",
    "ttft_ms",
    "native_ttft_ms",
    "context_prepare_ms",
    "full_reference_prompt_tokens",
    "resident_prompt_tokens",
    "full_reference_history_tokens",
    "resident_history_tokens",
    "cache_hits",
    "cache_misses",
    "unique_chunks",
    "cache_hit_rate",
    "reference_action_json",
    "generated_action_json",
    "reference_action_sha256",
    "generated_action_sha256",
    "action_exact_match",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def source_identity(payload: dict[str, Any], scenario: str) -> tuple[str, str]:
    context = payload.get("c2kv_eval_context") or {}
    task = context.get("task_id") or scenario
    decision = (
        f"{task}/turn-{context.get('user_turn')}/step-{context.get('step')}"
        f"/attempt-{context.get('attempt', 0)}"
    )
    return str(task), decision


def samples_from_steps(
    step_path: Path,
    scenario: str,
    payloads: list[dict[str, Any]],
    repeat: int,
    full_by_key: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    records = core.rows(step_path)
    if len(records) != len(payloads):
        raise ValueError(
            f"{step_path}: expected {len(payloads)} records, got {len(records)}"
        )
    output = []
    for record, payload in zip(records, payloads):
        task, decision_id = source_identity(payload, scenario)
        template = full_by_key.get((decision_id, repeat))
        if template is None:
            raise ValueError(f"missing Full template for {decision_id} rep{repeat}")
        traces = [
            trace
            for trace in record.get("generation_trace") or []
            if trace.get("phase") != "regeneration"
        ]
        if not traces:
            raise ValueError(f"missing draft trace in {step_path}")
        trace = traces[0]
        stats = ((trace.get("generation") or {}).get("stats") or {})
        timing = stats.get("speed_timing") or {}
        native = timing.get("native_ttft_ms")
        prepare = (record.get("controller_timing") or {}).get("prepare_seconds")
        if not isinstance(native, (int, float)) or native <= 0:
            raise ValueError(f"invalid native TTFT in {step_path}")
        if not isinstance(prepare, (int, float)) or prepare < 0:
            raise ValueError(f"invalid controller timing in {step_path}")
        extraction = stats.get("sglang_extraction") or {}
        hits = int(extraction.get("cache_hits") or 0)
        misses = int(extraction.get("cache_misses") or 0)
        unique = int(extraction.get("unique_chunks") or 0)
        if misses != 0 or hits != unique:
            raise ValueError(
                f"cache hit was not 100% for {decision_id}: {extraction}"
            )
        controller = trace.get("controller") or {}
        reference = controller.get("same_prefix_full_reference") or {}
        resident_prompt = trace.get("planned_resident_prompt_tokens")
        full_prompt = reference.get("full_prompt_tokens")
        if not isinstance(full_prompt, int):
            full_prompt = resident_prompt
        reference_history = reference.get(
            "full_history_tokens", controller.get("history_tokens")
        )
        generated = core.action_canonical(record.get("response") or {})
        generated_json = json.dumps(
            generated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        reference_json = template["reference_action_json"]
        row: dict[str, Any] = dict(template)
        row.update(
            method="proposed",
            scenario=scenario,
            decision_key=record.get("decision_key"),
            repeat=repeat,
            timed=True,
            ttft_ms=float(native) + float(prepare) * 1000.0,
            native_ttft_ms=float(native),
            context_prepare_ms=float(prepare) * 1000.0,
            full_reference_prompt_tokens=full_prompt,
            resident_prompt_tokens=resident_prompt,
            full_reference_history_tokens=reference_history,
            resident_history_tokens=controller.get(
                "actual_history_tokens", controller.get("history_tokens")
            ),
            cache_hits=hits,
            cache_misses=misses,
            unique_chunks=unique,
            cache_hit_rate=(hits / unique if unique else 1.0),
            generated_action_json=generated_json,
            generated_action_sha256=hashlib.sha256(
                generated_json.encode()
            ).hexdigest(),
            action_exact_match=int(generated_json == reference_json),
        )
        output.append(row)
    return output


def make_subset(plan_path: Path, destination: Path) -> list[dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan = load(plan_path)
    payloads = plan.get("decisions", {}).get(TASK_ID)
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"{TASK_ID} is absent from {plan_path}")
    subset = {
        "schema": plan.get("schema"),
        "scenarios": [TASK_ID],
        "tasks": [TASK_ID],
        "decisions": {TASK_ID: payloads},
        "decision_count": len(payloads),
    }
    destination.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payloads


def run_patch(args: argparse.Namespace, subset_path: Path) -> None:
    parsed = bfcl.parser().parse_args(
        [
            "--phase",
            "formal",
            "--arm",
            "racer",
            "--gpu",
            str(args.gpu),
            "--out",
            str(args.patch_out),
            "--repetitions",
            "3",
            "--server-port",
            str(args.server_port),
            "--controller-port",
            str(args.controller_port),
            "--startup-timeout",
            str(args.startup_timeout),
            "--task-timeout",
            str(args.task_timeout),
            "--decisions-file",
            str(subset_path),
            "--model",
            str(args.model),
            "--embedding-model",
            str(args.embedding_model),
            "--eval-policy",
            str(args.eval_policy),
            "--eval-capacity",
            str(args.eval_capacity),
            "--run-ids",
            TASK_ID,
        ]
    )
    # The shared runner supports arbitrary repeat counts internally. The public
    # paper CLI enforces three, so this recovery path calls it directly once.
    parsed.repetitions = 1
    core.selected_scenarios = bfcl.selected_tasks
    core.run_c1_cli = bfcl.run_c1_cli
    core.capture_inputs = bfcl.capture_inputs
    core.run(parsed)


def merge(args: argparse.Namespace, payloads: list[dict[str, Any]]) -> None:
    full_rows = load_csv(args.pilot_root / "full" / "samples.csv")
    full_by_key = {
        (row["source_decision_id"], int(row["repeat"])): row for row in full_rows
    }
    plan = load(args.plan)
    expected_decisions = {
        source_identity(payload, task)[1]
        for task, task_payloads in plan["decisions"].items()
        for payload in task_payloads
    }
    if len(expected_decisions) != 185:
        raise ValueError(
            f"expected 185 unique pilot Decisions, got {len(expected_decisions)}"
        )

    merged: list[dict[str, Any]] = []
    timed_root = args.pilot_root / "racer" / "timed" / "racer_cache_hit"
    for repeat in range(3):
        for scenario, task_payloads in plan["decisions"].items():
            if repeat == 2 and scenario == TASK_ID:
                continue
            step_path = (
                timed_root
                / f"rep{repeat}"
                / scenario
                / "task_shards"
                / scenario
                / "server"
                / "steps.jsonl"
            )
            if not step_path.exists():
                raise FileNotFoundError(f"unexpected missing replay: {step_path}")
            merged.extend(
                samples_from_steps(
                    step_path,
                    scenario,
                    task_payloads,
                    repeat,
                    full_by_key,
                )
            )

    patch_rows = load_csv(args.patch_out / "samples.csv")
    if len(patch_rows) != len(payloads):
        raise ValueError(
            f"Task 149 patch expected {len(payloads)} rows, got {len(patch_rows)}"
        )
    for row in patch_rows:
        row["repeat"] = 2
        row["timed"] = True
        merged.append(row)

    keys = [(row["source_decision_id"], int(row["repeat"])) for row in merged]
    if len(merged) != 555 or len(set(keys)) != 555:
        raise ValueError(
            f"merged Racer rows are not unique 555/555: "
            f"rows={len(merged)} unique={len(set(keys))}"
        )
    counts: dict[str, int] = {}
    for decision_id, _ in keys:
        counts[decision_id] = counts.get(decision_id, 0) + 1
    invalid = {key: value for key, value in counts.items() if value != 3}
    if set(counts) != expected_decisions or invalid:
        raise ValueError(
            f"Decision pairing failed: missing={expected_decisions - set(counts)}, "
            f"extra={set(counts) - expected_decisions}, invalid={invalid}"
        )
    merged.sort(
        key=lambda row: (
            int(row["repeat"]),
            row["source_task_id"],
            int(row["source_user_turn"]),
            int(row["source_step"]),
            int(row["source_attempt"]),
        )
    )
    destination = args.pilot_root / "racer" / "samples.csv"
    write_csv(destination, merged)
    report = {
        "schema": "decision-dual-gp-racer-recovery-v1",
        "recovered_task": TASK_ID,
        "recovered_repeat": 2,
        "recovered_rows": len(payloads),
        "merged_rows": len(merged),
        "unique_task_repeat_decisions": len(set(keys)),
        "unique_decisions": len(counts),
        "repeats_per_decision": 3,
        "cache_misses": sum(int(row["cache_misses"]) for row in merged),
        "samples_csv": str(destination.resolve()),
        "patch_output": str(args.patch_out.resolve()),
    }
    report_path = args.pilot_root / "racer" / "recovery_task149_rep2.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pilot-root", type=Path, required=True)
    result.add_argument("--patch-out", type=Path, required=True)
    result.add_argument("--gpu", type=int, default=4)
    result.add_argument("--server-port", type=int, default=52664)
    result.add_argument("--controller-port", type=int, default=53760)
    result.add_argument("--startup-timeout", type=int, default=1200)
    result.add_argument("--task-timeout", type=int, default=3600)
    result.add_argument("--model", type=Path, default=core.MODEL)
    result.add_argument("--embedding-model", type=Path, default=core.EMBED)
    result.add_argument("--eval-policy", type=Path, default=core.POLICY)
    result.add_argument("--eval-capacity", type=Path, default=core.CAPACITY)
    return result


def main() -> None:
    args = parser().parse_args()
    args.pilot_root = args.pilot_root.resolve()
    args.patch_out = args.patch_out.resolve()
    args.plan = args.pilot_root / "plan" / "decisions.json"
    if args.patch_out.exists():
        raise FileExistsError(f"refusing to overwrite {args.patch_out}")
    if (args.pilot_root / "racer" / "samples.csv").exists():
        raise FileExistsError("racer/samples.csv already exists; refusing to replace it")
    subset_path = args.patch_out.with_suffix(".decisions.json")
    if subset_path.exists():
        raise FileExistsError(f"refusing to overwrite {subset_path}")
    payloads = make_subset(args.plan, subset_path)
    try:
        run_patch(args, subset_path)
        merge(args, payloads)
    finally:
        subset_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
