#!/usr/bin/env python3
"""Paired BFCL Long TTFT on each task's final or longest decision.

Also measures server-side C2KV cache-miss materialization latency during the
excluded offline warm pass. Importing/planning is CPU-only; --execute owns GPU
processes explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import bfcl_long_length_bucket_ttft as base

DEFAULT_SOURCE = base.DEFAULT_SOURCE
DEFAULT_THRESHOLD = 20_000
BFCL_MAXIMUM_STEP_LIMIT = 20


def longest_per_task(rows):
    selected = {}
    for row in rows:
        key = (
            row["full_context_tokens"], row["user_turn"], row["step"],
            row["attempt"], row["source_decision_id"],
        )
        previous = selected.get(row["task_id"])
        if previous is None:
            selected[row["task_id"]] = row
            continue
        previous_key = (
            previous["full_context_tokens"], previous["user_turn"],
            previous["step"], previous["attempt"],
            previous["source_decision_id"],
        )
        if key > previous_key:
            selected[row["task_id"]] = row
    return sorted(selected.values(), key=lambda row: row["task_id"])


def final_per_task(rows):
    selected = {}
    for row in rows:
        key = (
            row["user_turn"], row["step"], row["attempt"],
            row["source_decision_id"],
        )
        previous = selected.get(row["task_id"])
        if previous is None:
            selected[row["task_id"]] = row
            continue
        previous_key = (
            previous["user_turn"], previous["step"], previous["attempt"],
            previous["source_decision_id"],
        )
        if key > previous_key:
            selected[row["task_id"]] = row
    return sorted(selected.values(), key=lambda row: row["task_id"])


def completed_history_eligibility(data_path, source_root, available_task_ids):
    """Audit the denominator for a completed-history timing experiment.

    BFCL Long contains a small number of official one-user-turn scenarios. They
    are valid BFCL tasks, but they cannot provide a decision after a completed
    user turn and therefore are not eligible for this history-compression
    timing protocol.
    """

    official = list(base.read_jsonl(data_path))
    source_files = base.source_step_files(source_root)
    step_files = {path.parents[1].name: path for path in source_files}
    if len(step_files) != len(source_files):
        raise ValueError("duplicate source step files for a BFCL task")

    rows = []
    expected_eligible = set()
    for task in official:
        task_id = task.get("id")
        questions = task.get("question")
        if not isinstance(task_id, str) or not isinstance(questions, list):
            raise ValueError(f"invalid BFCL task row in {data_path}")
        official_turns = len(questions)
        structurally_eligible = official_turns >= 2
        if structurally_eligible:
            expected_eligible.add(task_id)

        step_path = step_files.get(task_id)
        source_steps = list(base.read_jsonl(step_path)) if step_path else []
        observed_turns = []
        observed_steps = []
        for record in source_steps:
            payload = (record.get("harness_transport_normalization") or {}).get(
                "same_input_replay_request"
            ) or {}
            context = payload.get("c2kv_eval_context") or {}
            turn = context.get("user_turn")
            step = context.get("step")
            if isinstance(turn, int):
                observed_turns.append(turn)
            if isinstance(step, int):
                observed_steps.append(step)

        audit = {}
        if step_path:
            task_dir = step_path.parents[1]
            audit_paths = sorted(task_dir.glob("bfcl/bfcl/task_audit/tasks.jsonl"))
            if audit_paths:
                audit_rows = list(base.read_jsonl(audit_paths[0]))
                if len(audit_rows) != 1:
                    raise ValueError(f"expected one task audit row: {audit_paths[0]}")
                audit = audit_rows[0]
        base_error = audit.get("base_error")
        rows.append({
            "task_id": task_id,
            "official_user_turn_count": official_turns,
            "completed_history_eligible": structurally_eligible,
            "observed_completed_history_candidate": task_id in available_task_ids,
            "source_step_count": len(source_steps),
            "source_max_user_turn": max(observed_turns) if observed_turns else None,
            "source_max_step": max(observed_steps) if observed_steps else None,
            "source_reached_bfcl_step_limit": bool(
                observed_steps and max(observed_steps) >= BFCL_MAXIMUM_STEP_LIMIT
            ),
            "source_base_error_type": (
                base_error.get("type") if isinstance(base_error, dict) else None
            ),
            "source_base_error_message": (
                base_error.get("message") if isinstance(base_error, dict) else None
            ),
            "eligibility_reason": (
                "eligible_completed_history"
                if structurally_eligible
                else "ineligible_official_single_user_turn"
            ),
        })

    actual = set(available_task_ids)
    missing = sorted(expected_eligible - actual)
    unexpected = sorted(actual - expected_eligible)
    if missing or unexpected:
        raise ValueError(
            "completed-history denominator mismatch: "
            f"missing_eligible={missing}, unexpected={unexpected}"
        )
    return rows


def source_decision_id(record):
    payload = (record.get("harness_transport_normalization") or {}).get(
        "same_input_replay_request"
    )
    if not isinstance(payload, dict):
        raise ValueError("offline warm record lacks same-input replay identity")
    context = payload.get("c2kv_eval_context") or {}
    task = context.get("task_id")
    turn = context.get("user_turn")
    step = context.get("step")
    attempt = context.get("attempt", 0)
    if not isinstance(task, str) or not isinstance(turn, int) or not isinstance(step, int):
        raise ValueError("offline warm replay identity is incomplete")
    return f"{task}/turn-{turn}/step-{step}/attempt-{attempt}"


def offline_rows(root):
    result = []
    pattern = "runs/c2kv_gpu*/offline_warm/*/task_shards/*/server/steps.jsonl"
    for path in sorted(root.glob(pattern)):
        for record in base.read_jsonl(path):
            traces = [
                trace for trace in record.get("generation_trace") or []
                if trace.get("phase") != "regeneration"
                and trace.get("status") == "completed"
            ]
            if len(traces) != 1:
                raise ValueError(f"expected one offline draft trace in {path}")
            stats = (traces[0].get("generation") or {}).get("stats") or {}
            extraction = stats.get("sglang_extraction") or {}
            misses = extraction.get("cache_misses")
            model_calls = extraction.get("model_calls")
            duration = extraction.get("materialized_extraction_duration_ns_sum")
            gpu_duration = extraction.get(
                "materialized_gist_generation_duration_ns_sum"
            )
            if not isinstance(misses, int) or misses < 0 or model_calls != misses:
                raise ValueError(f"invalid offline materialization count: {path}")
            if misses > 0 and (not isinstance(duration, int) or duration <= 0):
                raise ValueError(f"offline extraction timing is unavailable: {path}")
            if misses > 0 and (not isinstance(gpu_duration, int) or gpu_duration <= 0):
                raise ValueError(f"offline gist GPU timing is unavailable: {path}")
            identity = source_decision_id(record)
            result.append({
                "source_decision_id": identity,
                "task_id": identity.split("/", 1)[0],
                "offline_extraction_wall_ms": (duration or 0) / 1e6,
                "offline_gist_gpu_ms": (gpu_duration or 0) / 1e6,
                "offline_materialized_tokens": int(
                    extraction.get("materialized_encoder_tokens") or 0
                ),
                "offline_materialized_chunks": misses,
                "offline_chunk_wall_max_ms": (
                    extraction.get("materialized_extraction_duration_ns_max") or 0
                ) / 1e6,
                "offline_chunk_gpu_max_ms": (
                    extraction.get("materialized_gist_generation_duration_ns_max") or 0
                ) / 1e6,
            })
    return result


def p(values, q):
    return base.percentile(list(values), q)


def summarize_group(name, rows):
    if not rows:
        return None
    full = [row["full_online_ttft_ms"] for row in rows]
    c2kv = [row["c2kv_online_ttft_ms"] for row in rows]
    offline = [row["offline_extraction_wall_ms"] for row in rows]
    gpu = [row["offline_gist_gpu_ms"] for row in rows]
    savings = [f - c for f, c in zip(full, c2kv)]
    net = [c + o for c, o in zip(c2kv, offline)]
    return {
        "group": name,
        "decision_count": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "full_context_tokens_p50": p((row["full_context_tokens"] for row in rows), .5),
        "full_context_tokens_p90": p((row["full_context_tokens"] for row in rows), .9),
        "full_context_tokens_p95": p((row["full_context_tokens"] for row in rows), .95),
        "full_online_ttft_p50_ms": p(full, .5),
        "c2kv_online_ttft_p50_ms": p(c2kv, .5),
        "online_ttft_p50_reduction": 1 - p(c2kv, .5) / p(full, .5),
        "paired_online_reduction_ge_50_rate": sum(
            row["online_ttft_reduction"] >= .5 for row in rows
        ) / len(rows),
        "incremental_required_gap_p50_ms": p(offline, .5),
        "incremental_required_gap_p95_ms": p(offline, .95),
        "offline_gist_gpu_p50_ms": p(gpu, .5),
        "offline_gist_gpu_p95_ms": p(gpu, .95),
        "offline_materialization_throughput_tokens_per_sec": (
            1000 * sum(row["offline_materialized_tokens"] for row in rows)
            / sum(offline)
        ),
        "offline_break_even_vs_online_saving_rate": sum(
            o <= max(0, s) for o, s in zip(offline, savings)
        ) / len(rows),
        "conservative_worst_step_plus_ttft_p50_reduction": (
            1 - p(net, .5) / p(full, .5)
        ),
        "offline_fits_250ms_gap_rate": sum(value <= 250 for value in offline) / len(rows),
        "offline_fits_500ms_gap_rate": sum(value <= 500 for value in offline) / len(rows),
        "offline_fits_1s_gap_rate": sum(value <= 1000 for value in offline) / len(rows),
        "offline_fits_2s_gap_rate": sum(value <= 2000 for value in offline) / len(rows),
        "offline_fits_5s_gap_rate": sum(value <= 5000 for value in offline) / len(rows),
    }


def summarize(root, threshold, selection):
    base.summarize(root)
    with (root / "paired_decisions.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))
    numeric = (
        "full_context_tokens", "full_online_ttft_ms", "c2kv_online_ttft_ms",
        "online_ttft_reduction",
    )
    for row in paired:
        for key in numeric:
            row[key] = float(row[key])
    offline = offline_rows(root)
    by_id = {row["source_decision_id"]: row for row in offline}
    if len(by_id) != len(offline):
        raise ValueError("duplicate offline timing identity")
    by_task = {}
    for timing in offline:
        by_task.setdefault(timing["task_id"], []).append(timing)
    joined = []
    for row in paired:
        selected_timing = by_id.get(row["source_decision_id"])
        task_timings = by_task.get(row["task_id"], [])
        if selected_timing is None or not task_timings:
            raise ValueError(f"missing offline timing: {row['source_decision_id']}")
        wall = [item["offline_extraction_wall_ms"] for item in task_timings]
        gpu = [item["offline_gist_gpu_ms"] for item in task_timings]
        aggregate = {
            "offline_incremental_step_count": len(task_timings),
            "offline_incremental_materialization_steps": sum(
                item["offline_materialized_chunks"] > 0 for item in task_timings
            ),
            "offline_extraction_wall_total_ms": sum(wall),
            "offline_extraction_wall_p95_step_ms": p(wall, .95),
            "offline_extraction_wall_ms": max(wall),
            "offline_gist_gpu_total_ms": sum(gpu),
            "offline_gist_gpu_p95_step_ms": p(gpu, .95),
            "offline_gist_gpu_ms": max(gpu),
            "offline_materialized_tokens": sum(
                item["offline_materialized_tokens"] for item in task_timings
            ),
            "offline_materialized_chunks": sum(
                item["offline_materialized_chunks"] for item in task_timings
            ),
            "offline_selected_decision_new_wall_ms": selected_timing[
                "offline_extraction_wall_ms"
            ],
        }
        joined.append({**row, **aggregate})
    base.write_csv(root / f"{selection}_decisions_with_offline.csv", joined)
    groups = [
        summarize_group(f"all_task_{selection}", joined),
        summarize_group(
            f"task_{selection}_ge_{threshold}_tokens",
            [row for row in joined if row["full_context_tokens"] >= threshold],
        ),
    ]
    groups = [row for row in groups if row is not None]
    base.write_csv(root / f"{selection}_summary.csv", groups)
    base.save(root / f"{selection}_summary.json", groups)
    return groups


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-full-root", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--bins", default=base.DEFAULT_BINS)
    p.add_argument("--long-context-threshold", type=int, default=DEFAULT_THRESHOLD)
    p.add_argument(
        "--selection", choices=("final", "longest"), default="final",
        help="Select each task's final valid decision (default) or longest prompt.",
    )
    p.add_argument(
        "--max-tasks", type=int,
        help="Debug-only deterministic length-spread subset; omit for all tasks.",
    )
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--port-base", type=int, default=42800)
    p.add_argument("--controller-port-base", type=int, default=20000)
    p.add_argument("--startup-timeout", type=int, default=1200)
    p.add_argument("--task-timeout", type=int, default=7200)
    p.add_argument("--client-python", type=Path, default=Path(
        "/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python"
    ))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--summarize", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.out, args.long_context_threshold, args.selection), indent=2))
        return
    if args.repetitions != 3:
        raise ValueError("paired timing requires exactly three repetitions")
    if args.long_context_threshold <= 0:
        raise ValueError("long-context threshold must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    edges = base.parse_bins(args.bins)
    pool = base.candidates(args.source_full_root, edges)
    selected = (
        final_per_task(pool) if args.selection == "final" else longest_per_task(pool)
    )
    if not selected:
        raise ValueError("source Full run has no completed-history decisions")
    full_selected_count = len(selected)
    available_task_ids = {row["task_id"] for row in selected}
    if args.max_tasks is not None:
        if not 4 <= args.max_tasks <= full_selected_count:
            raise ValueError("max-tasks must be at least four and no larger than the pool")
        ordered = sorted(selected, key=lambda row: (
            row["full_context_tokens"], row["source_decision_id"]
        ))
        indices = [
            round(i * (len(ordered) - 1) / (args.max_tasks - 1))
            for i in range(args.max_tasks)
        ]
        selected = [ordered[index] for index in indices]
    jobs = base.make_jobs(args, selected)
    all_by_task = {}
    for row in pool:
        if row["task_id"] in {item["task_id"] for item in selected}:
            all_by_task.setdefault(row["task_id"], []).append(row)
    for values in all_by_task.values():
        values.sort(key=lambda row: (
            row["user_turn"], row["step"], row["attempt"], row["source_decision_id"]
        ))
    for job in jobs:
        if job["arm"] != "c2kv":
            continue
        timed_payload = json.loads(Path(job["decision_file"]).read_text())
        warm_rows = [
            row for task in timed_payload["scenarios"] for row in all_by_task[task]
        ]
        warm_path = args.out / "shards" / f"c2kv_warm_gpu{job['gpu']}.json"
        base.decision_file(warm_path, warm_rows)
        job["warm_decision_file"] = str(warm_path)
        job["command"] += ["--warm-decisions-file", warm_path]
    official_data = (
        base.HERE.parents[2] / "bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
    )
    eligibility = completed_history_eligibility(
        official_data, args.source_full_root, available_task_ids
    )
    official = {row["task_id"] for row in eligibility}
    ineligible = [row for row in eligibility if not row["completed_history_eligible"]]
    base.save(args.out / "eligibility_audit.json", eligibility)
    base.write_csv(args.out / "eligibility_audit.csv", eligibility)
    selected_tasks = {row["task_id"] for row in selected}
    public = [{k: v for k, v in row.items() if k != "payload"} for row in selected]
    manifest = {
        "schema": "bfcl-long-task-selected-offline-ttft-v2",
        "source_full_root": str(args.source_full_root),
        "source_warning": (
            "speed-only frozen replay; regenerate source with parser-fixed Full for "
            "a formal trajectory-distribution claim"
        ),
        "selection_mode": args.selection,
        "selection": (
            "one final valid decision per task"
            if args.selection == "final"
            else "one maximum server-tokenized Full prompt per task"
        ),
        "selection_sha256": hashlib.sha256(json.dumps(
            [row["source_decision_id"] for row in selected], separators=(",", ":")
        ).encode()).hexdigest(),
        "official_task_count": len(official),
        "completed_history_eligible_task_count": len(eligibility) - len(ineligible),
        "completed_history_ineligible_task_count": len(ineligible),
        "completed_history_denominator": (
            "official tasks with at least two user turns, so the selected "
            "decision has at least one completed user-turn history document"
        ),
        "available_task_longest_count": full_selected_count,
        "selected_task_count": len(selected_tasks),
        "debug_max_tasks": args.max_tasks,
        "tasks_without_completed_history": sorted(official - available_task_ids),
        "tasks_without_completed_history_details": ineligible,
        "long_context_threshold": args.long_context_threshold,
        "repetitions": args.repetitions,
        "selected": public,
        "jobs": [{**job, "command": [str(item) for item in job["command"]]} for job in jobs],
        "timing": {
            "online_ttft": "controller prepare + native request through first token",
            "offline_extraction_wall": (
                "per-decision newly materialized chunks; task required gap is the maximum "
                "server c2kv_extract synchronous cache-miss duration"
            ),
            "offline_gist_gpu": "per-decision synchronized forward_c2kv_extract duration",
            "offline_order": (
                "source task turn/step order; timed replay uses the selected "
                f"{args.selection} decision only"
            ),
            "radix_cache": "disabled",
            "offline_warm_excluded_from_online_ttft": True,
        },
    }
    base.save(args.out / "selection_manifest.json", manifest)
    if not args.execute:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    base.execute_jobs(jobs)
    print(json.dumps(summarize(args.out, args.long_context_threshold, args.selection), indent=2))


if __name__ == "__main__":
    main()
