#!/usr/bin/env python3
"""Paired BFCL Long Full/C2KV TTFT by full-context length bucket.

The source Full run freezes visible decision requests and their server-tokenized
Full prompt lengths. Selected requests are replayed identically through Full
and C2KV. Importing/planning is CPU-only; GPU execution requires --execute.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

from c2kv_eval.portable.proxy import action_canonical
from ttft_process import stop_process

HERE = Path(__file__).resolve().parent
REPLAY = HERE / "bfcl_long_cache_hit_ttft.py"
DEFAULT_SOURCE = Path(
    "/home/zhuyuhan/project/bfcl-c2kv/results/"
    "bfcl_long_soft256_full200_gpu012_20261029_153723"
)
DEFAULT_BINS = "0,8192,16384,32768,65536,inf"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - point) + ordered[hi] * (point - lo)


def parse_bins(raw: str):
    edges = []
    for item in raw.split(","):
        item = item.strip().lower()
        edges.append(math.inf if item in {"inf", "+inf", "infinity"} else int(item))
    if len(edges) < 3 or edges[0] != 0 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("bins must be strictly increasing and start at zero")
    if not math.isinf(edges[-1]):
        raise ValueError("the final bin edge must be inf")
    return edges


def bin_label(lo, hi):
    def short(value):
        if math.isinf(value):
            return "inf"
        return f"{int(value / 1024)}K" if value and value % 1024 == 0 else str(int(value))
    return f"[{short(lo)}, {short(hi)})"


def source_step_files(root: Path):
    files = sorted(root.glob("gpu*/rep0/full_cold/run/task_shards/*/server/steps.jsonl"))
    if not files:
        files = sorted(root.glob("**/full_cold/run/task_shards/*/server/steps.jsonl"))
    if not files:
        raise FileNotFoundError(f"no Full steps.jsonl under {root}")
    return files


def candidates(root: Path, edges):
    result = []
    seen = set()
    for path in source_step_files(root):
        for record in read_jsonl(path):
            payload = (record.get("harness_transport_normalization") or {}).get(
                "same_input_replay_request"
            )
            if not isinstance(payload, dict):
                continue
            messages = payload.get("messages") or []
            if sum(message.get("role") == "user" for message in messages) < 2:
                continue
            traces = [
                trace for trace in record.get("generation_trace") or []
                if trace.get("phase") != "regeneration" and trace.get("status") == "completed"
            ]
            if not traces:
                continue
            full_tokens = traces[0].get("planned_resident_prompt_tokens")
            if not isinstance(full_tokens, int) or full_tokens <= 0:
                raise ValueError(f"invalid Full prompt token count in {path}")
            context = payload.get("c2kv_eval_context") or {}
            task_id = context.get("task_id")
            user_turn = context.get("user_turn")
            step = context.get("step")
            attempt = context.get("attempt", 0)
            if not isinstance(task_id, str) or not isinstance(user_turn, int) or not isinstance(step, int):
                raise ValueError(f"incomplete replay identity in {path}")
            decision_id = f"{task_id}/turn-{user_turn}/step-{step}/attempt-{attempt}"
            if decision_id in seen:
                raise ValueError(f"duplicate source decision: {decision_id}")
            seen.add(decision_id)
            index = next(
                (i for i, (lo, hi) in enumerate(zip(edges, edges[1:])) if lo <= full_tokens < hi),
                None,
            )
            if index is None:
                raise ValueError(f"cannot bucket {full_tokens} tokens")
            result.append({
                "source_decision_id": decision_id,
                "task_id": task_id,
                "user_turn": user_turn,
                "step": step,
                "attempt": attempt,
                "decision_key": record.get("decision_key"),
                "full_context_tokens": full_tokens,
                "full_history_tokens": (traces[0].get("controller") or {}).get("history_tokens"),
                "bucket_index": index,
                "bucket": bin_label(edges[index], edges[index + 1]),
                "payload": {
                    **payload,
                    "_replay_reference_action": action_canonical(
                        record.get("response") or {}
                    ),
                },
            })
    return result


def balanced_sample(rows, edges, per_bin, seed):
    rng = random.Random(seed)
    selected = []
    availability = []
    for index in range(len(edges) - 1):
        pool = [row for row in rows if row["bucket_index"] == index]
        by_task = {}
        for row in pool:
            by_task.setdefault(row["task_id"], []).append(row)
        task_ids = sorted(by_task)
        rng.shuffle(task_ids)
        for values in by_task.values():
            values.sort(key=lambda row: (row["user_turn"], row["step"], row["attempt"]))
            rng.shuffle(values)
        picked = []
        while len(picked) < per_bin and task_ids:
            next_round = []
            for task_id in task_ids:
                values = by_task[task_id]
                if values and len(picked) < per_bin:
                    picked.append(values.pop())
                if values:
                    next_round.append(task_id)
            task_ids = next_round
        availability.append({
            "bucket_index": index,
            "bucket": bin_label(edges[index], edges[index + 1]),
            "available_decisions": len(pool),
            "available_tasks": len(by_task),
            "selected_decisions": len(picked),
            "selected_tasks": len({row["task_id"] for row in picked}),
        })
        if len(picked) != per_bin:
            raise ValueError(
                f"bucket {availability[-1]['bucket']} has {len(pool)} candidates; "
                f"cannot select {per_bin}"
            )
        selected.extend(picked)
    selected.sort(key=lambda row: (
        row["bucket_index"], row["full_context_tokens"], row["source_decision_id"]
    ))
    return selected, availability


def decision_file(path: Path, rows):
    grouped = {}
    for row in sorted(rows, key=lambda item: (
        item["task_id"], item["user_turn"], item["step"], item["attempt"]
    )):
        grouped.setdefault(row["task_id"], []).append(row["payload"])
    scenarios = list(grouped)
    save(path, {
        "schema": "bfcl-long-length-bucket-decisions-v1",
        "scenarios": scenarios,
        "tasks": scenarios,
        "decisions": grouped,
        "decision_count": len(rows),
    })
    return scenarios


def partition_by_task(rows, count):
    groups = {}
    for row in rows:
        groups.setdefault(row["task_id"], []).append(row)
    shards = [[] for _ in range(count)]
    loads = [0] * count
    for _, values in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        slot = min(range(count), key=lambda i: (loads[i], i))
        shards[slot].extend(values)
        loads[slot] += len(values)
    if any(not shard for shard in shards):
        raise ValueError("more GPU shards than selected task groups")
    return shards


def make_jobs(args, selected):
    gpus = [int(item) for item in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("provide at least one unique physical GPU")
    shards = partition_by_task(selected, len(gpus))
    jobs = []
    for slot, (gpu, shard) in enumerate(zip(gpus, shards)):
        for arm in ("full", "c2kv"):
            root = args.out / "runs" / f"{arm}_gpu{gpu}"
            decision_path = args.out / "shards" / f"{arm}_gpu{gpu}.json"
            task_ids = decision_file(decision_path, shard)
            command = [
                str(args.client_python), str(REPLAY),
                "--phase", "formal", "--arm", arm,
                "--gpu", str(gpu), "--out", str(root),
                "--repetitions", str(args.repetitions),
                "--server-port", str(args.port_base + slot * 300),
                "--controller-port", str(
                    getattr(args, "controller_port_base", args.port_base + 1000)
                    + slot * 1000
                ),
                "--startup-timeout", str(args.startup_timeout),
                "--task-timeout", str(args.task_timeout),
                "--decisions-file", str(decision_path),
                "--run-ids", *task_ids,
            ]
            jobs.append({
                "arm": arm,
                "gpu": gpu,
                "gpu_stage": 0 if arm == "full" else 1,
                "decision_count": len(shard),
                "task_count": len(task_ids),
                "output": str(root),
                "decision_file": str(decision_path),
                "command": command,
            })
    return jobs


def launch_job(job):
    log_path = Path(job["output"]).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    process = subprocess.Popen(
        job["command"], cwd=HERE.parents[2], stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    return process, log, log_path


def execute_jobs(jobs):
    queues = {}
    for job in sorted(jobs, key=lambda item: (item["gpu"], item["gpu_stage"])):
        queues.setdefault(job["gpu"], []).append(job)
    active = {}
    try:
        for gpu, queue in queues.items():
            job = queue.pop(0)
            active[gpu] = (job, *launch_job(job))
        while active:
            for gpu, item in list(active.items()):
                job, process, log, log_path = item
                code = process.poll()
                if code is None:
                    continue
                log.close()
                if code:
                    raise RuntimeError(
                        f"{job['arm']} GPU{job['gpu']} exited {code}; see {log_path}"
                    )
                if queues[gpu]:
                    next_job = queues[gpu].pop(0)
                    active[gpu] = (next_job, *launch_job(next_job))
                else:
                    del active[gpu]
            if active:
                time.sleep(5)
    finally:
        for _, process, log, _ in active.values():
            stop_process(process)
            if not log.closed:
                log.close()


def load_samples(root: Path):
    result = []
    for path in sorted((root / "runs").glob("*/samples.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in (
                    "repeat", "source_user_turn", "source_step", "source_attempt",
                    "full_reference_prompt_tokens", "resident_prompt_tokens",
                    "full_reference_history_tokens", "resident_history_tokens",
                ):
                    if row.get(key) not in (None, "", "None"):
                        row[key] = int(float(row[key]))
                for key in ("ttft_ms", "native_ttft_ms", "context_prepare_ms"):
                    row[key] = float(row[key])
                if row.get("action_exact_match") not in (None, ""):
                    row["action_exact_match"] = int(row["action_exact_match"])
                result.append(row)
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path):
    manifest = json.loads((root / "selection_manifest.json").read_text())
    selected = manifest["selected"]
    metadata = {row["source_decision_id"]: row for row in selected}
    samples = load_samples(root)
    if not samples:
        raise ValueError("no replay samples found")
    expected_repeats = manifest["repetitions"]
    grouped = {}
    for row in samples:
        grouped.setdefault((row["method"], row["source_decision_id"]), []).append(row)
    paired = []
    for decision_id, source in metadata.items():
        entries = {}
        for method in ("full", "c2kv_only"):
            rows = grouped.get((method, decision_id), [])
            if len(rows) != expected_repeats:
                raise ValueError(
                    f"{decision_id} {method}: expected {expected_repeats} repeats, got {len(rows)}"
                )
            if any(
                row["full_reference_prompt_tokens"] != source["full_context_tokens"]
                for row in rows
            ):
                raise ValueError(f"Full-reference token mismatch: {decision_id} {method}")
            entries[method] = {
                "native": percentile([row["native_ttft_ms"] for row in rows], .5),
                "online": percentile([row["ttft_ms"] for row in rows], .5),
                "resident": int(percentile([row["resident_prompt_tokens"] for row in rows], .5)),
                "resident_history": int(percentile([
                    row.get("resident_history_tokens") or 0 for row in rows
                ], .5)),
                "action_fidelity": (
                    sum(row["action_exact_match"] for row in rows) / len(rows)
                    if all(row.get("action_exact_match") in (0, 1) for row in rows)
                    else None
                ),
            }
        full, c2kv = entries["full"], entries["c2kv_only"]
        paired.append({
            "source_decision_id": decision_id,
            "task_id": source["task_id"],
            "bucket_index": source["bucket_index"],
            "bucket": source["bucket"],
            "full_context_tokens": source["full_context_tokens"],
            "full_history_tokens": source.get("full_history_tokens"),
            "c2kv_resident_context_tokens": c2kv["resident"],
            "c2kv_resident_history_tokens": c2kv["resident_history"],
            "effective_context_compression": source["full_context_tokens"] / c2kv["resident"],
            "full_native_ttft_ms": full["native"],
            "c2kv_native_ttft_ms": c2kv["native"],
            "native_ttft_ratio": c2kv["native"] / full["native"],
            "native_ttft_reduction": 1 - c2kv["native"] / full["native"],
            "full_online_ttft_ms": full["online"],
            "c2kv_online_ttft_ms": c2kv["online"],
            "online_ttft_ratio": c2kv["online"] / full["online"],
            "online_ttft_reduction": 1 - c2kv["online"] / full["online"],
            "full_reference_action_fidelity": full["action_fidelity"],
            "c2kv_reference_action_fidelity": c2kv["action_fidelity"],
            "action_fidelity_delta_pp": (
                100 * (c2kv["action_fidelity"] - full["action_fidelity"])
                if full["action_fidelity"] is not None
                and c2kv["action_fidelity"] is not None else None
            ),
        })
    summary = []
    for bucket_index in sorted({row["bucket_index"] for row in paired}):
        rows = [row for row in paired if row["bucket_index"] == bucket_index]
        full_native = [row["full_native_ttft_ms"] for row in rows]
        c2_native = [row["c2kv_native_ttft_ms"] for row in rows]
        full_online = [row["full_online_ttft_ms"] for row in rows]
        c2_online = [row["c2kv_online_ttft_ms"] for row in rows]
        full_p50, c2_p50 = percentile(full_native, .5), percentile(c2_native, .5)
        full_online_p50 = percentile(full_online, .5)
        c2_online_p50 = percentile(c2_online, .5)
        full_fidelity = [
            row["full_reference_action_fidelity"] for row in rows
            if row["full_reference_action_fidelity"] is not None
        ]
        c2kv_fidelity = [
            row["c2kv_reference_action_fidelity"] for row in rows
            if row["c2kv_reference_action_fidelity"] is not None
        ]
        fidelity_delta = [
            row["action_fidelity_delta_pp"] for row in rows
            if row["action_fidelity_delta_pp"] is not None
        ]
        summary.append({
            "bucket_index": bucket_index,
            "bucket": rows[0]["bucket"],
            "decision_count": len(rows),
            "task_count": len({row["task_id"] for row in rows}),
            "full_context_tokens_min": min(row["full_context_tokens"] for row in rows),
            "full_context_tokens_mean": sum(row["full_context_tokens"] for row in rows) / len(rows),
            "full_context_tokens_p50": percentile([row["full_context_tokens"] for row in rows], .5),
            "full_context_tokens_max": max(row["full_context_tokens"] for row in rows),
            "c2kv_resident_context_tokens_p50": percentile([
                row["c2kv_resident_context_tokens"] for row in rows
            ], .5),
            "effective_context_compression_weighted": (
                sum(row["full_context_tokens"] for row in rows)
                / sum(row["c2kv_resident_context_tokens"] for row in rows)
            ),
            "full_native_ttft_p50_ms": full_p50,
            "full_native_ttft_p95_ms": percentile(full_native, .95),
            "c2kv_native_ttft_p50_ms": c2_p50,
            "c2kv_native_ttft_p95_ms": percentile(c2_native, .95),
            "native_ttft_ratio_of_p50": c2_p50 / full_p50,
            "native_ttft_reduction_of_p50": 1 - c2_p50 / full_p50,
            "paired_native_ttft_ratio_p50": percentile([
                row["native_ttft_ratio"] for row in rows
            ], .5),
            "full_online_ttft_p50_ms": full_online_p50,
            "c2kv_online_ttft_p50_ms": c2_online_p50,
            "online_ttft_ratio_of_p50": c2_online_p50 / full_online_p50,
            "online_ttft_reduction_of_p50": 1 - c2_online_p50 / full_online_p50,
            "full_reference_action_fidelity": (
                sum(full_fidelity) / len(full_fidelity) if full_fidelity else None
            ),
            "c2kv_reference_action_fidelity": (
                sum(c2kv_fidelity) / len(c2kv_fidelity) if c2kv_fidelity else None
            ),
            "action_fidelity_delta_pp": (
                sum(fidelity_delta) / len(fidelity_delta) if fidelity_delta else None
            ),
        })
    write_csv(root / "samples.csv", samples)
    write_csv(root / "paired_decisions.csv", paired)
    write_csv(root / "bucket_summary.csv", summary)
    save(root / "bucket_summary.json", summary)
    plot(root, summary)
    return summary


def plot(root: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["bucket"] for row in rows]
    x = list(range(len(rows)))
    fig, (latency, ratio) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    latency.plot(x, [row["full_native_ttft_p50_ms"] for row in rows], "o-", label="Full")
    latency.plot(x, [row["c2kv_native_ttft_p50_ms"] for row in rows], "o-", label="C2KV")
    latency.set_ylabel("Native TTFT P50 (ms)")
    latency.set_title("BFCL Long: paired TTFT by Full context length")
    latency.grid(alpha=.25)
    latency.legend()
    ratios = [100 * row["native_ttft_ratio_of_p50"] for row in rows]
    bars = ratio.bar(x, ratios, color="#4C78A8")
    ratio.axhline(100, color="#C44E52", linestyle="--", linewidth=1)
    ratio.set_ylabel("C2KV / Full TTFT (%)")
    ratio.set_xlabel("Full-context token bucket")
    ratio.set_xticks(x, labels)
    ratio.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, ratios):
        ratio.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%",
                   ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "ttft_by_context_bucket.png", dpi=180)
    fig.savefig(root / "ttft_by_context_bucket.pdf")
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-full-root", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--bins", default=DEFAULT_BINS)
    p.add_argument("--samples-per-bin", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--gpus", default="0,1,3,4,5,6,7")
    p.add_argument("--port-base", type=int, default=41000)
    p.add_argument("--controller-port-base", type=int, default=20000)
    p.add_argument("--startup-timeout", type=int, default=1200)
    p.add_argument("--task-timeout", type=int, default=3600)
    p.add_argument("--client-python", type=Path, default=Path(
        "/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python"
    ))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--summarize", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.summarize and not args.execute:
        print(json.dumps(summarize(args.out), indent=2, ensure_ascii=False))
        return
    if args.repetitions != 3:
        raise ValueError("the paired timing protocol requires exactly 3 repetitions")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    edges = parse_bins(args.bins)
    pool = candidates(args.source_full_root, edges)
    selected, availability = balanced_sample(pool, edges, args.samples_per_bin, args.seed)
    jobs = make_jobs(args, selected)
    public_selected = [
        {key: value for key, value in row.items() if key != "payload"}
        for row in selected
    ]
    manifest = {
        "schema": "bfcl-long-length-bucket-ttft-v1",
        "source_full_root": str(args.source_full_root),
        "source_step_file_count": len(source_step_files(args.source_full_root)),
        "bins": edges[:-1] + ["inf"],
        "samples_per_bin": args.samples_per_bin,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "selection_sha256": hashlib.sha256(json.dumps(
            [row["source_decision_id"] for row in selected], separators=(",", ":")
        ).encode()).hexdigest(),
        "availability": availability,
        "selected": public_selected,
        "jobs": [{**job, "command": [str(item) for item in job["command"]]} for job in jobs],
        "comparison": "same-input paired replay; bucketed by server-tokenized Full prompt length",
        "timing": {
            "primary": "native_ttft_ms",
            "secondary": "controller_prepare_ms + native_ttft_ms",
            "radix_cache": "disabled",
        },
    }
    save(args.out / "selection_manifest.json", manifest)
    if not args.execute:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    execute_jobs(jobs)
    print(json.dumps(summarize(args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
