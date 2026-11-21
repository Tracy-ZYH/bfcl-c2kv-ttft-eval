#!/usr/bin/env python3
"""BFCL Long same-input Full/C2KV/Racer cache-hit TTFT experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import toolsandbox_cache_hit_ttft as core


DATA = core.BFCL / "bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"


def official_ids() -> list[str]:
    ids = [row.get("id") for row in core.rows(DATA)]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("expected the fixed 200-example BFCL Long category")
    if any(
        not isinstance(task, str)
        or not task.startswith("multi_turn_long_context_")
        for task in ids
    ):
        raise ValueError("invalid BFCL Long task IDs")
    return ids


def selected_tasks(args) -> list[str]:
    ids = official_ids()
    if args.run_ids:
        if len(args.run_ids) != len(set(args.run_ids)) or any(
            task not in ids for task in args.run_ids
        ):
            raise ValueError("--run-ids must be unique official BFCL Long IDs")
        return list(args.run_ids)
    return ids[: args.quick_cases] if args.phase == "quick" else ids


def run_c1_cli(args, method, tasks, out, port):
    command = [
        str(args.client_python),
        str(core.C2KV / "runtime/history_system/run_c1.py"),
        "--benchmark",
        "bfcl",
        "--method",
        method,
        "--checkpoint",
        str(args.model),
        "--sglang-backend-url",
        f"http://127.0.0.1:{args.server_port}",
        "--embedding-model",
        str(args.embedding_model),
        "--embedding-device",
        "cpu",
        "--detector",
        "t02_risk",
        "--selector-threshold",
        "0.5",
        "--eval-policy-override",
        str(args.eval_policy),
        "--eval-capacity-override",
        str(args.eval_capacity),
        "--benchmark-dir",
        str(core.BFCL),
        "--bfcl-python",
        str(args.client_python),
        "--portable-root",
        str(core.BFCL),
        "--sglang-root",
        str(core.SGLANG),
        "--port",
        str(port),
        "--task-timeout",
        str(args.task_timeout),
        "--out",
        str(out),
    ]
    for task in tasks:
        command += ["--task-id", task]
    return command


def capture_inputs(args, tasks, root, env):
    capture = root / "capture_full"
    capture_log = root / "capture_full.log"
    if not (capture / "result.json").exists():
        command = run_c1_cli(args, "full", tasks, capture, args.controller_port)
        with capture_log.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=core.C2KV,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            raise RuntimeError(f"Full BFCL capture failed; see {capture_log}")

    decisions = {}
    for task in tasks:
        path = capture / "task_shards" / task / "server" / "steps.jsonl"
        picked = []
        for record in core.rows(path):
            payload = (
                record.get("harness_transport_normalization") or {}
            ).get("same_input_replay_request")
            if not isinstance(payload, dict):
                raise RuntimeError(f"missing same-input payload in {path}")
            messages = payload.get("messages") or []
            if sum(message.get("role") == "user" for message in messages) >= 2:
                picked.append(payload)
        if picked:
            decisions[task] = picked
    if not decisions:
        raise RuntimeError(
            "BFCL Full capture produced no decision with completed history"
        )

    core.save(
        root / "decisions.json",
        {
            "schema": "bfcl-long-same-input-decisions-v1",
            # The shared core validates this compatibility key.
            "scenarios": tasks,
            "tasks": tasks,
            "decisions": decisions,
            "task_ids_sha256": hashlib.sha256(
                json.dumps(tasks, separators=(",", ":")).encode()
            ).hexdigest(),
            "decision_count": sum(map(len, decisions.values())),
        },
    )
    return decisions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("quick", "formal"), default="quick")
    p.add_argument("--arm", choices=("all", "full", "c2kv", "racer"), default="all")
    p.add_argument("--decisions-file", type=Path)
    p.add_argument(
        "--warm-decisions-file", type=Path,
        help="Optional ordered per-task decisions used only for offline incremental warmup.",
    )
    p.add_argument("--decisions-wait", type=int, default=7200)
    p.add_argument("--quick-cases", type=int, choices=range(2, 21), default=10)
    p.add_argument("--run-ids", nargs="+")
    p.add_argument("--gpu", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--server-port", type=int, default=39970)
    p.add_argument("--controller-port", type=int, default=40000)
    p.add_argument("--startup-timeout", type=int, default=1200)
    p.add_argument("--task-timeout", type=int, default=3600)
    p.add_argument("--mem-fraction-static", type=float, default=0.65)
    p.add_argument("--c2kv-pool-fraction", type=float, default=0.15)
    p.add_argument("--model", type=Path, default=core.MODEL)
    p.add_argument("--embedding-model", type=Path, default=core.EMBED)
    p.add_argument("--server-python", type=Path, default=core.SERVER_PY)
    p.add_argument("--client-python", type=Path, default=core.CLIENT_PY)
    p.add_argument("--eval-policy", type=Path, default=core.POLICY)
    p.add_argument("--eval-capacity", type=Path, default=core.CAPACITY)
    return p


def main():
    args = parser().parse_args()
    required = [
        DATA,
        args.model / "config.json",
        args.embedding_model / "config.json",
        args.eval_policy,
        args.eval_capacity,
        args.server_python,
        args.client_python,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing prerequisites:\n" + "\n".join(missing))
    if args.repetitions != 3:
        raise SystemExit("the paper experiment requires exactly 3 repetitions")

    core.selected_scenarios = selected_tasks
    core.run_c1_cli = run_c1_cli
    core.capture_inputs = capture_inputs
    core.run(args)
    for name in ("manifest.json", "summary.json"):
        path = args.out / name
        payload = json.loads(path.read_text())
        payload["dataset"] = "bfcl_long"
        payload["task_count"] = payload.get("scenario_count")
        payload["schema"] = "bfcl-long-cache-hit-ttft-summary-v1" if name == "summary.json" else "bfcl-long-cache-hit-ttft-v1"
        core.save(path, payload)


if __name__ == "__main__":
    main()
