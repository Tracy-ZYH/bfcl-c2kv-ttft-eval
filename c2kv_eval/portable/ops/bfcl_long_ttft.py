"""BFCL Long speed suite. No model calls without --execute."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

PROJECT = Path("/home/zhuyuhan/project")
REPO = Path(__file__).resolve().parents[3]
ARMS = {"full_cold": "full", "full_cached": "full", "c2kv_only": "c2kv_only", "racer": "proposed"}
STOP = threading.Event()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='2,3', help='Physical nvidia-smi indices; execution resolves UUIDs')
    p.add_argument('--cases', type=int, default=2)
    p.add_argument('--arm-per-gpu', action='store_true',
                   help='Bind one arm to each GPU; every arm runs the full selected task set')
    p.add_argument('--arms', nargs='+', choices=tuple(ARMS),
                   help='Subset of arms to run; default runs all arms')
    p.add_argument('--run-ids', nargs='+',
                   help='Exact official task IDs for targeted regression; preserves supplied order')
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--execute', action='store_true', help='Required to query GPUs/start servers/run BFCL')
    p.add_argument('--summarize', type=Path)
    p.add_argument('--out', type=Path, default=PROJECT / 'bfcl-c2kv/results' /
                   ('bfcl_long_ttft_gpu23_' + time.strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--c2kv-root', type=Path, default=PROJECT / 'c2kv-retrieval-quality')
    p.add_argument('--sglang-root', type=Path, default=PROJECT / 'kvoffload-sglang-c2kv-ttft')
    p.add_argument('--bfcl-root', type=Path, default=REPO)
    p.add_argument('--checkpoint', type=Path,
                   default=Path('/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000'))
    p.add_argument('--embedding-model', type=Path,
                   default=Path('/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B'))
    p.add_argument('--server-python', type=Path,
                   default=Path('/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python'))
    p.add_argument('--client-python', type=Path,
                   default=Path('/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python'))
    p.add_argument('--port-base', type=int, default=39420)
    p.add_argument('--task-timeout', type=int, default=3600)
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument(
        '--max-tasks-per-server', type=int, default=40,
        help='Recycle SGLang after this many tasks; startup time is excluded from metrics',
    )
    p.add_argument('--min-free-mib', type=int, default=75000)
    return p


def main():
    args = parser().parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ttft_plan import build_plan, validate_profiles
    from ttft_summary import summarize
    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
        return 0
    plan = build_plan(args)
    plan['profiles'] = validate_profiles(args.out, args.port_base)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    from ttft_suite import execute
    execute(args, plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
