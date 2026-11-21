"""CPU-only planning and frozen-profile validation."""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from ttft_commands import (ARMS, C2KV, C2KV_COMPRESSION_RATIO, C2KV_HISTORY_BUDGET_BYTES, C2KV_HISTORY_BUDGET_TOKENS, EVAL_POLICY, REPO, task_command, server_command)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_plan(args):
    devices = [int(x) for x in args.gpus.split(',')]
    if not devices or min(devices) < 0 or len(devices) != len(set(devices)):
        raise ValueError('Use unique physical nvidia-smi GPU indices')
    path = REPO / 'bfcl_eval/data/BFCL_v4_multi_turn_long_context.json'
    official_ids = [row['id'] for row in rows(path)]
    if len(official_ids) != 200 or len(set(official_ids)) != 200:
        raise ValueError('Expected official fixed 200-example BFCL Long category')
    if any(not task.startswith('multi_turn_long_context_') for task in official_ids):
        raise ValueError('Wrong category')
    requested = getattr(args, 'run_ids', None)
    if requested:
        if len(requested) != len(set(requested)) or any(task not in official_ids for task in requested):
            raise ValueError('run-ids must be unique official BFCL Long task IDs')
        ids = list(requested)
    else:
        if not 1 <= args.cases <= 200:
            raise ValueError('cases must be within the official 200-example category')
        ids = official_ids[:args.cases]
    arm_names = list(getattr(args, 'arms', None) or ARMS)
    max_tasks_per_server = int(getattr(args, 'max_tasks_per_server', 40))
    if max_tasks_per_server < 1:
        raise ValueError('max-tasks-per-server must be positive')
    arm_count = len(arm_names)
    arm_per_gpu = bool(getattr(args, 'arm_per_gpu', False))
    if arm_per_gpu and len(devices) != arm_count:
        raise ValueError('arm-per-gpu requires exactly one physical GPU per selected arm')
    device_stride = args.repetitions * (2 if arm_per_gpu else arm_count * 2)
    last_port = args.port_base + len(devices) * device_stride - 1
    if args.repetitions < 1 or not 1024 <= args.port_base or last_port > 65535:
        raise ValueError('Invalid repetition or port range')
    jobs = []

    def add_jobs(gpu, repetition, arm, port, tasks, aggregate_output):
        chunks = [
            tasks[start:start + max_tasks_per_server]
            for start in range(0, len(tasks), max_tasks_per_server)
        ]
        for chunk_index, chunk in enumerate(chunks):
            output = (
                aggregate_output
                if len(chunks) == 1
                else aggregate_output / 'chunks' / f'chunk{chunk_index:03d}'
            )
            jobs.append(dict(
                gpu=gpu, repetition=repetition, arm=arm, port=port, tasks=chunk,
                output=str(output), aggregate_output=str(aggregate_output),
                chunk_index=chunk_index, chunk_count=len(chunks),
                server_command=server_command(
                    arm, port, getattr(args, 'task_timeout', 3600)
                ),
            ))

    if arm_per_gpu:
        for slot, (gpu, arm) in enumerate(zip(devices, arm_names)):
            for rep in range(args.repetitions):
                port = args.port_base + slot * device_stride + 2 * rep
                out = args.out / f'gpu{gpu}' / f'rep{rep}' / arm
                add_jobs(gpu, rep, arm, port, list(ids), out)
    else:
        for slot, gpu in enumerate(devices):
            tasks = ids[slot::len(devices)]
            if not tasks:
                continue
            for rep in range(args.repetitions):
                names = list(arm_names)
                names = names[rep % arm_count:] + names[:rep % arm_count]
                for stage, arm in enumerate(names):
                    port = args.port_base + slot * device_stride + 2 * (rep * arm_count + stage)
                    out = args.out / f'gpu{gpu}' / f'rep{rep}' / arm
                    add_jobs(gpu, rep, arm, port, tasks, out)
    return dict(schema='bfcl-long-ttft-v1', task_ids=ids, jobs=jobs,
                task_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                generation=dict(temperature=0, seed=0, max_new_tokens=4096, concurrency_per_gpu=1),
                c2kv=dict(compression_ratio=C2KV_COMPRESSION_RATIO, history_kv_budget_tokens=C2KV_HISTORY_BUDGET_TOKENS, history_kv_budget_bytes=C2KV_HISTORY_BUDGET_BYTES, eval_policy=str(EVAL_POLICY)),
                comparison='Closed-loop paired tasks, NOT same-input causal replay',
                retrieval_quality='not_assessed_no_gold_relevance_labels',
                max_tasks_per_server=max_tasks_per_server)


def validate_profiles(out, port):
    sys.path.insert(0, str(BFCL / 'runtime/history_system'))
    c1 = importlib.import_module('run_c1')
    profiles = {}
    for arm in ARMS:
        cmd = task_command(arm, ['multi_turn_long_context_0'], out / 'preview' / arm, port)
        args = c1.build_parser().parse_args(cmd[2:])
        c1.validate_args(args)
        _, profile = c1.build_profile(args)
        # Use the already validated controller file only through CPU preview.
        server, worker = c1.commands_for_task(args, 'multi_turn_long_context_0', out / 'preview' / arm / 'controller.json')
        profiles[arm] = profile | dict(controller_command=server, worker_command=worker)
    return profiles
