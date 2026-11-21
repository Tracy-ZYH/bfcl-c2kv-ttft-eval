"""Execution is explicit; importing this module starts no processes."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from ttft_commands import C2KV, REPO, SGLANG
from ttft_plan import save
from ttft_process import STOP, gpu_inventory
from ttft_summary import summarize
from ttft_worker import run_device


def consolidate_chunked_runs(plan):
    """Expose chunked jobs through the original one-run-per-arm layout."""
    groups = {}
    for job in plan["jobs"]:
        aggregate = job.get("aggregate_output", job["output"])
        key = (job["gpu"], job["repetition"], job["arm"], aggregate)
        groups.setdefault(key, []).append(job)
    for (_, _, _, aggregate), jobs in groups.items():
        jobs.sort(key=lambda item: item.get("chunk_index", 0))
        target_root = Path(aggregate)
        if len(jobs) == 1 and Path(jobs[0]["output"]) == target_root:
            continue
        target = target_root / "run"
        if target.exists():
            raise FileExistsError(f"Refusing existing consolidated run: {target}")
        shards = target / "task_shards"
        shards.mkdir(parents=True)
        task_rows = []
        metric_rows = []
        expected = []
        for job in jobs:
            source = Path(job["output"]) / "run"
            result = json.loads((source / "result.json").read_text())
            if result.get("status") != "completed":
                raise ValueError(f"Incomplete chunk result: {source}")
            actual = [row.get("task_id") for row in result.get("tasks", [])]
            if actual != job["tasks"]:
                raise ValueError(
                    f"Chunk task mismatch in {source}: "
                    f"expected={job['tasks']} actual={actual}"
                )
            expected.extend(job["tasks"])
            task_rows.extend(result["tasks"])
            metric_rows.extend(row["unified_metrics"] for row in result["tasks"])
            for task in job["tasks"]:
                source_shard = (source / "task_shards" / task).resolve()
                destination = shards / task
                if destination.exists() or destination.is_symlink():
                    raise ValueError(f"Duplicate consolidated task shard: {task}")
                os.symlink(source_shard, destination, target_is_directory=True)
        actual = [row["task_id"] for row in task_rows]
        if actual != expected or len(actual) != len(set(actual)):
            raise ValueError(f"Invalid consolidated task order under {target_root}")
        save(target / "result.json", {
            "schema": "c1-multibench-delivery-run-v1",
            "status": "completed",
            "tasks": task_rows,
        })
        save(target / "unified_summary.json", {
            "schema": "c1-unified-summary-v1",
            "rows": metric_rows,
        })
        with (target / "unified_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        save(target_root / "chunk_manifest.json", {
            "schema": "bfcl-long-server-chunks-v1",
            "task_count": len(expected),
            "chunk_count": len(jobs),
            "chunks": [
                {
                    "chunk_index": job.get("chunk_index", 0),
                    "tasks": job["tasks"],
                    "output": job["output"],
                }
                for job in jobs
            ],
        })


def execute(args, plan):
    if args.out.exists():
        raise FileExistsError(f'Refusing existing output: {args.out}')
    inventory = gpu_inventory(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,name,memory.free', '--format=csv,noheader,nounits'], text=True))
    devices = [inventory[i] for i in sorted({j['gpu'] for j in plan['jobs']})]
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    for gpu in devices:
        if gpu['free_mib'] < args.min_free_mib or gpu['uuid'] in busy:
            raise RuntimeError(f'Physical GPU {gpu["index"]} busy or insufficient free memory: {gpu}')
    args.out.mkdir(parents=True, exist_ok=False)
    plan.update(gpu_identities=devices, revisions={}, source_hashes={})
    for name, repo in [('c2kv', C2KV), ('sglang', SGLANG), ('bfcl', REPO)]:
        plan['revisions'][name] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        (args.out / f'{name}.diff').write_bytes(subprocess.check_output(['git', '-C', str(repo), 'diff', '--binary', 'HEAD']))
        changed = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-mo', '--exclude-standard', '-z']).decode().split('\0')
        for relative in sorted(set(changed) - {''}):
            path = repo / relative
            if path.is_file():
                plan['source_hashes'][name + '/' + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(args.out / 'plan.json', plan)
    print(f'Results: {args.out}; GPU identities: {devices}', flush=True)
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(run_device, args, gpu, [j for j in plan['jobs'] if j['gpu'] == gpu['index']]) for gpu in devices]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        consolidate_chunked_runs(plan)
        summarize(args.out)
        save(args.out / 'status.json', {'status': 'completed'})
    except BaseException as error:
        STOP.set()
        save(args.out / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    print(f'completed; results: {args.out}', flush=True)
