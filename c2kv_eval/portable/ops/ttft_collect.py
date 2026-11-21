"""Collect official outcomes; incomplete timing fails closed."""
import json
import math
from pathlib import Path
from ttft_plan import rows


def positive(row, key):
    value = row.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Missing/invalid timing: {key}')
    return value


def collect(plan):
    tasks, calls, decisions = [], [], []
    for job in plan['jobs']:
        run = Path(job['output']) / 'run'
        result = json.loads((run / 'result.json').read_text())
        if result.get('status') != 'completed' or [r['task_id'] for r in result['tasks']] != job['tasks']:
            raise ValueError(f'Incomplete or misaligned result: {run}')
        for task in result['tasks']:
            row = task['unified_metrics']
            if row['official_score'] not in (0, 1):
                raise ValueError('Expected binary official BFCL task accuracy')
            common = dict(arm=job['arm'], gpu=job['gpu'], repetition=job['repetition'], task_id=task['task_id'])
            tasks.append(row | common)
            server = run / 'task_shards' / task['task_id'] / 'server'
            found = []
            for record in rows(server / 'steps.jsonl'):
                for trace in record.get('generation_trace', []):
                    stats = (trace.get('generation') or {}).get('stats', {})
                    if stats.get('backend') == 'sglang_c2kv_native_packed':
                        timing = stats.get('speed_timing', {})
                        for key in ('native_ttft_ms', 'prefill_path_ms', 'native_pre_generation_ms'):
                            positive(timing, key)
                        found.append(common | timing | dict(phase=trace.get('phase'), source_tokens=stats.get('source_tokens')))
            if not found or len(found) != row['native_generate_requests']:
                raise ValueError(f'Incomplete native-call timing: {server}')
            calls.extend(found)
            latency = rows(server / 'decision_latency.jsonl')
            if not latency:
                raise ValueError(f'Missing final-action timing: {server}')
            for entry in latency:
                positive(entry, 'final_action_ready_ms')
                decisions.append(entry | common)
    return tasks, calls, decisions
