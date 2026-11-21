"""Descriptive latency summary; no fabricated retrieval-quality PASS."""
import csv
import json
from ttft_commands import ARMS
from ttft_collect import collect
from ttft_plan import save


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(root):
    plan = json.loads((root / 'plan.json').read_text())
    tasks, calls, decisions = collect(plan)
    arm_names = list(dict.fromkeys(job['arm'] for job in plan['jobs']))
    summary = []
    for arm in arm_names:
        ts = [r for r in tasks if r['arm'] == arm]
        cs = [r for r in calls if r['arm'] == arm]
        ds = [r for r in decisions if r['arm'] == arm]
        if not ts or not cs or not ds:
            raise ValueError(f'Incomplete arm: {arm}')
        row = dict(method=arm, official_correct_count=int(sum(r['official_score'] for r in ts)),
                   official_total_count=len(ts), native_generation_calls=len(cs), committed_steps=len(ds),
                   detector_calls=sum(r.get('detector_calls', 0) for r in ts),
                   recovery_count=sum(r.get('recovery_count', 0) for r in ts))
        row['bfcl_accuracy'] = row['official_correct_count'] / len(ts)
        for key, source in [('native_ttft_ms', cs), ('prefill_path_ms', cs),
                            ('native_pre_generation_ms', cs), ('final_action_ready_ms', ds)]:
            for label, q in [('p50', .5), ('p90', .9), ('p95', .95)]:
                row[f'{key}_{label}'] = quantile([r[key] for r in source], q)
        summary.append(row)
    full = next((row for row in summary if row['method'] == 'full_cold'), None)
    for row in summary:
        if full is None:
            row['model_prefill_p50_reduction_vs_full_cold'] = None
            row['injection_pipeline_ttft_p50_reduction_vs_full_cold'] = None
            row['accuracy_relative_drop_vs_full_cold'] = None
        else:
            if full['prefill_path_ms_p50'] <= 0 or full['native_ttft_ms_p50'] <= 0:
                raise ValueError('Invalid Full timing')
            row['model_prefill_p50_reduction_vs_full_cold'] = 1 - row['prefill_path_ms_p50'] / full['prefill_path_ms_p50']
            row['injection_pipeline_ttft_p50_reduction_vs_full_cold'] = 1 - row['native_ttft_ms_p50'] / full['native_ttft_ms_p50']
            row['accuracy_relative_drop_vs_full_cold'] = (
                1 - row['bfcl_accuracy'] / full['bfcl_accuracy']
                if full['bfcl_accuracy'] else None
            )
        row['claim_status'] = 'closed_loop_observation; same_input_replay_and_gold_relevance_still_required'
    for name, value in [('tasks', tasks), ('calls', calls), ('decisions', decisions), ('summary', summary)]:
        save(root / f'{name}.json', value)
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary
