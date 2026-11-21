#!/usr/bin/env python3
"""Summarize Full, C2KV, Fixed-20K, and GP-Length BFCL Long controls."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from c2kv_eval.portable.ops.summarize_gp_sjcs import (
    online_ttft, pair_key, pct, read_json, request_rows, score_rows,
)
ARMS = (
    ("full", "Full Recompute"),
    ("c2kv", "C2KV History 8x"),
    ("fixed20k_joint", "Fixed 20K Router"),
    ("gp_length_only", "GP-Length-only"),
)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path); ap.add_argument('--expected',type=int,required=True)
    args=ap.parse_args(); collected={}
    for arm,label in ARMS:
        rr=args.root/'runs'/arm
        summary=read_json(rr/f'summary_{arm}.json')
        scores=score_rows(rr); requests=request_rows(summary)
        if len(scores)!=args.expected or int(summary.get('official_total_count') or 0)!=args.expected:
            raise ValueError(f'{arm}: incomplete official tasks')
        collected[arm]=(label,summary,scores,requests)
    full={pair_key(r):r for r in collected['full'][3]}
    output=[]
    for arm,_ in ARMS:
        label,summary,scores,requests=collected[arm]
        vals=[online_ttft(r) for r in requests]
        pairs=[(full[pair_key(r)],r) for r in requests if pair_key(r) in full]
        base=[online_ttft(a) for a,_ in pairs]; method=[online_ttft(b) for _,b in pairs]
        rl=summary.get('request_log_summary') or {}
        output.append({
            'arm':arm,'method':label,'tasks':len(scores),'correct':sum(scores.values()),
            'accuracy':sum(scores.values())/len(scores),'decisions':len(vals),
            'total_online_ttft_ms':sum(vals),'mean_online_ttft_ms':sum(vals)/len(vals),
            'p50_online_ttft_ms':pct(vals,.5),'p95_online_ttft_ms':pct(vals,.95),
            'paired_decisions':len(pairs),'paired_total_reduction_vs_full':(
                1-sum(method)/sum(base) if base and sum(base)>0 else None),
            'selected_config_counts':json.dumps(rl.get('gp_selected_config_counts') or {},sort_keys=True),
            'tool_retention':rl.get('tool_retention'),'history_retention':rl.get('history_retention'),
            'e2e_kv_retention':rl.get('e2e_kv_retention'),
            'e2e_kv_compression_ratio':rl.get('e2e_kv_compression_ratio'),
        })
    out=args.root/'summary'; out.mkdir(parents=True,exist_ok=True)
    with (out/'non_gp_sjcs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    (out/'non_gp_sjcs.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(output,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
