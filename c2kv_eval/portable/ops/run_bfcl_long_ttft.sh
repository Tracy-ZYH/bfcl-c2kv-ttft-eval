#!/usr/bin/env bash
set -euo pipefail
exec /home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python /home/zhuyuhan/project/bfcl-c2kv-ttft-eval/c2kv_eval/portable/ops/bfcl_long_ttft.py "$@"
