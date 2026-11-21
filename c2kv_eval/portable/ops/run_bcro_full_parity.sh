#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
C1_ROOT="${C1_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/runtime}"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/third_party/sglang}"
MODEL="${MODEL:-/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000}"
EMBED="${EMBED:-/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B}"
BFCL_PY="${BFCL_PY:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
SGLANG_PY="${SGLANG_PY:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
GPU=0
SERVER_PORT=45720
PROXY_PORT=45820
CONTROLLER_PORT=45920
TASK_ID=multi_turn_long_context_24
RESULT_ROOT=""
TASK_TIMEOUT=3600

usage() {
  echo "Usage: $0 --gpu ID_OR_UUID [--server-port N] [--proxy-port N] [--controller-port N] [--task-id ID] [--out DIR] [--model PATH]"
}
while (( $# )); do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --proxy-port) PROXY_PORT="$2"; shift 2 ;;
    --controller-port) CONTROLLER_PORT="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --out) RESULT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bcro_full_parity_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
POLICY="${C1_ROOT}/runtime/configs/eval_policy_bfcl_long_fixed256.json"
CAPACITY="${C1_ROOT}/runtime/configs/eval_capacity_bfcl_long.json"
ARTIFACT="${ROOT}/results/bcro_stage_a2_20261111_022630/seed_0/bcro_artifact.json"
CLIENT_PATH="${ROOT}:${ROOT}/runtime/history_system:${PYTHONPATH:-}"

[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
for port in "${SERVER_PORT}" "${PROXY_PORT}" "${CONTROLLER_PORT}"; do
  ! ss -ltn "sport = :${port}" | grep -q LISTEN || { echo "port ${port} in use" >&2; exit 2; }
done
# run_c1 creates this directory itself.
mkdir -p "${RESULT_ROOT}/server" "${RESULT_ROOT}/portable"

SERVER_PID=""
cleanup() {
  status=$?
  set +e
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -0 "${SERVER_PID}" 2>/dev/null && kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  echo "Parity results: ${RESULT_ROOT}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

export PYTHONHASHSEED=0 NO_PROXY='*' no_proxy='*'
env PYTHONPATH="${CLIENT_PATH}" "${BFCL_PY}" -c '
from c2kv_eval.portable.backends.sglang import SglangBackend
from runtime.benchmarks.memory_runtime.event_native_bcrs import _portable_full_payload
print("IMPORT_PASS")
' >"${RESULT_ROOT}/import_preflight.log" 2>&1

unset CUDA_VISIBLE_DEVICES
setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
  CUDA_HOME=/usr/local/cuda-13.0 CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
  PATH="/usr/local/cuda-13.0/bin:${PATH}" CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
  NVCC_CCBIN=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 MAX_JOBS=1 \
  PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}" \
  "${SGLANG_PY}" -m sglang.launch_server \
    --model-path "${MODEL}" --tokenizer-path "${MODEL}" --served-model-name c2kv-agent \
    --host 127.0.0.1 --port "${SERVER_PORT}" --dtype bfloat16 \
    --tool-call-parser qwen25 --enable-c2kv --enable-streaming-session \
    --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
    --c2kv-query-proj base --c2kv-tools-dump full --enable-return-hidden-states \
    --c2kv-shadow-feature-layer -2 --disable-radix-cache --page-size 1 \
    --max-running-requests 1 --attention-backend triton \
    --decode-attention-backend triton --prefill-attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph --disable-piecewise-cuda-graph \
    --mem-fraction-static 0.65 --c2kv-pool-fraction 0.15 \
    --context-length 262144 --max-total-tokens 262144 --chunked-prefill-size 4096 \
    --random-seed 0 --watchdog-timeout "$((TASK_TIMEOUT + 300))" \
    >"${RESULT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 1200))
until curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server exited; see server/server.log" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "server startup timeout" >&2; exit 1; }
  sleep 2
done

env PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${BFCL_PY}" -m c2kv_eval.portable.run \
  --benchmark bfcl --categories multi_turn_long_context --arm full \
  --upstream "http://127.0.0.1:${SERVER_PORT}" --model c2kv-agent \
  --proxy-port "${PROXY_PORT}" --proxy-python "${SGLANG_PY}" \
  --bfcl-dir "${ROOT}" --run-ids "${TASK_ID}" --num-workers 1 \
  --upstream-timeout "${TASK_TIMEOUT}" --allow-unprofiled --doc-packing turn \
  --max-doc-length 1024 --max-doc-num 48 --tokenizer "${MODEL}" --parity-debug \
  --out "${RESULT_ROOT}/portable" --exact-out --run-name parity-portable-full \
  >"${RESULT_ROOT}/portable/launcher.log" 2>&1

set +e
env PYTHONPATH="${CLIENT_PATH}" "${BFCL_PY}" "${C1_ROOT}/runtime/history_system/run_c1.py" \
  --benchmark bfcl --method proposed --checkpoint "${MODEL}" \
  --sglang-backend-url "http://127.0.0.1:${SERVER_PORT}" \
  --embedding-model "${EMBED}" --embedding-device cpu --detector t02_risk \
  --selector-threshold 0.5 --eval-policy-override "${POLICY}" \
  --eval-capacity-override "${CAPACITY}" --benchmark-dir "${ROOT}" \
  --bfcl-python "${BFCL_PY}" --portable-root "${ROOT}" --sglang-root "${SGLANG_ROOT}" \
  --port "${CONTROLLER_PORT}" --task-timeout "${TASK_TIMEOUT}" \
  --out "${RESULT_ROOT}/bcro" --task-id "${TASK_ID}" \
  --bcrs-artifact "${ARTIFACT}" --bcrs-mode bcro_racer --bcrs-python-root "${ROOT}" \
  >"${RESULT_ROOT}/bcro_launcher.log" 2>&1
BCRO_STATUS=$?
set -e
echo "${BCRO_STATUS}" >"${RESULT_ROOT}/bcro_exit_status.txt"

set +e
"${BFCL_PY}" - "${RESULT_ROOT}" "${TASK_ID}" "${BCRO_STATUS}" <<'PY' >"${RESULT_ROOT}/parity_check.log" 2>&1
import json, pathlib, sys
root, task, bcro_status = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
logs = sorted((root / "portable" / "logs").glob("proxy_*.jsonl"))
assert len(logs) == 1, logs
portable = None
for line in logs[0].open(encoding="utf-8"):
    row = json.loads(line); ctx = row.get("eval_context") or {}
    if ctx.get("task_id") == task and ctx.get("user_turn") == 0 and ctx.get("step") == 0 and row.get("status") == "ok":
        portable = row; break
assert portable is not None, "portable first decision missing"
bcro = None
for path in sorted((root / "bcro").glob("task_shards/*/server/steps.jsonl")):
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("schema") == "a-ordinary-full-step-v1" and row.get("status") == "ok":
            bcro = row; break
    if bcro is not None: break
assert bcro is not None, "BCRO ordinary-Full first decision missing"
pp = portable.get("parity_debug") or {}
bp = bcro.get("ordinary_full_path") or {}
bdebug = (((bcro.get("generation_trace") or [{}])[0].get("generation") or {}).get("stats") or {}).get("server_token_parity") or {}
pt = (portable.get("native_tool_names") or [None])[0]
calls = (bcro.get("response") or {}).get("tool_calls") or []
bt = (calls[0].get("function") or {}).get("name") if calls else None
sampling = {k:v for k,v in (bp.get("sampling") or {}).items() if v is not None}
pids, bids = pp.get("prompt_ids") or [], bdebug.get("prompt_ids") or []
first_divergence = None
if pids and bids and pids != bids:
    limit = min(len(pids), len(bids))
    index = next((i for i in range(limit) if pids[i] != bids[i]), limit)
    first_divergence = {
        "index": index,
        "portable_window": pids[max(0,index-12):index+24],
        "bcro_window": bids[max(0,index-12):index+24],
        "portable_remaining": len(pids) - index,
        "bcro_remaining": len(bids) - index,
    }
checks = {
 "token_hash_equal": bool(pp.get("prompt_ids_sha256")) and pp.get("prompt_ids_sha256") == bp.get("server_prompt_ids_sha256"),
 "prompt_count_equal": (portable.get("usage") or {}).get("prompt_tokens") == bp.get("server_prompt_token_count"),
 "first_tool_equal": pt == bt,
 "temperature_zero": sampling.get("temperature") == 0.0,
 "max_completion_tokens_4096": sampling.get("max_completion_tokens") == 4096,
}
report = {
 "status": "PASS" if all(checks.values()) else "FAIL", "task_id": task,
 "bcro_process_exit_status": bcro_status, "checks": checks,
 "portable": {"prompt_hash": pp.get("prompt_ids_sha256"), "prompt_tokens": (portable.get("usage") or {}).get("prompt_tokens"), "first_tool": pt},
 "bcro": {"prompt_hash": bp.get("server_prompt_ids_sha256"), "prompt_tokens": bp.get("server_prompt_token_count"), "sampling": sampling, "first_tool": bt},
 "template_input": {
   "messages_hash_equal": bool(pp.get("chat_template_messages_sha256")) and pp.get("chat_template_messages_sha256") == bdebug.get("chat_template_messages_sha256"),
   "tools_hash_equal": bool(pp.get("chat_template_tools_sha256")) and pp.get("chat_template_tools_sha256") == bdebug.get("chat_template_tools_sha256"),
   "portable_content_format": pp.get("jinja_template_content_format"),
   "bcro_content_format": bdebug.get("jinja_template_content_format"),
   "portable_template_name": pp.get("chat_template_name"),
   "bcro_template_name": bdebug.get("chat_template_name"),
 },
 "first_prompt_divergence": first_divergence,
}
(root/"parity_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=False))
if report["status"] != "PASS": raise SystemExit("FULL_PATH_PARITY_FAIL")
print("FULL_PATH_PARITY_PASS")
PY
PARITY_STATUS=$?
set -e
cat "${RESULT_ROOT}/parity_check.log"
(( PARITY_STATUS == 0 )) || exit "${PARITY_STATUS}"
(( BCRO_STATUS == 0 )) || { echo "Parity passed but BCRO task failed; see bcro_launcher.log" >&2; exit 3; }
echo "completed: ${RESULT_ROOT}"

