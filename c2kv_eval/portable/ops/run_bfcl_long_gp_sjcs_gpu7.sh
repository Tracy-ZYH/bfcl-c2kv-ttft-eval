#!/usr/bin/env bash
# BFCL Long GP-SJCS on one physical GPU. Usage: smoke (20 held-out tasks) or full (200).
set -Eeuo pipefail

PHASE="${1:-smoke}"
[[ "${PHASE}" == "smoke" || "${PHASE}" == "full" ]] || {
  echo "usage: $0 smoke|full" >&2
  exit 2
}

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export PYTHONHASHSEED=42
export HISTORY_KV_TARGET_TOKENS=""
export GP_SJCS_HISTORY_BUDGET="${GP_SJCS_HISTORY_BUDGET:-256}"
export GP_SJCS_BETA="${GP_SJCS_BETA:-0.25}"

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/third_party/sglang}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
CUDA_HOME_FOR_JIT="${CUDA_HOME_FOR_JIT:-/usr/local/cuda-13.0}"
HISTORICAL="${HISTORICAL:-/home/zhuyuhan/project/bfcl-c2kv/results/bfcl_long_ttft_by_length_20261030_000701/paired_decisions.csv}"
GPU="${GPU:-7}"
SERVER_PORT="${SERVER_PORT:-39970}"
PROXY_PORT_BASE="${PROXY_PORT_BASE:-39980}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.65}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.05}"
CASES=20
[[ "${PHASE}" == "full" ]] && CASES=200
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bfcl_long_gp_sjcs_${PHASE}_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
DATA="${ROOT}/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
FIT="${ROOT}/c2kv_eval/portable/ops/fit_gp_sjcs.py"
SUMMARIZE="${ROOT}/c2kv_eval/portable/ops/summarize_gp_sjcs.py"

for required in "${MODEL_PATH}/config.json" "${HISTORICAL}" "${DATA}" "${FIT}" "${SUMMARIZE}"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 2; }
done
[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
if ss -ltn "sport = :${SERVER_PORT}" | grep -q LISTEN; then
  echo "server port ${SERVER_PORT} is already in use" >&2
  exit 2
fi
for offset in 0 1 2 3; do
  port=$((PROXY_PORT_BASE + offset))
  if ss -ltn "sport = :${port}" | grep -q LISTEN; then
    echo "proxy port ${port} is already in use" >&2
    exit 2
  fi
done
mkdir -p "${RESULT_ROOT}/artifacts" "${RESULT_ROOT}/server" "${RESULT_ROOT}/runs" "${RESULT_ROOT}/summary"
export PYTHONPATH="${ROOT}:${SGLANG_ROOT}/python:${PYTHONPATH:-}"

GP_SJCS_ARTIFACT="${RESULT_ROOT}/artifacts/gp_sjcs.json"
export GP_SJCS_ARTIFACT
"${BFCL_PYTHON}" "${FIT}"   --historical-paired "${HISTORICAL}"   --out "${GP_SJCS_ARTIFACT}"   --log "${RESULT_ROOT}/artifacts/fit_log.json"   --history-budget "${GP_SJCS_HISTORY_BUDGET}" --seed 42   2>&1 | tee "${RESULT_ROOT}/artifacts/fit.log"

RUN_IDS="$("${BFCL_PYTHON}" - "${DATA}" "${GP_SJCS_ARTIFACT}" "${PHASE}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
artifact = json.load(open(sys.argv[2], encoding="utf-8"))
phase = sys.argv[3]
official = [str(row["id"]) for row in rows]
if len(official) != 200 or len(set(official)) != 200:
    raise SystemExit("BFCL Long must contain exactly 200 unique official IDs")
if phase == "smoke":
    calibration = set(map(str, artifact["calibration_task_ids"]))
    selected = [task for task in official if task not in calibration][:20]
    if len(selected) != 20:
        raise SystemExit("could not choose 20 held-out smoke tasks")
else:
    selected = official
print(",".join(selected))
PY
)"
printf '%s\n' "${RUN_IDS//,/$'\n'}" > "${RESULT_ROOT}/task_ids.txt"

SERVER_PID=""
cleanup() {
  status=$?
  set +e
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    pgid="$(ps -o pgid= -p "${SERVER_PID}" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "${pgid}" && "${pgid}" == "${SERVER_PID}" ]] &&
      kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      [[ -n "${pgid}" && "${pgid}" == "${SERVER_PID}" ]] &&
        kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  echo "results: ${RESULT_ROOT}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo "starting SGLang on physical GPU ${GPU}"
setsid env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}"   CUDA_HOME="${CUDA_HOME_FOR_JIT}" CUDACXX="${CUDA_HOME_FOR_JIT}/bin/nvcc"   PATH="${CUDA_HOME_FOR_JIT}/bin:${PATH}" CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12   NVCC_CCBIN=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 MAX_JOBS=1   CMAKE_BUILD_PARALLEL_LEVEL=1 PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH}"   "${SGLANG_PYTHON}" -m sglang.launch_server     --model-path "${MODEL_PATH}" --tokenizer-path "${MODEL_PATH}"     --served-model-name c2kv-agent --host 127.0.0.1 --port "${SERVER_PORT}"     --dtype bfloat16 --tool-call-parser qwen25     --enable-c2kv --enable-streaming-session --c2kv-gist-type dynamic-interleave     --c2kv-gist-param qkv --c2kv-query-proj base --c2kv-tools-dump full     --disable-radix-cache --page-size 1 --max-running-requests 1     --attention-backend triton --decode-attention-backend triton     --prefill-attention-backend triton --sampling-backend pytorch     --disable-cuda-graph --disable-piecewise-cuda-graph     --mem-fraction-static "${MEM_FRACTION_STATIC}"     --c2kv-pool-fraction "${C2KV_POOL_FRACTION}"     --context-length 262144 --max-total-tokens 262144     --chunked-prefill-size 4096 --random-seed 42     --watchdog-timeout "$((TASK_TIMEOUT + 300))"     >"${RESULT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    echo "server exited; see ${RESULT_ROOT}/server/server.log" >&2
    exit 1
  }
  (( SECONDS < deadline )) || {
    echo "server startup timeout; see ${RESULT_ROOT}/server/server.log" >&2
    exit 1
  }
  sleep 2
done

# One untimed first-token request removes model/kernel initialization skew from Full.
curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions"   -H 'Content-Type: application/json'   -d '{"model":"c2kv-agent","messages":[{"role":"user","content":"Reply OK."}],"temperature":0,"top_p":1,"seed":42,"max_tokens":1}'   >"${RESULT_ROOT}/server/warmup.json"
curl --noproxy '*' -fsS -X POST   "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null

run_arm() {
  arm="$1"
  offset="$2"
  out="${RESULT_ROOT}/runs/${arm}"
  echo "[GPU ${GPU}] arm=${arm}, phase=${PHASE}, cases=${CASES}"
  env no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY=''     GP_SJCS_ARTIFACT="${GP_SJCS_ARTIFACT}"     GP_SJCS_HISTORY_BUDGET="${GP_SJCS_HISTORY_BUDGET}" GP_SJCS_BETA="${GP_SJCS_BETA}"     "${BFCL_PYTHON}" -m c2kv_eval.portable.run       --benchmark bfcl --categories multi_turn_long_context --arm "${arm}"       --upstream "http://127.0.0.1:${SERVER_PORT}" --model c2kv-agent       --proxy-port "$((PROXY_PORT_BASE + offset))"       --proxy-python "${SGLANG_PYTHON}"       --bfcl-dir "${ROOT}" --run-ids "${RUN_IDS}"       --num-workers 1 --upstream-timeout "${TASK_TIMEOUT}"       --allow-unprofiled --doc-packing turn --max-doc-length 1024 --max-doc-num 48       --tokenizer "${MODEL_PATH}" --out "${out}" --exact-out       --run-name "bfcl_long_${arm}"       2>&1 | tee "${RESULT_ROOT}/runs/${arm}.launcher.log"
}

arms=(full fixed20k_joint gp_length_only gp_sjcs)
for index in "${!arms[@]}"; do
  (( index == 0 )) || curl --noproxy '*' -fsS -X POST     "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null
  run_arm "${arms[$index]}" "${index}"
done

"${BFCL_PYTHON}" "${SUMMARIZE}" "${RESULT_ROOT}"   --expected "${CASES}" --artifact "${GP_SJCS_ARTIFACT}"   2>&1 | tee "${RESULT_ROOT}/summary/summarize.log"
echo "completed: ${RESULT_ROOT}"
