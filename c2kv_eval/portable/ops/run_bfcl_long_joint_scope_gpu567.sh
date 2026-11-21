#!/usr/bin/env bash
# BFCL Long 200: Tool-only, History-only and Tool+History SnapKV ablations
# on physical GPUs 5/6/7. Full is reused from the immutable 2026-11-09 run.
set -Eeuo pipefail

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export PYTHONHASHSEED=0
# Do not inherit the older fixed-256 override; these arms use 25% retention.
export HISTORY_KV_TARGET_TOKENS=""

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/third_party/sglang}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
CUDA_HOME_FOR_JIT="${CUDA_HOME_FOR_JIT:-/usr/local/cuda-13.0}"
CC_BIN="${CC_BIN:-/usr/bin/gcc-12}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-12}"
CASES="${CASES:-200}"
CATEGORY=multi_turn_long_context
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bfcl_long_joint_scope_gpu567_$(date +%Y%m%d_%H%M%S)}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.05}"
# BFCL Long contains canonical prompts above the checkpoint's nominal 262144
# metadata limit (observed task128 input reaches 358374 tokens). RoPE is position-computed, so serving
# can use an explicit larger bound without truncating official history/tools.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-393216}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-393216}"
PORTS=("${PORT_GPU5:-39475}" "${PORT_GPU6:-39476}" "${PORT_GPU7:-39477}")
GPUS=(5 6 7)
PROXY_PORT_BASE="${PROXY_PORT_BASE:-39570}"
FULL_SUMMARY="${FULL_SUMMARY:-${ROOT}/results/bfcl_long_joint_scope_gpu6_full200_20261109_230337/runs/full/summary_full.json}"
DATA="${ROOT}/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
SUMMARY="${ROOT}/c2kv_eval/portable/ops/summarize_bfcl_long_joint_scope.py"

[[ -d "${SGLANG_ROOT}/python/sglang" ]] || { echo "missing SGLang fork: ${SGLANG_ROOT}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${DATA}" ]] || { echo "missing BFCL Long data: ${DATA}" >&2; exit 2; }
[[ -f "${FULL_SUMMARY}" ]] || { echo "missing frozen Full summary: ${FULL_SUMMARY}" >&2; exit 2; }
[[ -x "${BFCL_PYTHON}" && -x "${SGLANG_PYTHON}" ]] || { echo "missing benchmark/serving Python" >&2; exit 2; }
[[ -x "${CC_BIN}" && -x "${CXX_BIN}" ]] || { echo "gcc-12/g++-12 required" >&2; exit 2; }
[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
mkdir -p "${RESULT_ROOT}/servers" "${RESULT_ROOT}/runs" "${RESULT_ROOT}/summary"

RUN_IDS_OVERRIDE="${RUN_IDS_OVERRIDE:-}"
RUN_IDS="$(${BFCL_PYTHON} - "${DATA}" "${CASES}" "${RUN_IDS_OVERRIDE}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
count = int(sys.argv[2])
override = [item.strip() for item in sys.argv[3].split(',') if item.strip()]
official = [str(row['id']) for row in rows]
if override:
    ids = override
    unknown = sorted(set(ids) - set(official))
    if unknown:
        raise SystemExit(f'RUN_IDS_OVERRIDE contains unknown IDs: {unknown}')
    if len(ids) != count:
        raise SystemExit(f'RUN_IDS_OVERRIDE count {len(ids)} != CASES {count}')
else:
    if count < 1 or count > len(rows):
        raise SystemExit(f'CASES must be in [1,{len(rows)}], got {count}')
    ids = official[:count]
if len(ids) != len(set(ids)):
    raise SystemExit('duplicate BFCL task IDs')
print(','.join(ids))
PY
)"

for port in "${PORTS[@]}"; do
  if ss -ltn "sport = :${port}" | grep -q LISTEN; then
    echo "server port ${port} is already in use" >&2
    exit 2
  fi
done

# Every arm owns a distinct local proxy. Check all of them before loading the
# model so a stale listener cannot abort a serial run after an expensive arm.
for offset in 0 1 2 3; do
  proxy_port="$((PROXY_PORT_BASE + offset))"
  if ss -ltn "sport = :${proxy_port}" | grep -q LISTEN; then
    echo "proxy port ${proxy_port} is already in use" >&2
    exit 2
  fi
done

SERVER_PIDS=()
cleanup() {
  local status=$?
  set +e
  for pid in "${SERVER_PIDS[@]:-}"; do
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ -n "${pgid}" && "${pgid}" == "${pid}" ]]; then
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    else
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for _ in $(seq 1 30); do
    alive=0
    for pid in "${SERVER_PIDS[@]:-}"; do kill -0 "${pid}" 2>/dev/null && alive=1; done
    [[ "${alive}" == 0 ]] && break
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if kill -0 "${pid}" 2>/dev/null; then
      [[ -n "${pgid}" && "${pgid}" == "${pid}" ]] && kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  echo "results: ${RESULT_ROOT}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

start_server() {
  local gpu="$1"
  local port="$2"
  local label="${3:-shared}"
  local log="${RESULT_ROOT}/servers/gpu${gpu}_${label}.log"
  echo "starting server on physical GPU ${gpu}, port ${port}"
  setsid env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    CUDA_HOME="${CUDA_HOME_FOR_JIT}" CUDACXX="${CUDA_HOME_FOR_JIT}/bin/nvcc" \
    PATH="${CUDA_HOME_FOR_JIT}/bin:${PATH}" CC="${CC_BIN}" CXX="${CXX_BIN}" \
    NVCC_CCBIN="${CXX_BIN}" CUDAHOSTCXX="${CXX_BIN}" MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
    PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}" \
    "${SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" --tokenizer-path "${MODEL_PATH}" \
      --served-model-name c2kv-agent --host 127.0.0.1 --port "${port}" \
      --dtype bfloat16 --tool-call-parser qwen25 \
      --enable-c2kv --enable-streaming-session --c2kv-gist-type dynamic-interleave \
      --c2kv-gist-param qkv --c2kv-query-proj base --c2kv-tools-dump full \
      --disable-radix-cache --page-size 1 --max-running-requests 1 \
      --attention-backend triton --decode-attention-backend triton \
      --prefill-attention-backend triton --sampling-backend pytorch \
      --disable-cuda-graph --disable-piecewise-cuda-graph \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --c2kv-pool-fraction "${C2KV_POOL_FRACTION}" \
      --context-length "${CONTEXT_LENGTH}" --max-total-tokens "${MAX_TOTAL_TOKENS}" \
      --chunked-prefill-size 4096 --random-seed 0 \
      --watchdog-timeout "$((TASK_TIMEOUT + 300))" \
      >"${log}" 2>&1 &
  SERVER_PIDS+=("$!")
  local pid="${SERVER_PIDS[-1]}" deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server GPU ${gpu} exited; see ${log}" >&2
      return 1
    fi
    (( SECONDS < deadline )) || { echo "server GPU ${gpu} startup timeout; see ${log}" >&2; return 1; }
    sleep 2
  done
}

stop_server() {
  local pid="$1"
  local pgid
  set +e
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if kill -0 "${pid}" 2>/dev/null; then
    [[ -n "${pgid}" && "${pgid}" == "${pid}" ]] &&
      kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    [[ -n "${pgid}" && "${pgid}" == "${pid}" ]] &&
      kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
  set -e
}

for i in "${!GPUS[@]}"; do
  start_server "${GPUS[$i]}" "${PORTS[$i]}" "shared"
done

export PYTHONPATH="${ROOT}:${SGLANG_ROOT}/python:${PYTHONPATH:-}"
run_arm() {
  local index="$1" arm="$2" proxy_port="$3"
  local out="${RESULT_ROOT}/runs/${arm}"
  echo "[physical GPU ${GPUS[$index]}] ${arm}"
  local server_pid="${SERVER_PIDS[$index]}"
  (
    env no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
      "${BFCL_PYTHON}" -m c2kv_eval.portable.run \
        --benchmark bfcl --categories "${CATEGORY}" --arm "${arm}" \
        --upstream "http://127.0.0.1:${PORTS[$index]}" --model c2kv-agent \
        --proxy-port "${proxy_port}" --proxy-python "${SGLANG_PYTHON}" \
        --bfcl-dir "${ROOT}" --run-ids "${RUN_IDS}" \
        --num-workers 1 --upstream-timeout "${TASK_TIMEOUT}" \
        --allow-unprofiled --doc-packing turn --max-doc-length 1024 --max-doc-num 48 \
        --tokenizer "${MODEL_PATH}" --out "${out}" --exact-out \
        --run-name "bfcl_long_${arm}" \
        2>&1 | tee "${RESULT_ROOT}/runs/${arm}.launcher.log"
  ) &
  local runner_pid=$!
  while kill -0 "${runner_pid}" 2>/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "server exited while arm=${arm}; aborting benchmark" >&2
      pkill -TERM -P "${runner_pid}" 2>/dev/null || true
      kill -TERM "${runner_pid}" 2>/dev/null || true
      wait "${runner_pid}" 2>/dev/null || true
      return 1
    fi
    sleep 5
  done
  set +e
  wait "${runner_pid}"
  local status=$?
  set -e
  (( status == 0 )) || return "${status}"
  kill -0 "${server_pid}" 2>/dev/null || {
    echo "server exited after arm=${arm}; refusing invalid result" >&2
    return 1
  }
}

validate_arm() {
  local arm="$1"
  local launcher_log="${RESULT_ROOT}/runs/${arm}.launcher.log"
  if [[ ! -f "${launcher_log}" ]] || rg -q 'Error occurred during inference' "${launcher_log}"; then
    echo "invalid arm ${arm}: BFCL harness reported an inference failure" >&2
    return 1
  fi
  "${BFCL_PYTHON}" - "${RESULT_ROOT}/runs/${arm}/summary_${arm}.json" "${CASES}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"missing arm summary: {path}")
summary = json.loads(path.read_text(encoding="utf-8"))
actual = int(summary.get("official_total_count") or 0)
request = summary.get("request_log_summary") or {}
errors = int(request.get("n_error") or 0)
ttft = int(request.get("server_ttft_ms_count") or 0)
if actual != expected or errors != 0 or ttft <= 0:
    raise SystemExit(
        f"invalid arm {summary.get('arm')}: tasks={actual}/{expected}, "
        f"request_errors={errors}, ttft_count={ttft}")
print(f"validated {summary.get('arm')}: tasks={actual}, request_errors=0, ttft_count={ttft}")
PY
}

# One compressed scope per physical GPU; Full is never rerun.
worker_gpu5() {
  run_arm 0 joint_snapkv_persistent_tool_only_r25 "$((PROXY_PORT_BASE + 0))"
}
worker_gpu6() {
  run_arm 1 joint_snapkv_persistent_history_only_r25 "$((PROXY_PORT_BASE + 1))"
}
worker_gpu7() {
  run_arm 2 joint_snapkv_persistent_joint_r25 "$((PROXY_PORT_BASE + 2))"
}

worker_gpu5 & P5=$!
worker_gpu6 & P6=$!
worker_gpu7 & P7=$!
status=0
wait "${P5}" || status=1
wait "${P6}" || status=1
wait "${P7}" || status=1
(( status == 0 )) || { echo "one or more BFCL Long arms failed" >&2; exit 1; }

for arm in   joint_snapkv_persistent_tool_only_r25   joint_snapkv_persistent_history_only_r25   joint_snapkv_persistent_joint_r25; do
  validate_arm "${arm}"
done

# Reuse the immutable Full200 summary without copying or rerunning Full.
mkdir -p "${RESULT_ROOT}/runs/full"
ln -s "${FULL_SUMMARY}" "${RESULT_ROOT}/runs/full/summary_full.json"

"${BFCL_PYTHON}" "${SUMMARY}" "${RESULT_ROOT}" --expected "${CASES}" \
  2>&1 | tee "${RESULT_ROOT}/summary/summarize.log"
echo "completed: ${RESULT_ROOT}"
