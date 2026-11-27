#!/usr/bin/env bash
# User-operated ModelArts launcher. This script starts one private NPU server.
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
CONFIG="${BCRO120_CONFIG:-${ROOT}/artifacts/bcro/bcro120_config.json}"
MODE="${MODE:-smoke}"
case "${MODE}" in
  preflight|smoke|full) ;;
  *) echo "MODE must be preflight, smoke, or full" >&2; exit 2 ;;
esac
if [[ "${MODE}" == "full" ]]; then
  MAX_TASKS="${MAX_TASKS:-120}"
elif [[ "${MODE}" == "smoke" ]]; then
  MAX_TASKS="${MAX_TASKS:-2}"
else
  MAX_TASKS="${MAX_TASKS:-120}"
fi
[[ "${MAX_TASKS}" =~ ^[0-9]+$ ]] && (( MAX_TASKS >= 1 && MAX_TASKS <= 120 )) || {
  echo "MAX_TASKS must be in [1, 120]" >&2; exit 2;
}

NPU_DEVICE="${NPU_DEVICE:-0}"
SERVER_PORT="${SERVER_PORT:-41080}"
CONTROLLER_PORT="${CONTROLLER_PORT:-41120}"
TASK_TIMEOUT="${TASK_TIMEOUT:-10800}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
NPU_PYTHON="${NPU_PYTHON:-python3}"
BFCL_PYTHON="${BFCL_PYTHON:-${NPU_PYTHON}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${NPU_PYTHON}}"
MODEL_PATH="${MODEL_PATH:-$(${NPU_PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "${CONFIG}")}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$(${NPU_PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["embedding_model"])' "${CONFIG}")}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bcro120_npu_$(date +%Y%m%d_%H%M%S)}"

# A ModelArts image may expose one of these setup scripts. An explicit value
# wins; failure to initialize CANN is caught by the runtime preflight.
if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_ENV_SCRIPT}"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
fi

export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export C2KV_SPEED_TIMING=1
unset CUDA_VISIBLE_DEVICES CUDA_HOME CUDACXX CUDAHOSTCXX

PYTHONPATH_COMMON="${ROOT}:${ROOT}/runtime/history_system:${ROOT}/third_party${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH_COMMON}"

[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
mkdir -p "${RESULT_ROOT}/server" "${RESULT_ROOT}/warm" "${RESULT_ROOT}/runs"

"${NPU_PYTHON}" -m c2kv_eval.portable.npu.preflight \
  --root "${ROOT}" --config "${CONFIG}" --model "${MODEL_PATH}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --json-out "${RESULT_ROOT}/npu_preflight.json"

"${NPU_PYTHON}" - "${CONFIG}" "${RESULT_ROOT}/run_config.json" \
  "${MODE}" "${MAX_TASKS}" "${NPU_DEVICE}" "${SERVER_PORT}" "${CONTROLLER_PORT}" <<'PY'
import json, os, sys
src, dst, mode, count, device, server, controller = sys.argv[1:]
cfg = json.load(open(src, encoding="utf-8"))
payload = {
    "schema": "bcro-120-npu-run-config-v1",
    "mode": mode,
    "task_count": int(count),
    "npu_device": device,
    "server_port": int(server),
    "controller_port": int(controller),
    "frozen_config": cfg,
    "guards": cfg["guards"],
    "environment_overrides": {
        "model_path": os.environ.get("MODEL_PATH"),
        "tokenizer_path": os.environ.get("TOKENIZER_PATH"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
    },
}
open(dst, "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
PY

if [[ "${MODE}" == "preflight" ]]; then
  echo "NPU preflight completed; no server started. Results: ${RESULT_ROOT}"
  exit 0
fi

mapfile -t ALL_TASKS < "${ROOT}/artifacts/bcro/task_ids.120.txt"
TASKS=("${ALL_TASKS[@]:0:${MAX_TASKS}}")
printf '%s\n' "${TASKS[@]}" > "${RESULT_ROOT}/task_ids.txt"

"${NPU_PYTHON}" - "${SERVER_PORT}" "${CONTROLLER_PORT}" <<'PY'
import socket, sys
for raw in sys.argv[1:]:
    port = int(raw)
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"port {port} is unavailable: {exc}")
PY

SERVER_PID=""
cleanup() {
  status=$?
  set +e
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    curl --noproxy '*' -fsS -X POST \
      "http://${SERVER_HOST}:${SERVER_PORT}/internal/graceful-stop" >/dev/null 2>&1 || true
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -0 "${SERVER_PID}" 2>/dev/null && kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  echo "Results: ${RESULT_ROOT}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

setsid env \
  ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}" \
  PYTHONPATH="${ROOT}/third_party:${ROOT}:${PYTHONPATH_COMMON}" \
  C2KV_SPEED_TIMING=1 \
  "${SGLANG_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --tokenizer-path "${TOKENIZER_PATH}" \
    --served-model-name c2kv-agent --host "${SERVER_HOST}" --port "${SERVER_PORT}" \
    --device npu --dtype bfloat16 --tool-call-parser qwen25 \
    --attention-backend ascend --prefill-attention-backend ascend \
    --decode-attention-backend ascend --sampling-backend pytorch \
    --enable-c2kv --enable-streaming-session \
    --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
    --c2kv-query-proj base --c2kv-tools-dump full \
    --enable-return-hidden-states --c2kv-shadow-feature-layer -2 \
    --disable-radix-cache --disable-cuda-graph --disable-piecewise-cuda-graph \
    --page-size 1 --max-running-requests 1 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.65}" \
    --c2kv-pool-fraction "${C2KV_POOL_FRACTION:-0.15}" \
    --context-length 262144 --max-total-tokens 262144 \
    --chunked-prefill-size 4096 --random-seed 0 \
    --watchdog-timeout "$((TASK_TIMEOUT + 300))" \
    >"${RESULT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --noproxy '*' -fsS "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    echo "NPU server exited during startup; see ${RESULT_ROOT}/server/server.log" >&2
    exit 1
  }
  (( SECONDS < deadline )) || {
    echo "NPU server startup timeout; see ${RESULT_ROOT}/server/server.log" >&2
    exit 1
  }
  sleep 2
done

run_method() {
  local group="$1" name="$2" method="$3"
  local out="${RESULT_ROOT}/${group}/${name}"
  local command=(
    "${BFCL_PYTHON}" "${ROOT}/runtime/history_system/run_c1.py"
    --benchmark bfcl --method "${method}" --checkpoint "${MODEL_PATH}"
    --sglang-backend-url "http://${SERVER_HOST}:${SERVER_PORT}"
    --embedding-model "${EMBEDDING_MODEL}" --embedding-device cpu
    --detector t02_risk --selector-threshold 0.5
    --eval-policy-override "${ROOT}/artifacts/bcro/eval_policy_bfcl_long_fixed256.json"
    --eval-capacity-override "${ROOT}/artifacts/bcro/eval_capacity_bfcl_long.json"
    --benchmark-dir "${ROOT}" --bfcl-python "${BFCL_PYTHON}"
    --portable-root "${ROOT}" --sglang-root "${ROOT}/third_party/sglang"
    --port "${CONTROLLER_PORT}" --task-timeout "${TASK_TIMEOUT}" --out "${out}"
  )
  local task
  for task in "${TASKS[@]}"; do command+=(--task-id "${task}"); done
  if [[ "${method}" == "proposed" ]]; then
    command+=(
      --selector-artifact "${ROOT}/runtime/artifacts/c1_risk.t02_v1.json"
      --bcrs-artifact "${ROOT}/artifacts/bcro/bcro_artifact.seed0.json"
      --bcrs-mode bcro_racer --bcrs-python-root "${ROOT}"
      --bcro-exact-identifier-guard
      --bcro-readonly-repeat-guard
      --bcro-post-tweet-literal-guard
      --bcro-cancel-booking-full-diversion
    )
  fi
  env no_proxy='*' NO_PROXY='*' \
    PYTHONPATH="${ROOT}:${ROOT}/runtime/history_system:${ROOT}/third_party" \
    "${command[@]}" 2>&1 | tee "${RESULT_ROOT}/${group}/${name}.log"
}

# Full is measured on the same NPU and never consumes a historical RTX result.
run_method runs full full

# Warm cache construction is excluded from measured output. The measured run
# still fails closed if any needed block misses or performs online extraction.
curl --noproxy '*' -fsS -X POST \
  "http://${SERVER_HOST}:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null
run_method warm bcro proposed
run_method runs bcro proposed

"${BFCL_PYTHON}" -m c2kv_eval.portable.npu.summarize \
  --full-root "${RESULT_ROOT}/runs/full" \
  --bcro-root "${RESULT_ROOT}/runs/bcro" \
  --out-dir "${RESULT_ROOT}/summary" \
  --config "${CONFIG}" --max-tasks "${MAX_TASKS}" \
  | tee "${RESULT_ROOT}/summary_stdout.log"

echo "completed: ${RESULT_ROOT}"
