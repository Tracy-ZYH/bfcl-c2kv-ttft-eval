#!/usr/bin/env bash
# Stage-B closed-loop smoke: Full, Fixed20K->Racer, frozen BCRS->Racer.
set -Eeuo pipefail

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONHASHSEED=42
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
C1_ROOT="${C1_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/runtime}"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/third_party/sglang}"
MODEL="${MODEL:-/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000}"
EMBED="${EMBED:-/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
BCRS_ARTIFACT="${BCRS_ARTIFACT:-${ROOT}/results/bcrs_stage_a_v2_final_20261110_210050/bcrs_v2_artifact.json}"
POLICY="${C1_ROOT}/runtime/configs/eval_policy_bfcl_long_fixed256.json"
CAPACITY="${C1_ROOT}/runtime/configs/eval_capacity_bfcl_long.json"
GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-39970}"
CONTROLLER_PORT="${CONTROLLER_PORT:-40010}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bfcl_long_bcrs_smoke_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
DATA="${ROOT}/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
TASK_IDS_CSV="${TASK_IDS_CSV:-}"
RUN_FULL="${RUN_FULL:-1}"
RUN_FIXED20K="${RUN_FIXED20K:-1}"
RUN_BCRS="${RUN_BCRS:-1}"

for path in "${MODEL}/config.json" "${EMBED}/config.json" "${BCRS_ARTIFACT}" \
            "${POLICY}" "${CAPACITY}" "${DATA}"; do
  [[ -e "${path}" ]] || { echo "missing: ${path}" >&2; exit 2; }
done
[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
for port in "${SERVER_PORT}" "${CONTROLLER_PORT}"; do
  ! ss -ltn "sport = :${port}" | grep -q LISTEN || {
    echo "port ${port} already in use" >&2; exit 2;
  }
done
mkdir -p "${RESULT_ROOT}/server" "${RESULT_ROOT}/warm" "${RESULT_ROOT}/runs" "${RESULT_ROOT}/summary"

mapfile -t TASKS < <("${BFCL_PYTHON}" - "${DATA}" "${TASK_IDS_CSV}" <<'PY'
import json, sys
rows=[json.loads(line) for line in open(sys.argv[1],encoding='utf-8') if line.strip()]
ids=sys.argv[2].split(',') if sys.argv[2] else [row['id'] for row in rows[:20]]
if not sys.argv[2]: assert ids == [f"multi_turn_long_context_{index}" for index in range(20)]
assert ids and len(ids) == len(set(ids)) and all(task_id in {row['id'] for row in rows} for task_id in ids)
print('\n'.join(ids))
PY
)
printf '%s\n' "${TASKS[@]}" > "${RESULT_ROOT}/task_ids.txt"

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
  echo "results: ${RESULT_ROOT}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

setsid env -u CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="${GPU}" \
  CUDA_HOME=/usr/local/cuda-13.0 CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
  PATH="/usr/local/cuda-13.0/bin:${PATH}" CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
  NVCC_CCBIN=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 MAX_JOBS=1 \
  PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}" \
  "${SGLANG_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL}" --tokenizer-path "${MODEL}" --served-model-name c2kv-agent \
    --host 127.0.0.1 --port "${SERVER_PORT}" --dtype bfloat16 \
    --tool-call-parser qwen25 --enable-c2kv --enable-streaming-session \
    --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
    --c2kv-query-proj base --c2kv-tools-dump full \
    --enable-return-hidden-states --c2kv-shadow-feature-layer -2 \
    --disable-radix-cache \
    --page-size 1 --max-running-requests 1 --attention-backend triton \
    --decode-attention-backend triton --prefill-attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph --disable-piecewise-cuda-graph \
    --mem-fraction-static 0.65 --c2kv-pool-fraction 0.15 \
    --context-length 262144 --max-total-tokens 262144 --chunked-prefill-size 4096 \
    --random-seed 42 --watchdog-timeout "$((TASK_TIMEOUT + 300))" \
    >"${RESULT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 1200))
until curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "server startup timeout" >&2; exit 1; }
  sleep 2
done

run_method() {
  group="$1"; name="$2"; method="$3"; mode="${4:-}"
  out="${RESULT_ROOT}/${group}/${name}"
  command=("${BFCL_PYTHON}" "${C1_ROOT}/runtime/history_system/run_c1.py"
    --benchmark bfcl --method "${method}" --checkpoint "${MODEL}"
    --sglang-backend-url "http://127.0.0.1:${SERVER_PORT}"
    --embedding-model "${EMBED}" --embedding-device cpu --detector t02_risk
    --selector-threshold 0.5 --eval-policy-override "${POLICY}"
    --eval-capacity-override "${CAPACITY}" --benchmark-dir "${ROOT}"
    --bfcl-python "${BFCL_PYTHON}" --portable-root "${ROOT}"
    --sglang-root "${SGLANG_ROOT}" --port "${CONTROLLER_PORT}"
    --task-timeout "${TASK_TIMEOUT}" --out "${out}")
  for task in "${TASKS[@]}"; do command+=(--task-id "${task}"); done
  if [[ -n "${mode}" ]]; then
    command+=(--bcrs-artifact "${BCRS_ARTIFACT}" --bcrs-mode "${mode}"
              --bcrs-python-root "${ROOT}")
  fi
  env no_proxy='*' NO_PROXY='*' PYTHONPATH="${ROOT}/runtime/history_system" \
    "${command[@]}" 2>&1 | tee "${RESULT_ROOT}/${group}/${name}.log"
}

# Full has no compression cache and is measured once with Radix disabled.
if [[ "${RUN_FULL}" == "1" ]]; then
run_method runs full full
fi

# Warm and measured passes are deliberately separate. The warm outputs are
# never included in TTFT/accuracy summaries. A measured cache miss fails in
# summarize_bcrs_smoke.py instead of silently becoming online extraction.
curl --noproxy '*' -fsS -X POST "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null
if [[ "${RUN_FIXED20K}" == "1" ]]; then
run_method warm fixed20k proposed fixed20k_racer
run_method runs fixed20k proposed fixed20k_racer
fi

curl --noproxy '*' -fsS -X POST "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null
if [[ "${RUN_BCRS}" == "1" ]]; then
run_method warm bcrs proposed bcrs_racer
run_method runs bcrs proposed bcrs_racer
fi

if [[ "${RUN_FULL}" == "1" && "${RUN_FIXED20K}" == "1" && "${RUN_BCRS}" == "1" ]]; then
PYTHONPATH="${ROOT}" "${BFCL_PYTHON}" \
  "${ROOT}/c2kv_eval/portable/ops/summarize_bcrs_smoke.py" "${RESULT_ROOT}" \
  | tee "${RESULT_ROOT}/summary/summarize.log"
else
  printf 'partial arm run: full=%s fixed20k=%s bcrs=%s; comparison summary skipped\n' "${RUN_FULL}" "${RUN_FIXED20K}" "${RUN_BCRS}" | tee "${RESULT_ROOT}/summary/partial_run.log"
fi
echo "completed: ${RESULT_ROOT}"
