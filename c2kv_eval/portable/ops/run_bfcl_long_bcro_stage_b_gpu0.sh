#!/usr/bin/env bash
# BCRO Stage-B: frozen Full subset + real Fixed20K/BCRO closed-loop Test117.
set -Eeuo pipefail

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONHASHSEED=0
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
C1_ROOT="${C1_ROOT:-/home/zhuyuhan/project/c2kv-retrieval-quality}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv-ttft}"
MODEL="${MODEL:-/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000}"
EMBED="${EMBED:-/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
FIXED_ARTIFACT="${FIXED_ARTIFACT:-${ROOT}/results/bcrs_stage_a_v2_final_20261110_210050/bcrs_v2_artifact.json}"
BCRO_ARTIFACT="${BCRO_ARTIFACT:-${ROOT}/results/bcro_stage_a2_20261111_022630/seed_0/bcro_artifact.json}"
SPLIT_CSV="${SPLIT_CSV:-${ROOT}/results/bcro_stage_a2_20261111_022630/seed_0/split.csv}"
FROZEN_FULL_ROOT="${FROZEN_FULL_ROOT:-${ROOT}/results/bfcl_long_joint_scope_gpu6_full200_20261109_230337/runs/full}"
POLICY="${C1_ROOT}/experiments/history_system/runtime/configs/eval_policy_bfcl_long_fixed256.json"
CAPACITY="${C1_ROOT}/experiments/history_system/runtime/configs/eval_capacity_bfcl_long.json"
GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-41170}"
CONTROLLER_PORT="${CONTROLLER_PORT:-41210}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bcro_stage_b_full117_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
DATA="${ROOT}/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
TASK_IDS_OVERRIDE="${TASK_IDS_OVERRIDE:-}"
BCRO_EXACT_IDENTIFIER_GUARD="${BCRO_EXACT_IDENTIFIER_GUARD:-0}"
BCRO_READONLY_REPEAT_GUARD="${BCRO_READONLY_REPEAT_GUARD:-0}"
BCRO_READONLY_REPEAT_PREVIOUS_TURN="${BCRO_READONLY_REPEAT_PREVIOUS_TURN:-0}"
BCRO_READONLY_REPEAT_NO_TOOLS_MAX_TOKENS="${BCRO_READONLY_REPEAT_NO_TOOLS_MAX_TOKENS:-}"
BCRO_POST_TWEET_LITERAL_GUARD="${BCRO_POST_TWEET_LITERAL_GUARD:-0}"
BCRO_CANCEL_BOOKING_FULL_DIVERSION="${BCRO_CANCEL_BOOKING_FULL_DIVERSION:-0}"
BCRO_QUERY_UNIQUE_RAW_EVIDENCE="${BCRO_QUERY_UNIQUE_RAW_EVIDENCE:-0}"
RUN_FIXED20K="${RUN_FIXED20K:-1}"
RUN_BCRO="${RUN_BCRO:-1}"

for path in "${MODEL}/config.json" "${EMBED}/config.json" "${FIXED_ARTIFACT}" \
            "${BCRO_ARTIFACT}" "${SPLIT_CSV}" "${POLICY}" "${CAPACITY}" "${DATA}" \
            "${FROZEN_FULL_ROOT}/summary_full.json" \
            "${FROZEN_FULL_ROOT}/logs/proxy_full_40270.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing: ${path}" >&2; exit 2; }
done
[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
for port in "${SERVER_PORT}" "${CONTROLLER_PORT}"; do
  ! ss -ltn "sport = :${port}" | grep -q LISTEN || {
    echo "port ${port} already in use" >&2; exit 2;
  }
done
mkdir -p "${RESULT_ROOT}/server" "${RESULT_ROOT}/warm" "${RESULT_ROOT}/runs" "${RESULT_ROOT}/summary"
# BCRO Full imports the portable Full builder at decision time. Fail before GPU startup.
env PYTHONPATH="${ROOT}:${C1_ROOT}/experiments/history_system:${PYTHONPATH:-}" \
  "${BFCL_PYTHON}" -c 'from c2kv_eval.portable.backends.sglang import SglangBackend; from runtime.benchmarks.memory_runtime.event_native_bcrs import _portable_full_payload'

if [[ -n "${TASK_IDS_OVERRIDE}" ]]; then
  IFS=',' read -r -a TASKS <<< "${TASK_IDS_OVERRIDE}"
  "${BFCL_PYTHON}" - "${DATA}" "${SPLIT_CSV}" "${TASKS[@]}" <<'PY'
import csv, json, sys
data_ids={json.loads(line)['id'] for line in open(sys.argv[1],encoding='utf-8') if line.strip()}
test_ids={row['task_id'] for row in csv.DictReader(open(sys.argv[2],encoding='utf-8')) if row['split']=='test'}
requested=sys.argv[3:]
assert requested and len(requested)==len(set(requested))
assert set(requested)<=data_ids and set(requested)<=test_ids
PY
else
  mapfile -t TASKS < <("${BFCL_PYTHON}" - "${DATA}" "${SPLIT_CSV}" <<'PY'
import csv, json, sys
data_ids={json.loads(line)['id'] for line in open(sys.argv[1],encoding='utf-8') if line.strip()}
rows=list(csv.DictReader(open(sys.argv[2],encoding='utf-8')))
ids=[row['task_id'] for row in rows if row['split']=='test']
non_test={row['task_id'] for row in rows if row['split']!='test'}
assert len(rows)==197 and len(ids)==117 and len(ids)==len(set(ids))
assert not set(ids)&non_test and set(ids)<=data_ids
print('\n'.join(ids))
PY
)
fi
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
    --random-seed 0 --watchdog-timeout "$((TASK_TIMEOUT + 300))" \
    >"${RESULT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 1200))
until curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "server startup timeout" >&2; exit 1; }
  sleep 2
done

run_method() {
  group="$1"; name="$2"; method="$3"; mode="${4:-}"; artifact="${5:-}"
  out="${RESULT_ROOT}/${group}/${name}"
  command=("${BFCL_PYTHON}" "${C1_ROOT}/experiments/history_system/run_c1.py"
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
    command+=(--bcrs-artifact "${artifact}" --bcrs-mode "${mode}"
              --bcrs-python-root "${ROOT}")
    if [[ "${BCRO_EXACT_IDENTIFIER_GUARD}" == "1" ]]; then
      command+=(--bcro-exact-identifier-guard)
    fi
    if [[ "${BCRO_READONLY_REPEAT_GUARD}" == "1" ]]; then
      command+=(--bcro-readonly-repeat-guard)
    fi
    if [[ "${BCRO_READONLY_REPEAT_PREVIOUS_TURN}" == "1" ]]; then
      command+=(--bcro-readonly-repeat-previous-turn)
    fi
    if [[ -n "${BCRO_READONLY_REPEAT_NO_TOOLS_MAX_TOKENS}" ]]; then
      command+=(--bcro-readonly-repeat-no-tools-max-tokens
                "${BCRO_READONLY_REPEAT_NO_TOOLS_MAX_TOKENS}")
    fi
    if [[ "${BCRO_QUERY_UNIQUE_RAW_EVIDENCE}" == "1" ]]; then
      command+=(--bcro-query-unique-raw-evidence)
    fi
    if [[ "${BCRO_POST_TWEET_LITERAL_GUARD}" == "1" ]]; then
      command+=(--bcro-post-tweet-literal-guard)
    fi
    if [[ "${BCRO_CANCEL_BOOKING_FULL_DIVERSION}" == "1" ]]; then
      command+=(--bcro-cancel-booking-full-diversion)
    fi
  fi
  env no_proxy='*' NO_PROXY='*' \
    PYTHONPATH="${ROOT}:${C1_ROOT}/experiments/history_system:${PYTHONPATH:-}" \
    "${command[@]}" 2>&1 | tee "${RESULT_ROOT}/${group}/${name}.log"
}

# Warm outputs are excluded. Measured misses remain visible and fail acceptance.
if [[ "${RUN_FIXED20K}" == "1" ]]; then
  curl --noproxy '*' -fsS -X POST "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null
  run_method warm fixed20k proposed fixed20k_racer "${FIXED_ARTIFACT}"
  run_method runs fixed20k proposed fixed20k_racer "${FIXED_ARTIFACT}"
fi

if [[ "${RUN_BCRO}" == "1" ]]; then
  curl --noproxy '*' -fsS -X POST "http://127.0.0.1:${SERVER_PORT}/flush_cache?timeout=120" >/dev/null
  run_method warm bcro proposed bcro_racer "${BCRO_ARTIFACT}"
  run_method runs bcro proposed bcro_racer "${BCRO_ARTIFACT}"
fi

if [[ "${RUN_FIXED20K}" == "1" && "${RUN_BCRO}" == "1" ]]; then
  PYTHONPATH="${ROOT}" "${BFCL_PYTHON}" \
    "${ROOT}/c2kv_eval/portable/ops/summarize_bcro_stage_b.py" \
    --result-root "${RESULT_ROOT}" --split-csv "${SPLIT_CSV}" \
    --full-root "${FROZEN_FULL_ROOT}" | tee "${RESULT_ROOT}/summary/summarize.log"
else
  printf 'partial run: fixed20k=%s bcro=%s; comparison skipped\n' \
    "${RUN_FIXED20K}" "${RUN_BCRO}" | tee "${RESULT_ROOT}/summary/partial_run.log"
fi
echo "completed: ${RESULT_ROOT}"
