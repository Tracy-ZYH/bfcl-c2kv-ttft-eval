#!/usr/bin/env bash
# Decision-level dual-objective GP: paired pilot, then one frozen 200-task run.
set -Eeuo pipefail

PHASE="${1:-}"
if [[ "${PHASE}" != "pilot" && "${PHASE}" != "final" ]]; then
  echo "usage: $0 pilot|final" >&2
  exit 2
fi

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
C1_ROOT="${C1_ROOT:-/home/zhuyuhan/project/c2kv-retrieval-quality}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv-ttft}"
MODEL="${MODEL:-/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000}"
EMBED="${EMBED:-/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python}"
GPU="${GPU:-6}"
SERVER_PORT="${SERVER_PORT:-42660}"
CONTROLLER_PORT="${CONTROLLER_PORT:-42760}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
SOURCE_FULL="${SOURCE_FULL:-/home/zhuyuhan/project/bfcl-c2kv/results/bfcl_long_soft256_full200_gpu012_20261029_153723}"
FROZEN_FULL="${FROZEN_FULL:-${ROOT}/results/bfcl_long_joint_scope_gpu6_full200_20261109_230337/runs/full}"
POLICY="${C1_ROOT}/experiments/history_system/runtime/configs/eval_policy_bfcl_long_fixed256.json"
CAPACITY="${C1_ROOT}/experiments/history_system/runtime/configs/eval_capacity_bfcl_long.json"
OPS="${ROOT}/c2kv_eval/portable/ops"
EXPERIMENT="${OPS}/decision_dual_gp_experiment.py"
REPLAY="${OPS}/bfcl_long_cache_hit_ttft.py"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONHASHSEED=42
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

for path in "${MODEL}/config.json" "${EMBED}/config.json" "${POLICY}"             "${CAPACITY}" "${EXPERIMENT}" "${REPLAY}"             "${FROZEN_FULL}/summary_full.json"; do
  [[ -e "${path}" ]] || { echo "missing: ${path}" >&2; exit 2; }
done

if [[ "${PHASE}" == "pilot" ]]; then
  RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/decision_dual_gp_pilot_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
  [[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
  mkdir -p "${RESULT_ROOT}"
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${BFCL_PYTHON}" "${EXPERIMENT}" prepare     --source-full-root "${SOURCE_FULL}" --model "${MODEL}"     --tasks-per-fold 4 --out "${RESULT_ROOT}/plan"
  mapfile -t TASKS < "${RESULT_ROOT}/plan/task_ids.txt"
  "${BFCL_PYTHON}" "${REPLAY}"     --phase formal --arm full --gpu "${GPU}"     --out "${RESULT_ROOT}/full" --repetitions 3     --server-port "${SERVER_PORT}" --controller-port "${CONTROLLER_PORT}"     --startup-timeout 1200 --task-timeout "${TASK_TIMEOUT}"     --decisions-file "${RESULT_ROOT}/plan/decisions.json"     --model "${MODEL}" --embedding-model "${EMBED}"     --eval-policy "${POLICY}" --eval-capacity "${CAPACITY}"     --run-ids "${TASKS[@]}"
  "${BFCL_PYTHON}" "${REPLAY}"     --phase formal --arm racer --gpu "${GPU}"     --out "${RESULT_ROOT}/racer" --repetitions 3     --server-port "$((SERVER_PORT + 1))"     --controller-port "$((CONTROLLER_PORT + 1000))"     --startup-timeout 1200 --task-timeout "${TASK_TIMEOUT}"     --decisions-file "${RESULT_ROOT}/plan/decisions.json"     --model "${MODEL}" --embedding-model "${EMBED}"     --eval-policy "${POLICY}" --eval-capacity "${CAPACITY}"     --run-ids "${TASKS[@]}"
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${BFCL_PYTHON}" "${EXPERIMENT}" fit     --plan "${RESULT_ROOT}/plan/selection_manifest.json"     --full-samples "${RESULT_ROOT}/full/samples.csv"     --racer-samples "${RESULT_ROOT}/racer/samples.csv"     --uncertainty-beta 1.0 --seed 42 --out "${RESULT_ROOT}/frozen"
  echo "pilot completed: ${RESULT_ROOT}"
  printf '%s\n' "${RESULT_ROOT}" > "${ROOT}/results/decision_dual_gp_latest_pilot.txt"
  echo "artifact: ${RESULT_ROOT}/frozen/decision_dual_gp_artifact.json"
  exit 0
fi

if [[ -z "${PILOT_ROOT:-}" ]]; then
  [[ -s "${ROOT}/results/decision_dual_gp_latest_pilot.txt" ]] || {
    echo "set PILOT_ROOT or run a successful pilot first" >&2; exit 2; }
  PILOT_ROOT="$(<"${ROOT}/results/decision_dual_gp_latest_pilot.txt")"
fi
GATE="${PILOT_ROOT}/frozen/pilot_gate.json"
ARTIFACT="${PILOT_ROOT}/frozen/decision_dual_gp_artifact.json"
"${BFCL_PYTHON}" - "${GATE}" "${ARTIFACT}" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate.get("passed") is True, gate
assert pathlib.Path(sys.argv[2]).is_file()
PY

RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/decision_dual_gp_final200_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
for port in "${SERVER_PORT}" "${CONTROLLER_PORT}"; do
  ! ss -ltn "sport = :${port}" | grep -q LISTEN || {
    echo "port ${port} already in use" >&2
    exit 2
  }
done
mkdir -p "${RESULT_ROOT}/server" "${RESULT_ROOT}/runs" "${RESULT_ROOT}/summary"
GPU_UUID="$(nvidia-smi --id="${GPU}" --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '[:space:]')"
[[ "${GPU_UUID}" == GPU-* ]] || { echo "cannot resolve physical GPU ${GPU}" >&2; exit 2; }

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

setsid env CUDA_VISIBLE_DEVICES="${GPU_UUID}"   CUDA_HOME=/usr/local/cuda-13.0 CUDACXX=/usr/local/cuda-13.0/bin/nvcc   PATH="/usr/local/cuda-13.0/bin:${PATH}" CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12   NVCC_CCBIN=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 MAX_JOBS=1   PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"   "${SGLANG_PYTHON}" -m sglang.launch_server     --model-path "${MODEL}" --tokenizer-path "${MODEL}" --served-model-name c2kv-agent     --host 127.0.0.1 --port "${SERVER_PORT}" --dtype bfloat16     --tool-call-parser qwen25 --enable-c2kv --enable-streaming-session     --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv     --c2kv-query-proj base --c2kv-tools-dump full     --enable-return-hidden-states --c2kv-shadow-feature-layer -2     --disable-radix-cache --page-size 1 --max-running-requests 1     --attention-backend triton --decode-attention-backend triton     --prefill-attention-backend triton --sampling-backend pytorch     --disable-cuda-graph --disable-piecewise-cuda-graph     --mem-fraction-static 0.65 --c2kv-pool-fraction 0.15     --context-length 262144 --max-total-tokens 262144     --chunked-prefill-size 4096 --random-seed 0     --watchdog-timeout "$((TASK_TIMEOUT + 300))"     >"${RESULT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 1200))
until curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "server startup timeout" >&2; exit 1; }
  sleep 2
done

mapfile -t TASKS < <("${BFCL_PYTHON}" - <<'PY'
print("\n".join(f"multi_turn_long_context_{i}" for i in range(200)))
PY
)
COMMAND=("${BFCL_PYTHON}" "${C1_ROOT}/experiments/history_system/run_c1.py"
  --benchmark bfcl --method proposed --checkpoint "${MODEL}"
  --sglang-backend-url "http://127.0.0.1:${SERVER_PORT}"
  --embedding-model "${EMBED}" --embedding-device cpu
  --detector t02_risk --selector-threshold 0.5
  --eval-policy-override "${POLICY}" --eval-capacity-override "${CAPACITY}"
  --benchmark-dir "${ROOT}" --bfcl-python "${BFCL_PYTHON}"
  --portable-root "${ROOT}" --sglang-root "${SGLANG_ROOT}"
  --port "${CONTROLLER_PORT}" --task-timeout "${TASK_TIMEOUT}"
  --bcrs-artifact "${ARTIFACT}" --bcrs-mode dual_gp_racer
  --bcrs-python-root "${ROOT}" --out "${RESULT_ROOT}/runs/dual_gp")
for task in "${TASKS[@]}"; do COMMAND+=(--task-id "${task}"); done
env no_proxy='*' NO_PROXY='*'   PYTHONPATH="${ROOT}:${C1_ROOT}/experiments/history_system:${PYTHONPATH:-}"   "${COMMAND[@]}" 2>&1 | tee "${RESULT_ROOT}/runs/dual_gp.log"

PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${BFCL_PYTHON}" "${EXPERIMENT}" summarize-final   --run-root "${RESULT_ROOT}/runs/dual_gp" --full-root "${FROZEN_FULL}"   --out "${RESULT_ROOT}/summary/final.json" | tee "${RESULT_ROOT}/summary/final.log"
echo "completed: ${RESULT_ROOT}"
