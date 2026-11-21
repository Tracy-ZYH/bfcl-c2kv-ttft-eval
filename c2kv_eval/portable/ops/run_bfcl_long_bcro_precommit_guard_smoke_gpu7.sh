#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
GPU="${GPU:-7}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bcro_precommit_guards_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
BASE="${ROOT}/c2kv_eval/portable/ops/run_bfcl_long_bcro_stage_b_gpu0.sh"
PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"

[[ ! -e "${RESULT_ROOT}" ]] || { echo "refusing to overwrite ${RESULT_ROOT}" >&2; exit 2; }
mkdir -p "${RESULT_ROOT}"

env GPU="${GPU}" SERVER_PORT=41770 CONTROLLER_PORT=41810 \
  RUN_FIXED20K=0 RUN_BCRO=1 \
  TASK_IDS_OVERRIDE="multi_turn_long_context_151,multi_turn_long_context_153,multi_turn_long_context_191,multi_turn_long_context_192,multi_turn_long_context_193" \
  BCRO_EXACT_IDENTIFIER_GUARD=1 BCRO_READONLY_REPEAT_GUARD=0 \
  RESULT_ROOT="${RESULT_ROOT}/exact_identifier" \
  bash "${BASE}"

env GPU="${GPU}" SERVER_PORT=41771 CONTROLLER_PORT=41811 \
  RUN_FIXED20K=0 RUN_BCRO=1 \
  TASK_IDS_OVERRIDE="multi_turn_long_context_113" \
  BCRO_EXACT_IDENTIFIER_GUARD=0 BCRO_READONLY_REPEAT_GUARD=1 \
  RESULT_ROOT="${RESULT_ROOT}/readonly_loop" \
  bash "${BASE}"

"${PYTHON}" "${ROOT}/c2kv_eval/portable/ops/summarize_bcro_precommit_guard_smoke.py" \
  --root "${RESULT_ROOT}" | tee "${RESULT_ROOT}/summary.log"
echo "completed: ${RESULT_ROOT}"
