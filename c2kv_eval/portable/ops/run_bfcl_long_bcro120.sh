#!/usr/bin/env bash
# Reproduce the frozen BCRO-120 configuration. This wrapper changes configuration
# only; routing, detector, recovery, compression, and generation implementations
# remain in the shared Stage-B launcher/runtime.
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv-ttft-eval}"
BASE="${ROOT}/c2kv_eval/portable/ops/run_bfcl_long_bcro_stage_b_gpu0.sh"
ARTIFACT_DIR="${ROOT}/artifacts/bcro"
CONFIG="${BCRO120_CONFIG:-${ARTIFACT_DIR}/bcro120_config.json}"
TASK_IDS_FILE="${BCRO120_TASK_IDS_FILE:-${ARTIFACT_DIR}/task_ids.120.txt}"
DATA="${ROOT}/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python}"
GPU="${GPU:-0}"
BCRO120_PREFLIGHT_ONLY="${BCRO120_PREFLIGHT_ONLY:-0}"
BCRO120_REQUIRE_EXACT_RUNTIME="${BCRO120_REQUIRE_EXACT_RUNTIME:-0}"

for path in "${BASE}" "${CONFIG}" "${TASK_IDS_FILE}" "${DATA}"   "${ARTIFACT_DIR}/bcro_artifact.seed0.json"   "${ARTIFACT_DIR}/split.seed0.csv"   "${ARTIFACT_DIR}/eval_policy_bfcl_long_fixed256.json"   "${ARTIFACT_DIR}/eval_capacity_bfcl_long.json"   "${ROOT}/runtime/artifacts/c1_risk.t02_v1.json"   "${ROOT}/runtime/configs/current_algorithm.json"; do
  [[ -f "${path}" ]] || { echo "missing BCRO-120 input: ${path}" >&2; exit 2; }
done

(
  cd "${ARTIFACT_DIR}"
  sha256sum -c manifest.sha256
)

"${BFCL_PYTHON}" - "${ROOT}" "${CONFIG}" "${TASK_IDS_FILE}" "${DATA}" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys

root, config_path, task_path, data_path = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text(encoding="utf-8"))
tasks = [line.strip() for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
data_ids = {
    json.loads(line)["id"]
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
split_rows = list(csv.DictReader((root / config["split"]).open(encoding="utf-8")))
test117 = {row["task_id"] for row in split_rows if row["split"] == "test"}
extras = set(config["additional_task_ids"])

assert config["schema"] == "bcro-120-reproduction-config-v1"
assert len(tasks) == config["task_count"] == 120
assert len(tasks) == len(set(tasks))
assert set(tasks) <= data_ids
assert set(tasks) == test117 | extras
assert len(test117) == config["stage_a2_test_count"] == 117
assert not test117 & extras
assert hashlib.sha256(task_path.read_bytes()).hexdigest() == config["task_manifest_sha256"]
assert config["guards"] == {
    "exact_identifier": True,
    "readonly_repeat_same_turn": True,
    "readonly_repeat_previous_turn": False,
    "readonly_repeat_no_tools_max_tokens": None,
    "post_tweet_literal": True,
    "cancel_booking_full_diversion": True,
    "query_unique_raw_evidence": False,
    "shared_regeneration_budget": 1,
}
model_config = Path(config["model"]) / "config.json"
embedding_config = Path(config["embedding_model"]) / "config.json"
detector_artifact = root / config["detector"]["artifact"]
assert model_config.is_file()
assert embedding_config.is_file()
assert detector_artifact.is_file()
assert hashlib.sha256(model_config.read_bytes()).hexdigest() == config["model_config_sha256"]
assert hashlib.sha256(detector_artifact.read_bytes()).hexdigest() == config["detector"]["artifact_sha256"]
sglang_package = root / "third_party/sglang"
assert (sglang_package / "__init__.py").is_file()
assert (sglang_package / "launch_server.py").is_file()
print("BCRO-120 manifest/config validation: PASS")
PY

env -i   PATH="/usr/bin:/bin"   PYTHONPATH="${ROOT}:${ROOT}/runtime/history_system"   PYTHONDONTWRITEBYTECODE=1   "${BFCL_PYTHON}" -c   'from c2kv_eval.portable.backends.sglang import SglangBackend; from runtime.benchmarks.memory_runtime.event_native_bcrs import _portable_full_payload; import run_c1; print("BCRO-120 isolated imports: PASS")'

EXPECTED_EXPERIMENT_SHA="e54fc79fa1ac7d6211f5c1832bba2b026cc5562bee5e0faa9d5b844025188661"
EXPECTED_SET_RETRIEVAL_SHA="d5e7056b842c2f83ad570ca81ac190ffc62436ac48dd3ef28e4109b754bacafd"
CURRENT_EXPERIMENT_SHA="$(sha256sum "${ROOT}/runtime/benchmarks/memory_runtime/recovery/experiment.py" | awk '{print $1}')"
CURRENT_SET_RETRIEVAL_SHA="$(sha256sum "${ROOT}/runtime/benchmarks/memory_runtime/recovery/set_retrieval.py" | awk '{print $1}')"
if [[ "${CURRENT_EXPERIMENT_SHA}" != "${EXPECTED_EXPERIMENT_SHA}" || "${CURRENT_SET_RETRIEVAL_SHA}" != "${EXPECTED_SET_RETRIEVAL_SHA}" ]]; then
  echo "WARNING: runtime differs from frozen BCRO-120 recovery implementation (commit a271a2d)." >&2
  if [[ "${BCRO120_REQUIRE_EXACT_RUNTIME}" == "1" ]]; then
    echo "Strict BCRO-120 runtime reproduction refused; use the frozen commit." >&2
    exit 3
  fi
else
  echo "BCRO-120 frozen runtime hashes: PASS"
fi

if [[ "${BCRO120_PREFLIGHT_ONLY}" == "1" ]]; then
  echo "BCRO-120 preflight completed; GPU/server not started."
  exit 0
fi

TASK_IDS_OVERRIDE="$(paste -sd, "${TASK_IDS_FILE}")"
export ROOT GPU BFCL_PYTHON TASK_IDS_OVERRIDE
export C1_ROOT="${ROOT}"
export SGLANG_ROOT="${ROOT}/third_party/sglang"
export ALLOW_TASK_IDS_OUTSIDE_SPLIT=1
export FIXED_ARTIFACT="${ARTIFACT_DIR}/bcrs_v2_artifact.json"
export BCRO_ARTIFACT="${ARTIFACT_DIR}/bcro_artifact.seed0.json"
export SPLIT_CSV="${ARTIFACT_DIR}/split.seed0.csv"
export POLICY="${ARTIFACT_DIR}/eval_policy_bfcl_long_fixed256.json"
export CAPACITY="${ARTIFACT_DIR}/eval_capacity_bfcl_long.json"
export RUN_FIXED20K=0
export RUN_BCRO=1

# Frozen BCRO-120 guard configuration recovered from request telemetry.
export BCRO_EXACT_IDENTIFIER_GUARD=1
export BCRO_READONLY_REPEAT_GUARD=1
export BCRO_READONLY_REPEAT_PREVIOUS_TURN=0
export BCRO_READONLY_REPEAT_NO_TOOLS_MAX_TOKENS=""
export BCRO_POST_TWEET_LITERAL_GUARD=1
export BCRO_CANCEL_BOOKING_FULL_DIVERSION=1
export BCRO_QUERY_UNIQUE_RAW_EVIDENCE=0

export RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/bcro_stage_b_full120_repro_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
exec bash "${BASE}"
