# Frozen BCRO-120 on Ascend NPU

This package runs only the formal 120-task BFCL `multi_turn_long_context`
comparison:

- `Full`: ordinary full-context SGLang chat, no BCRO or C2KV compression.
- `BCRO`: frozen Ridge+GP routing, 8x C2KV, 256-token history budget,
  frozen T02 at threshold 0.5, the validated guards, and normal recovery.

It does not use the BCRO-117 entry point and does not reuse RTX latency as the
NPU Full baseline. The task IDs, order, GP artifact, detector, policy, capacity,
model-config hash, generation settings, and guard settings are validated before
the server starts.

## Validation status

The vendored SGLang source contains an Ascend attention backend, paged NPU KV
pool, Qwen3 NPU kernels, C2KV gist/stored-KV injection, absolute-position/RoPE
handling, hidden-state return, output logprobs, and NPU synchronization paths.
Static and mock tests cover the wrapper and result accounting.

This checkout has not been exercised on Ascend hardware from the RTX host.
Real C2KV injection, hidden/logprob capture, recovery, and timing remain blocked
until the ModelArts smoke passes. The launcher fails instead of substituting
text truncation, ordinary raw Prefill, or simulated KV injection.

## ModelArts environment

Use a ModelArts image whose CANN, `torch`, `torch_npu`, and `sgl_kernel_npu`
versions are mutually compatible. Do not install a generic CUDA PyTorch wheel
into that image. The preflight records the exact versions in
`npu_preflight.json` and requires all three NPU modules and at least one visible
device. Install the repository's ordinary BFCL/client dependencies in the same
environment, or set `BFCL_PYTHON` to a compatible second environment.

The model and embedding directories may be mounted at different ModelArts
paths. `MODEL_PATH` is accepted only when its `config.json` matches the frozen
SHA256. No model weights, datasets, caches, credentials, or `.env` files are
part of this package.

```bash
cd /home/ma-user/work/bfcl-c2kv-ttft-eval
export CANN_ENV_SCRIPT=/usr/local/Ascend/ascend-toolkit/set_env.sh
export NPU_PYTHON=/path/to/npu-env/bin/python
export BFCL_PYTHON="$NPU_PYTHON"
export SGLANG_PYTHON="$NPU_PYTHON"
export MODEL_PATH=/path/to/checkpoint-1000
export TOKENIZER_PATH="$MODEL_PATH"
export EMBEDDING_MODEL=/path/to/Qwen3-Embedding-0.6B
```

Environment-only validation (does not start a server):

```bash
MODE=preflight NPU_DEVICE=0 \
  bash c2kv_eval/portable/npu/run_bcro120_npu.sh
```

Two-task smoke:

```bash
MODE=smoke MAX_TASKS=2 NPU_DEVICE=0 \
  RESULT_ROOT="$PWD/results/bcro120_npu_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash c2kv_eval/portable/npu/run_bcro120_npu.sh
```

Formal 120-task run:

```bash
MODE=full MAX_TASKS=120 NPU_DEVICE=0 \
  RESULT_ROOT="$PWD/results/bcro120_npu_full_$(date +%Y%m%d_%H%M%S)" \
  bash c2kv_eval/portable/npu/run_bcro120_npu.sh
```

The launcher starts one private NPU SGLang process, measures Full, legally warms
the BCRO gist cache, then measures BCRO. Warm outputs are separate and excluded.
The measured BCRO path retains the existing strict cache-hit receipts; a miss or
online extraction is not relabeled as a hit.

Final files:

- `results/bcro120_npu_<timestamp>/summary/summary.json`
- `results/bcro120_npu_<timestamp>/summary/summary.csv`
- `results/bcro120_npu_<timestamp>/npu_preflight.json`
- full raw task/request logs under the same run directory

`server_ttft_ms` is scheduler request creation through first sampled output;
`prefill_path_ms` is model-forward entry through that output. Native C2KV TTFT
adds planning, cache lookup/load, request assembly and host-side KV injection
reported by the native endpoint. System Draft TTFT is Full preparation plus
Server TTFT, or BCRO route + preparation + legal prewarm + Native TTFT. Final
Action Ready is the complete decision latency and is never added to recovery a
second time.
