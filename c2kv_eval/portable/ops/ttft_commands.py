"""Command construction only; importing this module starts no processes."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[3]
C2KV = PROJECT
SGLANG = PROJECT / 'third_party/sglang'
MODEL = Path('/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000')
EMBED = Path('/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B')
SERVER_PY = '/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python'
CLIENT_PY = '/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python'
ARMS = {'full_cold': 'full', 'full_cached': 'full', 'c2kv_only': 'c2kv_only', 'racer': 'proposed'}
C2KV_COMPRESSION_RATIO = 8
C2KV_KV_BYTES_PER_TOKEN = 147456
C2KV_HISTORY_BUDGET_TOKENS = 256
C2KV_HISTORY_BUDGET_BYTES = C2KV_KV_BYTES_PER_TOKEN * C2KV_HISTORY_BUDGET_TOKENS
EVAL_POLICY = C2KV / 'runtime/configs/eval_policy_bfcl_long_fixed256.json'


def server_command(arm, port, task_timeout=3600):
    watchdog_timeout = max(600, int(task_timeout) + 300)
    cmd = [SERVER_PY, '-m', 'sglang.launch_server',
           '--model-path', str(MODEL), '--tokenizer-path', str(MODEL),
           '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
           '--enable-c2kv', '--c2kv-gist-type', 'dynamic-interleave', '--c2kv-gist-param', 'qkv',
           '--c2kv-query-proj', 'base', '--c2kv-tools-dump', 'full',
           '--enable-return-hidden-states', '--c2kv-shadow-feature-layer', '-2',
           '--enable-metrics', '--attention-backend', 'triton', '--sampling-backend', 'pytorch',
           '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
           '--max-running-requests', '1', '--page-size', '1',
           '--mem-fraction-static', '0.65', '--c2kv-pool-fraction', '0.05',
           '--context-length', '262144', '--max-total-tokens', '262144',
           '--chunked-prefill-size', '4096', '--random-seed', '0']
    cmd += ['--watchdog-timeout', str(watchdog_timeout)]
    return cmd + (['--disable-radix-cache'] if arm == 'full_cold' else [])


def task_command(arm, tasks, out, port, timeout=3600):
    cmd = [CLIENT_PY, str(C2KV / 'runtime/history_system/run_c1.py'),
           '--benchmark', 'bfcl', '--method', ARMS[arm], '--checkpoint', str(MODEL),
           '--sglang-backend-url', f'http://127.0.0.1:{port}',
           '--embedding-model', str(EMBED), '--embedding-device', 'cpu',
           '--detector', 't02_risk', '--selector-threshold', '0.5',
           '--eval-policy-override', str(EVAL_POLICY),
           '--eval-capacity-override', str(C2KV / 'runtime/configs/eval_capacity_bfcl_long.json'),
           '--benchmark-dir', str(REPO), '--bfcl-python', CLIENT_PY,
           '--portable-root', str(REPO), '--sglang-root', str(SGLANG),
           '--port', str(port + 1), '--task-timeout', str(timeout), '--out', str(out)]
    for task in tasks:
        cmd += ['--task-id', task]
    return cmd
