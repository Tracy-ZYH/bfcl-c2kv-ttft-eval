#!/usr/bin/env python3
"""ToolSandbox same-input C2KV gist-cache-hit TTFT experiment.

The script owns one SGLang process. It first captures exact visible decisions
from a Full ToolSandbox trajectory, warms native C2KV chunks outside timing,
and then replays the identical decisions through fresh finite controllers.
Timed C2KV/Racer controllers disallow extraction misses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

from c2kv_eval.portable.proxy import action_canonical

PROJECT = HERE.parents[3]
HERE = Path(__file__).resolve().parent
BFCL = HERE.parents[2]
C2KV = BFCL / 'runtime'
SGLANG = BFCL / 'third_party/sglang'
MODEL = Path('/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000')
EMBED = Path('/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B')
SERVER_PY = Path('/home/zhuyuhan/miniconda3/envs/c2kv-sglang-gpu/bin/python')
CLIENT_PY = Path('/home/zhuyuhan/miniconda3/envs/bfcl-c2kv-gpu/bin/python')
POLICY = BFCL / 'runtime/configs/eval_policy_bfcl_long_fixed256.json'
CAPACITY = BFCL / 'runtime/configs/eval_capacity_bfcl_long.json'
QUICK_SCENARIOS = (
    'search_message_with_recency_latest_multiple_user_turn',
    'search_message_with_recency_oldest_multiple_user_turn',
    'modify_contact_with_message_recency_multiple_user_turn',
    'remove_contact_by_phone_multiple_user_turn',
    'send_message_with_contact_content_cellular_off_multiple_user_turn',
    'update_contact_relationship_with_relationship_multiple_user_turn',
    'find_days_till_holiday_multiple_user_turn',
    'find_temperature_f_with_location_and_time_diff_multiple_user_turn',
    'find_distance_with_location_name_multiple_user_turn',
    'add_reminder_content_and_date_and_time_multiple_user_turn',
)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError('no TTFT samples')
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - point) + ordered[hi] * (point - lo)


def http(url, body=None, timeout=3600):
    request = Request(url, data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        payload = response.read()
        return json.loads(payload) if payload else {}


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def wait_health(process, url, timeout, log):
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            http(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f'process did not become ready; see {log}')


def discover_scenarios(root: Path, python: Path) -> list[str]:
    code = ("import json; from tool_sandbox.scenarios import named_scenarios; "
            "print(json.dumps(sorted(named_scenarios(preferred_tool_backend=\"DEFAULT\"))))")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    output = subprocess.check_output(
        [str(python), "-c", code], cwd=root, env=env, text=True
    )
    value = json.loads(output)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise RuntimeError("cannot discover ToolSandbox named scenarios; use --scenario-file")
    return value


def selected_scenarios(args) -> list[str]:
    if args.scenario_file:
        selected = [line.strip() for line in args.scenario_file.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
    elif args.phase == 'quick':
        selected = list(QUICK_SCENARIOS[:args.quick_scenarios])
    else:
        selected = discover_scenarios(args.toolsandbox_root)[:50]
    expected = args.quick_scenarios if args.phase == 'quick' else 50
    if len(selected) != expected or len(set(selected)) != len(selected):
        raise ValueError(f'expected exactly {expected} unique scenarios, got {len(selected)}')
    return selected


def server_command(args):
    return [str(args.server_python), '-m', 'sglang.launch_server',
            '--model-path', str(args.model), '--tokenizer-path', str(args.model),
            '--host', '127.0.0.1', '--port', str(args.server_port), '--dtype', 'bfloat16',
            '--enable-c2kv', '--c2kv-gist-type', 'dynamic-interleave',
            '--c2kv-gist-param', 'qkv', '--c2kv-query-proj', 'base',
            '--c2kv-tools-dump', 'full', '--enable-return-hidden-states',
            '--c2kv-shadow-feature-layer', '-2', '--attention-backend', 'triton',
            '--sampling-backend', 'pytorch', '--enable-metrics', '--disable-cuda-graph',
            '--disable-piecewise-cuda-graph', '--disable-radix-cache',
            '--max-running-requests', '1', '--page-size', '1',
            '--mem-fraction-static', str(args.mem_fraction_static),
            '--c2kv-pool-fraction', str(args.c2kv_pool_fraction),
            '--context-length', '262144', '--max-total-tokens', '262144',
            '--chunked-prefill-size', '4096', '--random-seed', '0']


def run_c1_cli(args, method, scenarios, out, port):
    command = [str(args.client_python), str(C2KV / 'runtime/history_system/run_c1.py'),
        '--benchmark', 'toolsandbox', '--method', method, '--checkpoint', str(args.model),
        '--sglang-backend-url', f'http://127.0.0.1:{args.server_port}',
        '--embedding-model', str(args.embedding_model), '--embedding-device', 'cpu',
        '--detector', 't02_risk', '--selector-threshold', '0.5',
        '--eval-policy-override', str(args.eval_policy),
        '--eval-capacity-override', str(args.eval_capacity),
        '--toolsandbox-dir', str(args.toolsandbox_root),
        '--toolsandbox-python', str(args.toolsandbox_python),
        '--user-base-url', f'http://127.0.0.1:{args.server_port}',
        '--portable-root', str(BFCL), '--sglang-root', str(SGLANG),
        '--port', str(port), '--task-timeout', str(args.task_timeout), '--out', str(out)]
    for scenario in scenarios:
        command += ['--ts-scenario', scenario]
    return command


def capture_inputs(args, scenarios, root, env):
    capture = root / 'capture_full'
    if not (capture / 'result.json').exists():
        command = run_c1_cli(args, 'full', scenarios, capture, args.controller_port)
        with (root / 'capture_full.log').open('w') as log:
            completed = subprocess.run(command, cwd=C2KV, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Full capture failed; see {root / "capture_full.log"}')
    decisions = {}
    for scenario in scenarios:
        path = capture / 'task_shards' / scenario / 'server' / 'steps.jsonl'
        picked = []
        for record in rows(path):
            transport = record.get('harness_transport_normalization') or {}
            payload = transport.get('same_input_replay_request')
            if not isinstance(payload, dict):
                raise RuntimeError(f'missing same-input payload in {path}')
            messages = payload.get('messages') or []
            completed_history = sum(message.get('role') == 'user' for message in messages) >= 2
            if completed_history:
                picked.append(payload)
        if picked:
            decisions[scenario] = picked
    if not decisions:
        raise RuntimeError('Full capture produced no decision with non-empty completed history')
    save(root / 'decisions.json', {'schema': 'toolsandbox-same-input-decisions-v1',
         'scenarios': scenarios, 'decisions': decisions,
         'decision_count': sum(map(len, decisions.values()))})
    return decisions



def obtain_decisions(args, scenarios, root, env):
    shared = args.decisions_file
    if shared and shared.exists():
        payload = json.loads(shared.read_text())
        if payload.get("scenarios") != scenarios:
            raise RuntimeError(f"shared decision scenario mismatch: {shared}")
        return payload["decisions"]
    if args.arm != "full" and shared:
        deadline = time.monotonic() + args.decisions_wait
        while not shared.exists() and time.monotonic() < deadline:
            time.sleep(2)
        if not shared.exists():
            raise RuntimeError(f"timed out waiting for shared decisions: {shared}")
        payload = json.loads(shared.read_text())
        if payload.get("scenarios") != scenarios:
            raise RuntimeError(f"shared decision scenario mismatch: {shared}")
        return payload["decisions"]
    decisions = capture_inputs(args, scenarios, root, env)
    if shared:
        shared.parent.mkdir(parents=True, exist_ok=True)
        temporary = shared.with_suffix(shared.suffix + f".tmp.{os.getpid()}")
        shutil.copyfile(root / "decisions.json", temporary)
        os.replace(temporary, shared)
    return decisions

def c1_args(args, method, scenario, out, port):
    sys.path.insert(0, str(BFCL / 'runtime/history_system'))
    try:
        module = importlib.import_module('run_c1')
        cli = run_c1_cli(args, method, [scenario], out, port)
        parsed = module.build_parser().parse_args(cli[2:])
        module.validate_args(parsed)
        controller, profile = module.build_profile(parsed)
        out.mkdir(parents=True, exist_ok=False)
        controller_path = out / 'controller.json'
        module.save(controller_path, controller)
        module.save(out / 'profile.json', profile)
        command, _ = module.commands_for_task(parsed, scenario, controller_path)
        return module, parsed, command
    finally:
        sys.path.pop(0)


def replace_flag(command, name, value):
    command = list(command)
    index = command.index(name)
    command[index + 1] = str(value)
    return command


def available_loopback_port(start):
    for candidate in range(start, min(start + 256, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError(f"no available controller port at or above {start}")

def replay_once(args, method, scenario, payloads, out, port, env,
                *, timed, repeat, disallow_miss):
    port = available_loopback_port(port)
    module, parsed, command = c1_args(args, method, scenario, out, port)
    if disallow_miss:
        command = replace_flag(command, '--max-extraction-calls', 0)
    log_path = out / 'controller.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=C2KV / 'runtime',
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    try:
        wait_health(process, f'http://127.0.0.1:{port}/health', 300, log_path)
        model = method if method != 'proposed' else 'c1_t02_risk'
        for payload in payloads:
            # Replay-only scoring metadata must never reach the OpenAI schema.
            request = {
                key: value for key, value in payload.items()
                if not key.startswith('_replay_')
            }
            request['model'] = model
            request['max_completion_tokens'] = 4096
            request['temperature'] = 0
            request['top_p'] = 1
            request['seed'] = 0
            request['store'] = False
            http(f'http://127.0.0.1:{port}/v1/chat/completions', request,
                 timeout=args.task_timeout)
        try:
            http(f'http://127.0.0.1:{port}/internal/graceful-stop', {}, timeout=10)
        except Exception:
            pass
        process.wait(timeout=180)
    finally:
        stop(process)
    if process.returncode not in (0, None):
        raise RuntimeError(f'controller failed; see {log_path}')
    step_path = out / 'task_shards' / scenario / 'server' / 'steps.jsonl'
    records = rows(step_path)
    if not timed:
        return []
    if len(records) != len(payloads):
        raise RuntimeError(
            f'replay record count mismatch in {step_path}: '
            f'{len(records)} records for {len(payloads)} requests'
        )
    samples = []
    for record, payload in zip(records, payloads):
        traces = [trace for trace in record.get('generation_trace', [])
                  if trace.get('phase') != 'regeneration']
        if not traces:
            raise RuntimeError(f'no draft generation trace in {step_path}')
        stats = (traces[0].get('generation') or {}).get('stats') or {}
        timing = stats.get('speed_timing') or {}
        value = timing.get('native_ttft_ms')
        prepare_seconds = (record.get('controller_timing') or {}).get('prepare_seconds')
        if type(prepare_seconds) not in (int, float) or not math.isfinite(prepare_seconds) or prepare_seconds < 0:
            raise RuntimeError(f'missing controller prepare timing in {step_path}')
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f'missing native_ttft_ms in {step_path}')
        online_ttft_ms = float(value) + float(prepare_seconds) * 1000
        extraction = stats.get('sglang_extraction') or {}
        controller = traces[0].get('controller') or {}
        full_reference = controller.get('same_prefix_full_reference') or {}
        resident_prompt_tokens = traces[0].get('planned_resident_prompt_tokens')
        full_prompt_tokens = full_reference.get('full_prompt_tokens')
        if not isinstance(full_prompt_tokens, int):
            full_prompt_tokens = resident_prompt_tokens
        eval_context = payload.get('c2kv_eval_context') or {}
        source_task_id = eval_context.get('task_id') or scenario
        source_user_turn = eval_context.get('user_turn')
        source_step = eval_context.get('step')
        source_attempt = eval_context.get('attempt', 0)
        source_decision_id = (
            f'{source_task_id}/turn-{source_user_turn}/step-{source_step}'
            f'/attempt-{source_attempt}'
        )
        hits = int(extraction.get('cache_hits') or 0)
        misses = int(extraction.get('cache_misses') or 0)
        unique = int(extraction.get('unique_chunks') or 0)
        reference_history_tokens = full_reference.get(
            'full_history_tokens', controller.get('history_tokens')
        )
        no_compressible_history = (
            isinstance(reference_history_tokens, (int, float))
            and reference_history_tokens <= 0
        )
        if method != 'full' and disallow_miss and (
            misses != 0 or hits != unique or (unique <= 0 and not no_compressible_history)
        ):
            raise RuntimeError(f'cache hit was not 100% in {step_path}: {extraction}')
        reference_action = payload.get('_replay_reference_action')
        generated_action = action_canonical(record.get('response') or {})
        if reference_action is not None and not isinstance(reference_action, dict):
            raise RuntimeError(f'invalid replay reference action in {step_path}')
        reference_action_json = (
            json.dumps(reference_action, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'))
            if reference_action is not None else ''
        )
        generated_action_json = json.dumps(
            generated_action, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')
        )
        samples.append({'method': method, 'scenario': scenario,
                        'decision_key': record.get('decision_key'),
                        'source_decision_id': source_decision_id,
                        'source_task_id': source_task_id,
                        'source_user_turn': source_user_turn,
                        'source_step': source_step,
                        'source_attempt': source_attempt,
                        'repeat': repeat, 'timed': timed,
                        'ttft_ms': online_ttft_ms, 'native_ttft_ms': float(value),
                        'context_prepare_ms': float(prepare_seconds) * 1000,
                        'full_reference_prompt_tokens': full_prompt_tokens,
                        'resident_prompt_tokens': resident_prompt_tokens,
                        'full_reference_history_tokens': reference_history_tokens,
                        'resident_history_tokens': controller.get(
                            'actual_history_tokens', controller.get('history_tokens')
                        ),
                        'cache_hits': hits,
                        'cache_misses': misses, 'unique_chunks': unique,
                        'cache_hit_rate': (hits / unique if unique else 1.0),
                        'reference_action_json': reference_action_json,
                        'generated_action_json': generated_action_json,
                        'reference_action_sha256': (
                            hashlib.sha256(reference_action_json.encode()).hexdigest()
                            if reference_action_json else ''
                        ),
                        'generated_action_sha256': hashlib.sha256(
                            generated_action_json.encode()
                        ).hexdigest(),
                        'action_exact_match': (
                            int(generated_action == reference_action)
                            if reference_action is not None else ''
                        )})
    return samples


def physical_gpu_uuid(index: int) -> str:
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'
    ], text=True)
    inventory = {}
    for line in output.splitlines():
        raw_index, uuid = [part.strip() for part in line.split(',', 1)]
        inventory[int(raw_index)] = uuid
    if index not in inventory or not inventory[index].startswith('GPU-'):
        raise ValueError(f'physical GPU {index} was not reported by nvidia-smi')
    return inventory[index]

def run(args):
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite {args.out}')
    args.out.mkdir(parents=True)
    scenarios = selected_scenarios(args)
    if args.arm != "full" and args.decisions_file and not args.decisions_file.exists():
        deadline = time.monotonic() + args.decisions_wait
        while not args.decisions_file.exists() and time.monotonic() < deadline:
            time.sleep(2)
        if not args.decisions_file.exists():
            raise RuntimeError(f"timed out waiting for shared decisions: {args.decisions_file}")
    gpu_uuid = physical_gpu_uuid(args.gpu)
    save(args.out / 'manifest.json', {
        'schema': 'toolsandbox-cache-hit-ttft-v1', 'phase': args.phase,
        'scenarios': scenarios, 'scenario_sha256': hashlib.sha256(
            json.dumps(scenarios, separators=(',', ':')).encode()).hexdigest(),
        'ratio': 8, 'chunk_tokens': 768, 'chunk_overlap': 64,
        'max_chunks': 48, 'repetitions': args.repetitions,
        'radix_cache': 'disabled', 'timing': 'controller context preparation + native online request through first token',
        'offline_cache': 'server-side C2KV content-addressed gist pool; warmup excluded',
        'checkpoint': str(args.model),
        'physical_gpu_index': args.gpu, 'physical_gpu_uuid': gpu_uuid,
    })
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CC='/usr/bin/gcc-12', CXX='/usr/bin/g++-12',
               CUDAHOSTCXX='/usr/bin/g++-12',
               NVCC_CCBIN='/usr/bin/g++-12',
               CUDA_HOME='/usr/local/cuda-13.0', CUDA_PATH='/usr/local/cuda-13.0',
               PATH=str(BFCL / '.toolchains/gcc12/bin') + os.pathsep + '/usr/local/cuda-13.0/bin' + os.pathsep + os.environ.get('PATH', ''),
               PYTHONPATH=str(SGLANG / 'python'), PYTHONUNBUFFERED='1',
               C2KV_SPEED_TIMING='1',
               TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    server_log = args.out / 'server.log'
    with server_log.open('w') as log:
        server = subprocess.Popen(server_command(args), cwd=SGLANG, env=env,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        wait_health(server, f'http://127.0.0.1:{args.server_port}/health',
                    args.startup_timeout, server_log)
        for _ in range(2):
            http(f'http://127.0.0.1:{args.server_port}/generate',
                 {'text': 'Reply OK.', 'sampling_params': {'temperature': 0,
                  'max_new_tokens': 8}}, timeout=180)
        client_env = env.copy()
        client_env['CUDA_VISIBLE_DEVICES'] = ''
        client_env['PYTHONPATH'] = str(BFCL)
        client_env['TOOLSANDBOX_QWEN_TOOL_RESULT_COMPAT'] = '1'
        controller_env = env.copy()
        runtime = C2KV / 'runtime'
        controller_env['PYTHONPATH'] = os.pathsep.join((str(runtime / 'python'), str(runtime)))
        decisions = obtain_decisions(args, scenarios, args.out, client_env)
        warm_decisions = decisions
        warm_file = getattr(args, "warm_decisions_file", None)
        if warm_file is not None:
            warm_payload = json.loads(warm_file.read_text())
            if warm_payload.get("scenarios") != scenarios:
                raise RuntimeError(f"warm decision scenario mismatch: {warm_file}")
            warm_decisions = warm_payload.get("decisions")
            if not isinstance(warm_decisions, dict) or set(warm_decisions) != set(decisions):
                raise RuntimeError(f"warm decision task set mismatch: {warm_file}")
        samples = []
        serial = 0
        arm_specs = {
            "full": [("full_recompute", "full", "Full Recompute")],
            "c2kv": [("c2kv_cache_hit", "c2kv_only", "C2KV Cache Hit")],
            "racer": [("racer_cache_hit", "proposed", "Racer Cache Hit")],
            "all": [("full_recompute", "full", "Full Recompute"),
                    ("c2kv_cache_hit", "c2kv_only", "C2KV Cache Hit"),
                    ("racer_cache_hit", "proposed", "Racer Cache Hit")],
        }[args.arm]
        if any(method != "full" for _, method, _ in arm_specs):
            for scenario, payloads in warm_decisions.items():
                serial += 1
                replay_once(args, "c2kv_only", scenario, payloads,
                            args.out / "offline_warm" / scenario,
                            args.controller_port + serial, controller_env,
                            timed=False, repeat=-1, disallow_miss=False)
        for label, method, _ in arm_specs:
            for repeat in range(args.repetitions):
                for scenario, payloads in decisions.items():
                    serial += 1
                    samples += replay_once(
                        args, method, scenario, payloads,
                        args.out / "timed" / label / f"rep{repeat}" / scenario,
                        args.controller_port + serial, controller_env,
                        timed=True, repeat=repeat, disallow_miss=method != "full")
        with (args.out / "samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader(); writer.writerows(samples)
        summary = []
        for _, method, display in arm_specs:
            values = [row["ttft_ms"] for row in samples if row["method"] == method]
            summary.append({"method": display, "decision_repeats": len(values),
                            "ttft_p50_ms": percentile(values, .5),
                            "ttft_p95_ms": percentile(values, .95),
                            "reduction_vs_full": None})
        cache_rows = [row for row in samples if row["method"] != "full"]
        result = {"schema": "toolsandbox-cache-hit-ttft-summary-v1",
                  "scenario_count": len(scenarios),
                  "decision_count": sum(map(len, decisions.values())),
                  "repetitions": args.repetitions,
                  "gist_cache_hit_rate": (sum(row["cache_hits"] for row in cache_rows) /
                    max(1, sum(row["unique_chunks"] for row in cache_rows))) if cache_rows else None,
                  "rows": summary, "racer_reduction": None, "passes_50pct": None}
        save(args.out / 'summary.json', result)
        with (args.out / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        print(json.dumps(result, indent=2))
    finally:
        stop(server)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('quick', 'formal'), default='quick')
    p.add_argument("--arm", choices=("all", "full", "c2kv", "racer"), default="all")
    p.add_argument("--decisions-file", type=Path)
    p.add_argument("--decisions-wait", type=int, default=7200)
    p.add_argument('--gpu', type=int, default=7)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--toolsandbox-root', type=Path,
                   default=PROJECT / 'benchmarks/ToolSandbox')
    p.add_argument('--toolsandbox-python', type=Path,
                   default=Path('/home/zhuyuhan/project/benchmarks/.venv-toolsandbox/bin/python'))
    p.add_argument('--scenario-file', type=Path)
    p.add_argument('--quick-scenarios', type=int, choices=range(10, 21), default=10)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--server-port', type=int, default=39770)
    p.add_argument('--controller-port', type=int, default=39800)
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument('--task-timeout', type=int, default=3600)
    p.add_argument('--mem-fraction-static', type=float, default=.65)
    p.add_argument('--c2kv-pool-fraction', type=float, default=.15)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--embedding-model', type=Path, default=EMBED)
    p.add_argument('--server-python', type=Path, default=SERVER_PY)
    p.add_argument('--client-python', type=Path, default=CLIENT_PY)
    p.add_argument('--eval-policy', type=Path, default=POLICY)
    p.add_argument('--eval-capacity', type=Path, default=CAPACITY)
    return p


def main():
    args = parser().parse_args()
    required = [args.toolsandbox_root / 'tool_sandbox', args.model / 'config.json',
                args.embedding_model / 'config.json', args.eval_policy,
                args.eval_capacity, args.server_python, args.client_python]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit('missing prerequisites:\n' + '\n'.join(missing))
    probe = subprocess.run([str(args.toolsandbox_python), '-c', 'import tool_sandbox'], cwd=args.toolsandbox_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if probe.returncode:
        raise SystemExit('ToolSandbox Python import failed: ' + probe.stderr.strip())
    if args.repetitions != 3:
        raise SystemExit('the paper experiment requires exactly 3 repetitions')
    run(args)


if __name__ == '__main__':
    main()
