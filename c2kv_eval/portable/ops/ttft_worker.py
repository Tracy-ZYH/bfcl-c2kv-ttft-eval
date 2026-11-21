"""Serial arms per physical GPU; parallel task shards."""
import fcntl
import os
from pathlib import Path
import subprocess
import time
from ttft_commands import C2KV, REPO, SGLANG, task_command
from ttft_plan import save
from ttft_process import (
    STOP,
    add_loopback_no_proxy,
    http,
    monitor,
    wait_ports_free,
    stop_process,
    wait_ready,
)


def run_stage(args, info, job, env):
    root = Path(job['output'])
    wait_ports_free(job['port'])
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'job.json', job | {'gpu_identity': info})
    server = worker = None
    base = f'http://127.0.0.1:{job["port"]}'
    client_env = dict(env, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(REPO))
    print(f'GPU{info["index"]} {info["uuid"]} {job["arm"]} -> {root}', flush=True)
    try:
        with (root / 'server.log').open('w') as log:
            server = subprocess.Popen(job['server_command'], cwd=SGLANG, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_ready(server, base, args.startup_timeout)
            started = time.perf_counter()
            for _ in range(2):
                http(base + '/generate', {'text': 'Reply with OK.',
                     'sampling_params': {'temperature': 0, 'max_new_tokens': 16}}, timeout=180)
            save(root / 'warmup.json', dict(seconds=time.perf_counter() - started,
                 excluded_from_metrics=True, native_extraction_warmup=False))
            command = task_command(job['arm'], job['tasks'], root / 'run', job['port'], args.task_timeout)
            command[1] = str(Path(__file__).with_name('ttft_c1_entry.py'))
            with (root / 'benchmark.log').open('w') as bench:
                worker = subprocess.Popen(command, cwd=C2KV, env=client_env,
                                          stdout=bench, stderr=subprocess.STDOUT, start_new_session=True)
                monitor(worker, server, len(job['tasks']) * args.task_timeout + 300, root / 'benchmark.log')
    finally:
        stop_process(worker)
        stop_process(server)


def run_device(args, info, jobs):
    env = add_loopback_no_proxy(os.environ.copy())
    env.update(CUDA_VISIBLE_DEVICES=info['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONPATH=str(SGLANG / 'python'), PYTHONUNBUFFERED='1', C2KV_SPEED_TIMING='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               CUDA_HOME='/usr/local/cuda-13.0', CUDACXX='/usr/local/cuda-13.0/bin/nvcc')
    # Keep controller cleanup inside ttft_process.stop_process()'s grace
    # period so a failed SGLang server cannot strand event-native children.
    env.setdefault('C2KV_CONTROLLER_STOP_TIMEOUT_SECONDS', '30')
    env['PATH'] = '/usr/local/cuda-13.0/bin:' + env.get('PATH', '')
    if Path('/usr/bin/g++-12').exists():
        env.update(CC='/usr/bin/gcc-12', CXX='/usr/bin/g++-12')
    try:
        with (Path('/tmp') / f'bfcl-ttft-{info["uuid"]}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for job in jobs:
                if STOP.is_set():
                    raise InterruptedError('Suite interrupted')
                run_stage(args, info, job, env)
    except BaseException:
        STOP.set()
        raise
