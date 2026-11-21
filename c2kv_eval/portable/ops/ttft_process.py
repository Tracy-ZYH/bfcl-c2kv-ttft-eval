"""Owned-process lifecycle and UUID-only device binding."""
import csv
import json
import os
import signal
import socket
import subprocess
import threading
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener

STOP = threading.Event()


def add_loopback_no_proxy(env):
    """Keep external proxies while ensuring local benchmark traffic stays local."""
    entries = []
    for name in ('NO_PROXY', 'no_proxy'):
        entries.extend(part.strip() for part in env.get(name, '').split(',') if part.strip())
    for host in ('127.0.0.1', 'localhost', '::1'):
        if host not in entries:
            entries.append(host)
    value = ','.join(entries)
    env['NO_PROXY'] = value
    env['no_proxy'] = value
    return env


def gpu_inventory(text):
    result = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        index, uuid, bus, name, free = (x.strip() for x in row)
        if not uuid.startswith('GPU-'):
            raise ValueError('No GPU UUID; refusing ordinal fallback')
        result[int(index)] = dict(index=int(index), uuid=uuid, pci_bus_id=bus, name=name, free_mib=int(free))
    return result


def ports_free(port):
    for number in (port, port + 1):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', number))


def wait_ports_free(port, timeout=90):
    """Wait for an owned server's children to release both stage ports."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            ports_free(port)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    detail = f": {last_error}" if last_error is not None else ""
    raise OSError(
        getattr(last_error, "errno", None) or 98,
        f"ports {port}/{port + 1} did not become free within {timeout}s{detail}",
    )


def http(url, body=None, timeout=30):
    req = Request(url, data=None if body is None else json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    # Every endpoint used here is loopback. Ignore ambient HTTP_PROXY settings
    # rather than relying on the caller's shell to provide NO_PROXY.
    with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
        raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode()


def _process_tree(root_pid):
    """Return root and descendants before a parent can orphan new sessions."""
    pending = [int(root_pid)]
    found = []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen or pid <= 1:
            continue
        seen.add(pid)
        found.append(pid)
        try:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        pending.extend(int(child) for child in children)
    return found


def _signal_process_tree(root_pid, signum):
    pids = _process_tree(root_pid)
    groups = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    for pgid in groups:
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            pass
    for pid in pids:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def stop_process(process):
    if process is None:
        return
    # run_c1 gives its BFCL worker and event-native controller independent
    # sessions. Capture the complete owned tree before children are reparented.
    owned_pids = _process_tree(process.pid)
    try:
        _signal_process_tree(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        for pid in owned_pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def monitor(process, server, timeout, logfile):
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if STOP.wait(1):
            raise InterruptedError('Suite interrupted')
        server_code = server.poll()
        if server_code is not None:
            raise RuntimeError(
                f'Server exited with code {server_code} while worker was active: {logfile}'
            )
        if time.monotonic() > deadline:
            raise RuntimeError(f'Task timeout after {timeout}s: {logfile}')
    if process.returncode:
        raise RuntimeError(f'Worker exit={process.returncode}: {logfile}')


def wait_ready(server, base, timeout):
    deadline = time.monotonic() + timeout
    while server.poll() is None and not STOP.is_set() and time.monotonic() < deadline:
        try:
            http(base + '/health', timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError('Server startup failed; see server.log')
