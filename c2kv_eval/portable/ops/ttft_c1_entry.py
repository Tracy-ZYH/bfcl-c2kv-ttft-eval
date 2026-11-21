"""Experiment-only cache isolation; executed by the user-owned speed runner."""
import sys
from ttft_commands import C2KV
from ttft_process import http


def main():
    sys.path.insert(0, str(BFCL / 'runtime/history_system'))
    import run_c1
    original = run_c1.run_task

    def isolated_task(args, task, controller_path):
        http(args.sglang_backend_url + '/flush_cache', timeout=60)
        return original(args, task, controller_path)

    run_c1.run_task = isolated_task
    return run_c1.main()


if __name__ == '__main__':
    raise SystemExit(main())
