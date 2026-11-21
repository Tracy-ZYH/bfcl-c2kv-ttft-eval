import argparse
import sys
import tempfile
import os
import subprocess
import time
import unittest
from pathlib import Path

OPS = Path('/home/zhuyuhan/project/bfcl-c2kv-ttft-eval/c2kv_eval/portable/ops')
C1 = Path('/home/zhuyuhan/project/bfcl-c2kv-ttft-eval/runtime/history_system')
sys.path[:0] = [str(OPS), str(C1)]

import run_c1
from ttft_commands import ARMS, task_command
from ttft_plan import build_plan
from ttft_process import add_loopback_no_proxy, gpu_inventory, stop_process


class TestLongTTFT(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(gpus='2,3', cases=2, repetitions=1, port_base=39420,
                                  out=root, run_ids=None, arms=None, arm_per_gpu=False, bfcl_root=Path('/home/zhuyuhan/project/bfcl-c2kv-ttft-eval'))

    def test_fixed_ids_and_two_gpu_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan(self.args(Path(tmp)))
        self.assertEqual(plan['task_ids'], ['multi_turn_long_context_0', 'multi_turn_long_context_1'])
        self.assertEqual({job['gpu'] for job in plan['jobs']}, {2, 3})
        self.assertEqual(len(plan['jobs']), 8)
        bound_ports = [port for job in plan['jobs'] for port in (job['port'], job['port'] + 1)]
        self.assertEqual(len(bound_ports), len(set(bound_ports)))
        self.assertEqual([job['port'] for job in plan['jobs'] if job['gpu'] == 2], [39420, 39422, 39424, 39426])
        self.assertEqual(plan['retrieval_quality'], 'not_assessed_no_gold_relevance_labels')
    def test_exact_run_ids_for_targeted_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args.run_ids = ['multi_turn_long_context_100']
            plan = build_plan(args)
        self.assertEqual(plan['task_ids'], ['multi_turn_long_context_100'])
        self.assertTrue(all(job['tasks'] == ['multi_turn_long_context_100'] for job in plan['jobs']))

    def test_server_watchdog_covers_task_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args.task_timeout = 7200
            plan = build_plan(args)
        for job in plan['jobs']:
            command = job['server_command']
            self.assertEqual(command[command.index('--watchdog-timeout') + 1], '7500')


    def test_arm_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args.gpus = '7'
            args.arms = ['c2kv_only', 'racer']
            plan = build_plan(args)
        self.assertEqual([job['arm'] for job in plan['jobs']], ['c2kv_only', 'racer'])


    def test_one_arm_per_gpu_runs_identical_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args.gpus = '0,1,2,3'
            args.arm_per_gpu = True
            plan = build_plan(args)
        self.assertEqual([(job['gpu'], job['arm']) for job in plan['jobs']],
                         [(0, 'full_cold'), (1, 'full_cached'),
                          (2, 'c2kv_only'), (3, 'racer')])
        self.assertTrue(all(job['tasks'] == plan['task_ids'] for job in plan['jobs']))


    def test_physical_gpu_uuid_mapping(self):
        inventory = gpu_inventory('2, GPU-two, 0000:02:00.0, Test GPU, 90000\n3, GPU-three, 0000:03:00.0, Test GPU, 91000\n')
        self.assertEqual(inventory[2]['uuid'], 'GPU-two')
        self.assertEqual(inventory[3]['pci_bus_id'], '0000:03:00.0')


    def test_loopback_bypasses_proxy_without_dropping_existing_hosts(self):
        env = add_loopback_no_proxy({
            'HTTP_PROXY': 'http://127.0.0.1:17890',
            'NO_PROXY': 'internal.example',
        })
        self.assertEqual(env['HTTP_PROXY'], 'http://127.0.0.1:17890')
        for host in ('internal.example', '127.0.0.1', 'localhost', '::1'):
            self.assertIn(host, env['NO_PROXY'].split(','))
            self.assertIn(host, env['no_proxy'].split(','))
    def test_stop_process_terminates_descendant_session(self):
        child_code = (
            "import os,subprocess,sys,time\n"
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
            " start_new_session=True)\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", child_code], stdout=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        child_pid = int(parent.stdout.readline().strip())
        try:
            stop_process(parent)
            self.assertIsNotNone(parent.poll())
            deadline = time.monotonic() + 2
            while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
        finally:
            parent.stdout.close()
            for pid in (child_pid, parent.pid):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass


    def test_resolved_arm_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ARMS:
                cmd = task_command(arm, ['multi_turn_long_context_0'], Path(tmp) / arm, 39420)
                args = run_c1.build_parser().parse_args(cmd[2:])
                server, worker = run_c1.commands_for_task(
                    args, 'multi_turn_long_context_0', Path(tmp) / 'controller.json')
                self.assertIn('benchmarks.memory_runtime.event_native_bfcl', worker)
                view = server[server.index('--view-mode') + 1]
                if arm.startswith('full_'):
                    self.assertEqual(view, 'full_original')
                    self.assertNotIn('--s0-config', server)
                    self.assertNotIn('--compression-policy', server)
                else:
                    self.assertIn('--s0-config', server)
                    self.assertIn('--compression-policy', server)
                self.assertIn('--eval-capacity', server)
                if arm.startswith('full_'):
                    self.assertNotIn('--eval-policy', server)
                else:
                    self.assertEqual(Path(server[server.index('--eval-policy') + 1]).name,
                                     'eval_policy_bfcl_long_fixed256.json')
                capacity = Path(server[server.index('--eval-capacity') + 1])
                expected_capacity = ('eval_capacity_full_context.json'
                                     if arm.startswith('full_') else 'eval_capacity_bfcl_long.json')
                self.assertEqual(capacity.name, expected_capacity)
                self.assertEqual(server[server.index('--generation-backend') + 1], 'sglang')


if __name__ == '__main__':
    unittest.main()
