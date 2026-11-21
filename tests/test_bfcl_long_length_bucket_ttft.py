import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

OPS = Path(__file__).resolve().parents[1] / "c2kv_eval/portable/ops"
sys.path.insert(0, str(OPS))
import bfcl_long_length_bucket_ttft as bucket
import bfcl_long_longest_offline_ttft as longest


class TestLengthBucketTTFT(unittest.TestCase):
    def synthetic_rows(self):
        edges = [0, 10, 20, float("inf")]
        rows = []
        for index, values in enumerate(((1, 2, 3), (11, 12, 13), (21, 22, 23))):
            for offset, tokens in enumerate(values):
                task = f"task-{index}-{offset}"
                rows.append({
                    "source_decision_id": f"{task}/turn-1/step-0/attempt-0",
                    "task_id": task,
                    "user_turn": 1,
                    "step": 0,
                    "attempt": 0,
                    "decision_key": "turn-1/step-0",
                    "full_context_tokens": tokens,
                    "full_history_tokens": tokens // 2,
                    "bucket_index": index,
                    "bucket": bucket.bin_label(edges[index], edges[index + 1]),
                    "payload": {"messages": [], "c2kv_eval_context": {
                        "task_id": task, "user_turn": 1, "step": 0, "attempt": 0,
                    }},
                })
        return edges, rows

    def test_balanced_sampling_is_equal_and_deterministic(self):
        edges, rows = self.synthetic_rows()
        first, availability = bucket.balanced_sample(rows, edges, 2, 42)
        second, _ = bucket.balanced_sample(rows, edges, 2, 42)
        self.assertEqual(
            [row["source_decision_id"] for row in first],
            [row["source_decision_id"] for row in second],
        )
        self.assertEqual([row["selected_decisions"] for row in availability], [2, 2, 2])

    def test_every_gpu_runs_same_shard_full_then_c2kv(self):
        _, rows = self.synthetic_rows()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                gpus="0,1,3", out=Path(tmp),
                client_python=Path("/python"), repetitions=3,
                port_base=41000, startup_timeout=1, task_timeout=1,
            )
            jobs = bucket.make_jobs(args, rows)
        self.assertEqual(len(jobs), 6)
        for gpu in (0, 1, 3):
            pair = [job for job in jobs if job["gpu"] == gpu]
            self.assertEqual([job["arm"] for job in pair], ["full", "c2kv"])
            self.assertEqual(pair[0]["decision_count"], pair[1]["decision_count"])
            self.assertEqual(pair[0]["task_count"], pair[1]["task_count"])

    def test_single_gpu_runs_full_then_c2kv(self):
        _, rows = self.synthetic_rows()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                gpus="7", out=Path(tmp), client_python=Path("/python"),
                repetitions=3, port_base=41000, startup_timeout=1, task_timeout=1,
            )
            jobs = bucket.make_jobs(args, rows)
        self.assertEqual([job["arm"] for job in jobs], ["full", "c2kv"])
        self.assertTrue(all(job["gpu"] == 7 for job in jobs))
        self.assertEqual(jobs[0]["decision_count"], jobs[1]["decision_count"])

    def test_paired_summary_and_plot(self):
        decision_id = "task-0/turn-1/step-0/attempt-0"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bucket.save(root / "selection_manifest.json", {
                "repetitions": 3,
                "selected": [{
                    "source_decision_id": decision_id, "task_id": "task-0",
                    "bucket_index": 0, "bucket": "[0, 8K)",
                    "full_context_tokens": 8000, "full_history_tokens": 2000,
                }],
            })
            fields = [
                "method", "source_decision_id", "repeat", "source_user_turn",
                "source_step", "source_attempt", "full_reference_prompt_tokens",
                "resident_prompt_tokens", "full_reference_history_tokens",
                "resident_history_tokens", "ttft_ms", "native_ttft_ms",
                "context_prepare_ms", "action_exact_match",
            ]
            for method, native, resident in (("full", 100.0, 8000), ("c2kv_only", 40.0, 4000)):
                path = root / "runs" / method / "samples.csv"
                path.parent.mkdir(parents=True)
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for repeat in range(3):
                        writer.writerow({
                            "method": method, "source_decision_id": decision_id,
                            "repeat": repeat, "source_user_turn": 1, "source_step": 0,
                            "source_attempt": 0, "full_reference_prompt_tokens": 8000,
                            "resident_prompt_tokens": resident,
                            "full_reference_history_tokens": 2000,
                            "resident_history_tokens": 200 if method != "full" else 2000,
                            "ttft_ms": native + 10, "native_ttft_ms": native,
                            "context_prepare_ms": 10,
                            "action_exact_match": int(method == "full"),
                        })
            summary = bucket.summarize(root)
            self.assertAlmostEqual(summary[0]["native_ttft_ratio_of_p50"], .4)
            self.assertAlmostEqual(summary[0]["native_ttft_reduction_of_p50"], .6)
            self.assertEqual(summary[0]["full_reference_action_fidelity"], 1.0)
            self.assertEqual(summary[0]["c2kv_reference_action_fidelity"], 0.0)
            self.assertEqual(summary[0]["action_fidelity_delta_pp"], -100.0)
            self.assertTrue((root / "ttft_by_context_bucket.png").is_file())
            self.assertTrue((root / "paired_decisions.csv").is_file())

    def test_completed_history_eligibility_excludes_official_single_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "official.jsonl"
            data.write_text(
                json.dumps({"id": "single", "question": [[{"role": "user"}]]})
                + "\n"
                + json.dumps({
                    "id": "multi",
                    "question": [[{"role": "user"}], [{"role": "user"}]],
                })
                + "\n"
            )
            for task, turn in (("single", 0), ("multi", 1)):
                steps = (
                    root / "gpu0/rep0/full_cold/run/task_shards"
                    / task / "server/steps.jsonl"
                )
                steps.parent.mkdir(parents=True)
                steps.write_text(json.dumps({
                    "harness_transport_normalization": {
                        "same_input_replay_request": {
                            "c2kv_eval_context": {
                                "task_id": task, "user_turn": turn, "step": 0,
                            }
                        }
                    }
                }) + "\n")
            audit = longest.completed_history_eligibility(data, root, {"multi"})
            by_id = {row["task_id"]: row for row in audit}
            self.assertFalse(by_id["single"]["completed_history_eligible"])
            self.assertEqual(
                by_id["single"]["eligibility_reason"],
                "ineligible_official_single_user_turn",
            )
            self.assertTrue(by_id["multi"]["completed_history_eligible"])

    def test_completed_history_eligibility_fails_on_missing_multiturn_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "official.jsonl"
            data.write_text(json.dumps({
                "id": "multi",
                "question": [[{"role": "user"}], [{"role": "user"}]],
            }) + "\n")
            steps = root / "gpu0/rep0/full_cold/run/task_shards/multi/server/steps.jsonl"
            steps.parent.mkdir(parents=True)
            steps.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "missing_eligible=\\['multi'\\]"):
                longest.completed_history_eligibility(data, root, set())


if __name__ == "__main__":
    unittest.main()
