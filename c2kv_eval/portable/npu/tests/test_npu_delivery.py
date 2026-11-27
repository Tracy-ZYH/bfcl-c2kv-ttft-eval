from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from c2kv_eval.portable.npu.frozen import EXPECTED_GUARDS, load_and_validate
from c2kv_eval.portable.npu.summarize import build_summary, summarize_method, write_outputs
from c2kv_eval.portable.npu.timing import synchronize_accelerator


class _Sync:
    def __init__(self):
        self.calls = 0

    def synchronize(self):
        self.calls += 1


class _Torch:
    def __init__(self):
        self.npu = _Sync()
        self.cuda = _Sync()


def _write_method(root: Path, method: str, task_ids: list[str]) -> None:
    root.mkdir(parents=True)
    rows = []
    for index, task_id in enumerate(task_ids):
        rows.append({
            "task_id": task_id,
            "status": "completed",
            "official_score": 1.0 if index == 0 else 0.0,
        })
        server = root / "task_shards" / task_id / "server"
        server.mkdir(parents=True)
        speed = {
            "prefill_path_ms": 10.0 + index,
            "server_ttft_ms": 12.0 + index,
            "native_ttft_ms": 15.0 + index,
        }
        record = {
            "generation_trace": [{
                "phase": "draft",
                "status": "completed",
                "generation": {"stats": {"speed_timing": speed}},
            }],
            "controller_timing": {
                "prepare_seconds": 0.001,
                "bcrs_route_seconds": 0.002 if method == "BCRO" else 0.0,
                "legal_cache_prewarm_seconds": 0.003 if method == "BCRO" else 0.0,
            },
            "decision_runtime_seconds": 0.050 + index / 1000,
        }
        if method == "Full":
            record["ordinary_full_path"] = {"endpoint": "/v1/chat/completions"}
        else:
            record["bcrs"] = {"selected_arm": "c2kv_racer"}
            record["strict_cache_hit"] = [{"passed": True}]
        (server / "steps.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "unified_summary.json").write_text(
        json.dumps({"schema": "c1-unified-summary-v1", "rows": rows}), encoding="utf-8"
    )


class FrozenConfigTest(unittest.TestCase):
    def test_formal_config_and_static_npu_contract(self):
        frozen = load_and_validate(
            require_external_models=False,
            require_bfcl_data=True,
        )
        self.assertEqual(len(frozen.task_ids), 120)
        self.assertEqual(frozen.config["guards"], EXPECTED_GUARDS)
        self.assertEqual(frozen.config["compression"]["history_budget_tokens"], 256)
        self.assertEqual(frozen.config["detector"]["threshold"], 0.5)


class TimingTest(unittest.TestCase):
    def test_device_sync_is_explicit(self):
        fake = _Torch()
        synchronize_accelerator(fake, "npu:0")
        synchronize_accelerator(fake, "cuda:0")
        synchronize_accelerator(fake, "cpu")
        self.assertEqual(fake.npu.calls, 1)
        self.assertEqual(fake.cuda.calls, 1)

    def test_accuracy_and_timing_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            ids = ["multi_turn_long_context_1", "multi_turn_long_context_2"]
            _write_method(base / "full", "Full", ids)
            _write_method(base / "bcro", "BCRO", ids)
            full = summarize_method(base / "full", "Full", ids)
            bcro = summarize_method(base / "bcro", "BCRO", ids)
            summary = build_summary(full, bcro)
            self.assertEqual(full["official_correct_count"], 1)
            self.assertEqual(full["timing"]["draft_server_ttft_ms"]["count"], 2)
            self.assertEqual(bcro["validation"]["route_racer"], 2)
            write_outputs(summary, base / "summary")
            self.assertTrue((base / "summary/summary.json").is_file())
            self.assertTrue((base / "summary/summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
