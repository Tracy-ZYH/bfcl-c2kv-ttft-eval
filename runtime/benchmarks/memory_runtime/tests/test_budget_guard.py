"""Pre-generation accounting contracts for Full and compressed history."""

import unittest
from types import SimpleNamespace

from benchmarks.memory_runtime.budget_guard import history_budget_receipt


class _Memory:
    def __init__(self, *, gist_tokens=0):
        self.gist_tokens = gist_tokens

    def costs(self, ratio):
        assert ratio == 8
        return {"resident_kv_tokens": 100, "gist_tokens": self.gist_tokens}


def _full_metadata():
    return {
        "full_source_coverage": True,
        "omitted_event_ids": [],
        "resident_kv_tokens": 100,
        "history_bytes": 9000,
        "fixed_baseline_tokens": 10,
    }


class BudgetGuardTest(unittest.TestCase):
    def test_full_original_does_not_require_a_hybrid_budget(self):
        receipt = history_budget_receipt(
            _Memory(), _full_metadata(), SimpleNamespace(view_mode="full_original"),
            ratio=8, phase="draft",
        )
        self.assertEqual(receipt["status"], "passed")
        self.assertIsNone(receipt["history_budget_bytes"])
        self.assertEqual(receipt["active_history_bytes"], 9000)

    def test_full_original_still_rejects_non_full_representation(self):
        for gist_tokens, omitted in ((1, []), (0, ["event-1"])):
            with self.subTest(gist_tokens=gist_tokens, omitted=omitted):
                metadata = _full_metadata()
                metadata["omitted_event_ids"] = omitted
                receipt = history_budget_receipt(
                    _Memory(gist_tokens=gist_tokens), metadata,
                    SimpleNamespace(view_mode="full_original"), ratio=8, phase="draft",
                )
                self.assertEqual(receipt["status"], "rejected")
                self.assertEqual(
                    receipt["errors"], ["full_original_is_not_full_raw_history"]
                )

    def test_compressed_route_still_requires_explicit_budget_contract(self):
        with self.assertRaisesRegex(
            ValueError, "explicit native history budget contract"
        ):
            history_budget_receipt(
                _Memory(), {}, SimpleNamespace(view_mode="policy"),
                ratio=8, phase="draft",
            )


if __name__ == "__main__":
    unittest.main()
