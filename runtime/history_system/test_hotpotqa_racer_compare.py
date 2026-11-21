"""Compare query-only Racer retrieval stages on HotpotQA distractor."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_hotpotqa_racer_compare as compare
import test_hotpotqa_retrieval as fixtures

class FakeModels:
    def embed(self, *, texts, purpose, config):
        return [[float("alpha" in text.casefold()), float("city" in text.casefold()), 1.0] for text in texts]
    def rerank_retrieval_candidates(self, *, task, draft, documents, overflow_policy):
        self.last_draft = draft
        return [0.99 if "city alpha contains" in text.casefold() else 0.1 for text in documents]

class CompareTest(unittest.TestCase):
    def evaluate(self, row):
        return compare.evaluate_example(row, row_index=0, methods=compare.METHODS, models=FakeModels(), rrf_k=60, candidate_pool_size=3)
    def test_same_universe_and_metrics(self):
        result, anomaly = self.evaluate(fixtures.sample())
        self.assertIsNone(anomaly)
        universe = {row["document_id"] for row in result["candidate_documents"]}
        for method in compare.METHODS:
            self.assertEqual({row["document_id"] for row in result["methods"][method]["ranking"]}, universe)
        accumulators = compare._new_accumulators(compare.METHODS)
        compare.update_accumulators(accumulators, result)
        self.assertEqual(len(compare.summarize(accumulators)), 9)
    def test_gold_and_answer_do_not_affect_ranking(self):
        first = fixtures.sample()
        second = copy.deepcopy(first)
        second["answer"] = "adversarial"
        second["supporting_facts"] = {"title": ["Noise"], "sent_id": [0]}
        left, _ = self.evaluate(first)
        right, _ = self.evaluate(second)
        for method in compare.METHODS:
            self.assertEqual([(row["document_id"], row["score"]) for row in left["methods"][method]["ranking"]], [(row["document_id"], row["score"]) for row in right["methods"][method]["ranking"]])
    def test_reranker_is_query_only(self):
        models = FakeModels()
        candidates = compare.base.build_candidates(fixtures.sample()["context"])
        hybrid = compare.rank_hybrid_rrf(fixtures.sample()["question"], candidates, models)
        reranked = compare.rank_hybrid_reranker(fixtures.sample()["question"], hybrid, models, candidate_pool_size=3)
        self.assertEqual(models.last_draft, "")
        self.assertEqual({row["unit_id"] for row in reranked}, {row["unit_id"] for row in hybrid})

    def test_explicit_reranker_snapshot_requires_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            args = type("Args", (), {
                "reranker_model": str(root),
                "reranker_revision": compare.RERANKER_REVISION,
                "allow_model_download": False,
            })()
            with self.assertRaisesRegex(RuntimeError, "no model weights"):
                compare.resolve_pinned_reranker_snapshot(args)

    def test_explicit_reranker_snapshot_resolves_before_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"fixture")
            args = type("Args", (), {
                "reranker_model": str(root),
                "reranker_revision": compare.RERANKER_REVISION,
                "allow_model_download": False,
            })()
            resolved, receipt = compare.resolve_pinned_reranker_snapshot(args)
            self.assertEqual(Path(resolved), root.resolve())
            self.assertEqual(receipt["source"], "explicit_local_directory")
            self.assertEqual(receipt["weight_files"], ["model.safetensors"])

if __name__ == "__main__":
    unittest.main()
