import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_hotpotqa_retrieval as evaluation


def sample():
    return {
        "id": "sample-1",
        "question": "Which city contains the Alpha tower?",
        "answer": "unused answer",
        "context": {
            "title": ["Alpha tower", "Other place", "City Alpha", "Noise"],
            "sentences": [
                ["Alpha tower is a landmark."],
                ["This paragraph is unrelated."],
                ["City Alpha contains the Alpha tower."],
                ["More unrelated material."],
            ],
        },
        "supporting_facts": {
            "title": ["Alpha tower", "City Alpha"],
            "sent_id": [0, 0],
        },
    }


class HotpotRetrievalTest(unittest.TestCase):
    def test_query_only_ranking_and_metrics(self):
        result, anomaly = evaluation.evaluate_example(sample(), row_index=0)
        self.assertIsNone(anomaly)
        self.assertTrue(result["metric_eligible"])
        self.assertEqual(len(result["candidate_documents"]), 4)
        self.assertEqual(result["top_k"]["2"]["tp"], 2)
        self.assertEqual(result["top_k"]["2"]["fn"], 0)
        self.assertEqual(result["top_k"]["2"]["fp"], 0)
        self.assertEqual(result["top_k"]["2"]["tn"], 2)
        self.assertEqual(result["top_k"]["2"]["recall"], 1.0)
        self.assertEqual(result["top_k"]["2"]["filtering"], 1.0)

    def test_answer_and_gold_never_enter_ranking(self):
        first = sample()
        second = copy.deepcopy(first)
        second["answer"] = "adversarial Alpha tower City Alpha"
        second["supporting_facts"] = {"title": ["Noise"], "sent_id": [0]}
        ranked_first = evaluation.rank_query_only(
            first["question"], evaluation.build_candidates(first["context"])
        )
        ranked_second = evaluation.rank_query_only(
            second["question"], evaluation.build_candidates(second["context"])
        )
        self.assertEqual(
            [(row["unit_id"], row["score"]) for row in ranked_first],
            [(row["unit_id"], row["score"]) for row in ranked_second],
        )

    def test_missing_gold_is_separate_anomaly(self):
        value = sample()
        value["supporting_facts"]["title"].append("Missing article")
        value["supporting_facts"]["sent_id"].append(0)
        result, anomaly = evaluation.evaluate_example(value, row_index=3)
        self.assertFalse(result["metric_eligible"])
        self.assertIn("missing_gold_title", anomaly["anomaly_types"])
        self.assertEqual(anomaly["missing_gold_titles"], ["Missing article"])

    def test_micro_macro_summary(self):
        accumulators = {k: evaluation._empty_accumulator() for k in evaluation.TOP_KS}
        result, _ = evaluation.evaluate_example(sample(), row_index=0)
        evaluation.update_accumulators(accumulators, result)
        rows = evaluation.summarize(accumulators)
        top2 = next(row for row in rows if row["top_k"] == 2)
        self.assertEqual(top2["micro_recall"], 1.0)
        self.assertEqual(top2["macro_recall"], 1.0)
        self.assertEqual(top2["samples_meeting_both_ratio"], 1.0)

    def test_paragraph_recency_is_neutralized(self):
        candidates = evaluation.build_candidates(sample()["context"])
        self.assertTrue(all(row["source_indices"] == [0] for row in candidates))


if __name__ == "__main__":
    unittest.main()
