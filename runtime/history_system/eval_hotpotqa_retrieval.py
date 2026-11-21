"""Evaluate Racer's lexical ranker on HotpotQA distractor paragraphs.

This is a retriever-only adapter. It does not run C2KV, mutate KV state, call
an LLM, or use answers/supporting facts as retrieval input. Supporting facts
are read only after ranking to score the selected paragraph IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "python")]

from benchmarks.memory_runtime.recovery.selection import (  # noqa: E402
    _catalog_records,
    _lexical_score,
    _sort_ranked,
)


DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
EXPECTED_VALIDATION_ROWS = 7405
DEFAULT_SEED = 42
TOP_KS = (2, 3, 4)
ADAPTER_NAME = "racer-lexical-query-only-retriever-v1"
DEPENDENCY_VERSIONS = {
    "pyarrow": "24.0.0",
    "huggingface_hub": "1.31.0",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def dependency_versions(*, strict: bool = True) -> dict[str, str]:
    observed = {}
    mismatches = {}
    for distribution, expected in DEPENDENCY_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        observed[distribution] = actual
        if actual != expected:
            mismatches[distribution] = {"expected": expected, "actual": actual}
    if strict and mismatches:
        raise RuntimeError(
            "HotpotQA retrieval dependency versions differ from the pinned "
            f"contract: {mismatches}"
        )
    return observed


def load_hotpotqa_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the pinned official HF parquet without dataset-info format coupling."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    pattern = f"{args.dataset_config}/{args.split}-*.parquet"
    snapshot = Path(snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.dataset_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        allow_patterns=[pattern],
        local_files_only=args.local_files_only,
    ))
    files = sorted(snapshot.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Pinned Hugging Face snapshot contains no files matching {pattern!r}"
        )
    file_receipts = [{
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    } for path in files]
    table = pa.concat_tables([pq.read_table(path) for path in files])
    fingerprint = _sha256_bytes(_json_line([{"bytes": row["bytes"], "sha256": row["sha256"]} for row in file_receipts]).encode("utf-8"))
    return table.to_pylist(), {
        "snapshot_path": str(snapshot.resolve()),
        "parquet_pattern": pattern,
        "parquet_files": file_receipts,
        "fingerprint": fingerprint,
    }


def _context_pairs(context: Any) -> list[tuple[str, list[str]]]:
    if isinstance(context, Mapping):
        titles = context.get("title")
        sentence_groups = context.get("sentences")
        if not _is_sequence(titles) or not _is_sequence(sentence_groups):
            raise ValueError("context must contain title and sentences sequences")
        if len(titles) != len(sentence_groups):
            raise ValueError("context.title and context.sentences lengths differ")
        rows = zip(titles, sentence_groups, strict=True)
    elif _is_sequence(context):
        rows = []
        for item in context:
            if not isinstance(item, Mapping):
                raise ValueError("context list entries must be mappings")
            rows.append((item.get("title"), item.get("sentences")))
    else:
        raise ValueError("context must be a mapping or sequence")

    result = []
    for title, sentences in rows:
        if not isinstance(title, str) or not title:
            raise ValueError("context title must be a nonempty string")
        if not _is_sequence(sentences) or any(
            not isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError(f"context sentences are invalid for title {title!r}")
        result.append((title, list(sentences)))
    if not result:
        raise ValueError("context contains no candidate paragraphs")
    return result


def _supporting_titles(supporting_facts: Any) -> list[str]:
    if isinstance(supporting_facts, Mapping):
        titles = supporting_facts.get("title")
        sentence_ids = supporting_facts.get("sent_id")
        if not _is_sequence(titles) or not _is_sequence(sentence_ids):
            raise ValueError("supporting_facts must contain title and sent_id sequences")
        if len(titles) != len(sentence_ids):
            raise ValueError("supporting_facts.title and sent_id lengths differ")
    elif _is_sequence(supporting_facts):
        titles = []
        for item in supporting_facts:
            if not isinstance(item, Mapping):
                raise ValueError("supporting_facts list entries must be mappings")
            titles.append(item.get("title"))
    else:
        raise ValueError("supporting_facts must be a mapping or sequence")
    if any(not isinstance(title, str) or not title for title in titles):
        raise ValueError("supporting_facts contains an invalid title")
    return list(dict.fromkeys(titles))


def build_candidates(context: Any) -> list[dict[str, Any]]:
    candidates = []
    for index, (title, sentences) in enumerate(_context_pairs(context)):
        paragraph = " ".join(sentence.strip() for sentence in sentences).strip()
        text = f"{title}\n{paragraph}" if paragraph else title
        digest = _sha256_bytes((title + "\0" + paragraph).encode("utf-8"))[:12]
        candidates.append({
            "unit_id": f"doc-{index:02d}-{digest}",
            "document_index": index,
            "title": title,
            "sentences": sentences,
            "text": text,
            "source_type": "paragraph",
            # Hotpot paragraphs are an unordered candidate set, not temporal
            # history. Neutralize Racer's recency tie-break for this adapter.
            "source_indices": [0],
            "metadata": {"title": title, "document_index": index},
        })
    return candidates


def rank_query_only(question: str, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Use the exact Racer lexical scorer without draft, answer, or gold input."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    records = _catalog_records(list(candidates))
    scored = []
    for record in records:
        score, components = _lexical_score(question, record)
        scored.append({**record, "score": score, "components": components})
    # Racer normally drops zero-score rows before variable-size candidate
    # selection. Fixed Top-K evaluation must retain that deterministic tail.
    return _sort_ranked(scored)


def confusion(candidate_ids: Sequence[str], gold_ids: set[str], selected_ids: set[str]) -> dict[str, Any]:
    universe = set(candidate_ids)
    if not selected_ids <= universe or not gold_ids <= universe:
        raise ValueError("gold and selected IDs must belong to the candidate universe")
    tp = len(selected_ids & gold_ids)
    fp = len(selected_ids - gold_ids)
    fn = len(gold_ids - selected_ids)
    tn = len(universe - selected_ids - gold_ids)
    recall = tp / (tp + fn) if tp + fn else 0.0
    filtering = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "filtering": filtering, "precision": precision,
        "meets_recall_85_filtering_85": recall >= 0.85 and filtering >= 0.85,
    }


def evaluate_example(row: Mapping[str, Any], *, row_index: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample_id = row.get("id") or row.get("_id")
    question = row.get("question")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample id is missing")
    candidates = build_candidates(row.get("context"))
    gold_titles = _supporting_titles(row.get("supporting_facts"))

    title_to_ids: dict[str, list[str]] = {}
    for candidate in candidates:
        title_to_ids.setdefault(candidate["title"], []).append(candidate["unit_id"])
    missing_titles = [title for title in gold_titles if title not in title_to_ids]
    ambiguous_titles = {
        title: title_to_ids[title] for title in gold_titles
        if len(title_to_ids.get(title, ())) > 1
    }
    duplicate_candidate_titles = {
        title: ids for title, ids in title_to_ids.items() if len(ids) > 1
    }
    gold_ids = {
        document_id for title in gold_titles
        for document_id in title_to_ids.get(title, ())
    }
    anomalies = []
    if missing_titles:
        anomalies.append("missing_gold_title")
    if ambiguous_titles:
        anomalies.append("ambiguous_gold_title")
    if duplicate_candidate_titles:
        anomalies.append("duplicate_candidate_title")
    if not gold_titles:
        anomalies.append("empty_gold_titles")
    if not gold_ids:
        anomalies.append("empty_gold_documents")

    ranked = rank_query_only(question, candidates)
    candidate_ids = [candidate["unit_id"] for candidate in candidates]
    rank_rows = []
    for rank, record in enumerate(ranked, start=1):
        unit = record["unit"]
        rank_rows.append({
            "rank": rank,
            "document_id": record["unit_id"],
            "title": unit["title"],
            "score": record["score"],
            "score_components": record["components"],
            "is_relevant_gold": record["unit_id"] in gold_ids,
        })

    topk = {}
    for k in TOP_KS:
        selected = {record["unit_id"] for record in ranked[:k]}
        topk[str(k)] = {
            "selected_document_ids": [record["unit_id"] for record in ranked[:k]],
            **confusion(candidate_ids, gold_ids, selected),
        }

    metric_eligible = not missing_titles and not ambiguous_titles and bool(gold_ids)
    result = {
        "schema": "hotpotqa-racer-retrieval-sample-v1",
        "dataset_row_index": row_index,
        "sample_id": sample_id,
        "question": question,
        "adapter": ADAPTER_NAME,
        "retrieval_input_contract": {
            "uses_question": True,
            "uses_agent_draft": False,
            "uses_answer": False,
            "uses_gold": False,
            "gold_used_after_ranking_for_metrics_only": True,
        },
        "candidate_documents": [{
            "document_id": candidate["unit_id"],
            "document_index": candidate["document_index"],
            "title": candidate["title"],
            "sentences": candidate["sentences"],
            "text": candidate["text"],
        } for candidate in candidates],
        "gold_titles": gold_titles,
        "gold_document_ids": sorted(gold_ids),
        "gold_complete_coverage": not missing_titles and not ambiguous_titles,
        "metric_eligible": metric_eligible,
        "ranking": rank_rows,
        "top_k": topk,
        "anomalies": anomalies,
        "claim_scope": "retrieval_selection_only_not_actual_kv_context_filtering",
    }
    anomaly = None
    if anomalies:
        anomaly = {
            "schema": "hotpotqa-retrieval-anomaly-v1",
            "dataset_row_index": row_index,
            "sample_id": sample_id,
            "anomaly_types": anomalies,
            "gold_titles": gold_titles,
            "missing_gold_titles": missing_titles,
            "ambiguous_gold_titles": ambiguous_titles,
            "duplicate_candidate_titles": duplicate_candidate_titles,
            "candidate_titles": [candidate["title"] for candidate in candidates],
        }
    return result, anomaly


def _empty_accumulator() -> dict[str, Any]:
    return {
        "eligible": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "recall_sum": 0.0, "filtering_sum": 0.0, "precision_sum": 0.0,
        "both_count": 0,
    }


def update_accumulators(accumulators: dict[int, dict[str, Any]], result: Mapping[str, Any]) -> None:
    if not result.get("metric_eligible"):
        return
    for k in TOP_KS:
        metric = result["top_k"][str(k)]
        acc = accumulators[k]
        acc["eligible"] += 1
        for name in ("tp", "fp", "tn", "fn"):
            acc[name] += metric[name]
        for name in ("recall", "filtering", "precision"):
            acc[name + "_sum"] += metric[name]
        acc["both_count"] += int(metric["meets_recall_85_filtering_85"])


def summarize(accumulators: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for k in TOP_KS:
        acc = accumulators[k]
        eligible = acc["eligible"]
        tp, fp, tn, fn = (acc[name] for name in ("tp", "fp", "tn", "fn"))
        rows.append({
            "top_k": k,
            "eligible_samples": eligible,
            "micro_tp": tp, "micro_fp": fp, "micro_tn": tn, "micro_fn": fn,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
            "micro_filtering": tn / (tn + fp) if tn + fp else 0.0,
            "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "macro_recall": acc["recall_sum"] / eligible if eligible else 0.0,
            "macro_filtering": acc["filtering_sum"] / eligible if eligible else 0.0,
            "macro_precision": acc["precision_sum"] / eligible if eligible else 0.0,
            "samples_meeting_both_count": acc["both_count"],
            "samples_meeting_both_ratio": acc["both_count"] / eligible if eligible else 0.0,
        })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--expected-size", type=int, default=EXPECTED_VALIDATION_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int,
                        help="CPU development smoke only; omit for the required full validation split")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Require the pinned Hugging Face snapshot to exist in the local cache")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    versions = dependency_versions(strict=not args.allow_version_mismatch)
    random.seed(args.seed)

    dataset, dataset_receipt = load_hotpotqa_rows(args)
    full_size = len(dataset)
    if args.max_samples is None and full_size != args.expected_size:
        raise RuntimeError(
            f"Expected {args.expected_size} rows, found {full_size}; refusing a noncanonical full run"
        )
    limit = full_size if args.max_samples is None else min(full_size, args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = args.output_dir / "samples.jsonl"
    anomalies_path = args.output_dir / "anomalies.jsonl"
    summary_path = args.output_dir / "summary.csv"
    run_path = args.output_dir / "run_config.json"

    accumulators = {k: _empty_accumulator() for k in TOP_KS}
    anomaly_count = schema_error_count = eligible_count = 0
    started = time.perf_counter()
    with samples_path.open("w", encoding="utf-8") as samples_file, anomalies_path.open(
        "w", encoding="utf-8"
    ) as anomalies_file:
        for index in range(limit):
            row = dataset[index]
            try:
                result, anomaly = evaluate_example(row, row_index=index)
            except Exception as error:
                schema_error_count += 1
                result = {
                    "schema": "hotpotqa-racer-retrieval-sample-v1",
                    "dataset_row_index": index,
                    "sample_id": row.get("id") or row.get("_id"),
                    "status": "schema_error",
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "metric_eligible": False,
                    "claim_scope": "retrieval_selection_only_not_actual_kv_context_filtering",
                }
                anomaly = {
                    "schema": "hotpotqa-retrieval-anomaly-v1",
                    "dataset_row_index": index,
                    "sample_id": result["sample_id"],
                    "anomaly_types": ["schema_error"],
                    "error": result["error"],
                }
            samples_file.write(_json_line(result) + "\n")
            if anomaly is not None:
                anomaly_count += 1
                anomalies_file.write(_json_line(anomaly) + "\n")
            if result.get("metric_eligible"):
                eligible_count += 1
            update_accumulators(accumulators, result)
            if (index + 1) % 500 == 0 or index + 1 == limit:
                print(f"processed {index + 1}/{limit}", flush=True)

    summary_rows = summarize(accumulators)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    run_config = {
        "schema": "hotpotqa-racer-retrieval-run-v1",
        "status": "completed",
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": dataset_receipt["fingerprint"],
        "dataset_snapshot": dataset_receipt,
        "dataset_rows": full_size,
        "processed_rows": limit,
        "expected_full_rows": args.expected_size,
        "is_full_validation_run": args.max_samples is None and limit == args.expected_size,
        "seed": args.seed,
        "adapter": ADAPTER_NAME,
        "adapter_semantics": "Racer lexical score; question-only; paragraph recency neutralized; zero-score tail retained for fixed Top-K",
        "top_k": list(TOP_KS),
        "eligible_samples": eligible_count,
        "anomaly_samples": anomaly_count,
        "schema_error_samples": schema_error_count,
        "dependency_versions": versions,
        "source_files": {
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__))},
            "racer_selection": {
                "path": str((RUNTIME / "benchmarks/memory_runtime/recovery/selection.py").resolve()),
                "sha256": _sha256_file(RUNTIME / "benchmarks/memory_runtime/recovery/selection.py"),
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "samples_jsonl": str(samples_path),
            "anomalies_jsonl": str(anomalies_path),
            "summary_csv": str(summary_path),
        },
        "claim_scope": "retrieval_selection_only_not_actual_kv_context_filtering",
        "summary": summary_rows,
    }
    run_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
