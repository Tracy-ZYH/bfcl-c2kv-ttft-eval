"""Compare query-only Racer retrieval stages on HotpotQA distractor."""


from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path[:0] = [str(HERE), str(RUNTIME), str(RUNTIME / "python")]

import eval_hotpotqa_retrieval as base  # noqa: E402
from benchmarks.memory_runtime.recovery.local_selection_models import (  # noqa: E402
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    LocalSelectionModels,
    LocalSelectionModelError,
)
from benchmarks.memory_runtime.recovery.selection import (  # noqa: E402
    _catalog_records,
    _cosine,
    _lexical_score,
    _sort_ranked,
    _validate_embeddings,
)

METHODS = ("lexical", "hybrid_rrf", "hybrid_reranker")
ADAPTER_NAME = "racer-query-only-retriever-comparison-v1"
DEFAULT_LOCAL_EMBEDDING = Path("/home/zhuyuhan/lyc/models/Qwen3-Embedding-0.6B")
def resolve_pinned_reranker_snapshot(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Resolve the pinned reranker before creating benchmark outputs."""
    requested = str(args.reranker_model)
    if requested == RERANKER_MODEL:
        from huggingface_hub import snapshot_download
        try:
            snapshot = Path(snapshot_download(
                repo_id=RERANKER_MODEL,
                revision=args.reranker_revision,
                local_files_only=not args.allow_model_download,
            )).resolve()
        except Exception as error:
            mode = "download-enabled" if args.allow_model_download else "local-only"
            raise RuntimeError(
                f"Unable to resolve pinned reranker {RERANKER_MODEL}@"
                f"{args.reranker_revision} in {mode} mode. Pass "
                "--allow-model-download to provision the missing snapshot."
            ) from error
        source = "pinned_huggingface_snapshot"
    else:
        snapshot = Path(requested).expanduser().resolve()
        source = "explicit_local_directory"
    if not snapshot.is_dir():
        raise RuntimeError(f"Reranker model directory does not exist: {snapshot}")
    weight_patterns = ("*.safetensors", "pytorch_model*.bin")
    weights = sorted({
        path.resolve()
        for pattern in weight_patterns
        for path in snapshot.glob(pattern)
        if path.is_file()
    })
    if not weights:
        raise RuntimeError(
            f"Reranker snapshot has config/tokenizer metadata but no model weights: "
            f"{snapshot}. Re-run with --allow-model-download."
        )
    if not (snapshot / "config.json").is_file():
        raise RuntimeError(f"Reranker snapshot lacks config.json: {snapshot}")
    return str(snapshot), {
        "source": source,
        "repo_id": RERANKER_MODEL if requested == RERANKER_MODEL else None,
        "requested_revision": args.reranker_revision if requested == RERANKER_MODEL else None,
        "snapshot_path": str(snapshot),
        "weight_files": [path.name for path in weights],
    }



def _lexical_rows(
    question: str, candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for record in _catalog_records(list(candidates)):
        score, components = _lexical_score(question, record)
        rows.append({**record, "score": score, "components": components})
    return _sort_ranked(rows)


def rank_hybrid_rrf(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    models: LocalSelectionModels,
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse the exact Racer lexical score and Qwen semantic cosine ranks."""
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    lexical = _lexical_rows(question, candidates)
    if not lexical:
        return []
    query_vectors = models.embed(texts=[question], purpose="query", config={})
    document_vectors = models.embed(
        texts=[row["search_text"] for row in lexical],
        purpose="document",
        config={},
    )
    vectors = _validate_embeddings(
        query_vectors + document_vectors, 1 + len(lexical)
    )
    semantic_scores = {
        row["unit_id"]: (_cosine(vectors[0], vector) + 1.0) / 2.0
        for row, vector in zip(lexical, vectors[1:], strict=True)
    }
    semantic = sorted(
        lexical,
        key=lambda row: (-semantic_scores[row["unit_id"]], row["unit_id"]),
    )
    lexical_rank = {
        row["unit_id"]: rank for rank, row in enumerate(lexical, 1)
    }
    semantic_rank = {
        row["unit_id"]: rank for rank, row in enumerate(semantic, 1)
    }
    fused = []
    for row in lexical:
        lr = lexical_rank[row["unit_id"]]
        sr = semantic_rank[row["unit_id"]]
        fused.append(
            {
                **row,
                "score": 1.0 / (rrf_k + lr) + 1.0 / (rrf_k + sr),
                "components": {
                    **row["components"],
                    "semantic_cosine_normalized": semantic_scores[row["unit_id"]],
                    "hybrid_fusion": "rrf",
                    "rrf_k": rrf_k,
                    "lexical_rank": lr,
                    "semantic_rank": sr,
                },
            }
        )
    return _sort_ranked(fused)


def rank_hybrid_reranker(
    question: str,
    hybrid_rows: Sequence[Mapping[str, Any]],
    models: LocalSelectionModels,
    *,
    candidate_pool_size: int = 8,
) -> list[dict[str, Any]]:
    """Rerank Racer's bounded hybrid pool and preserve the hybrid tail."""
    if not isinstance(candidate_pool_size, int) or isinstance(
        candidate_pool_size, bool
    ):
        raise TypeError("candidate_pool_size must be an integer")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")
    rows = [dict(row) for row in hybrid_rows]
    pool = rows[:candidate_pool_size]
    if not pool:
        return rows
    scores = models.rerank_retrieval_candidates(
        task=question,
        draft="",
        documents=[row["search_text"] for row in pool],
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    if len(scores) != len(pool) or any(
        not isinstance(score, (int, float)) or not math.isfinite(float(score))
        for score in scores
    ):
        raise RuntimeError("Racer reranker returned invalid scores")
    reranked = []
    for original_rank, (row, score) in enumerate(
        zip(pool, scores, strict=True), 1
    ):
        reranked.append(
            {
                **row,
                "score": float(score),
                "components": {
                    **row["components"],
                    "hybrid_candidate_rank": original_rank,
                    "reranker_probability": float(score),
                    "reranker_draft_input": "empty_query_only_adapter",
                },
            }
        )
    reranked = _sort_ranked(reranked)
    for tail_rank, row in enumerate(
        rows[candidate_pool_size:], candidate_pool_size + 1
    ):
        reranked.append(
            {
                **row,
                "components": {
                    **row["components"],
                    "hybrid_candidate_rank": tail_rank,
                },
            }
        )
    return reranked


def _gold_and_candidates(
    row: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], set[str], dict[str, Any]]:
    candidates = base.build_candidates(row.get("context"))
    gold_titles = base._supporting_titles(row.get("supporting_facts"))
    title_to_ids: dict[str, list[str]] = {}
    for candidate in candidates:
        title_to_ids.setdefault(candidate["title"], []).append(candidate["unit_id"])
    missing = [title for title in gold_titles if title not in title_to_ids]
    ambiguous = {
        title: title_to_ids[title]
        for title in gold_titles
        if len(title_to_ids.get(title, ())) > 1
    }
    duplicates = {
        title: ids for title, ids in title_to_ids.items() if len(ids) > 1
    }
    gold_ids = {
        unit_id
        for title in gold_titles
        for unit_id in title_to_ids.get(title, ())
    }
    anomalies = []
    if missing:
        anomalies.append("missing_gold_title")
    if ambiguous:
        anomalies.append("ambiguous_gold_title")
    if duplicates:
        anomalies.append("duplicate_candidate_title")
    if not gold_titles:
        anomalies.append("empty_gold_titles")
    if not gold_ids:
        anomalies.append("empty_gold_documents")
    audit = {
        "types": anomalies,
        "missing_gold_titles": missing,
        "ambiguous_gold_titles": ambiguous,
        "duplicate_candidate_titles": duplicates,
        "metric_eligible": not missing and not ambiguous and bool(gold_ids),
    }
    return candidates, gold_titles, gold_ids, audit


def _method_result(
    ranked: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    gold_ids: set[str],
) -> dict[str, Any]:
    ranking = []
    for rank, row in enumerate(ranked, 1):
        unit = row["unit"]
        ranking.append(
            {
                "rank": rank,
                "document_id": row["unit_id"],
                "title": unit["title"],
                "score": row["score"],
                "score_components": row["components"],
                "is_relevant_gold": row["unit_id"] in gold_ids,
            }
        )
    candidate_ids = [candidate["unit_id"] for candidate in candidates]
    top_k = {}
    for k in base.TOP_KS:
        selected_ids = [row["unit_id"] for row in ranked[:k]]
        top_k[str(k)] = {
            "selected_document_ids": selected_ids,
            **base.confusion(candidate_ids, gold_ids, set(selected_ids)),
        }
    return {"ranking": ranking, "top_k": top_k}


def evaluate_example(
    row: Mapping[str, Any],
    *,
    row_index: int,
    methods: Sequence[str],
    models: LocalSelectionModels | None,
    rrf_k: int,
    candidate_pool_size: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample_id = row.get("id") or row.get("_id")
    question = row.get("question")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample id is missing")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    candidates, gold_titles, gold_ids, audit = _gold_and_candidates(row)
    lexical = _lexical_rows(question, candidates)
    rankings: dict[str, list[dict[str, Any]]] = {"lexical": lexical}
    method_seconds: dict[str, float] = {}
    if any(method != "lexical" for method in methods):
        if models is None:
            raise RuntimeError(
                "model-backed retrieval requested without LocalSelectionModels"
            )
        started = time.perf_counter()
        hybrid = rank_hybrid_rrf(question, candidates, models, rrf_k=rrf_k)
        method_seconds["hybrid_rrf"] = time.perf_counter() - started
        rankings["hybrid_rrf"] = hybrid
        if "hybrid_reranker" in methods:
            started = time.perf_counter()
            rankings["hybrid_reranker"] = rank_hybrid_reranker(
                question,
                hybrid,
                models,
                candidate_pool_size=candidate_pool_size,
            )
            method_seconds["hybrid_reranker"] = time.perf_counter() - started
    result = {
        "schema": "hotpotqa-racer-retrieval-comparison-sample-v1",
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
        "candidate_documents": [
            {
                "document_id": candidate["unit_id"],
                "document_index": candidate["document_index"],
                "title": candidate["title"],
                "sentences": candidate["sentences"],
                "text": candidate["text"],
            }
            for candidate in candidates
        ],
        "gold_titles": gold_titles,
        "gold_document_ids": sorted(gold_ids),
        "gold_complete_coverage": not audit["missing_gold_titles"]
        and not audit["ambiguous_gold_titles"],
        "metric_eligible": audit["metric_eligible"],
        "methods": {
            method: {
                **_method_result(rankings[method], candidates, gold_ids),
                "elapsed_seconds": method_seconds.get(method),
            }
            for method in methods
        },
        "anomalies": audit["types"],
        "claim_scope": "retrieval_selection_only_not_actual_kv_context_filtering",
    }
    anomaly = None
    if audit["types"]:
        anomaly = {
            "schema": "hotpotqa-retrieval-anomaly-v1",
            "dataset_row_index": row_index,
            "sample_id": sample_id,
            "anomaly_types": audit["types"],
            "gold_titles": gold_titles,
            "candidate_titles": [candidate["title"] for candidate in candidates],
            **{
                key: audit[key]
                for key in (
                    "missing_gold_titles",
                    "ambiguous_gold_titles",
                    "duplicate_candidate_titles",
                )
            },
        }
    return result, anomaly


def _new_accumulators(
    methods: Sequence[str],
) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        method: {k: base._empty_accumulator() for k in base.TOP_KS}
        for method in methods
    }


def update_accumulators(
    accumulators: dict[str, dict[int, dict[str, Any]]],
    result: Mapping[str, Any],
) -> None:
    if not result.get("metric_eligible"):
        return
    for method, method_result in result["methods"].items():
        shim = {"metric_eligible": True, "top_k": method_result["top_k"]}
        base.update_accumulators(accumulators[method], shim)


def summarize(
    accumulators: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for method, values in accumulators.items():
        for row in base.summarize(values):
            rows.append({"method": method, **row})
    return rows


def model_dependency_versions() -> dict[str, str]:
    names = ("torch", "transformers", "accelerate", "safetensors", "numpy")
    return {name: importlib.metadata.version(name) for name in names}

def model_config(args: argparse.Namespace) -> dict[str, Any]:
    embedding_path = str(args.embedding_model)
    reranker_path = str(args.reranker_model)
    return {
        "embedding": {
            "model_name_or_path": embedding_path,
            "revision": args.embedding_revision if embedding_path == EMBEDDING_MODEL else None,
            "device": args.device,
            "dtype": args.dtype,
            "local_files_only": not args.allow_model_download,
            "batch_size": args.embedding_batch_size,
            "cache_size": args.embedding_cache_size,
        },
        "reranker": {
            "model_name_or_path": reranker_path,
            "revision": args.reranker_revision if reranker_path == RERANKER_MODEL else None,
            "device": args.device,
            "dtype": args.dtype,
            "local_files_only": not args.allow_model_download,
            "batch_size": args.reranker_batch_size,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dataset", default=base.DATASET_ID)
    parser.add_argument("--dataset-config", default=base.DATASET_CONFIG)
    parser.add_argument("--split", default=base.DATASET_SPLIT)
    parser.add_argument("--dataset-revision", default=base.DATASET_REVISION)
    parser.add_argument("--expected-size", type=int, default=base.EXPECTED_VALIDATION_ROWS)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--embedding-model", default=str(DEFAULT_LOCAL_EMBEDDING))
    parser.add_argument("--embedding-revision", default=EMBEDDING_REVISION)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--reranker-revision", default=RERANKER_REVISION)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache-size", type=int, default=4096)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser


def _receipt_summary(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capabilities = Counter(str(row.get("capability")) for row in receipts)
    return {
        "receipt_count": len(receipts),
        "capability_counts": dict(sorted(capabilities.items())),
        "latency_seconds": sum(float(row.get("latency_seconds") or 0.0) for row in receipts),
        "model_loads": [dict(row) for row in receipts if row.get("capability") == "model_load"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = list(dict.fromkeys(args.methods))
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.candidate_pool_size <= 0 or args.rrf_k <= 0:
        raise ValueError("candidate-pool-size and rrf-k must be positive")
    reranker_snapshot = None
    if "hybrid_reranker" in methods:
        args.reranker_model, reranker_snapshot = resolve_pinned_reranker_snapshot(args)
    versions = base.dependency_versions(strict=not args.allow_version_mismatch)
    random.seed(args.seed)
    dataset, dataset_receipt = base.load_hotpotqa_rows(args)
    full_size = len(dataset)
    if args.max_samples is None and full_size != args.expected_size:
        raise RuntimeError(
            f"Expected {args.expected_size} rows, found {full_size}; "
            "refusing a noncanonical full run"
        )
    limit = full_size if args.max_samples is None else min(full_size, args.max_samples)
    config = model_config(args)
    needs_models = any(method != "lexical" for method in methods)
    models = LocalSelectionModels(config) if needs_models else None

    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = args.output_dir / "samples.jsonl"
    anomalies_path = args.output_dir / "anomalies.jsonl"
    summary_path = args.output_dir / "summary.csv"
    run_path = args.output_dir / "run_config.json"
    accumulators = _new_accumulators(methods)
    anomaly_count = schema_error_count = eligible_count = 0
    receipts: list[dict[str, Any]] = []
    started = time.perf_counter()
    with samples_path.open("w", encoding="utf-8") as samples_file, anomalies_path.open(
        "w", encoding="utf-8"
    ) as anomalies_file:
        for index in range(limit):
            row = dataset[index]
            try:
                result, anomaly = evaluate_example(
                    row,
                    row_index=index,
                    methods=methods,
                    models=models,
                    rrf_k=args.rrf_k,
                    candidate_pool_size=args.candidate_pool_size,
                )
            except (TypeError, ValueError) as error:
                if isinstance(error, LocalSelectionModelError):
                    raise
                schema_error_count += 1
                result = {
                    "schema": "hotpotqa-racer-retrieval-comparison-sample-v1",
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
            if models is not None:
                receipts.extend(models.drain_receipts())
            samples_file.write(base._json_line(result) + "\n")
            if anomaly is not None:
                anomaly_count += 1
                anomalies_file.write(base._json_line(anomaly) + "\n")
            if result.get("metric_eligible"):
                eligible_count += 1
            update_accumulators(accumulators, result)
            if (index + 1) % 100 == 0 or index + 1 == limit:
                print(f"processed {index + 1}/{limit}", flush=True)

    summary_rows = summarize(accumulators)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    run_config = {
        "schema": "hotpotqa-racer-retrieval-comparison-run-v1",
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
        "methods": methods,
        "method_semantics": {
            "lexical": "exact Racer lexical score, question only",
            "hybrid_rrf": "Racer lexical rank + Qwen query/document embedding rank, RRF",
            "hybrid_reranker": "hybrid RRF pool reranked by Racer Qwen reranker; empty draft",
        },
        "rrf_k": args.rrf_k,
        "candidate_pool_size": args.candidate_pool_size,
        "eligible_samples": eligible_count,
        "anomaly_samples": anomaly_count,
        "schema_error_samples": schema_error_count,
        "dependency_versions": versions,
        "model_dependency_versions": model_dependency_versions(),
        "local_selection_model_config": models.public_config() if models is not None else None,
        "reranker_snapshot": reranker_snapshot,
        "model_receipt_summary": _receipt_summary(receipts),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "samples_jsonl": str(samples_path),
            "anomalies_jsonl": str(anomalies_path),
            "summary_csv": str(summary_path),
        },
        "claim_scope": "retrieval_selection_only_not_actual_kv_context_filtering",
        "summary": summary_rows,
    }
    run_path.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
