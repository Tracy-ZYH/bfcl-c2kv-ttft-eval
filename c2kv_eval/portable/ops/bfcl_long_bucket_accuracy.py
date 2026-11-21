#!/usr/bin/env python3
"""Join BFCL Long timing buckets with parser-clean official task scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_run(summary_path: Path, expected_task_ids):
    run_root = summary_path.parent
    audits = sorted(run_root.glob("task_shards/*/bfcl/bfcl/task_audit/tasks.jsonl"))
    if not audits:
        raise ValueError(f"task audit telemetry is missing under {run_root}")
    errors = []
    audited = set()
    for path in audits:
        for row in read_jsonl(path):
            task_id = row.get("task_id")
            if task_id not in expected_task_ids:
                continue
            if isinstance(task_id, str):
                audited.add(task_id)
            for field in ("base_error", "check_error", "retry_check_error"):
                if row.get(field) is not None:
                    errors.append({
                        "task_id": task_id,
                        "field": field,
                        "value": row[field],
                    })
    if errors:
        preview = json.dumps(errors[:5], ensure_ascii=False)
        raise ValueError(f"parser/checker errors in {len(errors)} task audits: {preview}")
    return audited


def official_scores(paths):
    if isinstance(paths, Path):
        paths = [paths]
    result = {}
    for path in paths:
        payload = json.loads(path.read_text())
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"invalid unified summary: {path}")
        part = {}
        for row in rows:
            task_id = row.get("task_id")
            score = row.get("official_score")
            if not isinstance(task_id, str) or score not in (0, 1, 0.0, 1.0):
                raise ValueError(f"non-binary official score in {path}: {row}")
            if task_id in part or task_id in result:
                raise ValueError(f"duplicate task score across summaries: {task_id}")
            part[task_id] = int(score)
        audited = validate_run(path, set(part))
        if set(part) != audited:
            raise ValueError(
                f"score/audit task mismatch in {path}: "
                f"scores={len(part)} audits={len(audited)}"
            )
        result.update(part)
    return result


def load_timing_rows(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task_id", "bucket_index", "bucket"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid paired timing CSV: {path}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("accuracy bucketing requires one final decision per task")
    return rows


def load_bucket_rows(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "bucket_index" not in rows[0]:
        raise ValueError(f"invalid bucket summary CSV: {path}")
    return rows


def enrich(timing_rows, bucket_rows, full_scores, compressed_scores):
    timing_ids = {row["task_id"] for row in timing_rows}
    missing_full = sorted(timing_ids - set(full_scores))
    missing_compressed = sorted(timing_ids - set(compressed_scores))
    if missing_full or missing_compressed:
        raise ValueError(
            "official scores missing selected timing tasks: "
            f"full={missing_full[:5]} compressed={missing_compressed[:5]}"
        )
    by_bucket = {}
    for row in timing_rows:
        by_bucket.setdefault(row["bucket_index"], []).append(row["task_id"])
    output = []
    for timing in bucket_rows:
        task_ids = by_bucket.get(timing["bucket_index"], [])
        if not task_ids:
            raise ValueError(f"empty timing bucket: {timing['bucket_index']}")
        full = [full_scores[task_id] for task_id in task_ids]
        compressed = [compressed_scores[task_id] for task_id in task_ids]
        total = len(task_ids)
        full_correct = sum(full)
        compressed_correct = sum(compressed)
        full_accuracy = full_correct / total
        compressed_accuracy = compressed_correct / total
        both_correct = sum(f == 1 and c == 1 for f, c in zip(full, compressed))
        full_only = sum(f == 1 and c == 0 for f, c in zip(full, compressed))
        compressed_only = sum(f == 0 and c == 1 for f, c in zip(full, compressed))
        both_wrong = total - both_correct - full_only - compressed_only
        output.append({
            **timing,
            "accuracy_task_count": total,
            "full_correct": full_correct,
            "full_accuracy": full_accuracy,
            "compressed_correct": compressed_correct,
            "compressed_accuracy": compressed_accuracy,
            "accuracy_delta_pp": 100 * (compressed_accuracy - full_accuracy),
            "accuracy_ratio": (
                compressed_accuracy / full_accuracy if full_accuracy else None
            ),
            "accuracy_relative_change": (
                compressed_accuracy / full_accuracy - 1 if full_accuracy else None
            ),
            "both_correct": both_correct,
            "full_only": full_only,
            "compressed_only": compressed_only,
            "both_wrong": both_wrong,
        })
    return output


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing-root", type=Path, required=True)
    p.add_argument("--full-summary", type=Path, action="append", required=True)
    p.add_argument("--compressed-summary", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    timing_rows = load_timing_rows(args.timing_root / "paired_decisions.csv")
    bucket_rows = load_bucket_rows(args.timing_root / "bucket_summary.csv")
    rows = enrich(
        timing_rows,
        bucket_rows,
        official_scores(args.full_summary),
        official_scores(args.compressed_summary),
    )
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "bucket_time_accuracy.csv", rows)
    (args.out / "bucket_time_accuracy.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
