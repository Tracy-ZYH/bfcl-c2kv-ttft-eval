#!/usr/bin/env python3
"""Summarize BFCL Long Tool/History scope ablations against Full."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ARMS = (
    ("full", "Full"),
    ("joint_snapkv_persistent_tool_only_r25", "Tool Definition only"),
    ("joint_snapkv_persistent_history_only_r25", "Completed History only"),
    ("joint_snapkv_persistent_joint_r25", "Tool + History"),
)


def load_summary(root: Path, arm: str) -> dict[str, Any]:
    hits = list((root / "runs" / arm).glob(f"summary_{arm}.json"))
    if len(hits) != 1:
        raise ValueError(f"expected one summary for {arm}, found {len(hits)}: {hits}")
    return json.loads(hits[0].read_text(encoding="utf-8"))


def number(value: Any, field: str, arm: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{arm}: missing/invalid {field}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{arm}: non-finite {field}: {value!r}")
    return result


def pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("cannot compute relative change from zero Full baseline")
    return 100.0 * (value / baseline - 1.0)


def summarize(root: Path, expected: int) -> list[dict[str, Any]]:
    loaded = {arm: load_summary(root, arm) for arm, _ in ARMS}
    full = loaded["full"]
    full_acc = number(full.get("official_score"), "official_score", "full")
    full_rl = full.get("request_log_summary") or {}
    full_ttft = number(full_rl.get("server_ttft_ms_p50"), "server_ttft_ms_p50", "full")

    rows: list[dict[str, Any]] = []
    for arm, label in ARMS:
        summary = loaded[arm]
        total = int(summary.get("official_total_count") or 0)
        correct = int(summary.get("official_correct_count") or 0)
        if total != expected:
            raise ValueError(f"{arm}: official_total_count={total}, expected={expected}")
        if correct < 0 or correct > total:
            raise ValueError(f"{arm}: invalid correct count {correct}/{total}")
        rl = summary.get("request_log_summary") or {}
        if int(rl.get("n_error") or 0):
            raise ValueError(f"{arm}: request errors present: {rl.get('error_kinds')}")
        ttft_count = int(rl.get("server_ttft_ms_count") or 0)
        if ttft_count <= 0:
            raise ValueError(f"{arm}: no server TTFT telemetry")
        acc = number(summary.get("official_score"), "official_score", arm)
        p50 = number(rl.get("server_ttft_ms_p50"), "server_ttft_ms_p50", arm)
        p95 = number(rl.get("server_ttft_ms_p95"), "server_ttft_ms_p95", arm)

        full_tool = rl.get("full_tool_kv")
        active_tool = rl.get("active_tool_kv")
        full_history = rl.get("full_history_kv")
        active_history = rl.get("active_history_kv")
        if arm == "full":
            scope_retention = 1.0
            scope_compression = 1.0
            overall_retention = 1.0
            overall_compression = 1.0
        else:
            values = (full_tool, active_tool, full_history, active_history)
            if not all(isinstance(value, (int, float)) for value in values):
                raise ValueError(f"{arm}: incomplete joint Tool/History accounting: {values}")
            full_scope = float(full_tool) + float(full_history)
            active_scope = float(active_tool) + float(active_history)
            if full_scope <= 0 or active_scope <= 0:
                raise ValueError(f"{arm}: invalid joint scope counts {full_scope}/{active_scope}")
            scope_retention = active_scope / full_scope
            scope_compression = full_scope / active_scope
            overall_retention = number(
                rl.get("e2e_kv_retention"), "e2e_kv_retention", arm)
            overall_compression = number(
                rl.get("e2e_kv_compression_ratio"),
                "e2e_kv_compression_ratio", arm)

        rows.append({
            "arm": arm,
            "method": label,
            "official_correct_count": correct,
            "official_total_count": total,
            "bfcl_accuracy": acc,
            "accuracy_change_pp_vs_full": 100.0 * (acc - full_acc),
            "accuracy_relative_change_pct_vs_full": (
                pct_change(acc, full_acc) if full_acc != 0 else None
            ),
            "server_ttft_ms_p50": p50,
            "server_ttft_ms_p95": p95,
            "ttft_p50_change_pct_vs_full": pct_change(p50, full_ttft),
            "ttft_p50_reduction_pct_vs_full": -pct_change(p50, full_ttft),
            "full_tool_kv": full_tool,
            "active_tool_kv": active_tool,
            "tool_retention": rl.get("tool_retention") if arm != "full" else 1.0,
            "full_history_kv": full_history,
            "active_history_kv": active_history,
            "history_retention": rl.get("history_retention") if arm != "full" else 1.0,
            "tool_history_scope_retention": scope_retention,
            "tool_history_scope_compression_ratio": scope_compression,
            "overall_e2e_kv_retention": overall_retention,
            "overall_e2e_kv_compression_ratio": overall_compression,
            "ttft_request_count": ttft_count,
            "summary_path": str(root / "runs" / arm / f"summary_{arm}.json"),
        })
    return rows


def write_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "bfcl_long_joint_scope.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "bfcl_long_joint_scope.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# BFCL Long Tool/History Scope Ablation",
        "",
        "Overall compression includes protected System/template KV and the full Current Turn; Tool+History scope compression is retained separately in CSV/JSON.",
        "",
        "| Method | Accuracy | Correct | Acc delta | TTFT P50 (ms) | TTFT P95 (ms) | TTFT reduction | Overall compression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['bfcl_accuracy']:.4f} | "
            f"{row['official_correct_count']}/{row['official_total_count']} | "
            f"{row['accuracy_change_pp_vs_full']:+.2f} pp | "
            f"{row['server_ttft_ms_p50']:.2f} | {row['server_ttft_ms_p95']:.2f} | "
            f"{row['ttft_p50_reduction_pct_vs_full']:+.2f}% | "
            f"{row['overall_e2e_kv_compression_ratio']:.3f}x |"
        )
    (out / "bfcl_long_joint_scope.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, default=200)
    args = parser.parse_args()
    rows = summarize(args.root, args.expected)
    write_outputs(args.root, rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
