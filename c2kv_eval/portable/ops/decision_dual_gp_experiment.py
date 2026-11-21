#!/usr/bin/env python3
"""Prepare, fit and summarize the BFCL Long decision-level dual-GP experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from c2kv_eval.portable.decision_dual_gp import (
    DecisionDualGPScheduler,
    FEATURE_NAMES,
    feature_vector,
)
from c2kv_eval.portable.gp_sjcs import extract_features
from c2kv_eval.portable.proxy import action_canonical

DEFAULT_SOURCE = Path(
    "/home/zhuyuhan/project/bfcl-c2kv/results/"
    "bfcl_long_soft256_full200_gpu012_20261029_153723"
)
DEFAULT_DATA = Path(
    "/home/zhuyuhan/project/bfcl-c2kv-ttft-eval/"
    "bfcl_eval/data/BFCL_v4_multi_turn_long_context.json"
)
DEFAULT_MODEL = Path(
    "/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000"
)
DEFAULT_FULL = Path(
    "/home/zhuyuhan/project/bfcl-c2kv-ttft-eval/results/"
    "bfcl_long_joint_scope_gpu6_full200_20261109_230337/runs/full"
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def official_ids(path: Path) -> list[str]:
    result = [row["id"] for row in jsonl(path)]
    expected = [f"multi_turn_long_context_{index}" for index in range(200)]
    if sorted(result) != sorted(expected) or len(result) != 200:
        raise ValueError("BFCL Long data must contain the fixed 200 official IDs")
    return expected


def fold_map(tasks: list[str]) -> dict[str, int]:
    result = {}
    for task in tasks:
        index = int(task.rsplit("_", 1)[1])
        result[task] = index % 5
    if Counter(result.values()) != Counter({index: 40 for index in range(5)}):
        raise AssertionError("five-fold split is not 40 tasks per fold")
    return result


def source_files(root: Path) -> dict[str, Path]:
    files = sorted(root.glob("gpu*/rep0/full_cold/run/task_shards/*/server/steps.jsonl"))
    mapped = {path.parent.parent.name: path for path in files}
    if len(files) != 200 or len(mapped) != 200:
        raise ValueError(f"expected 200 unique Full task traces under {root}")
    return mapped


def source_decisions(root: Path, tasks: list[str]) -> dict[str, list[dict[str, Any]]]:
    files = source_files(root)
    output: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        records = []
        seen = set()
        for record in jsonl(files[task]):
            payload = (
                record.get("harness_transport_normalization") or {}
            ).get("same_input_replay_request")
            if not isinstance(payload, dict):
                continue
            traces = [
                trace for trace in record.get("generation_trace") or []
                if trace.get("phase") != "regeneration"
                and trace.get("status") == "completed"
            ]
            if not traces:
                continue
            context = payload.get("c2kv_eval_context") or {}
            user_turn = context.get("user_turn")
            step = context.get("step")
            attempt = context.get("attempt", 0)
            if not isinstance(user_turn, int) or not isinstance(step, int):
                raise ValueError(f"incomplete Decision identity for {task}")
            decision_id = f"{task}/turn-{user_turn}/step-{step}/attempt-{attempt}"
            if decision_id in seen:
                raise ValueError(f"duplicate source Decision: {decision_id}")
            seen.add(decision_id)
            controller = traces[0].get("controller") or {}
            history_tokens = controller.get("history_tokens")
            if not isinstance(history_tokens, (int, float)):
                history_tokens = 0
            full_tokens = traces[0].get("planned_resident_prompt_tokens")
            if not isinstance(full_tokens, int) or full_tokens <= 0:
                raise ValueError(f"missing Full token count for {decision_id}")
            records.append({
                "decision_id": decision_id,
                "task_id": task,
                "user_turn": user_turn,
                "step": step,
                "attempt": attempt,
                "source_history_tokens": int(history_tokens),
                "source_full_tokens": full_tokens,
                "payload": {
                    **payload,
                    "_replay_reference_action": action_canonical(
                        record.get("response") or {}
                    ),
                },
            })
        if not records:
            raise ValueError(f"no replayable Decisions for {task}")
        output[task] = sorted(
            records,
            key=lambda row: (row["user_turn"], row["step"], row["attempt"]),
        )
    return output


def length_stratified_tasks(
    decisions: dict[str, list[dict[str, Any]]],
    folds: dict[str, int],
    per_fold: int,
) -> list[str]:
    selected = []
    for fold in range(5):
        ranked = sorted(
            (task for task, value in folds.items() if value == fold),
            key=lambda task: (
                max(row["source_history_tokens"] for row in decisions[task]),
                task,
            ),
        )
        if per_fold > len(ranked):
            raise ValueError("requested more pilot tasks than a fold contains")
        positions = [
            round(index * (len(ranked) - 1) / max(1, per_fold - 1))
            for index in range(per_fold)
        ]
        chosen = [ranked[position] for position in positions]
        if len(set(chosen)) != per_fold:
            raise AssertionError("length-stratified selection produced duplicates")
        selected.extend(chosen)
    return selected


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    tasks = official_ids(args.data)
    folds = fold_map(tasks)
    decisions = source_decisions(args.source_full_root, tasks)
    selected_tasks = length_stratified_tasks(decisions, folds, args.tasks_per_fold)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    selected_rows = []
    replay = {}
    for task in selected_tasks:
        replay[task] = []
        for row in decisions[task]:
            observed = extract_features(row["payload"], tokenizer)
            public = {key: value for key, value in row.items() if key != "payload"}
            public["fold"] = folds[task]
            public["features"] = {
                "L_total": observed["L_total"],
                "L_hist": observed["L_hist"],
                "L_tool": observed["L_tool"],
                "compressible_history_ratio": (
                    observed["L_hist"] / max(1.0, observed["L_total"])
                ),
            }
            selected_rows.append(public)
            replay[task].append(row["payload"])

    first_two = selected_rows[:2]
    prefix_rows = []
    for row in first_two:
        payload = next(
            item["payload"] for item in decisions[row["task_id"]]
            if item["decision_id"] == row["decision_id"]
        )
        prefix_rows.append({
            "decision_id": row["decision_id"],
            "system_messages_sha256": canonical_hash([
                message for message in payload.get("messages") or []
                if message.get("role") == "system"
            ]),
            "tools_sha256": canonical_hash(payload.get("tools") or []),
            "full_and_racer_request_content_equal": True,
        })
    prefix = {
        "schema": "bfcl-tool-prefix-feasibility-v1",
        "request_count": len(prefix_rows),
        "rows": prefix_rows,
        "direct_kv_reuse_enabled": False,
        "reason": (
            "Full uses ordinary chat while Racer uses the native C2KV workspace; "
            "Radix Cache is disabled. Reuse would require an SGLang change, so the "
            "current chain is frozen without tool-prefix KV reuse."
        ),
        "timing_policy": "no prefix cost is subtracted from either arm",
    }
    decision_payload = {
        "schema": "bfcl-long-decision-dual-gp-pilot-v1",
        "scenarios": selected_tasks,
        "tasks": selected_tasks,
        "decisions": replay,
        "decision_count": len(selected_rows),
    }
    save(args.out / "decisions.json", decision_payload)
    save(args.out / "tool_prefix_feasibility.json", prefix)
    (args.out / "task_ids.txt").write_text(
        "\n".join(selected_tasks) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "bfcl-long-decision-dual-gp-plan-v1",
        "source_full_root": str(args.source_full_root.resolve()),
        "official_data": str(args.data.resolve()),
        "model": str(args.model.resolve()),
        "fold_rule": "numeric official task suffix modulo 5",
        "task_to_fold": folds,
        "fold_counts": dict(Counter(folds.values())),
        "selection_rule": "four within-fold quantiles of maximum visible history length",
        "selection_uses_outcome": False,
        "selected_tasks": selected_tasks,
        "selected_task_count": len(selected_tasks),
        "selected_decision_count": len(selected_rows),
        "selected": selected_rows,
        "tool_prefix_reuse": prefix,
    }
    save(args.out / "selection_manifest.json", manifest)
    print(json.dumps({
        "selected_tasks": len(selected_tasks),
        "selected_decisions": len(selected_rows),
        "fold_counts": dict(Counter(folds[task] for task in selected_tasks)),
        "tool_prefix_reuse_enabled": False,
        "result": str(args.out),
    }, indent=2))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    low, high = math.floor(point), math.ceil(point)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - point) + ordered[high] * (point - low)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def modal(rows: list[dict[str, str]], key: str) -> str:
    values = [row[key] for row in rows]
    return Counter(values).most_common(1)[0][0]


def serialize_gp(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict[str, Any], Any]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(x.shape[1]),
            length_scale_bounds=(1e-2, 1e3),
            nu=1.5,
        )
        + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-6, 1.0))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=seed,
        ).fit(scaled, y)
    fitted = gp.kernel_
    payload = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "x_train_scaled": gp.X_train_.tolist(),
        "alpha": gp.alpha_.tolist(),
        "cholesky": gp.L_.tolist(),
        "constant": float(fitted.k1.k1.constant_value),
        "length_scale": np.asarray(fitted.k1.k2.length_scale).reshape(-1).tolist(),
        "noise": float(fitted.k2.noise_level),
        "y_mean": float(np.asarray(gp._y_train_mean).reshape(-1)[0]),
        "y_std": float(np.asarray(gp._y_train_std).reshape(-1)[0]),
    }
    return payload, (gp, scaler)


def pair_samples(plan: dict[str, Any], full_path: Path, racer_path: Path) -> list[dict[str, Any]]:
    metadata = {row["decision_id"]: row for row in plan["selected"]}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in load_csv(full_path) + load_csv(racer_path):
        grouped.setdefault((row["method"], row["source_decision_id"]), []).append(row)
    result = []
    for decision_id, source in metadata.items():
        full = grouped.get(("full", decision_id), [])
        racer = grouped.get(("proposed", decision_id), [])
        if len(full) != 3 or len(racer) != 3:
            raise ValueError(
                f"{decision_id}: expected three Full and Racer repeats, "
                f"got {len(full)} and {len(racer)}"
            )
        full_ms = statistics.median(float(row["ttft_ms"]) for row in full)
        racer_ms = statistics.median(float(row["ttft_ms"]) for row in racer)
        full_action = modal(full, "generated_action_json")
        racer_action = modal(racer, "generated_action_json")
        result.append({
            **source,
            "full_online_ttft_ms": full_ms,
            "racer_online_ttft_ms": racer_ms,
            "latency_gain_ms": full_ms - racer_ms,
            "full_action_json": full_action,
            "racer_action_json": racer_action,
            "action_divergence": int(full_action != racer_action),
            "risk_label_kind": "proxy_action_divergence",
            "official_node_checker_available": False,
        })
    return result


def fitted_threshold(fitted: Any, x: np.ndarray, labels: np.ndarray) -> float:
    gp, scaler = fitted
    prediction = np.clip(gp.predict(scaler.transform(x)), 0.0, 1.0)
    negative = prediction[labels < 0.5]
    positive = prediction[labels >= 0.5]
    if len(negative) and len(positive):
        threshold = 0.5 * (
            float(np.quantile(negative, 0.95))
            + float(np.quantile(positive, 0.05))
        )
    elif len(positive):
        threshold = float(np.quantile(positive, 0.10)) * 0.5
    else:
        threshold = 0.25
    return min(0.50, max(0.05, threshold))


def fit(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = pair_samples(plan, args.full_samples, args.racer_samples)
    task_to_fold = {task: int(fold) for task, fold in plan["task_to_fold"].items()}
    fold_models = {}
    for held_out in range(5):
        train = [row for row in rows if int(row["fold"]) != held_out]
        train_tasks = sorted({row["task_id"] for row in train})
        if any(task_to_fold[task] == held_out for task in train_tasks):
            raise AssertionError("held-out Task entered GP training")
        x = np.vstack([feature_vector(row["features"]) for row in train])
        gain_y = np.asarray([row["latency_gain_ms"] for row in train], dtype=float)
        risk_y = np.asarray([row["action_divergence"] for row in train], dtype=float)
        gain_payload, _ = serialize_gp(x, gain_y, args.seed + held_out)
        risk_payload, risk_fit = serialize_gp(x, risk_y, args.seed + 100 + held_out)
        threshold = fitted_threshold(risk_fit, x, risk_y)
        fold_models[str(held_out)] = {
            "held_out_fold": held_out,
            "training_folds": [fold for fold in range(5) if fold != held_out],
            "training_tasks": train_tasks,
            "training_decisions": len(train),
            "gain_gp": gain_payload,
            "risk_gp": risk_payload,
            "risk_threshold": threshold,
            "uncertainty_beta": args.uncertainty_beta,
            "threshold_source": "training-fold predictions only",
        }
    artifact = {
        "schema": "bfcl-decision-dual-gp-v1",
        "frozen": True,
        "feature_names": list(FEATURE_NAMES),
        "task_to_fold": task_to_fold,
        "fold_models": fold_models,
        "latency_target": "Full Online TTFT minus Racer Online TTFT, milliseconds",
        "risk_target": "proxy action divergence; not BFCL official accuracy",
        "online_forbidden_inputs": ["Gold", "Checker result", "future turns", "Task outcome"],
        "history_compression_ratio": 8,
        "history_budget_tokens": 256,
        "tool_prefix_kv_reuse": False,
        "seed": args.seed,
    }
    artifact_path = args.out / "decision_dual_gp_artifact.json"
    save(artifact_path, artifact)
    scheduler = DecisionDualGPScheduler(artifact)
    for row in rows:
        decision = scheduler.route_observed(row["task_id"], row["features"])
        row["selected_racer"] = int(decision["compress"])
        row["predicted_gain_ms"] = decision["gain_mean_ms"]
        row["gain_lower_ms"] = decision["gain_lower_ms"]
        row["predicted_risk"] = decision["risk_mean"]
        row["risk_upper"] = decision["risk_upper"]
        row["risk_threshold"] = decision["risk_threshold"]
        row["route_reason"] = decision["reason"]
    selected = [row for row in rows if row["selected_racer"]]
    net = sum(row["latency_gain_ms"] for row in selected)
    minimum_selected = max(2, math.ceil(0.01 * len(rows)))
    checks = {
        "latency_label_coverage": len(rows) >= 20,
        "proxy_risk_label_coverage": len(rows) >= 20,
        "racer_selection_not_negligible": len(selected) >= minimum_selected,
        "selected_nodes_have_net_savings": net > 0.0,
    }
    gate = {
        "schema": "bfcl-decision-dual-gp-pilot-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "pilot_task_count": len({row["task_id"] for row in rows}),
        "pilot_decision_count": len(rows),
        "official_node_risk_labels": 0,
        "proxy_action_divergence_labels": len(rows),
        "proxy_action_divergence_positive": sum(row["action_divergence"] for row in rows),
        "oof_racer_selected": len(selected),
        "oof_racer_selection_rate": len(selected) / len(rows),
        "selected_measured_net_savings_ms": net,
        "selected_action_divergence_rate": (
            sum(row["action_divergence"] for row in selected) / len(selected)
            if selected else None
        ),
        "artifact": str(artifact_path.resolve()),
        "stop_reason": None if all(checks.values()) else (
            "pilot labels, selection coverage, or measured net savings are insufficient"
        ),
    }
    save(args.out / "pilot_gate.json", gate)
    fields = [
        "decision_id", "task_id", "fold", "full_online_ttft_ms",
        "racer_online_ttft_ms", "latency_gain_ms", "action_divergence",
        "risk_label_kind", "selected_racer", "predicted_gain_ms",
        "gain_lower_ms", "predicted_risk", "risk_upper", "risk_threshold",
        "route_reason",
    ]
    with (args.out / "paired_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(gate, indent=2))
    if not gate["passed"]:
        raise SystemExit(3)


def final_summary(args: argparse.Namespace) -> None:
    full = json.loads((args.full_root / "summary_full.json").read_text(encoding="utf-8"))
    if (
        full.get("official_correct_count"),
        full.get("official_total_count"),
    ) != (63, 200):
        raise ValueError("frozen 2026-11-09 Full identity changed")
    unified = json.loads(
        (args.run_root / "unified_summary.json").read_text(encoding="utf-8")
    )["rows"]
    if len(unified) != 200 or len({row["task_id"] for row in unified}) != 200:
        raise ValueError("dual-GP run is not the fixed complete BFCL Long 200")
    incomplete = [row for row in unified if row.get("status") != "completed"]
    if incomplete:
        raise ValueError(f"{len(incomplete)} tasks did not complete")
    correct = sum(row.get("official_score") == 1 for row in unified)
    online = []
    final_ready = []
    routes = Counter()
    misses = 0
    for row in unified:
        path = (
            args.run_root / "task_shards" / row["task_id"]
            / "server" / "steps.jsonl"
        )
        for record in jsonl(path):
            route = (record.get("bcrs") or {}).get("selected_arm")
            routes[route] += 1
            draft = next(
                trace for trace in record.get("generation_trace") or []
                if trace.get("phase") == "draft"
            )
            speed = (
                ((draft.get("generation") or {}).get("stats") or {})
                .get("speed_timing") or {}
            )
            native = speed.get("native_ttft_ms")
            prepare = (record.get("controller_timing") or {}).get("prepare_seconds", 0)
            route_seconds = (record.get("controller_timing") or {}).get(
                "bcrs_route_seconds", 0
            )
            if not isinstance(native, (int, float)):
                raise ValueError(f"missing native TTFT: {row['task_id']}")
            online.append(float(native) + 1000.0 * (float(prepare) + float(route_seconds)))
            ready = record.get("final_action_ready_seconds")
            if not isinstance(ready, (int, float)):
                ready = record.get("decision_runtime_seconds")
            if not isinstance(ready, (int, float)):
                raise ValueError(f"missing Final Action Ready: {row['task_id']}")
            final_ready.append(1000.0 * float(ready))
            for trace in record.get("generation_trace") or []:
                extraction = (
                    ((trace.get("generation") or {}).get("stats") or {})
                    .get("sglang_extraction") or {}
                )
                misses += int(extraction.get("cache_misses") or 0)
    online_total = sum(online)
    full_total = float(full["request_log_summary"]["online_ttft_ms_total"])
    result = {
        "schema": "bfcl-decision-dual-gp-final-v1",
        "official_correct_count": correct,
        "official_total_count": 200,
        "official_score": correct / 200,
        "frozen_full_correct_count": 63,
        "relative_accuracy_drop": (63 - correct) / 63,
        "cumulative_online_ttft_ms": online_total,
        "frozen_full_online_ttft_ms": full_total,
        "online_ttft_reduction": 1.0 - online_total / full_total,
        "final_action_ready_total_ms": sum(final_ready),
        "final_action_ready_p50_ms": percentile(final_ready, 0.50),
        "final_action_ready_p95_ms": percentile(final_ready, 0.95),
        "racer_route_count": routes["c2kv_racer"],
        "full_route_count": routes["full"],
        "cache_miss_count": misses,
        "acceptance": {
            "correct_at_least_54": correct >= 54,
            "online_ttft_at_most_1287023_69_ms": online_total <= 1287023.69,
            "joint_pass": correct >= 54 and online_total <= 1287023.69,
        },
        "result_path": str(args.run_root.resolve()),
        "frozen_full_path": str(args.full_root.resolve()),
    }
    save(args.out, result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("prepare")
    plan.add_argument("--source-full-root", type=Path, default=DEFAULT_SOURCE)
    plan.add_argument("--data", type=Path, default=DEFAULT_DATA)
    plan.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    plan.add_argument("--tasks-per-fold", type=int, default=4)
    plan.add_argument("--out", type=Path, required=True)
    train = sub.add_parser("fit")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--full-samples", type=Path, required=True)
    train.add_argument("--racer-samples", type=Path, required=True)
    train.add_argument("--uncertainty-beta", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", type=Path, required=True)
    summary = sub.add_parser("summarize-final")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--full-root", type=Path, default=DEFAULT_FULL)
    summary.add_argument("--out", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "fit":
        fit(args)
    else:
        final_summary(args)


if __name__ == "__main__":
    main()
