"""Check the assembled history representation immediately before generation."""
from __future__ import annotations


def history_budget_receipt(memory, metadata, controller, *, ratio, phase):
    if getattr(controller, "view_mode", None) == "full_original":
        costs = memory.costs(ratio)
        resident = costs["resident_kv_tokens"]
        errors = []
        if (
            costs["gist_tokens"]
            or metadata.get("full_source_coverage") is not True
            or metadata.get("omitted_event_ids")
        ):
            errors.append("full_original_is_not_full_raw_history")
        if resident != metadata.get("resident_kv_tokens"):
            errors.append("resident_prompt_differs_from_controller_accounting")
        return {
            "schema": "a-full-original-pre-generation-integrity-v1",
            "phase": phase,
            "status": "rejected" if errors else "passed",
            "errors": errors,
            "history_budget_bytes": None,
            "active_history_bytes": metadata.get("history_bytes"),
            "declared_active_history_bytes": metadata.get("history_bytes"),
            "common_prompt_tokens": metadata.get("fixed_baseline_tokens"),
            "resident_prompt_tokens": resident,
            "active_gist_bytes": 0,
            "scope": "Full-original raw history integrity; history budget is intentionally unused.",
            "all_derived_workspace_content_charged": True,
        }
    policy = getattr(controller, "policy_config", None)
    unit = getattr(controller, "kv_bytes_per_token", None)
    common = metadata.get("common_raw_prompt_tokens")
    if policy is None or type(unit) is not int or type(common) is not int:
        raise ValueError("Hybrid search requires an explicit native history budget contract")
    costs = memory.costs(ratio)
    resident = costs["resident_kv_tokens"]
    history_tokens = resident - common
    active = history_tokens * unit
    declared = metadata.get("actual_history_bytes")
    cap = min(policy.history_budget_bytes, policy.workspace_budget_bytes)
    errors = []
    if history_tokens < 0:
        errors.append("negative_history_after_common_boundary")
    if active != declared:
        errors.append("assembled_history_differs_from_controller_accounting")
    soft_overflow = bool(getattr(policy, "whole_event_soft_budget", False))
    if active > cap and not soft_overflow:
        errors.append("active_history_exceeds_budget")
    return {
        "schema": "a-hybrid-pre-generation-budget-v1", "phase": phase,
        "status": "rejected" if errors else "passed", "errors": errors,
        "history_budget_bytes": cap, "requested_history_budget_bytes": cap,
        "active_history_bytes": active, "actual_history_tokens": history_tokens,
        "history_budget_overflow_bytes": max(0, active - cap),
        "whole_event_soft_budget": soft_overflow,
        "declared_active_history_bytes": declared, "kv_bytes_per_token": unit,
        "common_prompt_tokens": common, "resident_prompt_tokens": resident,
        "resident_prompt_kv_bytes": resident * unit,
        "active_gist_bytes": costs["gist_tokens"] * unit,
        "active_raw_and_derived_history_bytes": (history_tokens - costs["gist_tokens"]) * unit,
        "scope": "Assembled BF16 KV representation at loaded-model geometry; excludes weights, allocator overhead and CPU/disk archives.",
        "all_derived_workspace_content_charged": True,
    }
