from __future__ import annotations

import json

from c2kv_eval.portable.gp_sjcs import (
    CANDIDATES, GPSJCSScheduler, extract_features,
)
from c2kv_eval.portable.reqlog import summarize


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking, tools=None):
        assert tokenize and add_generation_prompt and not enable_thinking
        size = sum(len(json.dumps(message, sort_keys=True)) for message in messages)
        size += len(json.dumps(tools, sort_keys=True)) if tools else 0
        return list(range(max(1, size // 4)))

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(max(1, len(text) // 4))) if text else []


def test_a0_is_uncompressed_and_history_budget_is_capped():
    assert CANDIDATES[0].history_target(5000, 256) is None
    assert CANDIDATES[1].history_target(5000, 256) == 256
    assert CANDIDATES[1].history_target(800, 256) == 100

def test_fixed_candidate_profile_arm_selects_requested_candidate():
    # Profile arms bypass GP routing but retain the exact runtime materialization.
    artifact = {
        "schema": "gp-sjcs-artifact-v1",
        "models": {"semantic": {
            "scaler_mean": [0.0] * 10,
            "scaler_scale": [1.0] * 10,
            "x_train_scaled": [[0.0] * 10],
            "alpha": [0.0],
            "cholesky": [[1.0]],
            "constant": 1.0,
            "length_scale": [1.0] * 10,
            "noise": 0.0,
            "y_mean": 0.0,
            "y_std": 1.0,
        }},
    }
    scheduler = GPSJCSScheduler(artifact, "gp_profile_a3", history_budget=256)
    decision = scheduler.route({
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
    }, FakeTokenizer())
    assert decision["selected_config"] == "A3"
    assert decision["selection_reason"] == "fixed_candidate_profile"
    assert decision["history_kv"]["retention_ratio"] == 0.5


def test_query_only_features_do_not_require_gold():
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "remember account 42"},
            {"role": "assistant", "content": "account 42 is active"},
            {"role": "user", "content": "fetch account 42"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "fetch_account",
                "description": "fetch account",
                "parameters": {"type": "object", "properties": {
                    "account_id": {"type": "string"}}},
            },
        }],
    }
    features = extract_features(payload, FakeTokenizer())
    assert features["L_total"] > 0
    assert features["L_tool"] > 0
    assert features["L_hist"] > 0
    assert features["S_tool"] > 0
    assert features["S_hist"] > 0


def test_online_ttft_includes_proxy_preparation():
    rows = [{
        "status": "ok",
        "conv_id": "x",
        "server_ttft_ms": 100.0,
        "assemble_sec": 0.01,
        "gp_controller_preparation_ms": 2.0,
        "gp_sjcs": {"selected_config": "A2"},
    }]
    summary = summarize(rows)
    assert summary["online_ttft_ms_total"] == 110.0
    assert summary["gp_selected_config_counts"] == {"A2": 1}
