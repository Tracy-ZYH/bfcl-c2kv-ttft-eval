"""Binary BCRS delegation over the existing Full and Racer runners."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from types import SimpleNamespace

from .event_native_step import EventNativeDecisionRunner
BFCL_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_PORTABLE_CHAT_FIELDS = frozenset({
    "messages", "model", "tools", "temperature", "max_tokens",
    "max_completion_tokens", "stream", "seed", "store", "top_p",
    "presence_penalty", "frequency_penalty", "chat_template_kwargs",
    "tool_choice", "parallel_tool_calls", "n", "c2kv_eval_context",
})
_PORTABLE_FULL_ARM = SimpleNamespace(
    history_kv=None, kv_reuse=None, constrain_tools=False,
)


def _portable_full_payload(
    payload, *, disable_tools=False, max_new_tokens=None,
):
    """Build BCRO Full with the existing portable SGLang Full entrypoint."""

    staged = {
        key: copy.deepcopy(value)
        for key, value in dict(payload).items()
        if key in _PORTABLE_CHAT_FIELDS
    }
    if disable_tools:
        staged.pop("tools", None)
        staged.pop("tool_choice", None)
        staged.pop("parallel_tool_calls", None)
    if max_new_tokens is not None:
        staged.pop("max_tokens", None)
        staged["max_completion_tokens"] = int(max_new_tokens)
    messages = list(staged.get("messages") or ())
    if not any(message.get("role") == "system" for message in messages):
        messages.insert(
            0,
            {"role": "system", "content": BFCL_DEFAULT_SYSTEM_PROMPT},
        )
    staged["messages"] = messages
    try:
        from c2kv_eval.portable.backends.sglang import SglangBackend
    except ImportError as error:
        raise RuntimeError(
            "BCRO Full requires the portable SGLang request builder"
        ) from error
    return SglangBackend(lambda *_args, **_kwargs: None).prepare_chat(
        staged, _PORTABLE_FULL_ARM, None, context={},
    )


class _OrdinaryFullRunner:
    """Use the ordinary SGLang chat path for BCRS Full decisions."""

    def __init__(self, generator, *, model_name, max_new_tokens,
                 max_generation_calls, journal):
        self.generator = generator
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_generation_calls = max_generation_calls
        self.journal = journal
        self.generation_calls = 0

    def run(self, payload, *, disable_tools=False, max_new_tokens=None):
        started = time.perf_counter()
        record = {
            "schema": "a-ordinary-full-step-v1",
            "status": "started",
            "session_id": payload["session_id"],
            "decision_key": payload["decision_key"],
            "generation_trace": [],
            "response": None,
            "controller_timing": {
                "prepare_seconds": 0.0,
                "reconsider_seconds": 0.0,
            },
        }
        request_id = json.dumps(
            [payload["session_id"], payload["decision_key"]],
            separators=(",", ":"),
        )
        handle = self.journal.start(
            "generation", self.generation_calls + 1, request_id,
            {
                "task_id": payload["session_id"],
                "decision_id": payload["decision_key"],
            },
        )
        self.generation_calls += 1
        try:
            effective_max_new_tokens = (
                self.max_new_tokens
                if max_new_tokens is None else int(max_new_tokens)
            )
            prepare_started = time.perf_counter()
            prepared_payload = _portable_full_payload(
                payload, disable_tools=disable_tools,
                max_new_tokens=effective_max_new_tokens,
            )
            record["controller_timing"]["prepare_seconds"] = (
                time.perf_counter() - prepare_started
            )
            data = self.generator.generate_ordinary_full(
                prepared_payload, model_name=self.model_name,
                max_new_tokens=effective_max_new_tokens,
            )
            choices = data.get("choices")
            usage = data.get("usage")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError(
                    "ordinary Full response must contain exactly one choice"
                )
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError(
                    "ordinary Full response has no assistant message"
                )
            if not isinstance(usage, dict):
                raise ValueError("ordinary Full response has no usage")
            usage_snapshot = {
                key: int(usage[key])
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            }
            metadata = data.get("metadata") or {}
            parity = metadata.get("parity_debug") or {}
            prompt_hash = parity.get("prompt_ids_sha256")
            prompt_count = parity.get("prompt_token_count")
            if (
                not isinstance(prompt_hash, str)
                or len(prompt_hash) != 64
                or any(char not in "0123456789abcdef" for char in prompt_hash)
                or type(prompt_count) is not int
                or prompt_count != usage_snapshot["prompt_tokens"]
                or parity.get("canonical_reconstruction_matches") is not True
            ):
                raise ValueError(
                    "ordinary Full server token parity receipt is missing or inconsistent"
                )
            speed = copy.deepcopy(metadata.get("speed_timing") or {})
            if "native_ttft_ms" not in speed and isinstance(
                    speed.get("server_ttft_ms"), (int, float)):
                speed["native_ttft_ms"] = speed["server_ttft_ms"]
                speed["native_ttft_source"] = "ordinary_chat_server_ttft_ms"
            record["generation_trace"].append(
                {
                    "attempt_uid": handle.attempt_uid,
                    "attempt_index": handle.attempt_index,
                    "phase": "draft",
                    "status": "completed",
                    "discarded": False,
                    "usage": usage_snapshot,
                    "generation": {
                        "stats": {
                            "ordinary_full_path": True,
                            "tools_disabled": bool(disable_tools),
                            "max_new_tokens": effective_max_new_tokens,
                            "ordinary_full_wall_ms": data[
                                "_ordinary_full_wall_ms"
                            ],
                            "speed_timing": speed,
                            "server_token_parity": parity,
                        }
                    },
                }
            )
            record["response"] = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls") or [],
                "reasoning_content": message.get("reasoning_content"),
                "native_parse_status": "ordinary_sglang",
                "native_parse_reason": None,
                "finish_reason": choice.get("finish_reason") or "stop",
            }
            record["generation_usage_total"] = usage_snapshot
            record["generation_usage_known"] = copy.deepcopy(usage_snapshot)
            record["generation_attempts"] = 1
            record["generation_completed"] = 1
            record["ordinary_full_path"] = {
                "request_contract": {
                    "messages_sha256": _canonical_sha256(data["_ordinary_full_request"].get("messages") or []),
                    "tools_sha256": _canonical_sha256(data["_ordinary_full_request"].get("tools") or []),
                    "chat_template_kwargs_sha256": _canonical_sha256(data["_ordinary_full_request"].get("chat_template_kwargs") or {}),
                    "parallel_tool_calls": data["_ordinary_full_request"].get("parallel_tool_calls"),
                    "tool_choice": data["_ordinary_full_request"].get("tool_choice"),
                    "model": data["_ordinary_full_request"].get("model"),
                    "logprobs": data["_ordinary_full_request"].get("logprobs"),
                    "top_logprobs": data["_ordinary_full_request"].get("top_logprobs"),
                    "rid_present": bool(data["_ordinary_full_request"].get("rid")),
                },
                "endpoint": "/v1/chat/completions",
                "server_prompt_ids_sha256": prompt_hash,
                "server_prompt_token_count": prompt_count,
                "sampling": {
                    key: data["_ordinary_full_request"].get(key)
                    for key in (
                        "temperature", "top_p", "top_k", "seed",
                        "max_tokens", "max_completion_tokens", "stream",
                    )
                    if key in data["_ordinary_full_request"]
                },
            }
            record["status"] = "ok"
            self.journal.finish(handle, "completed", usage=usage_snapshot)
            return record
        except Exception as error:
            record["status"] = "failed"
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            self.journal.finish(handle, "failed")
            from .event_native_step import EventNativeStepError
            raise EventNativeStepError(str(error), record) from error
        finally:
            record["decision_runtime_seconds"] = time.perf_counter() - started



class BCRSDecisionRunner:
    """Select an existing runner; never reimplement generation or recovery."""

    def __init__(self, racer_controller, full_controller, generator, tokenizer, *,
                 scheduler, ratio, max_new_tokens, max_generation_calls, journal, model_name,
                 exact_identifier_guard=False, readonly_repeat_guard=False,
                 readonly_repeat_previous_turn=False,
                 post_tweet_literal_guard=False,
                 cancel_booking_full_diversion=False,
                 readonly_repeat_no_tools_max_tokens=None):
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.generator = generator
        self.max_generation_calls = max_generation_calls
        self.generation_calls = 0
        self.guard_config = {
            "exact_identifier": bool(exact_identifier_guard),
            "readonly_repeat": bool(readonly_repeat_guard),
            "readonly_repeat_previous_turn": bool(readonly_repeat_previous_turn),
            "post_tweet_literal": bool(post_tweet_literal_guard),
            "cancel_booking_full_diversion": bool(cancel_booking_full_diversion),
            "readonly_repeat_no_tools_max_tokens": (
                int(readonly_repeat_no_tools_max_tokens)
                if readonly_repeat_no_tools_max_tokens is not None else None
            ),
        }
        no_tools_limit = self.guard_config["readonly_repeat_no_tools_max_tokens"]
        if no_tools_limit is not None and (
            not self.guard_config["readonly_repeat"] or no_tools_limit <= 0
        ):
            raise ValueError("bounded no-tools retry requires readonly-repeat guard and a positive limit")
        pre_recovery_guard = None
        if cancel_booking_full_diversion:
            from .precommit_guards import booking_cancel_full_diversion_guard
            pre_recovery_guard = booking_cancel_full_diversion_guard
        common = dict(
            generator=generator,
            tokenizer=tokenizer,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            max_generation_calls=max_generation_calls,
            journal=journal,
        )
        self.runners = {
            "c2kv_racer": EventNativeDecisionRunner(
                racer_controller, **common, legal_cache_prewarm=True,
                require_draft_cache_hit=True,
                pre_recovery_guard=pre_recovery_guard),
            "full": _OrdinaryFullRunner(
                generator, model_name=model_name, max_new_tokens=max_new_tokens,
                max_generation_calls=max_generation_calls, journal=journal),
        }

    def run(self, payload):
        started = time.perf_counter()
        decision = self.scheduler.route(payload, self.tokenizer)
        route_seconds = time.perf_counter() - started
        selected = decision.get("selected_arm")
        if selected not in self.runners:
            raise ValueError(f"BCRS selected unsupported runner: {selected!r}")
        runner = self.runners[selected]
        runner.generation_calls = self.generation_calls
        record = runner.run(payload)
        self.generation_calls = runner.generation_calls
        diversion = record.get("pre_recovery_diversion") or {}
        if selected == "c2kv_racer" and diversion.get("triggered"):
            record = self._apply_cancel_booking_full_diversion(payload, record)
        elif selected == "c2kv_racer" and any(
            self.guard_config[name]
            for name in ("exact_identifier", "readonly_repeat", "post_tweet_literal")
        ):
            record = self._apply_precommit_guard(payload, record)
        result = copy.deepcopy(record)
        result["bcrs"] = copy.deepcopy(decision)
        result["bcrs"]["measured_route_seconds"] = route_seconds
        result["decision_runtime_seconds"] = (
            float(result.get("decision_runtime_seconds") or 0.0) + route_seconds
        )
        if "final_action_ready_seconds" in result:
            result["final_action_ready_seconds"] = (
                float(result["final_action_ready_seconds"]) + route_seconds)
        timing = result.setdefault("controller_timing", {})
        timing["bcrs_route_seconds"] = route_seconds
        return result

    @staticmethod
    def _draft_message(record):
        response = record.get("response") or {}
        return {
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": copy.deepcopy(response.get("tool_calls") or []),
        }

    @staticmethod
    def _raw_source_indices(record):
        trace = record.get("generation_trace") or []
        if not trace:
            return []
        controller = trace[0].get("controller") or {}
        return list(controller.get("raw_source_indices") or [])

    @staticmethod
    def _recovery_already_used(record):
        return int(record.get("generation_attempts") or 0) > 1

    @staticmethod
    def _apply_post_tweet_literal(record, check):
        result = copy.deepcopy(record)
        calls = (result.get("response") or {}).get("tool_calls") or []
        index = check["tool_call_index"]
        if not 0 <= index < len(calls):
            raise ValueError("post_tweet literal guard call index is out of range")
        function = calls[index].get("function") or {}
        raw_arguments = function.get("arguments") or {}
        was_string = isinstance(raw_arguments, str)
        arguments = json.loads(raw_arguments) if was_string else copy.deepcopy(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("post_tweet literal guard arguments are not an object")
        if arguments.get("content") != check["before_content"]:
            raise ValueError("post_tweet literal guard draft changed before commit")
        before_other = {
            key: copy.deepcopy(value) for key, value in arguments.items()
            if key != "content"
        }
        arguments["content"] = check["after_content"]
        after_other = {
            key: copy.deepcopy(value) for key, value in arguments.items()
            if key != "content"
        }
        if before_other != after_other:
            raise AssertionError("post_tweet literal guard changed another argument")
        function["arguments"] = (
            json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            if was_string else arguments
        )
        return result

    @staticmethod
    def _merge_full_regeneration(draft_record, full_record, receipt):
        result = copy.deepcopy(draft_record)
        traces = result.setdefault("generation_trace", [])
        if traces:
            traces[-1]["discarded"] = True
        retry_traces = copy.deepcopy(full_record.get("generation_trace") or [])
        for trace in retry_traces:
            trace["phase"] = "regeneration"
            trace["regeneration_kind"] = "precommit_full_guard"
            trace["discarded"] = False
        traces.extend(retry_traces)
        result["response"] = copy.deepcopy(full_record["response"])
        result["precommit_guard"] = receipt
        result["generation_attempts"] = len(traces)
        result["generation_completed"] = sum(
            trace.get("status") == "completed" for trace in traces
        )
        for total_key in ("generation_usage_total", "generation_usage_known"):
            merged = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                left = (draft_record.get(total_key) or {}).get(key)
                right = (full_record.get(total_key) or {}).get(key)
                merged[key] = (
                    left + right
                    if isinstance(left, (int, float)) and isinstance(right, (int, float))
                    else None
                )
            result[total_key] = merged
        retry_seconds = float(full_record.get("decision_runtime_seconds") or 0.0)
        result["decision_runtime_seconds"] = (
            float(draft_record.get("decision_runtime_seconds") or 0.0)
            + retry_seconds
        )
        result["final_action_ready_seconds"] = result["decision_runtime_seconds"]
        timing = result.setdefault("controller_timing", {})
        timing["precommit_full_regeneration_seconds"] = retry_seconds
        return result

    def _apply_cancel_booking_full_diversion(self, payload, record):
        diversion = copy.deepcopy(record.get("pre_recovery_diversion") or {})
        traces = record.get("generation_trace") or []
        if len(traces) != 1 or traces[0].get("phase") != "draft":
            raise RuntimeError(
                "cancel-booking Full diversion must replace recovery after one draft"
            )
        if self._recovery_already_used(record):
            raise RuntimeError(
                "cancel-booking Full diversion cannot follow Racer recovery"
            )
        full_runner = self.runners["full"]
        full_runner.generation_calls = self.generation_calls
        full_record = full_runner.run(payload)
        self.generation_calls = full_runner.generation_calls
        receipt = {
            "schema": "bcro-recovery-diversion-v1",
            "enabled": True,
            "triggered": True,
            "triggered_guard": diversion.get("guard"),
            "trigger_reason": diversion.get("reason"),
            "shared_regeneration_budget": 1,
            "regeneration_count": 1,
            "racer_recovery_count": 0,
            "commit_source": "full_regeneration",
            "commit_reason": "cancel_booking_draft_without_tool_call",
            "draft_response_sha256": _canonical_sha256(record.get("response")),
            "final_response_sha256": _canonical_sha256(full_record.get("response")),
        }
        result = self._merge_full_regeneration(record, full_record, receipt)
        result["recovery_diversion"] = copy.deepcopy(receipt)
        return result

    def _apply_precommit_guard(self, payload, record):
        from .precommit_guards import (
            exact_identifier_guard,
            post_tweet_quoted_content_guard,
            readonly_repeat_guard,
        )

        raw = self._raw_source_indices(record)
        draft = self._draft_message(record)
        checks = []
        working_record = record
        if self.guard_config.get("post_tweet_literal", False):
            literal_check = post_tweet_quoted_content_guard(payload, draft, raw)
            checks.append(literal_check)
            if literal_check.get("triggered"):
                working_record = self._apply_post_tweet_literal(record, literal_check)
                draft = self._draft_message(working_record)
        if self.guard_config["exact_identifier"]:
            checks.append(exact_identifier_guard(payload, draft, raw))
        if self.guard_config["readonly_repeat"]:
            checks.append(readonly_repeat_guard(
                payload, draft, raw,
                include_previous_user_turn=self.guard_config.get(
                    "readonly_repeat_previous_turn", False),
            ))
        triggered = [check for check in checks if check.get("triggered")]
        regeneration_triggers = [
            check for check in triggered if check.get("regeneration_required", True)
        ]
        receipt = {
            "schema": "bcro-precommit-guard-v1",
            "enabled": copy.deepcopy(self.guard_config),
            "checks": checks,
            "triggered": bool(triggered),
            "triggered_guard": triggered[0]["guard"] if triggered else None,
            "shared_regeneration_budget": 1,
            "regeneration_count": 0,
            "commit_source": (
                "draft_normalized" if triggered and not regeneration_triggers else "draft"
            ),
        }
        if not regeneration_triggers:
            result = copy.deepcopy(working_record)
            result["precommit_guard"] = receipt
            if triggered:
                receipt["commit_reason"] = "post_tweet_quoted_content_preserved"
            result["final_action_ready_seconds"] = float(
                result.get("decision_runtime_seconds") or 0.0
            )
            return result
        if self._recovery_already_used(record):
            receipt["commit_reason"] = "shared_regeneration_budget_already_used"
            result = copy.deepcopy(record)
            result["precommit_guard"] = receipt
            result["final_action_ready_seconds"] = float(
                result.get("decision_runtime_seconds") or 0.0
            )
            return result
        readonly_repeat_triggered = any(
            check.get("guard") == "same_turn_readonly_repeat"
            for check in regeneration_triggers
        )
        no_tools_limit = (
            self.guard_config.get("readonly_repeat_no_tools_max_tokens")
            if readonly_repeat_triggered else None
        )
        full_runner = self.runners["full"]
        full_runner.generation_calls = self.generation_calls
        if no_tools_limit is None:
            full_record = full_runner.run(payload)
        else:
            full_record = full_runner.run(
                payload,
                disable_tools=True,
                max_new_tokens=no_tools_limit,
            )
        self.generation_calls = full_runner.generation_calls
        receipt.update({
            "regeneration_count": 1,
            "commit_source": "full_regeneration",
            "commit_reason": "precommit_guard_triggered",
            "regeneration_tools_disabled": no_tools_limit is not None,
            "regeneration_max_new_tokens": no_tools_limit,
            "draft_response_sha256": _canonical_sha256(record.get("response")),
            "final_response_sha256": _canonical_sha256(full_record.get("response")),
        })
        return self._merge_full_regeneration(record, full_record, receipt)

    def close(self):
        self.generator.close_session()

