import json
import unittest

from ..event_native_bcrs import BCRSDecisionRunner
from ..precommit_guards import (
    booking_cancel_full_diversion_guard,
    exact_identifier_guard,
    post_tweet_quoted_content_guard,
    readonly_repeat_guard,
)


def tool(name, properties, required=None, description=""):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in properties},
                "required": list(required or properties),
            },
        },
    }


class PrecommitGuardTest(unittest.TestCase):
    def test_post_tweet_preserves_unique_quoted_body_without_regeneration(self):
        payload = {
            "messages": [{
                "role": "user",
                "content": (
                    "Post 'Just cancelled my trip to LA' with tags "
                    "'TravelUpdate' and 'BusinessTrip'."
                ),
            }],
        }
        draft = {"tool_calls": [{"function": {
            "name": "post_tweet",
            "arguments": json.dumps({
                "content": "Just cancelled my trip to LA.",
                "tags": ["#TravelUpdate", "#BusinessTrip"],
            }),
        }}]}
        receipt = post_tweet_quoted_content_guard(payload, draft, [])
        self.assertTrue(receipt["triggered"])
        self.assertFalse(receipt["regeneration_required"])
        runner = object.__new__(BCRSDecisionRunner)
        result = runner._apply_post_tweet_literal({"response": draft}, receipt)
        arguments = json.loads(
            result["response"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["content"], "Just cancelled my trip to LA")
        self.assertEqual(arguments["tags"], ["#TravelUpdate", "#BusinessTrip"])

    def test_post_tweet_rule_rejects_nonunique_or_nonperiod_change(self):
        draft = {"tool_calls": [{"function": {
            "name": "post_tweet",
            "arguments": {"content": "hello!", "tags": []},
        }}]}
        payload = {"messages": [{"role": "user", "content": "Post 'hello'."}]}
        self.assertFalse(
            post_tweet_quoted_content_guard(payload, draft, [])["triggered"])
        draft["tool_calls"][0]["function"]["arguments"]["content"] = "hello."
        payload["messages"][0]["content"] = "Use 'hello', exactly 'hello'."
        self.assertFalse(
            post_tweet_quoted_content_guard(payload, draft, [])["triggered"])

    def test_booking_cancel_diversion_is_request_local_and_fail_closed(self):
        empty = {"content": "", "tool_calls": []}
        for text in (
            "Please cancel my recent booking.",
            "Reverse the reservation I just made.",
        ):
            payload = {"messages": [{"role": "user", "content": text}]}
            receipt = booking_cancel_full_diversion_guard(payload, empty, [])
            self.assertTrue(receipt["triggered"])
            self.assertEqual(
                receipt["reason"],
                "explicit_cancel_request_with_no_draft_tool_call",
            )
            decorated = dict(payload, task_id="ignored", gold={"ignored": True})
            self.assertEqual(
                receipt,
                booking_cancel_full_diversion_guard(decorated, empty, []),
            )

        payload = {
            "messages": [{"role": "user", "content": "Cancel my booking"}],
        }
        draft = {"tool_calls": [{"function": {
            "name": "cancel_booking", "arguments": {"booking_id": "B-7"},
        }}]}
        self.assertFalse(
            booking_cancel_full_diversion_guard(payload, draft, [])["triggered"])
        payload["messages"] = [{
            "role": "user", "content": "Do not cancel my booking",
        }]
        self.assertFalse(
            booking_cancel_full_diversion_guard(payload, empty, [])["triggered"])
        payload["messages"] = [
            {"role": "user", "content": "Cancel my booking"},
            {"role": "assistant", "tool_calls": [draft["tool_calls"][0]]},
        ]
        receipt = booking_cancel_full_diversion_guard(payload, empty, [])
        self.assertFalse(receipt["triggered"])
        self.assertTrue(receipt["cancel_already_called_this_turn"])

    def test_exact_identifier_requires_unprotected_history(self):
        payload = {
            "decision_key": "turn-1/step-0",
            "tools": [tool("cancel_booking", ["booking_id", "access_token"])],
            "messages": [
                {"role": "user", "content": "Find my booking"},
                {"role": "tool", "content": '{"booking_id":"B-7","access_token":"T-9"}'},
                {"role": "user", "content": "Cancel the booking"},
            ],
        }
        draft = {
            "tool_calls": [{"id": "c", "type": "function", "function": {
                "name": "cancel_booking",
                "arguments": '{"booking_id":"B-7","access_token":"T-9"}',
            }}]
        }
        receipt = exact_identifier_guard(payload, draft, [])
        self.assertTrue(receipt["triggered"])
        self.assertEqual({item["field"] for item in receipt["missing_evidence"]},
                         {"booking_id", "access_token"})
        self.assertFalse(exact_identifier_guard(payload, draft, [1])["triggered"])

    def test_current_user_identifier_does_not_trigger(self):
        payload = {
            "decision_key": "turn-1/step-0",
            "tools": [tool("retrieve_invoice", ["booking_id"])],
            "messages": [
                {"role": "tool", "content": '{"booking_id":"B-7"}'},
                {"role": "user", "content": "Retrieve invoice for booking B-7"},
            ],
        }
        draft = {"tool_calls": [{"function": {
            "name": "retrieve_invoice", "arguments": {"booking_id": "B-7"},
        }}]}
        self.assertFalse(exact_identifier_guard(payload, draft, [])["triggered"])

    def test_readonly_repeat_cycle_triggers(self):
        payload = {
            "decision_key": "turn-1/step-4",
            "tools": [tool("get_stock_info", ["symbol"], description="Get stock information")],
            "messages": [
                {"role": "user", "content": "Review my watchlist"},
                {"role": "assistant", "tool_calls": [{"id": "a", "function": {
                    "name": "get_stock_info", "arguments": '{"symbol":"TSLA"}'}}]},
                {"role": "tool", "tool_call_id": "a", "content": '{"price":1}'},
                {"role": "assistant", "tool_calls": [{"id": "b", "function": {
                    "name": "get_stock_info", "arguments": '{"symbol":"AAPL"}'}}]},
                {"role": "tool", "tool_call_id": "b", "content": '{"price":2}'},
            ],
        }
        draft = {"tool_calls": [{"id": "c", "function": {
            "name": "get_stock_info", "arguments": '{"symbol":"TSLA"}'}}]}
        self.assertTrue(readonly_repeat_guard(payload, draft, [])["triggered"])

    def test_mutation_breaks_repeat_cycle(self):
        payload = {
            "decision_key": "turn-1/step-3",
            "tools": [
                tool("get_stock_info", ["symbol"], description="Get stock information"),
                tool("place_order", ["symbol"]),
            ],
            "messages": [
                {"role": "user", "content": "Trade"},
                {"role": "assistant", "tool_calls": [{"id": "a", "function": {
                    "name": "get_stock_info", "arguments": {"symbol": "TSLA"}}}]},
                {"role": "tool", "tool_call_id": "a", "content": "old"},
                {"role": "assistant", "tool_calls": [{"id": "b", "function": {
                    "name": "place_order", "arguments": {"symbol": "TSLA"}}}]},
                {"role": "tool", "tool_call_id": "b", "content": "done"},
            ],
        }
        draft = {"tool_calls": [{"function": {
            "name": "get_stock_info", "arguments": {"symbol": "TSLA"}}}]}
        self.assertFalse(readonly_repeat_guard(payload, draft, [])["triggered"])

    def test_runner_uses_only_one_shared_regeneration(self):
        payload = {
            "session_id": "s", "decision_key": "turn-1/step-0",
            "tools": [tool("cancel_booking", ["booking_id"])],
            "messages": [
                {"role": "tool", "content": '{"booking_id":"B-7"}'},
                {"role": "user", "content": "Cancel the booking"},
            ],
        }
        record = {
            "response": {"content": "", "tool_calls": [{"function": {
                "name": "cancel_booking", "arguments": '{"booking_id":"B-7"}'}}]},
            "generation_trace": [{
                "status": "completed", "discarded": False,
                "controller": {"raw_source_indices": []},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }],
            "generation_attempts": 1,
            "generation_usage_total": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "generation_usage_known": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "decision_runtime_seconds": 1.0,
        }

        class Full:
            generation_calls = 0
            calls = 0
            def run(self, _payload):
                self.calls += 1
                self.generation_calls += 1
                usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
                return {
                    "response": {"content": "full", "tool_calls": []},
                    "generation_trace": [{
                        "phase": "draft", "status": "completed",
                        "discarded": False, "usage": usage}],
                    "generation_usage_total": usage,
                    "generation_usage_known": usage,
                    "decision_runtime_seconds": 2.0,
                }

        runner = object.__new__(BCRSDecisionRunner)
        runner.guard_config = {"exact_identifier": True, "readonly_repeat": True}
        runner.generation_calls = 1
        full = Full()
        runner.runners = {"full": full}
        result = runner._apply_precommit_guard(payload, record)
        self.assertEqual(full.calls, 1)
        self.assertEqual(result["precommit_guard"]["regeneration_count"], 1)
        self.assertEqual(result["response"]["content"], "full")
        self.assertEqual(result["generation_trace"][-1]["phase"], "regeneration")
        self.assertEqual(result["generation_trace"][-1]["regeneration_kind"], "precommit_full_guard")
        self.assertEqual(result["final_action_ready_seconds"], 3.0)


    def test_cancel_booking_diversion_uses_one_full_retry_without_racer_recovery(self):
        usage1 = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        record = {
            "pre_recovery_diversion": {
                "guard": "booking_cancel_full_diversion",
                "triggered": True,
                "reason": "explicit_cancel_request_with_no_draft_tool_call",
            },
            "response": {"content": "", "tool_calls": []},
            "generation_trace": [{
                "phase": "draft", "status": "completed",
                "discarded": False, "usage": usage1,
            }],
            "generation_attempts": 1,
            "generation_usage_total": usage1,
            "generation_usage_known": usage1,
            "decision_runtime_seconds": 1.0,
        }

        class Full:
            generation_calls = 1
            calls = 0
            def run(self, _payload):
                self.calls += 1
                self.generation_calls += 1
                usage = {"prompt_tokens": 2, "completion_tokens": 1,
                         "total_tokens": 3}
                return {
                    "response": {"content": "", "tool_calls": [{"function": {
                        "name": "cancel_booking",
                        "arguments": {"booking_id": "B-7"},
                    }}]},
                    "generation_trace": [{
                        "phase": "draft", "status": "completed",
                        "discarded": False, "usage": usage,
                    }],
                    "generation_usage_total": usage,
                    "generation_usage_known": usage,
                    "decision_runtime_seconds": 2.0,
                }

        runner = object.__new__(BCRSDecisionRunner)
        runner.generation_calls = 1
        full = Full()
        runner.runners = {"full": full}
        result = runner._apply_cancel_booking_full_diversion({}, record)
        self.assertEqual(full.calls, 1)
        self.assertEqual(result["generation_attempts"], 2)
        self.assertEqual(result["recovery_diversion"]["racer_recovery_count"], 0)
        self.assertEqual(result["recovery_diversion"]["regeneration_count"], 1)
        self.assertTrue(result["generation_trace"][0]["discarded"])
        self.assertEqual(result["generation_trace"][1]["phase"], "regeneration")
        self.assertEqual(result["final_action_ready_seconds"], 3.0)

    def test_adjacent_previous_turn_readonly_repeat_is_opt_in(self):
        payload = {
            "tools": [tool("get_available_stocks", [], description="Get stock symbols")],
            "messages": [
                {"role": "user", "content": "List Technology stocks."},
                {"role": "assistant", "tool_calls": [{"id": "a", "function": {
                    "name": "get_available_stocks",
                    "arguments": {"sector": "Technology"},
                }}]},
                {"role": "tool", "tool_call_id": "a", "content": "[\"NVDA\"]"},
                {"role": "user", "content": "Now analyze NVDA."},
            ],
        }
        draft = {"tool_calls": [{"id": "b", "function": {
            "name": "get_available_stocks",
            "arguments": '{"sector":"Technology"}',
        }}]}
        self.assertFalse(readonly_repeat_guard(payload, draft, [])['triggered'])
        receipt = readonly_repeat_guard(
            payload, draft, [], include_previous_user_turn=True,
        )
        self.assertTrue(receipt['triggered'])
        self.assertEqual(receipt['guard'], 'adjacent_turn_readonly_repeat')

    def test_adjacent_repeat_blocks_refresh_and_mutation(self):
        base = [
            {"role": "user", "content": "List Technology stocks."},
            {"role": "assistant", "tool_calls": [{"id": "a", "function": {
                "name": "get_available_stocks", "arguments": {"sector": "Technology"},
            }}]},
            {"role": "tool", "tool_call_id": "a", "content": "ok"},
        ]
        draft = {"tool_calls": [{"id": "b", "function": {
            "name": "get_available_stocks", "arguments": {"sector": "Technology"},
        }}]}
        refreshed = {"tools": [tool("get_available_stocks", [], description="Get stock symbols")],
                     "messages": base + [{"role": "user", "content": "Refresh the Technology list again."}]}
        self.assertFalse(readonly_repeat_guard(
            refreshed, draft, [], include_previous_user_turn=True,
        )['triggered'])
        mutated = {"tools": [tool("get_available_stocks", [], description="Get stock symbols"),
                              tool("place_order", ["symbol"])],
                   "messages": base + [
                       {"role": "assistant", "tool_calls": [{"id": "m", "function": {
                           "name": "place_order", "arguments": {"symbol": "NVDA"},
                       }}]},
                       {"role": "tool", "tool_call_id": "m", "content": "ok"},
                       {"role": "user", "content": "Analyze NVDA."},
                   ]}
        self.assertFalse(readonly_repeat_guard(
            mutated, draft, [], include_previous_user_turn=True,
        )['triggered'])


if __name__ == "__main__":
    unittest.main()
