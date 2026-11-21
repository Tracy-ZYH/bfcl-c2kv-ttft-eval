"""Ground-truth-free guards evaluated after a draft and before submission."""
from __future__ import annotations

import json
import re
from typing import Any


_EXACT_SLOT = re.compile(
    r"(?:^|_)(?:id|token|key|code|number|reference|ref)(?:$|_)", re.I,
)
_READ_PREFIXES = (
    "get_", "list_", "search_", "view_", "retrieve_", "filter_",
    "check_", "lookup_", "find_", "fetch_", "read_", "query_",
    "describe_", "inspect_", "show_",
)
_MUTATING_WORDS = {
    "add", "cancel", "create", "delete", "fund", "login", "logout",
    "notify", "place", "post", "remove", "send", "set", "update",
    "withdraw", "write",
}
_TOOL_CONCEPT_STOP = {"get", "api", "tool", "function"}
_EXPLICIT_QUOTE = re.compile(
    r"(?<![A-Za-z0-9])(?P<quote>['\"])(?P<text>[^'\"\n]+)(?P=quote)(?![A-Za-z0-9])"
)
_BOOKING_CANCEL_INTENT = re.compile(
    r"\b(?:cancel|cancellation|reverse|revoke|void)\b", re.I,
)
_BOOKING_CANCEL_OBJECT = re.compile(
    r"\b(?:booking|reservation|flight)\b", re.I,
)
_BOOKING_CANCEL_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|not)\s+(?:cancel|reverse|revoke|void)\b", re.I,
)
_READONLY_REFRESH_OR_SCOPE_CHANGE = re.compile(
    r"\b(?:refresh|again|once\s+more|re-?run|rerun|update|latest|new|different|another|change|switch|expand|narrow|include|exclude)\b",
    re.I,
)


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _call_parts(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    arguments = _json_value(function.get("arguments") or {})
    return name, arguments if isinstance(arguments, dict) else {}


def _canonical_call(call: dict[str, Any]) -> tuple[str, str]:
    name, arguments = _call_parts(call)
    return name, json.dumps(
        arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _tools_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            result[str(function["name"])] = function
    return result


def _current_user_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _walk_field(value: Any, field: str) -> list[Any]:
    value = _json_value(value)
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == field.lower() and not isinstance(item, (dict, list)):
                found.append(item)
            found.extend(_walk_field(item, field))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_field(item, field))
    return found


def _message_field_values(message: dict[str, Any], field: str) -> list[Any]:
    values = _walk_field(message.get("content"), field)
    for call in _tool_calls(message):
        values.extend(_walk_field(_call_parts(call)[1], field))
    if values:
        return values
    label = re.escape(field).replace(r"\_", r"[ _-]")
    pattern = re.compile(
        rf"(?i)(?:\"?{label}\"?)\s*[:=]\s*[\"']?([A-Za-z0-9_.:@/+\-]+)",
    )
    return pattern.findall(_content_text(message))


def _candidate_tools(
    draft: dict[str, Any], tools: dict[str, dict[str, Any]],
    current_text: str, *, step_zero: bool,
) -> list[tuple[str, dict[str, Any]]]:
    calls = _tool_calls(draft)
    if calls:
        return [_call_parts(call) for call in calls]
    if not step_zero:
        return []
    normalized = re.sub(r"[^a-z0-9]+", " ", current_text.lower())
    words = set(normalized.split())
    inferred = []
    for name in tools:
        concepts = [
            part for part in re.split(r"[_\W]+", name.lower())
            if part and part not in _TOOL_CONCEPT_STOP
        ]
        if concepts and all(concept in words for concept in concepts):
            inferred.append((name, {}))
    return inferred if len(inferred) == 1 else []


def exact_identifier_guard(
    payload: dict[str, Any], draft: dict[str, Any], raw_source_indices: list[int],
) -> dict[str, Any]:
    """Trigger when an exact schema identifier exists only outside raw evidence."""

    messages = list(payload.get("messages") or [])
    user_index = _current_user_index(messages)
    receipt = {
        "guard": "exact_identifier_raw_evidence",
        "triggered": False,
        "candidate_tools": [],
        "missing_evidence": [],
    }
    if user_index is None:
        receipt["reason"] = "no_current_user"
        return receipt
    tools = _tools_by_name(list(payload.get("tools") or []))
    step_zero = str(payload.get("decision_key") or "").endswith("/step-0")
    candidates = _candidate_tools(
        draft, tools, _content_text(messages[user_index]), step_zero=step_zero,
    )
    receipt["candidate_tools"] = [name for name, _ in candidates]
    raw = {int(index) for index in raw_source_indices or []}
    current_text = _content_text(messages[user_index]).lower()
    for name, draft_args in candidates:
        schema = ((tools.get(name) or {}).get("parameters") or {})
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or properties)
        for field in properties:
            if field not in required or not _EXACT_SLOT.search(field):
                continue
            proposed = draft_args.get(field)
            if proposed is not None and str(proposed).lower() in current_text:
                continue
            evidence = []
            for index, message in enumerate(messages[:user_index]):
                values = _message_field_values(message, field)
                if proposed is not None:
                    values = [value for value in values if str(value) == str(proposed)]
                if values:
                    evidence.append({"message_index": index, "values": values})
            if evidence and not any(item["message_index"] in raw for item in evidence):
                receipt["missing_evidence"].append({
                    "tool": name,
                    "field": field,
                    "proposed_value": proposed,
                    "evidence_message_indices": [
                        item["message_index"] for item in evidence
                    ],
                })
    receipt["triggered"] = bool(receipt["missing_evidence"])
    receipt["reason"] = (
        "exact_identifier_evidence_not_raw"
        if receipt["triggered"] else "no_unprotected_exact_identifier"
    )
    return receipt


def post_tweet_quoted_content_guard(
    payload: dict[str, Any], draft: dict[str, Any], _raw_source_indices: list[int],
) -> dict[str, Any]:
    """Preserve an explicitly quoted tweet body when only a final period differs."""

    messages = list(payload.get("messages") or [])
    user_index = _current_user_index(messages)
    receipt = {
        "guard": "post_tweet_quoted_content",
        "triggered": False,
        "regeneration_required": False,
    }
    if user_index is None:
        receipt["reason"] = "no_current_user"
        return receipt

    quoted = [
        match.group("text")
        for match in _EXPLICIT_QUOTE.finditer(_content_text(messages[user_index]))
    ]
    post_calls = []
    for index, call in enumerate(_tool_calls(draft)):
        name, arguments = _call_parts(call)
        if name == "post_tweet":
            post_calls.append((index, arguments))
    if len(post_calls) != 1:
        receipt["reason"] = "post_tweet_call_not_unique"
        receipt["post_tweet_call_count"] = len(post_calls)
        return receipt

    call_index, arguments = post_calls[0]
    content = arguments.get("content")
    if not isinstance(content, str):
        receipt["reason"] = "post_tweet_content_not_string"
        return receipt
    matches = [literal for literal in quoted if content == literal + "."]
    if len(matches) != 1:
        receipt["reason"] = "quoted_content_match_not_unique"
        receipt["quoted_match_count"] = len(matches)
        return receipt

    literal = matches[0]
    receipt.update({
        "triggered": True,
        "reason": "unique_quoted_content_differs_by_one_trailing_period",
        "tool_call_index": call_index,
        "before_content": content,
        "after_content": literal,
        "changed_fields": ["content"],
    })
    return receipt


def booking_cancel_full_diversion_guard(
    payload: dict[str, Any], draft: dict[str, Any], _raw_source_indices: list[int],
) -> dict[str, Any]:
    """Divert an unacted explicit booking cancellation to ordinary Full.

    This request-local check does not inspect a checker, expected action,
    task ID, or future tool result, and it never forces a tool choice.
    """

    messages = list(payload.get("messages") or [])
    user_index = _current_user_index(messages)
    receipt = {
        "guard": "booking_cancel_full_diversion",
        "triggered": False,
        "regeneration_required": True,
        "draft_tool_call_count": len(_tool_calls(draft)),
        "cancel_already_called_this_turn": False,
    }
    if user_index is None:
        receipt["reason"] = "no_current_user"
        return receipt
    current_text = _content_text(messages[user_index])
    explicit_request = bool(
        _BOOKING_CANCEL_INTENT.search(current_text)
        and _BOOKING_CANCEL_OBJECT.search(current_text)
        and not _BOOKING_CANCEL_NEGATION.search(current_text)
    )
    receipt["explicit_booking_cancel_request"] = explicit_request
    if not explicit_request:
        receipt["reason"] = "no_explicit_booking_cancel_request"
        return receipt
    if receipt["draft_tool_call_count"]:
        receipt["reason"] = "draft_already_has_tool_call"
        return receipt
    for message in messages[user_index + 1:]:
        for call in _tool_calls(message):
            name, _arguments = _call_parts(call)
            if name == "cancel_booking":
                receipt["cancel_already_called_this_turn"] = True
                receipt["reason"] = "cancel_booking_already_called_this_turn"
                return receipt
    receipt["triggered"] = True
    receipt["reason"] = "explicit_cancel_request_with_no_draft_tool_call"
    return receipt


def _is_read_only(name: str, tool: dict[str, Any] | None) -> bool:
    lower = name.lower()
    first = lower.split("_", 1)[0]
    if first in _MUTATING_WORDS:
        return False
    if lower.startswith(_READ_PREFIXES):
        return True
    description = str((tool or {}).get("description") or "").lower()
    return "get " in description or "list " in description or "search " in description


def readonly_repeat_guard(
    payload: dict[str, Any], draft: dict[str, Any], _raw_source_indices: list[int],
    *, include_previous_user_turn: bool = False,
) -> dict[str, Any]:
    """Detect a repeated read-only call with no intervening write.

    The historical same-turn behavior is unchanged. The optional cross-turn
    mode inspects only the immediately preceding completed user turn and is
    deliberately opt-in.
    """

    messages = list(payload.get("messages") or [])
    user_index = _current_user_index(messages)
    receipt = {
        "guard": (
            "adjacent_turn_readonly_repeat"
            if include_previous_user_turn else "same_turn_readonly_repeat"
        ),
        "triggered": False,
        "repeated_calls": [],
        "cross_turn_enabled": bool(include_previous_user_turn),
    }
    if user_index is None:
        receipt["reason"] = "no_current_user"
        return receipt
    current_text = _content_text(messages[user_index])
    history_start = user_index
    previous_user_index = None
    cross_turn_allowed = False
    if include_previous_user_turn:
        previous_user_index = next(
            (index for index in range(user_index - 1, -1, -1)
             if messages[index].get("role") == "user"),
            None,
        )
        if previous_user_index is None:
            receipt["cross_turn_reason"] = "no_adjacent_completed_user_turn"
        elif _READONLY_REFRESH_OR_SCOPE_CHANGE.search(current_text):
            receipt["cross_turn_reason"] = "current_user_requests_refresh_or_scope_change"
            receipt["scope_change_blocked"] = True
        else:
            cross_turn_allowed = True
            history_start = previous_user_index
    tools = _tools_by_name(list(payload.get("tools") or []))
    past: list[tuple[tuple[str, str], bool, bool]] = []
    if cross_turn_allowed:
        history_messages = (
            messages[history_start:user_index] + messages[user_index + 1:]
        )
    else:
        # Preserve the original same-turn behavior exactly.
        history_messages = messages[user_index + 1:]
    result_ids = {
        str(message.get("tool_call_id"))
        for message in history_messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    for message in history_messages:
        for call in _tool_calls(message):
            name, _ = _call_parts(call)
            past.append((
                _canonical_call(call),
                _is_read_only(name, tools.get(name)),
                str(call.get("id") or "") in result_ids,
            ))
    for call in _tool_calls(draft):
        name, arguments = _call_parts(call)
        signature = _canonical_call(call)
        if not _is_read_only(name, tools.get(name)):
            continue
        prior = [index for index, item in enumerate(past) if item[0] == signature]
        if not prior:
            continue
        previous = prior[-1]
        prior_result_visible = past[previous][2]
        no_intervening_write = all(item[1] for item in past[previous + 1:])
        if prior_result_visible and no_intervening_write:
            receipt["repeated_calls"].append({
                "tool": name,
                "arguments": arguments,
                "previous_call_ordinal": previous,
            })
    receipt["triggered"] = bool(receipt["repeated_calls"])
    receipt["reason"] = (
        "read_only_signature_already_observed_without_intervening_write"
        if receipt["triggered"] else "no_redundant_read_only_call"
    )
    if previous_user_index is not None:
        receipt["previous_user_turn_index"] = previous_user_index
    if include_previous_user_turn and cross_turn_allowed and not receipt["triggered"]:
        receipt["reason"] = "no_adjacent_completed_read_only_signature"
    return receipt

