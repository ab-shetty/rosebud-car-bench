"""Deterministic p3i32 input reformulation and state-ledger transforms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

if __package__:
    from .policy_rag import (
        POLICY_SUBJECT_TOOL_FAMILIES,
        PolicyCorpus,
        PolicySection,
        build_policy_corpus,
        extract_policy_subjects,
        policy_query_keys,
        policy_token_count,
        retrieve_policy_sections,
    )
else:  # pragma: no cover - loose-script compatibility
    from policy_rag import (
        POLICY_SUBJECT_TOOL_FAMILIES,
        PolicyCorpus,
        PolicySection,
        build_policy_corpus,
        extract_policy_subjects,
        policy_query_keys,
        policy_token_count,
        retrieve_policy_sections,
    )


REFORMULATION_TOKEN_CAP = 170
REFORMULATION_RULE_LIMIT = 2
REFORMULATION_TOOL_LIMIT = 3
LEDGER_TOOL_NAME = "state_ledger"


@dataclass(frozen=True)
class ReformulationBlock:
    text: str
    token_count: int
    rule_ids: tuple[str, ...]
    tool_names: tuple[str, ...]


@dataclass(frozen=True)
class LedgerView:
    messages: list[dict[str, Any]]
    content: str
    token_count: int


def build_reformulation_block(
    *,
    user_text: str,
    system_text: str,
    tools: list[dict[str, Any]],
    token_cap: int = REFORMULATION_TOKEN_CAP,
) -> ReformulationBlock:
    """Build a stable whole-item-capped advisory from the supplied policy/catalog."""

    subjects = extract_policy_subjects(user_text)
    ranked_tools = _rank_tools(user_text, subjects, tools)
    tool_names = tuple(name for name, _ in ranked_tools)
    corpus = build_policy_corpus(
        system_text,
        tool_names=(
            str(tool.get("function", {}).get("name") or "") for tool in tools
        ),
    )
    # Census-core sections remain in the candidate corpus, but are not forced into
    # every turn. This makes them eligible under the same top-2 retrieval ranking.
    ranked_corpus = PolicyCorpus(
        sections=corpus.sections,
        always_on_ids=(),
        split_parent_count=corpus.split_parent_count,
        split_child_count=corpus.split_child_count,
        parent_key_document_frequency=corpus.parent_key_document_frequency,
        core_token_count=corpus.core_token_count,
        rejected_core_ids=corpus.rejected_core_ids,
    )
    rules = retrieve_policy_sections(
        ranked_corpus,
        policy_query_keys(tool_names=tool_names, subjects=subjects),
        top_k=REFORMULATION_RULE_LIMIT,
        query_text=user_text,
    ).matched_sections
    rule_items = [(_one_line(section.text), section.section_id) for section in rules]
    tool_items = [(signature, name) for name, signature in ranked_tools]

    included_rules: list[tuple[str, str]] = []
    included_tools: list[tuple[str, str]] = []
    for item in rule_items:
        candidate = _format_block([*included_rules, item], included_tools)
        if policy_token_count(candidate) <= token_cap:
            included_rules.append(item)
    for item in tool_items:
        candidate = _format_block(included_rules, [*included_tools, item])
        if policy_token_count(candidate) <= token_cap:
            included_tools.append(item)

    text = _format_block(included_rules, included_tools)
    count = policy_token_count(text)
    if count > token_cap:  # Defensive for impractically tiny test caps.
        text = _format_block([], [])
        count = policy_token_count(text)
    return ReformulationBlock(
        text=text,
        token_count=count,
        rule_ids=tuple(section_id for _, section_id in included_rules),
        tool_names=tuple(name for _, name in included_tools),
    )


def append_reformulation_block(
    messages: list[dict[str, Any]], block: ReformulationBlock
) -> list[dict[str, Any]]:
    """Append guidance to the latest user message without mutating history."""

    latest_user = _latest_user_index(messages)
    if latest_user is None:
        return list(messages)
    transformed = list(messages)
    message = dict(messages[latest_user])
    content = str(message.get("content") or "")
    message["content"] = f"{content}\n\n{block.text}" if content else block.text
    transformed[latest_user] = message
    return transformed


def reconstruct_state_ledger(messages: list[dict[str, Any]]) -> LedgerView:
    """Fold prior-turn tool results into one deterministic tool-role ledger."""

    latest_user = _latest_user_index(messages)
    if latest_user is None:
        return LedgerView(list(messages), "", 0)
    prior_tool_indexes = [
        index
        for index, message in enumerate(messages[:latest_user])
        if message.get("role") == "tool"
    ]
    if not prior_tool_indexes:
        return LedgerView(list(messages), "", 0)

    ledger = parse_state_ledger(messages, through_index=latest_user)
    content = "[state ledger] " + json.dumps(
        ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    prior_tool_set = set(prior_tool_indexes)
    transformed = [
        message for index, message in enumerate(messages) if index not in prior_tool_set
    ]
    new_user_index = next(
        index
        for index, message in enumerate(transformed)
        if message is messages[latest_user]
    )
    transformed.insert(
        new_user_index,
        {
            "role": "tool",
            "name": LEDGER_TOOL_NAME,
            "tool_call_id": LEDGER_TOOL_NAME,
            "content": content,
        },
    )
    return LedgerView(transformed, content, policy_token_count(content))


def parse_state_ledger(
    messages: list[dict[str, Any]], *, through_index: int | None = None
) -> dict[str, Any]:
    """Parse tool results generically; unknown result fields are retained."""

    limit = len(messages) if through_index is None else through_index
    call_by_id, calls_in_order = _tool_calls(messages[:limit])
    state: dict[str, Any] = {}
    navigation: dict[str, Any] = {}
    mutations: list[dict[str, Any]] = []
    call_cursor = 0
    for message in messages[:limit]:
        if message.get("role") != "tool":
            continue
        tool_id = str(message.get("tool_call_id") or "")
        call = call_by_id.get(tool_id)
        if call is None and call_cursor < len(calls_in_order):
            call = calls_in_order[call_cursor]
        call_cursor += 1
        name = str(
            message.get("name")
            or (call or {}).get("name")
            or f"tool_result_{call_cursor}"
        )
        decoded = _decode_content(message.get("content"))
        payload = (
            decoded.get("result")
            if isinstance(decoded, dict) and "result" in decoded
            else decoded
        )
        if isinstance(payload, dict):
            for key, value in payload.items():
                state[str(key)] = value
            _update_navigation(navigation, payload)
        else:
            state[name] = payload
        _update_navigation(navigation, decoded)
        if _is_mutation(name):
            arguments = (call or {}).get("arguments", {})
            _update_navigation(navigation, arguments)
            mutations.append(
                {
                    "tool": name,
                    "args": arguments,
                    "status": _result_status(decoded),
                }
            )

    pending = _pending_confirmation(messages[:limit])
    return {
        "state": state,
        "navigation": navigation,
        "mutations": mutations,
        "open_items": {"pending_confirmation": pending},
    }


def history_token_count(messages: list[dict[str, Any]]) -> int:
    return policy_token_count(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _rank_tools(
    user_text: str,
    subjects: Iterable[str],
    tools: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    sections: list[PolicySection] = []
    for order, tool in enumerate(
        sorted(tools, key=lambda item: str(item.get("function", {}).get("name") or ""))
    ):
        function = tool.get("function", {})
        name = str(function.get("name") or "")
        if not name:
            continue
        signature = _tool_signature(tool)
        metadata = " ".join(
            (
                name,
                str(function.get("description") or ""),
                json.dumps(function.get("parameters") or {}),
            )
        )
        tool_subjects = set(extract_policy_subjects(metadata))
        lowered = name.casefold()
        for subject, fragments in POLICY_SUBJECT_TOOL_FAMILIES.items():
            if any(fragment in lowered for fragment in fragments):
                tool_subjects.add(subject)
        keys = frozenset(
            {f"subject:{subject}" for subject in tool_subjects}
            | {f"tool:{name.casefold()}"}
        )
        sections.append(
            PolicySection(
                section_id=f"catalog-{name}",
                text=signature,
                order=order,
                keys=keys,
            )
        )
    corpus = PolicyCorpus(tuple(sections), ())
    explicitly_named = [
        section.text.split("(", 1)[0]
        for section in sections
        if re.search(
            rf"(?<![a-z0-9_])"
            rf"{re.escape(section.text.split('(', 1)[0].casefold())}"
            rf"(?![a-z0-9_])",
            user_text.casefold(),
        )
    ]
    retrieval = retrieve_policy_sections(
        corpus,
        policy_query_keys(tool_names=explicitly_named, subjects=subjects),
        top_k=REFORMULATION_TOOL_LIMIT,
        query_text=user_text,
    )
    return [
        (section.section_id.removeprefix("catalog-"), section.text)
        for section in retrieval.matched_sections
    ]


def _tool_signature(tool: dict[str, Any]) -> str:
    function = tool.get("function", {})
    name = str(function.get("name") or "")
    schema = function.get("parameters") or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    arguments: list[str] = []
    if isinstance(properties, dict):
        for argument, details in properties.items():
            kind = str(details.get("type") or "any") if isinstance(details, dict) else "any"
            optional = "" if argument in required else "?"
            arguments.append(f"{argument}{optional}:{kind}")
    return f"{name}({', '.join(arguments)})"


def _format_block(
    rules: list[tuple[str, str]], tools: list[tuple[str, str]]
) -> str:
    rule_text = " | ".join(text for text, _ in rules) or "none"
    tool_text = " | ".join(text for text, _ in tools) or "none"
    return f"[assistant guidance — policy: {rule_text}; likely tools: {tool_text}]"


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _latest_user_index(messages: list[dict[str, Any]]) -> int | None:
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )


def _decode_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def _tool_calls(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            name = str(
                function.get("name")
                or raw.get("tool_name")
                or raw.get("name")
                or ""
            )
            arguments = function.get("arguments", raw.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            call = {"name": name, "arguments": arguments or {}}
            ordered.append(call)
            call_id = str(raw.get("id") or raw.get("tool_call_id") or "")
            if call_id:
                by_id[call_id] = call
    return by_id, ordered


def _update_navigation(navigation: dict[str, Any], value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if normalized in {"route_id", "route_ids", "routes_to_final_destination_id"}:
                navigation["route_id"] = item
            elif normalized in {
                "origin",
                "origin_id",
                "start",
                "start_id",
                "start_location",
            }:
                navigation["origin"] = item
            elif normalized in {
                "destination",
                "destination_id",
                "final_destination",
                "final_destination_id",
            }:
                navigation["destination"] = item
            elif normalized in {"waypoint", "waypoints", "waypoint_ids", "waypoints_id"}:
                navigation["waypoints"] = item
            elif normalized in {"navigation_active", "active_navigation", "route_active"}:
                navigation["active"] = item
            _update_navigation(navigation, item)
    elif isinstance(value, list):
        for item in value:
            _update_navigation(navigation, item)


def _is_mutation(name: str) -> bool:
    lowered = name.casefold()
    return lowered.startswith(
        (
            "set_",
            "open_close_",
            "activate_",
            "deactivate_",
            "delete_",
            "navigation_",
            "send_",
            "call_",
            "create_",
            "update_",
            "replace_",
            "add_",
            "remove_",
        )
    )


def _result_status(value: Any) -> str:
    if isinstance(value, dict):
        status = str(value.get("status") or "").casefold()
        if status in {"failure", "error", "failed"} or value.get("errors"):
            return "error"
    return "ok"


def _pending_confirmation(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content if content.endswith("?") else None
    return None
