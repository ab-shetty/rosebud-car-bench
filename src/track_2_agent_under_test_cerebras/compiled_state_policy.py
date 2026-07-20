"""Deterministic compiled-state brief and conservative catalog affordances."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

if __package__:
    from .reformulation_ledger import history_token_count, parse_state_ledger
else:  # pragma: no cover - loose-script compatibility
    from reformulation_ledger import history_token_count, parse_state_ledger


@dataclass(frozen=True)
class CompiledBrief:
    messages: list[dict[str, Any]]
    ledger: dict[str, Any]
    raw_history_tokens: int
    brief_tokens: int
    prior_request_count: int
    fresh_tool_result_count: int


@dataclass(frozen=True)
class WithheldTool:
    name: str
    family: str
    resource: str
    contradiction: str


_CREATION_ACTIONS = frozenset({"create", "start", "activate"})
_EDIT_ACTIONS = frozenset({"edit", "update", "replace", "delete", "remove", "add"})
_RESOURCE_QUALIFIERS = frozenset(
    {
        "current",
        "new",
        "one",
        "final",
        "the",
        "a",
        "an",
        "to",
        "from",
        "by",
    }
)
_ACTIVE_FIELDS = frozenset({"active", "exists", "present", "available"})


def compile_state_brief(messages: list[dict[str, Any]]) -> CompiledBrief:
    """Compile raw history into the exact deterministic CSP decision surface."""

    latest_user = _latest_user_index(messages)
    if latest_user is None:
        ledger = parse_state_ledger(messages)
        live_user = ""
        prior_requests: list[str] = []
        fresh_results: list[dict[str, Any]] = []
    else:
        ledger = parse_state_ledger(messages, through_index=latest_user)
        live_user = str(messages[latest_user].get("content") or "")
        prior_requests = [
            str(message.get("content") or "")
            for message in messages[:latest_user]
            if message.get("role") == "user"
        ]
        fresh_results = [
            dict(message)
            for message in messages[latest_user + 1 :]
            if message.get("role") == "tool"
        ]

    last_assistant = next(
        (
            _assistant_message_for_brief(message)
            for message in reversed(messages)
            if message.get("role") == "assistant"
        ),
        None,
    )
    pending = bool(
        (ledger.get("open_items") or {}).get("pending_confirmation")
        if isinstance(ledger, dict)
        else False
    )
    header = "\n".join(
        (
            "[requests]",
            json.dumps(prior_requests, ensure_ascii=False, separators=(",", ":")),
            "[your last reply]",
            json.dumps(
                last_assistant,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "[state ledger]",
            json.dumps(
                ledger,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "[open]",
            json.dumps(
                {"pending_confirmation": pending},
                separators=(",", ":"),
            ),
        )
    )
    system_messages = [
        dict(message) for message in messages if message.get("role") == "system"
    ]
    compiled_messages = [
        *system_messages,
        {"role": "user", "content": header},
        *fresh_results,
        {"role": "user", "content": live_user},
    ]
    return CompiledBrief(
        messages=compiled_messages,
        ledger=ledger,
        raw_history_tokens=history_token_count(messages),
        brief_tokens=history_token_count(compiled_messages),
        prior_request_count=len(prior_requests),
        fresh_tool_result_count=len(fresh_results),
    )


def prior_state_ledger(messages: list[dict[str, Any]]) -> dict[str, Any]:
    latest_user = _latest_user_index(messages)
    return parse_state_ledger(
        messages,
        through_index=latest_user if latest_user is not None else len(messages),
    )


def gate_catalog_by_affordance(
    tools: list[dict[str, Any]], ledger: dict[str, Any]
) -> tuple[list[dict[str, Any]], tuple[WithheldTool, ...]]:
    """Withhold only catalog actions contradicted by explicit ledger evidence."""

    kept: list[dict[str, Any]] = []
    withheld: list[WithheldTool] = []
    for tool in tools:
        name = str(tool.get("function", {}).get("name") or "")
        semantics = catalog_action_semantics(name)
        if semantics is None:
            kept.append(tool)
            continue
        family, resource_tokens = semantics
        status = resource_status(ledger, resource_tokens)
        contradiction = None
        if family == "creation" and status is True:
            contradiction = "resource_active"
        elif family == "edit_delete" and status is False:
            contradiction = "resource_absent"
        if contradiction is None:
            kept.append(tool)
            continue
        withheld.append(
            WithheldTool(
                name=name,
                family=family,
                resource="_".join(sorted(resource_tokens)),
                contradiction=contradiction,
            )
        )
    return kept, tuple(withheld)


def catalog_action_semantics(
    name: str,
) -> tuple[str, frozenset[str]] | None:
    """Derive action family and resource tokens from a catalog function name."""

    tokens = re.findall(r"[a-z0-9]+", name.casefold().replace("_", " "))
    if len(tokens) >= 3 and tokens[:2] == ["set", "new"]:
        resource = _resource_tokens(tokens[2:])
        return ("creation", resource) if resource else None

    action_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token in _CREATION_ACTIONS or token in _EDIT_ACTIONS
        ),
        None,
    )
    if action_index is None:
        return None
    action = tokens[action_index]
    family = "creation" if action in _CREATION_ACTIONS else "edit_delete"
    resource_source = (
        tokens[:action_index] if action_index > 0 else tokens[action_index + 1 :]
    )
    resource = _resource_tokens(resource_source)
    return (family, resource) if resource else None


def resource_status(
    ledger: dict[str, Any], resource_tokens: frozenset[str]
) -> bool | None:
    """Return active/absent only when the ledger carries hard resource evidence."""

    evidence: set[bool] = set()
    for path, value in _flatten(ledger):
        path_tokens = set(
            re.findall(r"[a-z0-9]+", " ".join(path).casefold().replace("_", " "))
        )
        if not resource_tokens <= path_tokens:
            continue
        leaf_tokens = set(
            re.findall(r"[a-z0-9]+", path[-1].casefold().replace("_", " "))
        )
        if leaf_tokens & _ACTIVE_FIELDS and isinstance(value, bool):
            evidence.add(value)
        elif "id" in leaf_tokens:
            if value is None or value == "" or value == []:
                evidence.add(False)
            elif isinstance(value, (str, int, list)):
                evidence.add(True)
    return next(iter(evidence)) if len(evidence) == 1 else None


def requested_withheld_tool(
    action: dict[str, Any], withheld: tuple[WithheldTool, ...]
) -> str | None:
    names = {item.name for item in withheld}
    return next(
        (
            str(call.get("tool_name") or "")
            for call in action.get("tool_calls") or []
            if str(call.get("tool_name") or "") in names
        ),
        None,
    )


def _assistant_message_for_brief(message: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {"content": message.get("content")}
    if message.get("tool_calls") is not None:
        rendered["tool_calls"] = message.get("tool_calls")
    return rendered


def _resource_tokens(tokens: list[str]) -> frozenset[str]:
    return frozenset(token for token in tokens if token not in _RESOURCE_QUALIFIERS)


def _latest_user_index(messages: list[dict[str, Any]]) -> int | None:
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )


def _flatten(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        rows: list[tuple[tuple[str, ...], Any]] = []
        for key, item in value.items():
            rows.extend(_flatten(item, (*path, str(key))))
        return rows
    return [(path or ("value",), value)]
