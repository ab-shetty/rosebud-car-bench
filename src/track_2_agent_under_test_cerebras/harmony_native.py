"""Native openai-harmony transport for adaptive-minimal Cerebras execution."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from openai_harmony import (  # type: ignore[import-untyped]
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    ToolNamespaceConfig,
    load_harmony_encoding,
)

if __package__:
    from . import cerebras_client as cb
else:  # pragma: no cover - loose-script compatibility
    import cerebras_client as cb


TRANSPORT_NAME = "harmony_native"
DEFAULT_PARSE_RETRIES = 1
STOP_SEQUENCES = ["<|return|>", "<|call|>"]
FUNCTION_NAMESPACE = "functions"
SYSTEM_TOOL_PLACEMENT = "system"
DEVELOPER_TOOL_PLACEMENT = "developer"
TOOL_PLACEMENTS = {SYSTEM_TOOL_PLACEMENT, DEVELOPER_TOOL_PLACEMENT}

MODEL_IDENTITY = (
    "You are the CAR-bench in-car voice assistant. Use only formally defined "
    "functions. Put private reasoning in analysis, function calls in commentary, "
    "and concise user-facing replies in final."
)


@dataclass(frozen=True)
class HarmonyActionCall:
    """Parsed adaptive action plus convention-compatible provider accounting."""

    action: dict[str, Any]
    completion: cb.CompletionCallResult
    call_count: int
    parse_failures: int
    analysis_text: str
    rendered_input_tokens: int


class HarmonyNativeParseError(cb.MalformedModelResponseError):
    """Bounded Harmony parse failure carrying all successfully billed calls."""

    def __init__(
        self,
        message: str,
        *,
        completion: cb.CompletionCallResult | None,
        call_count: int,
        parse_failures: int,
    ) -> None:
        super().__init__(message)
        self.completion = completion
        self.call_count = call_count
        self.parse_failures = parse_failures
        self.parse_failures_counted = True


class HarmonyNativeClient(cb.CerebrasCompletionClient):
    """Cerebras ``/completions`` client with strict Harmony render and parse."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.parse_retries = max(
            0,
            int(os.getenv("TRACK2_HARMONY_PARSE_RETRIES", DEFAULT_PARSE_RETRIES)),
        )
        placement = (
            os.getenv("TRACK2_HARMONY_TOOL_PLACEMENT", SYSTEM_TOOL_PLACEMENT)
            .strip()
            .casefold()
        )
        if placement not in TOOL_PLACEMENTS:
            raise ValueError(
                "TRACK2_HARMONY_TOOL_PLACEMENT must be system or developer"
            )
        self.tool_placement = placement

    def generate_action(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        developer_instructions: str,
        fault_text: str | None,
        max_completion_tokens: int,
        temperature: float | None,
        reasoning_effort: str,
        analysis_to_replay: str | None,
    ) -> HarmonyActionCall:
        with self._request_lock:
            return self._generate_action_locked(
                model=model,
                messages=messages,
                tools=tools,
                developer_instructions=developer_instructions,
                fault_text=fault_text,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                analysis_to_replay=analysis_to_replay,
            )

    def _generate_action_locked(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        developer_instructions: str,
        fault_text: str | None,
        max_completion_tokens: int,
        temperature: float | None,
        reasoning_effort: str,
        analysis_to_replay: str | None,
    ) -> HarmonyActionCall:
        conversation = build_harmony_conversation(
            messages=messages,
            tools=tools,
            developer_instructions=developer_instructions,
            fault_text=fault_text,
            reasoning_effort=reasoning_effort,
            analysis_to_replay=analysis_to_replay,
            tool_placement=self.tool_placement,
        )
        prompt_tokens = self.encoding.render_conversation_for_completion(
            conversation,
            Role.ASSISTANT,
        )
        prompt_text = self.encoding.decode(prompt_tokens)

        total: cb.CompletionCallResult | None = None
        parse_failures = 0
        call_count = 0
        last_error: Exception | None = None
        for attempt in range(self.parse_retries + 1):
            call = self._generate_harmony_locked(
                model=model,
                prompt_text=prompt_text,
                prompt_token_count=len(prompt_tokens),
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            total = _add_completion_results(total, call)
            call_count += 1
            try:
                action, analysis_text = parse_harmony_action(
                    self.encoding,
                    call.text,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                parse_failures += 1
                last_error = exc
                if self.logger:
                    self.logger.warning(
                        "Native Harmony parse failure",
                        attempt=attempt + 1,
                        retrying=attempt < self.parse_retries,
                        error=str(exc),
                    )
                continue
            return HarmonyActionCall(
                action=action,
                completion=total,
                call_count=call_count,
                parse_failures=parse_failures,
                analysis_text=analysis_text,
                rendered_input_tokens=len(prompt_tokens),
            )

        raise HarmonyNativeParseError(
            f"Native Harmony completion did not parse after {call_count} call(s): "
            f"{last_error}",
            completion=total,
            call_count=call_count,
            parse_failures=parse_failures,
        )

    def _generate_harmony_locked(
        self,
        *,
        model: str,
        prompt_text: str,
        prompt_token_count: int,
        max_completion_tokens: int,
        temperature: float | None,
        reasoning_effort: str,
    ) -> cb.CompletionCallResult:
        normalized_model = cb.normalize_cerebras_model(model)
        estimated_tokens = math.ceil(
            (prompt_token_count + max(0, max_completion_tokens))
            * cb.DEFAULT_TOKEN_SAFETY_FACTOR
        )
        quota_wait_ms = 0.0
        rate_limit_retries = 0
        queue_retries = 0
        edge_retries = 0
        report_messages = [{"role": "harmony_prompt", "content": prompt_text}]

        while True:
            previous_request_state = self._record_attempt(
                model=normalized_model,
                estimated_tokens=estimated_tokens,
            )
            kwargs: dict[str, Any] = {
                "model": normalized_model,
                "prompt": prompt_text,
                "max_tokens": max_completion_tokens,
                "stop": STOP_SEQUENCES,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if self.logger:
                self.logger.info(
                    "Sending Cerebras request",
                    model=normalized_model,
                    transport=TRANSPORT_NAME,
                    service_tier=None,
                    reasoning_effort=reasoning_effort,
                    estimated_request_tokens=estimated_tokens,
                    rendered_prompt_tokens=prompt_token_count,
                    previous_estimated_request_tokens=previous_request_state[
                        "previous_estimated_request_tokens"
                    ],
                    estimated_request_token_delta_since_previous=(
                        previous_request_state[
                            "estimated_request_token_delta_since_previous"
                        ]
                    ),
                    previous_successful_token_usage=previous_request_state[
                        "previous_successful_token_usage"
                    ],
                    previous_rate_limit_headers=previous_request_state[
                        "previous_rate_limit_headers"
                    ],
                    max_completion_tokens=max_completion_tokens,
                    has_output_schema=False,
                    tool_placement=self.tool_placement,
                )
            start = time.perf_counter()
            try:
                raw_response = self._client.completions.with_raw_response.create(
                    **kwargs
                )
                completion = raw_response.parse()
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                details, signal, report_path = self._handle_completion_error(
                    exc=exc,
                    model=normalized_model,
                    messages=report_messages,
                    response_schema=None,
                    response_schema_name=None,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                    estimated_tokens=estimated_tokens,
                    duration_ms=duration_ms,
                    queue_retry_attempt=queue_retries + 1,
                )
                self._log_cerebras_error(
                    exc,
                    details=details,
                    rate_limit_signal=signal,
                    report_path=report_path,
                )
                if (
                    signal is not None
                    and signal.schedule_wait_seconds is not None
                    and signal.schedule_wait_seconds > 0
                ):
                    wait_seconds = signal.schedule_wait_seconds
                    rate_limit_retries += 1
                    if cb._is_queue_rate_limit_signal(signal):
                        queue_retries += 1
                    if signal.quota_wait_eligible:
                        quota_wait_ms += wait_seconds * 1000.0
                    if self.logger:
                        self.logger.warning(
                            "Native Harmony rate-limit retry",
                            model=normalized_model,
                            wait_seconds=round(wait_seconds, 3),
                            retry_count=rate_limit_retries,
                            quota_wait_eligible=signal.quota_wait_eligible,
                            report_path=str(report_path) if report_path else None,
                        )
                    time.sleep(wait_seconds)
                    continue
                edge_wait = cb._edge_retry_wait_seconds(
                    details=details,
                    attempt=edge_retries + 1,
                    max_attempts=self.edge_retry_attempts,
                    initial_seconds=self.edge_retry_initial_seconds or 0.0,
                    cap_seconds=self.edge_retry_cap_seconds or 0.0,
                )
                if edge_wait is not None:
                    edge_retries += 1
                    if self.logger:
                        self.logger.warning(
                            "Native Harmony transport retry",
                            model=normalized_model,
                            wait_seconds=round(edge_wait, 3),
                            retry_count=edge_retries,
                            max_attempts=self.edge_retry_attempts,
                        )
                    time.sleep(edge_wait)
                    continue
                raise cb.CerebrasTemplateError(
                    f"Cerebras native Harmony completion failed for "
                    f"{normalized_model}: {exc}"
                ) from exc

            duration_ms = (time.perf_counter() - start) * 1000.0
            choice = completion.choices[0]
            text = getattr(choice, "text", None)
            if not isinstance(text, str):
                text = ""
            finish_reason = getattr(choice, "finish_reason", None)
            usage = cb.TokenUsage.from_provider_usage(
                getattr(completion, "usage", None)
            )
            # Native Harmony includes analysis in completion_tokens and exposes
            # no separate reasoning subtotal. Keeping reasoning at zero prevents
            # billed-token double counting in the existing audit convention.
            if usage is not None:
                usage.reasoning_output_tokens = 0
            headers = cb.CerebrasRateLimitHeaders.from_headers(
                getattr(raw_response, "headers", None)
            )
            self._record_successful_call(
                model=normalized_model,
                token_usage=usage,
                rate_limit_headers=headers,
            )
            if self.logger:
                self.logger.info(
                    "Cerebras response received",
                    model=getattr(completion, "model", None) or normalized_model,
                    transport=TRANSPORT_NAME,
                    reasoning_effort=reasoning_effort,
                    finish_reason=finish_reason,
                    duration_ms=round(duration_ms, 1),
                    estimated_request_tokens=estimated_tokens,
                    rendered_prompt_tokens=prompt_token_count,
                    token_usage=cb._token_usage_to_dict(usage),
                    time_info=cb._time_info_to_dict(
                        getattr(completion, "time_info", None)
                    ),
                    rate_limit_headers=(headers.as_dict() if headers else None),
                    quota_wait_ms=round(quota_wait_ms, 1),
                    cf_edge_retries=edge_retries,
                )
            return cb.CompletionCallResult(
                text=text,
                duration_ms=duration_ms,
                model=getattr(completion, "model", None) or normalized_model,
                finish_reason=finish_reason,
                token_usage=usage,
                cost=0.0,
                estimated_request_tokens=estimated_tokens,
                rate_limit_headers=headers,
                quota_wait_ms=quota_wait_ms,
            )


def build_harmony_conversation(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    developer_instructions: str,
    fault_text: str | None,
    reasoning_effort: str,
    analysis_to_replay: str | None,
    tool_placement: str = SYSTEM_TOOL_PLACEMENT,
) -> Conversation:
    """Render one formal tool namespace and a reasoning-hygienic history."""

    namespace = _tool_namespace(tools)
    system = (
        SystemContent.new()
        .with_model_identity(MODEL_IDENTITY)
        .with_reasoning_effort(_reasoning_effort(reasoning_effort))
    )
    developer_text = developer_instructions
    system_context = "\n\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system" and message.get("content")
    )
    if system_context:
        developer_text += "\n\nCAR-BENCH SYSTEM CONTEXT:\n" + system_context
    if fault_text:
        developer_text += "\n\nCONCRETE INTERNAL FAULT:\n" + fault_text
    developer = DeveloperContent.new().with_instructions(developer_text)
    if tool_placement == SYSTEM_TOOL_PLACEMENT:
        system.with_tools(namespace)
    elif tool_placement == DEVELOPER_TOOL_PLACEMENT:
        developer.with_tools(namespace)
    else:
        raise ValueError(f"unknown Harmony tool placement: {tool_placement!r}")

    rendered = [
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.DEVELOPER, developer),
    ]
    last_tool_call_index = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("tool_calls")
        ),
        default=-1,
    )
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            rendered.append(
                Message.from_role_and_content(
                    Role.USER,
                    _content_text(message.get("content")),
                )
            )
            continue
        if role == "assistant" and message.get("tool_calls"):
            if index == last_tool_call_index and analysis_to_replay:
                rendered.append(
                    Message.from_role_and_content(
                        Role.ASSISTANT,
                        analysis_to_replay,
                    ).with_channel("analysis")
                )
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        arguments or {}, ensure_ascii=False, separators=(",", ":")
                    )
                rendered.append(
                    Message.from_role_and_content(Role.ASSISTANT, arguments)
                    .with_channel("commentary")
                    .with_recipient(f"{FUNCTION_NAMESPACE}.{name}")
                    .with_content_type("<|constrain|>json")
                )
            continue
        if role == "assistant":
            rendered.append(
                Message.from_role_and_content(
                    Role.ASSISTANT,
                    _content_text(message.get("content")),
                ).with_channel("final")
            )
            continue
        if role == "tool":
            name = str(message.get("name") or "unknown")
            rendered.append(
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"{FUNCTION_NAMESPACE}.{name}"),
                    _content_text(message.get("content")),
                ).with_channel("commentary")
            )
    return Conversation.from_messages(rendered)


def parse_harmony_action(
    encoding: HarmonyEncoding,
    raw_text: str,
) -> tuple[dict[str, Any], str]:
    if not raw_text or not raw_text.strip():
        raise ValueError("empty native Harmony completion")
    tokens = encoding.encode(raw_text, allowed_special="all")
    try:
        messages = encoding.parse_messages_from_completion_tokens(
            tokens,
            role=Role.ASSISTANT,
            strict=True,
        )
    except Exception as exc:
        raise ValueError(f"strict Harmony parse failed: {exc}") from exc
    if not messages:
        raise ValueError("Harmony parser returned no assistant messages")
    analysis_text = "\n".join(
        _message_text(message)
        for message in messages
        if message.channel == "analysis"
    ).strip()
    calls = [message for message in messages if message.recipient is not None]
    if calls:
        normalized = []
        for message in calls:
            recipient = str(message.recipient or "")
            prefix = f"{FUNCTION_NAMESPACE}."
            name = recipient[len(prefix) :] if recipient.startswith(prefix) else recipient
            arguments = json.loads(_message_text(message))
            if not isinstance(arguments, dict):
                raise ValueError("Harmony tool-call arguments must be a JSON object")
            normalized.append({"tool_name": name, "arguments": arguments})
        return {"action": "tool_calls", "tool_calls": normalized}, analysis_text

    finals = [
        _message_text(message)
        for message in messages
        if message.channel == "final" and message.recipient is None
    ]
    content = "\n".join(item for item in finals if item).strip()
    if not content:
        raise ValueError("Harmony completion contained neither a tool call nor final text")
    return {"action": "respond", "content": content}, analysis_text


def _tool_namespace(tools: list[dict[str, Any]]) -> ToolNamespaceConfig:
    descriptions = []
    for tool in tools:
        function = tool.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        parameters = function.get("parameters") or {"type": "object"}
        descriptions.append(
            ToolDescription.new(
                name,
                str(function.get("description") or ""),
                None if _is_zero_argument_schema(parameters) else parameters,
            )
        )
    return ToolNamespaceConfig(
        name=FUNCTION_NAMESPACE,
        description="CAR-bench environment functions available for this episode.",
        tools=descriptions,
    )


def _is_zero_argument_schema(schema: dict[str, Any]) -> bool:
    """Return whether a schema is a closed object with no possible arguments."""

    return bool(
        schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and not schema.get("properties")
        and not schema.get("required")
        and not schema.get("patternProperties")
        and not schema.get("allOf")
        and not schema.get("anyOf")
        and not schema.get("oneOf")
    )


def _reasoning_effort(value: str) -> ReasoningEffort:
    return {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }.get(value.casefold(), ReasoningEffort.MEDIUM)


def _message_text(message: Message) -> str:
    return "".join(getattr(content, "text", "") for content in message.content)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _add_completion_results(
    left: cb.CompletionCallResult | None,
    right: cb.CompletionCallResult,
) -> cb.CompletionCallResult:
    if left is None:
        return right
    return cb.CompletionCallResult(
        text=right.text,
        duration_ms=left.duration_ms + right.duration_ms,
        model=right.model,
        finish_reason=right.finish_reason,
        token_usage=cb.add_token_usage(left.token_usage, right.token_usage),
        cost=left.cost + right.cost,
        estimated_request_tokens=(
            left.estimated_request_tokens + right.estimated_request_tokens
        ),
        rate_limit_headers=right.rate_limit_headers,
        quota_wait_ms=left.quota_wait_ms + right.quota_wait_ms,
    )
