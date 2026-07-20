"""Low-latency adaptive harness for the Track 2 Cerebras agent.

The normal path is one schema-constrained executor call.  The only additional
model call is a single correction after a hard internal fault.  Read-before-set
grounding and fastest-route disclosure are deterministic.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__:
    from .car_bench_agent import (
        AgentInferenceResult,
        NEXT_ACTION_OUTPUT_SCHEMA,
        _messages_for_prompt,
        parse_next_action,
    )
    from .cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        CerebrasCompletionClient,
        MalformedModelResponseError,
        TokenUsage,
        add_token_usage,
    )
    from .consensus_planner import (
        DEFAULT_MODEL2VEC_MODEL_PATH,
        Model2VecBackend,
        _decision_selected_route_ids,
        _route_disclosure_values,
        _selected_fastest_route_ids,
    )
    from .fewshot_rag import (
        EVENT_TRIGGERED_EXAMPLES,
        FewShotRetriever,
        FewShotSelection,
        SYNTHETIC_EXAMPLES,
        SYNTHETIC_EXAMPLES_V12R,
        append_selection_to_latest_user,
        selection_for_example,
    )
    from .compiled_state_policy import (
        WithheldTool,
        compile_state_brief,
        gate_catalog_by_affordance,
        prior_state_ledger,
        requested_withheld_tool,
    )
    from .harmony_native import (
        TRANSPORT_NAME as HARMONY_NATIVE_TRANSPORT,
        HarmonyNativeClient,
        HarmonyNativeParseError,
    )
    from .policy_rag import policy_token_count
    from .reformulation_ledger import (
        ReformulationBlock,
        append_reformulation_block,
        build_reformulation_block,
        history_token_count,
        reconstruct_state_ledger,
    )
else:  # pragma: no cover - loose-script compatibility
    from car_bench_agent import (
        AgentInferenceResult,
        NEXT_ACTION_OUTPUT_SCHEMA,
        _messages_for_prompt,
        parse_next_action,
    )
    from cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        CerebrasCompletionClient,
        MalformedModelResponseError,
        TokenUsage,
        add_token_usage,
    )
    from consensus_planner import (
        DEFAULT_MODEL2VEC_MODEL_PATH,
        Model2VecBackend,
        _decision_selected_route_ids,
        _route_disclosure_values,
        _selected_fastest_route_ids,
    )
    from fewshot_rag import (
        EVENT_TRIGGERED_EXAMPLES,
        FewShotRetriever,
        FewShotSelection,
        SYNTHETIC_EXAMPLES,
        SYNTHETIC_EXAMPLES_V12R,
        append_selection_to_latest_user,
        selection_for_example,
    )
    from compiled_state_policy import (
        WithheldTool,
        compile_state_brief,
        gate_catalog_by_affordance,
        prior_state_ledger,
        requested_withheld_tool,
    )
    from harmony_native import (
        TRANSPORT_NAME as HARMONY_NATIVE_TRANSPORT,
        HarmonyNativeClient,
        HarmonyNativeParseError,
    )
    from policy_rag import policy_token_count
    from reformulation_ledger import (
        ReformulationBlock,
        append_reformulation_block,
        build_reformulation_block,
        history_token_count,
        reconstruct_state_ledger,
    )


HARNESS_NAME = "adaptive_minimal"
DEFAULT_TRANSPORT = "chat"
EXECUTOR_REASONING_EFFORT = "medium"
DEFAULT_ESCALATION_BUDGET = 3
MICRO_PROMPT_TOKEN_CAP = 500
PROCEDURES_TOKEN_CAP = 80
PREFETCH_TOOL_CAP = 20
LLM_CONSENSUS_JUDGE_EPISODE_CAP = 6
LLM_CONSENSUS_JUDGE_COMPLETION_CAP = 512
LLM_CONSENSUS_JUDGE_TIMEOUT_SECONDS = 15.0
LLM_CONSENSUS_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}
LLM_ASK_TRIAGE_EPISODE_CAP = 2
LLM_ASK_TRIAGE_COMPLETION_CAP = 512
LLM_ASK_TRIAGE_TIMEOUT_SECONDS = 15.0
LLM_ASK_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "RESOLVABLE_BY_READ",
                "GENUINE_AMBIGUITY",
                "NO_AMBIGUITY",
            ],
        },
        "tool_name": {"type": "string"},
        "arguments_json": {"type": "string"},
    },
    "required": ["label", "tool_name", "arguments_json"],
    "additionalProperties": False,
}
LLM_LIMITATION_CLASSIFIER_COMPLETION_CAP = 512
LLM_LIMITATION_CLASSIFIER_TIMEOUT_SECONDS = 15.0
LLM_LIMITATION_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "CAPABILITY_UNAVAILABLE_TERMINATE",
                "CAPABILITY_AVAILABLE_CONTINUE",
            ],
        },
        "tool_name": {"type": "string"},
        "arguments_json": {"type": "string"},
        "finding": {"type": "string"},
    },
    "required": ["label", "tool_name", "arguments_json", "finding"],
    "additionalProperties": False,
}
ROUTE_BUDGET_DEFAULT = 6
# Prefetch runs before the model can bind selector arguments.  Keep its
# semantic mode deliberately narrower than the tool schema: these getters are
# the catalog's genuinely zero-argument reads.  In particular, lookup getters
# with optional-looking JSON schemas can still impose an "at least one of"
# runtime contract and must never receive an empty prefetch call.
SEMANTIC_PREFETCH_ZERO_ARGUMENT_GETTERS = frozenset(
    {
        "get_ambient_light_status_and_color",
        "get_car_color",
        "get_charging_status",
        "get_climate_settings",
        "get_current_navigation_state",
        "get_exterior_lights_status",
        "get_fuel_information",
        "get_reading_lights_status",
        "get_seat_heating_level",
        "get_seats_occupancy",
        "get_steering_wheel_heating_level",
        "get_sunroof_and_sunshade_position",
        "get_temperature_inside_car",
        "get_trunk_door_position",
        "get_user_preferences",
        "get_window_positions",
    }
)
P3I31_CHAT_COMPLETION_TOKENS = 2048
TRUNCATION_RESCUE_CAPS = (4096, 8192)
TERMINAL_RESPOND_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["respond"]},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["tool_name", "arguments_json"],
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments_json": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "required": ["action", "content", "tool_calls"],
    "additionalProperties": False,
}

# The measured policy-corpus cure lines are copied verbatim from the frozen
# p3i23.5 build (908f03d).  The final invariants preserve the corresponding
# SCATTER_SYSTEM / FINAL_REVIEW_SYSTEM corpus language.
MICRO_PROMPT = """POLICY REMINDERS (from the in-car system policy; they override style habits):
- If a tool is marked REQUIRES_CONFIRMATION, or system policy conditions an action on user approval such as approval required under certain weather or vehicle states, you MUST ask exactly once with the exact tool name and parameter values, and act only after an explicit yes. This rule wins over everything below.
- Otherwise, a direct user command for an available car-state change is itself the authorization: do not ask a discretionary confirmation, do not re-read a state already read this episode for the same subject, and do not reply with an explanation instead of the action.
- After carrying out a direct car-state command, report what you did.
- Report only what tool results confirm. Never assert an outcome you cannot verify with a tool; state what you did, not what it achieved.
- Never send essentially the same message twice. If the user repeats a request after your refusal, do the closest available action instead of refusing again.
- When a requested capability is impossible with your tools, say so plainly once, state what you did instead, and do not end with a counter-question about that same capability.

EXISTING-RESOURCE RULE: before creating or setting a new resource that may already exist, such as active navigation, an existing plan, or an already-set mode, first establish its current state from the transcript or a get_* read. Then use the tool that edits, replaces, or updates it rather than one that creates a new one. A create or set-new call on an already active resource fails.

READ-BEFORE-SET RULE: before any state-changing tool call (set_*, open_close_*, activate/deactivate) on a vehicle subsystem, first call that subsystem's read tool (get_* / *_status / positions) if one is supplied and you have not already read it this turn -- e.g. read climate settings before changing climate, read window positions before moving windows. These reads are required context, not optional; skipping the relevant read is a failure even when you are confident of the value.

minimal-action-bias: choose the smallest complete action that satisfies the current user request and avoid extra state changes.
Every tool must be available, and every argument must be schema-valid and grounded in the transcript, tool results, policy text, or tool schema."""

MICRO_PROMPT_TOKEN_COUNT = policy_token_count(MICRO_PROMPT)
assert MICRO_PROMPT_TOKEN_COUNT <= MICRO_PROMPT_TOKEN_CAP, (
    "adaptive_minimal micro-prompt exceeds its hard cap: "
    f"{MICRO_PROMPT_TOKEN_COUNT} > {MICRO_PROMPT_TOKEN_CAP}"
)

# This generic procedure summary is distilled only from ground-truth action
# sequences in the train/discovery partition.  It contains no task facts,
# identifiers, arguments, or answer mappings.
PROCEDURES_BLOCK = """CANONICAL PROCEDURES:
- Clear window fog: read climate and window state, enable defrost, direct airflow to the windshield, close open windows, and use the needed fan and air conditioning.
- Change active navigation: read current navigation, get routes, then edit or replace the active route instead of creating a new one."""
PROCEDURES_BLOCK_TOKEN_COUNT = policy_token_count(PROCEDURES_BLOCK)
assert PROCEDURES_BLOCK_TOKEN_COUNT <= PROCEDURES_TOKEN_CAP, (
    "adaptive_minimal procedures block exceeds its hard cap: "
    f"{PROCEDURES_BLOCK_TOKEN_COUNT} > {PROCEDURES_TOKEN_CAP}"
)
MICRO_PROMPT_V2 = f"{MICRO_PROMPT}\n\n{PROCEDURES_BLOCK}"
MICRO_PROMPT_V2_TOKEN_COUNT = policy_token_count(MICRO_PROMPT_V2)
assert MICRO_PROMPT_V2_TOKEN_COUNT <= MICRO_PROMPT_TOKEN_CAP, (
    "adaptive_minimal v2 micro-prompt exceeds its hard cap: "
    f"{MICRO_PROMPT_V2_TOKEN_COUNT} > {MICRO_PROMPT_TOKEN_CAP}"
)

CSP_ONE_CONFIRM_LINE = (
    "ONE-CONFIRM PLAN: for multi-step requests, state the complete plan in one "
    "message, ask for confirmation at most once, then execute all steps without "
    "further questions; never re-confirm mid-plan."
)
CSP_ONE_CONFIRM_LINE_TOKEN_COUNT = policy_token_count(CSP_ONE_CONFIRM_LINE)
MICRO_PROMPT_CSP = f"{MICRO_PROMPT}\n{CSP_ONE_CONFIRM_LINE}"
MICRO_PROMPT_CSP_TOKEN_COUNT = policy_token_count(MICRO_PROMPT_CSP)
assert MICRO_PROMPT_CSP_TOKEN_COUNT <= MICRO_PROMPT_TOKEN_CAP, (
    "adaptive_minimal CSP micro-prompt exceeds its hard cap: "
    f"{MICRO_PROMPT_CSP_TOKEN_COUNT} > {MICRO_PROMPT_TOKEN_CAP}"
)

# p3i31 restores the measured p3i23.5 policy-corpus lines verbatim and adds
# only generic task-free guidance. Blocks are ordered by the binding spec's
# keep priority; the lowest-priority tail is trimmed until the hard mass cap
# is satisfied.
_MICRO_PROMPT_V3_CANDIDATE_BLOCKS: tuple[tuple[str, str], ...] = (
    (
        "tool_grounded_honesty",
        """POLICY REMINDERS (from the in-car system policy; they override style habits):
- Report only what tool results confirm. Never assert an outcome you cannot verify with a tool; state what you did, not what it achieved.""",
    ),
    (
        "preference_fidelity",
        """- If the user says to use a stored preference/setting, read it with the preferences tool and apply the returned value; do not ask the user to supply the value.
- When the user refers to a stored or preferred setting, read user preferences and apply the exact resolved value; if the preference is conditional, evaluate it against the vehicle state after your planned changes; never ask the user to choose between preference values you can retrieve.""",
    ),
    (
        "confirmation_protocol",
        """- Before calling any tool marked REQUIRES_CONFIRMATION: first send a message that names the action AND its exact parameter values (e.g. "I'll call set_head_lights_high_beams with on: true - confirm?"), then wait for an explicit yes.
- If a tool is marked REQUIRES_CONFIRMATION, or system policy conditions an action on user approval such as approval required under certain weather or vehicle states, you MUST ask exactly once with the exact tool name and parameter values, and act only after an explicit yes. This rule wins over everything below.""",
    ),
    (
        "navigation_origin_grounding",
        "NAVIGATION-ORIGIN GROUNDING: when modifying or extending navigation, "
        "ground route searches on the ACTIVE route's endpoints, not the current "
        "location, unless the user says otherwise.",
    ),
    (
        "unavailability_closure",
        "UNAVAILABILITY CLOSURE: if a state or capability is absent, say so "
        "decisively once, offer the nearest available alternative, close the "
        "topic, and do not re-query it.",
    ),
    (
        "existing_resource",
        "EXISTING-RESOURCE RULE: before creating or setting a new resource that "
        "may already exist, such as active navigation, an existing plan, or an "
        "already-set mode, first establish its current state from the transcript "
        "or a get_* read. Then use the tool that edits, replaces, or updates it "
        "rather than one that creates a new one. A create or set-new call on an "
        "already active resource fails.",
    ),
    (
        "proceed_bias",
        """- Otherwise, a direct user command for an available car-state change is itself the authorization: do not ask a discretionary confirmation, do not re-read a state already read this episode for the same subject, and do not reply with an explanation instead of the action.
- After carrying out a direct car-state command, report what you did.""",
    ),
    (
        "minimal_action_argument_grounding",
        """minimal-action-bias: choose the smallest complete action that satisfies the current user request and avoid extra state changes.
Every tool must be available, and every argument must be schema-valid and grounded in the transcript, tool results, policy text, or tool schema.""",
    ),
    (
        "read_before_set",
        "READ-BEFORE-SET RULE: before any state-changing tool call (set_*, "
        "open_close_*, activate/deactivate) on a vehicle subsystem, first call "
        "that subsystem's read tool (get_* / *_status / positions) if one is "
        "supplied and you have not already read it this turn -- e.g. read climate "
        "settings before changing climate, read window positions before moving "
        "windows. These reads are required context, not optional; skipping the "
        "relevant read is a failure even when you are confident of the value.",
    ),
    ("procedures", PROCEDURES_BLOCK),
)


def _mass_capped_prompt(
    blocks: tuple[tuple[str, str], ...], cap: int
) -> tuple[str, tuple[str, ...]]:
    kept = list(blocks)
    while kept:
        prompt = "\n".join(text for _, text in kept)
        if policy_token_count(prompt) <= cap:
            return prompt, tuple(name for name, _ in kept)
        kept.pop()
    raise AssertionError("adaptive_minimal prompt blocks cannot satisfy mass cap")


MICRO_PROMPT_V3, MICRO_PROMPT_V3_COMPOSITION = _mass_capped_prompt(
    _MICRO_PROMPT_V3_CANDIDATE_BLOCKS,
    MICRO_PROMPT_TOKEN_CAP,
)
MICRO_PROMPT_V3_TOKEN_COUNT = policy_token_count(MICRO_PROMPT_V3)
assert MICRO_PROMPT_V3_TOKEN_COUNT <= MICRO_PROMPT_TOKEN_CAP, (
    "adaptive_minimal v3 micro-prompt exceeds its hard cap: "
    f"{MICRO_PROMPT_V3_TOKEN_COUNT} > {MICRO_PROMPT_TOKEN_CAP}"
)

SIGNAL_TOOL_NOT_IN_CATALOG = "tool_not_in_catalog"
SIGNAL_SCHEMA_VALIDATION = "schema_validation"
SIGNAL_TOOL_ERROR = "tool_error"
SIGNAL_MALFORMED_OR_EMPTY = "malformed_or_empty"
SIGNAL_REPETITION_LOOP = "repetition_loop"
SIGNAL_TURN_GUARD = "turn_guard"
SIGNAL_UNAVAILABILITY_LOOP = "unavailability_loop"
SIGNAL_MUTATION_LOG_CHECK = "mutation_log_check"
SIGNAL_GROUNDED_RESPOND = "grounded_respond"
SIGNAL_TERMINAL_READBACK = "terminal_readback"
ESCALATION_SIGNALS = (
    SIGNAL_TOOL_NOT_IN_CATALOG,
    SIGNAL_SCHEMA_VALIDATION,
    SIGNAL_TOOL_ERROR,
    SIGNAL_MALFORMED_OR_EMPTY,
    SIGNAL_REPETITION_LOOP,
    SIGNAL_TURN_GUARD,
    SIGNAL_UNAVAILABILITY_LOOP,
    SIGNAL_MUTATION_LOG_CHECK,
    SIGNAL_GROUNDED_RESPOND,
    SIGNAL_TERMINAL_READBACK,
)

METRIC_INJECTED_READS = "adaptive_minimal_injected_reads"
METRIC_ESCALATIONS_FIRED = "adaptive_minimal_escalations_fired"
METRIC_MICRO_PROMPT_TOKENS = "adaptive_minimal_micro_prompt_tokens"
METRIC_DECISIONS = "adaptive_minimal_decisions"
METRIC_UNDEFINED_TOOL_CALLS = "adaptive_minimal_undefined_tool_calls"
METRIC_SCHEMA_VIOLATIONS = "adaptive_minimal_schema_violations"
METRIC_PARSE_FAILURES = "adaptive_minimal_parse_failures"
METRIC_INPUT_TOKENS = "adaptive_minimal_input_tokens"
METRIC_TRANSPORT_CALLS = "adaptive_minimal_transport_calls"
METRIC_PREFETCH_READS = "adaptive_minimal_prefetch_reads"
METRIC_PREFETCH_ERROR_DROPS = "adaptive_minimal_prefetch_error_drops"
METRIC_PREFETCH_SEMANTIC_EMITTED = (
    "adaptive_minimal_prefetch_semantic_emitted"
)
METRIC_PREFETCH_SEMANTIC_SUPPRESSED = (
    "adaptive_minimal_prefetch_semantic_suppressed"
)
METRIC_EVENT_EXEMPLAR_PREFIX = "adaptive_minimal_event_exemplar_"
METRIC_EVENT_EXEMPLAR_TOKENS = "adaptive_minimal_event_exemplar_tokens"
METRIC_TERMINAL_READBACK_FIRES = "adaptive_minimal_terminal_readback_fires"
METRIC_TERMINAL_READBACK_READS = "adaptive_minimal_terminal_readback_reads"
METRIC_TERMINAL_READBACK_MISMATCHES = (
    "adaptive_minimal_terminal_readback_mismatches"
)
METRIC_TERMINAL_READBACK_REVISES = "adaptive_minimal_terminal_readback_revises"
METRIC_EVENT_E4_SKIPS = "adaptive_minimal_event_e4_skips"
METRIC_PHASE_GATE_DECISIONS = "adaptive_minimal_phase_gate_decisions"
METRIC_PHASE_GATE_WITHHELD_TOOLS = "adaptive_minimal_phase_gate_withheld_tools"
METRIC_PHASE_GATE_FAIL_OPENS = "adaptive_minimal_phase_gate_fail_opens"
METRIC_PHASE_GATE_HARMFUL_WITHHOLDS = (
    "adaptive_minimal_phase_gate_harmful_withholds"
)
METRIC_TERMINAL_EFFORT_HIGH_CALLS = (
    "adaptive_minimal_terminal_effort_high_calls"
)
METRIC_TERMINAL_EFFORT_MEDIUM_CALLS = (
    "adaptive_minimal_terminal_effort_medium_calls"
)
METRIC_ARGUMENT_BINDING_RELATIVE_CLARIFICATIONS = (
    "adaptive_minimal_argument_binding_relative_clarifications"
)
METRIC_ARGUMENT_BINDING_ROUTE_CORRECTIONS = (
    "adaptive_minimal_argument_binding_route_corrections"
)
METRIC_DISCLOSURE_CONFIRMATION_REASKS = (
    "adaptive_minimal_disclosure_confirmation_reasks"
)
METRIC_DISCLOSURE_UNAVAILABLE_ACKS = (
    "adaptive_minimal_disclosure_unavailable_acks"
)
METRIC_TRUNCATION_RESCUE_FIRES = "adaptive_minimal_truncation_rescue_fires"
METRIC_PLACEHOLDER_GUARD_FIRES = "adaptive_minimal_placeholder_guard_fires"
METRIC_VAGUE_DEGREE_CLARIFICATIONS = (
    "adaptive_minimal_vague_degree_clarifications"
)
METRIC_VAGUE_DEGREE_PREFERENCE_REDIRECTS = (
    "adaptive_minimal_vague_degree_preference_redirects"
)
METRIC_VAGUE_DEGREE_PREFERENCE_APPLIES = (
    "adaptive_minimal_vague_degree_preference_applies"
)
METRIC_SCHEMA_PREFLIGHT_BOUNCES = "adaptive_minimal_schema_preflight_bounces"
METRIC_SCHEMA_PREFLIGHT_PASS_THROUGHS = (
    "adaptive_minimal_schema_preflight_pass_throughs"
)
METRIC_VALUE_P1_CONTEXT_APPLIES = "adaptive_minimal_value_p1_context_applies"
METRIC_VALUE_P1_ASK_SUPPRESSIONS = (
    "adaptive_minimal_value_p1_ask_suppressions"
)
METRIC_VALUE_P2_BINDING_BOUNCES = (
    "adaptive_minimal_value_p2_binding_bounces"
)
METRIC_VALUE_P2_PASS_THROUGHS = "adaptive_minimal_value_p2_pass_throughs"
METRIC_VALUE_P3_PREFERENCE_READS = (
    "adaptive_minimal_value_p3_preference_reads"
)
METRIC_VALUE_P3_FALLBACK_ASKS = "adaptive_minimal_value_p3_fallback_asks"
METRIC_VALUE_P4_NAV_REDIRECTS = "adaptive_minimal_value_p4_nav_redirects"
METRIC_VALUE_P4_NAV_BLOCKS = "adaptive_minimal_value_p4_nav_blocks"
METRIC_VALUE_P5_OCCUPANCY_READS = "adaptive_minimal_value_p5_occupancy_reads"
METRIC_VALUE_P6_CLAIM_REVISES = "adaptive_minimal_value_p6_claim_revises"
METRIC_TIME_FORMAT_REVISES = "adaptive_minimal_time_format_revises"
METRIC_INJECTED_ASKS = "adaptive_minimal_injected_asks"
METRIC_ASK_BUDGET_SUPPRESSED = "adaptive_minimal_ask_budget_suppressed"
METRIC_MALFORMED_ARGUMENT_RESCUE_FIRES = (
    "adaptive_minimal_malformed_argument_rescue_fires"
)
METRIC_REPEATED_READ_BLOCKS = "adaptive_minimal_repeated_read_blocks"
METRIC_ROUTE_REFERENCE_BOUNCES = "adaptive_minimal_route_reference_bounces"
METRIC_POLICY_LINT_REVISES = "adaptive_minimal_policy_lint_revises"
METRIC_POLICY_LINT_ZONE_DIFFERENCE_REVISES = (
    "adaptive_minimal_policy_lint_zone_difference_revises"
)
METRIC_POLICY_LINT_TEMPERATURE_UNIT_REVISES = (
    "adaptive_minimal_policy_lint_temperature_unit_revises"
)
METRIC_ROUTE_RESOLVER_FIRES = "adaptive_minimal_route_resolver_fires"
METRIC_ROUTE_RESOLVER_BLOCKED_READS = (
    "adaptive_minimal_route_resolver_blocked_reads"
)
METRIC_ASK_TYPE_GATE_SUPPRESSIONS = (
    "adaptive_minimal_ask_type_gate_suppressions"
)
METRIC_TEXTCALL_GUARD_FIRES = "adaptive_minimal_textcall_guard_fires"
METRIC_TEXTCALL_GUARD_EXECUTES = "adaptive_minimal_textcall_guard_executes"
METRIC_TEXTCALL_GUARD_REDECIDES = "adaptive_minimal_textcall_guard_redecides"
METRIC_ARG_LINT_FIRES = "adaptive_minimal_arg_lint_fires"
METRIC_ARG_LINT_ARGUMENT_BOUNCES = (
    "adaptive_minimal_arg_lint_argument_bounces"
)
METRIC_ARG_LINT_DISCLOSURE_REVISES = (
    "adaptive_minimal_arg_lint_disclosure_revises"
)
METRIC_ROUTE_BUDGET_FIRES = "adaptive_minimal_route_budget_fires"
METRIC_ROUTE_BUDGET_BLOCKED_READS = (
    "adaptive_minimal_route_budget_blocked_reads"
)
METRIC_ROUTE_BUDGET_TERMINAL_LIMITATIONS = (
    "adaptive_minimal_route_budget_terminal_limitations"
)
METRIC_NAV_INTENT_PREFLIGHT_BOUNCES = (
    "adaptive_minimal_nav_intent_preflight_bounces"
)
METRIC_NAV_INTENT_PREFLIGHT_PASS_THROUGHS = (
    "adaptive_minimal_nav_intent_preflight_pass_throughs"
)
METRIC_STEP_COVERAGE_FIRES = "adaptive_minimal_step_coverage_fires"
METRIC_STEP_COVERAGE_REDECISIONS = (
    "adaptive_minimal_step_coverage_redecisions"
)
METRIC_P3_ASK_GATE_V2_SUPPRESSIONS = (
    "adaptive_minimal_p3_ask_gate_v2_suppressions"
)
METRIC_LLM_LIMITATION_CLASSIFIER_CALLS = (
    "adaptive_minimal_llm_limitation_classifier_calls"
)
METRIC_LLM_LIMITATION_CLASSIFIER_TERMINATES = (
    "adaptive_minimal_llm_limitation_classifier_terminates"
)
METRIC_LLM_LIMITATION_CLASSIFIER_CONTINUES = (
    "adaptive_minimal_llm_limitation_classifier_continues"
)
METRIC_LLM_LIMITATION_CLASSIFIER_ERRORS = (
    "adaptive_minimal_llm_limitation_classifier_errors"
)
METRIC_LLM_LIMITATION_CLASSIFIER_TIMEOUTS = (
    "adaptive_minimal_llm_limitation_classifier_timeouts"
)
METRIC_LLM_LIMITATION_CLASSIFIER_MALFORMED = (
    "adaptive_minimal_llm_limitation_classifier_malformed"
)
METRIC_LLM_LIMITATION_CLASSIFIER_ADDED_LATENCY_MS = (
    "adaptive_minimal_llm_limitation_classifier_added_latency_ms"
)
METRIC_READ_RESOLVE_REDIRECTS = "adaptive_minimal_read_resolve_redirects"
METRIC_GROUNDED_ASK_FIRES = "adaptive_minimal_grounded_ask_fires"
METRIC_GROUNDED_ASK_READS = "adaptive_minimal_grounded_ask_reads"
METRIC_GROUNDED_ASK_REDRAFT_ASKS = (
    "adaptive_minimal_grounded_ask_redraft_asks"
)
METRIC_GROUNDED_ASK_REDRAFT_ACTS = (
    "adaptive_minimal_grounded_ask_redraft_acts"
)
METRIC_GROUNDED_ASK_REDRAFT_RESPONDS = (
    "adaptive_minimal_grounded_ask_redraft_responds"
)
METRIC_ASK_CONTENT_CONSENSUS_INVOCATIONS = (
    "adaptive_minimal_ask_content_consensus_invocations"
)
METRIC_ASK_CONTENT_CONSENSUS_MAJORITY_SELECTIONS = (
    "adaptive_minimal_ask_content_consensus_majority_selections"
)
METRIC_ASK_CONTENT_CONSENSUS_FALLBACKS = (
    "adaptive_minimal_ask_content_consensus_fallbacks"
)
METRIC_ASK_CONTENT_CONSENSUS_EXTRA_CALLS = (
    "adaptive_minimal_ask_content_consensus_extra_llm_calls"
)
METRIC_ASK_CONTENT_CONSENSUS_ADDED_LATENCY_MS = (
    "adaptive_minimal_ask_content_consensus_added_latency_ms"
)
METRIC_LLM_CONSENSUS_JUDGE_CALLS = (
    "adaptive_minimal_llm_consensus_judge_calls"
)
METRIC_LLM_CONSENSUS_JUDGE_MAJORITIES = (
    "adaptive_minimal_llm_consensus_judge_majorities"
)
METRIC_LLM_CONSENSUS_JUDGE_OVERRIDES = (
    "adaptive_minimal_llm_consensus_judge_overrides"
)
METRIC_LLM_CONSENSUS_JUDGE_NO_MAJORITY = (
    "adaptive_minimal_llm_consensus_judge_no_majority"
)
METRIC_LLM_CONSENSUS_JUDGE_ERRORS = (
    "adaptive_minimal_llm_consensus_judge_errors"
)
METRIC_LLM_CONSENSUS_JUDGE_TIMEOUTS = (
    "adaptive_minimal_llm_consensus_judge_timeouts"
)
METRIC_LLM_CONSENSUS_JUDGE_MALFORMED = (
    "adaptive_minimal_llm_consensus_judge_malformed"
)
METRIC_LLM_CONSENSUS_JUDGE_BUDGET_SUPPRESSED = (
    "adaptive_minimal_llm_consensus_judge_budget_suppressed"
)
METRIC_LLM_CONSENSUS_JUDGE_ADDED_LATENCY_MS = (
    "adaptive_minimal_llm_consensus_judge_added_latency_ms"
)
METRIC_LLM_ASK_TRIAGE_CALLS = "adaptive_minimal_llm_ask_triage_calls"
METRIC_LLM_ASK_TRIAGE_RESOLVABLE = (
    "adaptive_minimal_llm_ask_triage_resolvable_labels"
)
METRIC_LLM_ASK_TRIAGE_GENUINE = (
    "adaptive_minimal_llm_ask_triage_genuine_ambiguity_labels"
)
METRIC_LLM_ASK_TRIAGE_NO_AMBIGUITY = (
    "adaptive_minimal_llm_ask_triage_no_ambiguity_labels"
)
METRIC_LLM_ASK_TRIAGE_FIRES = "adaptive_minimal_llm_ask_triage_fires"
METRIC_LLM_ASK_TRIAGE_READS = "adaptive_minimal_llm_ask_triage_reads"
METRIC_LLM_ASK_TRIAGE_INVALID = (
    "adaptive_minimal_llm_ask_triage_invalid_proposals"
)
METRIC_LLM_ASK_TRIAGE_ERRORS = "adaptive_minimal_llm_ask_triage_errors"
METRIC_LLM_ASK_TRIAGE_TIMEOUTS = "adaptive_minimal_llm_ask_triage_timeouts"
METRIC_LLM_ASK_TRIAGE_MALFORMED = "adaptive_minimal_llm_ask_triage_malformed"
METRIC_LLM_ASK_TRIAGE_BUDGET_SUPPRESSED = (
    "adaptive_minimal_llm_ask_triage_budget_suppressed"
)
METRIC_LLM_ASK_TRIAGE_REDRAFT_ASKS = (
    "adaptive_minimal_llm_ask_triage_redraft_asks"
)
METRIC_LLM_ASK_TRIAGE_REDRAFT_ACTS = (
    "adaptive_minimal_llm_ask_triage_redraft_acts"
)
METRIC_LLM_ASK_TRIAGE_REDRAFT_RESPONDS = (
    "adaptive_minimal_llm_ask_triage_redraft_responds"
)
METRIC_LLM_ASK_TRIAGE_ADDED_LATENCY_MS = (
    "adaptive_minimal_llm_ask_triage_added_latency_ms"
)
METRIC_MUTATION_CONSENSUS_INVOCATIONS = (
    "adaptive_minimal_mutation_consensus_invocations"
)
METRIC_MUTATION_CONSENSUS_MAJORITY_AGREEMENTS = (
    "adaptive_minimal_mutation_consensus_majority_agreements"
)
METRIC_MUTATION_CONSENSUS_MAJORITY_OVERRIDES = (
    "adaptive_minimal_mutation_consensus_majority_overrides"
)
METRIC_MUTATION_CONSENSUS_NO_MAJORITY = (
    "adaptive_minimal_mutation_consensus_no_majority_fallbacks"
)
METRIC_MUTATION_CONSENSUS_EXTRA_CALLS = (
    "adaptive_minimal_mutation_consensus_extra_llm_calls"
)
METRIC_MUTATION_CONSENSUS_ADDED_LATENCY_MS = (
    "adaptive_minimal_mutation_consensus_added_latency_ms"
)
METRIC_MUTATION_CONSENSUS_DEEPENINGS = (
    "adaptive_minimal_mutation_consensus_deepenings"
)
METRIC_MUTATION_CONSENSUS_DEEP_MAJORITIES = (
    "adaptive_minimal_mutation_consensus_deep_majorities"
)
METRIC_MUTATION_CONSENSUS_DEEP_OVERRIDES = (
    "adaptive_minimal_mutation_consensus_deep_overrides"
)
METRIC_MUTATION_CONSENSUS_STILL_NO_MAJORITY = (
    "adaptive_minimal_mutation_consensus_still_no_majority_fallbacks"
)
METRIC_MUTATION_CONSENSUS_DEEP_EXTRA_CALLS = (
    "adaptive_minimal_mutation_consensus_deep_extra_llm_calls"
)
METRIC_TERMINAL_CONSENSUS_INVOCATIONS = (
    "adaptive_minimal_terminal_consensus_invocations"
)
METRIC_TERMINAL_CONSENSUS_RESPOND_MAJORITIES = (
    "adaptive_minimal_terminal_consensus_respond_majorities"
)
METRIC_TERMINAL_CONSENSUS_ACTION_MAJORITIES = (
    "adaptive_minimal_terminal_consensus_action_majorities"
)
METRIC_TERMINAL_CONSENSUS_ACTION_OVERRIDES = (
    "adaptive_minimal_terminal_consensus_action_overrides"
)
METRIC_TERMINAL_CONSENSUS_NO_MAJORITY = (
    "adaptive_minimal_terminal_consensus_no_majority_fallbacks"
)
METRIC_TERMINAL_CONSENSUS_EXTRA_CALLS = (
    "adaptive_minimal_terminal_consensus_extra_llm_calls"
)
METRIC_TERMINAL_CONSENSUS_ADDED_LATENCY_MS = (
    "adaptive_minimal_terminal_consensus_added_latency_ms"
)
METRIC_TERMINAL_MEDIUM_REISSUES = (
    "adaptive_minimal_terminal_medium_reissues"
)
METRIC_TERMINAL_MEDIUM_RESPONDS_KEPT = (
    "adaptive_minimal_terminal_medium_responds_kept"
)
METRIC_TERMINAL_MEDIUM_TURNED_ACTION = (
    "adaptive_minimal_terminal_medium_turned_action"
)
METRIC_TERMINAL_MEDIUM_ADDED_LATENCY_MS = (
    "adaptive_minimal_terminal_medium_added_latency_ms"
)
METRIC_REFORMULATION_BLOCK_TOKENS = "adaptive_minimal_reformulation_block_tokens"
METRIC_REFORMULATION_RULES_HIT = "adaptive_minimal_reformulation_rules_hit"
METRIC_REFORMULATION_TOOLS_SUGGESTED = (
    "adaptive_minimal_reformulation_tools_suggested"
)
METRIC_HISTORY_TOKENS_WITHOUT_LEDGER = (
    "adaptive_minimal_history_tokens_without_ledger"
)
METRIC_HISTORY_TOKENS_WITH_LEDGER = "adaptive_minimal_history_tokens_with_ledger"
METRIC_LEDGER_TOKENS = "adaptive_minimal_ledger_tokens"
METRIC_CSP_RAW_HISTORY_TOKENS = "adaptive_minimal_csp_raw_history_tokens"
METRIC_CSP_BRIEF_TOKENS = "adaptive_minimal_csp_brief_tokens"
METRIC_CSP_INPUT_TOKENS = "adaptive_minimal_csp_input_tokens"
METRIC_CSP_INPUT_CALLS = "adaptive_minimal_csp_input_calls"
METRIC_CSP_WITHHELD_TOOLS = "adaptive_minimal_csp_withheld_tools"
METRIC_CSP_FAIL_OPEN_REOPENS = "adaptive_minimal_csp_fail_open_reopens"
METRIC_CSP_USER_TURNS = "adaptive_minimal_csp_user_turns"
METRIC_CSP_ASSISTANT_ASKS = "adaptive_minimal_csp_assistant_asks"
METRIC_FEWSHOT_TOKENS = "adaptive_minimal_fewshot_rag_tokens"
METRIC_FEWSHOT_SELECTION_PREFIX = "adaptive_minimal_fewshot_selection_"
METRIC_ESCALATION_BY_SIGNAL = {
    signal: f"adaptive_minimal_escalations_{signal}" for signal in ESCALATION_SIGNALS
}


@dataclass(frozen=True)
class AdaptiveMinimalConfig:
    escalation_budget: int = DEFAULT_ESCALATION_BUDGET
    prefetch: bool = False
    prefetch_semantic: bool = False
    turn_guard: bool = False
    procedures: bool = False
    autopsy_fixes: bool = False
    reformulate: bool = False
    ledger: bool = False
    csp_brief: bool = False
    csp_afford: bool = False
    prompt_file: str | None = None
    completion_cap_2048: bool = False
    mutation_log_check: bool = False
    grounded_respond: bool = False
    tool_description_enrich: bool = False
    tool_description_wave2: bool = False
    allow_prompt_overage: bool = False
    fewshot_rag: bool = False
    fewshot_rag_v12r: bool = False
    event_exemplars: bool = False
    terminal_readback: bool = False
    phase_gate: bool = False
    terminal_effort_high: bool = False
    argument_binding_guard: bool = False
    disclosure_guard: bool = False
    truncation_rescue: bool = False
    placeholder_guard: bool = False
    vague_degree_clarify: bool = False
    schema_preflight: bool = False
    value_provenance: bool = False
    time_format_revise: bool = False
    ask_budget: bool = False
    repeated_read_breaker: bool = False
    route_reference_preflight: bool = False
    policy_lint: bool = False
    rescue_quality: bool = False
    initial_cap_4096: bool = False
    initial_cap_8192: bool = False
    mutation_consensus: bool = False
    consensus_mixed_effort: bool = False
    executor_effort_high: bool = False
    struggle_effort: bool = False
    terminal_consensus: bool = False
    consensus_deepen: bool = False
    terminal_medium: bool = False
    route_resolver: bool = False
    route_budget: bool = False
    route_budget_limit: int = ROUTE_BUDGET_DEFAULT
    nav_intent_preflight: bool = False
    step_coverage: bool = False
    p3_ask_gate_v2: bool = False
    ask_type_gate: bool = False
    textcall_guard: bool = False
    arg_lint: bool = False
    read_resolve: bool = False
    grounded_ask: bool = False
    ask_content_consensus: bool = False
    llm_consensus_judge: bool = False
    llm_ask_triage: bool = False
    llm_limitation_classifier: bool = False

    @classmethod
    def from_env(cls) -> "AdaptiveMinimalConfig":
        raw = os.getenv("TRACK2_AM_ESCALATION_BUDGET")
        budget = DEFAULT_ESCALATION_BUDGET if not raw or not raw.strip() else int(raw)
        route_raw = os.getenv("TRACK2_AM_ROUTE_BUDGET_LIMIT")
        route_budget_limit = (
            ROUTE_BUDGET_DEFAULT
            if not route_raw or not route_raw.strip()
            else max(1, int(route_raw))
        )
        return cls(
            escalation_budget=max(0, budget),
            prefetch=_env_flag("TRACK2_AM_PREFETCH"),
            prefetch_semantic=_env_flag("TRACK2_AM_PREFETCH_SEMANTIC"),
            turn_guard=_env_flag("TRACK2_AM_TURN_GUARD"),
            procedures=_env_flag("TRACK2_AM_PROCEDURES"),
            autopsy_fixes=_env_flag("TRACK2_AM_AUTOPSY_FIXES"),
            reformulate=_env_flag("TRACK2_AM_REFORMULATE"),
            ledger=_env_flag("TRACK2_AM_LEDGER"),
            csp_brief=_env_flag("TRACK2_AM_CSP_BRIEF"),
            csp_afford=_env_flag("TRACK2_AM_CSP_AFFORD"),
            prompt_file=os.getenv("TRACK2_AM_PROMPT_FILE") or None,
            completion_cap_2048=_env_flag("TRACK2_AM_CAP_2048"),
            mutation_log_check=_env_flag("TRACK2_AM_MUTATION_LOG_CHECK"),
            grounded_respond=_env_flag("TRACK2_AM_GROUNDED_RESPOND"),
            tool_description_enrich=_env_flag("TRACK2_AM_TOOL_DESC_ENRICH"),
            tool_description_wave2=_env_flag("TRACK2_AM_TOOL_DESC_WAVE2"),
            allow_prompt_overage=_env_flag("TRACK2_AM_ALLOW_PROMPT_OVERAGE"),
            fewshot_rag=_env_flag("TRACK2_AM_FEWSHOT_RAG"),
            fewshot_rag_v12r=_env_flag("TRACK2_AM_FEWSHOT_RAG_V12R"),
            event_exemplars=_env_flag("TRACK2_AM_EVENT_EXEMPLARS"),
            terminal_readback=_env_flag("TRACK2_AM_TERMINAL_READBACK"),
            phase_gate=_env_flag("TRACK2_AM_PHASE_GATE"),
            terminal_effort_high=_env_flag("TRACK2_AM_TERMINAL_EFFORT_HIGH"),
            argument_binding_guard=_env_flag(
                "TRACK2_AM_ARGUMENT_BINDING_GUARD"
            ),
            disclosure_guard=_env_flag("TRACK2_AM_DISCLOSURE_GUARD"),
            truncation_rescue=_env_flag("TRACK2_AM_TRUNCATION_RESCUE"),
            placeholder_guard=_env_flag("TRACK2_AM_PLACEHOLDER_GUARD"),
            vague_degree_clarify=_env_flag("TRACK2_AM_VAGUE_DEGREE_CLARIFY"),
            schema_preflight=_env_flag("TRACK2_AM_SCHEMA_PREFLIGHT"),
            value_provenance=_env_flag("TRACK2_AM_VALUE_PROVENANCE"),
            time_format_revise=_env_flag("TRACK2_AM_TIME_FORMAT_REVISE"),
            ask_budget=_env_flag("TRACK2_AM_ASK_BUDGET"),
            repeated_read_breaker=_env_flag(
                "TRACK2_AM_REPEATED_READ_BREAKER"
            ),
            route_reference_preflight=_env_flag(
                "TRACK2_AM_ROUTE_REFERENCE_PREFLIGHT"
            ),
            policy_lint=_env_flag("TRACK2_AM_POLICY_LINT"),
            rescue_quality=_env_flag("TRACK2_AM_RESCUE_QUALITY"),
            initial_cap_4096=_env_flag("TRACK2_AM_INITIAL_CAP_4096"),
            initial_cap_8192=_env_flag("TRACK2_AM_INITIAL_CAP_8192"),
            mutation_consensus=_env_flag("TRACK2_AM_MUTATION_CONSENSUS"),
            consensus_mixed_effort=_env_flag(
                "TRACK2_AM_CONSENSUS_MIXED_EFFORT"
            ),
            executor_effort_high=_env_flag(
                "TRACK2_AM_EXECUTOR_EFFORT_HIGH"
            ),
            struggle_effort=_env_flag("TRACK2_AM_STRUGGLE_EFFORT"),
            terminal_consensus=_env_flag("TRACK2_AM_TERMINAL_CONSENSUS"),
            consensus_deepen=_env_flag("TRACK2_AM_CONSENSUS_DEEPEN"),
            terminal_medium=_env_flag("TRACK2_AM_TERMINAL_MEDIUM"),
            route_resolver=_env_flag("TRACK2_AM_ROUTE_RESOLVER"),
            route_budget=_env_flag("TRACK2_AM_ROUTE_BUDGET"),
            route_budget_limit=route_budget_limit,
            nav_intent_preflight=_env_flag(
                "TRACK2_AM_NAV_INTENT_PREFLIGHT"
            ),
            step_coverage=_env_flag("TRACK2_AM_STEP_COVERAGE"),
            p3_ask_gate_v2=_env_flag("TRACK2_AM_P3_ASK_GATE_V2"),
            ask_type_gate=_env_flag("TRACK2_AM_ASK_TYPE_GATE"),
            textcall_guard=_env_flag("TRACK2_AM_TEXTCALL_GUARD"),
            arg_lint=_env_flag("TRACK2_AM_ARG_LINT"),
            read_resolve=_env_flag("TRACK2_AM_READ_RESOLVE"),
            grounded_ask=_env_flag("TRACK2_AM_GROUNDED_ASK"),
            ask_content_consensus=_env_flag(
                "TRACK2_AM_ASK_CONTENT_CONSENSUS"
            ),
            llm_consensus_judge=_env_flag(
                "TRACK2_AM_LLM_CONSENSUS_JUDGE"
            ),
            llm_ask_triage=_env_flag("TRACK2_AM_LLM_ASK_TRIAGE"),
            llm_limitation_classifier=_env_flag(
                "TRACK2_AM_LLM_LIMITATION_CLASSIFIER"
            ),
        )


@dataclass(frozen=True)
class InternalFault:
    signal: str
    text: str


@dataclass
class _EpisodeState:
    pairings: dict[str, str]
    tools_by_name: dict[str, dict[str, Any]]
    read_ledger: set[str] = field(default_factory=set)
    processed_tool_results: set[tuple[str, str, str]] = field(default_factory=set)
    pending_action: dict[str, Any] | None = None
    pending_reads: set[str] = field(default_factory=set)
    escalations_fired: int = 0
    escalation_counts: dict[str, int] = field(
        default_factory=lambda: {signal: 0 for signal in ESCALATION_SIGNALS}
    )
    injected_reads: int = 0
    prefetch_attempted: bool = False
    prefetch_candidates: list[str] = field(default_factory=list)
    prefetch_tools_emitted: list[str] = field(default_factory=list)
    prefetch_reads: int = 0
    prefetch_semantic_emitted: int = 0
    prefetch_semantic_suppressed: int = 0
    prefetch_results_pending: bool = False
    prefetch_result_keys: set[tuple[str, str, str]] = field(default_factory=set)
    dropped_prefetch_result_keys: set[tuple[str, str, str]] = field(
        default_factory=set
    )
    prefetch_error_drops: int = 0
    non_prefetch_tool_calls_executed: int = 0
    turn_guard_fired: bool = False
    unavailability_loop_fired: bool = False
    unavailable_tool_signatures: set[str] = field(default_factory=set)
    successful_mutation_signatures: set[str] = field(default_factory=set)
    grounded_respond_fired: bool = False
    fewshot_turn_key: tuple[int, str] | None = None
    fewshot_selection: FewShotSelection | None = None
    fewshot_metrics_pending: bool = False
    fewshot_selection_counts: dict[str, int] = field(default_factory=dict)
    event_exemplar_counts: dict[str, int] = field(
        default_factory=lambda: {f"E{index}": 0 for index in range(1, 6)}
    )
    event_exemplar_turn_fires: set[tuple[str, int]] = field(default_factory=set)
    event_e5_revised: bool = False
    event_e4_skips: int = 0
    successful_mutations: list[dict[str, Any]] = field(default_factory=list)
    terminal_readback_checked_mutations: int = 0
    terminal_readback_pending_reads: set[str] = field(default_factory=set)
    terminal_readback_pending_mutation_count: int = 0
    terminal_readback_fires: int = 0
    terminal_readback_reads: int = 0
    terminal_readback_mismatches: int = 0
    terminal_readback_revises: int = 0
    terminal_readback_revise_fired: bool = False
    repetition_signature: str | None = None
    repetition_count: int = 0
    last_tool_call_analysis: str | None = None
    reformulation_turn_key: tuple[int, str] | None = None
    reformulation_block: ReformulationBlock | None = None
    reformulation_metrics_pending: bool = False
    csp_last_presented_tools_by_name: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    csp_last_withheld: tuple[WithheldTool, ...] = ()
    csp_fail_open_next: bool = False
    csp_withheld_counts: dict[str, int] = field(default_factory=dict)
    csp_fail_open_reopens: int = 0
    csp_user_turns_episode: int = 0
    csp_user_turns_pending: int = 0
    csp_assistant_asks_episode: int = 0
    csp_assistant_asks_pending: int = 0
    phase_gate_last_presented_tools_by_name: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    phase_gate_last_withheld: tuple[str, ...] = ()
    phase_gate_fail_open_next: bool = False
    phase_gate_decisions: int = 0
    phase_gate_withheld_tools: int = 0
    phase_gate_fail_opens: int = 0
    phase_gate_harmful_withholds: int = 0
    terminal_effort_high_calls: int = 0
    terminal_effort_medium_calls: int = 0
    argument_binding_relative_clarifications: int = 0
    argument_binding_route_corrections: int = 0
    argument_binding_turn_fires: set[int] = field(default_factory=set)
    disclosure_confirmation_reasks: int = 0
    disclosure_unavailable_acks: int = 0
    truncation_rescue_fires: int = 0
    placeholder_guard_fires: int = 0
    vague_degree_clarifications: int = 0
    vague_degree_turn_fires: set[int] = field(default_factory=set)
    vague_degree_preference_redirects: int = 0
    vague_degree_preference_applies: int = 0
    vague_degree_preference_turn_reads: set[int] = field(default_factory=set)
    schema_preflight_bounces: int = 0
    schema_preflight_pass_throughs: int = 0
    value_p1_context_applies: int = 0
    value_p1_ask_suppressions: int = 0
    value_p2_binding_bounces: int = 0
    value_p2_pass_throughs: int = 0
    value_p3_preference_reads: int = 0
    value_p3_fallback_asks: int = 0
    value_p4_nav_redirects: int = 0
    value_p4_nav_blocks: int = 0
    value_p5_occupancy_reads: int = 0
    value_p6_claim_revises: int = 0
    time_format_revises: int = 0
    injected_asks: int = 0
    ask_budget_suppressed: int = 0
    malformed_argument_rescue_fires: int = 0
    successful_get_results: dict[str, list[str]] = field(default_factory=dict)
    successful_tool_names: set[str] = field(default_factory=set)
    route_reference_catalog: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )
    route_candidates_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    successful_get_signatures_by_tool: dict[str, set[str]] = field(
        default_factory=dict
    )
    navigation_waypoints: list[str] = field(default_factory=list)
    repeated_read_blocks: int = 0
    route_reference_bounces: int = 0
    policy_lint_revises: int = 0
    policy_lint_zone_difference_revises: int = 0
    policy_lint_temperature_unit_revises: int = 0
    mutation_consensus_invocations: int = 0
    mutation_consensus_majority_agreements: int = 0
    mutation_consensus_majority_overrides: int = 0
    mutation_consensus_no_majority_fallbacks: int = 0
    mutation_consensus_extra_calls: int = 0
    mutation_consensus_added_latency_ms: float = 0.0
    mutation_consensus_deepenings: int = 0
    mutation_consensus_deep_majorities: int = 0
    mutation_consensus_deep_overrides: int = 0
    mutation_consensus_still_no_majority: int = 0
    mutation_consensus_deep_extra_calls: int = 0
    terminal_consensus_invocations: int = 0
    terminal_consensus_respond_majorities: int = 0
    terminal_consensus_action_majorities: int = 0
    terminal_consensus_action_overrides: int = 0
    terminal_consensus_no_majority_fallbacks: int = 0
    terminal_consensus_extra_calls: int = 0
    terminal_consensus_added_latency_ms: float = 0.0
    struggle_effort_escalated: bool = False
    struggle_effort_trigger: str | None = None
    struggle_effort_high_calls: int = 0
    struggle_effort_added_latency_ms: float = 0.0
    terminal_medium_reissues: int = 0
    terminal_medium_responds_kept: int = 0
    terminal_medium_turned_action: int = 0
    terminal_medium_added_latency_ms: float = 0.0
    route_resolver_fires: int = 0
    route_resolver_blocked_reads: int = 0
    route_getter_call_counts: dict[str, int] = field(default_factory=dict)
    route_budget_redirected_tools: set[str] = field(default_factory=set)
    route_budget_fires: int = 0
    route_budget_blocked_reads: int = 0
    route_budget_terminal_limitations: int = 0
    nav_intent_preflight_bounced: bool = False
    nav_intent_preflight_bounces: int = 0
    nav_intent_preflight_pass_throughs: int = 0
    step_coverage_redecided: bool = False
    step_coverage_fires: int = 0
    step_coverage_redecisions: int = 0
    p3_ask_gate_v2_suppressions: int = 0
    ask_type_gate_suppressions: int = 0
    textcall_guard_fires: int = 0
    textcall_guard_executes: int = 0
    textcall_guard_redecides: int = 0
    arg_lint_fires: int = 0
    arg_lint_argument_bounces: int = 0
    arg_lint_disclosure_revises: int = 0
    arg_lint_pending_disclosures: set[str] = field(default_factory=set)
    read_resolve_redirects: int = 0
    grounded_ask_seen_referents: set[str] = field(default_factory=set)
    grounded_ask_pending: dict[str, Any] | None = None
    grounded_ask_fires: int = 0
    grounded_ask_reads: int = 0
    grounded_ask_redraft_asks: int = 0
    grounded_ask_redraft_acts: int = 0
    grounded_ask_redraft_responds: int = 0
    ask_content_consensus_invocations: int = 0
    ask_content_consensus_majority_selections: int = 0
    ask_content_consensus_fallbacks: int = 0
    ask_content_consensus_extra_calls: int = 0
    ask_content_consensus_added_latency_ms: float = 0.0
    llm_consensus_judge_calls: int = 0
    llm_consensus_judge_majorities: int = 0
    llm_consensus_judge_overrides: int = 0
    llm_consensus_judge_no_majority: int = 0
    llm_consensus_judge_errors: int = 0
    llm_consensus_judge_timeouts: int = 0
    llm_consensus_judge_malformed: int = 0
    llm_consensus_judge_budget_suppressed: int = 0
    llm_consensus_judge_added_latency_ms: float = 0.0
    llm_ask_triage_pending: dict[str, Any] | None = None
    llm_ask_triage_calls: int = 0
    llm_ask_triage_resolvable: int = 0
    llm_ask_triage_genuine: int = 0
    llm_ask_triage_no_ambiguity: int = 0
    llm_ask_triage_fires: int = 0
    llm_ask_triage_reads: int = 0
    llm_ask_triage_invalid: int = 0
    llm_ask_triage_errors: int = 0
    llm_ask_triage_timeouts: int = 0
    llm_ask_triage_malformed: int = 0
    llm_ask_triage_budget_suppressed: int = 0
    llm_ask_triage_redraft_asks: int = 0
    llm_ask_triage_redraft_acts: int = 0
    llm_ask_triage_redraft_responds: int = 0
    llm_ask_triage_added_latency_ms: float = 0.0
    unavailability_evidence: list[str] = field(default_factory=list)
    llm_limitation_classifier_calls: int = 0
    llm_limitation_classifier_terminates: int = 0
    llm_limitation_classifier_continues: int = 0
    llm_limitation_classifier_errors: int = 0
    llm_limitation_classifier_timeouts: int = 0
    llm_limitation_classifier_malformed: int = 0
    llm_limitation_classifier_added_latency_ms: float = 0.0


@dataclass
class _CallTally:
    token_usage: TokenUsage | None = None
    cost: float = 0.0
    quota_wait_ms: float = 0.0
    duration_ms: float = 0.0
    calls: int = 0
    parse_failures: int = 0
    undefined_tool_calls: int = 0
    schema_violations: int = 0
    last_analysis: str | None = None
    reformulation_block_tokens: int = 0
    reformulation_rules_hit: int = 0
    reformulation_tools_suggested: int = 0
    history_tokens_without_ledger: int = 0
    history_tokens_with_ledger: int = 0
    ledger_tokens: int = 0
    csp_raw_history_tokens: int = 0
    csp_brief_tokens: int = 0
    csp_withheld_counts: dict[str, int] = field(default_factory=dict)
    csp_fail_open_reopens: int = 0
    fewshot_tokens: int = 0
    fewshot_selection_counts: dict[str, int] = field(default_factory=dict)
    event_exemplar_tokens: int = 0
    event_exemplar_fires: dict[str, int] = field(default_factory=dict)
    event_e4_skips: int = 0
    phase_gate_decisions: int = 0
    phase_gate_withheld_tools: int = 0
    phase_gate_fail_opens: int = 0
    phase_gate_harmful_withholds: int = 0
    terminal_effort_high_calls: int = 0
    terminal_effort_medium_calls: int = 0
    argument_binding_relative_clarifications: int = 0
    argument_binding_route_corrections: int = 0
    disclosure_confirmation_reasks: int = 0
    disclosure_unavailable_acks: int = 0
    truncation_rescue_fires: int = 0
    placeholder_guard_fires: int = 0
    vague_degree_clarifications: int = 0
    vague_degree_preference_redirects: int = 0
    vague_degree_preference_applies: int = 0
    schema_preflight_bounces: int = 0
    schema_preflight_pass_throughs: int = 0
    value_p1_context_applies: int = 0
    value_p1_ask_suppressions: int = 0
    value_p2_binding_bounces: int = 0
    value_p2_pass_throughs: int = 0
    value_p3_preference_reads: int = 0
    value_p3_fallback_asks: int = 0
    value_p4_nav_redirects: int = 0
    value_p4_nav_blocks: int = 0
    value_p5_occupancy_reads: int = 0
    value_p6_claim_revises: int = 0
    time_format_revises: int = 0
    injected_asks: int = 0
    ask_budget_suppressed: int = 0
    malformed_argument_rescue_fires: int = 0
    repeated_read_blocks: int = 0
    route_reference_bounces: int = 0
    policy_lint_revises: int = 0
    policy_lint_zone_difference_revises: int = 0
    policy_lint_temperature_unit_revises: int = 0
    mutation_consensus_invocations: int = 0
    mutation_consensus_majority_agreements: int = 0
    mutation_consensus_majority_overrides: int = 0
    mutation_consensus_no_majority_fallbacks: int = 0
    mutation_consensus_extra_calls: int = 0
    mutation_consensus_added_latency_ms: float = 0.0
    mutation_consensus_deepenings: int = 0
    mutation_consensus_deep_majorities: int = 0
    mutation_consensus_deep_overrides: int = 0
    mutation_consensus_still_no_majority: int = 0
    mutation_consensus_deep_extra_calls: int = 0
    terminal_consensus_invocations: int = 0
    terminal_consensus_respond_majorities: int = 0
    terminal_consensus_action_majorities: int = 0
    terminal_consensus_action_overrides: int = 0
    terminal_consensus_no_majority_fallbacks: int = 0
    terminal_consensus_extra_calls: int = 0
    terminal_consensus_added_latency_ms: float = 0.0
    terminal_medium_reissues: int = 0
    terminal_medium_responds_kept: int = 0
    terminal_medium_turned_action: int = 0
    terminal_medium_added_latency_ms: float = 0.0
    route_resolver_fires: int = 0
    route_resolver_blocked_reads: int = 0
    route_budget_fires: int = 0
    route_budget_blocked_reads: int = 0
    route_budget_terminal_limitations: int = 0
    nav_intent_preflight_bounces: int = 0
    nav_intent_preflight_pass_throughs: int = 0
    step_coverage_fires: int = 0
    step_coverage_redecisions: int = 0
    p3_ask_gate_v2_suppressions: int = 0
    ask_type_gate_suppressions: int = 0
    textcall_guard_fires: int = 0
    textcall_guard_executes: int = 0
    textcall_guard_redecides: int = 0
    arg_lint_fires: int = 0
    arg_lint_argument_bounces: int = 0
    arg_lint_disclosure_revises: int = 0
    read_resolve_redirects: int = 0
    grounded_ask_fires: int = 0
    grounded_ask_reads: int = 0
    grounded_ask_redraft_asks: int = 0
    grounded_ask_redraft_acts: int = 0
    grounded_ask_redraft_responds: int = 0
    ask_content_consensus_invocations: int = 0
    ask_content_consensus_majority_selections: int = 0
    ask_content_consensus_fallbacks: int = 0
    ask_content_consensus_extra_calls: int = 0
    ask_content_consensus_added_latency_ms: float = 0.0
    llm_consensus_judge_calls: int = 0
    llm_consensus_judge_majorities: int = 0
    llm_consensus_judge_overrides: int = 0
    llm_consensus_judge_no_majority: int = 0
    llm_consensus_judge_errors: int = 0
    llm_consensus_judge_timeouts: int = 0
    llm_consensus_judge_malformed: int = 0
    llm_consensus_judge_budget_suppressed: int = 0
    llm_consensus_judge_added_latency_ms: float = 0.0
    llm_ask_triage_calls: int = 0
    llm_ask_triage_resolvable: int = 0
    llm_ask_triage_genuine: int = 0
    llm_ask_triage_no_ambiguity: int = 0
    llm_ask_triage_fires: int = 0
    llm_ask_triage_reads: int = 0
    llm_ask_triage_invalid: int = 0
    llm_ask_triage_errors: int = 0
    llm_ask_triage_timeouts: int = 0
    llm_ask_triage_malformed: int = 0
    llm_ask_triage_budget_suppressed: int = 0
    llm_ask_triage_redraft_asks: int = 0
    llm_ask_triage_redraft_acts: int = 0
    llm_ask_triage_redraft_responds: int = 0
    llm_ask_triage_added_latency_ms: float = 0.0
    llm_limitation_classifier_calls: int = 0
    llm_limitation_classifier_terminates: int = 0
    llm_limitation_classifier_continues: int = 0
    llm_limitation_classifier_errors: int = 0
    llm_limitation_classifier_timeouts: int = 0
    llm_limitation_classifier_malformed: int = 0
    llm_limitation_classifier_added_latency_ms: float = 0.0

    def add(
        self,
        result: Any,
        *,
        calls: int = 1,
        parse_failures: int = 0,
        analysis_text: str | None = None,
    ) -> None:
        self.token_usage = add_token_usage(self.token_usage, result.token_usage)
        self.cost += result.cost
        self.quota_wait_ms += result.quota_wait_ms
        self.duration_ms += result.duration_ms
        self.calls += max(0, calls)
        self.parse_failures += max(0, parse_failures)
        self.last_analysis = analysis_text


def _completion_has_usable_action(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    try:
        action = parse_next_action(text)
    except (MalformedModelResponseError, json.JSONDecodeError, ValueError):
        return False
    if action.get("action") == "respond":
        return bool(str(action.get("content") or "").strip())
    if action.get("action") == "tool_calls":
        return bool(action.get("tool_calls"))
    return False


def _completion_has_empty_action(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return not str(payload.get("content") or "").strip() and not (
        payload.get("tool_calls") or []
    )


def _is_truncated_empty_completion(result: Any, completion_cap: int) -> bool:
    """Recognize provider length stops or reasoning-budget empty actions."""

    finish_reason = str(getattr(result, "finish_reason", "") or "").casefold()
    usage = getattr(result, "token_usage", None)
    thinking_tokens = int(getattr(usage, "reasoning_output_tokens", 0) or 0)
    near_cap = thinking_tokens >= max(0, completion_cap - 16)
    return finish_reason == "length" or (
        _completion_has_empty_action(getattr(result, "text", None)) and near_cap
    )


def _completion_has_malformed_tool_arguments(text: str | None) -> bool:
    """Recognize a structured action whose arguments_json is not JSON."""

    if not text or not text.strip():
        return False
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("action") != "tool_calls":
        return False
    for call in payload.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        raw_arguments = call.get("arguments_json")
        if not isinstance(raw_arguments, str):
            continue
        try:
            json.loads(raw_arguments)
        except json.JSONDecodeError:
            return True
    return False


def _record_reasoning_effort_call(
    reasoning_effort: str,
    *,
    state: _EpisodeState,
    tally: _CallTally,
) -> None:
    if reasoning_effort == "high":
        state.terminal_effort_high_calls += 1
        tally.terminal_effort_high_calls += 1
    elif reasoning_effort == "medium":
        state.terminal_effort_medium_calls += 1
        tally.terminal_effort_medium_calls += 1


class AdaptiveMinimalPlanner:
    """One-call executor with deterministic state grounding and hard-fault repair."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str = DEFAULT_CEREBRAS_API_BASE,
        service_tier: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int = 1024,
        transport: str = DEFAULT_TRANSPORT,
        config: AdaptiveMinimalConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.transport = (transport or DEFAULT_TRANSPORT).strip().casefold()
        if self.transport not in {DEFAULT_TRANSPORT, HARMONY_NATIVE_TRANSPORT}:
            raise ValueError(
                "adaptive-minimal transport must be chat or harmony_native"
            )
        self.config = config or AdaptiveMinimalConfig()
        self.executor_reasoning_effort = (
            "high"
            if self.config.executor_effort_high
            else EXECUTOR_REASONING_EFFORT
        )
        self.max_completion_tokens = (
            8192
            if self.config.initial_cap_8192
            else 4096
            if self.config.initial_cap_4096
            else P3I31_CHAT_COMPLETION_TOKENS
            if (self.config.autopsy_fixes or self.config.completion_cap_2048)
            and self.transport == DEFAULT_TRANSPORT
            else max_completion_tokens
        )
        self.repetition_guard_enabled = temperature is not None
        base_micro_prompt = (
            MICRO_PROMPT_V3
            if self.config.autopsy_fixes
            else MICRO_PROMPT_V2
            if self.config.procedures
            else MICRO_PROMPT
        )
        self.csp_enabled = self.config.csp_brief or self.config.csp_afford
        self.micro_prompt = (
            f"{base_micro_prompt}\n{CSP_ONE_CONFIRM_LINE}"
            if self.csp_enabled
            else base_micro_prompt
        )
        if self.config.prompt_file:
            prompt_swap = Path(self.config.prompt_file).read_text().strip()
            if not prompt_swap:
                raise ValueError("TRACK2_AM_PROMPT_FILE is empty")
            if prompt_swap.startswith("REPLACE_BASE\n"):
                self.micro_prompt = prompt_swap.removeprefix("REPLACE_BASE\n").strip()
            else:
                self.micro_prompt = f"{base_micro_prompt}\n{prompt_swap}"
        self.micro_prompt_token_count = policy_token_count(self.micro_prompt)
        if not self.config.allow_prompt_overage:
            assert self.micro_prompt_token_count <= MICRO_PROMPT_TOKEN_CAP
        self.fewshot_enabled = self.config.fewshot_rag or self.config.fewshot_rag_v12r
        self.fewshot_examples = (
            SYNTHETIC_EXAMPLES_V12R
            if self.config.fewshot_rag_v12r
            else SYNTHETIC_EXAMPLES
        )
        self.fewshot_retriever = (
            FewShotRetriever(
                Model2VecBackend(DEFAULT_MODEL2VEC_MODEL_PATH),
                examples=self.fewshot_examples,
            )
            if self.fewshot_enabled
            else None
        )
        self.logger = logger
        client_type = (
            HarmonyNativeClient
            if self.transport == HARMONY_NATIVE_TRANSPORT
            else CerebrasCompletionClient
        )
        self.client = client_type(
            api_base=api_base,
            service_tier=service_tier,
            logger=(logger.bind(context="cerebras") if logger is not None else None),
        )
        self._state_by_context: dict[str, _EpisodeState] = {}
        self.last_decision: dict[str, Any] | None = None

    def clear_context(self, context_id: str) -> None:
        self._state_by_context.pop(context_id, None)

    def _activate_struggle_effort(
        self,
        state: _EpisodeState,
        trigger: str,
        ctx_logger: Any,
    ) -> bool:
        """Escalate an episode once using only generic internal signals."""

        if (
            not self.config.struggle_effort
            or state.struggle_effort_escalated
        ):
            return False
        state.struggle_effort_escalated = True
        state.struggle_effort_trigger = trigger
        ctx_logger.info(
            "Adaptive-minimal struggle effort escalated",
            trigger=trigger,
        )
        return True

    def _episode_reasoning_effort(self, state: _EpisodeState) -> str:
        if self.config.struggle_effort and state.struggle_effort_escalated:
            return "high"
        return self.executor_reasoning_effort

    def _record_struggle_effort_call(
        self,
        *,
        state: _EpisodeState,
        reasoning_effort: str,
        duration_ms: float,
        calls: int = 1,
    ) -> None:
        if not (
            self.config.struggle_effort
            and state.struggle_effort_escalated
            and reasoning_effort == "high"
        ):
            return
        state.struggle_effort_high_calls += max(0, calls)
        # This is observed latency of struggle-gated high calls, not an
        # unobservable counterfactual delta against medium effort.
        state.struggle_effort_added_latency_ms += max(0.0, duration_ms)

    def plan(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> AgentInferenceResult:
        state = self._state_by_context.get(context_id)
        if state is None:
            tools_by_name = _tools_by_name(tools)
            state = _EpisodeState(
                pairings=derive_read_pairings(tools),
                tools_by_name=tools_by_name,
            )
            self._state_by_context[context_id] = state
            ctx_logger.info(
                "Adaptive-minimal catalog pairing derived",
                pairings=state.pairings,
                micro_prompt_tokens=self.micro_prompt_token_count,
            )

        if self.csp_enabled:
            observed_user_turns = sum(
                message.get("role") == "user" for message in messages
            )
            if observed_user_turns > state.csp_user_turns_episode:
                state.csp_user_turns_pending += (
                    observed_user_turns - state.csp_user_turns_episode
                )
                state.csp_user_turns_episode = observed_user_turns

        prefetch_result = self._maybe_prefetch(
            state=state, messages=messages, ctx_logger=ctx_logger
        )
        if prefetch_result is not None:
            return prefetch_result

        messages, prefetch_error_drops = _drop_failed_prefetch_results(
            messages, state, ctx_logger
        )
        tool_fault = _ingest_tool_results(messages, state)
        grounded_ask_redraft = (
            state.grounded_ask_pending
            if self.config.grounded_ask
            else None
        )
        if grounded_ask_redraft is not None:
            # The grounding reads were emitted by the preceding planner turn.
            # Consume the latch before calling the model so this mechanism can
            # never recursively ground its own re-draft.
            state.grounded_ask_pending = None
        llm_ask_triage_redraft = (
            state.llm_ask_triage_pending
            if self.config.llm_ask_triage
            else None
        )
        if llm_ask_triage_redraft is not None:
            # Consume before the model call. The re-draft is final for this
            # mechanism whatever its decision shape, so triage cannot loop.
            state.llm_ask_triage_pending = None
        if tool_fault is not None and tool_fault.signal == SIGNAL_TOOL_ERROR:
            self._activate_struggle_effort(
                state, "tool_execution_error", ctx_logger
            )
        assistant_turn_count = sum(
            message.get("role") == "assistant" for message in messages
        )
        if assistant_turn_count > 12:
            self._activate_struggle_effort(
                state, "assistant_turns_over_12", ctx_logger
            )
        readback_fault = (
            _complete_terminal_readback(messages, state, ctx_logger)
            if self.config.terminal_readback
            else None
        )
        if state.pending_action is not None:
            if tool_fault is not None:
                state.pending_action = None
                state.pending_reads.clear()
            elif state.pending_reads <= state.read_ledger:
                action = state.pending_action
                state.pending_action = None
                state.pending_reads.clear()
                direct_tally = _CallTally()
                action = _apply_pre_mutation_guards(
                    action,
                    state=state,
                    messages=messages,
                    config=self.config,
                    tally=direct_tally,
                    ctx_logger=ctx_logger,
                )
                if self.config.event_exemplars:
                    action, skipped = _skip_satisfied_mutations(action, state)
                    if skipped:
                        direct_tally.event_e4_skips += skipped
                        ctx_logger.info(
                            "Adaptive-minimal EVENTv2 E4 mutation skipped",
                            skipped=skipped,
                            skips_episode=state.event_e4_skips,
                            route="pending_mutation_release",
                        )
                action, injected_reads = _gate_mutations(action, state)
                if injected_reads:
                    state.injected_reads += injected_reads
                    self.last_decision = self._decision_meta(
                        state=state,
                        route="pending_read_chain",
                        calls=0,
                        injected_reads=injected_reads,
                        escalation_signal=None,
                        disclosure_fastest_fixes=0,
                    )
                    ctx_logger.info(
                        "Adaptive-minimal issued next canonical pending read",
                        tool_names=[
                            call.get("tool_name", "")
                            for call in action.get("tool_calls") or []
                        ],
                        read_ledger=sorted(state.read_ledger),
                    )
                    return self._result(
                        action,
                        direct_tally,
                        state=state,
                        injected_reads=injected_reads,
                        escalation_signal=None,
                    )
                state.last_tool_call_analysis = None
                self.last_decision = self._decision_meta(
                    state=state,
                    route="pending_mutation_release",
                    calls=0,
                    injected_reads=0,
                    escalation_signal=None,
                    disclosure_fastest_fixes=0,
                )
                ctx_logger.info(
                    "Adaptive-minimal released mutation after successful read",
                    tool_names=[
                        call.get("tool_name", "")
                        for call in action.get("tool_calls") or []
                    ],
                    read_ledger=sorted(state.read_ledger),
                )
                return self._result(
                    action,
                    direct_tally,
                    state=state,
                    injected_reads=0,
                    escalation_signal=None,
                )

        tally = _CallTally()
        escalation_signal: str | None = None
        initial_fault = tool_fault or readback_fault
        if initial_fault is not None and self._can_escalate(state):
            escalation_signal = self._record_escalation(state, initial_fault, ctx_logger)

        action: dict[str, Any] | None = None
        fault_for_call = initial_fault if escalation_signal is not None else None
        if llm_ask_triage_redraft is not None:
            original_question = str(
                llm_ask_triage_redraft.get("question") or ""
            )
            triage_note = (
                "You previously drafted this clarification question: "
                f"{original_question!r}. A classifier-selected read has now "
                "returned in the conversation. Re-draft exactly once using "
                "that result. The new decision may ask the user, issue tool "
                "calls, or respond; asking remains allowed."
            )
            if initial_fault is not None:
                triage_note += " Also account for: " + initial_fault.text
            fault_for_call = InternalFault("llm_ask_triage", triage_note)
        elif grounded_ask_redraft is not None:
            original_question = str(
                grounded_ask_redraft.get("question") or ""
            )
            grounding_note = (
                "You previously drafted this clarification question: "
                f"{original_question!r}. The deterministic grounding reads "
                "you requested are now in the conversation as tool results. "
                "Re-draft the decision exactly once using those results. The "
                "new decision may ask the user, issue tool calls, or respond; "
                "do not assume that asking is forbidden."
            )
            if initial_fault is not None:
                grounding_note += " Also account for: " + initial_fault.text
            fault_for_call = InternalFault("grounded_ask", grounding_note)
        terminal_high_reviewed = bool(
            self.config.terminal_effort_high
            and fault_for_call is not None
            and fault_for_call.signal == SIGNAL_TERMINAL_READBACK
        )
        next_reasoning_effort = (
            "high" if terminal_high_reviewed else self._episode_reasoning_effort(state)
        )
        next_terminal_draft: str | None = None
        if terminal_high_reviewed:
            ctx_logger.info(
                "Adaptive-minimal selective high effort scheduled",
                reason="terminal_readback_mismatch",
            )
        event_selection: FewShotSelection | None = None
        if self.config.event_exemplars:
            event = _detect_event_exemplar(messages, state)
            if event is not None:
                event_selection = _record_event_exemplar(
                    event,
                    state=state,
                    tally=tally,
                    messages=messages,
                    ctx_logger=ctx_logger,
                )
        placeholder_redecided = False
        schema_preflight_bounces = 0
        value_binding_bounced = False
        nav_active_redirected = False
        claim_provenance_revised = False
        time_format_revised = False
        route_reference_bounced = False
        policy_lint_revised = False
        arg_lint_argument_bounced = False
        arg_lint_disclosure_revised = False
        injected_asks_before_draft = state.injected_asks
        last_executor_context: tuple[
            InternalFault | None, FewShotSelection | None, str, str | None
        ] | None = None
        while action is None:
            call_reasoning_effort = (
                "high"
                if self.config.struggle_effort
                and state.struggle_effort_escalated
                else next_reasoning_effort
            )
            terminal_draft_for_call = next_terminal_draft
            event_selection_for_call = event_selection
            try:
                action = self._call_executor(
                    state=state,
                    messages=messages,
                    tools=tools,
                    fault=fault_for_call,
                    event_selection=event_selection,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    reasoning_effort=call_reasoning_effort,
                    terminal_draft=terminal_draft_for_call,
                )
                if (
                    self.config.terminal_medium
                    and call_reasoning_effort == "high"
                    and _action_is_pure_respond(action)
                    and state.terminal_medium_reissues < 3
                ):
                    duration_before_reissue = tally.duration_ms
                    state.terminal_medium_reissues += 1
                    tally.terminal_medium_reissues += 1
                    action = self._call_executor(
                        state=state,
                        messages=messages,
                        tools=tools,
                        fault=fault_for_call,
                        event_selection=event_selection_for_call,
                        tally=tally,
                        ctx_logger=ctx_logger,
                        reasoning_effort="medium",
                        terminal_draft=terminal_draft_for_call,
                    )
                    added_latency_ms = max(
                        0.0, tally.duration_ms - duration_before_reissue
                    )
                    state.terminal_medium_added_latency_ms += added_latency_ms
                    tally.terminal_medium_added_latency_ms += added_latency_ms
                    if _action_is_pure_respond(action):
                        state.terminal_medium_responds_kept += 1
                        tally.terminal_medium_responds_kept += 1
                        medium_outcome = "respond"
                    else:
                        state.terminal_medium_turned_action += 1
                        tally.terminal_medium_turned_action += 1
                        medium_outcome = "tool_calls"
                    ctx_logger.info(
                        "Adaptive-minimal terminal response reissued at medium effort",
                        reissues_episode=state.terminal_medium_reissues,
                        outcome=medium_outcome,
                        added_latency_ms=round(added_latency_ms, 3),
                    )
                    call_reasoning_effort = "medium"
                last_executor_context = (
                    fault_for_call,
                    event_selection_for_call,
                    call_reasoning_effort,
                    terminal_draft_for_call,
                )
                next_reasoning_effort = self._episode_reasoning_effort(state)
                next_terminal_draft = None
                event_selection = None
            except (MalformedModelResponseError, json.JSONDecodeError) as exc:
                if not getattr(exc, "parse_failures_counted", False):
                    tally.parse_failures += 1
                if getattr(exc, "rescue_exhausted", False):
                    ctx_logger.warning(
                        "Adaptive-minimal malformed-argument rescue exhausted",
                        fires_episode=state.malformed_argument_rescue_fires,
                    )
                    action = {
                        "action": "respond",
                        "content": (
                            "I couldn't safely complete the requested action, "
                            "so I did not claim that it was done."
                        ),
                    }
                    break
                fault = InternalFault(
                    SIGNAL_MALFORMED_OR_EMPTY,
                    f"Completion was malformed or empty: {exc}",
                )
                if escalation_signal is None and self._can_escalate(state):
                    escalation_signal = self._record_escalation(state, fault, ctx_logger)
                    fault_for_call = fault
                    continue
                action = _fault_fallback_action(fault)
                break

            validation_tools = state.tools_by_name
            if self.config.csp_afford:
                validation_tools = state.csp_last_presented_tools_by_name
            if self.config.phase_gate:
                validation_tools = state.phase_gate_last_presented_tools_by_name
            schema_preflight_pass_through = False
            if self.config.schema_preflight:
                preflight_fault = _schema_preflight_fault(action, validation_tools)
                if preflight_fault is not None:
                    if schema_preflight_bounces < 2:
                        schema_preflight_bounces += 1
                        state.schema_preflight_bounces += 1
                        tally.schema_preflight_bounces += 1
                        ctx_logger.info(
                            "Adaptive-minimal schema preflight bounced tool call",
                            fault=preflight_fault.text,
                            bounce=schema_preflight_bounces,
                            bounces_episode=state.schema_preflight_bounces,
                        )
                        self._activate_struggle_effort(
                            state, "schema_preflight_bounce", ctx_logger
                        )
                        fault_for_call = preflight_fault
                        action = None
                        continue
                    schema_preflight_pass_through = True
                    state.schema_preflight_pass_throughs += 1
                    tally.schema_preflight_pass_throughs += 1
                    ctx_logger.info(
                        "Adaptive-minimal schema preflight passed through after bound",
                        fault=preflight_fault.text,
                        pass_throughs_episode=(
                            state.schema_preflight_pass_throughs
                        ),
                    )
            validation_fault = validate_next_action(action, validation_tools)
            if (
                schema_preflight_pass_through
                and validation_fault is not None
                and validation_fault.signal == SIGNAL_SCHEMA_VALIDATION
            ):
                validation_fault = None
            if validation_fault is not None:
                if validation_fault.signal == SIGNAL_TOOL_NOT_IN_CATALOG:
                    tally.undefined_tool_calls += 1
                elif validation_fault.signal == SIGNAL_SCHEMA_VALIDATION:
                    tally.schema_violations += 1
                withheld_request = (
                    requested_withheld_tool(action, state.csp_last_withheld)
                    if validation_fault.signal == SIGNAL_TOOL_NOT_IN_CATALOG
                    and self.config.csp_afford
                    else None
                )
                phase_withheld_request = (
                    next(
                        (
                            str(call.get("tool_name") or "")
                            for call in action.get("tool_calls") or []
                            if str(call.get("tool_name") or "")
                            in state.phase_gate_last_withheld
                        ),
                        None,
                    )
                    if validation_fault.signal == SIGNAL_TOOL_NOT_IN_CATALOG
                    and self.config.phase_gate
                    else None
                )
                if phase_withheld_request is not None:
                    state.phase_gate_fail_open_next = True
                    state.phase_gate_harmful_withholds += 1
                    tally.phase_gate_harmful_withholds += 1
                    ctx_logger.info(
                        "Adaptive-minimal phase gate harmful withhold detected",
                        requested_tool=phase_withheld_request,
                        withheld_tools=list(state.phase_gate_last_withheld),
                        harmful_withholds_episode=(
                            state.phase_gate_harmful_withholds
                        ),
                    )
                    if escalation_signal is None and self._can_escalate(state):
                        escalation_signal = self._record_escalation(
                            state, validation_fault, ctx_logger
                        )
                    fault_for_call = validation_fault
                    action = None
                    continue
                if escalation_signal is None and self._can_escalate(state):
                    if withheld_request is not None:
                        state.csp_fail_open_next = True
                        ctx_logger.info(
                            "Adaptive-minimal CSP fail-open scheduled",
                            requested_tool=withheld_request,
                            withheld_tools=[
                                item.name for item in state.csp_last_withheld
                            ],
                        )
                    escalation_signal = self._record_escalation(
                        state, validation_fault, ctx_logger
                    )
                    fault_for_call = validation_fault
                    action = None
                    continue
                action = _fault_fallback_action(validation_fault)
                break

            if self.config.textcall_guard:
                _, textcall_detected = _textcall_guard_candidate(
                    action, state
                )
                if textcall_detected and state.textcall_guard_fires < 2:
                    state.textcall_guard_fires += 1
                    tally.textcall_guard_fires += 1
                    state.textcall_guard_redecides += 1
                    tally.textcall_guard_redecides += 1
                    outcome = "redecide_only"
                    fault_for_call = InternalFault(
                        "textcall_guard",
                        "The draft described a tool call as user-visible text "
                        "instead of executing it. Re-decide once. If the "
                        "mutation is still requested and confirmed, issue a "
                        "schema-valid call whose identifiers and addresses "
                        "exactly match successful read results; otherwise state "
                        "plainly why it cannot be executed. Never reconstruct "
                        "or direct-execute a call parsed from prose.",
                    )
                    action = None
                    ctx_logger.info(
                        "Adaptive-minimal textcall guard fired",
                        outcome=outcome,
                        fires_episode=state.textcall_guard_fires,
                        executes_episode=state.textcall_guard_executes,
                        redecides_episode=state.textcall_guard_redecides,
                    )
                    continue

            if self.config.read_resolve:
                resolved_read = _read_resolve_action(
                    action, messages=messages, state=state
                )
                if resolved_read is not None:
                    action = resolved_read
                    state.read_resolve_redirects += 1
                    tally.read_resolve_redirects += 1
                    ctx_logger.info(
                        "Adaptive-minimal read-resolve redirected answerable ask",
                        tool_names=[
                            call.get("tool_name")
                            for call in action.get("tool_calls") or []
                        ],
                        redirects_episode=state.read_resolve_redirects,
                    )

            if self.config.arg_lint and action.get("action") == "tool_calls":
                argument_lint_fault = _argument_policy_lint_fault(action)
                if (
                    argument_lint_fault is not None
                    and not arg_lint_argument_bounced
                    and state.arg_lint_argument_bounces < 2
                ):
                    arg_lint_argument_bounced = True
                    state.arg_lint_fires += 1
                    tally.arg_lint_fires += 1
                    state.arg_lint_argument_bounces += 1
                    tally.arg_lint_argument_bounces += 1
                    fault_for_call = argument_lint_fault
                    ctx_logger.info(
                        "Adaptive-minimal argument lint bounced mutation",
                        fault=argument_lint_fault.text,
                        fires_episode=state.arg_lint_fires,
                        bounces_episode=state.arg_lint_argument_bounces,
                    )
                    action = None
                    continue

            if self.config.route_reference_preflight:
                route_fault = _route_reference_preflight_fault(action, state)
                if route_fault is not None and not route_reference_bounced:
                    route_reference_bounced = True
                    state.route_reference_bounces += 1
                    tally.route_reference_bounces += 1
                    fault_for_call = route_fault
                    ctx_logger.info(
                        "Adaptive-minimal route-reference preflight bounced call",
                        fault=route_fault.text,
                        bounces_episode=state.route_reference_bounces,
                    )
                    action = None
                    continue

            if self.config.nav_intent_preflight:
                nav_fault = _nav_intent_preflight_fault(
                    action, messages=messages, state=state
                )
                if nav_fault is not None:
                    if not state.nav_intent_preflight_bounced:
                        state.nav_intent_preflight_bounced = True
                        state.nav_intent_preflight_bounces += 1
                        tally.nav_intent_preflight_bounces += 1
                        fault_for_call = nav_fault
                        ctx_logger.info(
                            "Adaptive-minimal navigation-intent preflight bounced",
                            fault=nav_fault.text,
                            bounces_episode=state.nav_intent_preflight_bounces,
                        )
                        action = None
                        continue
                    state.nav_intent_preflight_pass_throughs += 1
                    tally.nav_intent_preflight_pass_throughs += 1
                    ctx_logger.info(
                        "Adaptive-minimal navigation-intent preflight failed open",
                        fault=nav_fault.text,
                        pass_throughs_episode=(
                            state.nav_intent_preflight_pass_throughs
                        ),
                    )

            if self.config.route_resolver:
                action, resolver_blocked, resolver_note = _apply_route_resolver(
                    action, state
                )
                if resolver_blocked:
                    state.route_resolver_fires += 1
                    tally.route_resolver_fires += 1
                    state.route_resolver_blocked_reads += resolver_blocked
                    tally.route_resolver_blocked_reads += resolver_blocked
                    ctx_logger.info(
                        "Adaptive-minimal route resolver injected candidates",
                        blocked=resolver_blocked,
                        fires_episode=state.route_resolver_fires,
                        blocked_reads_episode=state.route_resolver_blocked_reads,
                    )
                    if action is None:
                        fault_for_call = InternalFault(
                            "route_resolver", str(resolver_note or "Use cached routes.")
                        )
                        continue

            if self.config.route_budget:
                action, budget_fault, blocked, terminal = _apply_route_budget(
                    action, state, limit=self.config.route_budget_limit
                )
                if blocked:
                    state.route_budget_fires += 1
                    tally.route_budget_fires += 1
                    state.route_budget_blocked_reads += blocked
                    tally.route_budget_blocked_reads += blocked
                    if terminal:
                        state.route_budget_terminal_limitations += 1
                        tally.route_budget_terminal_limitations += 1
                    ctx_logger.info(
                        "Adaptive-minimal route budget blocked calls",
                        blocked=blocked,
                        terminal_limitation=terminal,
                        fires_episode=state.route_budget_fires,
                    )
                    if budget_fault is not None:
                        fault_for_call = budget_fault
                        continue

            if self.config.repeated_read_breaker:
                action, blocked_reads = _split_repeated_get_calls(action, state)
                if blocked_reads:
                    state.repeated_read_blocks += blocked_reads
                    tally.repeated_read_blocks += blocked_reads
                    ctx_logger.info(
                        "Adaptive-minimal repeated-read loop breaker blocked calls",
                        blocked=blocked_reads,
                        blocks_episode=state.repeated_read_blocks,
                    )
                    if action is None:
                        fault_for_call = InternalFault(
                            "repeated_read_loop",
                            "This exact read already returned identical results "
                            "twice. Use the results already in context and decide; "
                            "do not execute the identical read again.",
                        )
                        continue

            if self.config.placeholder_guard:
                guarded_action, placeholder_index = _split_calls_before_placeholder(
                    action
                )
                if placeholder_index is not None:
                    state.placeholder_guard_fires += 1
                    tally.placeholder_guard_fires += 1
                    ctx_logger.info(
                        "Adaptive-minimal placeholder guard fired",
                        first_placeholder_call_index=placeholder_index,
                        concrete_prefix_calls=(
                            len(guarded_action.get("tool_calls") or [])
                            if guarded_action is not None
                            else 0
                        ),
                        fires_episode=state.placeholder_guard_fires,
                    )
                    self._activate_struggle_effort(
                        state, "placeholder_guard", ctx_logger
                    )
                    if guarded_action is not None:
                        action = guarded_action
                    elif not placeholder_redecided:
                        placeholder_redecided = True
                        fault_for_call = InternalFault(
                            SIGNAL_SCHEMA_VALIDATION,
                            "A proposed tool argument contained a placeholder. "
                            "Do not execute it. Re-issue the decision using only "
                            "concrete argument values grounded in prior messages "
                            "or tool results.",
                        )
                        action = None
                        continue
                    else:
                        action = {
                            "action": "respond",
                            "content": (
                                "I need a concrete value from the available "
                                "results before I can issue that dependent action."
                            ),
                        }

            if self.config.value_provenance:
                binding_fault = _clarification_answer_binding_fault(
                    action,
                    messages=messages,
                    tools_by_name=state.tools_by_name,
                )
                if binding_fault is not None:
                    if not value_binding_bounced:
                        value_binding_bounced = True
                        state.value_p2_binding_bounces += 1
                        tally.value_p2_binding_bounces += 1
                        ctx_logger.info(
                            "Adaptive-minimal value provenance P2 binding bounced",
                            fault=binding_fault.text,
                            bounces_episode=state.value_p2_binding_bounces,
                        )
                        self._activate_struggle_effort(
                            state, "p2_binding_bounce", ctx_logger
                        )
                        fault_for_call = binding_fault
                        action = None
                        continue
                    state.value_p2_pass_throughs += 1
                    tally.value_p2_pass_throughs += 1
                    ctx_logger.info(
                        "Adaptive-minimal value provenance P2 passed through after bound",
                        fault=binding_fault.text,
                        pass_throughs_episode=state.value_p2_pass_throughs,
                    )

                if _set_new_navigation_while_active(action, messages):
                    if not nav_active_redirected:
                        nav_active_redirected = True
                        state.value_p4_nav_redirects += 1
                        tally.value_p4_nav_redirects += 1
                        fault_for_call = InternalFault(
                            "value_provenance",
                            "Navigation is already active. Do not call "
                            "set_new_navigation. Re-decide using the available "
                            "navigation editing tools, delete current navigation "
                            "first if replacement is truly required, or answer "
                            "from the active navigation state when it already "
                            "satisfies the request.",
                        )
                        ctx_logger.info(
                            "Adaptive-minimal value provenance P4 active-navigation redirect",
                            redirects_episode=state.value_p4_nav_redirects,
                        )
                        action = None
                        continue
                    state.value_p4_nav_blocks += 1
                    tally.value_p4_nav_blocks += 1
                    ctx_logger.info(
                        "Adaptive-minimal value provenance P4 repeated set blocked",
                        blocks_episode=state.value_p4_nav_blocks,
                    )
                    action = {
                        "action": "respond",
                        "content": (
                            "Navigation is already active, so I did not start a "
                            "second new navigation. I need to use the available "
                            "editing controls or delete the active navigation "
                            "before replacing it."
                        ),
                    }

            if self.config.llm_limitation_classifier:
                limitation_evidence = _limitation_classifier_trigger(
                    action, messages=messages, state=state
                )
                if limitation_evidence is not None:
                    limitation_fault = self._apply_llm_limitation_classifier(
                        action=action,
                        evidence=limitation_evidence,
                        state=state,
                        messages=messages,
                        tally=tally,
                        ctx_logger=ctx_logger,
                    )
                    if limitation_fault is not None:
                        fault_for_call = limitation_fault
                        action = None
                        continue

            if self.config.step_coverage and not state.step_coverage_redecided:
                missing_steps = _step_coverage_missing_tools(
                    action, messages=messages, state=state
                )
                if missing_steps:
                    state.step_coverage_redecided = True
                    state.step_coverage_fires += 1
                    state.step_coverage_redecisions += 1
                    tally.step_coverage_fires += 1
                    tally.step_coverage_redecisions += 1
                    fault_for_call = InternalFault(
                        "step_coverage",
                        "The terminal draft leaves explicit requested or "
                        "catalog/policy-dependent operations unaddressed: "
                        + ", ".join(missing_steps)
                        + ". Re-decide once and execute only those schema-valid "
                        "missing steps; do not invent unrelated actions.",
                    )
                    ctx_logger.info(
                        "Adaptive-minimal step coverage scheduled re-decision",
                        missing_tools=missing_steps,
                        fires_episode=state.step_coverage_fires,
                    )
                    action = None
                    continue

            if self.config.value_provenance:
                if (
                    _respond_claims_performed_action(action)
                    and not state.successful_mutations
                    and not claim_provenance_revised
                    and state.value_p6_claim_revises < 2
                ):
                    claim_provenance_revised = True
                    state.value_p6_claim_revises += 1
                    tally.value_p6_claim_revises += 1
                    fault_for_call = InternalFault(
                        "value_provenance",
                        "The drafted response claims an action was performed "
                        "or is in progress, but this episode has no successful "
                        "mutation tool call. Only state actions actually "
                        "executed via tools. Execute the needed tool first or "
                        "state plainly what you cannot do.",
                    )
                    ctx_logger.info(
                        "Adaptive-minimal value provenance P6 claim revise fired",
                        revises_episode=state.value_p6_claim_revises,
                    )
                    action = None
                    continue

            if (
                self.config.time_format_revise
                and _respond_has_am_pm_time(action)
                and not time_format_revised
            ):
                time_format_revised = True
                state.time_format_revises += 1
                tally.time_format_revises += 1
                fault_for_call = InternalFault(
                    "time_format",
                    "The drafted response used an am/pm time. Revise it once "
                    "so every clock time is written in 24-hour format while "
                    "preserving the same facts and action.",
                )
                ctx_logger.info(
                    "Adaptive-minimal 24-hour time-format revise fired",
                    revises_episode=state.time_format_revises,
                )
                action = None
                continue

            if self.config.arg_lint and action.get("action") == "respond":
                disclosure_fault = _argument_disclosure_fault(action, state)
                if (
                    disclosure_fault is not None
                    and not arg_lint_disclosure_revised
                    and state.arg_lint_disclosure_revises < 2
                ):
                    arg_lint_disclosure_revised = True
                    state.arg_lint_fires += 1
                    tally.arg_lint_fires += 1
                    state.arg_lint_disclosure_revises += 1
                    tally.arg_lint_disclosure_revises += 1
                    fault_for_call = disclosure_fault
                    ctx_logger.info(
                        "Adaptive-minimal argument lint revised disclosure",
                        fault=disclosure_fault.text,
                        fires_episode=state.arg_lint_fires,
                        revises_episode=state.arg_lint_disclosure_revises,
                    )
                    action = None
                    continue
                if disclosure_fault is None:
                    state.arg_lint_pending_disclosures.clear()

            if self.config.policy_lint and not policy_lint_revised:
                policy_violations = _policy_lint_violations(
                    action, messages=messages, state=state
                )
                if policy_violations:
                    policy_lint_revised = True
                    state.policy_lint_revises += 1
                    tally.policy_lint_revises += 1
                    rule_names = [rule for rule, _ in policy_violations]
                    if "zone_difference" in rule_names:
                        state.policy_lint_zone_difference_revises += 1
                        tally.policy_lint_zone_difference_revises += 1
                    if "temperature_unit" in rule_names:
                        state.policy_lint_temperature_unit_revises += 1
                        tally.policy_lint_temperature_unit_revises += 1
                    fault_for_call = InternalFault(
                        "policy_lint",
                        " ".join(note for _, note in policy_violations),
                    )
                    ctx_logger.info(
                        "Adaptive-minimal policy post-condition lint revised response",
                        rules=rule_names,
                        revises_episode=state.policy_lint_revises,
                    )
                    action = None
                    continue

            repetition_fault = self._repetition_fault(action, state)
            turn_guard_fault = (
                _turn_guard_fault(
                    action,
                    non_prefetch_tool_calls_executed=(
                        state.non_prefetch_tool_calls_executed
                    ),
                    already_fired=state.turn_guard_fired,
                )
                if self.config.turn_guard
                else None
            )
            unavailability_fault = (
                _unavailability_loop_fault(
                    action,
                    unavailable_tool_signatures=state.unavailable_tool_signatures,
                    already_fired=state.unavailability_loop_fired,
                )
                if self.config.autopsy_fixes
                else None
            )
            mutation_fault = (
                _mutation_log_fault(action, state, messages)
                if self.config.mutation_log_check
                else None
            )
            grounded_fault = (
                _grounded_respond_fault(
                    action, messages, already_fired=state.grounded_respond_fired
                )
                if self.config.grounded_respond
                else None
            )
            if (
                self.config.event_exemplars
                and not state.event_e5_revised
                and _draft_has_ungrounded_state_claim(action, messages)
            ):
                state.event_e5_revised = True
                event_selection = _record_event_exemplar(
                    "E5",
                    state=state,
                    tally=tally,
                    messages=messages,
                    ctx_logger=ctx_logger,
                )
                if self.config.terminal_effort_high:
                    terminal_high_reviewed = True
                    next_reasoning_effort = "high"
                    next_terminal_draft = str(action.get("content") or "")
                    ctx_logger.info(
                        "Adaptive-minimal selective high effort scheduled",
                        reason="event_e5_mismatch",
                    )
                action = None
                continue
            corrective_fault = (
                unavailability_fault
                or mutation_fault
                or grounded_fault
                or repetition_fault
                or turn_guard_fault
            )
            if (
                corrective_fault is not None
                and (
                    escalation_signal is None
                    or (
                        unavailability_fault is not None
                        and escalation_signal == SIGNAL_TOOL_ERROR
                    )
                )
                and self._can_escalate(state)
            ):
                escalation_signal = self._record_escalation(
                    state, corrective_fault, ctx_logger
                )
                fault_for_call = corrective_fault
                action = None
                continue
            if (
                self.config.terminal_effort_high
                and action.get("action") == "respond"
                and not terminal_high_reviewed
            ):
                terminal_high_reviewed = True
                next_reasoning_effort = "high"
                next_terminal_draft = str(action.get("content") or "")
                ctx_logger.info(
                    "Adaptive-minimal selective high effort scheduled",
                    reason="terminal_respond",
                )
                action = None
                continue
            break

        if (
            self.config.llm_ask_triage
            and llm_ask_triage_redraft is None
            and _is_clarification_question(action)
            and state.injected_asks == injected_asks_before_draft
        ):
            triage_read, triage_outcome = self._apply_llm_ask_triage(
                action=action,
                state=state,
                messages=messages,
                tools=tools,
                tally=tally,
                ctx_logger=ctx_logger,
            )
            if triage_read is not None:
                state.llm_ask_triage_pending = {
                    "question": str(action.get("content") or ""),
                }
                state.llm_ask_triage_fires += 1
                state.llm_ask_triage_reads += 1
                state.injected_reads += 1
                tally.llm_ask_triage_fires += 1
                tally.llm_ask_triage_reads += 1
                self.last_decision = self._decision_meta(
                    state=state,
                    route="llm_ask_triage_read",
                    calls=tally.calls,
                    injected_reads=1,
                    escalation_signal=escalation_signal,
                    disclosure_fastest_fixes=0,
                )
                ctx_logger.info(
                    "Adaptive-minimal LLM ask triage injected free read",
                    outcome=triage_outcome,
                    tool_name=(triage_read.get("tool_calls") or [{}])[0].get(
                        "tool_name"
                    ),
                    calls_episode=state.llm_ask_triage_calls,
                    fires_episode=state.llm_ask_triage_fires,
                    reads_episode=state.llm_ask_triage_reads,
                )
                ctx_logger.info("Adaptive-minimal decision", **self.last_decision)
                return self._result(
                    triage_read,
                    tally,
                    state=state,
                    injected_reads=1,
                    escalation_signal=escalation_signal,
                    prefetch_error_drops=prefetch_error_drops,
                )

        if self.config.grounded_ask and grounded_ask_redraft is None:
            grounded = _grounded_ask_read_action(
                action,
                messages=messages,
                state=state,
            )
            if grounded is not None:
                read_action, referent_key = grounded
                read_count = len(read_action.get("tool_calls") or [])
                state.grounded_ask_seen_referents.add(referent_key)
                state.grounded_ask_pending = {
                    "question": str(action.get("content") or ""),
                    "referent_key": referent_key,
                }
                state.grounded_ask_fires += 1
                state.grounded_ask_reads += read_count
                tally.grounded_ask_fires += 1
                tally.grounded_ask_reads += read_count
                self.last_decision = self._decision_meta(
                    state=state,
                    route="grounded_ask_reads",
                    calls=tally.calls,
                    injected_reads=0,
                    escalation_signal=escalation_signal,
                    disclosure_fastest_fixes=0,
                )
                ctx_logger.info(
                    "Adaptive-minimal grounded ask injected free reads",
                    tool_names=[
                        call.get("tool_name")
                        for call in read_action.get("tool_calls") or []
                    ],
                    fires_episode=state.grounded_ask_fires,
                    reads_episode=state.grounded_ask_reads,
                )
                ctx_logger.info("Adaptive-minimal decision", **self.last_decision)
                return self._result(
                    read_action,
                    tally,
                    state=state,
                    injected_reads=0,
                    escalation_signal=escalation_signal,
                    prefetch_error_drops=prefetch_error_drops,
                )

        action = _apply_pre_mutation_guards(
            action,
            state=state,
            messages=messages,
            config=self.config,
            tally=tally,
            ctx_logger=ctx_logger,
        )

        if (
            self.config.mutation_consensus
            and _action_contains_mutation(action)
            and state.mutation_consensus_invocations < 6
            and last_executor_context is not None
        ):
            action = self._apply_mutation_consensus(
                original_action=action,
                state=state,
                messages=messages,
                tools=tools,
                executor_context=last_executor_context,
                tally=tally,
                ctx_logger=ctx_logger,
            )
        elif (
            self.config.terminal_consensus
            and _action_is_pure_respond(action)
            and not _conversation_has_ended(messages)
            and state.terminal_consensus_invocations < 4
            and last_executor_context is not None
        ):
            action = self._apply_terminal_consensus(
                original_action=action,
                state=state,
                messages=messages,
                tools=tools,
                executor_context=last_executor_context,
                tally=tally,
                ctx_logger=ctx_logger,
            )

        if grounded_ask_redraft is not None:
            if _is_clarification_question(action):
                state.grounded_ask_redraft_asks += 1
                tally.grounded_ask_redraft_asks += 1
                grounded_outcome = "ask"
            elif action.get("action") == "tool_calls":
                state.grounded_ask_redraft_acts += 1
                tally.grounded_ask_redraft_acts += 1
                grounded_outcome = "act"
            else:
                state.grounded_ask_redraft_responds += 1
                tally.grounded_ask_redraft_responds += 1
                grounded_outcome = "respond"
            ctx_logger.info(
                "Adaptive-minimal grounded ask re-draft completed",
                outcome=grounded_outcome,
                redraft_asks_episode=state.grounded_ask_redraft_asks,
                redraft_acts_episode=state.grounded_ask_redraft_acts,
                redraft_responds_episode=(
                    state.grounded_ask_redraft_responds
                ),
            )

        if llm_ask_triage_redraft is not None:
            if _is_clarification_question(action):
                state.llm_ask_triage_redraft_asks += 1
                tally.llm_ask_triage_redraft_asks += 1
                triage_redraft_outcome = "ask"
            elif action.get("action") == "tool_calls":
                state.llm_ask_triage_redraft_acts += 1
                tally.llm_ask_triage_redraft_acts += 1
                triage_redraft_outcome = "act"
            else:
                state.llm_ask_triage_redraft_responds += 1
                tally.llm_ask_triage_redraft_responds += 1
                triage_redraft_outcome = "respond"
            ctx_logger.info(
                "Adaptive-minimal LLM ask triage re-draft completed",
                outcome=triage_redraft_outcome,
                redraft_asks_episode=state.llm_ask_triage_redraft_asks,
                redraft_acts_episode=state.llm_ask_triage_redraft_acts,
                redraft_responds_episode=(
                    state.llm_ask_triage_redraft_responds
                ),
            )

        if (
            self.config.ask_content_consensus
            and _is_clarification_question(action)
            and state.ask_content_consensus_invocations < 3
            and last_executor_context is not None
        ):
            action = self._apply_ask_content_consensus(
                original_action=action,
                state=state,
                messages=messages,
                tools=tools,
                executor_context=last_executor_context,
                tally=tally,
                ctx_logger=ctx_logger,
            )

        if self.config.terminal_readback and action.get("action") == "respond":
            readback_action, read_names = _terminal_readback_action(state)
            if readback_action is not None:
                self.last_decision = self._decision_meta(
                    state=state,
                    route="terminal_readback",
                    calls=tally.calls,
                    injected_reads=0,
                    escalation_signal=escalation_signal,
                    disclosure_fastest_fixes=0,
                )
                ctx_logger.info(
                    "Adaptive-minimal terminal read-back fired",
                    read_tools=read_names,
                    mutation_count=state.terminal_readback_pending_mutation_count,
                    fires_episode=state.terminal_readback_fires,
                    reads_episode=state.terminal_readback_reads,
                )
                ctx_logger.info("Adaptive-minimal decision", **self.last_decision)
                return self._result(
                    readback_action,
                    tally,
                    state=state,
                    injected_reads=0,
                    escalation_signal=escalation_signal,
                    prefetch_error_drops=prefetch_error_drops,
                )

        disclosure_fastest_fixes = 0
        if action.get("action") == "respond":
            content, disclosure_fastest_fixes = _apply_fastest_route_disclosure(
                str(action.get("content") or ""), messages
            )
            if self.config.disclosure_guard:
                content, unavailable_ack = _apply_unavailable_disclosure_guard(
                    content, messages
                )
                if unavailable_ack:
                    state.disclosure_unavailable_acks += 1
                    tally.disclosure_unavailable_acks += 1
                    ctx_logger.info(
                        "Adaptive-minimal unavailable disclosure guard fired",
                        acknowledgments_episode=(
                            state.disclosure_unavailable_acks
                        ),
                    )
            action = {"action": "respond", "content": content}
            if self.csp_enabled and "?" in content:
                state.csp_assistant_asks_episode += 1
                state.csp_assistant_asks_pending += 1

        if self.config.event_exemplars:
            action, skipped = _skip_satisfied_mutations(action, state)
            if skipped:
                tally.event_e4_skips += skipped
                ctx_logger.info(
                    "Adaptive-minimal EVENTv2 E4 mutation skipped",
                    skipped=skipped,
                    skips_episode=state.event_e4_skips,
                    route="executor",
                )
        action, injected_reads = _gate_mutations(action, state)
        if self.transport == HARMONY_NATIVE_TRANSPORT:
            state.last_tool_call_analysis = (
                tally.last_analysis
                if action.get("action") == "tool_calls"
                else None
            )
        state.injected_reads += injected_reads
        self.last_decision = self._decision_meta(
            state=state,
            route="executor",
            calls=tally.calls,
            injected_reads=injected_reads,
            escalation_signal=escalation_signal,
            disclosure_fastest_fixes=disclosure_fastest_fixes,
        )
        ctx_logger.info(
            "Adaptive-minimal decision",
            **self.last_decision,
        )
        return self._result(
            action,
            tally,
            state=state,
            injected_reads=injected_reads,
            escalation_signal=escalation_signal,
            prefetch_error_drops=prefetch_error_drops,
        )

    def _maybe_prefetch(
        self,
        *,
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> AgentInferenceResult | None:
        if not self.config.prefetch or state.prefetch_attempted:
            return None
        state.prefetch_attempted = True
        user_text = _user_turn_key(messages)[1]
        candidate_calls = derive_prefetch_calls(
            list(state.tools_by_name.values()), user_text=user_text
        )
        if self.config.prefetch_semantic:
            semantic_calls: list[dict[str, Any]] = []
            suppressed: list[str] = []
            for call in candidate_calls:
                tool_name = str(call.get("tool_name") or "")
                tool = state.tools_by_name.get(tool_name) or {}
                if _semantic_prefetch_call_valid(tool, call.get("arguments") or {}):
                    semantic_calls.append(call)
                else:
                    suppressed.append(tool_name)
            candidate_calls = semantic_calls
            state.prefetch_semantic_suppressed = len(suppressed)
            ctx_logger.info(
                "Adaptive-minimal semantic prefetch filtered",
                emitted=[call["tool_name"] for call in candidate_calls],
                suppressed=suppressed,
                suppressed_count=len(suppressed),
            )
        all_candidate_names = [call["tool_name"] for call in candidate_calls]
        candidate_count = len(candidate_calls)
        if candidate_count > PREFETCH_TOOL_CAP:
            candidate_calls = candidate_calls[:PREFETCH_TOOL_CAP]
            ctx_logger.info(
                "Adaptive-minimal prefetch capped",
                candidate_count=candidate_count,
                cap=PREFETCH_TOOL_CAP,
                candidates=all_candidate_names,
                emitted=[call["tool_name"] for call in candidate_calls],
            )

        state.prefetch_candidates = all_candidate_names
        state.prefetch_tools_emitted = [
            call["tool_name"] for call in candidate_calls
        ]
        state.prefetch_reads = len(candidate_calls)
        state.prefetch_semantic_emitted = (
            len(candidate_calls) if self.config.prefetch_semantic else 0
        )
        state.prefetch_results_pending = bool(state.prefetch_tools_emitted)
        ctx_logger.info(
            "Adaptive-minimal prefetch emitted",
            candidate_count=candidate_count,
            cap=PREFETCH_TOOL_CAP,
            candidates=state.prefetch_candidates,
            emitted=state.prefetch_tools_emitted,
        )
        if not state.prefetch_tools_emitted:
            return None
        action = {
            "action": "tool_calls",
            "tool_calls": candidate_calls,
        }
        self.last_decision = self._decision_meta(
            state=state,
            route="episode_prefetch",
            calls=0,
            injected_reads=0,
            escalation_signal=None,
            disclosure_fastest_fixes=0,
        )
        return self._result(
            action,
            _CallTally(),
            state=state,
            injected_reads=0,
            escalation_signal=None,
            prefetch_reads=len(candidate_calls),
        )

    def _repetition_fault(
        self,
        action: dict[str, Any],
        state: _EpisodeState,
    ) -> InternalFault | None:
        if not self.repetition_guard_enabled:
            return None
        if action.get("action") != "tool_calls":
            state.repetition_signature = None
            state.repetition_count = 0
            return None
        for call in action.get("tool_calls") or []:
            signature = _tool_call_signature(call)
            if signature == state.repetition_signature:
                state.repetition_count += 1
            else:
                state.repetition_signature = signature
                state.repetition_count = 1
            if state.repetition_count >= 3:
                return InternalFault(
                    SIGNAL_REPETITION_LOOP,
                    "The same tool call with identical arguments was emitted "
                    "three times consecutively. Break the repetition loop: use "
                    "new grounded information, choose a different valid next "
                    "action, or respond once if the task is complete.",
                )
        return None

    def _call_executor(
        self,
        *,
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        fault: InternalFault | None,
        event_selection: FewShotSelection | None,
        tally: _CallTally,
        ctx_logger: Any,
        reasoning_effort: str = EXECUTOR_REASONING_EFFORT,
        terminal_draft: str | None = None,
    ) -> dict[str, Any]:
        executor_messages = messages
        ledger_tokens = 0
        compiled_brief = None
        if self.config.csp_brief:
            compiled_brief = compile_state_brief(messages)
            executor_messages = compiled_brief.messages
            tally.csp_raw_history_tokens += compiled_brief.raw_history_tokens
            tally.csp_brief_tokens += compiled_brief.brief_tokens
        elif self.config.reformulate:
            user_turn_key = _user_turn_key(messages)
            user_text = user_turn_key[1]
            if user_turn_key != state.reformulation_turn_key:
                system_text = next(
                    (
                        str(message.get("content") or "")
                        for message in messages
                        if message.get("role") == "system"
                    ),
                    "",
                )
                state.reformulation_block = build_reformulation_block(
                    user_text=user_text,
                    system_text=system_text,
                    tools=tools,
                )
                state.reformulation_turn_key = user_turn_key
                state.reformulation_metrics_pending = True
            if state.reformulation_block is not None:
                executor_messages = append_reformulation_block(
                    executor_messages, state.reformulation_block
                )
                if state.reformulation_metrics_pending:
                    tally.reformulation_block_tokens += (
                        state.reformulation_block.token_count
                    )
                    tally.reformulation_rules_hit += len(
                        state.reformulation_block.rule_ids
                    )
                    tally.reformulation_tools_suggested += len(
                        state.reformulation_block.tool_names
                    )
                    state.reformulation_metrics_pending = False

        if self.fewshot_enabled:
            user_turn_key = _user_turn_key(messages)
            if user_turn_key != state.fewshot_turn_key:
                assert self.fewshot_retriever is not None
                state.fewshot_selection = self.fewshot_retriever.select(
                    user_turn_key[1]
                )
                state.fewshot_turn_key = user_turn_key
                state.fewshot_metrics_pending = True
                example_id = state.fewshot_selection.example_id
                state.fewshot_selection_counts[example_id] = (
                    state.fewshot_selection_counts.get(example_id, 0) + 1
                )
                ctx_logger.info(
                    "Adaptive-minimal few-shot example selected",
                    example_id=example_id,
                    example_tokens=state.fewshot_selection.token_count,
                    score=round(state.fewshot_selection.score, 6),
                    user_turn=user_turn_key[0],
                )
            if state.fewshot_selection is not None:
                executor_messages = append_selection_to_latest_user(
                    executor_messages, state.fewshot_selection
                )
                if state.fewshot_metrics_pending:
                    tally.fewshot_tokens += state.fewshot_selection.token_count
                    tally.fewshot_selection_counts[
                        state.fewshot_selection.example_id
                    ] = 1
                    state.fewshot_metrics_pending = False

        if event_selection is not None:
            executor_messages = append_selection_to_latest_user(
                executor_messages, event_selection
            )

        history_without_ledger = 0
        history_with_ledger = 0
        if self.config.ledger and not self.config.csp_brief:
            history_without_ledger = history_token_count(
                _messages_for_prompt(executor_messages)
            )
            ledger_view = reconstruct_state_ledger(executor_messages)
            executor_messages = ledger_view.messages
            ledger_tokens = ledger_view.token_count
            history_with_ledger = history_token_count(
                _messages_for_prompt(executor_messages)
            )
            tally.history_tokens_without_ledger += history_without_ledger
            tally.history_tokens_with_ledger += history_with_ledger
            tally.ledger_tokens += ledger_tokens

        presented_tools = tools
        enriched_tool_count = 0
        if self.config.csp_afford:
            if state.csp_fail_open_next:
                state.csp_fail_open_next = False
                state.csp_last_withheld = ()
                state.csp_fail_open_reopens += 1
                tally.csp_fail_open_reopens += 1
                ctx_logger.info(
                    "Adaptive-minimal CSP catalog fail-open",
                    presented_tool_count=len(tools),
                    fail_open_reopens_episode=state.csp_fail_open_reopens,
                )
            else:
                affordance_ledger = (
                    compiled_brief.ledger
                    if compiled_brief is not None
                    else prior_state_ledger(messages)
                )
                presented_tools, withheld = gate_catalog_by_affordance(
                    tools, affordance_ledger
                )
                state.csp_last_withheld = withheld
                for item in withheld:
                    state.csp_withheld_counts[item.name] = (
                        state.csp_withheld_counts.get(item.name, 0) + 1
                    )
                    tally.csp_withheld_counts[item.name] = (
                        tally.csp_withheld_counts.get(item.name, 0) + 1
                    )
                if withheld:
                    ctx_logger.info(
                        "Adaptive-minimal CSP tools withheld",
                        withheld=[
                            {
                                "tool": item.name,
                                "family": item.family,
                                "resource": item.resource,
                                "contradiction": item.contradiction,
                            }
                            for item in withheld
                        ],
                        count=len(withheld),
                    )
            state.csp_last_presented_tools_by_name = _tools_by_name(presented_tools)

        if self.config.phase_gate:
            state.phase_gate_decisions += 1
            tally.phase_gate_decisions += 1
            if state.phase_gate_fail_open_next:
                state.phase_gate_fail_open_next = False
                state.phase_gate_last_withheld = ()
                state.phase_gate_fail_opens += 1
                tally.phase_gate_fail_opens += 1
                ctx_logger.info(
                    "Adaptive-minimal phase gate fail-open",
                    reason="previous_withheld_tool_requested",
                    presented_tool_count=len(presented_tools),
                    fail_opens_episode=state.phase_gate_fail_opens,
                )
            else:
                (
                    presented_tools,
                    phase_withheld,
                    phase_fail_open_reason,
                    referenced_tokens,
                ) = gate_catalog_by_phase(
                    presented_tools,
                    messages,
                    pairings=state.pairings,
                )
                state.phase_gate_last_withheld = phase_withheld
                state.phase_gate_withheld_tools += len(phase_withheld)
                tally.phase_gate_withheld_tools += len(phase_withheld)
                if phase_fail_open_reason is not None:
                    state.phase_gate_fail_opens += 1
                    tally.phase_gate_fail_opens += 1
                    ctx_logger.info(
                        "Adaptive-minimal phase gate fail-open",
                        reason=phase_fail_open_reason,
                        referenced_tokens=sorted(referenced_tokens),
                        presented_tool_count=len(presented_tools),
                        fail_opens_episode=state.phase_gate_fail_opens,
                    )
                else:
                    ctx_logger.info(
                        "Adaptive-minimal phase gate applied",
                        referenced_tokens=sorted(referenced_tokens),
                        withheld_tools=list(phase_withheld),
                        withheld_count=len(phase_withheld),
                        presented_tool_count=len(presented_tools),
                        decisions_episode=state.phase_gate_decisions,
                    )
            state.phase_gate_last_presented_tools_by_name = _tools_by_name(
                presented_tools
            )

        if self.config.tool_description_enrich or self.config.tool_description_wave2:
            presented_tools = enrich_tool_descriptions(
                presented_tools,
                wave2_notes=self.config.tool_description_wave2,
            )
            enriched_tool_count = sum(
                str(enriched.get("function", {}).get("description") or "")
                != str(original.get("function", {}).get("description") or "")
                for original, enriched in zip(tools, presented_tools)
            )

        prompt_fault = fault
        if terminal_draft is not None:
            presented_tools = []
            prompt_fault = InternalFault(
                "terminal_effort_review",
                "The medium-effort decision selected a terminal response. "
                "Produce the final grounded terminal response now; do not issue "
                "a tool call. Medium draft: "
                + terminal_draft,
            )
        prompt = build_adaptive_prompt(
            messages=executor_messages, tools=presented_tools, fault=prompt_fault
        )
        instrumentation: dict[str, Any] = {}
        if self.config.tool_description_enrich or self.config.tool_description_wave2:
            instrumentation["tool_descriptions_enriched"] = enriched_tool_count
            instrumentation["tool_description_wave2"] = self.config.tool_description_wave2
        if self.config.reformulate and state.reformulation_block is not None:
            instrumentation.update(
                reformulation_block_tokens=state.reformulation_block.token_count,
                reformulation_rules_hit=len(state.reformulation_block.rule_ids),
                reformulation_tools_suggested=list(
                    state.reformulation_block.tool_names
                ),
            )
        if self.config.ledger:
            instrumentation.update(
                history_tokens_without_ledger=history_without_ledger,
                history_tokens_with_ledger=history_with_ledger,
                ledger_tokens=ledger_tokens,
            )
        if compiled_brief is not None:
            instrumentation.update(
                csp_raw_history_tokens=compiled_brief.raw_history_tokens,
                csp_brief_tokens=compiled_brief.brief_tokens,
                csp_prior_requests=compiled_brief.prior_request_count,
                csp_fresh_tool_results=compiled_brief.fresh_tool_result_count,
            )
        if self.config.csp_afford:
            instrumentation.update(
                csp_presented_tool_count=len(presented_tools),
                csp_withheld_tools=[
                    item.name for item in state.csp_last_withheld
                ],
                csp_fail_open_reopens_episode=state.csp_fail_open_reopens,
            )
        if self.config.phase_gate:
            instrumentation.update(
                phase_gate_presented_tool_count=len(presented_tools),
                phase_gate_withheld_tools=list(state.phase_gate_last_withheld),
                phase_gate_decisions_episode=state.phase_gate_decisions,
                phase_gate_fail_opens_episode=state.phase_gate_fail_opens,
                phase_gate_harmful_withholds_episode=(
                    state.phase_gate_harmful_withholds
                ),
            )
        if self.fewshot_enabled and state.fewshot_selection is not None:
            instrumentation.update(
                fewshot_example_id=state.fewshot_selection.example_id,
                fewshot_example_tokens=state.fewshot_selection.token_count,
                fewshot_selection_counts_episode=dict(
                    state.fewshot_selection_counts
                ),
            )
        if event_selection is not None:
            instrumentation.update(
                event_exemplar_id=event_selection.example_id,
                event_exemplar_tokens=event_selection.token_count,
                event_exemplar_counts_episode=dict(state.event_exemplar_counts),
            )
        ctx_logger.debug(
            "Calling adaptive-minimal executor",
            model=self.model,
            call=tally.calls + 1,
            escalation_signal=fault.signal if fault is not None else None,
            reasoning_effort=reasoning_effort,
            prompt_chars=len(prompt),
            transport=self.transport,
            **instrumentation,
        )
        if self.transport == HARMONY_NATIVE_TRANSPORT:
            def record_native_call(
                completion: Any,
                *,
                calls: int,
                parse_failures: int,
                analysis_text: str | None,
                effort: str,
            ) -> None:
                tally.add(
                    completion,
                    calls=calls,
                    parse_failures=parse_failures,
                    analysis_text=analysis_text,
                )
                if effort == "high":
                    state.terminal_effort_high_calls += calls
                    tally.terminal_effort_high_calls += calls
                else:
                    state.terminal_effort_medium_calls += calls
                    tally.terminal_effort_medium_calls += calls
                self._record_struggle_effort_call(
                    state=state,
                    reasoning_effort=effort,
                    duration_ms=float(completion.duration_ms),
                    calls=calls,
                )

            def native_attempt(cap: int, effort: str) -> Any:
                return self.client.generate_action(
                    model=self.model,
                    messages=executor_messages,
                    tools=presented_tools,
                    developer_instructions=self.micro_prompt,
                    fault_text=(
                        prompt_fault.text if prompt_fault is not None else None
                    ),
                    max_completion_tokens=cap,
                    temperature=self.temperature,
                    reasoning_effort=effort,
                    analysis_to_replay=state.last_tool_call_analysis,
                )

            try:
                native = native_attempt(
                    self.max_completion_tokens, reasoning_effort
                )
            except HarmonyNativeParseError as exc:
                if exc.completion is not None:
                    record_native_call(
                        exc.completion,
                        calls=exc.call_count,
                        parse_failures=exc.parse_failures,
                        analysis_text=None,
                        effort=reasoning_effort,
                    )
                else:
                    tally.parse_failures += exc.parse_failures
                if not (
                    self.config.truncation_rescue
                    and exc.completion is not None
                    and _is_truncated_empty_completion(
                        exc.completion, self.max_completion_tokens
                    )
                ):
                    raise
                native = None

            native_truncated = bool(
                native is None
                or _is_truncated_empty_completion(
                    native.completion, self.max_completion_tokens
                )
            )
            if native is not None:
                record_native_call(
                    native.completion,
                    calls=native.call_count,
                    parse_failures=native.parse_failures,
                    analysis_text=native.analysis_text,
                    effort=reasoning_effort,
                )
            if self.config.truncation_rescue and native_truncated:
                state.truncation_rescue_fires += 1
                tally.truncation_rescue_fires += 1
                ctx_logger.warning(
                    "Adaptive-minimal native Harmony truncation rescue fired",
                    initial_cap=self.max_completion_tokens,
                    finish_reason=(
                        native.completion.finish_reason
                        if native is not None
                        else "length"
                    ),
                    fires_episode=state.truncation_rescue_fires,
                )
                high_mode = bool(
                    self.config.executor_effort_high
                    or (
                        self.config.struggle_effort
                        and state.struggle_effort_escalated
                    )
                )
                rescue_attempts = (
                    [(TRUNCATION_RESCUE_CAPS[-1], "high"),
                     (TRUNCATION_RESCUE_CAPS[-1], "medium")]
                    if self.config.rescue_quality and high_mode
                    else [(TRUNCATION_RESCUE_CAPS[0], reasoning_effort),
                          (TRUNCATION_RESCUE_CAPS[-1], "medium")]
                )
                last_parse_error: HarmonyNativeParseError | None = None
                for attempt, (cap, retry_effort) in enumerate(
                    rescue_attempts, start=1
                ):
                    try:
                        retry = native_attempt(cap, retry_effort)
                    except HarmonyNativeParseError as exc:
                        last_parse_error = exc
                        if exc.completion is not None:
                            record_native_call(
                                exc.completion,
                                calls=exc.call_count,
                                parse_failures=exc.parse_failures,
                                analysis_text=None,
                                effort=retry_effort,
                            )
                        else:
                            tally.parse_failures += exc.parse_failures
                        if not (
                            exc.completion is not None
                            and _is_truncated_empty_completion(
                                exc.completion, cap
                            )
                        ):
                            raise
                        ctx_logger.info(
                            "Adaptive-minimal native Harmony rescue attempt",
                            attempt=attempt,
                            max_completion_tokens=cap,
                            reasoning_effort=retry_effort,
                            finish_reason="length",
                            usable=False,
                        )
                        continue
                    record_native_call(
                        retry.completion,
                        calls=retry.call_count,
                        parse_failures=retry.parse_failures,
                        analysis_text=retry.analysis_text,
                        effort=retry_effort,
                    )
                    still_truncated = _is_truncated_empty_completion(
                        retry.completion, cap
                    )
                    ctx_logger.info(
                        "Adaptive-minimal native Harmony rescue attempt",
                        attempt=attempt,
                        max_completion_tokens=cap,
                        reasoning_effort=retry_effort,
                        finish_reason=retry.completion.finish_reason,
                        usable=not still_truncated,
                    )
                    if not still_truncated:
                        native = retry
                        break
                else:
                    if last_parse_error is not None:
                        raise last_parse_error
                    raise MalformedModelResponseError(
                        "native Harmony completion remained truncated after rescue"
                    )
            if native is None:
                raise MalformedModelResponseError(
                    "native Harmony completion was unavailable after rescue"
                )
            return native.action

        request_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.micro_prompt},
                {"role": "user", "content": prompt},
            ],
            response_schema=(
                TERMINAL_RESPOND_OUTPUT_SCHEMA
                if terminal_draft is not None
                else NEXT_ACTION_OUTPUT_SCHEMA
            ),
            response_schema_name=(
                "terminal_respond"
                if terminal_draft is not None
                else "next_action"
            ),
            temperature=self.temperature,
        )
        result = self.client.generate(
            **request_kwargs,
            max_completion_tokens=self.max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        tally.add(result)
        _record_reasoning_effort_call(
            reasoning_effort, state=state, tally=tally
        )
        self._record_struggle_effort_call(
            state=state,
            reasoning_effort=reasoning_effort,
            duration_ms=float(result.duration_ms),
        )
        malformed_arguments = _completion_has_malformed_tool_arguments(result.text)
        truncated_empty = _is_truncated_empty_completion(
            result, self.max_completion_tokens
        )
        if self.config.truncation_rescue and (
            truncated_empty or malformed_arguments
        ):
            state.truncation_rescue_fires += 1
            tally.truncation_rescue_fires += 1
            if malformed_arguments:
                state.malformed_argument_rescue_fires += 1
                tally.malformed_argument_rescue_fires += 1
            self._activate_struggle_effort(
                state,
                (
                    "malformed_argument"
                    if malformed_arguments
                    else "truncation_rescue"
                ),
                ctx_logger,
            )
            ctx_logger.warning(
                "Adaptive-minimal truncation rescue fired",
                classifier=(
                    "malformed_tool_arguments"
                    if malformed_arguments
                    else "truncated_empty"
                ),
                initial_cap=self.max_completion_tokens,
                finish_reason=result.finish_reason,
                thinking_tokens=(
                    result.token_usage.reasoning_output_tokens
                    if result.token_usage is not None
                    else 0
                ),
                fires_episode=state.truncation_rescue_fires,
            )
            if self.config.rescue_quality:
                high_rescue_mode = bool(
                    self.config.executor_effort_high
                    or (
                        self.config.struggle_effort
                        and state.struggle_effort_escalated
                    )
                )
                rescue_attempts = (
                    [
                        (
                            self.max_completion_tokens,
                            "high" if high_rescue_mode else reasoning_effort,
                        )
                    ]
                    if malformed_arguments
                    else []
                )
                if high_rescue_mode:
                    # p3i61: an 8192-high initial call retries that same cap
                    # once at high, then once at medium. A malformed result
                    # already seeded the same-cap high retry above.
                    if self.config.initial_cap_8192:
                        if not rescue_attempts:
                            rescue_attempts.append(
                                (TRUNCATION_RESCUE_CAPS[-1], "high")
                            )
                        rescue_attempts.append(
                            (TRUNCATION_RESCUE_CAPS[-1], "medium")
                        )
                    else:
                        rescue_attempts.extend(
                            [
                                (TRUNCATION_RESCUE_CAPS[-1], "high"),
                                (TRUNCATION_RESCUE_CAPS[-1], "medium"),
                            ]
                        )
                else:
                    if TRUNCATION_RESCUE_CAPS[0] > self.max_completion_tokens:
                        rescue_attempts.append(
                            (TRUNCATION_RESCUE_CAPS[0], reasoning_effort)
                        )
                    if TRUNCATION_RESCUE_CAPS[1] > self.max_completion_tokens:
                        rescue_attempts.append(
                            (TRUNCATION_RESCUE_CAPS[1], "medium")
                        )
            else:
                rescue_caps = (
                    (self.max_completion_tokens, *TRUNCATION_RESCUE_CAPS)
                    if malformed_arguments
                    else TRUNCATION_RESCUE_CAPS
                )
                rescue_attempts = [
                    (
                        cap,
                        "low" if cap == TRUNCATION_RESCUE_CAPS[-1] else reasoning_effort,
                    )
                    for cap in rescue_caps
                ]
            for attempt, (cap, retry_effort) in enumerate(
                rescue_attempts, start=1
            ):
                retry = self.client.generate(
                    **request_kwargs,
                    max_completion_tokens=cap,
                    reasoning_effort=retry_effort,
                )
                tally.add(retry)
                _record_reasoning_effort_call(
                    retry_effort, state=state, tally=tally
                )
                self._record_struggle_effort_call(
                    state=state,
                    reasoning_effort=retry_effort,
                    duration_ms=float(retry.duration_ms),
                )
                ctx_logger.info(
                    "Adaptive-minimal truncation rescue attempt",
                    attempt=attempt,
                    max_completion_tokens=cap,
                    reasoning_effort=retry_effort,
                    finish_reason=retry.finish_reason,
                    usable=_completion_has_usable_action(retry.text),
                )
                result = retry
                if (
                    self.config.rescue_quality
                    and not self.config.executor_effort_high
                    and not (
                        self.config.struggle_effort
                        and state.struggle_effort_escalated
                    )
                    and cap == TRUNCATION_RESCUE_CAPS[-1]
                    and retry_effort == "medium"
                    and _is_truncated_empty_completion(retry, cap)
                ):
                    # p3i46 L4: preserve reasoning quality at 8192. Only an
                    # actual second-escalation truncation earns a final low-
                    # effort retry at the same generous cap.
                    rescue_attempts.append((cap, "low"))
                if (
                    _completion_has_usable_action(retry.text)
                    and not _is_truncated_empty_completion(retry, cap)
                    and not _completion_has_malformed_tool_arguments(retry.text)
                ):
                    break
            if _completion_has_malformed_tool_arguments(result.text):
                exhausted = MalformedModelResponseError(
                    "tool-call arguments remained invalid after rescue"
                )
                exhausted.rescue_exhausted = True
                raise exhausted
        if not result.text or not result.text.strip():
            raise MalformedModelResponseError("empty completion")
        return parse_next_action(result.text)

    def _apply_mutation_consensus(
        self,
        *,
        original_action: dict[str, Any],
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        executor_context: tuple[
            InternalFault | None, FewShotSelection | None, str, str | None
        ],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any]:
        """Sample two guarded mutation drafts and select an exact 2-of-3 mode.

        Mutation calls are irreversible scoring forks: an incorrect call remains
        penalized even if a later action corrects it.  The extra samples use the
        exact executor context that produced the original draft.  Each losing
        candidate is checked against a deep-copied episode state, so it cannot
        consume an ask budget, alter a read ledger, or execute a guard side
        effect.  Only the selected decision reaches the real execution path.
        """

        fault, event_selection, reasoning_effort, terminal_draft = executor_context
        state.mutation_consensus_invocations += 1
        tally.mutation_consensus_invocations += 1
        calls_before = tally.calls
        duration_before = tally.duration_ms
        raw_candidates: list[dict[str, Any] | None] = [original_action]
        guarded_candidates: list[dict[str, Any] | None] = [original_action]
        candidate_faults: list[str | None] = [None]

        sample_reasoning_efforts: list[str] = []
        for sample_index in range(2):
            sample_reasoning_effort = (
                ("high", "medium")[sample_index]
                if self.config.consensus_mixed_effort
                else reasoning_effort
            )
            sample_reasoning_efforts.append(sample_reasoning_effort)
            try:
                sampled = self._call_executor(
                    state=state,
                    messages=messages,
                    tools=tools,
                    fault=fault,
                    event_selection=event_selection,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    reasoning_effort=sample_reasoning_effort,
                    terminal_draft=terminal_draft,
                )
            except (
                MalformedModelResponseError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                raw_candidates.append(None)
                guarded_candidates.append(None)
                candidate_faults.append(f"malformed sample {sample_index + 1}: {exc}")
                continue
            raw_candidates.append(sampled)
            guarded, guard_fault = _guard_mutation_consensus_candidate(
                sampled,
                state=state,
                messages=messages,
                config=self.config,
            )
            guarded_candidates.append(guarded)
            candidate_faults.append(guard_fault)

        selected, agreed, overridden, signatures = _select_mutation_consensus(
            guarded_candidates
        )
        judge_outcome = "not_enabled"
        if not agreed and self.config.llm_consensus_judge:
            if (
                state.llm_consensus_judge_calls
                >= LLM_CONSENSUS_JUDGE_EPISODE_CAP
            ):
                state.llm_consensus_judge_budget_suppressed += 1
                tally.llm_consensus_judge_budget_suppressed += 1
                judge_outcome = "budget_suppressed"
            else:
                (
                    selected,
                    agreed,
                    overridden,
                    judge_outcome,
                ) = self._apply_llm_consensus_judge(
                    raw_candidates=raw_candidates,
                    guarded_candidates=guarded_candidates,
                    state=state,
                    tally=tally,
                    ctx_logger=ctx_logger,
                )
        deepened = False
        deep_majority = False
        deep_extra_calls = 0
        deep_added_latency_ms = 0.0
        if not agreed and self.config.consensus_deepen:
            deepened = True
            state.mutation_consensus_deepenings += 1
            tally.mutation_consensus_deepenings += 1
            deep_calls_before = tally.calls
            deep_duration_before = tally.duration_ms
            for sample_index in range(2, 4):
                try:
                    sampled = self._call_executor(
                        state=state,
                        messages=messages,
                        tools=tools,
                        fault=fault,
                        event_selection=event_selection,
                        tally=tally,
                        ctx_logger=ctx_logger,
                        reasoning_effort=reasoning_effort,
                        terminal_draft=terminal_draft,
                    )
                except (
                    MalformedModelResponseError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    raw_candidates.append(None)
                    guarded_candidates.append(None)
                    candidate_faults.append(
                        f"malformed sample {sample_index + 1}: {exc}"
                    )
                    continue
                raw_candidates.append(sampled)
                guarded, guard_fault = _guard_mutation_consensus_candidate(
                    sampled,
                    state=state,
                    messages=messages,
                    config=self.config,
                )
                guarded_candidates.append(guarded)
                candidate_faults.append(guard_fault)
            deep_extra_calls = tally.calls - deep_calls_before
            deep_added_latency_ms = tally.duration_ms - deep_duration_before
            state.mutation_consensus_deep_extra_calls += deep_extra_calls
            tally.mutation_consensus_deep_extra_calls += deep_extra_calls
            selected, agreed, overridden, signatures = (
                _select_deepened_mutation_consensus(guarded_candidates)
            )
            deep_majority = agreed

        extra_calls = tally.calls - calls_before
        added_latency_ms = tally.duration_ms - duration_before
        state.mutation_consensus_extra_calls += extra_calls
        state.mutation_consensus_added_latency_ms += added_latency_ms
        tally.mutation_consensus_extra_calls += extra_calls
        tally.mutation_consensus_added_latency_ms += added_latency_ms

        selected_index = next(
            index
            for index, candidate in enumerate(guarded_candidates)
            if candidate is selected
        )
        if overridden:
            raw_selected = raw_candidates[selected_index]
            assert raw_selected is not None
            selected = self._apply_consensus_winner_guards(
                raw_selected,
                state=state,
                messages=messages,
                tally=tally,
                ctx_logger=ctx_logger,
            )
        if agreed:
            state.mutation_consensus_majority_agreements += 1
            tally.mutation_consensus_majority_agreements += 1
            if deep_majority:
                state.mutation_consensus_deep_majorities += 1
                tally.mutation_consensus_deep_majorities += 1
            if overridden:
                state.mutation_consensus_majority_overrides += 1
                tally.mutation_consensus_majority_overrides += 1
                if deep_majority:
                    state.mutation_consensus_deep_overrides += 1
                    tally.mutation_consensus_deep_overrides += 1
        else:
            state.mutation_consensus_no_majority_fallbacks += 1
            tally.mutation_consensus_no_majority_fallbacks += 1
            if deepened:
                state.mutation_consensus_still_no_majority += 1
                tally.mutation_consensus_still_no_majority += 1
            self._activate_struggle_effort(
                state, "consensus_no_majority", ctx_logger
            )

        ctx_logger.info(
            "Adaptive-minimal mutation-point consensus completed",
            invocation_episode=state.mutation_consensus_invocations,
            majority_agreement=agreed,
            majority_override=overridden,
            no_majority_fallback=not agreed,
            deepened=deepened,
            deep_majority=deep_majority,
            deep_override=deep_majority and overridden,
            still_no_majority=deepened and not agreed,
            deep_extra_llm_calls=deep_extra_calls,
            deep_added_latency_ms=round(deep_added_latency_ms, 3),
            extra_llm_calls=extra_calls,
            added_latency_ms=round(added_latency_ms, 3),
            sampling_mode="sequential",
            sample_reasoning_efforts=sample_reasoning_efforts,
            decision_signatures=signatures,
            candidate_guard_faults=candidate_faults,
            llm_consensus_judge_outcome=judge_outcome,
        )
        return selected

    def _apply_llm_consensus_judge(
        self,
        *,
        raw_candidates: list[dict[str, Any] | None],
        guarded_candidates: list[dict[str, Any] | None],
        state: _EpisodeState,
        tally: _CallTally,
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], bool, bool, str]:
        """Try one bounded semantic vote after exact consensus has failed.

        The classifier receives only the three drafted call sets. It cannot
        create or edit a decision. A strict, validated 2-of-3 grouping selects
        the earliest existing draft in that group; every error fails open to
        the original draft.
        """

        original = guarded_candidates[0]
        if original is None:
            raise ValueError("mutation consensus original cannot be rejected")

        state.llm_consensus_judge_calls += 1
        tally.llm_consensus_judge_calls += 1
        started = time.perf_counter()
        try:
            result = self.client.generate(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify three drafted tool-call decisions by "
                            "semantic equivalence. Two drafts are equivalent "
                            "only when they use the same ordered tools and have "
                            "the same intended effect. Pure formatting or "
                            "serialization differences such as string versus "
                            "number, a percent sign, case, or key order are "
                            "equivalent. Different target entities or different "
                            "values are not equivalent. Return a partition of "
                            "candidate indices 0, 1, and 2 in strict JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "candidates": [
                                    _llm_consensus_judge_candidate(candidate)
                                    for candidate in raw_candidates
                                ]
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                response_schema=LLM_CONSENSUS_JUDGE_SCHEMA,
                response_schema_name="mutation_semantic_groups",
                max_completion_tokens=LLM_CONSENSUS_JUDGE_COMPLETION_CAP,
                temperature=0.0,
                reasoning_effort="low",
                request_timeout_seconds=LLM_CONSENSUS_JUDGE_TIMEOUT_SECONDS,
                fail_fast=True,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            tally.calls += 1
            tally.duration_ms += elapsed_ms
            state.llm_consensus_judge_added_latency_ms += elapsed_ms
            tally.llm_consensus_judge_added_latency_ms += elapsed_ms
            state.llm_consensus_judge_errors += 1
            tally.llm_consensus_judge_errors += 1
            timeout = _is_timeout_error(exc)
            if timeout:
                state.llm_consensus_judge_timeouts += 1
                tally.llm_consensus_judge_timeouts += 1
            ctx_logger.info(
                "Adaptive-minimal LLM consensus judge failed open",
                outcome="timeout" if timeout else "error",
                exception_type=type(exc).__name__,
                calls_episode=state.llm_consensus_judge_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return original, False, False, "timeout" if timeout else "error"

        tally.add(result)
        elapsed_ms = float(result.duration_ms)
        state.llm_consensus_judge_added_latency_ms += elapsed_ms
        tally.llm_consensus_judge_added_latency_ms += elapsed_ms
        try:
            selected_index, majority_indices = _parse_llm_consensus_judgment(
                result.text, raw_candidates
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            state.llm_consensus_judge_malformed += 1
            tally.llm_consensus_judge_malformed += 1
            ctx_logger.info(
                "Adaptive-minimal LLM consensus judge failed open",
                outcome="malformed",
                exception_type=type(exc).__name__,
                calls_episode=state.llm_consensus_judge_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return original, False, False, "malformed"

        if selected_index is None:
            state.llm_consensus_judge_no_majority += 1
            tally.llm_consensus_judge_no_majority += 1
            ctx_logger.info(
                "Adaptive-minimal LLM consensus judge failed open",
                outcome="no_majority",
                calls_episode=state.llm_consensus_judge_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return original, False, False, "no_majority"

        selected = guarded_candidates[selected_index]
        if selected is None:
            state.llm_consensus_judge_malformed += 1
            tally.llm_consensus_judge_malformed += 1
            ctx_logger.info(
                "Adaptive-minimal LLM consensus judge failed open",
                outcome="selected_guard_rejected",
                selected_index=selected_index,
                calls_episode=state.llm_consensus_judge_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return original, False, False, "selected_guard_rejected"
        overridden = selected_index != 0
        state.llm_consensus_judge_majorities += 1
        tally.llm_consensus_judge_majorities += 1
        if overridden:
            state.llm_consensus_judge_overrides += 1
            tally.llm_consensus_judge_overrides += 1
        ctx_logger.info(
            "Adaptive-minimal LLM consensus judge selected majority",
            outcome="semantic_majority",
            majority_indices=list(majority_indices),
            selected_index=selected_index,
            override=overridden,
            calls_episode=state.llm_consensus_judge_calls,
            added_latency_ms=round(elapsed_ms, 3),
        )
        return selected, True, overridden, "semantic_majority"

    def _apply_llm_ask_triage(
        self,
        *,
        action: dict[str, Any],
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> tuple[dict[str, Any] | None, str]:
        """Classify one model-authored ask and fail open unless a read is valid."""

        if state.llm_ask_triage_calls >= LLM_ASK_TRIAGE_EPISODE_CAP:
            state.llm_ask_triage_budget_suppressed += 1
            tally.llm_ask_triage_budget_suppressed += 1
            ctx_logger.info(
                "Adaptive-minimal LLM ask triage passed ask through",
                outcome="budget_suppressed",
                calls_episode=state.llm_ask_triage_calls,
            )
            return None, "budget_suppressed"

        state.llm_ask_triage_calls += 1
        tally.llm_ask_triage_calls += 1
        tool_descriptions = [
            {
                "name": str(tool.get("function", {}).get("name") or ""),
                "description": str(
                    tool.get("function", {}).get("description") or ""
                ),
            }
            for tool in tools
            if str(tool.get("function", {}).get("name") or "")
        ]
        started = time.perf_counter()
        try:
            result = self.client.generate(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the assistant's drafted clarification "
                            "question using only the dialogue and available "
                            "tool descriptions. Use RESOLVABLE_BY_READ only "
                            "when one available read-only get_* call can resolve "
                            "the question internally; provide that exact call. "
                            "Use GENUINE_AMBIGUITY when user intent really must "
                            "be clarified. Use NO_AMBIGUITY when the draft is not "
                            "a genuine clarification need. Never propose a "
                            "mutation. For non-read labels, return an empty tool "
                            "name and '{}' arguments."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "dialogue": _messages_for_prompt(messages),
                                "drafted_ask": str(action.get("content") or ""),
                                "available_tools": tool_descriptions,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                response_schema=LLM_ASK_TRIAGE_SCHEMA,
                response_schema_name="ask_triage",
                max_completion_tokens=LLM_ASK_TRIAGE_COMPLETION_CAP,
                temperature=0.0,
                reasoning_effort="low",
                request_timeout_seconds=LLM_ASK_TRIAGE_TIMEOUT_SECONDS,
                fail_fast=True,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            tally.calls += 1
            tally.duration_ms += elapsed_ms
            state.llm_ask_triage_added_latency_ms += elapsed_ms
            tally.llm_ask_triage_added_latency_ms += elapsed_ms
            state.llm_ask_triage_errors += 1
            tally.llm_ask_triage_errors += 1
            timeout = _is_timeout_error(exc)
            if timeout:
                state.llm_ask_triage_timeouts += 1
                tally.llm_ask_triage_timeouts += 1
            outcome = "timeout" if timeout else "error"
            ctx_logger.info(
                "Adaptive-minimal LLM ask triage passed ask through",
                outcome=outcome,
                exception_type=type(exc).__name__,
                calls_episode=state.llm_ask_triage_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return None, outcome

        tally.add(result)
        elapsed_ms = float(result.duration_ms)
        state.llm_ask_triage_added_latency_ms += elapsed_ms
        tally.llm_ask_triage_added_latency_ms += elapsed_ms
        try:
            label, tool_name, arguments = _parse_llm_ask_triage_payload(
                result.text
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            state.llm_ask_triage_malformed += 1
            tally.llm_ask_triage_malformed += 1
            ctx_logger.info(
                "Adaptive-minimal LLM ask triage passed ask through",
                outcome="malformed",
                exception_type=type(exc).__name__,
                calls_episode=state.llm_ask_triage_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return None, "malformed"

        if label == "GENUINE_AMBIGUITY":
            state.llm_ask_triage_genuine += 1
            tally.llm_ask_triage_genuine += 1
            outcome = "genuine_ambiguity"
        elif label == "NO_AMBIGUITY":
            state.llm_ask_triage_no_ambiguity += 1
            tally.llm_ask_triage_no_ambiguity += 1
            outcome = "no_ambiguity"
        else:
            state.llm_ask_triage_resolvable += 1
            tally.llm_ask_triage_resolvable += 1
            outcome = "resolvable_by_read"
            proposed = {
                "action": "tool_calls",
                "tool_calls": [
                    {"tool_name": tool_name, "arguments": arguments}
                ],
            }
            tool = state.tools_by_name.get(tool_name)
            invalid = (
                tool is None
                or not tool_name.casefold().startswith("get_")
                or _schema_preflight_fault(proposed, state.tools_by_name)
                is not None
                or (
                    not arguments
                    and tool is not None
                    and not _semantic_prefetch_call_valid(tool, arguments)
                )
            )
            if invalid:
                state.llm_ask_triage_invalid += 1
                tally.llm_ask_triage_invalid += 1
                ctx_logger.info(
                    "Adaptive-minimal LLM ask triage passed ask through",
                    outcome="invalid_read_proposal",
                    calls_episode=state.llm_ask_triage_calls,
                    added_latency_ms=round(elapsed_ms, 3),
                )
                return None, "invalid_read_proposal"
            ctx_logger.info(
                "Adaptive-minimal LLM ask triage classified draft",
                outcome=outcome,
                calls_episode=state.llm_ask_triage_calls,
                added_latency_ms=round(elapsed_ms, 3),
            )
            return proposed, outcome

        ctx_logger.info(
            "Adaptive-minimal LLM ask triage passed ask through",
            outcome=outcome,
            calls_episode=state.llm_ask_triage_calls,
            added_latency_ms=round(elapsed_ms, 3),
        )
        return None, outcome

    def _apply_llm_limitation_classifier(
        self,
        *,
        action: dict[str, Any],
        evidence: str,
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> InternalFault | None:
        """Classify one unavailable-capability fork; every failure fails open."""

        if state.llm_limitation_classifier_calls >= 1:
            return None
        state.llm_limitation_classifier_calls += 1
        tally.llm_limitation_classifier_calls += 1
        tools = [
            {
                "name": name,
                "description": str(
                    tool.get("function", {}).get("description") or ""
                ),
            }
            for name, tool in sorted(state.tools_by_name.items())
        ]
        started = time.perf_counter()
        try:
            result = self.client.generate(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Decide whether the user's requested capability is "
                            "actually unavailable from the supplied tool catalog "
                            "and evidence. Use CAPABILITY_UNAVAILABLE_TERMINATE "
                            "only when no schema-valid available call can satisfy "
                            "the request. Otherwise use "
                            "CAPABILITY_AVAILABLE_CONTINUE and name the exact "
                            "schema-valid call. Return strict JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_request": _all_user_text(messages),
                                "draft": action,
                                "available_tools": tools,
                                "unavailability_evidence": evidence,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                response_schema=LLM_LIMITATION_CLASSIFIER_SCHEMA,
                response_schema_name="limitation_classifier",
                max_completion_tokens=LLM_LIMITATION_CLASSIFIER_COMPLETION_CAP,
                temperature=0.0,
                reasoning_effort="low",
                request_timeout_seconds=LLM_LIMITATION_CLASSIFIER_TIMEOUT_SECONDS,
                fail_fast=True,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            tally.calls += 1
            tally.duration_ms += elapsed_ms
            state.llm_limitation_classifier_added_latency_ms += elapsed_ms
            tally.llm_limitation_classifier_added_latency_ms += elapsed_ms
            state.llm_limitation_classifier_errors += 1
            tally.llm_limitation_classifier_errors += 1
            timeout = _is_timeout_error(exc)
            if timeout:
                state.llm_limitation_classifier_timeouts += 1
                tally.llm_limitation_classifier_timeouts += 1
            ctx_logger.info(
                "Adaptive-minimal LLM limitation classifier failed open",
                outcome="timeout" if timeout else "error",
            )
            return None
        tally.add(result)
        elapsed_ms = float(result.duration_ms)
        state.llm_limitation_classifier_added_latency_ms += elapsed_ms
        tally.llm_limitation_classifier_added_latency_ms += elapsed_ms
        try:
            label, tool_name, arguments, finding = _parse_llm_limitation_payload(
                result.text
            )
            if label == "CAPABILITY_AVAILABLE_CONTINUE":
                candidate = {
                    "action": "tool_calls",
                    "tool_calls": [
                        {"tool_name": tool_name, "arguments": arguments}
                    ],
                }
                if (
                    validate_next_action(candidate, state.tools_by_name) is not None
                    or _schema_preflight_fault(candidate, state.tools_by_name)
                    is not None
                ):
                    raise ValueError("classifier proposed a schema-invalid call")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            state.llm_limitation_classifier_malformed += 1
            tally.llm_limitation_classifier_malformed += 1
            ctx_logger.info(
                "Adaptive-minimal LLM limitation classifier failed open",
                outcome="malformed",
                exception_type=type(exc).__name__,
            )
            return None
        if label == "CAPABILITY_AVAILABLE_CONTINUE":
            state.llm_limitation_classifier_continues += 1
            tally.llm_limitation_classifier_continues += 1
            ctx_logger.info(
                "Adaptive-minimal LLM limitation classifier continued",
                proposed_tool=tool_name,
            )
            return None
        state.llm_limitation_classifier_terminates += 1
        tally.llm_limitation_classifier_terminates += 1
        ctx_logger.info(
            "Adaptive-minimal LLM limitation classifier scheduled re-decision",
            finding=finding,
        )
        return InternalFault(
            "llm_limitation_classifier",
            "A low-effort catalog classifier found the requested capability "
            "unavailable: " + finding + " Re-draft exactly once. Write your "
            "own grounded limitation acknowledgment; do not retry substitute "
            "calls and do not assume a forced conversation ending.",
        )

    def _apply_consensus_winner_guards(
        self,
        action: dict[str, Any],
        *,
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any]:
        """Commit guard effects for an extra sample selected by the majority."""

        if self.config.repeated_read_breaker:
            action, blocked_reads = _split_repeated_get_calls(action, state)
            if blocked_reads:
                state.repeated_read_blocks += blocked_reads
                tally.repeated_read_blocks += blocked_reads
                ctx_logger.info(
                    "Adaptive-minimal repeated-read loop breaker blocked calls",
                    blocked=blocked_reads,
                    blocks_episode=state.repeated_read_blocks,
                    route="mutation_consensus_winner",
                )
            assert action is not None
        if self.config.placeholder_guard:
            action, placeholder_index = _split_calls_before_placeholder(action)
            if placeholder_index is not None:
                state.placeholder_guard_fires += 1
                tally.placeholder_guard_fires += 1
                ctx_logger.info(
                    "Adaptive-minimal placeholder guard fired",
                    first_placeholder_call_index=placeholder_index,
                    concrete_prefix_calls=len(action.get("tool_calls") or []),
                    fires_episode=state.placeholder_guard_fires,
                    route="mutation_consensus_winner",
                )
            assert action is not None
        action = _apply_pre_mutation_guards(
            action,
            state=state,
            messages=messages,
            config=self.config,
            tally=tally,
            ctx_logger=ctx_logger,
        )
        if self.config.event_exemplars:
            action, skipped = _skip_satisfied_mutations(action, state)
            if skipped:
                tally.event_e4_skips += skipped
                ctx_logger.info(
                    "Adaptive-minimal EVENTv2 E4 mutation skipped",
                    skipped=skipped,
                    skips_episode=state.event_e4_skips,
                    route="mutation_consensus_winner",
                )
        return action

    def _apply_terminal_consensus(
        self,
        *,
        original_action: dict[str, Any],
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        executor_context: tuple[
            InternalFault | None, FewShotSelection | None, str, str | None
        ],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any]:
        """Expand a terminal draft and override only for an exact action mode.

        Respond text is deliberately never voted on: a respond majority keeps
        the original text.  Only two identical guarded call-set signatures can
        replace the original response, targeting modal under-execution without
        introducing a content-selection mechanism.
        """

        fault, event_selection, _reasoning_effort, terminal_draft = (
            executor_context
        )
        state.terminal_consensus_invocations += 1
        tally.terminal_consensus_invocations += 1
        calls_before = tally.calls
        duration_before = tally.duration_ms
        raw_candidates: list[dict[str, Any] | None] = [original_action]
        guarded_candidates: list[dict[str, Any] | None] = [original_action]
        candidate_faults: list[str | None] = [None]

        for sample_index in range(2):
            try:
                sampled = self._call_executor(
                    state=state,
                    messages=messages,
                    tools=tools,
                    fault=fault,
                    event_selection=event_selection,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    reasoning_effort=EXECUTOR_REASONING_EFFORT,
                    terminal_draft=terminal_draft,
                )
            except (
                MalformedModelResponseError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                raw_candidates.append(None)
                guarded_candidates.append(None)
                candidate_faults.append(
                    f"malformed sample {sample_index + 1}: {exc}"
                )
                continue
            raw_candidates.append(sampled)
            guarded, guard_fault = _guard_mutation_consensus_candidate(
                sampled,
                state=state,
                messages=messages,
                config=self.config,
            )
            guarded_candidates.append(guarded)
            candidate_faults.append(guard_fault)

        extra_calls = tally.calls - calls_before
        added_latency_ms = tally.duration_ms - duration_before
        state.terminal_consensus_extra_calls += extra_calls
        state.terminal_consensus_added_latency_ms += added_latency_ms
        tally.terminal_consensus_extra_calls += extra_calls
        tally.terminal_consensus_added_latency_ms += added_latency_ms

        selected, outcome, signatures = _select_terminal_consensus(
            guarded_candidates
        )
        if outcome == "respond_majority":
            state.terminal_consensus_respond_majorities += 1
            tally.terminal_consensus_respond_majorities += 1
        elif outcome == "action_majority":
            state.terminal_consensus_action_majorities += 1
            state.terminal_consensus_action_overrides += 1
            tally.terminal_consensus_action_majorities += 1
            tally.terminal_consensus_action_overrides += 1
            selected_index = next(
                index
                for index, candidate in enumerate(guarded_candidates)
                if candidate is selected
            )
            raw_selected = raw_candidates[selected_index]
            assert raw_selected is not None
            selected = self._apply_consensus_winner_guards(
                raw_selected,
                state=state,
                messages=messages,
                tally=tally,
                ctx_logger=ctx_logger,
            )
        else:
            state.terminal_consensus_no_majority_fallbacks += 1
            tally.terminal_consensus_no_majority_fallbacks += 1

        ctx_logger.info(
            "Adaptive-minimal terminal-respond consensus completed",
            invocation_episode=state.terminal_consensus_invocations,
            outcome=outcome,
            action_override=outcome == "action_majority",
            extra_llm_calls=extra_calls,
            added_latency_ms=round(added_latency_ms, 3),
            sampling_mode="sequential",
            decision_signatures=signatures,
            candidate_guard_faults=candidate_faults,
        )
        return selected

    def _apply_ask_content_consensus(
        self,
        *,
        original_action: dict[str, Any],
        state: _EpisodeState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        executor_context: tuple[
            InternalFault | None, FewShotSelection | None, str, str | None
        ],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any]:
        """Vote only among clarification asks; never change decision shape.

        Both extra samples must themselves be asks before any content vote is
        eligible. Actions and non-question responses are ignored, so this
        mechanism is structurally unable to convert an ask into another action
        class. A 2-of-3 normalized slot majority selects the first ask bearing
        that signature; every other case fails open to the original ask.
        """

        fault, event_selection, reasoning_effort, terminal_draft = executor_context
        state.ask_content_consensus_invocations += 1
        tally.ask_content_consensus_invocations += 1
        calls_before = tally.calls
        duration_before = tally.duration_ms
        candidates: list[dict[str, Any] | None] = [original_action]
        sample_is_ask: list[bool] = []
        sample_errors: list[str | None] = []
        for sample_index in range(2):
            try:
                sampled = self._call_executor(
                    state=state,
                    messages=messages,
                    tools=tools,
                    fault=fault,
                    event_selection=event_selection,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    reasoning_effort=reasoning_effort,
                    terminal_draft=terminal_draft,
                )
            except (
                MalformedModelResponseError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                candidates.append(None)
                sample_is_ask.append(False)
                sample_errors.append(
                    f"malformed sample {sample_index + 1}: {exc}"
                )
                continue
            candidates.append(sampled)
            sample_is_ask.append(_is_clarification_question(sampled))
            sample_errors.append(None)

        extra_calls = tally.calls - calls_before
        added_latency_ms = tally.duration_ms - duration_before
        state.ask_content_consensus_extra_calls += extra_calls
        state.ask_content_consensus_added_latency_ms += added_latency_ms
        tally.ask_content_consensus_extra_calls += extra_calls
        tally.ask_content_consensus_added_latency_ms += added_latency_ms

        selected, majority, signatures = _select_ask_content_consensus(
            candidates,
            tools_by_name=state.tools_by_name,
        )
        if majority:
            state.ask_content_consensus_majority_selections += 1
            tally.ask_content_consensus_majority_selections += 1
            outcome = "majority_selection"
        else:
            state.ask_content_consensus_fallbacks += 1
            tally.ask_content_consensus_fallbacks += 1
            outcome = "original_fallback"

        # The selector's type contract is a second hard boundary around the
        # never-convert law, independent of provider output or slot parsing.
        assert _is_clarification_question(selected)
        ctx_logger.info(
            "Adaptive-minimal ask-content consensus completed",
            invocation_episode=state.ask_content_consensus_invocations,
            outcome=outcome,
            majority_selection=majority,
            sample_asks=sum(sample_is_ask),
            extra_llm_calls=extra_calls,
            added_latency_ms=round(added_latency_ms, 3),
            slot_signatures=signatures,
            sample_errors=sample_errors,
        )
        return selected

    def _can_escalate(self, state: _EpisodeState) -> bool:
        return state.escalations_fired < self.config.escalation_budget

    @staticmethod
    def _record_escalation(
        state: _EpisodeState,
        fault: InternalFault,
        ctx_logger: Any,
    ) -> str:
        state.escalations_fired += 1
        state.escalation_counts[fault.signal] += 1
        if fault.signal == SIGNAL_TURN_GUARD:
            state.turn_guard_fired = True
        elif fault.signal == SIGNAL_UNAVAILABILITY_LOOP:
            state.unavailability_loop_fired = True
        elif fault.signal == SIGNAL_GROUNDED_RESPOND:
            state.grounded_respond_fired = True
        ctx_logger.info(
            "Adaptive-minimal escalation fired",
            signal=fault.signal,
            fault=fault.text,
            episode_count=state.escalations_fired,
        )
        return fault.signal

    def _decision_meta(
        self,
        *,
        state: _EpisodeState,
        route: str,
        calls: int,
        injected_reads: int,
        escalation_signal: str | None,
        disclosure_fastest_fixes: int,
    ) -> dict[str, Any]:
        meta = {
            "route": route,
            "total_calls": calls,
            "injected_reads": injected_reads,
            "injected_reads_episode": state.injected_reads,
            "read_ledger": sorted(state.read_ledger),
            "escalation_signal": escalation_signal,
            "escalations_fired_episode": state.escalations_fired,
            "escalations_by_signal_episode": dict(state.escalation_counts),
            "prefetch_candidates": list(state.prefetch_candidates),
            "prefetch_tools_emitted": list(state.prefetch_tools_emitted),
            "prefetch_reads_episode": state.prefetch_reads,
            "prefetch_semantic_emitted_episode": (
                state.prefetch_semantic_emitted
            ),
            "prefetch_semantic_suppressed_episode": (
                state.prefetch_semantic_suppressed
            ),
            "prefetch_error_drops_episode": state.prefetch_error_drops,
            "non_prefetch_tool_calls_executed_episode": (
                state.non_prefetch_tool_calls_executed
            ),
            "turn_guard_fired_episode": state.turn_guard_fired,
            "unavailability_loop_fired_episode": state.unavailability_loop_fired,
            "micro_prompt_tokens": self.micro_prompt_token_count,
            "disclosure_fastest_fixes": disclosure_fastest_fixes,
            "event_exemplar_counts_episode": dict(state.event_exemplar_counts),
            "event_e4_skips_episode": state.event_e4_skips,
            "terminal_readback_fires_episode": state.terminal_readback_fires,
            "terminal_readback_reads_episode": state.terminal_readback_reads,
            "terminal_readback_mismatches_episode": (
                state.terminal_readback_mismatches
            ),
            "terminal_readback_revises_episode": state.terminal_readback_revises,
            "terminal_effort_high_calls_episode": (
                state.terminal_effort_high_calls
            ),
            "terminal_effort_medium_calls_episode": (
                state.terminal_effort_medium_calls
            ),
            "argument_binding_relative_clarifications_episode": (
                state.argument_binding_relative_clarifications
            ),
            "argument_binding_route_corrections_episode": (
                state.argument_binding_route_corrections
            ),
            "disclosure_confirmation_reasks_episode": (
                state.disclosure_confirmation_reasks
            ),
            "disclosure_unavailable_acks_episode": (
                state.disclosure_unavailable_acks
            ),
            "truncation_rescue_fires_episode": state.truncation_rescue_fires,
            "malformed_argument_rescue_fires_episode": (
                state.malformed_argument_rescue_fires
            ),
            "placeholder_guard_fires_episode": state.placeholder_guard_fires,
            "vague_degree_clarifications_episode": (
                state.vague_degree_clarifications
            ),
            "vague_degree_preference_redirects_episode": (
                state.vague_degree_preference_redirects
            ),
            "vague_degree_preference_applies_episode": (
                state.vague_degree_preference_applies
            ),
            "schema_preflight_bounces_episode": state.schema_preflight_bounces,
            "schema_preflight_pass_throughs_episode": (
                state.schema_preflight_pass_throughs
            ),
            "value_p1_context_applies_episode": state.value_p1_context_applies,
            "value_p1_ask_suppressions_episode": (
                state.value_p1_ask_suppressions
            ),
            "value_p2_binding_bounces_episode": state.value_p2_binding_bounces,
            "value_p2_pass_throughs_episode": state.value_p2_pass_throughs,
            "value_p3_preference_reads_episode": state.value_p3_preference_reads,
            "value_p3_fallback_asks_episode": state.value_p3_fallback_asks,
            "value_p4_nav_redirects_episode": state.value_p4_nav_redirects,
            "value_p4_nav_blocks_episode": state.value_p4_nav_blocks,
            "value_p5_occupancy_reads_episode": state.value_p5_occupancy_reads,
            "value_p6_claim_revises_episode": state.value_p6_claim_revises,
            "time_format_revises_episode": state.time_format_revises,
            "injected_asks_episode": state.injected_asks,
            "ask_budget_suppressed_episode": state.ask_budget_suppressed,
            "repeated_read_blocks_episode": state.repeated_read_blocks,
            "route_reference_bounces_episode": state.route_reference_bounces,
            "policy_lint_revises_episode": state.policy_lint_revises,
            "policy_lint_zone_difference_revises_episode": (
                state.policy_lint_zone_difference_revises
            ),
            "policy_lint_temperature_unit_revises_episode": (
                state.policy_lint_temperature_unit_revises
            ),
            "mutation_consensus_invocations_episode": (
                state.mutation_consensus_invocations
            ),
            "mutation_consensus_majority_agreements_episode": (
                state.mutation_consensus_majority_agreements
            ),
            "mutation_consensus_majority_overrides_episode": (
                state.mutation_consensus_majority_overrides
            ),
            "mutation_consensus_no_majority_fallbacks_episode": (
                state.mutation_consensus_no_majority_fallbacks
            ),
            "mutation_consensus_extra_llm_calls_episode": (
                state.mutation_consensus_extra_calls
            ),
            "mutation_consensus_added_latency_ms_episode": round(
                state.mutation_consensus_added_latency_ms, 3
            ),
            "mutation_consensus_deepenings_episode": (
                state.mutation_consensus_deepenings
            ),
            "mutation_consensus_deep_majorities_episode": (
                state.mutation_consensus_deep_majorities
            ),
            "mutation_consensus_deep_overrides_episode": (
                state.mutation_consensus_deep_overrides
            ),
            "mutation_consensus_still_no_majority_fallbacks_episode": (
                state.mutation_consensus_still_no_majority
            ),
            "mutation_consensus_deep_extra_llm_calls_episode": (
                state.mutation_consensus_deep_extra_calls
            ),
            "terminal_consensus_invocations_episode": (
                state.terminal_consensus_invocations
            ),
            "terminal_consensus_respond_majorities_episode": (
                state.terminal_consensus_respond_majorities
            ),
            "terminal_consensus_action_majorities_episode": (
                state.terminal_consensus_action_majorities
            ),
            "terminal_consensus_action_overrides_episode": (
                state.terminal_consensus_action_overrides
            ),
            "terminal_consensus_no_majority_fallbacks_episode": (
                state.terminal_consensus_no_majority_fallbacks
            ),
            "terminal_consensus_extra_llm_calls_episode": (
                state.terminal_consensus_extra_calls
            ),
            "terminal_consensus_added_latency_ms_episode": round(
                state.terminal_consensus_added_latency_ms, 3
            ),
            "struggle_effort_escalated_episode": (
                state.struggle_effort_escalated
            ),
            "struggle_effort_trigger_episode": (
                state.struggle_effort_trigger
            ),
            "struggle_effort_high_calls_episode": (
                state.struggle_effort_high_calls
            ),
            "struggle_effort_added_latency_ms_episode": round(
                state.struggle_effort_added_latency_ms, 3
            ),
            "terminal_medium_reissues_episode": state.terminal_medium_reissues,
            "terminal_medium_responds_kept_episode": (
                state.terminal_medium_responds_kept
            ),
            "terminal_medium_turned_action_episode": (
                state.terminal_medium_turned_action
            ),
            "terminal_medium_added_latency_ms_episode": round(
                state.terminal_medium_added_latency_ms, 3
            ),
            "route_resolver_fires_episode": state.route_resolver_fires,
            "route_resolver_blocked_reads_episode": (
                state.route_resolver_blocked_reads
            ),
            "route_budget_fires_episode": state.route_budget_fires,
            "route_budget_blocked_reads_episode": (
                state.route_budget_blocked_reads
            ),
            "route_budget_terminal_limitations_episode": (
                state.route_budget_terminal_limitations
            ),
            "nav_intent_preflight_bounces_episode": (
                state.nav_intent_preflight_bounces
            ),
            "nav_intent_preflight_pass_throughs_episode": (
                state.nav_intent_preflight_pass_throughs
            ),
            "step_coverage_fires_episode": state.step_coverage_fires,
            "step_coverage_redecisions_episode": (
                state.step_coverage_redecisions
            ),
            "p3_ask_gate_v2_suppressions_episode": (
                state.p3_ask_gate_v2_suppressions
            ),
            "ask_type_gate_suppressions_episode": (
                state.ask_type_gate_suppressions
            ),
            "textcall_guard_fires_episode": state.textcall_guard_fires,
            "textcall_guard_executes_episode": state.textcall_guard_executes,
            "textcall_guard_redecides_episode": state.textcall_guard_redecides,
            "arg_lint_fires_episode": state.arg_lint_fires,
            "arg_lint_argument_bounces_episode": (
                state.arg_lint_argument_bounces
            ),
            "arg_lint_disclosure_revises_episode": (
                state.arg_lint_disclosure_revises
            ),
            "read_resolve_redirects_episode": state.read_resolve_redirects,
            "grounded_ask_fires_episode": state.grounded_ask_fires,
            "grounded_ask_reads_episode": state.grounded_ask_reads,
            "grounded_ask_redraft_asks_episode": (
                state.grounded_ask_redraft_asks
            ),
            "grounded_ask_redraft_acts_episode": (
                state.grounded_ask_redraft_acts
            ),
            "grounded_ask_redraft_responds_episode": (
                state.grounded_ask_redraft_responds
            ),
            "ask_content_consensus_invocations_episode": (
                state.ask_content_consensus_invocations
            ),
            "ask_content_consensus_majority_selections_episode": (
                state.ask_content_consensus_majority_selections
            ),
            "ask_content_consensus_fallbacks_episode": (
                state.ask_content_consensus_fallbacks
            ),
            "ask_content_consensus_extra_llm_calls_episode": (
                state.ask_content_consensus_extra_calls
            ),
            "ask_content_consensus_added_latency_ms_episode": round(
                state.ask_content_consensus_added_latency_ms, 3
            ),
            "llm_consensus_judge_calls_episode": (
                state.llm_consensus_judge_calls
            ),
            "llm_consensus_judge_majorities_episode": (
                state.llm_consensus_judge_majorities
            ),
            "llm_consensus_judge_overrides_episode": (
                state.llm_consensus_judge_overrides
            ),
            "llm_consensus_judge_no_majority_episode": (
                state.llm_consensus_judge_no_majority
            ),
            "llm_consensus_judge_errors_episode": (
                state.llm_consensus_judge_errors
            ),
            "llm_consensus_judge_timeouts_episode": (
                state.llm_consensus_judge_timeouts
            ),
            "llm_consensus_judge_malformed_episode": (
                state.llm_consensus_judge_malformed
            ),
            "llm_consensus_judge_budget_suppressed_episode": (
                state.llm_consensus_judge_budget_suppressed
            ),
            "llm_consensus_judge_added_latency_ms_episode": round(
                state.llm_consensus_judge_added_latency_ms, 3
            ),
            "llm_ask_triage_calls_episode": state.llm_ask_triage_calls,
            "llm_ask_triage_resolvable_labels_episode": (
                state.llm_ask_triage_resolvable
            ),
            "llm_ask_triage_genuine_ambiguity_labels_episode": (
                state.llm_ask_triage_genuine
            ),
            "llm_ask_triage_no_ambiguity_labels_episode": (
                state.llm_ask_triage_no_ambiguity
            ),
            "llm_ask_triage_fires_episode": state.llm_ask_triage_fires,
            "llm_ask_triage_reads_episode": state.llm_ask_triage_reads,
            "llm_ask_triage_invalid_proposals_episode": (
                state.llm_ask_triage_invalid
            ),
            "llm_ask_triage_errors_episode": state.llm_ask_triage_errors,
            "llm_ask_triage_timeouts_episode": state.llm_ask_triage_timeouts,
            "llm_ask_triage_malformed_episode": state.llm_ask_triage_malformed,
            "llm_ask_triage_budget_suppressed_episode": (
                state.llm_ask_triage_budget_suppressed
            ),
            "llm_ask_triage_redraft_asks_episode": (
                state.llm_ask_triage_redraft_asks
            ),
            "llm_ask_triage_redraft_acts_episode": (
                state.llm_ask_triage_redraft_acts
            ),
            "llm_ask_triage_redraft_responds_episode": (
                state.llm_ask_triage_redraft_responds
            ),
            "llm_ask_triage_added_latency_ms_episode": round(
                state.llm_ask_triage_added_latency_ms, 3
            ),
            "llm_limitation_classifier_calls_episode": (
                state.llm_limitation_classifier_calls
            ),
            "llm_limitation_classifier_terminates_episode": (
                state.llm_limitation_classifier_terminates
            ),
            "llm_limitation_classifier_continues_episode": (
                state.llm_limitation_classifier_continues
            ),
            "llm_limitation_classifier_errors_episode": (
                state.llm_limitation_classifier_errors
            ),
            "llm_limitation_classifier_timeouts_episode": (
                state.llm_limitation_classifier_timeouts
            ),
            "llm_limitation_classifier_malformed_episode": (
                state.llm_limitation_classifier_malformed
            ),
            "llm_limitation_classifier_added_latency_ms_episode": round(
                state.llm_limitation_classifier_added_latency_ms, 3
            ),
        }
        if self.config.phase_gate:
            meta.update(
                phase_gate_decisions_episode=state.phase_gate_decisions,
                phase_gate_withheld_tools_episode=(
                    state.phase_gate_withheld_tools
                ),
                phase_gate_fail_opens_episode=state.phase_gate_fail_opens,
                phase_gate_harmful_withholds_episode=(
                    state.phase_gate_harmful_withholds
                ),
            )
        if self.csp_enabled:
            meta.update(
                csp_user_turns_episode=state.csp_user_turns_episode,
                csp_assistant_asks_episode=state.csp_assistant_asks_episode,
                csp_withheld_counts_episode=dict(state.csp_withheld_counts),
                csp_fail_open_reopens_episode=state.csp_fail_open_reopens,
            )
        if self.fewshot_enabled:
            meta.update(
                fewshot_selection_counts_episode=dict(
                    state.fewshot_selection_counts
                ),
                fewshot_example_id=(
                    state.fewshot_selection.example_id
                    if state.fewshot_selection is not None
                    else None
                ),
            )
        return meta

    def _result(
        self,
        action: dict[str, Any],
        tally: _CallTally,
        *,
        state: _EpisodeState,
        injected_reads: int,
        escalation_signal: str | None,
        prefetch_reads: int = 0,
        prefetch_error_drops: int = 0,
    ) -> AgentInferenceResult:
        metrics: dict[str, Any] = {
            METRIC_INJECTED_READS: injected_reads,
            METRIC_ESCALATIONS_FIRED: int(escalation_signal is not None),
            METRIC_MICRO_PROMPT_TOKENS: self.micro_prompt_token_count,
            METRIC_DECISIONS: int(tally.calls > 0),
            METRIC_UNDEFINED_TOOL_CALLS: tally.undefined_tool_calls,
            METRIC_SCHEMA_VIOLATIONS: tally.schema_violations,
            METRIC_PARSE_FAILURES: tally.parse_failures,
            METRIC_INPUT_TOKENS: (
                tally.token_usage.input_tokens
                if tally.token_usage is not None
                else 0
            ),
            METRIC_TRANSPORT_CALLS: tally.calls,
            METRIC_PREFETCH_READS: prefetch_reads,
            METRIC_PREFETCH_ERROR_DROPS: prefetch_error_drops,
            METRIC_PREFETCH_SEMANTIC_EMITTED: (
                state.prefetch_semantic_emitted
            ),
            METRIC_PREFETCH_SEMANTIC_SUPPRESSED: (
                state.prefetch_semantic_suppressed
            ),
            METRIC_EVENT_EXEMPLAR_TOKENS: tally.event_exemplar_tokens,
            METRIC_EVENT_E4_SKIPS: tally.event_e4_skips,
            METRIC_PHASE_GATE_DECISIONS: tally.phase_gate_decisions,
            METRIC_PHASE_GATE_WITHHELD_TOOLS: tally.phase_gate_withheld_tools,
            METRIC_PHASE_GATE_FAIL_OPENS: tally.phase_gate_fail_opens,
            METRIC_PHASE_GATE_HARMFUL_WITHHOLDS: (
                tally.phase_gate_harmful_withholds
            ),
            METRIC_TERMINAL_EFFORT_HIGH_CALLS: (
                tally.terminal_effort_high_calls
            ),
            METRIC_TERMINAL_EFFORT_MEDIUM_CALLS: (
                tally.terminal_effort_medium_calls
            ),
            METRIC_ARGUMENT_BINDING_RELATIVE_CLARIFICATIONS: (
                tally.argument_binding_relative_clarifications
            ),
            METRIC_ARGUMENT_BINDING_ROUTE_CORRECTIONS: (
                tally.argument_binding_route_corrections
            ),
            METRIC_DISCLOSURE_CONFIRMATION_REASKS: (
                tally.disclosure_confirmation_reasks
            ),
            METRIC_DISCLOSURE_UNAVAILABLE_ACKS: (
                tally.disclosure_unavailable_acks
            ),
            METRIC_TRUNCATION_RESCUE_FIRES: tally.truncation_rescue_fires,
            METRIC_PLACEHOLDER_GUARD_FIRES: tally.placeholder_guard_fires,
            METRIC_VAGUE_DEGREE_CLARIFICATIONS: (
                tally.vague_degree_clarifications
            ),
            METRIC_VAGUE_DEGREE_PREFERENCE_REDIRECTS: (
                tally.vague_degree_preference_redirects
            ),
            METRIC_VAGUE_DEGREE_PREFERENCE_APPLIES: (
                tally.vague_degree_preference_applies
            ),
            METRIC_SCHEMA_PREFLIGHT_BOUNCES: tally.schema_preflight_bounces,
            METRIC_SCHEMA_PREFLIGHT_PASS_THROUGHS: (
                tally.schema_preflight_pass_throughs
            ),
            METRIC_VALUE_P1_CONTEXT_APPLIES: tally.value_p1_context_applies,
            METRIC_VALUE_P1_ASK_SUPPRESSIONS: tally.value_p1_ask_suppressions,
            METRIC_VALUE_P2_BINDING_BOUNCES: tally.value_p2_binding_bounces,
            METRIC_VALUE_P2_PASS_THROUGHS: tally.value_p2_pass_throughs,
            METRIC_VALUE_P3_PREFERENCE_READS: tally.value_p3_preference_reads,
            METRIC_VALUE_P3_FALLBACK_ASKS: tally.value_p3_fallback_asks,
            METRIC_VALUE_P4_NAV_REDIRECTS: tally.value_p4_nav_redirects,
            METRIC_VALUE_P4_NAV_BLOCKS: tally.value_p4_nav_blocks,
            METRIC_VALUE_P5_OCCUPANCY_READS: tally.value_p5_occupancy_reads,
            METRIC_VALUE_P6_CLAIM_REVISES: tally.value_p6_claim_revises,
            METRIC_TIME_FORMAT_REVISES: tally.time_format_revises,
            METRIC_INJECTED_ASKS: tally.injected_asks,
            METRIC_ASK_BUDGET_SUPPRESSED: tally.ask_budget_suppressed,
            METRIC_MALFORMED_ARGUMENT_RESCUE_FIRES: (
                tally.malformed_argument_rescue_fires
            ),
            METRIC_REPEATED_READ_BLOCKS: tally.repeated_read_blocks,
            METRIC_ROUTE_REFERENCE_BOUNCES: tally.route_reference_bounces,
            METRIC_POLICY_LINT_REVISES: tally.policy_lint_revises,
            METRIC_POLICY_LINT_ZONE_DIFFERENCE_REVISES: (
                tally.policy_lint_zone_difference_revises
            ),
            METRIC_POLICY_LINT_TEMPERATURE_UNIT_REVISES: (
                tally.policy_lint_temperature_unit_revises
            ),
            METRIC_MUTATION_CONSENSUS_INVOCATIONS: (
                tally.mutation_consensus_invocations
            ),
            METRIC_MUTATION_CONSENSUS_MAJORITY_AGREEMENTS: (
                tally.mutation_consensus_majority_agreements
            ),
            METRIC_MUTATION_CONSENSUS_MAJORITY_OVERRIDES: (
                tally.mutation_consensus_majority_overrides
            ),
            METRIC_MUTATION_CONSENSUS_NO_MAJORITY: (
                tally.mutation_consensus_no_majority_fallbacks
            ),
            METRIC_MUTATION_CONSENSUS_EXTRA_CALLS: (
                tally.mutation_consensus_extra_calls
            ),
            METRIC_MUTATION_CONSENSUS_ADDED_LATENCY_MS: round(
                tally.mutation_consensus_added_latency_ms, 3
            ),
            METRIC_MUTATION_CONSENSUS_DEEPENINGS: (
                tally.mutation_consensus_deepenings
            ),
            METRIC_MUTATION_CONSENSUS_DEEP_MAJORITIES: (
                tally.mutation_consensus_deep_majorities
            ),
            METRIC_MUTATION_CONSENSUS_DEEP_OVERRIDES: (
                tally.mutation_consensus_deep_overrides
            ),
            METRIC_MUTATION_CONSENSUS_STILL_NO_MAJORITY: (
                tally.mutation_consensus_still_no_majority
            ),
            METRIC_MUTATION_CONSENSUS_DEEP_EXTRA_CALLS: (
                tally.mutation_consensus_deep_extra_calls
            ),
            METRIC_TERMINAL_CONSENSUS_INVOCATIONS: (
                tally.terminal_consensus_invocations
            ),
            METRIC_TERMINAL_CONSENSUS_RESPOND_MAJORITIES: (
                tally.terminal_consensus_respond_majorities
            ),
            METRIC_TERMINAL_CONSENSUS_ACTION_MAJORITIES: (
                tally.terminal_consensus_action_majorities
            ),
            METRIC_TERMINAL_CONSENSUS_ACTION_OVERRIDES: (
                tally.terminal_consensus_action_overrides
            ),
            METRIC_TERMINAL_CONSENSUS_NO_MAJORITY: (
                tally.terminal_consensus_no_majority_fallbacks
            ),
            METRIC_TERMINAL_CONSENSUS_EXTRA_CALLS: (
                tally.terminal_consensus_extra_calls
            ),
            METRIC_TERMINAL_CONSENSUS_ADDED_LATENCY_MS: round(
                tally.terminal_consensus_added_latency_ms, 3
            ),
            METRIC_TERMINAL_MEDIUM_REISSUES: tally.terminal_medium_reissues,
            METRIC_TERMINAL_MEDIUM_RESPONDS_KEPT: (
                tally.terminal_medium_responds_kept
            ),
            METRIC_TERMINAL_MEDIUM_TURNED_ACTION: (
                tally.terminal_medium_turned_action
            ),
            METRIC_TERMINAL_MEDIUM_ADDED_LATENCY_MS: round(
                tally.terminal_medium_added_latency_ms, 3
            ),
            METRIC_ROUTE_RESOLVER_FIRES: tally.route_resolver_fires,
            METRIC_ROUTE_RESOLVER_BLOCKED_READS: (
                tally.route_resolver_blocked_reads
            ),
            METRIC_ROUTE_BUDGET_FIRES: tally.route_budget_fires,
            METRIC_ROUTE_BUDGET_BLOCKED_READS: (
                tally.route_budget_blocked_reads
            ),
            METRIC_ROUTE_BUDGET_TERMINAL_LIMITATIONS: (
                tally.route_budget_terminal_limitations
            ),
            METRIC_NAV_INTENT_PREFLIGHT_BOUNCES: (
                tally.nav_intent_preflight_bounces
            ),
            METRIC_NAV_INTENT_PREFLIGHT_PASS_THROUGHS: (
                tally.nav_intent_preflight_pass_throughs
            ),
            METRIC_STEP_COVERAGE_FIRES: tally.step_coverage_fires,
            METRIC_STEP_COVERAGE_REDECISIONS: (
                tally.step_coverage_redecisions
            ),
            METRIC_P3_ASK_GATE_V2_SUPPRESSIONS: (
                tally.p3_ask_gate_v2_suppressions
            ),
            METRIC_ASK_TYPE_GATE_SUPPRESSIONS: (
                tally.ask_type_gate_suppressions
            ),
            METRIC_TEXTCALL_GUARD_FIRES: tally.textcall_guard_fires,
            METRIC_TEXTCALL_GUARD_EXECUTES: tally.textcall_guard_executes,
            METRIC_TEXTCALL_GUARD_REDECIDES: tally.textcall_guard_redecides,
            METRIC_ARG_LINT_FIRES: tally.arg_lint_fires,
            METRIC_ARG_LINT_ARGUMENT_BOUNCES: (
                tally.arg_lint_argument_bounces
            ),
            METRIC_ARG_LINT_DISCLOSURE_REVISES: (
                tally.arg_lint_disclosure_revises
            ),
            METRIC_READ_RESOLVE_REDIRECTS: tally.read_resolve_redirects,
            METRIC_GROUNDED_ASK_FIRES: tally.grounded_ask_fires,
            METRIC_GROUNDED_ASK_READS: tally.grounded_ask_reads,
            METRIC_GROUNDED_ASK_REDRAFT_ASKS: (
                tally.grounded_ask_redraft_asks
            ),
            METRIC_GROUNDED_ASK_REDRAFT_ACTS: (
                tally.grounded_ask_redraft_acts
            ),
            METRIC_GROUNDED_ASK_REDRAFT_RESPONDS: (
                tally.grounded_ask_redraft_responds
            ),
            METRIC_ASK_CONTENT_CONSENSUS_INVOCATIONS: (
                tally.ask_content_consensus_invocations
            ),
            METRIC_ASK_CONTENT_CONSENSUS_MAJORITY_SELECTIONS: (
                tally.ask_content_consensus_majority_selections
            ),
            METRIC_ASK_CONTENT_CONSENSUS_FALLBACKS: (
                tally.ask_content_consensus_fallbacks
            ),
            METRIC_ASK_CONTENT_CONSENSUS_EXTRA_CALLS: (
                tally.ask_content_consensus_extra_calls
            ),
            METRIC_ASK_CONTENT_CONSENSUS_ADDED_LATENCY_MS: round(
                tally.ask_content_consensus_added_latency_ms, 3
            ),
            METRIC_LLM_CONSENSUS_JUDGE_CALLS: (
                tally.llm_consensus_judge_calls
            ),
            METRIC_LLM_CONSENSUS_JUDGE_MAJORITIES: (
                tally.llm_consensus_judge_majorities
            ),
            METRIC_LLM_CONSENSUS_JUDGE_OVERRIDES: (
                tally.llm_consensus_judge_overrides
            ),
            METRIC_LLM_CONSENSUS_JUDGE_NO_MAJORITY: (
                tally.llm_consensus_judge_no_majority
            ),
            METRIC_LLM_CONSENSUS_JUDGE_ERRORS: (
                tally.llm_consensus_judge_errors
            ),
            METRIC_LLM_CONSENSUS_JUDGE_TIMEOUTS: (
                tally.llm_consensus_judge_timeouts
            ),
            METRIC_LLM_CONSENSUS_JUDGE_MALFORMED: (
                tally.llm_consensus_judge_malformed
            ),
            METRIC_LLM_CONSENSUS_JUDGE_BUDGET_SUPPRESSED: (
                tally.llm_consensus_judge_budget_suppressed
            ),
            METRIC_LLM_CONSENSUS_JUDGE_ADDED_LATENCY_MS: round(
                tally.llm_consensus_judge_added_latency_ms, 3
            ),
            METRIC_LLM_ASK_TRIAGE_CALLS: tally.llm_ask_triage_calls,
            METRIC_LLM_ASK_TRIAGE_RESOLVABLE: (
                tally.llm_ask_triage_resolvable
            ),
            METRIC_LLM_ASK_TRIAGE_GENUINE: tally.llm_ask_triage_genuine,
            METRIC_LLM_ASK_TRIAGE_NO_AMBIGUITY: (
                tally.llm_ask_triage_no_ambiguity
            ),
            METRIC_LLM_ASK_TRIAGE_FIRES: tally.llm_ask_triage_fires,
            METRIC_LLM_ASK_TRIAGE_READS: tally.llm_ask_triage_reads,
            METRIC_LLM_ASK_TRIAGE_INVALID: tally.llm_ask_triage_invalid,
            METRIC_LLM_ASK_TRIAGE_ERRORS: tally.llm_ask_triage_errors,
            METRIC_LLM_ASK_TRIAGE_TIMEOUTS: tally.llm_ask_triage_timeouts,
            METRIC_LLM_ASK_TRIAGE_MALFORMED: tally.llm_ask_triage_malformed,
            METRIC_LLM_ASK_TRIAGE_BUDGET_SUPPRESSED: (
                tally.llm_ask_triage_budget_suppressed
            ),
            METRIC_LLM_ASK_TRIAGE_REDRAFT_ASKS: (
                tally.llm_ask_triage_redraft_asks
            ),
            METRIC_LLM_ASK_TRIAGE_REDRAFT_ACTS: (
                tally.llm_ask_triage_redraft_acts
            ),
            METRIC_LLM_ASK_TRIAGE_REDRAFT_RESPONDS: (
                tally.llm_ask_triage_redraft_responds
            ),
            METRIC_LLM_ASK_TRIAGE_ADDED_LATENCY_MS: round(
                tally.llm_ask_triage_added_latency_ms, 3
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_CALLS: (
                tally.llm_limitation_classifier_calls
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_TERMINATES: (
                tally.llm_limitation_classifier_terminates
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_CONTINUES: (
                tally.llm_limitation_classifier_continues
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_ERRORS: (
                tally.llm_limitation_classifier_errors
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_TIMEOUTS: (
                tally.llm_limitation_classifier_timeouts
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_MALFORMED: (
                tally.llm_limitation_classifier_malformed
            ),
            METRIC_LLM_LIMITATION_CLASSIFIER_ADDED_LATENCY_MS: round(
                tally.llm_limitation_classifier_added_latency_ms, 3
            ),
            METRIC_TERMINAL_READBACK_FIRES: int(
                action.get("action") == "tool_calls"
                and bool(state.terminal_readback_pending_reads)
            ),
            METRIC_TERMINAL_READBACK_READS: (
                len(action.get("tool_calls") or [])
                if action.get("action") == "tool_calls"
                and bool(state.terminal_readback_pending_reads)
                else 0
            ),
            METRIC_TERMINAL_READBACK_MISMATCHES: int(
                escalation_signal == SIGNAL_TERMINAL_READBACK
            ),
            METRIC_TERMINAL_READBACK_REVISES: int(
                escalation_signal == SIGNAL_TERMINAL_READBACK
            ),
            **{
                f"{METRIC_EVENT_EXEMPLAR_PREFIX}{event.casefold()}": (
                    tally.event_exemplar_fires.get(event, 0)
                )
                for event in EVENT_EXEMPLAR_PRIORITY
            },
            **{
                metric: int(signal == escalation_signal)
                for signal, metric in METRIC_ESCALATION_BY_SIGNAL.items()
            },
        }
        if self.config.reformulate:
            metrics.update(
                {
                    METRIC_REFORMULATION_BLOCK_TOKENS: (
                        tally.reformulation_block_tokens
                    ),
                    METRIC_REFORMULATION_RULES_HIT: tally.reformulation_rules_hit,
                    METRIC_REFORMULATION_TOOLS_SUGGESTED: (
                        tally.reformulation_tools_suggested
                    ),
                }
            )
        if self.config.ledger:
            metrics.update(
                {
                    METRIC_HISTORY_TOKENS_WITHOUT_LEDGER: (
                        tally.history_tokens_without_ledger
                    ),
                    METRIC_HISTORY_TOKENS_WITH_LEDGER: (
                        tally.history_tokens_with_ledger
                    ),
                    METRIC_LEDGER_TOKENS: tally.ledger_tokens,
                }
            )
        if self.fewshot_enabled:
            metrics.update(
                {
                    METRIC_FEWSHOT_TOKENS: tally.fewshot_tokens,
                    **{
                        f"{METRIC_FEWSHOT_SELECTION_PREFIX}{example.example_id}": (
                            tally.fewshot_selection_counts.get(
                                example.example_id, 0
                            )
                        )
                        for example in self.fewshot_examples
                    },
                }
            )
        if self.config.csp_brief:
            metrics.update(
                {
                    METRIC_CSP_RAW_HISTORY_TOKENS: tally.csp_raw_history_tokens,
                    METRIC_CSP_BRIEF_TOKENS: tally.csp_brief_tokens,
                    METRIC_CSP_INPUT_TOKENS: (
                        tally.token_usage.input_tokens
                        if tally.token_usage is not None
                        else 0
                    ),
                    METRIC_CSP_INPUT_CALLS: tally.calls,
                }
            )
        if self.config.csp_afford:
            metrics.update(
                {
                    METRIC_CSP_WITHHELD_TOOLS: sum(
                        tally.csp_withheld_counts.values()
                    ),
                    METRIC_CSP_FAIL_OPEN_REOPENS: tally.csp_fail_open_reopens,
                    **{
                        f"adaptive_minimal_csp_withheld_{name}": count
                        for name, count in tally.csp_withheld_counts.items()
                    },
                }
            )
        if self.csp_enabled:
            metrics[METRIC_CSP_USER_TURNS] = state.csp_user_turns_pending
            metrics[METRIC_CSP_ASSISTANT_ASKS] = state.csp_assistant_asks_pending
            state.csp_user_turns_pending = 0
            state.csp_assistant_asks_pending = 0
        return AgentInferenceResult(
            next_action=action,
            elapsed_ms=tally.duration_ms,
            token_usage=tally.token_usage,
            cost=tally.cost,
            internal_calls=tally.calls,
            quota_wait_ms=tally.quota_wait_ms,
            harness_metrics=metrics,
        )


def build_adaptive_prompt(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    fault: InternalFault | None = None,
) -> str:
    """Build the unchanged decision context, plus only a concrete fault on repair."""

    prompt: dict[str, Any] = {
        "task": "Choose exactly one next assistant action for this CAR-bench turn.",
        "available_tools": tools,
        "conversation_transcript": _messages_for_prompt(messages),
        "output_contract": {
            "respond": "Use when speaking naturally to the user.",
            "tool_calls": "Use one or more supplied CAR-bench environment tools.",
        },
    }
    if fault is not None:
        prompt["fault"] = fault.text
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _user_turn_key(messages: list[dict[str, Any]]) -> tuple[int, str]:
    user_messages = [message for message in messages if message.get("role") == "user"]
    return (
        len(user_messages),
        str(user_messages[-1].get("content") or "") if user_messages else "",
    )


# E4 is no longer an injected exemplar in EVENTv2. It is a deterministic
# exact-signature skip immediately before mutation release.
EVENT_EXEMPLAR_PRIORITY = ("E2", "E1", "E3", "E5")
EVENT_EXEMPLAR_IDS = {
    "E1": "unavailable_observable",
    "E2": "approval_execute",
    "E3": "target_time",
    "E4": "minimal_diff",
    "E5": "grounded_claim",
}


def _previous_assistant_before_latest_user(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return None
    return next(
        (
            messages[index]
            for index in range(latest_user_index - 1, -1, -1)
            if messages[index].get("role") == "assistant"
        ),
        None,
    )


def _approval_event(messages: list[dict[str, Any]]) -> bool:
    _, user_text = _user_turn_key(messages)
    normalized = " ".join(re.findall(r"[a-z0-9]+", user_text.casefold()))
    affirmation = bool(
        re.match(
            r"^(?:yes|yes please|ok|okay|sure|confirmed|confirm|do it|go ahead|proceed|please do)(?:\b|$)",
            normalized,
        )
    )
    previous = _previous_assistant_before_latest_user(messages)
    if not affirmation or previous is None:
        return False
    content = str(previous.get("content") or "")
    proposal_markers = re.compile(
        r"\b(?:confirm|proceed|shall i|should i|would you like|do you want|"
        r"send|set|change|enable|disable|open|close|call|start|delete|remove|add)\b",
        re.IGNORECASE,
    )
    return "?" in content and bool(proposal_markers.search(content))


def _unavailable_requested_event(messages: list[dict[str, Any]]) -> bool:
    trailing = []
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        trailing.append(message)
    if not trailing:
        return False
    user_tokens = set().union(
        *(
            _tokens(str(message.get("content") or ""))
            for message in messages
            if message.get("role") == "user"
        )
    )
    for message in trailing:
        content = str(message.get("content") or "")
        if not _tool_result_is_unavailable(content):
            continue
        call = _tool_result_call(messages, message)
        tool_name = str(message.get("name") or (call or {}).get("tool_name") or "")
        subject_tokens = _name_core_tokens(tool_name)
        unavailable_tokens = _tokens(content) - {
            "unknown",
            "unavailable",
            "not",
            "available",
            "absent",
            "found",
            "null",
            "none",
            "error",
            "status",
            "result",
        }
        if not subject_tokens or (subject_tokens | unavailable_tokens) & user_tokens:
            return True
    return False


def _target_time_event(messages: list[dict[str, Any]]) -> bool:
    """Match a future event and a conditional action in the *current* turn."""

    _, user_text = _user_turn_key(messages)
    lowered = user_text.casefold()
    future_reference = bool(
        re.search(
            r"\b(?:upon\s+arrival|at\s+arrival|later|tonight|tomorrow|"
            r"when\s+(?:i|we)\s+(?:arrive|get\s+there|reach\s+(?:it|there|the\s+destination))|"
            r"at\s+(?:[01]?\d|2[0-3]):[0-5]\d)\b",
            lowered,
        )
    )
    condition = bool(
        re.search(
            r"\b(?:if|unless|depending\s+on|based\s+on\s+whether|when)\b",
            lowered,
        )
    )
    pending_decision = bool(
        re.search(
            r"\b(?:set|change|enable|disable|turn|open|close|start|stop|"
            r"navigate|route|choose|select|use|find|send|call|activate|"
            r"deactivate|add|remove|replace|update)\b",
            lowered,
        )
    )
    return future_reference and condition and pending_decision


def _minimal_diff_event(
    messages: list[dict[str, Any]], state: _EpisodeState
) -> bool:
    _, user_text = _user_turn_key(messages)
    user_tokens = _tokens(user_text)
    return any(
        bool(set(mutation.get("subsystem_tokens") or ()) & user_tokens)
        for mutation in state.successful_mutations
    )


def _draft_has_ungrounded_state_claim(
    action: dict[str, Any], messages: list[dict[str, Any]]
) -> bool:
    if action.get("action") != "respond":
        return False
    content = str(action.get("content") or "")
    if not content or "?" in content:
        return False
    if _grounded_respond_fault(action, messages, already_fired=False) is not None:
        return True
    tool_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    ).casefold()
    state_values = re.findall(
        r"\b(?:is|are|now|remains?|at)\s+"
        r"(true|false|on|off|open|closed|active|inactive|enabled|disabled|[A-Z][A-Z_]{2,})\b",
        content,
    )
    return any(value.casefold() not in tool_text for value in state_values)


def _detect_event_exemplar(
    messages: list[dict[str, Any]], state: _EpisodeState
) -> str | None:
    user_turn = _user_turn_key(messages)[0]
    if _approval_event(messages) and (
        "E2", user_turn
    ) not in state.event_exemplar_turn_fires:
        return "E2"
    e1_active = _unavailable_requested_event(messages)
    if e1_active:
        # E1 wins over E3 even after E1 already fired for this user turn. This
        # prevents an unavailable result from cascading into target-time prose.
        return (
            "E1"
            if ("E1", user_turn) not in state.event_exemplar_turn_fires
            else None
        )
    if _target_time_event(messages) and (
        "E3", user_turn
    ) not in state.event_exemplar_turn_fires:
        return "E3"
    return None


def _record_event_exemplar(
    event: str,
    *,
    state: _EpisodeState,
    tally: _CallTally,
    messages: list[dict[str, Any]],
    ctx_logger: Any,
) -> FewShotSelection:
    selection = selection_for_example(EVENT_EXEMPLAR_IDS[event])
    state.event_exemplar_counts[event] += 1
    state.event_exemplar_turn_fires.add((event, _user_turn_key(messages)[0]))
    tally.event_exemplar_fires[event] = tally.event_exemplar_fires.get(event, 0) + 1
    tally.event_exemplar_tokens += selection.token_count
    ctx_logger.info(
        "Adaptive-minimal event exemplar fired",
        detector=event,
        example_id=selection.example_id,
        example_tokens=selection.token_count,
        priority=list(EVENT_EXEMPLAR_PRIORITY),
        counts_episode=dict(state.event_exemplar_counts),
    )
    return selection


def _tools_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(tool.get("function", {}).get("name") or ""): tool
        for tool in tools
        if str(tool.get("function", {}).get("name") or "")
    }


def derive_prefetch_calls(
    tools: list[dict[str, Any]], *, user_text: str = ""
) -> list[dict[str, Any]]:
    """Return free safe reads plus mentioned-subsystem reads in stable order."""

    user_tokens = _tokens(user_text)
    eligible: list[dict[str, Any]] = []
    for name, tool in _tools_by_name(tools).items():
        if not name.casefold().startswith("get_"):
            continue
        arguments = _safe_read_arguments(tool)
        if arguments is None:
            continue
        schema = tool.get("function", {}).get("parameters") or {}
        required = schema.get("required") or []
        zero_argument = isinstance(required, list) and not required
        subsystem_mentioned = bool(_name_core_tokens(name) & user_tokens)
        by_lookup = bool(re.match(r"^get_.+_by_.+$", name.casefold()))
        if (zero_argument and not by_lookup) or subsystem_mentioned:
            eligible.append({"tool_name": name, "arguments": arguments})
    return sorted(eligible, key=lambda call: str(call["tool_name"]))


def _semantic_prefetch_call_valid(
    tool: dict[str, Any], arguments: dict[str, Any]
) -> bool:
    """Accept only catalogued zero-argument reads with schema-valid emptiness.

    JSON Schema often models an at-least-one selector as ``anyOf`` branches,
    but the contact catalog historically exposed the same runtime contract
    without a top-level ``required`` field.  The explicit allowlist is the
    fail-closed semantic boundary; recursive validation additionally enforces
    any standard ``anyOf``/``oneOf`` constraints when present.
    """

    function = tool.get("function") or {}
    name = str(function.get("name") or "").casefold()
    if name not in SEMANTIC_PREFETCH_ZERO_ARGUMENT_GETTERS or arguments:
        return False
    schema = function.get("parameters") or {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    if not isinstance(schema, dict):
        return False
    return _validate_schema({}, schema, path="$arguments") is None


def derive_prefetch_tools(
    tools: list[dict[str, Any]], *, user_text: str = ""
) -> list[str]:
    """Compatibility view of :func:`derive_prefetch_calls`."""

    return [
        str(call["tool_name"])
        for call in derive_prefetch_calls(tools, user_text=user_text)
    ]


def _tool_call_signature(call: dict[str, Any]) -> str:
    name = str(call.get("tool_name") or "")
    arguments = call.get("arguments") or {}
    return json.dumps(
        [name, arguments],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _skip_satisfied_mutations(
    action: dict[str, Any], state: _EpisodeState
) -> tuple[dict[str, Any], int]:
    """Drop exact mutations already recorded as successful in this episode."""

    if action.get("action") != "tool_calls":
        return action, 0
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        if (
            _is_mutating_tool_name(name)
            and _tool_call_signature(call) in state.successful_mutation_signatures
        ):
            skipped.append(call)
        else:
            kept.append(call)
    if not skipped:
        return action, 0
    state.event_e4_skips += len(skipped)
    if kept:
        return {"action": "tool_calls", "tool_calls": kept}, len(skipped)
    return {
        "action": "respond",
        "content": (
            "That requested setting is already in place, so I didn't repeat "
            "the action."
        ),
    }, len(skipped)


TURN_GUARD_FAULT_TEXT = (
    "If any part of the reply depends on vehicle/route/weather/contact state "
    "not already in context, call the needed tool first; if the user's request "
    "is actionable now, execute it; otherwise proceed."
)

UNAVAILABILITY_LOOP_FAULT_TEXT = (
    "the observable/capability is unavailable — state that decisively to the "
    "user, offer the nearest available alternative, close the topic, do not "
    "re-attempt."
)


def _unavailability_loop_fault(
    action: dict[str, Any],
    *,
    unavailable_tool_signatures: set[str],
    already_fired: bool,
) -> InternalFault | None:
    if already_fired or action.get("action") != "tool_calls":
        return None
    if not any(
        _tool_call_signature(call) in unavailable_tool_signatures
        for call in action.get("tool_calls") or []
    ):
        return None
    return InternalFault(
        SIGNAL_UNAVAILABILITY_LOOP,
        UNAVAILABILITY_LOOP_FAULT_TEXT,
    )


def _turn_guard_fault(
    action: dict[str, Any],
    *,
    non_prefetch_tool_calls_executed: int,
    already_fired: bool,
) -> InternalFault | None:
    if (
        action.get("action") != "respond"
        or non_prefetch_tool_calls_executed > 0
        or already_fired
    ):
        return None
    return InternalFault(SIGNAL_TURN_GUARD, TURN_GUARD_FAULT_TEXT)


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


_READ_PREFIXES = ("get_",)
_MUTATION_PREFIXES = (
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
_NAME_STOPWORDS = {
    "get",
    "set",
    "open",
    "close",
    "activate",
    "deactivate",
    "delete",
    "send",
    "call",
    "create",
    "update",
    "replace",
    "add",
    "remove",
    "status",
    "state",
    "current",
    "position",
    "setting",
    "information",
    "inside",
    "car",
    "vehicle",
    "one",
    "final",
    "new",
    "and",
    "by",
    "to",
    "from",
}
_STATE_READ_HINTS = {"status", "state", "current", "position", "setting", "level"}


def derive_read_pairings(tools: list[dict[str, Any]]) -> dict[str, str]:
    """Derive conservative mutation -> get_* pairs from catalog lexical schemas.

    There are no task, task-description, or tool-specific mapping tables here.
    All evidence is recomputed from names and the supplied function schemas.  A
    tie or weak match is deliberately left unpaired.
    """

    tools_by_name = _tools_by_name(tools)
    reads = {
        name: tool
        for name, tool in tools_by_name.items()
        if name.casefold().startswith(_READ_PREFIXES)
        and _safe_read_arguments(tool) is not None
    }
    mutations = {
        name: tool
        for name, tool in tools_by_name.items()
        if _is_mutating_tool_name(name)
    }
    pairings: dict[str, str] = {}
    for mutation_name, mutation_tool in sorted(mutations.items()):
        mutation_core = _name_core_tokens(mutation_name)
        mutation_args = _parameter_name_tokens(mutation_tool)
        if not mutation_core:
            continue
        ranked: list[tuple[int, str]] = []
        for read_name, read_tool in reads.items():
            read_name_tokens = _name_core_tokens(read_name)
            read_schema_tokens = _schema_tokens(read_tool)
            name_overlap = mutation_core & read_name_tokens
            schema_overlap = mutation_core & read_schema_tokens
            if not name_overlap and not schema_overlap:
                continue
            score = (
                5 * len(name_overlap)
                + 2 * len(schema_overlap)
                + 2 * len(mutation_args & read_schema_tokens)
                + 4 * len(name_overlap | schema_overlap)
                + 3 * int(bool(_tokens(read_name) & _STATE_READ_HINTS))
            )
            if score >= 8:
                ranked.append((score, read_name))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 2:
            continue
        pairings[mutation_name] = ranked[0][1]
    return pairings


def enrich_tool_descriptions(
    tools: list[dict[str, Any]], *, wave2_notes: bool = False
) -> list[dict[str, Any]]:
    """Append catalog-derived usage notes without mutating the supplied catalog."""

    enriched = deepcopy(tools)
    by_name = _tools_by_name(enriched)
    pairings = derive_read_pairings(enriched)
    edit_verbs = {"edit", "modify", "replace", "update"}
    creation_verbs = {"add", "create", "new"}
    for name, tool in by_name.items():
        function = tool.get("function") or {}
        description = str(function.get("description") or "").rstrip()
        lowered = name.casefold()
        tokens = _tokens(name)
        notes: list[str] = []
        if lowered.startswith(_READ_PREFIXES):
            notes.append(
                "Usage note: read this observable before answering questions "
                "about the subsystem; do not guess its state."
            )
        if tokens & creation_verbs:
            subject = _name_core_tokens(name) - creation_verbs
            alternatives = sorted(
                candidate
                for candidate in by_name
                if candidate != name
                and (_tokens(candidate) & edit_verbs)
                and (_name_core_tokens(candidate) & subject)
            )
            note = (
                "Usage note: creation can fail when the matching resource is active; "
                "modify the active resource instead"
            )
            if alternatives:
                note += " with " + ", ".join(alternatives[:4])
            notes.append(note + ".")
        read_name = pairings.get(name)
        if read_name is not None:
            notes.append(
                f"Usage note: inspect current state with {read_name} before this mutation."
            )
        if wave2_notes and lowered.startswith(_READ_PREFIXES) and (
            tokens & {"weather", "condition", "forecast"}
        ):
            notes.append(
                "Usage note: when a condition applies at a future event "
                "(e.g. arrival time), query at that event's time, not the current time."
            )
        if wave2_notes and _is_mutating_tool_name(name):
            notes.append(
                "Usage note: invoke only for fields the user explicitly requested; "
                "if the requested state is already satisfied, do not re-execute."
            )
        if notes:
            function["description"] = "\n".join(
                part for part in (description, *notes) if part
            )
    return enriched


def _is_mutating_tool_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered == "set_new_navigation" or lowered.startswith(_MUTATION_PREFIXES)


def _action_contains_mutation(action: dict[str, Any]) -> bool:
    """Treat every non-read tool as a mutation-point consensus trigger."""

    return action.get("action") == "tool_calls" and any(
        not str(call.get("tool_name") or "").casefold().startswith(_READ_PREFIXES)
        for call in action.get("tool_calls") or []
    )


def _action_is_pure_respond(action: dict[str, Any]) -> bool:
    return action.get("action") == "respond" and not action.get("tool_calls")


def _conversation_has_ended(messages: list[dict[str, Any]]) -> bool:
    """Return whether the last conversational event is a terminal response."""

    for message in reversed(messages):
        role = str(message.get("role") or "").casefold()
        if role == "assistant":
            return not bool(message.get("tool_calls"))
        if role in {"user", "tool"}:
            return False
    return False


def _canonical_consensus_value(value: Any) -> Any:
    """Normalize JSON values so superficial numeric formatting cannot split a vote."""

    if isinstance(value, dict):
        return {
            str(key): _canonical_consensus_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_consensus_value(item) for item in value]
    if isinstance(value, float):
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        if math.isfinite(value):
            return float(format(value, ".15g"))
    return value


def _llm_consensus_judge_candidate(
    action: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expose only an existing draft's ordered calls to the classifier."""

    if action is None:
        return {"rejected": True, "calls": []}
    return {
        "rejected": False,
        "calls": [
            {
                "tool_name": str(call.get("tool_name") or ""),
                "arguments": call.get("arguments") or {},
            }
            for call in action.get("tool_calls") or []
        ],
    }


def _llm_consensus_tool_signature(action: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(call.get("tool_name") or "")
        for call in action.get("tool_calls") or []
    )


def _parse_llm_consensus_judgment(
    text: str,
    candidates: list[dict[str, Any] | None],
) -> tuple[int | None, tuple[int, ...]]:
    """Validate a complete partition and return its semantic majority."""

    if len(candidates) != 3 or candidates[0] is None:
        raise ValueError("semantic judge requires one original and two samples")
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"groups"}:
        raise ValueError("semantic judge output must contain only groups")
    raw_groups = payload["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("semantic judge groups must be a non-empty list")

    groups: list[tuple[int, ...]] = []
    flattened: list[int] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError("semantic judge groups must be non-empty arrays")
        group: list[int] = []
        for index in raw_group:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("semantic judge indices must be integers")
            if index not in {0, 1, 2}:
                raise ValueError("semantic judge index is out of range")
            group.append(index)
        if len(set(group)) != len(group):
            raise ValueError("semantic judge group repeats an index")
        normalized = tuple(sorted(group))
        groups.append(normalized)
        flattened.extend(normalized)
    if sorted(flattened) != [0, 1, 2]:
        raise ValueError("semantic judge output must partition all candidates")

    majority_groups = [group for group in groups if len(group) >= 2]
    if not majority_groups:
        return None, ()
    if len(majority_groups) != 1:
        raise ValueError("semantic judge returned multiple majority groups")
    majority = majority_groups[0]
    majority_candidates = [candidates[index] for index in majority]
    if any(candidate is None for candidate in majority_candidates):
        raise ValueError("semantic judge grouped a rejected candidate")
    tool_signatures = {
        _llm_consensus_tool_signature(candidate)
        for candidate in majority_candidates
        if candidate is not None
    }
    if len(tool_signatures) != 1:
        raise ValueError("semantic majority changed the ordered tool set")
    return min(majority), majority


def _parse_llm_ask_triage_payload(
    text: str,
) -> tuple[str, str, dict[str, Any]]:
    """Validate the ask classifier's strict label and proposed read payload."""

    payload = json.loads(text)
    required_keys = {"label", "tool_name", "arguments_json"}
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError("ask triage output must contain exactly the schema keys")
    label = payload["label"]
    tool_name = payload["tool_name"]
    arguments_json = payload["arguments_json"]
    if label not in {
        "RESOLVABLE_BY_READ",
        "GENUINE_AMBIGUITY",
        "NO_AMBIGUITY",
    }:
        raise ValueError("ask triage label is invalid")
    if not isinstance(tool_name, str) or not isinstance(arguments_json, str):
        raise TypeError("ask triage call fields must be strings")
    arguments = json.loads(arguments_json)
    if not isinstance(arguments, dict):
        raise TypeError("ask triage arguments_json must decode to an object")
    if label != "RESOLVABLE_BY_READ" and (tool_name or arguments):
        raise ValueError("non-read labels must not propose a tool call")
    if label == "RESOLVABLE_BY_READ" and not tool_name:
        raise ValueError("resolvable label requires a tool name")
    return label, tool_name, arguments


def _parse_llm_limitation_payload(
    text: str,
) -> tuple[str, str, dict[str, Any], str]:
    payload = json.loads(text)
    keys = {"label", "tool_name", "arguments_json", "finding"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("limitation classifier output has invalid keys")
    label = payload["label"]
    if label not in {
        "CAPABILITY_UNAVAILABLE_TERMINATE",
        "CAPABILITY_AVAILABLE_CONTINUE",
    }:
        raise ValueError("limitation classifier label is invalid")
    tool_name = payload["tool_name"]
    finding = payload["finding"]
    if not isinstance(tool_name, str) or not isinstance(finding, str):
        raise TypeError("limitation classifier text fields must be strings")
    arguments = json.loads(payload["arguments_json"])
    if not isinstance(arguments, dict):
        raise TypeError("limitation classifier arguments must be an object")
    if label == "CAPABILITY_UNAVAILABLE_TERMINATE" and (tool_name or arguments):
        raise ValueError("terminate label must not propose a call")
    if label == "CAPABILITY_AVAILABLE_CONTINUE" and not tool_name:
        raise ValueError("continue label requires a call")
    if not finding.strip():
        raise ValueError("limitation classifier finding is empty")
    return label, tool_name, arguments, finding.strip()


def _is_timeout_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, TimeoutError):
            return True
        text = f"{type(current).__name__}: {current}".casefold()
        if "timeout" in text or "timed out" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _mutation_consensus_signature(action: dict[str, Any]) -> str:
    """Return the ordered exact (tool, canonical arguments) decision signature."""

    calls = [] if action.get("action") != "tool_calls" else (
        action.get("tool_calls") or []
    )
    signature = [
        [
            str(call.get("tool_name") or ""),
            _canonical_consensus_value(call.get("arguments") or {}),
        ]
        for call in calls
    ]
    return json.dumps(
        signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _select_mutation_consensus(
    candidates: list[dict[str, Any] | None],
) -> tuple[dict[str, Any], bool, bool, list[str]]:
    """Select a 2-of-3 exact signature, or fail open to the original draft."""

    if len(candidates) != 3 or candidates[0] is None:
        raise ValueError("mutation consensus requires one original and two samples")
    signatures = [
        _mutation_consensus_signature(candidate)
        if candidate is not None
        else f"<guard-rejected:{index}>"
        for index, candidate in enumerate(candidates)
    ]
    counts = {signature: signatures.count(signature) for signature in signatures}
    majority_signature = next(
        (signature for signature in signatures if counts[signature] >= 2), None
    )
    original = candidates[0]
    assert original is not None
    if majority_signature is None:
        return original, False, False, signatures
    majority_indices = [
        index
        for index, signature in enumerate(signatures)
        if signature == majority_signature and candidates[index] is not None
    ]
    if not majority_indices:
        return original, False, False, signatures
    selected_index = 0 if 0 in majority_indices else majority_indices[0]
    selected = candidates[selected_index]
    assert selected is not None
    return selected, True, signatures[0] != majority_signature, signatures


def _select_deepened_mutation_consensus(
    candidates: list[dict[str, Any] | None],
) -> tuple[dict[str, Any], bool, bool, list[str]]:
    """Select an exact 3-of-5 signature, or fail open to the original draft."""

    if len(candidates) != 5 or candidates[0] is None:
        raise ValueError(
            "deepened mutation consensus requires one original and four samples"
        )
    signatures = [
        _mutation_consensus_signature(candidate)
        if candidate is not None
        else f"<guard-rejected:{index}>"
        for index, candidate in enumerate(candidates)
    ]
    counts = {signature: signatures.count(signature) for signature in signatures}
    majority_signature = next(
        (signature for signature in signatures if counts[signature] >= 3), None
    )
    original = candidates[0]
    assert original is not None
    if majority_signature is None:
        return original, False, False, signatures
    majority_indices = [
        index
        for index, signature in enumerate(signatures)
        if signature == majority_signature and candidates[index] is not None
    ]
    if not majority_indices:
        return original, False, False, signatures
    selected_index = 0 if 0 in majority_indices else majority_indices[0]
    selected = candidates[selected_index]
    assert selected is not None
    return selected, True, signatures[0] != majority_signature, signatures


def _terminal_consensus_signature(action: dict[str, Any]) -> str:
    if _action_is_pure_respond(action):
        return "<respond>"
    if action.get("action") == "tool_calls" and action.get("tool_calls"):
        return _mutation_consensus_signature(action)
    return "<other>"


def _select_terminal_consensus(
    candidates: list[dict[str, Any] | None],
) -> tuple[dict[str, Any], str, list[str]]:
    """Keep original text on respond mode; override only on exact action mode."""

    if len(candidates) != 3 or candidates[0] is None:
        raise ValueError("terminal consensus requires one original and two samples")
    original = candidates[0]
    assert original is not None and _action_is_pure_respond(original)
    signatures = [
        _terminal_consensus_signature(candidate)
        if candidate is not None
        else f"<guard-rejected:{index}>"
        for index, candidate in enumerate(candidates)
    ]
    if signatures.count("<respond>") >= 2:
        return original, "respond_majority", signatures
    counts = {signature: signatures.count(signature) for signature in signatures}
    majority_signature = next(
        (
            signature
            for signature in signatures
            if signature not in {"<respond>", "<other>"}
            and not signature.startswith("<guard-rejected:")
            and counts[signature] >= 2
        ),
        None,
    )
    if majority_signature is not None:
        selected_index = signatures.index(majority_signature)
        selected = candidates[selected_index]
        assert selected is not None
        return selected, "action_majority", signatures
    return original, "no_majority", signatures


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.casefold().replace("_", " "))
    normalized: set[str] = set()
    for token in raw:
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        normalized.add(token)
    return normalized


def _name_core_tokens(name: str) -> set[str]:
    return _tokens(name) - _NAME_STOPWORDS


_PHASE_REFERENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "be",
    "do",
    "for",
    "i",
    "it",
    "me",
    "my",
    "now",
    "of",
    "on",
    "off",
    "please",
    "that",
    "the",
    "this",
    "to",
    "you",
}


def _phase_tokens_overlap(left: set[str], right: set[str]) -> bool:
    """Conservative lexical/stem overlap without a subsystem lookup table."""

    if left & right:
        return True
    return any(
        len(a) >= 4
        and len(b) >= 4
        and (a.startswith(b) or b.startswith(a))
        for a in left
        for b in right
    )


def _phase_reference_tokens(messages: list[dict[str, Any]]) -> set[str]:
    # Full history is retained. Assistant proposals are included so a terse
    # approval such as "yes" retains the subsystem explicitly proposed just
    # before it; tool-result payloads are excluded because their field inventory
    # is not a user/assistant reference to requested scope.
    text = " ".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") in {"user", "assistant"}
    )
    return _tokens(text) - _PHASE_REFERENCE_STOPWORDS


def gate_catalog_by_phase(
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    pairings: dict[str, str],
) -> tuple[list[dict[str, Any]], tuple[str, ...], str | None, set[str]]:
    """Keep reads and only confidently referenced mutation subsystems.

    Any classification uncertainty fails open to the supplied full catalog.
    The function is task-free: subsystem evidence is derived from catalog names,
    argument names, read pairings, and the full user/assistant text history.
    """

    references = _phase_reference_tokens(messages)
    mutations: list[tuple[dict[str, Any], str, set[str], str | None]] = []
    for tool in tools:
        name = str(tool.get("function", {}).get("name") or "")
        if not _is_mutating_tool_name(name):
            continue
        tokens = _name_core_tokens(name) | _parameter_name_tokens(tool)
        read_name = pairings.get(name)
        if read_name:
            tokens |= _name_core_tokens(read_name)
        tokens -= _PHASE_REFERENCE_STOPWORDS
        if not tokens:
            return tools, (), f"unclassified_mutation:{name}", references
        mutations.append((tool, name, tokens, read_name))
    if not mutations:
        return tools, (), None, references
    if not references:
        return tools, (), "no_episode_subsystem_reference", references

    directly_matched = {
        name
        for _, name, tokens, _ in mutations
        if _phase_tokens_overlap(tokens, references)
    }
    if not directly_matched:
        return tools, (), "no_confident_subsystem_match", references
    matched_read_groups = {
        read_name
        for _, name, _, read_name in mutations
        if name in directly_matched and read_name is not None
    }
    allowed_mutations = {
        name
        for _, name, _, read_name in mutations
        if name in directly_matched
        or (read_name is not None and read_name in matched_read_groups)
    }
    withheld = tuple(
        sorted(name for _, name, _, _ in mutations if name not in allowed_mutations)
    )
    if not withheld:
        return tools, (), None, references
    kept = [
        tool
        for tool in tools
        if not _is_mutating_tool_name(
            str(tool.get("function", {}).get("name") or "")
        )
        or str(tool.get("function", {}).get("name") or "")
        in allowed_mutations
    ]
    return kept, withheld, None, references


def _schema_tokens(tool: dict[str, Any]) -> set[str]:
    function = tool.get("function", {})
    return _tokens(
        " ".join(
            (
                str(function.get("name") or ""),
                str(function.get("description") or ""),
                json.dumps(function.get("parameters") or {}, ensure_ascii=False),
            )
        )
    )


def _parameter_name_tokens(tool: dict[str, Any]) -> set[str]:
    properties = (
        tool.get("function", {}).get("parameters", {}).get("properties", {})
    )
    if not isinstance(properties, dict):
        return set()
    return {
        token
        for name in properties
        if isinstance(name, str)
        for token in _tokens(name)
    }


def _safe_read_arguments(tool: dict[str, Any]) -> dict[str, Any] | None:
    schema = tool.get("function", {}).get("parameters") or {}
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    arguments: dict[str, Any] = {}
    for name in required:
        prop = properties.get(name)
        if not isinstance(prop, dict) or "default" not in prop:
            return None
        arguments[name] = prop["default"]
    return arguments


_RELATIVE_NUMERIC_REQUEST = re.compile(
    r"\b(?:turn\s+(?:it|them|the\s+\w+)?\s*(?:up|down)|increase|decrease|"
    r"lower|raise|reduce|warmer|cooler|brighter|dimmer)\b",
    re.IGNORECASE,
)
_EXPLICIT_NUMBER = re.compile(
    r"\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
    re.IGNORECASE,
)
_VAGUE_DEGREE_REQUEST = re.compile(
    r"\b(?:partially|partly|somewhat|some|slightly|halfway)\b|\ba\s+bit\b",
    re.IGNORECASE,
)
_PERCENTAGE_OR_LEVEL_FIELD = re.compile(
    r"(?:^|_)(?:percentage|percent|level)(?:$|_)", re.IGNORECASE
)
_PLACEHOLDER_ARGUMENT = re.compile(
    r"\{\{[^{}]+\}\}|<[^<>]+>|placeholder|to_be_filled|from_previous_call",
    re.IGNORECASE,
)
_DEGREE_WORDS = {"partially", "partly", "somewhat", "some", "slightly", "halfway"}
_DEGREE_BRIDGE_WORDS = {
    "a",
    "an",
    "be",
    "close",
    "down",
    "it",
    "less",
    "lower",
    "make",
    "more",
    "move",
    "open",
    "only",
    "raise",
    "set",
    "the",
    "them",
    "turn",
    "up",
}
_NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
_STORED_PREFERENCE_REQUEST = re.compile(
    r"\b(?:default|preference|preferred|saved|stored)\b", re.IGNORECASE
)
_EXACT_VALUE_QUESTION = re.compile(
    r"\b(?:exact\s+value|what\s+(?:percentage|percent|level|temperature|value)|"
    r"how\s+much|which\s+(?:percentage|level|setting))\b",
    re.IGNORECASE,
)
_AM_PM_TIME = re.compile(
    r"\b(?:0?[1-9]|1[0-2])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_PERFORMED_ACTION_CLAIM = re.compile(
    r"(?:"
    r"\b(?:i|we)(?:(?:['’]ve)|\s+have)\s+"
    r"(?:set|changed|activated|updated|deleted|added|replaced|selected|"
    r"scheduled|sent|called)\s+(?:the\s+|your\s+|a\s+|an\s+|that\s+|it\s+)?"
    r"[a-z0-9]"
    r"|\b(?:i|we)\s+(?:set|changed|activated|updated|deleted|added|replaced|"
    r"selected|scheduled|sent|called)\s+"
    r"(?:the\s+|your\s+|a\s+|an\s+|that\s+|it\s+)?[a-z0-9]"
    r"|\b(?:i|we)(?:(?:['’]m)|\s+(?:am|are))\s+"
    r"(?:placing|starting|navigating|sending|calling|scheduling|updating|"
    r"deleting|adding|replacing|selecting)\s+(?:the\s+|your\s+|a\s+|an\s+|"
    r"that\s+|it\s+)?[a-z0-9]"
    r"|\b(?:i|we)(?:(?:['’]ve)|\s+have)?\s+turned\s+(?:on|off)\s+"
    r"(?:the\s+|your\s+|a\s+|an\s+)?[a-z0-9]"
    r"|\b(?:set|changed|activated|updated|deleted|added|replaced|selected|"
    r"scheduled|sent|called)\s+(?:the|your|that)\s+[a-z0-9]"
    r"|\b(?:placing|starting|navigating|sending|calling|scheduling|updating|"
    r"deleting|adding|replacing|selecting)\s+(?:the|your|a|an|that)\s+[a-z0-9]"
    r"|\bturned\s+(?:on|off)\s+(?:the|your)\s+[a-z0-9]"
    r"|\b(?:done|all\s+set)\s*[-—,:]\s*(?:the|your)\s+[a-z0-9]"
    r")",
    re.IGNORECASE,
)


def _numeric_mutation_targets(
    action: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if action.get("action") != "tool_calls":
        return targets
    for call_index, call in enumerate(action.get("tool_calls") or []):
        tool_name = str(call.get("tool_name") or "")
        if not _is_mutating_tool_name(tool_name):
            continue
        arguments = call.get("arguments") or {}
        tool = tools_by_name.get(tool_name) or {}
        properties = (
            tool.get("function", {}).get("parameters", {}).get("properties", {})
        )
        for field, value in arguments.items():
            schema = properties.get(field) or {}
            if schema.get("type") not in {"integer", "number"} or isinstance(
                value, bool
            ):
                continue
            description = str(schema.get("description") or "")
            targets.append(
                {
                    "call_index": call_index,
                    "tool_name": tool_name,
                    "field": str(field),
                    "value": value,
                    "schema": schema,
                    "device_concepts": _name_core_tokens(tool_name),
                    "field_concepts": _tokens(str(field)) | _tokens(description),
                }
            )
    return targets


def _injected_exact_ask_has_numeric_user_target(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
) -> bool:
    """Admit an injected exact-value ask only for a user-grounded numeric field.

    A model may draft numeric support mutations (for example a guessed fan
    level while turning on a boolean AC control).  That does not make the
    user's request numerically ambiguous.  Numeric schema plus a user mention
    of the pending device/value concept are both required.
    """

    targets = _numeric_mutation_targets(action, tools_by_name)
    if not targets:
        return False
    user_tokens = _tokens(_all_user_text(messages))
    generic = {
        "all", "change", "control", "exact", "level", "make", "on", "off",
        "percent", "percentage", "set", "setting", "turn", "value",
    }
    for target in targets:
        concepts = _target_concepts(target) - generic
        if concepts & user_tokens:
            return True
    return False


def _p3_ask_gate_v2_allows(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
) -> bool:
    """Allow P3's synthetic ask only for a truly source-less required number."""

    for target in _numeric_mutation_targets(action, tools_by_name):
        tool = tools_by_name.get(str(target["tool_name"])) or {}
        schema = tool.get("function", {}).get("parameters", {})
        required = set(schema.get("required") or [])
        if target["field"] not in required:
            continue
        field_schema = target.get("schema") or {}
        if "default" in field_schema:
            continue
        if _user_supplied_value_for_target(messages, target):
            continue
        concepts = _target_concepts(target)
        sourced = False
        for message in messages:
            if message.get("role") not in {"system", "tool", "assistant"}:
                continue
            text = str(message.get("content") or "")
            if _number_mentions(text) and concepts & _tokens(text):
                sourced = True
                break
        if not sourced:
            return True
    return False


def _target_concepts(target: dict[str, Any]) -> set[str]:
    return set(target["device_concepts"]) | set(target["field_concepts"])


def _number_mentions(text: str) -> list[tuple[int, int | float, str]]:
    words = list(re.finditer(r"[a-z0-9]+(?:\.\d+)?", text.casefold()))
    mentions: list[tuple[int, int | float, str]] = []
    for index, match in enumerate(words):
        token = match.group()
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            value: int | float = float(token)
            if value.is_integer():
                value = int(value)
            mentions.append((index, value, match.group()))
        elif token in _NUMBER_WORD_VALUES:
            mentions.append((index, _NUMBER_WORD_VALUES[token], match.group()))
    return mentions


def _text_mentions_target_value(text: str, target: dict[str, Any]) -> bool:
    words = _ordered_words(text)
    concepts = _target_concepts(target)
    concept_positions = [i for i, word in enumerate(words) if word in concepts]
    mentions = _number_mentions(text)
    return bool(
        mentions
        and (
            (len(mentions) == 1 and len(concept_positions) == 1)
            or any(
                abs(number_index - concept_index) <= 8
                for number_index, _, _ in mentions
                for concept_index in concept_positions
            )
        )
    )


def _user_supplied_value_for_target(
    messages: list[dict[str, Any]], target: dict[str, Any]
) -> bool:
    return any(
        message.get("role") == "user"
        and _text_mentions_target_value(str(message.get("content") or ""), target)
        for message in messages
    )


def _latest_clarification_answer(
    messages: list[dict[str, Any]], target: dict[str, Any]
) -> tuple[int | float, str] | None:
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return None
    latest_text = str(messages[latest_user_index].get("content") or "")
    mentions = _number_mentions(latest_text)
    if not mentions:
        return None
    question = next(
        (
            str(messages[index].get("content") or "")
            for index in range(latest_user_index - 1, -1, -1)
            if messages[index].get("role") == "assistant"
            and str(messages[index].get("content") or "").strip()
        ),
        "",
    )
    if "?" not in question or not _EXACT_VALUE_QUESTION.search(question):
        return None
    question_tokens = _tokens(question)
    if not (question_tokens & _target_concepts(target)) and len(mentions) != 1:
        return None
    if len(mentions) == 1:
        _, value, literal = mentions[0]
        return value, literal
    words = _ordered_words(latest_text)
    concepts = _target_concepts(target)
    ranked = sorted(
        (
            min(
                (abs(index - pos) for pos, word in enumerate(words) if word in concepts),
                default=10_000,
            ),
            value,
            literal,
        )
        for index, value, literal in mentions
    )
    return (ranked[0][1], ranked[0][2]) if ranked[0][0] <= 8 else None


def _clarification_answer_binding_fault(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
) -> InternalFault | None:
    for target in _numeric_mutation_targets(action, tools_by_name):
        answer = _latest_clarification_answer(messages, target)
        if answer is None:
            continue
        value, literal = answer
        drafted = target["value"]
        if isinstance(drafted, (int, float)) and math.isclose(
            float(drafted), float(value), abs_tol=1e-9
        ):
            continue
        return InternalFault(
            "value_provenance",
            "The user's literal clarification answer was "
            f"{literal!r} for {target['field']} on {target['tool_name']}. "
            f"Use that literal numeric value ({value}) unchanged; do not "
            "invert, complement, reinterpret, or normalize it.",
        )
    return None


def _navigation_is_active(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool" or str(message.get("name") or "") != (
            "get_current_navigation_state"
        ):
            continue
        payload = _successful_result_payload(message)
        if isinstance(payload, dict) and payload.get("navigation_active") is True:
            return True
    return False


def _set_new_navigation_while_active(
    action: dict[str, Any], messages: list[dict[str, Any]]
) -> bool:
    return bool(
        _navigation_is_active(messages)
        and action.get("action") == "tool_calls"
        and any(
            str(call.get("tool_name") or "").casefold() == "set_new_navigation"
            for call in action.get("tool_calls") or []
        )
    )


def _respond_has_am_pm_time(action: dict[str, Any]) -> bool:
    return bool(
        action.get("action") == "respond"
        and _AM_PM_TIME.search(str(action.get("content") or ""))
    )


def _respond_claims_performed_action(action: dict[str, Any]) -> bool:
    return bool(
        action.get("action") == "respond"
        and _PERFORMED_ACTION_CLAIM.search(str(action.get("content") or ""))
    )


_SEAT_POSITION_VALUES = {
    "DRIVER",
    "PASSENGER",
    "DRIVER_REAR",
    "PASSENGER_REAR",
    "ALL",
}


def _tool_was_called(messages: list[dict[str, Any]], tool_name: str) -> bool:
    target = tool_name.casefold()
    for message in messages:
        if (
            message.get("role") == "tool"
            and str(message.get("name") or "").casefold() == target
        ):
            return True
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name") or call.get("tool_name") or ""
            if str(name).casefold() == target:
                return True
    return False


def _schema_contains_seat_position(schema: dict[str, Any]) -> bool:
    enum = {str(value).upper() for value in schema.get("enum") or []}
    if enum & _SEAT_POSITION_VALUES:
        return True
    for branch in schema.get("anyOf") or schema.get("oneOf") or []:
        if isinstance(branch, dict) and _schema_contains_seat_position(branch):
            return True
    return False


def _seat_scoped_mutation(
    action: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> bool:
    if action.get("action") != "tool_calls":
        return False
    for call in action.get("tool_calls") or []:
        tool_name = str(call.get("tool_name") or "")
        if not _is_mutating_tool_name(tool_name):
            continue
        tool = tools_by_name.get(tool_name) or {}
        properties = (
            tool.get("function", {}).get("parameters", {}).get("properties", {})
        )
        if any(
            isinstance(schema, dict) and _schema_contains_seat_position(schema)
            for schema in properties.values()
        ):
            return True
        # Fail closed for the two published seat-scoped mutation families if a
        # provider catalog omits enum metadata while retaining their stable names.
        normalized = tool_name.casefold()
        if normalized in {"set_reading_light", "set_seat_heating"}:
            return True
    return False


def _apply_seat_occupancy_gate(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Ground each seat-scoped mutation in one episode occupancy read.

    This is the seat analogue of numeric value provenance: per-seat actions
    must be based on observed occupancy.  The injection is capped at one per
    episode, so an unavailable or failed read cannot create a loop.
    """

    read_name = "get_seats_occupancy"
    if (
        state.value_p5_occupancy_reads > 0
        or read_name not in state.tools_by_name
        or _tool_was_called(messages, read_name)
        or not _seat_scoped_mutation(action, state.tools_by_name)
    ):
        return action, False
    arguments = _safe_read_arguments(state.tools_by_name[read_name])
    if arguments is None:
        return action, False
    return (
        {
            "action": "tool_calls",
            "tool_calls": [{"tool_name": read_name, "arguments": arguments}],
        },
        True,
    )


def _apply_injected_ask_budget(
    proposed_action: dict[str, Any],
    *,
    original_action: dict[str, Any],
    state: _EpisodeState,
    tally: _CallTally,
    ctx_logger: Any,
    ask_path: str,
) -> tuple[dict[str, Any], bool]:
    """Admit at most one deterministic clarification ask per episode.

    A rejected ask restores the model's original draft verbatim.  There is no
    replacement response or instruction: later decisions therefore fall
    through to unmodified model judgment.  This budget is deliberately
    episode-scoped so a non-numeric reply can never trigger an injected
    re-ask, while P2 can still bind a literal numeric reply if one arrives.
    """

    if state.injected_asks >= 1:
        state.ask_budget_suppressed += 1
        tally.ask_budget_suppressed += 1
        ctx_logger.info(
            "Adaptive-minimal injected ask suppressed by episode budget",
            ask_path=ask_path,
            injected_asks_episode=state.injected_asks,
            suppressions_episode=state.ask_budget_suppressed,
        )
        return original_action, False
    state.injected_asks += 1
    tally.injected_asks += 1
    ctx_logger.info(
        "Adaptive-minimal injected ask admitted by episode budget",
        ask_path=ask_path,
        injected_asks_episode=state.injected_asks,
    )
    return proposed_action, True


def _apply_pre_mutation_guards(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
    config: AdaptiveMinimalConfig,
    tally: _CallTally,
    ctx_logger: Any,
) -> dict[str, Any]:
    value_kind: str | None = None
    if config.value_provenance:
        action, occupancy_read = _apply_seat_occupancy_gate(
            action, state=state, messages=messages
        )
        if occupancy_read:
            state.value_p5_occupancy_reads += 1
            tally.value_p5_occupancy_reads += 1
            ctx_logger.info(
                "Adaptive-minimal value provenance P5 occupancy read injected",
                reads_episode=state.value_p5_occupancy_reads,
            )
            return action
        pre_value_action = action
        action, value_kind = _apply_value_provenance_before_mutation(
            action, state=state, messages=messages
        )
        if value_kind == "p1_context_apply":
            state.value_p1_context_applies += 1
            tally.value_p1_context_applies += 1
            ctx_logger.info(
                "Adaptive-minimal value provenance P1 context value applied",
                applies_episode=state.value_p1_context_applies,
            )
        elif value_kind == "p3_preference_read":
            state.value_p3_preference_reads += 1
            tally.value_p3_preference_reads += 1
            ctx_logger.info(
                "Adaptive-minimal value provenance P3 preference read injected",
                reads_episode=state.value_p3_preference_reads,
            )
        elif value_kind == "p3_fallback_ask":
            if config.p3_ask_gate_v2 and not _p3_ask_gate_v2_allows(
                pre_value_action,
                messages=messages,
                tools_by_name=state.tools_by_name,
            ):
                action = pre_value_action
                value_kind = None
                state.p3_ask_gate_v2_suppressions += 1
                tally.p3_ask_gate_v2_suppressions += 1
                ctx_logger.info(
                    "Adaptive-minimal P3 ask gate v2 suppressed fallback ask",
                    suppressions_episode=state.p3_ask_gate_v2_suppressions,
                )
            elif config.ask_type_gate and not _injected_exact_ask_has_numeric_user_target(
                pre_value_action,
                messages=messages,
                tools_by_name=state.tools_by_name,
            ):
                action = pre_value_action
                value_kind = None
                state.ask_type_gate_suppressions += 1
                tally.ask_type_gate_suppressions += 1
                ctx_logger.info(
                    "Adaptive-minimal ask-type gate suppressed exact-value ask",
                    ask_path="p3_fallback_exact",
                    suppressions_episode=state.ask_type_gate_suppressions,
                )
            if config.ask_budget:
                if value_kind == "p3_fallback_ask":
                    action, admitted = _apply_injected_ask_budget(
                        action,
                        original_action=pre_value_action,
                        state=state,
                        tally=tally,
                        ctx_logger=ctx_logger,
                        ask_path="p3_fallback_exact",
                    )
                    if not admitted:
                        value_kind = None
            if value_kind == "p3_fallback_ask":
                state.value_p3_fallback_asks += 1
                tally.value_p3_fallback_asks += 1
                ctx_logger.info(
                    "Adaptive-minimal value provenance P3 fallback ask emitted",
                    asks_episode=state.value_p3_fallback_asks,
                )
    if config.vague_degree_clarify and value_kind != "p1_context_apply":
        pre_vague_action = action
        action, vague_kind = _apply_vague_degree_preference_gate(
            action, state=state, messages=messages
        )
        if vague_kind == "clarification":
            if config.ask_type_gate and not _injected_exact_ask_has_numeric_user_target(
                pre_vague_action,
                messages=messages,
                tools_by_name=state.tools_by_name,
            ):
                action = pre_vague_action
                vague_kind = None
                state.ask_type_gate_suppressions += 1
                tally.ask_type_gate_suppressions += 1
                ctx_logger.info(
                    "Adaptive-minimal ask-type gate suppressed exact-value ask",
                    ask_path="vague_degree_clarification",
                    suppressions_episode=state.ask_type_gate_suppressions,
                )
            if config.ask_budget:
                if vague_kind == "clarification":
                    action, admitted = _apply_injected_ask_budget(
                        action,
                        original_action=pre_vague_action,
                        state=state,
                        tally=tally,
                        ctx_logger=ctx_logger,
                        ask_path="vague_degree_clarification",
                    )
                    if not admitted:
                        vague_kind = None
            if vague_kind == "clarification":
                state.vague_degree_clarifications += 1
                tally.vague_degree_clarifications += 1
                ctx_logger.info(
                    "Adaptive-minimal vague-degree clarification fired",
                    clarifications_episode=state.vague_degree_clarifications,
                )
        elif vague_kind == "preference_redirect":
            state.vague_degree_preference_redirects += 1
            tally.vague_degree_preference_redirects += 1
            ctx_logger.info(
                "Adaptive-minimal vague-degree preference read redirected",
                redirects_episode=state.vague_degree_preference_redirects,
            )
        elif vague_kind == "preference_apply":
            state.vague_degree_preference_applies += 1
            tally.vague_degree_preference_applies += 1
            ctx_logger.info(
                "Adaptive-minimal vague-degree stored preference applied",
                applies_episode=state.vague_degree_preference_applies,
            )
    if config.argument_binding_guard:
        pre_binding_action = action
        guarded_action, binding_kind = _apply_argument_binding_guard(
            action, state=state, messages=messages
        )
        if (
            config.value_provenance
            and value_kind == "p1_context_apply"
            and binding_kind == "relative_clarification"
        ):
            action = pre_binding_action
            binding_kind = None
            state.value_p1_ask_suppressions += 1
            tally.value_p1_ask_suppressions += 1
            ctx_logger.info(
                "Adaptive-minimal value provenance P1 redundant ask suppressed",
                suppressions_episode=state.value_p1_ask_suppressions,
            )
        else:
            if (
                config.ask_type_gate
                and binding_kind == "relative_clarification"
                and not _injected_exact_ask_has_numeric_user_target(
                    pre_binding_action,
                    messages=messages,
                    tools_by_name=state.tools_by_name,
                )
            ):
                action = pre_binding_action
                binding_kind = None
                state.ask_type_gate_suppressions += 1
                tally.ask_type_gate_suppressions += 1
                ctx_logger.info(
                    "Adaptive-minimal ask-type gate suppressed exact-value ask",
                    ask_path="relative_numeric_clarification",
                    suppressions_episode=state.ask_type_gate_suppressions,
                )
            elif config.ask_budget and binding_kind == "relative_clarification":
                action, admitted = _apply_injected_ask_budget(
                    guarded_action,
                    original_action=pre_binding_action,
                    state=state,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    ask_path="relative_numeric_clarification",
                )
                if not admitted:
                    binding_kind = None
            else:
                action = guarded_action
        if binding_kind == "relative_clarification":
            state.argument_binding_relative_clarifications += 1
            tally.argument_binding_relative_clarifications += 1
            ctx_logger.info(
                "Adaptive-minimal argument-binding clarification fired",
                clarifications_episode=(
                    state.argument_binding_relative_clarifications
                ),
            )
        elif binding_kind == "route_correction":
            state.argument_binding_route_corrections += 1
            tally.argument_binding_route_corrections += 1
            ctx_logger.info(
                "Adaptive-minimal route-lineage correction fired",
                corrections_episode=state.argument_binding_route_corrections,
            )
    if config.disclosure_guard:
        action, confirmation_reask = _apply_confirmation_disclosure_guard(
            action, state=state, messages=messages
        )
        if confirmation_reask:
            state.disclosure_confirmation_reasks += 1
            tally.disclosure_confirmation_reasks += 1
            ctx_logger.info(
                "Adaptive-minimal exact confirmation guard fired",
                reasks_episode=state.disclosure_confirmation_reasks,
            )
    return action


def _all_user_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    )


def _percentage_or_level_arguments(
    action: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> list[str]:
    return sorted(
        {str(target["field"]) for target in _percentage_or_level_targets(action, tools_by_name)}
    )


def _percentage_or_level_targets(
    action: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if action.get("action") != "tool_calls":
        return targets
    for call_index, call in enumerate(action.get("tool_calls") or []):
        tool_name = str(call.get("tool_name") or "")
        if not _is_mutating_tool_name(tool_name):
            continue
        tool = tools_by_name.get(tool_name) or {}
        properties = (
            tool.get("function", {}).get("parameters", {}).get("properties", {})
        )
        for name, value in (call.get("arguments") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            schema = properties.get(name) if isinstance(properties, dict) else None
            description = str((schema or {}).get("description") or "")
            if _PERCENTAGE_OR_LEVEL_FIELD.search(str(name)) or re.search(
                r"\b(?:percentage|percent|level)\b", description, re.IGNORECASE
            ):
                device_concepts = _name_core_tokens(tool_name)
                field_concepts = _tokens(str(name)) | _tokens(description)
                targets.append(
                    {
                        "call_index": call_index,
                        "field": str(name),
                        "schema": schema or {},
                        "device_concepts": device_concepts,
                        "field_concepts": field_concepts,
                    }
                )
    return targets


def _ordered_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold().replace("_", " "))


def _degree_modifies_target(user_text: str, target: dict[str, Any]) -> bool:
    words = _ordered_words(user_text)
    concepts = set(target["device_concepts"]) | set(target["field_concepts"])
    degree_positions = [
        index for index, word in enumerate(words) if word in _DEGREE_WORDS
    ]
    degree_positions.extend(
        index + 1
        for index in range(len(words) - 1)
        if words[index : index + 2] == ["a", "bit"]
    )
    concept_positions = [
        index for index, word in enumerate(words) if word in concepts
    ]
    for degree in degree_positions:
        for concept in concept_positions:
            if degree < concept and concept - degree <= 3:
                if set(words[degree + 1 : concept]) <= _DEGREE_BRIDGE_WORDS:
                    return True
            elif concept < degree and degree - concept <= 3:
                if set(words[concept + 1 : degree]) <= _DEGREE_BRIDGE_WORDS:
                    return True
    return False


def _preference_tool_was_called(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "tool"
        and str(message.get("name") or "") == "get_user_preferences"
        for message in messages
    )


def _boolean_schema_selection(schema: dict[str, Any]) -> Any | None:
    if schema.get("type") == "boolean":
        return True
    if schema.get("type") != "object":
        return None
    selected: dict[str, Any] = {}
    for name, child_schema in (schema.get("properties") or {}).items():
        if not isinstance(child_schema, dict):
            continue
        child = _boolean_schema_selection(child_schema)
        if child is not None:
            selected[str(name)] = child
    return selected or None


def _preference_read_arguments(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Build a schema-valid, read-only request for all preference categories."""

    schema = tool.get("function", {}).get("parameters") or {"type": "object"}
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    fields = required or (
        ["preference_categories"] if "preference_categories" in properties else []
    )
    arguments: dict[str, Any] = {}
    for field in fields:
        child_schema = properties.get(field)
        if not isinstance(child_schema, dict):
            return None
        if "default" in child_schema:
            arguments[field] = child_schema["default"]
            continue
        selected = _boolean_schema_selection(child_schema)
        if selected is None:
            return None
        arguments[field] = selected
    if _validate_schema(arguments, schema, path="$arguments") is not None:
        return None
    return arguments


def _preference_text_rows(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, dict):
        rows: list[str] = []
        for key, item in value.items():
            rows.extend(_preference_text_rows(item, (*path, str(key))))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_preference_text_rows(item, path))
        return rows
    if isinstance(value, str):
        return [" ".join((*path, value))]
    return []


def _stored_preference_texts(messages: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for message in messages:
        if message.get("role") != "tool" or str(message.get("name") or "") != (
            "get_user_preferences"
        ):
            continue
        payload = _successful_result_payload(message)
        if payload is not None:
            rows.extend(_preference_text_rows(payload))
    return rows


def _numeric_value_from_text(text: str, schema: dict[str, Any]) -> int | float | None:
    match = re.search(r"(?<![a-z0-9])-?\d+(?:\.\d+)?(?![a-z0-9])", text.casefold())
    if match:
        value: int | float = float(match.group())
        if value.is_integer():
            value = int(value)
    else:
        value = next(
            (
                _NUMBER_WORD_VALUES[word]
                for word in _ordered_words(text)
                if word in _NUMBER_WORD_VALUES
            ),
            None,
        )
        if value is None:
            return None
    expected = schema.get("type")
    if expected == "integer" and not isinstance(value, int):
        return None
    if "minimum" in schema and value < schema["minimum"]:
        return None
    if "maximum" in schema and value > schema["maximum"]:
        return None
    if "enum" in schema and value not in schema["enum"]:
        return None
    return value


def _rewrite_from_stored_preference(
    action: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    preference_rows = _stored_preference_texts(messages)
    if not preference_rows:
        return None
    calls = [deepcopy(call) for call in action.get("tool_calls") or []]
    changed = False
    for target in targets:
        device = set(target["device_concepts"])
        field = set(target["field_concepts"])
        for row in preference_rows:
            row_tokens = _tokens(row)
            if not (row_tokens & device) or not (row_tokens & field):
                continue
            value = _numeric_value_from_text(row, target["schema"])
            if value is None:
                continue
            arguments = dict(calls[target["call_index"]].get("arguments") or {})
            arguments[target["field"]] = value
            calls[target["call_index"]]["arguments"] = arguments
            changed = True
            break
    return {"action": "tool_calls", "tool_calls": calls} if changed else None


def _rewrite_numeric_targets_from_preferences(
    action: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[tuple[int, str]]]:
    preference_rows = _stored_preference_texts(messages)
    calls = [deepcopy(call) for call in action.get("tool_calls") or []]
    applied: set[tuple[int, str]] = set()
    for target in targets:
        device = set(target["device_concepts"])
        field = set(target["field_concepts"])
        for row in preference_rows:
            row_tokens = _tokens(row)
            if not (row_tokens & device) or not (row_tokens & field):
                continue
            value = _numeric_value_from_text(row, target["schema"])
            if value is None:
                continue
            arguments = dict(calls[target["call_index"]].get("arguments") or {})
            arguments[target["field"]] = value
            calls[target["call_index"]]["arguments"] = arguments
            applied.add((target["call_index"], target["field"]))
            break
    return {"action": "tool_calls", "tool_calls": calls}, applied


def _user_qualitatively_specified_target(
    messages: list[dict[str, Any]], target: dict[str, Any]
) -> bool:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        if not re.search(r"\b(?:maximum|max|minimum|min|off|zero)\b", text, re.I):
            continue
        words = _ordered_words(text)
        concepts = _target_concepts(target)
        qualitative = [
            index
            for index, word in enumerate(words)
            if word in {"maximum", "max", "minimum", "min", "off", "zero"}
        ]
        targets = [index for index, word in enumerate(words) if word in concepts]
        if any(abs(left - right) <= 8 for left in qualitative for right in targets):
            return True
    return False


def _apply_value_provenance_before_mutation(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    """Require a source for every numeric mutation value.

    Provenance is ordered and task-free: a literal user value (including a
    literal clarification answer), an applicable stored preference, or an
    exact-value clarification after a preference read found nothing.  A value
    already present in context is applied and must never be asked for again.
    """

    targets = _numeric_mutation_targets(action, state.tools_by_name)
    if not targets:
        return action, None
    rewritten, preference_applied = _rewrite_numeric_targets_from_preferences(
        action, targets=targets, messages=messages
    )
    unresolved = [
        target
        for target in targets
        if (target["call_index"], target["field"]) not in preference_applied
        and not _user_supplied_value_for_target(messages, target)
        and not _user_qualitatively_specified_target(messages, target)
    ]
    if preference_applied and not unresolved:
        return rewritten, "p1_context_apply"
    if not unresolved:
        return action, None

    preference_tool = state.tools_by_name.get("get_user_preferences")
    if preference_tool is not None and not _preference_tool_was_called(messages):
        arguments = _preference_read_arguments(preference_tool)
        if arguments is not None:
            return (
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": "get_user_preferences",
                            "arguments": arguments,
                        }
                    ],
                },
                "p3_preference_read",
            )

    fields = sorted({str(target["field"]) for target in unresolved})
    labels = ", ".join(field.replace("_", " ") for field in fields)
    return (
        {
            "action": "respond",
            "content": f"What exact value should I use for {labels}?",
        },
        "p3_fallback_ask",
    )


def _apply_vague_degree_preference_gate(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    targets = _percentage_or_level_targets(action, state.tools_by_name)
    if not targets or _EXPLICIT_NUMBER.search(_all_user_text(messages)):
        return action, None

    latest_user = _latest_user_text(messages)
    relevant = any(_degree_modifies_target(latest_user, target) for target in targets)
    preference_requested = bool(_STORED_PREFERENCE_REQUEST.search(latest_user))
    if not relevant and not preference_requested:
        return action, None

    rewritten = _rewrite_from_stored_preference(
        action, targets=targets, messages=messages
    )
    if rewritten is not None:
        return rewritten, "preference_apply"

    user_turn = sum(message.get("role") == "user" for message in messages)
    if (
        "get_user_preferences" in state.tools_by_name
        and not _preference_tool_was_called(messages)
        and user_turn not in state.vague_degree_preference_turn_reads
    ):
        preference_arguments = _preference_read_arguments(
            state.tools_by_name["get_user_preferences"]
        )
        if preference_arguments is not None:
            state.vague_degree_preference_turn_reads.add(user_turn)
            return (
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": "get_user_preferences",
                            "arguments": preference_arguments,
                        }
                    ],
                },
                "preference_redirect",
            )

    if relevant and user_turn not in state.vague_degree_turn_fires:
        state.vague_degree_turn_fires.add(user_turn)
        fields = sorted({str(target["field"]) for target in targets})
        labels = ", ".join(field.replace("_", " ") for field in fields)
        return (
            {
                "action": "respond",
                "content": f"What exact value should I use for {labels}?",
            },
            "clarification",
        )
    return action, None


def _apply_vague_degree_clarification(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Ask once before guessing a vague percentage/level mutation value."""

    guarded, kind = _apply_vague_degree_preference_gate(
        action, state=state, messages=messages
    )
    return guarded, kind == "clarification"


def _value_contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_PLACEHOLDER_ARGUMENT.search(value))
    if isinstance(value, dict):
        return any(_value_contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_value_contains_placeholder(item) for item in value)
    return False


def _split_calls_before_placeholder(
    action: dict[str, Any],
) -> tuple[dict[str, Any] | None, int | None]:
    """Keep only concrete calls before the first placeholder-bearing call."""

    if action.get("action") != "tool_calls":
        return action, None
    calls = list(action.get("tool_calls") or [])
    for index, call in enumerate(calls):
        if _value_contains_placeholder(call.get("arguments") or {}):
            if index == 0:
                return None, index
            return {"action": "tool_calls", "tool_calls": calls[:index]}, index
    return action, None


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    return next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )


def _apply_argument_binding_guard(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    if action.get("action") != "tool_calls":
        return action, None
    calls = action.get("tool_calls") or []
    user_text = _latest_user_text(messages)
    user_turn = sum(message.get("role") == "user" for message in messages)

    numeric_arguments = sorted(
        {
            str(name)
            for call in calls
            if _is_mutating_tool_name(str(call.get("tool_name") or ""))
            for name, value in (call.get("arguments") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    if (
        numeric_arguments
        and _RELATIVE_NUMERIC_REQUEST.search(user_text)
        and not _EXPLICIT_NUMBER.search(user_text)
        and user_turn not in state.argument_binding_turn_fires
    ):
        state.argument_binding_turn_fires.add(user_turn)
        fields = ", ".join(name.replace("_", " ") for name in numeric_arguments)
        return (
            {
                "action": "respond",
                "content": (
                    "What exact value should I use for "
                    f"{fields} on the requested controls?"
                ),
            },
            "relative_clarification",
        )

    corrected = _correct_active_navigation_lineage(
        action, messages=messages, tools_by_name=state.tools_by_name
    )
    if corrected is not None:
        return corrected, "route_correction"
    return action, None


def _successful_result_payload(message: dict[str, Any]) -> Any | None:
    if message.get("role") != "tool":
        return None
    content = str(message.get("content") or "")
    succeeded, _ = _tool_result_status(content)
    if not succeeded:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and "result" in payload:
        return payload.get("result")
    return payload


def _active_navigation_start(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "tool" or str(message.get("name") or "") != (
            "get_current_navigation_state"
        ):
            continue
        result = _successful_result_payload(message)
        if not isinstance(result, dict) or result.get("navigation_active") is not True:
            continue
        waypoints = result.get("waypoints_id") or result.get("waypoints") or []
        if isinstance(waypoints, list) and waypoints:
            return str(waypoints[0])
    return None


def _known_routes(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        result = _successful_result_payload(message)
        if not isinstance(result, dict) or not isinstance(result.get("routes"), list):
            continue
        for route in result["routes"]:
            if isinstance(route, dict) and route.get("route_id"):
                routes[str(route["route_id"])] = route
    return routes


def _navigation_edit_requested(user_text: str) -> bool:
    normalized = user_text.casefold()
    return bool(
        re.search(r"\b(?:change|replace|update|edit|reroute|start)\b", normalized)
        and re.search(r"\b(?:navigation|route|destination)\b", normalized)
    )


def _route_read_action(
    *, start_id: str, destination_id: str, tools_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    name = "get_routes_from_start_to_destination"
    if name not in tools_by_name:
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": name,
                "arguments": {
                    "start_id": start_id,
                    "destination_id": destination_id,
                },
            }
        ],
    }


def _correct_active_navigation_lineage(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    active_start = _active_navigation_start(messages)
    if active_start is None:
        return None
    calls = action.get("tool_calls") or []
    latest_user = _latest_user_text(messages)

    if _navigation_edit_requested(latest_user):
        corrected_calls = [dict(call) for call in calls]
        changed = False
        for call in corrected_calls:
            if str(call.get("tool_name") or "") != (
                "get_routes_from_start_to_destination"
            ):
                continue
            arguments = dict(call.get("arguments") or {})
            if arguments.get("start_id") == active_start:
                continue
            if not arguments.get("destination_id"):
                continue
            arguments["start_id"] = active_start
            call["arguments"] = arguments
            changed = True
        if changed:
            return {"action": "tool_calls", "tool_calls": corrected_calls}

    routes = _known_routes(messages)
    for call_index, call in enumerate(calls):
        name = str(call.get("tool_name") or "")
        arguments = dict(call.get("arguments") or {})
        route_id = str(
            arguments.get("route_id_leading_to_new_destination")
            or next(iter(arguments.get("route_ids") or []), "")
        )
        route = routes.get(route_id)
        if route is None or str(route.get("start_id") or "") == active_start:
            continue
        destination_id = str(
            arguments.get("new_destination_id")
            or route.get("destination_id")
            or ""
        )
        candidates = [
            item
            for item in routes.values()
            if str(item.get("start_id") or "") == active_start
            and str(item.get("destination_id") or "") == destination_id
        ]
        if candidates and name == "navigation_replace_final_destination":
            selected = next(
                (
                    item
                    for item in candidates
                    if "fastest" in (item.get("alias") or [])
                ),
                candidates[0],
            )
            corrected_calls = [dict(item) for item in calls]
            corrected_arguments = dict(arguments)
            corrected_arguments["route_id_leading_to_new_destination"] = str(
                selected["route_id"]
            )
            corrected_calls[call_index] = {
                **call,
                "arguments": corrected_arguments,
            }
            return {"action": "tool_calls", "tool_calls": corrected_calls}
        read_action = _route_read_action(
            start_id=active_start,
            destination_id=destination_id,
            tools_by_name=tools_by_name,
        )
        if read_action is not None:
            return read_action
    return None


def _requires_confirmation_tool(tool: dict[str, Any] | None) -> bool:
    if not isinstance(tool, dict):
        return False
    description = str(tool.get("function", {}).get("description") or "")
    return description.lstrip().startswith("REQUIRES_CONFIRMATION")


def _confirmation_matches_calls(
    content: str, calls: list[dict[str, Any]]
) -> bool:
    normalized = content.casefold()
    for call in calls:
        name = str(call.get("tool_name") or "")
        if name.casefold() not in normalized:
            return False
        for key, value in (call.get("arguments") or {}).items():
            if str(key).casefold() not in normalized:
                return False
            value_text = json.dumps(value, ensure_ascii=False).casefold()
            alternatives = {value_text}
            if value is True:
                alternatives.add("on")
            elif value is False:
                alternatives.add("off")
            if not any(item in normalized for item in alternatives):
                return False
    return True


def _apply_confirmation_disclosure_guard(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if action.get("action") != "tool_calls":
        return action, False
    requiring = [
        call
        for call in action.get("tool_calls") or []
        if _requires_confirmation_tool(
            state.tools_by_name.get(str(call.get("tool_name") or ""))
        )
    ]
    if not requiring:
        return action, False
    previous = _previous_assistant_before_latest_user(messages)
    previous_content = str((previous or {}).get("content") or "")
    if _approval_event(messages) and _confirmation_matches_calls(
        previous_content, requiring
    ):
        return action, False
    details = "; ".join(
        f"{call.get('tool_name')} with parameters "
        f"{json.dumps(call.get('arguments') or {}, ensure_ascii=False, sort_keys=True)}"
        for call in requiring
    )
    return (
        {
            "action": "respond",
            "content": f"I will call {details}. Please confirm?",
        },
        True,
    )


def _requested_unavailable_labels(messages: list[dict[str, Any]]) -> list[str]:
    user_tokens = set().union(
        *(
            _tokens(str(message.get("content") or ""))
            for message in messages
            if message.get("role") == "user"
        )
    )
    labels: list[str] = []
    marker_tokens = {
        "unknown",
        "unavailable",
        "not",
        "available",
        "absent",
        "found",
        "null",
        "none",
        "error",
        "status",
        "result",
        "success",
    }
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") != "tool" or not _tool_result_is_unavailable(content):
            continue
        call = _tool_result_call(messages, message)
        tool_name = str(message.get("name") or (call or {}).get("tool_name") or "")
        related = (_name_core_tokens(tool_name) | _tokens(content)) - marker_tokens
        if related & user_tokens:
            labels.append(tool_name.replace("_", " ") or "requested information")
    return sorted(set(labels))


def _apply_unavailable_disclosure_guard(
    content: str, messages: list[dict[str, Any]]
) -> tuple[str, bool]:
    labels = _requested_unavailable_labels(messages)
    if not labels:
        return content, False
    explicit_limitation = re.search(
        r"\b(?:cannot|can't|unable to|not able to)\b.{0,120}"
        r"\b(?:retrieve|look up|lookup|check|access|provide|determine|perform)\b",
        content.casefold(),
    )
    if explicit_limitation:
        return content, False
    subjects = ", ".join(labels)
    addition = (
        f"I cannot perform the requested {subjects} lookup with the available "
        "tools, so I cannot reliably complete dependent steps without guessing."
    )
    return (f"{content.rstrip()} {addition}" if content.strip() else addition), True


def _gate_mutations(
    action: dict[str, Any], state: _EpisodeState
) -> tuple[dict[str, Any], int]:
    if action.get("action") != "tool_calls":
        return action, 0
    calls = action.get("tool_calls") or []
    read_support: dict[str, int] = {}
    for call in calls:
        name = str(call.get("tool_name") or "")
        read_name = state.pairings.get(name)
        if read_name is None or read_name in state.read_ledger:
            continue
        read_support[read_name] = read_support.get(read_name, 0) + 1
    needed_reads = set(read_support)
    if not needed_reads:
        return action, 0

    # A compound action can contain several state-changing calls in one
    # subsystem plus a secondary subsystem. Issue the uniquely dominant
    # subsystem's canonical read first, then re-gate the pending action after
    # that read succeeds. Ties remain batched; no arbitrary subsystem wins.
    if len(needed_reads) > 1:
        strongest = max(read_support.values())
        canonical = {
            read_name
            for read_name, support in read_support.items()
            if support == strongest
        }
        if len(canonical) == 1:
            needed_reads = canonical

    original_reads = {
        str(call.get("tool_name") or ""): call
        for call in calls
        if str(call.get("tool_name") or "").startswith(_READ_PREFIXES)
    }
    emitted_reads: list[dict[str, Any]] = []
    injected = 0
    for read_name in sorted(needed_reads):
        if read_name in original_reads:
            emitted_reads.append(original_reads[read_name])
            continue
        read_tool = state.tools_by_name[read_name]
        arguments = _safe_read_arguments(read_tool)
        if arguments is None:  # Defensive: derivation already excludes this.
            continue
        emitted_reads.append({"tool_name": read_name, "arguments": arguments})
        injected += 1

    if not emitted_reads:
        return action, 0
    emitted_names = {str(call.get("tool_name") or "") for call in emitted_reads}
    pending_calls = [
        call
        for call in calls
        if str(call.get("tool_name") or "") not in emitted_names
    ]
    if not pending_calls:
        return {"action": "tool_calls", "tool_calls": emitted_reads}, injected
    state.pending_action = {"action": "tool_calls", "tool_calls": pending_calls}
    state.pending_reads = set(needed_reads)
    return {"action": "tool_calls", "tool_calls": emitted_reads}, injected


def _terminal_readback_action(
    state: _EpisodeState,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build deterministic post-mutation reads without consulting task identity."""

    mutation_count = len(state.successful_mutations)
    if mutation_count <= state.terminal_readback_checked_mutations:
        return None, []
    pending = state.successful_mutations[
        state.terminal_readback_checked_mutations : mutation_count
    ]
    calls_by_name: dict[str, dict[str, Any]] = {}
    for mutation in pending:
        read_name = str(mutation.get("read_name") or "")
        read_tool = state.tools_by_name.get(read_name)
        arguments = _safe_read_arguments(read_tool) if read_tool is not None else None
        if read_name and arguments is not None:
            calls_by_name[read_name] = {
                "tool_name": read_name,
                "arguments": arguments,
            }
    if not calls_by_name:
        state.terminal_readback_checked_mutations = mutation_count
        return None, []
    calls = [calls_by_name[name] for name in sorted(calls_by_name)]
    state.terminal_readback_pending_reads = set(calls_by_name)
    state.terminal_readback_pending_mutation_count = mutation_count
    state.terminal_readback_fires += 1
    state.terminal_readback_reads += len(calls)
    return {"action": "tool_calls", "tool_calls": calls}, sorted(calls_by_name)


def _flatten_result_values(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        rows: list[tuple[tuple[str, ...], Any]] = []
        for key, item in value.items():
            rows.extend(_flatten_result_values(item, (*path, str(key))))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten_result_values(item, (*path, str(index))))
        return rows
    return [(path, value)]


def _readback_result_payload(content: str) -> Any | None:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return payload
    succeeded, error = _tool_result_status(content)
    if not succeeded or error is not None:
        return None
    return payload.get("result", payload)


def _normalized_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().replace("_", " ").split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _mutation_readback_mismatches(
    mutation: dict[str, Any], result: Any
) -> tuple[list[str], int]:
    flattened = _flatten_result_values(result)
    mismatches: list[str] = []
    unverified = 0
    for argument, expected in (mutation.get("arguments") or {}).items():
        if isinstance(expected, (dict, list)):
            unverified += 1
            continue
        argument_tokens = _tokens(str(argument))
        ranked: list[tuple[int, tuple[str, ...], Any]] = []
        for path, observed in flattened:
            if isinstance(observed, (dict, list)):
                continue
            leaf_tokens = _tokens(path[-1]) if path else set()
            path_tokens = set().union(*(_tokens(part) for part in path)) if path else set()
            score = 8 * int(leaf_tokens == argument_tokens) + 3 * len(
                argument_tokens & leaf_tokens
            ) + len(argument_tokens & path_tokens)
            if score > 0:
                ranked.append((score, path, observed))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            unverified += 1
            continue
        _, path, observed = ranked[0]
        if _normalized_scalar(observed) != _normalized_scalar(expected):
            mismatches.append(
                f"{mutation.get('tool_name')}.{argument}: expected {expected!r}, "
                f"observed {observed!r} at {'.'.join(path)}"
            )
    return mismatches, unverified


def _complete_terminal_readback(
    messages: list[dict[str, Any]], state: _EpisodeState, ctx_logger: Any
) -> InternalFault | None:
    if not state.terminal_readback_pending_reads:
        return None
    trailing: dict[str, Any] = {}
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        name = str(message.get("name") or "")
        if name in state.terminal_readback_pending_reads and name not in trailing:
            payload = _readback_result_payload(str(message.get("content") or ""))
            if payload is not None:
                trailing[name] = payload
    if not state.terminal_readback_pending_reads <= set(trailing):
        return None

    mutation_count = state.terminal_readback_pending_mutation_count
    start = state.terminal_readback_checked_mutations
    mismatches: list[str] = []
    unverified = 0
    for mutation in state.successful_mutations[start:mutation_count]:
        read_name = str(mutation.get("read_name") or "")
        if read_name not in trailing:
            unverified += len(mutation.get("arguments") or {})
            continue
        found, skipped = _mutation_readback_mismatches(
            mutation, trailing[read_name]
        )
        mismatches.extend(found)
        unverified += skipped

    state.terminal_readback_checked_mutations = mutation_count
    state.terminal_readback_pending_reads.clear()
    state.terminal_readback_pending_mutation_count = 0
    if mismatches:
        state.terminal_readback_mismatches += 1
    revise = bool(mismatches) and not state.terminal_readback_revise_fired
    if revise:
        state.terminal_readback_revise_fired = True
        state.terminal_readback_revises += 1
    ctx_logger.info(
        "Adaptive-minimal terminal read-back completed",
        mismatches=mismatches,
        unverified_fields=unverified,
        revise=revise,
        fires_episode=state.terminal_readback_fires,
        reads_episode=state.terminal_readback_reads,
        mismatches_episode=state.terminal_readback_mismatches,
        revises_episode=state.terminal_readback_revises,
    )
    if not revise:
        return None
    return InternalFault(
        SIGNAL_TERMINAL_READBACK,
        "Terminal read-back found a realized-state mismatch after mutation: "
        + "; ".join(mismatches[:4])
        + ". Revise once before the terminal response. Issue only the exact "
        "first-time-right mutation needed to satisfy the user's requested change; "
        "never speculate, undo, or touch unrelated fields.",
    )


def _canonical_tool_result(content: str) -> str:
    """Return a stable comparison key for a successful tool result."""

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return " ".join(content.split())
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _catalog_route_references(value: Any, state: _EpisodeState) -> None:
    """Catalog route endpoint provenance from successful tool results."""

    if isinstance(value, list):
        for item in value:
            _catalog_route_references(item, state)
        return
    if not isinstance(value, dict):
        return
    route_id = value.get("route_id")
    start_id = value.get("start_id")
    destination_id = value.get("destination_id")
    if all(isinstance(item, str) and item for item in (route_id, start_id, destination_id)):
        state.route_reference_catalog[str(route_id)] = (
            str(start_id),
            str(destination_id),
        )
        pair = (str(start_id), str(destination_id))
        candidates = state.route_candidates_by_pair.setdefault(pair, [])
        if not any(str(item.get("route_id") or "") == str(route_id) for item in candidates):
            # Keep only JSON-compatible route metadata already returned by the
            # environment.  The resolver never invents route candidates.
            try:
                candidate = json.loads(json.dumps(value, ensure_ascii=False))
            except (TypeError, ValueError):
                candidate = {
                    "route_id": str(route_id),
                    "start_id": str(start_id),
                    "destination_id": str(destination_id),
                }
            candidates.append(candidate)
    for item in value.values():
        _catalog_route_references(item, state)


def _ingest_navigation_waypoints(value: Any, state: _EpisodeState) -> None:
    if not isinstance(value, dict):
        return
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    waypoints = result.get("waypoints_id") if isinstance(result, dict) else None
    if isinstance(waypoints, list) and all(isinstance(item, str) for item in waypoints):
        state.navigation_waypoints = list(waypoints)


def _ingest_tool_results(
    messages: list[dict[str, Any]], state: _EpisodeState
) -> InternalFault | None:
    faults: list[str] = []
    trailing: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        trailing.append(message)
    for message in reversed(trailing):
        name = str(message.get("name") or "")
        content = str(message.get("content") or "")
        key = _tool_result_key(message)
        if key in state.processed_tool_results:
            continue
        state.processed_tool_results.add(key)
        if key not in state.prefetch_result_keys:
            state.non_prefetch_tool_calls_executed += 1
        succeeded, error = _tool_result_status(content)
        call = _tool_result_call(messages, message)
        if call is not None and _is_route_candidate_getter(call, state.tools_by_name):
            state.route_getter_call_counts[name] = (
                state.route_getter_call_counts.get(name, 0) + 1
            )
        if succeeded and name:
            state.successful_tool_names.add(name)
        if _tool_result_is_unavailable(content):
            signature = _tool_result_signature(messages, message)
            if signature is not None:
                state.unavailable_tool_signatures.add(signature)
            evidence = f"{name or '<unknown tool>'}: {content[:500]}"
            if evidence not in state.unavailability_evidence:
                state.unavailability_evidence.append(evidence)
                del state.unavailability_evidence[:-8]
        if succeeded and name.startswith(_READ_PREFIXES):
            state.read_ledger.add(name)
            if name.startswith("get_") and call is not None:
                signature = _tool_call_signature(call)
                history = state.successful_get_results.setdefault(signature, [])
                history.append(_canonical_tool_result(content))
                # Only the two most recent identical observations are needed
                # for the L1 third-read preflight.
                del history[:-2]
                state.successful_get_signatures_by_tool.setdefault(name, set()).add(
                    signature
                )
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                payload = None
            if payload is not None:
                _catalog_route_references(payload, state)
                if name == "get_current_navigation_state":
                    _ingest_navigation_waypoints(payload, state)
        elif succeeded and _is_mutating_tool_name(name):
            signature = _tool_call_signature(call) if call is not None else None
            if signature is not None and signature not in state.successful_mutation_signatures:
                state.successful_mutation_signatures.add(signature)
                assert call is not None
                state.successful_mutations.append(
                    {
                        "signature": signature,
                        "tool_name": call["tool_name"],
                        "arguments": dict(call.get("arguments") or {}),
                        "read_name": state.pairings.get(str(call["tool_name"])),
                        "subsystem_tokens": sorted(
                            _name_core_tokens(str(call["tool_name"]))
                        ),
                    }
                )
                _record_argument_disclosure_requirements(call, state)
        elif error is not None:
            faults.append(f"{name or '<unknown tool>'}: {error}")
    if not faults:
        return None
    return InternalFault(
        SIGNAL_TOOL_ERROR,
        "Tool call returned an error: " + "; ".join(faults),
    )


def _mutation_log_fault(
    action: dict[str, Any], state: _EpisodeState, messages: list[dict[str, Any]]
) -> InternalFault | None:
    """Reject duplicate successes and mutations with no user-mentioned subject."""

    if action.get("action") != "tool_calls":
        return None
    user_tokens = set().union(
        *(
            _tokens(str(message.get("content") or ""))
            for message in messages
            if message.get("role") == "user"
        )
    )
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        if not _is_mutating_tool_name(name):
            continue
        if _tool_call_signature(call) in state.successful_mutation_signatures:
            return InternalFault(
                SIGNAL_MUTATION_LOG_CHECK,
                "The episode mutation log shows this exact mutation already succeeded. "
                "Do not repeat it; complete only any still-unsatisfied request.",
            )
        subject = _name_core_tokens(name) - {"new", "activate", "deactivate"}
        if subject and not (subject & user_tokens):
            return InternalFault(
                SIGNAL_MUTATION_LOG_CHECK,
                "This mutation changes a state field the user did not mention. "
                "Use the smallest diff: omit unrelated state changes.",
            )
    return None


def _grounded_respond_fault(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    already_fired: bool,
) -> InternalFault | None:
    """Require numeric final-state claims to have a literal tool-result source."""

    if already_fired or action.get("action") != "respond":
        return None
    content = str(action.get("content") or "")
    if not content or "?" in content:
        return None
    tool_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    )
    claims = re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:\s*%|\s*°?C|\s*km)?", content)
    missing = [value for value in claims if value.strip() not in tool_text]
    if missing:
        return InternalFault(
            SIGNAL_GROUNDED_RESPOND,
            "The proposed final response quotes state values with no matching tool-result "
            f"source ({', '.join(missing[:3])}). Revise once: remove or qualify unsupported "
            "values, or read them with an available tool before responding.",
        )
    return None


def _drop_failed_prefetch_results(
    messages: list[dict[str, Any]],
    state: _EpisodeState,
    ctx_logger: Any,
) -> tuple[list[dict[str, Any]], int]:
    dropped_this_call = 0
    if state.prefetch_results_pending:
        trailing: list[dict[str, Any]] = []
        for message in reversed(messages):
            if message.get("role") != "tool":
                break
            trailing.append(message)
        if trailing:
            emitted = set(state.prefetch_tools_emitted)
            for message in reversed(trailing):
                name = str(message.get("name") or "")
                if name not in emitted:
                    continue
                key = _tool_result_key(message)
                state.prefetch_result_keys.add(key)
                _, error = _tool_result_status(str(message.get("content") or ""))
                if error is None:
                    continue
                state.dropped_prefetch_result_keys.add(key)
                state.prefetch_error_drops += 1
                dropped_this_call += 1
                signature = _tool_result_signature(messages, message)
                if signature is not None:
                    state.unavailable_tool_signatures.add(signature)
                ctx_logger.info(
                    "Adaptive-minimal dropped failed prefetch result",
                    tool_name=name,
                    dropped_count_episode=state.prefetch_error_drops,
                )
            state.prefetch_results_pending = False

    filtered = [
        message
        for message in messages
        if not (
            message.get("role") == "tool"
            and _tool_result_key(message) in state.dropped_prefetch_result_keys
        )
    ]
    return filtered, dropped_this_call


def _tool_result_key(message: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(message.get("tool_call_id") or ""),
        str(message.get("name") or ""),
        str(message.get("content") or ""),
    )


def _tool_result_status(content: str) -> tuple[bool, str | None]:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        lowered = content.casefold()
        if re.search(r"\b(error|failed|failure|invalid)\b", lowered):
            return False, content[:500]
        return True, None
    if not isinstance(payload, dict):
        return True, None
    status = payload.get("status")
    if isinstance(status, str):
        if status.casefold() == "success":
            return True, None
        return False, json.dumps(payload, ensure_ascii=False)[:500]
    if payload.get("success") is False or payload.get("error") is not None:
        return False, json.dumps(payload, ensure_ascii=False)[:500]
    return True, None


def _tool_result_is_unavailable(content: str) -> bool:
    """Classify generic unknown/error/absent results without tool-name rules."""

    _, error = _tool_result_status(content)
    if error is not None:
        return True
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return bool(
            re.search(
                r"\b(?:unknown|unavailable|not available|absent|not found)\b",
                content.casefold(),
            )
        )
    if not isinstance(payload, dict):
        return payload is None or payload == "" or payload == []
    if "result" not in payload:
        # A status-bearing envelope with no result is an absent result. A plain
        # dictionary may itself be a valid direct result, so leave it alone.
        return "status" in payload or "success" in payload
    result = payload.get("result")
    if result is None or result == "" or result == {} or result == []:
        return True
    return _contains_unavailable_marker(result)


def _contains_unavailable_marker(value: Any) -> bool:
    if isinstance(value, str):
        normalized = " ".join(value.casefold().replace("_", " ").split())
        return normalized in {
            "unknown",
            "unavailable",
            "not available",
            "absent",
            "not found",
        }
    if isinstance(value, dict):
        return any(_contains_unavailable_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unavailable_marker(item) for item in value)
    return False


def _tool_result_call(
    messages: list[dict[str, Any]], result_message: dict[str, Any]
) -> dict[str, Any] | None:
    call_id = str(result_message.get("tool_call_id") or "")
    result_name = str(result_message.get("name") or "")
    fallback: dict[str, Any] | None = None
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        for raw_call in reversed(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments") or {}
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    continue
            else:
                arguments = raw_arguments
            if not isinstance(arguments, dict):
                continue
            call = {"tool_name": name, "arguments": arguments}
            if call_id and str(raw_call.get("id") or "") == call_id:
                return call
            if fallback is None and result_name and name == result_name:
                fallback = call
        if fallback is not None and not call_id:
            break
    return fallback


def _tool_result_signature(
    messages: list[dict[str, Any]], result_message: dict[str, Any]
) -> str | None:
    call = _tool_result_call(messages, result_message)
    return _tool_call_signature(call) if call is not None else None


def validate_next_action(
    action: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> InternalFault | None:
    if action.get("action") == "respond" and not str(action.get("content") or "").strip():
        return InternalFault(
            SIGNAL_MALFORMED_OR_EMPTY,
            "Completion was malformed or empty: respond content was empty.",
        )
    if action.get("action") != "tool_calls":
        return None
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        tool = tools_by_name.get(name)
        if tool is None:
            return InternalFault(
                SIGNAL_TOOL_NOT_IN_CATALOG,
                f"Proposed tool {name!r} is not in the available tool catalog.",
            )
        arguments = call.get("arguments") or {}
        schema = tool.get("function", {}).get("parameters") or {"type": "object"}
        error = _validate_schema(arguments, schema, path="$arguments")
        if error is not None:
            return InternalFault(
                SIGNAL_SCHEMA_VALIDATION,
                f"Arguments for tool {name!r} fail schema validation: {error}",
            )
    return None


def _schema_preflight_fault(
    action: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> InternalFault | None:
    """Return a deterministic catalog-schema fault before tool execution.

    Some lookup catalogs express a choice of selector fields only in the tool
    description, leaving every individual selector optional in JSON Schema.
    Empty lookup calls are still structurally incomplete, so treat a `get_*_by_*`
    tool with declared id/name selectors and no supplied selector as a missing-
    parameter fault.  The rule is catalog-derived and independent of task data.
    """

    if action.get("action") != "tool_calls":
        return None
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        tool = tools_by_name.get(name)
        if tool is None:
            continue
        arguments = call.get("arguments")
        schema = tool.get("function", {}).get("parameters") or {"type": "object"}
        error = _validate_schema(arguments, schema, path="$arguments")
        if error is None and isinstance(arguments, dict):
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            for field in required:
                value = arguments.get(field)
                if (
                    isinstance(value, str)
                    and not value.strip()
                    and re.search(r"(?:^|_)(?:id|name)(?:_|$)", str(field))
                ):
                    error = f"$arguments.{field} must not be empty"
                    break
            selector_fields = {
                str(field)
                for field in properties
                if re.search(r"(?:^|_)(?:id|name)(?:_|$)", str(field))
            }
            if (
                error is None
                and re.match(r"^get_.+_by_.+", name)
                and selector_fields
                and not any(field in arguments for field in selector_fields)
            ):
                fields = ", ".join(sorted(selector_fields))
                error = (
                    "$arguments requires at least one catalog selector "
                    f"from: {fields}"
                )
        if error is not None:
            return InternalFault(
                SIGNAL_SCHEMA_VALIDATION,
                f"Arguments for tool {name!r} fail schema preflight: {error}. "
                "Correct the arguments and re-decide without executing this call.",
            )
    return None


def _split_repeated_get_calls(
    action: dict[str, Any], state: _EpisodeState
) -> tuple[dict[str, Any] | None, int]:
    """Block only third-or-later identical successful reads, never mutations."""

    if action.get("action") != "tool_calls":
        return action, 0
    kept: list[dict[str, Any]] = []
    blocked = 0
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        history = state.successful_get_results.get(_tool_call_signature(call), [])
        if name.startswith("get_") and len(history) >= 2 and history[-2] == history[-1]:
            blocked += 1
        else:
            kept.append(call)
    if not blocked:
        return action, 0
    if not kept:
        return None, blocked
    return {"action": "tool_calls", "tool_calls": kept}, blocked


ROUTE_RESOLVER_DISTINCT_READ_BUDGET = 4
ROUTE_RESOLVER_REDIRECT_CAP = 2


def _is_route_candidate_getter(
    call: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> bool:
    name = str(call.get("tool_name") or "")
    if not name.casefold().startswith("get_"):
        return False
    tool = tools_by_name.get(name) or {}
    properties = (
        tool.get("function", {}).get("parameters", {}).get("properties", {})
    )
    fields = {str(field).casefold() for field in properties}
    return bool(
        {"start_id", "destination_id"} <= fields
        and "route" in name.casefold()
    )


def _route_candidate_context(state: _EpisodeState) -> str | None:
    rows: list[dict[str, Any]] = []
    for (start_id, destination_id), candidates in sorted(
        state.route_candidates_by_pair.items()
    ):
        for candidate in candidates:
            row = {
                "start_id": start_id,
                "destination_id": destination_id,
                "route_id": candidate.get("route_id"),
            }
            for field in (
                "alias",
                "name_via",
                "distance",
                "duration",
                "includes_toll",
                "road_types",
            ):
                if field in candidate:
                    row[field] = candidate[field]
            rows.append(row)
            if len(rows) >= 24:
                break
        if len(rows) >= 24:
            break
    if not rows:
        return None
    rendered = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return rendered[:6000]


def _apply_route_resolver(
    action: dict[str, Any], state: _EpisodeState
) -> tuple[dict[str, Any] | None, int, str | None]:
    """Bound varying route reads and replay only environment-grounded candidates.

    L1 handles exact repeats.  This complementary guard fires only after four
    distinct successful signatures for the same route getter and only when a
    candidate catalog exists.  Two redirects per episode bound the additional
    decision work; afterward it fails open.
    """

    if (
        action.get("action") != "tool_calls"
        or state.route_resolver_fires >= ROUTE_RESOLVER_REDIRECT_CAP
    ):
        return action, 0, None
    context = _route_candidate_context(state)
    if context is None:
        return action, 0, None
    kept: list[dict[str, Any]] = []
    blocked = 0
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        distinct = state.successful_get_signatures_by_tool.get(name, set())
        if (
            _is_route_candidate_getter(call, state.tools_by_name)
            and len(distinct) >= ROUTE_RESOLVER_DISTINCT_READ_BUDGET
        ):
            blocked += 1
        else:
            kept.append(call)
    if not blocked:
        return action, 0, None
    note = (
        "The route getter has already succeeded with at least four distinct "
        "argument signatures. Do not read it again. Use these route candidates "
        "already grounded in tool results and decide now: "
        + context
    )
    if kept:
        return {"action": "tool_calls", "tool_calls": kept}, blocked, note
    return None, blocked, note


def _apply_route_budget(
    action: dict[str, Any], state: _EpisodeState, *, limit: int
) -> tuple[dict[str, Any] | None, InternalFault | None, int, bool]:
    """Stop every route-getter call after a per-tool execution budget.

    Unlike the retired resolver, the count includes successful and failed calls
    and ignores argument variation. The first block schedules one catalog-
    grounded re-decision. A repeated refusal to consume the catalog terminates
    with a grounded limitation instead of permitting another loop iteration.
    """

    if action.get("action") != "tool_calls":
        return action, None, 0, False
    blocked_names = [
        str(call.get("tool_name") or "")
        for call in action.get("tool_calls") or []
        if _is_route_candidate_getter(call, state.tools_by_name)
        and state.route_getter_call_counts.get(
            str(call.get("tool_name") or ""), 0
        ) >= max(1, limit)
    ]
    if not blocked_names:
        return action, None, 0, False
    unique = sorted(set(blocked_names))
    context = _route_candidate_context(state)
    fresh = [name for name in unique if name not in state.route_budget_redirected_tools]
    if fresh:
        state.route_budget_redirected_tools.update(fresh)
        catalog_note = context or "[] (no usable route candidate was returned)"
        return (
            None,
            InternalFault(
                "route_budget",
                "The route getter budget is exhausted. Do not call the route "
                "getter again. Consume the cached environment-grounded "
                "(start_id, destination_id, route_id) catalog and take the "
                "requested next action now. Cached catalog: " + catalog_note,
            ),
            len(blocked_names),
            False,
        )
    return (
        {
            "action": "respond",
            "content": (
                "I could not complete the requested route change from the "
                "route candidates returned by the navigation system, and I "
                "did not issue another speculative route lookup or mutation."
            ),
        },
        None,
        len(blocked_names),
        True,
    )


_NAV_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "delete_waypoint",
        re.compile(
            r"\b(?:remove|delete|drop|cancel)\b.{0,50}"
            r"\b(?:waypoint|intermediate\s+stop|stop)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "replace_waypoint",
        re.compile(
            r"\b(?:replace|change|swap)\b.{0,50}"
            r"\b(?:waypoint|intermediate\s+stop|stop)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delete_final_destination",
        re.compile(
            r"\b(?:remove|delete|drop|cancel)\b.{0,60}"
            r"\b(?:final\s+destination|destination|endpoint|from\s+(?:the|my|your)\s+route)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "replace_final_destination",
        re.compile(
            r"\b(?:replace|change|switch)\b.{0,60}"
            r"\b(?:final\s+destination|destination|endpoint)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "start_new_navigation",
        re.compile(
            r"\b(?:navigate|navigation|route)\b.{0,60}"
            r"\b(?:to|start|set\s+up|begin)\b|"
            r"\b(?:start|set\s+up|begin)\b.{0,60}\bnavigation\b",
            re.IGNORECASE,
        ),
    ),
)


def _requested_navigation_intent(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        for intent, pattern in _NAV_INTENT_PATTERNS:
            if pattern.search(text):
                return intent
    return None


def _navigation_call_intent(name: str) -> str | None:
    lowered = name.casefold()
    if lowered == "navigation_delete_destination":
        return "delete_final_destination"
    if lowered == "navigation_replace_final_destination":
        return "replace_final_destination"
    if lowered == "navigation_delete_waypoint":
        return "delete_waypoint"
    if lowered == "navigation_replace_one_waypoint":
        return "replace_waypoint"
    if lowered == "set_new_navigation":
        return "start_new_navigation"
    if lowered == "delete_current_navigation":
        return "clear_all_navigation"
    return None


def _nav_intent_preflight_fault(
    action: dict[str, Any], *, messages: list[dict[str, Any]], state: _EpisodeState
) -> InternalFault | None:
    if action.get("action") != "tool_calls":
        return None
    requested = _requested_navigation_intent(messages)
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        drafted = _navigation_call_intent(name)
        if drafted is None:
            continue
        arguments = call.get("arguments") or {}
        if requested is not None and drafted != requested:
            return InternalFault(
                "nav_intent_preflight",
                f"The user requested navigation intent {requested!r}, but "
                f"the drafted call {name!r} performs {drafted!r}. Do not "
                "substitute clear-all, waypoint, destination, replace, or "
                "start operations for one another; re-decide once.",
            )
        if name.casefold() == "navigation_delete_destination":
            target = arguments.get("destination_id_to_delete")
            final = state.navigation_waypoints[-1] if state.navigation_waypoints else None
            if final is not None and target != final:
                return InternalFault(
                    "nav_intent_preflight",
                    f"The drafted destination deletion targets {target!r}, "
                    f"but the latest navigation state says the final "
                    f"destination is {final!r}. A non-final waypoint must not "
                    "be deleted with the final-destination tool.",
                )
        if name.casefold() == "set_new_navigation":
            tool = state.tools_by_name.get(name) or {}
            properties = (
                tool.get("function", {})
                .get("parameters", {})
                .get("properties", {})
            )
            route_fields = [field for field in properties if "route_id" in str(field)]
            if not route_fields or not any(arguments.get(field) for field in route_fields):
                return InternalFault(
                    "nav_intent_preflight",
                    "Starting navigation requires an available, non-empty "
                    "route-id argument grounded in the route catalog. That "
                    "required capability is absent from this drafted call.",
                )
            selected = {
                str(route_id)
                for field in route_fields
                for route_id in (
                    arguments.get(field)
                    if isinstance(arguments.get(field), list)
                    else [arguments.get(field)]
                )
                if isinstance(route_id, str)
            }
            user_text = _all_user_text(messages).casefold()
            wants_no_toll = bool(
                re.search(
                    r"\b(?:no\s+tolls?|without\s+tolls?|toll[- ]free|"
                    r"does(?:n['’]t| not)\s+(?:use|include)\s+tolls?)\b",
                    user_text,
                )
            )
            if wants_no_toll:
                selected_rows = [
                    candidate
                    for candidates in state.route_candidates_by_pair.values()
                    for candidate in candidates
                    if str(candidate.get("route_id") or "") in selected
                ]
                has_toll_free_alternative = any(
                    candidate.get("includes_toll") is False
                    for candidates in state.route_candidates_by_pair.values()
                    for candidate in candidates
                )
                if has_toll_free_alternative and any(
                    candidate.get("includes_toll") is True
                    for candidate in selected_rows
                ):
                    return InternalFault(
                        "nav_intent_preflight",
                        "The drafted new-navigation route includes tolls even "
                        "though the user selected a toll-free route and the "
                        "catalog contains a toll-free candidate. Re-decide "
                        "from the grounded route catalog before mutating.",
                    )
            selected_rows = [
                candidate
                for candidates in state.route_candidates_by_pair.values()
                for candidate in candidates
                if str(candidate.get("route_id") or "") in selected
            ]
            has_toll_free_alternative = any(
                candidate.get("includes_toll") is False
                for candidates in state.route_candidates_by_pair.values()
                for candidate in candidates
            )
            accepted_tolls = bool(
                re.search(
                    r"\b(?:tolls?\s+(?:are|is)\s+(?:okay|ok|fine|acceptable)|"
                    r"accept\s+tolls?|pay\s+(?:the\s+)?tolls?|tolls?\s+are\s+fine)\b",
                    user_text,
                )
            )
            if (
                not wants_no_toll
                and not accepted_tolls
                and has_toll_free_alternative
                and any(
                    candidate.get("includes_toll") is True
                    for candidate in selected_rows
                )
            ):
                return InternalFault(
                    "nav_intent_preflight",
                    "The drafted route includes tolls and the catalog contains "
                    "a toll-free alternative, but the user has not accepted or "
                    "rejected tolls yet. Present that grounded distinction and "
                    "re-decide before the irreversible navigation mutation.",
                )
    return None


def _explicit_confirmation_for_weather(messages: list[dict[str, Any]]) -> bool:
    latest_user = _latest_user_text(messages).casefold().strip()
    if latest_user not in {"yes", "yes.", "confirm", "confirmed", "go ahead", "proceed"}:
        return False
    return any(
        message.get("role") == "assistant"
        and "fog" in str(message.get("content") or "").casefold()
        and "weather" in str(message.get("content") or "").casefold()
        for message in messages[-6:]
    )


def _step_coverage_missing_tools(
    action: dict[str, Any], *, messages: list[dict[str, Any]], state: _EpisodeState
) -> list[str]:
    """Return explicit/catalog-derived requested steps absent at terminal time."""

    if action.get("action") != "respond" or "?" in str(action.get("content") or ""):
        return []
    user_text = _all_user_text(messages).casefold()
    user_tokens = _tokens(user_text)
    missing: set[str] = set()
    available = state.tools_by_name
    done = state.successful_tool_names

    computed_rules = {
        "calculate_charging_time_by_soc": bool(
            re.search(r"\b(?:charging|charge)\s+time\b|\btime\b.{0,30}\bcharg", user_text)
        ),
        "get_distance_by_soc": bool(
            re.search(
                r"\b(?:driving\s+distance|how\s+far|range|charging\s+stops?|"
                r"distance\s+from|state\s+of\s+charge|soc)\b",
                user_text,
            )
        ),
        "get_charging_specs_and_status": bool(
            re.search(r"\b(?:battery|charging)\b", user_text)
            and re.search(r"\b(?:range|make\s+it|charging\s+station|charger|power)\b", user_text)
        ),
    }
    for name, requested in computed_rules.items():
        if requested and name in available and name not in done:
            missing.add(name)

    for name, tool in available.items():
        if not _is_mutating_tool_name(name) or name in done:
            continue
        core = _name_core_tokens(name)
        if len(core & user_tokens) >= 2:
            missing.add(name)
        elif name == "send_email" and {"send", "email"} <= user_tokens:
            missing.add(name)
        elif name == "set_new_navigation" and bool(
            {"navigate", "navigation"} & user_tokens
        ):
            missing.add(name)

    if (
        "set_fog_lights" in done
        and "get_weather" in available
        and "get_weather" not in done
        and not _explicit_confirmation_for_weather(messages)
    ):
        missing.add("get_weather")
    return sorted(missing)


_TEXTCALL_DIRECT = re.compile(
    r"\bi\s+will\s+call\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+with\s+parameters\s*",
    re.IGNORECASE,
)
_TEXTCALL_NEAR_TERMINAL = re.compile(
    r"\b(?:i(?:['’]m|\s+am)\s+(?:ready|going)\s+to\s+"
    r"(?:send|call|set|start)|i\s+will\s+(?:call|send)\s+"
    r"[a-zA-Z_][a-zA-Z0-9_]*|(?:ready|about)\s+to\s+send)\b",
    re.IGNORECASE,
)


def _textcall_guard_candidate(
    action: dict[str, Any], state: _EpisodeState
) -> tuple[dict[str, Any] | None, bool]:
    """Detect a mutation serialized as prose; execution is never synthesized."""

    if action.get("action") != "respond":
        return None, False
    content = str(action.get("content") or "")
    if not _TEXTCALL_NEAR_TERMINAL.search(content):
        return None, False
    return None, True


_INABILITY_DRAFT = re.compile(
    r"\b(?:cannot|can't|unable|do not have|don't have|no available tool|"
    r"not available|is unavailable|could not)\b",
    re.IGNORECASE,
)


def _limitation_classifier_trigger(
    action: dict[str, Any], *, messages: list[dict[str, Any]], state: _EpisodeState
) -> str | None:
    """Return generic catalog/result evidence for one careful classifier call."""

    draft = str(action.get("content") or "")
    user_text = _all_user_text(messages).casefold()
    validation_fault = validate_next_action(action, state.tools_by_name)
    if validation_fault is None:
        validation_fault = _schema_preflight_fault(action, state.tools_by_name)
    calls = action.get("tool_calls") or [] if action.get("action") == "tool_calls" else []
    retries_unavailable = any(
        _tool_call_signature(call) in state.unavailable_tool_signatures
        for call in calls
    )
    substitute = any(
        str(call.get("tool_name") or "").casefold() in {"calculate_math"}
        for call in calls
    )
    computed_catalog_gap = bool(
        re.search(
            r"\b(?:range|driving\s+distance|how\s+far|how\s+many\s+"
            r"(?:miles?|kilometers?|kilometres?))\b",
            user_text,
        )
        and "get_distance_by_soc" not in state.tools_by_name
        and action.get("action") == "respond"
        and bool(re.search(r"\d", draft))
    )
    repeats_inability = bool(_INABILITY_DRAFT.search(draft))
    if not (
        validation_fault is not None
        or retries_unavailable
        or substitute
        or computed_catalog_gap
        or (state.unavailability_evidence and repeats_inability)
    ):
        return None
    evidence = list(state.unavailability_evidence[-4:])
    if validation_fault is not None:
        evidence.append(validation_fault.text)
    if computed_catalog_gap:
        evidence.append(
            "The user requested a range/distance computation, but the catalog "
            "contains no get_distance_by_soc capability and the draft states a "
            "numeric result without that authoritative call."
        )
    if substitute:
        evidence.append("The draft retries a substitute calculation call.")
    if repeats_inability:
        evidence.append("The draft repeats an inability statement.")
    return " ".join(evidence)[-4000:]


def _walk_string_arguments(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            rows.extend(_walk_string_arguments(item, (*path, str(key))))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_walk_string_arguments(item, (*path, str(index))))
        return rows
    if isinstance(value, str):
        return [(".".join(path), value)]
    return []


def _argument_policy_lint_fault(action: dict[str, Any]) -> InternalFault | None:
    if action.get("action") != "tool_calls":
        return None
    violations: list[str] = []
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        if not _is_mutating_tool_name(name):
            continue
        for path, value in _walk_string_arguments(call.get("arguments") or {}):
            found = _AM_PM_TIME.search(value)
            if found:
                violations.append(f"{name}.{path} contains {found.group(0)!r}")
    if not violations:
        return None
    return InternalFault(
        "argument_policy_lint",
        "Mutation arguments must use 24-hour time. Revise the same intended "
        "call without changing recipients or semantics: " + "; ".join(violations[:4]),
    )


def _record_argument_disclosure_requirements(
    call: dict[str, Any], state: _EpisodeState
) -> None:
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict):
        return
    route_ids: list[str] = []
    for field, value in arguments.items():
        if "route_id" not in str(field).casefold():
            continue
        if isinstance(value, str):
            route_ids.append(value)
        elif isinstance(value, list):
            route_ids.extend(str(item) for item in value if isinstance(item, str))
    candidates = {
        str(item.get("route_id") or ""): item
        for items in state.route_candidates_by_pair.values()
        for item in items
    }
    for route_id in route_ids:
        candidate = candidates.get(route_id) or {}
        aliases = {str(item).casefold() for item in candidate.get("alias") or []}
        if "fastest" in aliases:
            state.arg_lint_pending_disclosures.add("fastest_route")
        if candidate.get("includes_toll") is True:
            state.arg_lint_pending_disclosures.add("toll_route")


def _argument_disclosure_fault(
    action: dict[str, Any], state: _EpisodeState
) -> InternalFault | None:
    if action.get("action") != "respond" or not state.arg_lint_pending_disclosures:
        return None
    content = str(action.get("content") or "").casefold()
    missing: list[str] = []
    if "fastest_route" in state.arg_lint_pending_disclosures and "fastest" not in content:
        missing.append("state that the selected route is the fastest")
    if "toll_route" in state.arg_lint_pending_disclosures and "toll" not in content:
        missing.append("disclose that the selected route includes toll roads")
    if not missing:
        return None
    return InternalFault(
        "argument_policy_lint",
        "The executed mutation selected route arguments with required user "
        "disclosures. Preserve the grounded action summary and " + " and ".join(missing) + ".",
    )


_REFERENT_PREFERENCE = re.compile(
    r"\b(?:preference|preferred|default|usual|saved|stored|secretary)\b",
    re.IGNORECASE,
)
_QUESTION_RESOLUTION = re.compile(
    r"\b(?:which|who|what|whether|do\s+you\s+mean|"
    r"(?:can|could)\s+you\s+tell\s+me|low[- ]?beam|high[- ]?beam)\b",
    re.IGNORECASE,
)
_MODEL_CLARIFICATION_QUESTION = re.compile(
    r"\b(?:do\s+you\s+(?:want|mean|prefer)|would\s+you\s+(?:like|prefer)|"
    r"should\s+i|which|who|what\s+(?:exact|specific)|"
    r"(?:can|could|would)\s+you\s+(?:clarify|confirm|tell|provide)|"
    r"please\s+(?:clarify|confirm))\b",
    re.IGNORECASE,
)
_GROUNDING_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "do", "exact", "for",
    "i", "is", "it", "me", "my", "of", "or", "please", "should",
    "specific", "tell", "the", "to", "use", "want", "what", "which",
    "who", "would", "you", "your",
}
_ASK_SLOT_STOPWORDS = _GROUNDING_QUESTION_STOPWORDS | {
    "about", "ask", "clarify", "confirm", "did", "does", "give", "help",
    "know", "like", "mean", "need", "prefer", "provide", "say", "select",
    "specify", "tell", "that", "them", "these", "this", "those", "using",
    "would",
    # Values and polarity words distinguish candidate answers, not the slot
    # being requested.
    "active", "closed", "current", "exact", "high", "low", "off", "on",
    "open", "preferred", "specific",
}
_ASK_SLOT_CANONICAL = {
    "assistant": "contact",
    "contact": "contact",
    "person": "contact",
    "recipient": "contact",
    "secretary": "contact",
    "email": "email",
    "mail": "email",
    "telephone": "phone",
    "phone": "phone",
    "amount": "value",
    "degree": "value",
    "level": "value",
    "percent": "value",
    "percentage": "value",
    "setting": "value",
    "value": "value",
    "destination": "location",
    "location": "location",
    "place": "location",
    "beam": "light",
    "headlight": "light",
    "light": "light",
    "path": "route",
    "route": "route",
    "colour": "color",
    "color": "color",
    "hour": "time",
    "time": "time",
    "position": "seat",
    "seat": "seat",
    "temp": "temperature",
    "temperature": "temperature",
}
_COMMON_CAPITALIZED = {
    "alright", "can", "could", "dear", "good", "great", "hello", "hey",
    "hi", "i", "okay", "please", "sure", "thanks", "thank", "the", "yes",
    "use",
}


def _exact_person_names(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        for first, last in re.findall(
            r"(?=\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b)", text
        ):
            if first.casefold() in _COMMON_CAPITALIZED:
                continue
            pair = (first, last)
            if pair not in names:
                names.append(pair)
    return names


def _contact_resolution_read(
    question: str,
    messages: list[dict[str, Any]],
    state: _EpisodeState,
) -> dict[str, Any] | None:
    tool_name = next(
        (
            name
            for name in state.tools_by_name
            if name.casefold() == "get_contact_id_by_contact_name"
        ),
        None,
    )
    if tool_name is None or _tool_was_called(messages, tool_name):
        return None
    lowered = question.casefold()
    candidates = [
        pair
        for pair in _exact_person_names(messages)
        if pair[0].casefold() in lowered or pair[1].casefold() in lowered
    ]
    if not candidates:
        return None
    first, last = candidates[-1]
    properties = (
        state.tools_by_name[tool_name]
        .get("function", {})
        .get("parameters", {})
        .get("properties", {})
    )
    arguments: dict[str, Any] = {}
    if "contact_first_name" in properties:
        arguments["contact_first_name"] = first
    if "contact_last_name" in properties:
        arguments["contact_last_name"] = last
    action = {
        "action": "tool_calls",
        "tool_calls": [{"tool_name": tool_name, "arguments": arguments}],
    }
    return action if _schema_preflight_fault(action, state.tools_by_name) is None else None


def _read_resolve_action(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    state: _EpisodeState,
) -> dict[str, Any] | None:
    if (
        action.get("action") != "respond"
        or state.read_resolve_redirects >= 1
    ):
        return None
    question = str(action.get("content") or "")
    if "?" not in question or not _QUESTION_RESOLUTION.search(question):
        return None

    contact = _contact_resolution_read(question, messages, state)
    if contact is not None:
        return contact

    all_user = _all_user_text(messages)
    preference_tool = state.tools_by_name.get("get_user_preferences")
    if (
        preference_tool is not None
        and not _preference_tool_was_called(messages)
        and _REFERENT_PREFERENCE.search(question + "\n" + all_user)
    ):
        arguments = _preference_read_arguments(preference_tool)
        if arguments is not None:
            return {
                "action": "tool_calls",
                "tool_calls": [
                    {"tool_name": "get_user_preferences", "arguments": arguments}
                ],
            }

    question_tokens = _tokens(question)
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for name, tool in state.tools_by_name.items():
        if not name.casefold().startswith("get_") or _tool_was_called(messages, name):
            continue
        arguments = _safe_read_arguments(tool)
        if arguments is None:
            continue
        score = len(question_tokens & _schema_tokens(tool))
        if score:
            ranked.append((score, name, arguments))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return None
    _, name, arguments = ranked[0]
    return {
        "action": "tool_calls",
        "tool_calls": [{"tool_name": name, "arguments": arguments}],
    }


def _is_clarification_question(action: dict[str, Any]) -> bool:
    """Conservatively recognize a model-authored request for user resolution."""

    if action.get("action") != "respond":
        return False
    content = str(action.get("content") or "")
    return "?" in content and bool(_MODEL_CLARIFICATION_QUESTION.search(content))


def _ask_slot_signature(action: dict[str, Any]) -> str | None:
    """Return a normalized set of parameter/referent nouns for an ask."""

    if not _is_clarification_question(action):
        return None
    content = str(action.get("content") or "").casefold()
    content = re.sub(r"\be[\s-]?mail\s+address(?:es)?\b", "email", content)
    content = re.sub(r"\bphone\s+number(?:s)?\b", "phone", content)
    slots = {
        _ASK_SLOT_CANONICAL.get(token, token)
        for token in _tokens(content)
        if token not in _ASK_SLOT_STOPWORDS
    }
    return json.dumps(sorted(slots), ensure_ascii=False, separators=(",", ":"))


def _select_ask_content_consensus(
    candidates: list[dict[str, Any] | None],
    *,
    tools_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool, list[str | None]]:
    """Select a 2-of-3 ask-slot mode, with a hard ask-only fail-open.

    ``tools_by_name`` is accepted for an explicit catalog-context interface;
    slot extraction itself intentionally uses only the draft's internal nouns.
    This keeps the vote independent of task IDs and evaluator signals.
    """

    del tools_by_name
    if len(candidates) != 3 or candidates[0] is None:
        raise ValueError(
            "ask-content consensus requires one original and two samples"
        )
    original = candidates[0]
    assert original is not None and _is_clarification_question(original)
    signatures = [
        _ask_slot_signature(candidate) if candidate is not None else None
        for candidate in candidates
    ]
    # HARD LAW: both additional completions must be asks. An action, response,
    # malformed sample, or missing slot signature makes the vote ineligible.
    if signatures[1] is None or signatures[2] is None:
        return original, False, signatures
    counts = Counter(signature for signature in signatures if signature is not None)
    majority_signature = next(
        (signature for signature in signatures if counts[signature] >= 2),
        None,
    )
    if majority_signature is None:
        return original, False, signatures
    selected_index = signatures.index(majority_signature)
    selected = candidates[selected_index]
    assert selected is not None and _is_clarification_question(selected)
    return selected, True, signatures


def _grounded_ask_read_action(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    state: _EpisodeState,
) -> tuple[dict[str, Any], str] | None:
    """Derive at most two free reads before a model-authored clarification.

    This is deliberately non-suppressive.  It only replaces the draft for one
    tool turn so the next model decision can see enumerated state; the planner
    then accepts that re-draft whether it asks, acts, or responds.  A call-set
    signature is the referent key, bounding repeated wording about the same
    catalogued entity/state without task IDs or evaluator signals.
    """

    if not _is_clarification_question(action):
        return None
    question = str(action.get("content") or "")
    calls: list[dict[str, Any]] = []

    contact = _contact_resolution_read(question, messages, state)
    if contact is not None:
        calls.extend(contact.get("tool_calls") or [])

    all_user = _all_user_text(messages)
    preference_tool = state.tools_by_name.get("get_user_preferences")
    if (
        len(calls) < 2
        and preference_tool is not None
        and not _preference_tool_was_called(messages)
        and _REFERENT_PREFERENCE.search(question + "\n" + all_user)
        and not any(
            call.get("tool_name") == "get_user_preferences" for call in calls
        )
    ):
        arguments = _preference_read_arguments(preference_tool)
        if arguments is not None:
            candidate = {
                "tool_name": "get_user_preferences",
                "arguments": arguments,
            }
            candidate_action = {"action": "tool_calls", "tool_calls": [candidate]}
            if _schema_preflight_fault(candidate_action, state.tools_by_name) is None:
                calls.append(candidate)

    question_tokens = _tokens(question) - _GROUNDING_QUESTION_STOPWORDS
    selected_names = {str(call.get("tool_name") or "") for call in calls}
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for name, tool in state.tools_by_name.items():
        if (
            not name.casefold().startswith("get_")
            or name in selected_names
            or _tool_was_called(messages, name)
        ):
            continue
        arguments = _safe_read_arguments(tool)
        if arguments is None:
            continue
        if not arguments and not _semantic_prefetch_call_valid(tool, arguments):
            # Optional-looking lookup schemas can still reject an empty call at
            # runtime.  Grounding reads must be semantically valid before they
            # are emitted because a later correction cannot erase tool errors.
            continue
        name_overlap = question_tokens & (
            _name_core_tokens(name) | _parameter_name_tokens(tool)
        )
        schema_overlap = question_tokens & _schema_tokens(tool)
        # A single description word is too weak (for example "current").
        # Name/parameter overlap or two independent schema words is required.
        if not name_overlap and len(schema_overlap) < 2:
            continue
        score = 5 * len(name_overlap) + 2 * len(schema_overlap)
        ranked.append((score, name, arguments))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    for _, name, arguments in ranked:
        if len(calls) >= 2:
            break
        calls.append({"tool_name": name, "arguments": arguments})

    if not calls:
        return None
    calls = calls[:2]
    referent_key = json.dumps(
        [
            {
                "tool_name": str(call.get("tool_name") or ""),
                "arguments": call.get("arguments") or {},
            }
            for call in calls
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if referent_key in state.grounded_ask_seen_referents:
        return None
    return {"action": "tool_calls", "tool_calls": calls}, referent_key


def _route_reference_preflight_fault(
    action: dict[str, Any], state: _EpisodeState
) -> InternalFault | None:
    """Validate navigation route IDs against endpoint provenance already read."""

    if action.get("action") != "tool_calls":
        return None
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        if "navigation" not in name.casefold():
            continue
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        route_fields = [
            (str(field), str(value))
            for field, value in arguments.items()
            if "route_id" in str(field).casefold() and isinstance(value, str)
        ]
        for field, route_id in route_fields:
            endpoints = state.route_reference_catalog.get(route_id)
            if endpoints is None:
                continue
            start_id, destination_id = endpoints
            field_key = field.casefold()
            expected_start: str | None = None
            expected_label = "waypoint"
            waypoint_fields = [
                (str(key), str(value))
                for key, value in arguments.items()
                if "waypoint_id" in str(key).casefold()
                and "delete" not in str(key).casefold()
                and isinstance(value, str)
            ]
            if "waypoint" in field_key and waypoint_fields:
                expected_label, expected_start = waypoint_fields[0]
            elif (
                "replace_final_destination" in name.casefold()
                and len(state.navigation_waypoints) >= 2
            ):
                expected_start = state.navigation_waypoints[-2]
                expected_label = "active penultimate waypoint"
            if expected_start is not None and start_id != expected_start:
                return InternalFault(
                    "route_reference_preflight",
                    f"Route {route_id} starts at {start_id}, but the "
                    f"{expected_label} is {expected_start}. Pick a route whose "
                    "start matches that waypoint and re-decide without executing "
                    "this call.",
                )
            destination_fields = [
                (str(key), str(value))
                for key, value in arguments.items()
                if "destination_id" in str(key).casefold()
                and isinstance(value, str)
            ]
            if "destination" in field_key and destination_fields:
                destination_label, expected_destination = destination_fields[0]
                if destination_id != expected_destination:
                    return InternalFault(
                        "route_reference_preflight",
                        f"Route {route_id} ends at {destination_id}, but the "
                        f"{destination_label} is {expected_destination}. Pick a "
                        "route whose destination matches and re-decide without "
                        "executing this call.",
                    )
    return None


class _ConsensusGuardLogger:
    """Discard cloned guard logs; the consensus summary records the outcome."""

    def info(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _guard_mutation_consensus_candidate(
    action: dict[str, Any],
    *,
    state: _EpisodeState,
    messages: list[dict[str, Any]],
    config: AdaptiveMinimalConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run a sampled draft through isolated deterministic execution guards.

    Guard paths that normally request another model decision reject this sample
    instead: M1 is allowed exactly two extra completions and never starts a
    correction or re-sampling loop.  Pure rewrites/injections run on a copied
    state and are eligible for exact-signature voting.
    """

    guarded = deepcopy(action)
    guard_state = deepcopy(state)
    guard_tally = _CallTally()
    guard_logger = _ConsensusGuardLogger()
    validation_tools = guard_state.tools_by_name
    if config.csp_afford and guard_state.csp_last_presented_tools_by_name:
        validation_tools = guard_state.csp_last_presented_tools_by_name
    if config.phase_gate and guard_state.phase_gate_last_presented_tools_by_name:
        validation_tools = guard_state.phase_gate_last_presented_tools_by_name

    if config.schema_preflight:
        fault = _schema_preflight_fault(guarded, validation_tools)
        if fault is not None:
            return None, fault.text
    fault = validate_next_action(guarded, validation_tools)
    if fault is not None:
        return None, fault.text
    if config.route_reference_preflight:
        fault = _route_reference_preflight_fault(guarded, guard_state)
        if fault is not None:
            return None, fault.text
    if config.nav_intent_preflight:
        fault = _nav_intent_preflight_fault(
            guarded, messages=messages, state=guard_state
        )
        if fault is not None:
            return None, fault.text
    if config.repeated_read_breaker:
        guarded, blocked = _split_repeated_get_calls(guarded, guard_state)
        if guarded is None:
            return None, f"repeated-read guard blocked {blocked} calls"
    if config.placeholder_guard:
        guarded, placeholder_index = _split_calls_before_placeholder(guarded)
        if guarded is None:
            return None, f"placeholder guard rejected call {placeholder_index}"
    if config.value_provenance:
        fault = _clarification_answer_binding_fault(
            guarded, messages=messages, tools_by_name=guard_state.tools_by_name
        )
        if fault is not None:
            return None, fault.text
        if _set_new_navigation_while_active(guarded, messages):
            return None, "active-navigation provenance guard requested a redirect"
        if (
            _respond_claims_performed_action(guarded)
            and not guard_state.successful_mutations
        ):
            return None, "claim-provenance guard requested a revise"
    if config.time_format_revise and _respond_has_am_pm_time(guarded):
        return None, "24-hour policy guard requested a revise"
    if config.policy_lint:
        violations = _policy_lint_violations(
            guarded, messages=messages, state=guard_state
        )
        if violations:
            return None, "policy lint requested a revise: " + ", ".join(
                rule for rule, _ in violations
            )
    if config.arg_lint:
        fault = _argument_policy_lint_fault(guarded)
        if fault is not None:
            return None, fault.text
    if config.turn_guard:
        fault = _turn_guard_fault(
            guarded,
            non_prefetch_tool_calls_executed=(
                guard_state.non_prefetch_tool_calls_executed
            ),
            already_fired=guard_state.turn_guard_fired,
        )
        if fault is not None:
            return None, fault.text
    if config.autopsy_fixes:
        fault = _unavailability_loop_fault(
            guarded,
            unavailable_tool_signatures=guard_state.unavailable_tool_signatures,
            already_fired=guard_state.unavailability_loop_fired,
        )
        if fault is not None:
            return None, fault.text
    if config.mutation_log_check:
        fault = _mutation_log_fault(guarded, guard_state, messages)
        if fault is not None:
            return None, fault.text
    if config.grounded_respond:
        fault = _grounded_respond_fault(
            guarded, messages, already_fired=guard_state.grounded_respond_fired
        )
        if fault is not None:
            return None, fault.text

    guarded = _apply_pre_mutation_guards(
        guarded,
        state=guard_state,
        messages=messages,
        config=config,
        tally=guard_tally,
        ctx_logger=guard_logger,
    )
    if config.event_exemplars:
        guarded, _ = _skip_satisfied_mutations(guarded, guard_state)
    fault = validate_next_action(guarded, guard_state.tools_by_name)
    if fault is not None:
        return None, fault.text
    return guarded, None


def _collect_zone_temperatures(value: Any, zones: dict[str, float]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_zone_temperatures(item, zones)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        match = re.fullmatch(
            r"climate_temperature_(driver|passenger)", str(key).casefold()
        )
        if match and isinstance(item, (int, float)) and not isinstance(item, bool):
            zones[match.group(1)] = float(item)
        else:
            _collect_zone_temperatures(item, zones)


def _resulting_front_zone_temperatures(
    messages: list[dict[str, Any]], state: _EpisodeState
) -> dict[str, float]:
    zones: dict[str, float] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        succeeded, _ = _tool_result_status(str(message.get("content") or ""))
        if not succeeded:
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except (TypeError, json.JSONDecodeError):
            continue
        _collect_zone_temperatures(payload, zones)
    for mutation in state.successful_mutations:
        if "climate_temperature" not in str(mutation.get("tool_name") or "").casefold():
            continue
        arguments = mutation.get("arguments") or {}
        temperature = arguments.get("temperature")
        seat_zone = str(arguments.get("seat_zone") or "").casefold()
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            continue
        if "driver" in seat_zone:
            zones["driver"] = float(temperature)
        elif "passenger" in seat_zone:
            zones["passenger"] = float(temperature)
        elif seat_zone in {"all", "front", "both"}:
            zones["driver"] = zones["passenger"] = float(temperature)
    return zones


def _draft_mentions_zone_difference(content: str) -> bool:
    lowered = content.casefold()
    return bool(
        re.search(r"\b(?:differ|difference|warmer|cooler)\w*\b", lowered)
        or ("driver" in lowered and "passenger" in lowered)
    )


def _draft_has_unitless_temperature(content: str) -> bool:
    # Agent policy 002: temperature must use degree Celsius.
    for match in re.finditer(
        r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", content
    ):
        before = content[max(0, match.start() - 32) : match.start()].casefold()
        suffix = content[match.end() : match.end() + 24].lstrip().casefold()
        has_degree_marker = bool(re.match(r"^(?:°|degrees?\b)", suffix))
        is_temperature_context = bool(
            re.search(r"\btemperature\b[^\n]{0,24}$", before)
        )
        has_celsius = bool(
            re.match(
                r"^(?:°\s*c\b|c\b|degrees?\s+celsius\b)", suffix
            )
        )
        if (has_degree_marker or is_temperature_context) and not has_celsius:
            return True
    return False


def _policy_lint_violations(
    action: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    state: _EpisodeState,
) -> list[tuple[str, str]]:
    if action.get("action") != "respond":
        return []
    content = str(action.get("content") or "")

    def zone_difference_missing() -> bool:
        # Agent policy 012: disclose a post-mutation single-zone difference >3 C.
        has_climate_mutation = any(
            "climate_temperature" in str(item.get("tool_name") or "").casefold()
            for item in state.successful_mutations
        )
        zones = _resulting_front_zone_temperatures(messages, state)
        return bool(
            has_climate_mutation
            and {"driver", "passenger"} <= zones.keys()
            and abs(zones["driver"] - zones["passenger"]) > 3.0
            and not _draft_mentions_zone_difference(content)
        )

    # Compiled from the stable agent-policy rules above. New post-conditions can
    # be appended without changing the executor control flow.
    post_conditions = (
        (
            "zone_difference",
            zone_difference_missing,
            "After the climate mutation, the driver and passenger temperatures "
            "differ by more than 3 degrees Celsius. Inform the user of that "
            "large zone difference and preserve the executed-action summary.",
        ),
        (
            "temperature_unit",
            lambda: _draft_has_unitless_temperature(content),
            "A temperature value lacks its Celsius unit. Revise it so every "
            "temperature value explicitly carries °C or degrees Celsius.",
        ),
    )
    return [(rule, note) for rule, check, note in post_conditions if check()]


def _validate_schema(value: Any, schema: dict[str, Any], *, path: str) -> str | None:
    if "const" in schema and value != schema["const"]:
        return f"{path} must equal {schema['const']!r}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']!r}"
    for keyword in ("allOf",):
        for branch in schema.get(keyword) or []:
            if isinstance(branch, dict):
                error = _validate_schema(value, branch, path=path)
                if error is not None:
                    return error
    for keyword in ("anyOf", "oneOf"):
        branches = [branch for branch in schema.get(keyword) or [] if isinstance(branch, dict)]
        if branches:
            valid = sum(_validate_schema(value, branch, path=path) is None for branch in branches)
            expected = 1 if keyword == "oneOf" else 0
            if (keyword == "oneOf" and valid != expected) or (keyword == "anyOf" and valid == 0):
                return f"{path} does not satisfy {keyword}"

    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(_matches_type(value, item) for item in expected_type):
            return f"{path} has the wrong type; expected one of {expected_type!r}"
    elif isinstance(expected_type, str) and not _matches_type(value, expected_type):
        return f"{path} has the wrong type; expected {expected_type}"

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for name in required:
            if name not in value:
                return f"{path}.{name} is required"
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                return f"{path} contains unsupported argument(s): {extras!r}"
        for name, item in value.items():
            prop = properties.get(name)
            if isinstance(prop, dict):
                error = _validate_schema(item, prop, path=f"{path}.{name}")
                if error is not None:
                    return error
    elif isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            return f"{path} has fewer than {schema['minItems']} items"
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            return f"{path} has more than {schema['maxItems']} items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_schema(item, item_schema, path=f"{path}[{index}]")
                if error is not None:
                    return error
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} must be >= {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} must be <= {schema['maximum']}"
        multiple = schema.get("multipleOf")
        if isinstance(multiple, (int, float)) and multiple:
            quotient = value / multiple
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                return f"{path} must be a multiple of {multiple}"
    return None


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def _fault_fallback_action(fault: InternalFault) -> dict[str, Any]:
    return {
        "action": "respond",
        "content": f"I couldn't issue that action because {fault.text}",
    }


def _apply_fastest_route_disclosure(
    content: str, messages: list[dict[str, Any]]
) -> tuple[str, int]:
    """Port p3i27.1 fastest disclosure selection semantics exactly."""

    _, fastest_ids = _route_disclosure_values(messages)
    normalized = content.casefold()
    if not fastest_ids or "alternative" in normalized or "more information" in normalized:
        return content, 0
    if "fastest" in normalized:
        addition = "Ask if you want information on alternative routes."
    else:
        selected_ids = _decision_selected_route_ids(content, messages, [])
        selected_fastest_ids = _selected_fastest_route_ids(messages, selected_ids)
        if not selected_fastest_ids:
            return content, 0
        ids = ", ".join(selected_fastest_ids)
        addition = (
            f"I selected the fastest available route ({ids}); ask if you want "
            "information on alternative routes."
        )
    return (f"{content.rstrip()} {addition}" if content.strip() else addition), 1


def build_planner_from_env(
    *,
    model: str,
    api_base: str,
    service_tier: str | None,
    reasoning_effort: str | None,
    temperature: float | None = None,
    max_completion_tokens: int = 1024,
    transport: str | None = None,
    logger: Any | None = None,
) -> AdaptiveMinimalPlanner:
    # Adaptive-minimal is intentionally fixed at the measured medium optimum.
    del reasoning_effort
    return AdaptiveMinimalPlanner(
        model=model,
        api_base=api_base,
        service_tier=service_tier,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        transport=(
            transport
            if transport is not None
            else os.getenv("TRACK2_TRANSPORT", DEFAULT_TRANSPORT)
        ),
        config=AdaptiveMinimalConfig.from_env(),
        logger=logger,
    )
