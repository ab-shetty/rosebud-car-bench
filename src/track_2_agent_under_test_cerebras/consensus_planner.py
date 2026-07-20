"""Consensus-routed planning harness for the Track 2 Cerebras agent.

Stage A/B/C of harness v2:

1. Scatter independent next-action drafts in one parallel fan-out.
2. Cluster the draft actions by action-equivalence.
3. Route by semantic/action agreement:
   COMMIT, DELIBERATE, CLARIFY, or VERIFY.
4. Route CLARIFY/VERIFY through standalone behavioral branches.
5. Verify the selected action with a weak-verifier ensemble.

This module does not import or extend the v1 guard layer.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

if __package__:
    from .car_bench_agent import (
        AgentInferenceResult,
        CEREBRAS_DEVELOPER_INSTRUCTIONS,
        NEXT_ACTION_OUTPUT_SCHEMA,
        _messages_for_prompt,
        parse_next_action,
    )
    from .cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        CerebrasCompletionClient,
        CerebrasTemplateError,
        MalformedModelResponseError,
    )
    from .scatter_sharpen import (
        ADVERSARIAL_SCHEMA,
        ADVERSARIAL_SYSTEM,
        CANDIDATE_REVIEW_SCHEMA,
        CANDIDATE_REVIEW_SYSTEM,
        DISPOSITIONS,
        PROPOSITION_KINDS,
        RECOMMENDATIONS,
        SCATTER_PASS_SCHEMA,
        SEQUENTIAL_LLM_BUDGET,
        SHARPEN_SCHEMA,
        SHARPEN_SYSTEM,
        Proposition,
        _adversarial_prompt,
        _CallTally,
        _candidate_review_prompt,
        _scatter_messages_for_pass,
        _scatter_prompt,
        _scatter_temperature_for_pass,
        _sharpen_prompt,
    )
    from .policy_rag import (
        PolicyCorpus,
        PolicySection,
        build_policy_corpus,
        extract_policy_subjects,
        policy_query_keys,
        policy_token_count,
        retrieve_policy_sections,
    )
else:  # pragma: no cover - exercised only when run as a loose script
    from car_bench_agent import (
        AgentInferenceResult,
        CEREBRAS_DEVELOPER_INSTRUCTIONS,
        NEXT_ACTION_OUTPUT_SCHEMA,
        _messages_for_prompt,
        parse_next_action,
    )
    from cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        CerebrasCompletionClient,
        CerebrasTemplateError,
        MalformedModelResponseError,
    )
    from scatter_sharpen import (
        ADVERSARIAL_SCHEMA,
        ADVERSARIAL_SYSTEM,
        CANDIDATE_REVIEW_SCHEMA,
        CANDIDATE_REVIEW_SYSTEM,
        DISPOSITIONS,
        PROPOSITION_KINDS,
        RECOMMENDATIONS,
        SCATTER_PASS_SCHEMA,
        SEQUENTIAL_LLM_BUDGET,
        SHARPEN_SCHEMA,
        SHARPEN_SYSTEM,
        Proposition,
        _adversarial_prompt,
        _CallTally,
        _candidate_review_prompt,
        _scatter_messages_for_pass,
        _scatter_prompt,
        _scatter_temperature_for_pass,
        _sharpen_prompt,
    )
    from policy_rag import (
        PolicyCorpus,
        PolicySection,
        build_policy_corpus,
        extract_policy_subjects,
        policy_query_keys,
        policy_token_count,
        retrieve_policy_sections,
    )


PLANNER_NAME = "consensus"
ROUTES = ("commit", "deliberate", "clarify", "verify")
AXES = ("none", "action", "intent", "claim")
DEFAULT_MODEL2VEC_MODEL_PATH = str(
    Path(__file__).resolve().parent / "assets" / "potion-base-8M"
)
DETERMINISTIC_VERIFIERS = (
    "tool_exists",
    "schema_valid",
    "args_grounded_in_trace",
    "not_user_rejected",
    "completion_check",
)
PROMPTED_VERIFIERS = (
    "policy_consistent",
    "read_before_act",
    "no_unsupported_claim",
)
# Fixed Stage B priors. Deterministic schema/tool checks receive higher weight
# because they are exact over the supplied tool schemas; prompted heads remain
# useful but noisier. Config can override this with verifier_weights later.
DEFAULT_VERIFIER_WEIGHTS = {
    "tool_exists": 1.5,
    "schema_valid": 1.4,
    "args_grounded_in_trace": 1.2,
    "not_user_rejected": 1.1,
    "completion_check": 1.0,
    "policy_consistent": 1.0,
    "read_before_act": 0.8,
    "no_unsupported_claim": 1.0,
}

POLICY_REMINDERS = """POLICY REMINDERS (from the in-car system policy; they override style habits):
- All datetimes in responses use 24h format (e.g. 18:00, never 6 PM). Distances in km/m; temperatures in deg C.
- Before calling any tool marked REQUIRES_CONFIRMATION: first send a message that names the action AND its exact parameter values (e.g. "I'll call set_head_lights_high_beams with on: true - confirm?"), then wait for an explicit yes.
- If a tool is marked REQUIRES_CONFIRMATION, or system policy conditions an action on user approval such as approval required under certain weather or vehicle states, you MUST ask exactly once with the exact tool name and parameter values, and act only after an explicit yes. This rule wins over everything below.
- Otherwise, a direct user command for an available car-state change is itself the authorization: do not ask a discretionary confirmation, do not re-read a state already read this episode for the same subject, and do not reply with an explanation instead of the action.
- For direct car-state commands, map scope and degree words to canonical tool arguments: all/every -> ALL or ALL_ZONES, fully/completely -> 100, close/off -> 0 or false or off, and on -> true. Use numeric values and zones stated by the user at runtime.
- After carrying out a direct car-state command, report what you did.
- When presenting route options, state whether each includes toll roads.
- If the user explicitly asks you to run a read/get tool that exists in your tool list, call it and report its actual output (even if values come back unknown) instead of refusing.
- If the user says to use a stored preference/setting, read it with the preferences tool and apply the returned value; do not ask the user to supply the value.
- When the user refers to a stored or preferred setting, read user preferences and apply the exact resolved value; if the preference is conditional, evaluate it against the vehicle state after your planned changes; never ask the user to choose between preference values you can retrieve.
- When the user commands a destination change, the job is finished only by navigation_replace_final_destination, or by set_new_navigation when no route is active; fetching routes alone changes nothing.
- To change the destination while navigation is active, call navigation_replace_final_destination directly; never delete the destination or add the new destination as a waypoint. If no route matches named roads/constraints and no route-pagination argument exists, pick the fastest available route, complete the change, and state the requested constraint was unavailable.
- Report only what tool results confirm. Never assert an outcome you cannot verify with a tool; state what you did, not what it achieved.
- Never send essentially the same message twice. If the user repeats a request after your refusal, do the closest available action instead of refusing again.
- When a requested capability is impossible with your tools, say so plainly once, state what you did instead, and do not end with a counter-question about that same capability."""

# Tiers 1+2 are the agent-skills layer (procedural behavior); tier 3 remains
# policy-only retrieval.  Line numbers are the frozen 16-line static inventory.
_POLICY_REMINDER_LINES = tuple(POLICY_REMINDERS.splitlines())
POLICY_INJECTION_TOKEN_BUDGET = policy_token_count(POLICY_REMINDERS)
POLICY_TIER1_LINE_NUMBERS = (1, 5, 7, 14, 15, 16)
POLICY_TIER2_LINE_NUMBERS: dict[str, tuple[int, ...]] = {
    "confirmation_protocol": (3, 4),
    "scope_degree_mapping": (6,),
    "route_presentation_disclosure": (8,),
    "explicit_read_actual_output": (9,),
    "preference_fidelity": (10, 11),
    "destination_mutation_completion": (12,),
    "active_navigation_mutation": (13,),
}
POLICY_TIER3_LINE_NUMBERS = (2,)

_TIER2_SCOPE_DEGREE_RE = re.compile(
    r"\b(?:all|every|entire|fully|completely|close|closed|off|on)\b",
    re.IGNORECASE,
)
_TIER2_DESTINATION_CHANGE_RE = re.compile(
    r"\b(?:change|replace|set|start|navigate|reroute|switch|update)\b"
    r"[^.;!?]{0,80}\b(?:destination|navigation|route|waypoint|stop)\b"
    r"|\b(?:destination|navigation|route|waypoint|stop)\b"
    r"[^.;!?]{0,80}\b(?:change|replace|set|start|navigate|reroute|switch|update)\b",
    re.IGNORECASE,
)
_TIER2_EXPLICIT_READ_RE = re.compile(
    r"\b(?:check|fetch|get|look up|read|retrieve|run|show|tell me|find)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyInjection:
    text: str
    tier1_tokens: int
    tier2_fired: tuple[str, ...]
    tier2_tokens: int
    tier3_core_tokens: int
    tier3_tail_tokens: int
    injected_tokens_total: int

STALL_NOTICE = """STALL NOTICE: You have made {count} consecutive attempts without progress on this request.
If the needed tool/argument is not in the supplied tool list, tell the user plainly that
this capability is unavailable, offer the closest available alternative once, and stop.
Do not ask another question about a capability you lack."""

REPEAT_NOTICE = (
    "REPEAT DETECTED: your last reply said this already; the user re-asked - "
    "execute the requested available tool or take a different concrete action."
)

REPEAT_CAP_NOTICE = """REPEAT LIMIT: The {tool_name} result has repeated without new information.
Tell the user the value is unavailable or unchanged based on the tool output, give the final
answer using what is known, and do not ask another counter-question or re-check the same tool."""

UNKNOWN_PROCEED_NOTICE = """UNKNOWN VALUE RULE: An unknown or unreadable value from a get/read tool is not by itself a reason to refuse an explicitly authorized action.
Report that the value is unknown, then proceed with the requested available action. If the selected tool requires confirmation, ask for confirmation with exact arguments first.
Do not state or assume the unknown value."""

NAV_ACTIVE_PREFLIGHT_NOTICE = (
    "PREFLIGHT: navigation active: use navigation_replace_final_destination / "
    "waypoint editing tools."
)

NAV_READ_PREFLIGHT_NOTICE = (
    "PREFLIGHT: before set_new_navigation, read get_current_navigation_state and "
    "only start a new navigation if navigation_active is false."
)

NAV_UNGROUNDED_PREFLIGHT_NOTICE = (
    "PREFLIGHT: argument not grounded in any tool result. Use a get/read tool first "
    "or choose an id that appears verbatim in a prior tool result."
)

NAME_GUESS_PREFLIGHT_NOTICE = (
    "PREFLIGHT: location name was not grounded in the user request or prior tool "
    "results. Ask the user or use a name they gave you."
)

OUTPUT_GUARD_TIME_PATTERN = re.compile(
    r"(?<!\w)(1[0-2]|[1-9])([:.][0-5][0-9])?\s?(a\.m\.|p\.m\.|AM|PM)(?!\w)",
    re.IGNORECASE,
)

VERIFIER_SYSTEM = """You are a CAR-bench action verifier.
Use only the provided conversation transcript, system/policy text visible in it,
the proposed assistant action, the sampled draft actions, and the supplied tool
schemas. Do not use external tool catalogs, hidden labels, external scores,
hidden categories, or external evaluation behavior. Return only JSON matching
the requested schema."""

PROMPTED_VERIFIER_SCHEMA = {
    "type": "object",
    "required": list(PROMPTED_VERIFIERS),
    "properties": {
        name: {
            "type": "object",
            "required": ["score", "rationale"],
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "1 means the proposed action passes this check.",
                },
                "rationale": {"type": "string"},
            },
            "additionalProperties": False,
        }
        for name in PROMPTED_VERIFIERS
    },
    "additionalProperties": False,
}

CLARIFY_SYSTEM = """You are a CAR-bench clarification planner.
Use only the supplied conversation transcript, system/policy text visible in it,
the sampled draft action representatives, and the supplied tool schemas. Ask one
short clarifying question that targets the disagreement. Do not introduce options
or facts not present in those inputs. Return only JSON matching the requested
schema."""

VOI_CLARIFY_SYSTEM = """You are a CAR-bench value-of-information question writer.
Use only the supplied conversation transcript, the disputed aspect, and the
supplied tool schemas. Ask exactly one short question about that aspect only.
Do not list internal candidates, draft actions, tool names, or hidden options.
Return only JSON matching the requested schema."""

CLARIFY_SCHEMA = {
    "type": "object",
    "required": ["action", "content", "tool_calls"],
    "properties": {
        "action": {"type": "string", "enum": ["respond"]},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": json.loads(
                json.dumps(NEXT_ACTION_OUTPUT_SCHEMA["properties"]["tool_calls"]["items"])
            ),
        },
    },
    "additionalProperties": False,
}


@dataclass
class ConsensusPlannerConfig:
    """Tunable knobs for consensus routing and verifier aggregation."""

    scatter_width: int = 6
    max_sharpen_iters: int = 2
    resolve_threshold: float = 0.8
    max_rescatters: int = 2
    scatter_temperature: float | None = 0.7
    sharpen_temperature: float | None = 0.2
    scatter_max_completion_tokens: int = 2048
    sharpen_max_completion_tokens: int = 2048
    adversarial_max_completion_tokens: int = 1024
    max_parallel_clients: int = 8
    scatter_diverse: bool = True
    enable_consensus_route: bool = True
    commit_threshold: float = 0.7
    enable_verifier_ensemble: bool = True
    verifier_veto_threshold: float = 0.5
    verifier_weights: dict[str, float] | None = None
    enable_clarify_branch: bool = True
    enable_verify_branch: bool = True
    enable_voi_clarify: bool = False
    enable_grounded_retry: bool = False
    scatter_quorum_shadow: bool = False
    scatter_quorum: int = 0
    quorum_live: bool = False
    fold_clarify: bool = False
    policy_reminders: bool = False
    policy_rag: bool = False
    decisive_limits: bool = False
    output_guards: bool = False
    autpol_cascade: bool = False
    spiral_cap: bool = False
    spiral_cap_tokens: int = 900_000
    spiral_cap_turns: int = 10
    spiral_cap_token_min_turns: int = 20
    unknown_proceed: bool = False
    response_guards_v2: bool = False
    nav_preflight: bool = False
    repair_containment: bool = False
    loop_breaker: bool = False
    nav_detailed_read: bool = False
    completion_check: bool = False
    scope_shadow: bool = False
    act_on_approval: bool = False
    voi_evpi_alpha: float = 0.1
    voi_redundancy_lambda: float = 0.5
    deterministic_commit: bool = False
    embedding_model: str = DEFAULT_MODEL2VEC_MODEL_PATH
    embedding_cluster_threshold: float = 0.45
    token_overlap_cluster_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.policy_reminders and self.policy_rag:
            raise ValueError(
                "TRACK2_POLICY_REMINDERS and TRACK2_POLICY_RAG are mutually exclusive"
            )

    @classmethod
    def from_env(cls) -> "ConsensusPlannerConfig":
        def _int(name: str, default: int) -> int:
            value = os.getenv(name)
            return int(value) if value and value.strip() else default

        def _clamped_int(name: str, default: int, low: int, high: int) -> int:
            return max(low, min(high, _int(name, default)))

        def _float(name: str, default: float) -> float:
            value = os.getenv(name)
            return float(value) if value and value.strip() else default

        def _opt_float(name: str, default: float | None) -> float | None:
            value = os.getenv(name)
            if value is None or not value.strip():
                return default
            return float(value)

        def _bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None or not value.strip():
                return default
            return value.strip().lower() not in {"0", "false", "no", "off"}

        return cls(
            scatter_width=_int("TRACK2_SCATTER_WIDTH", cls.scatter_width),
            max_sharpen_iters=_clamped_int(
                "TRACK2_SHARPEN_MAX_ITERS", cls.max_sharpen_iters, 0, 2
            ),
            resolve_threshold=_float(
                "TRACK2_SCATTER_RESOLVE_THRESHOLD", cls.resolve_threshold
            ),
            max_rescatters=_int("TRACK2_SHARPEN_MAX_RESCATTERS", cls.max_rescatters),
            scatter_temperature=_opt_float(
                "TRACK2_SCATTER_TEMPERATURE", cls.scatter_temperature
            ),
            sharpen_temperature=_opt_float(
                "TRACK2_SHARPEN_TEMPERATURE", cls.sharpen_temperature
            ),
            scatter_max_completion_tokens=_int(
                "TRACK2_SCATTER_MAX_COMPLETION_TOKENS",
                cls.scatter_max_completion_tokens,
            ),
            sharpen_max_completion_tokens=_int(
                "TRACK2_SHARPEN_MAX_COMPLETION_TOKENS",
                cls.sharpen_max_completion_tokens,
            ),
            adversarial_max_completion_tokens=_int(
                "TRACK2_ADVERSARIAL_MAX_COMPLETION_TOKENS",
                cls.adversarial_max_completion_tokens,
            ),
            max_parallel_clients=_int(
                "TRACK2_SCATTER_MAX_PARALLEL_CLIENTS", cls.max_parallel_clients
            ),
            scatter_diverse=_bool(
                "TRACK2_SCATTER_DIVERSE", cls.scatter_diverse
            ),
            enable_consensus_route=_bool(
                "TRACK2_ENABLE_CONSENSUS_ROUTE", cls.enable_consensus_route
            ),
            commit_threshold=_float(
                "TRACK2_COMMIT_THRESHOLD", cls.commit_threshold
            ),
            enable_verifier_ensemble=_bool(
                "TRACK2_ENABLE_VERIFIER_ENSEMBLE",
                cls.enable_verifier_ensemble,
            ),
            verifier_veto_threshold=_float(
                "TRACK2_VERIFIER_VETO_THRESHOLD",
                cls.verifier_veto_threshold,
            ),
            verifier_weights=_verifier_weights_from_env(),
            enable_clarify_branch=_bool(
                "TRACK2_ENABLE_CLARIFY_BRANCH",
                cls.enable_clarify_branch,
            ),
            enable_verify_branch=_bool(
                "TRACK2_ENABLE_VERIFY_BRANCH",
                cls.enable_verify_branch,
            ),
            enable_voi_clarify=_bool(
                "TRACK2_ENABLE_VOI_CLARIFY",
                cls.enable_voi_clarify,
            ),
            enable_grounded_retry=_bool(
                "TRACK2_ENABLE_GROUNDED_RETRY",
                cls.enable_grounded_retry,
            ),
            scatter_quorum_shadow=_bool(
                "TRACK2_SCATTER_QUORUM_SHADOW",
                cls.scatter_quorum_shadow,
            ),
            scatter_quorum=_int(
                "TRACK2_SCATTER_QUORUM",
                cls.scatter_quorum,
            ),
            quorum_live=_bool(
                "TRACK2_QUORUM_LIVE",
                cls.quorum_live,
            ),
            fold_clarify=_bool(
                "TRACK2_FOLD_CLARIFY",
                cls.fold_clarify,
            ),
            policy_reminders=_bool(
                "TRACK2_POLICY_REMINDERS",
                cls.policy_reminders,
            ),
            policy_rag=_bool(
                "TRACK2_POLICY_RAG",
                cls.policy_rag,
            ),
            decisive_limits=_bool(
                "TRACK2_DECISIVE_LIMITS",
                cls.decisive_limits,
            ),
            output_guards=_bool(
                "TRACK2_OUTPUT_GUARDS",
                cls.output_guards,
            ),
            autpol_cascade=_bool(
                "TRACK2_AUTPOL_CASCADE",
                cls.autpol_cascade,
            ),
            spiral_cap=_bool(
                "TRACK2_SPIRAL_CAP",
                cls.spiral_cap,
            ),
            spiral_cap_tokens=_int(
                "SPIRAL_CAP_TOKENS",
                cls.spiral_cap_tokens,
            ),
            spiral_cap_turns=_int(
                "SPIRAL_CAP_TURNS",
                cls.spiral_cap_turns,
            ),
            spiral_cap_token_min_turns=_int(
                "SPIRAL_CAP_TOKEN_MIN_TURNS",
                cls.spiral_cap_token_min_turns,
            ),
            unknown_proceed=_bool(
                "TRACK2_UNKNOWN_PROCEED",
                cls.unknown_proceed,
            ),
            response_guards_v2=_bool(
                "TRACK2_RESPONSE_GUARDS_V2",
                cls.response_guards_v2,
            ),
            nav_preflight=_bool(
                "TRACK2_NAV_PREFLIGHT",
                cls.nav_preflight,
            ),
            repair_containment=_bool(
                "TRACK2_REPAIR_CONTAINMENT",
                cls.repair_containment,
            ),
            loop_breaker=_bool(
                "TRACK2_LOOP_BREAKER",
                cls.loop_breaker,
            ),
            nav_detailed_read=_bool(
                "TRACK2_NAV_DETAILED_READ",
                cls.nav_detailed_read,
            ),
            completion_check=_bool(
                "TRACK2_COMPLETION_CHECK",
                cls.completion_check,
            ),
            scope_shadow=_bool(
                "TRACK2_SCOPE_SHADOW",
                cls.scope_shadow,
            ),
            act_on_approval=_bool(
                "TRACK2_ACT_ON_APPROVAL",
                cls.act_on_approval,
            ),
            voi_evpi_alpha=_float(
                "TRACK2_VOI_EVPI_ALPHA",
                cls.voi_evpi_alpha,
            ),
            voi_redundancy_lambda=_float(
                "TRACK2_VOI_REDUNDANCY_LAMBDA",
                cls.voi_redundancy_lambda,
            ),
            deterministic_commit=_bool(
                "TRACK2_DETERMINISTIC_COMMIT",
                cls.deterministic_commit,
            ),
            embedding_model=(
                os.getenv("TRACK2_EMBEDDING_MODEL", cls.embedding_model).strip()
                or cls.embedding_model
            ),
            embedding_cluster_threshold=_float(
                "TRACK2_EMBEDDING_CLUSTER_THRESHOLD",
                cls.embedding_cluster_threshold,
            ),
            token_overlap_cluster_threshold=_float(
                "TRACK2_TOKEN_OVERLAP_CLUSTER_THRESHOLD",
                cls.token_overlap_cluster_threshold,
            ),
        )


@dataclass
class ClusterResult:
    """Draft action clusters and routing signals."""

    clusters: list[list[dict[str, Any]]]
    top_share: float
    entropy: float
    axis: str
    representatives: list[dict[str, Any]]


@dataclass
class VerifierVote:
    """One weak verifier's normalized vote over the selected action."""

    name: str
    score: float
    veto: bool
    repair: dict[str, Any] | None
    rationale: str


@dataclass
class VerifierEnsembleResult:
    """Weighted verifier aggregate and selected repaired/final action."""

    action: dict[str, Any]
    votes: list[VerifierVote]
    score: float
    decision: str
    vetoed: bool
    repaired: bool


@dataclass(frozen=True)
class VoiAspect:
    """The single ambiguous aspect selected for read/ask/act routing."""

    name: str
    tokens: frozenset[str]
    kind: str


@dataclass
class GroundedRetryState:
    """Per-episode bounds for grounded tool-error correction."""

    total_rounds: int = 0
    corrected_call_sites: set[str] = field(default_factory=set)
    awaiting_corrective_result: bool = False


@dataclass
class StallState:
    """Per-episode no-progress tracking for decisive limit nudges."""

    processed_messages: int = 0
    successful_call_sites: set[str] = field(default_factory=set)
    stall_count: int = 0
    max_stall_count: int = 0
    nudges_injected: int = 0
    hard_stops: int = 0


@dataclass
class GuardState:
    """Per-episode deterministic guard memory."""

    confirmed_call_sites: set[str] = field(default_factory=set)
    emitted_cascade_sites: set[str] = field(default_factory=set)
    unknown_proceed_refusal_sites: dict[str, int] = field(default_factory=dict)


@dataclass
class RepeatNudgeState:
    """Per-episode repeat-nudge budget."""

    nudges_injected: int = 0
    capped: int = 0


@dataclass
class SpiralCapState:
    """Per-episode token/turn economy-cap accounting."""

    billed_tokens: int = 0
    assistant_turns: int = 0
    engaged: bool = False
    engaged_tokens: int = 0
    engaged_turns: int = 0


@dataclass
class RepairState:
    """Per-episode verifier-repair instrumentation."""

    grounding_repair_fires: int = 0


@dataclass
class PreflightState:
    """Per-episode final-preflight block accounting."""

    block_signatures: dict[tuple[str, str], int] = field(default_factory=dict)
    response_templates: dict[tuple[str, str], int] = field(default_factory=dict)
    corrective_reads: int = 0
    corrective_reads_suppressed: bool = False
    max_segment_mismatch_blocks: int = 0
    terminal_honesty_signatures: set[tuple[str, str]] = field(default_factory=set)
    route_substitution_hints: int = 0
    route_arg_substitutions: int = 0
    substitution_tool_call_key: str | None = None
    substitution_emission_pending: bool = False
    post_substitution_extra_nav_mutations: int = 0


@dataclass
class AnnouncedCallState:
    """Per-episode bounds for executing fully specified response announcements."""

    executions: int = 0


@dataclass
class CompletionState:
    """Per-episode completion-check escalation and deferral accounting."""

    notice_fires: dict[str, int] = field(default_factory=dict)
    respond_deferred: bool = False


@dataclass
class LoopBreakerState:
    """Per-episode loop-breaker one-shot state."""

    fired: bool = False
    suppressed_repeats: int = 0
    pending_action: dict[str, Any] | None = None
    fired_message_index: int | None = None


@dataclass(frozen=True)
class GroundedRetrySignal:
    """A concrete tool failure observation that can or cannot be corrected."""

    error_text: str
    carveout: bool
    reason: str
    tool_name: str | None
    tool_call_id: str | None
    failed_call: dict[str, Any] | None
    call_site_key: str


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return one vector per text, or None when embeddings are unavailable."""


class Model2VecBackend:
    """Lazy local model2vec backend with no torch dependency."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._failed = False

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if self._failed:
            return None
        try:
            if self._model is None:
                from model2vec import StaticModel

                self._model = StaticModel.from_pretrained(self.model_path)
            vectors = self._model.encode(
                texts,
                show_progress_bar=False,
                use_multiprocessing=False,
            )
            rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
            return [_normalized_vector(row) for row in rows]
        except Exception:
            self._failed = True
            return None


class SentenceTransformerBackend:
    """Lazy sentence-transformer backend retained for explicit experiments."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._failed = False

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if self._failed:
            return None
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            vectors = self._model.encode(
                texts,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            return [list(map(float, vector)) for vector in vectors]
        except Exception:
            self._failed = True
            return None


class ConsensusPlanner:
    """Produces one CAR-bench next action via consensus-routed scatter."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str = DEFAULT_CEREBRAS_API_BASE,
        service_tier: str | None = None,
        reasoning_effort: str | None = None,
        config: ConsensusPlannerConfig | None = None,
        logger: Any | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort
        self.config = config or ConsensusPlannerConfig()
        self.logger = logger
        self.embedding_backend = embedding_backend or Model2VecBackend(
            self.config.embedding_model
        )
        self._client_pool: list[CerebrasCompletionClient] = []
        self._grounded_retry_state_by_context: dict[str, GroundedRetryState] = {}
        self._stall_state_by_context: dict[str, StallState] = {}
        self._guard_state_by_context: dict[str, GuardState] = {}
        self._repeat_state_by_context: dict[str, RepeatNudgeState] = {}
        self._spiral_state_by_context: dict[str, SpiralCapState] = {}
        self._repair_state_by_context: dict[str, RepairState] = {}
        self._preflight_state_by_context: dict[str, PreflightState] = {}
        self._completion_state_by_context: dict[str, CompletionState] = {}
        self._announced_call_state_by_context: dict[str, AnnouncedCallState] = {}
        self._loop_breaker_state_by_context: dict[str, LoopBreakerState] = {}
        self._policy_corpus_by_context: dict[str, PolicyCorpus] = {}
        self._policy_rag_local = threading.local()
        self._pending_straggler_usage: list[Any] = []
        self._pending_straggler_lock = threading.Lock()
        self._last_scatter_meta: dict[str, Any] = {}
        self.last_decision: dict[str, Any] | None = None

    def plan(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> AgentInferenceResult:
        tally = _CallTally()
        straggler_tokens_deferred = self._drain_pending_stragglers(tally)
        transcript = _messages_for_prompt(messages)
        self._policy_rag_local.corpus = self._policy_corpus_for_context(
            context_id=context_id,
            messages=messages,
            tools=tools,
        )
        self._policy_rag_local.available_tool_names = frozenset(
            _available_tool_names(tools)
        )
        self._policy_rag_local.requires_confirmation_tool_names = frozenset(
            _requires_confirmation_tool_names(tools)
        )
        self._policy_rag_local.policy_conditioned_confirmation_tool_names = (
            _policy_conditioned_confirmation_tool_names(
                self._policy_rag_local.corpus
            )
        )
        completion_negation_meta = _completion_negation_meta(messages)
        spiral_state = self._update_spiral_turns(context_id, messages)
        economy_mode = self._spiral_cap_active(spiral_state)
        stall_state = self._update_stall_state(context_id, messages)
        stall_count = stall_state.stall_count if self.config.decisive_limits else 0
        stall_notice = (
            STALL_NOTICE.format(count=stall_count)
            if self.config.decisive_limits and stall_count >= 3
            else None
        )
        if stall_notice is not None:
            stall_state.nudges_injected += 1
        grounded_retry_correction, grounded_retry_meta = self._grounded_retry_correction(
            context_id=context_id,
            messages=messages,
            ctx_logger=ctx_logger,
        )
        repeat_notice, repeat_capped = (
            self._repeat_notice_for_context(context_id, messages)
            if self.config.policy_reminders or self.config.policy_rag
            else (None, False)
        )
        completion_notice_fires = 0
        confirmed_action = _confirmed_tool_call_from_latest_affirmation(
            messages,
            tools=tools,
            guard_state=self._guard_state_by_context.setdefault(context_id, GuardState()),
        )
        loop_breaker_confirmed = False
        if confirmed_action is None and self.config.loop_breaker:
            confirmed_action = self._loop_breaker_confirmed_action(
                context_id=context_id,
                messages=messages,
                tools=tools,
            )
            loop_breaker_confirmed = confirmed_action is not None
        if confirmed_action is not None:
            self.last_decision = {
                "route": "confirmed_tool_call",
                "action": confirmed_action.get("action"),
                "confirmed_tool_call": 1,
                "loop_breaker_confirmed": int(loop_breaker_confirmed),
                "loop_breaker_fires": 0,
                "loop_breaker_suppressed_repeats": 0,
                "loop_breaker_kind": None,
                "loop_breaker_message_index": None,
                "nav_detailed_rewrites": 0,
                "spiral_cap_enabled": self.config.spiral_cap,
                "spiral_cap_engaged": self._spiral_cap_active(spiral_state),
                "spiral_cap_episode_tokens": spiral_state.billed_tokens,
                "spiral_cap_episode_turns": spiral_state.assistant_turns,
                "repeat_nudge_injected": repeat_notice is not None,
                "repeat_nudge_capped": repeat_capped,
                "sequential_calls": 0,
                "total_calls": 0,
                "seq_budget_skips": 0,
                "completion_notices": completion_notice_fires,
                "completion_vote_zeros": 0,
                "completion_respond_deferrals": 0,
                **completion_negation_meta,
                "terminal_honesty_escalations": 0,
                "route_substitution_hints": 0,
                "route_arg_substitutions": 0,
                "announced_call_executions": 0,
                "post_substitution_extra_nav_mutations": 0,
                "scope_shadow_flags": 0,
                "scope_shadow_inverse_flags": 0,
                **_policy_rag_meta(tally),
            }
            ctx_logger.info(
                "Consensus deterministic confirmed tool call",
                context=context_id,
                tool_call=confirmed_action.get("tool_calls", [{}])[0].get("tool_name"),
                **completion_negation_meta,
            )
            confirmed_action, preflight_meta = self._final_preflight_or_respond(
                context_id=context_id,
                action=confirmed_action,
                messages=messages,
                ctx_logger=ctx_logger,
            )
            confirmed_action, nav_detail_meta = self._apply_nav_detailed_read(
                confirmed_action,
                ctx_logger=ctx_logger,
            )
            post_substitution_meta = self._post_substitution_mutation_meta(
                context_id=context_id,
                action=confirmed_action,
                ctx_logger=ctx_logger,
            )
            mandated_ask_meta = _mandated_ask_completion_meta(
                action=confirmed_action,
                messages=messages,
                tools=tools,
                completion_notices=0,
                completion_respond_deferrals=0,
                completion_pivotal_veto=False,
            )
            self.last_decision.update(preflight_meta)
            self.last_decision.update(nav_detail_meta)
            self.last_decision.update(post_substitution_meta)
            self.last_decision.update(mandated_ask_meta)
            self.last_decision["action"] = confirmed_action.get("action")
            return self._result(confirmed_action, tally)

        passes = self._scatter(
            transcript,
            tools,
            tally,
            ctx_logger,
            grounded_retry_correction=grounded_retry_correction,
            stall_notice=stall_notice,
            repeat_notice=repeat_notice,
            economy_mode=economy_mode,
        )
        tally.sequential_calls += 1
        if not passes:
            ctx_logger.warning("Consensus scatter empty; using single-action fallback")
            action = self._single_action(
                transcript,
                tools,
                tally,
                ctx_logger,
                grounded_retry_correction=grounded_retry_correction,
            )
            loop_breaker_meta = {
                "loop_breaker_fires": 0,
                "loop_breaker_suppressed_repeats": 0,
                "loop_breaker_kind": None,
                "loop_breaker_message_index": None,
            }
            if self.config.loop_breaker:
                action, loop_breaker_meta = self._apply_loop_breaker(
                    context_id=context_id,
                    final_action=action,
                    passes=[],
                    messages=messages,
                    tools=tools,
                    ctx_logger=ctx_logger,
                )
            action, announced_meta = self._apply_announced_call_execution(
                context_id=context_id,
                action=action,
                messages=messages,
                tools=tools,
                ctx_logger=ctx_logger,
            )
            action, preflight_meta = self._final_preflight_or_respond(
                context_id=context_id,
                action=action,
                messages=messages,
                ctx_logger=ctx_logger,
            )
            action, nav_detail_meta = self._apply_nav_detailed_read(
                action,
                ctx_logger=ctx_logger,
            )
            post_substitution_meta = self._post_substitution_mutation_meta(
                context_id=context_id,
                action=action,
                ctx_logger=ctx_logger,
            )
            mandated_ask_meta = _mandated_ask_completion_meta(
                action=action,
                messages=messages,
                tools=tools,
                completion_notices=0,
                completion_respond_deferrals=0,
                completion_pivotal_veto=False,
            )
            self.last_decision = {
                "route": "single_action_fallback",
                **loop_breaker_meta,
                **announced_meta,
                **preflight_meta,
                **nav_detail_meta,
                **post_substitution_meta,
                **mandated_ask_meta,
                "action": action.get("action"),
                "sequential_calls": tally.sequential_calls,
                "total_calls": tally.total_calls,
                "seq_budget_skips": tally.seq_budget_skips,
                "completion_notices": completion_notice_fires,
                "completion_vote_zeros": 0,
                "completion_respond_deferrals": 0,
                **completion_negation_meta,
                "scope_shadow_flags": 0,
                "scope_shadow_inverse_flags": 0,
                **_policy_rag_meta(tally),
            }
            return self._result(action, tally)

        plan_action, queue = self._aggregate(passes)
        if self.config.completion_check and plan_action.get("action") == "respond":
            completion_notice, completion_notice_fires = _completion_notice(
                messages,
                tools,
                self._completion_state_by_context.setdefault(
                    context_id, CompletionState()
                ),
            )
            if completion_notice is not None:
                stall_notice = _append_notice(stall_notice, completion_notice)
        cluster_result, route = scatter_cluster_drafts_route(
            passes,
            commit_threshold=self.config.commit_threshold,
            embedding_backend=self.embedding_backend,
            embedding_cluster_threshold=self.config.embedding_cluster_threshold,
            token_overlap_cluster_threshold=(
                self.config.token_overlap_cluster_threshold
            ),
        )
        scatter_meta = dict(self._last_scatter_meta)
        live_quorum_committed = bool(scatter_meta.get("quorum_commit"))

        stall_hard_stop = False
        if (
            self.config.decisive_limits
            and stall_count >= 6
            and _acknowledge_limit_majority(passes)
        ):
            final_action = _select_acknowledge_limit_action(passes)
            queue = []
            route = "commit"
            stall_hard_stop = True
            stall_state.hard_stops += 1
            deliberate_meta = {
                "branch": "stall_hard_stop",
                "candidate_reviewed": False,
                "sharpen_iters": 0,
                "rescatters": 0,
                "final_reviewed": False,
            }
        elif not self.config.enable_consensus_route:
            route = "deliberate"
            final_action, queue, deliberate_meta = self._run_deliberate_pipeline(
                transcript=transcript,
                tools=tools,
                passes=passes,
                route=route,
                plan_action=plan_action,
                queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
                stall_notice=stall_notice,
                economy_mode=economy_mode,
            )
        elif route == "commit":
            plan_action = _select_committed_action(
                cluster_result,
                deterministic=self.config.deterministic_commit,
                route=route,
                ctx_logger=ctx_logger,
            )
            final_action = plan_action
            queue = []
            deliberate_meta = {
                "branch": "commit",
                "candidate_reviewed": False,
                "sharpen_iters": 0,
                "rescatters": 0,
                "final_reviewed": False,
            }
        elif route == "clarify":
            if self.config.fold_clarify:
                final_action, queue, deliberate_meta = self._run_deliberate_pipeline(
                    transcript=transcript,
                    tools=tools,
                    passes=passes,
                    route=route,
                    plan_action=plan_action,
                    queue=queue,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    stall_notice=stall_notice,
                    economy_mode=economy_mode,
                )
                deliberate_meta["clarify_folded"] = True
            elif self.config.enable_clarify_branch:
                if self.config.enable_voi_clarify:
                    final_action, deliberate_meta = self._run_voi_branch(
                        transcript=transcript,
                        tools=tools,
                        passes=passes,
                        cluster_result=cluster_result,
                        tally=tally,
                        ctx_logger=ctx_logger,
                    )
                else:
                    final_action, deliberate_meta = self._run_clarify_branch(
                        transcript=transcript,
                        tools=tools,
                        passes=passes,
                        cluster_result=cluster_result,
                        tally=tally,
                        ctx_logger=ctx_logger,
                    )
            else:
                final_action = _deterministic_clarify_action(
                    cluster_result,
                    transcript=transcript,
                    tools=tools,
                )
                deliberate_meta = _branch_meta("clarify_disabled")
            queue = []
        elif route == "verify":
            # v3 removes the standalone verify branch. Claim disagreement is
            # handled by the same deliberation core and verifier ensemble as
            # other uncertain decisions; enable_verify_branch remains only as a
            # no-op ablation alias.
            final_action, queue, deliberate_meta = self._run_deliberate_pipeline(
                transcript=transcript,
                tools=tools,
                passes=passes,
                route=route,
                plan_action=plan_action,
                queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
                stall_notice=stall_notice,
                economy_mode=economy_mode,
            )
        else:
            final_action, queue, deliberate_meta = self._run_deliberate_pipeline(
                transcript=transcript,
                tools=tools,
                passes=passes,
                route=route,
                plan_action=plan_action,
                queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
                stall_notice=stall_notice,
                economy_mode=economy_mode,
            )

        reroute_meta: dict[str, Any] = {
            "unknown_proceed_reroutes": 0,
            "unknown_proceed_draft_override": 0,
            "unknown_proceed_confirm_ask": 0,
            "preflight_nav_active_blocks": 0,
            "preflight_navread_blocks": 0,
            "preflight_ungrounded_blocks": 0,
            "preflight_segment_mismatch_blocks": 0,
            "preflight_corrective_reads": 0,
            "preflight_corrective_read_suppressed": 0,
            "preflight_segment_mismatch_regressions": 0,
            "terminal_honesty_escalations": 0,
            "route_substitution_hints": 0,
            "route_arg_substitutions": 0,
            "preflight_capped_template_blocks": 0,
            "preflight_type_blocks": 0,
            "preflight_name_guess_blocks": 0,
            "nav_detailed_rewrites": 0,
        }
        guard_state = self._guard_state_by_context.setdefault(context_id, GuardState())
        if self.config.unknown_proceed:
            transformed, transform_counter = _unknown_proceed_transform(
                final_action=final_action,
                passes=passes,
                messages=messages,
                tools=tools,
                guard_state=guard_state,
            )
            if transformed is not None and transform_counter is not None:
                final_action = transformed
                reroute_meta[transform_counter] = 1
                live_quorum_committed = False

        verifier_meta: dict[str, Any] = {
            "verifier_ensemble": False,
            "verifier_score": None,
            "verifier_decision": "skipped",
            "verifier_vetoed": False,
            "verifier_repaired": False,
            "grounding_repair_reroutes": 0,
            "grounding_repair_fires": self._repair_state_by_context.setdefault(
                context_id, RepairState()
            ).grounding_repair_fires,
            "repair_reverts_block": 0,
            "repair_reverts_newtool": 0,
            "pref_read_commits": 0,
            "pref_arg_rewrites": 0,
            "grounding_fallback_commits": 0,
            "completion_vote_zeros": 0,
        }
        pre_repair_action_for_containment: dict[str, Any] | None = None
        if self.config.enable_verifier_ensemble and not live_quorum_committed:
            if economy_mode:
                verifier_result = self._run_deterministic_verifier_ensemble(
                    plan_action=final_action,
                    transcript=transcript,
                    tools=tools,
                    passes=passes,
                    ctx_logger=ctx_logger,
                )
            else:
                verifier_result = self._run_verifier_ensemble(
                    plan_action=final_action,
                    transcript=transcript,
                    tools=tools,
                    passes=passes,
                    tally=tally,
                    ctx_logger=ctx_logger,
                )
            pre_verifier_action = final_action
            final_action = verifier_result.action
            repair_state = self._repair_state_by_context.setdefault(
                context_id, RepairState()
            )
            if _verifier_grounding_repair_rerouted(verifier_result):
                repair_state.grounding_repair_fires += 1
                pre_repair_action_for_containment = pre_verifier_action
                ctx_logger.info(
                    "Consensus grounding repair action pair",
                    grounding_repair_fires=repair_state.grounding_repair_fires,
                    pre_action=_compact_action_summary(pre_verifier_action),
                    post_action=_compact_action_summary(final_action),
                )
                if (
                    self.config.repair_containment
                    and not _repair_tool_names_subset(pre_verifier_action, final_action)
                ):
                    ctx_logger.info(
                        "Consensus grounding repair reverted new tool",
                        pre_action=_compact_action_summary(pre_verifier_action),
                        post_action=_compact_action_summary(final_action),
                    )
                    final_action = pre_verifier_action
                    pre_repair_action_for_containment = None
            verifier_meta = {
                "verifier_ensemble": True,
                "verifier_score": verifier_result.score,
                "verifier_decision": verifier_result.decision,
                "verifier_vetoed": verifier_result.vetoed,
                "verifier_repaired": verifier_result.repaired,
                "grounding_repair_reroutes": int(
                    _verifier_grounding_repair_rerouted(verifier_result)
                ),
                "grounding_repair_fires": repair_state.grounding_repair_fires,
                "repair_reverts_block": 0,
                "repair_reverts_newtool": int(
                    self.config.repair_containment
                    and _verifier_grounding_repair_rerouted(verifier_result)
                    and not _repair_tool_names_subset(pre_verifier_action, verifier_result.action)
                ),
                **_verifier_grounding_repair_counters(verifier_result),
                "completion_vote_zeros": _completion_vote_zero_count(verifier_result),
            }

        if self.config.enable_voi_clarify:
            final_action = _sanitize_candidate_leak_response(
                final_action,
                disputed_aspect=deliberate_meta.get("disputed_aspect"),
                route=route,
                ctx_logger=ctx_logger,
            )

        completion_respond_deferrals = 0
        if self.config.completion_check and final_action.get("action") == "respond":
            completion_state = self._completion_state_by_context.setdefault(
                context_id, CompletionState()
            )
            completion_signal = _completion_signal(messages, tools)
            if completion_signal is not None and not completion_state.respond_deferred:
                if tally.has_sequential_budget():
                    completion_state.respond_deferred = True
                    completion_respond_deferrals = 1
                    completion_notice, added_fires = _completion_notice(
                        messages,
                        tools,
                        completion_state,
                        force_escalated=True,
                    )
                    completion_notice_fires += added_fires
                    final_action, queue, completion_meta = self._run_deliberate_pipeline(
                        transcript=transcript,
                        tools=tools,
                        passes=passes,
                        route="deliberate",
                        plan_action=final_action,
                        queue=queue,
                        tally=tally,
                        ctx_logger=ctx_logger,
                        stall_notice=_append_notice(stall_notice, completion_notice),
                        economy_mode=economy_mode,
                    )
                    deliberate_meta.update(completion_meta)
                    ctx_logger.info(
                        "Consensus completion respond deferred once",
                        candidates=completion_signal["candidates"],
                        sequential_calls=tally.sequential_calls,
                        action=_compact_action_summary(final_action),
                    )
                else:
                    tally.skip_sequential()
                    ctx_logger.info(
                        "Consensus completion respond deferral skipped for sequential budget",
                        candidates=completion_signal["candidates"],
                        sequential_calls=tally.sequential_calls,
                        seq_budget_skips=tally.seq_budget_skips,
                    )

        loop_breaker_meta = {
            "loop_breaker_fires": 0,
            "loop_breaker_suppressed_repeats": 0,
            "loop_breaker_kind": None,
            "loop_breaker_message_index": None,
        }
        if self.config.loop_breaker:
            final_action, loop_breaker_meta = self._apply_loop_breaker(
                context_id=context_id,
                final_action=final_action,
                passes=passes,
                messages=messages,
                tools=tools,
                ctx_logger=ctx_logger,
            )

        final_action, announced_meta = self._apply_announced_call_execution(
            context_id=context_id,
            action=final_action,
            messages=messages,
            tools=tools,
            ctx_logger=ctx_logger,
        )

        final_action, output_guard_meta = _apply_output_guards(
            final_action=final_action,
            passes=passes,
            tools=tools,
            messages=messages,
            guard_state=guard_state,
            enabled=self.config.output_guards,
            cascade_enabled=self.config.autpol_cascade,
            response_guards_v2=self.config.response_guards_v2,
        )
        if self.config.nav_preflight:
            final_action, queue, preflight_meta, rerouted_meta = (
                self._final_preflight_reroute_once(
                    context_id=context_id,
                    action=final_action,
                    repair_fallback_action=(
                        pre_repair_action_for_containment
                        if self.config.repair_containment
                        else None
                    ),
                    messages=messages,
                    transcript=transcript,
                    tools=tools,
                    passes=passes,
                    queue=queue,
                    tally=tally,
                    ctx_logger=ctx_logger,
                    stall_notice=stall_notice,
                    economy_mode=economy_mode,
                )
            )
            verifier_meta["repair_reverts_block"] = preflight_meta.pop(
                "repair_reverts_block", 0
            )
            reroute_meta.update(preflight_meta)
            deliberate_meta.update(rerouted_meta)
            if any(preflight_meta.values()):
                live_quorum_committed = False
        final_action, nav_detail_meta = self._apply_nav_detailed_read(
            final_action,
            ctx_logger=ctx_logger,
        )
        post_substitution_meta = self._post_substitution_mutation_meta(
            context_id=context_id,
            action=final_action,
            ctx_logger=ctx_logger,
        )
        mandated_ask_meta = _mandated_ask_completion_meta(
            action=final_action,
            messages=messages,
            tools=tools,
            completion_notices=completion_notice_fires,
            completion_respond_deferrals=completion_respond_deferrals,
            completion_pivotal_veto=False,
        )
        scope_shadow_meta = (
            _scope_shadow_meta(final_action, messages)
            if self.config.scope_shadow
            else {
                "scope_shadow_flags": 0,
                "scope_shadow_details": [],
                "scope_shadow_inverse_flags": 0,
                "scope_shadow_inverse_details": [],
            }
        )
        reroute_meta["nav_detailed_rewrites"] += nav_detail_meta["nav_detailed_rewrites"]
        self._account_spiral_tokens(spiral_state, tally)
        spiral_meta = self._spiral_meta(spiral_state)

        decision_meta = {
            "route": route,
            "top_share": cluster_result.top_share,
            "entropy": cluster_result.entropy,
            "axis": cluster_result.axis,
            "unresolved": len(queue),
            "voi_route": deliberate_meta.get("voi_route"),
            "disputed_aspect": deliberate_meta.get("disputed_aspect"),
            "read_resolvable": deliberate_meta.get("read_resolvable"),
            "evpi": deliberate_meta.get("evpi"),
            "redundancy_cost": deliberate_meta.get("redundancy_cost"),
            **deliberate_meta,
            **reroute_meta,
            **verifier_meta,
            **grounded_retry_meta,
            **loop_breaker_meta,
            **announced_meta,
            **post_substitution_meta,
            **mandated_ask_meta,
            **output_guard_meta,
            **scope_shadow_meta,
            **spiral_meta,
            "repeat_nudge_injected": repeat_notice is not None,
            "repeat_nudge_capped": repeat_capped,
            "stall_count": stall_count,
            "stall_max_count": stall_state.max_stall_count,
            "stall_nudge_injected": stall_notice is not None,
            "stall_nudges_injected": stall_state.nudges_injected,
            "stall_hard_stop": stall_hard_stop,
            "stall_hard_stops": stall_state.hard_stops,
            "sequential_calls": tally.sequential_calls,
            "total_calls": tally.total_calls,
            "seq_budget_skips": tally.seq_budget_skips,
            "completion_notices": completion_notice_fires,
            "completion_respond_deferrals": completion_respond_deferrals,
            **completion_negation_meta,
            **_policy_rag_meta(tally),
            "call_comp_scatter": len(passes),
            "call_comp_candidate_review": int(
                deliberate_meta.get("candidate_reviewed", False)
            ),
            "call_comp_sharpen": int(deliberate_meta.get("sharpen_iters") or 0),
            "call_comp_verifier": int(verifier_meta.get("verifier_ensemble", False)),
            "call_comp_total_llm": tally.total_calls,
            "action": final_action.get("action"),
        }
        if scatter_meta.get("quorum3_unanimous"):
            scatter_meta["quorum3_matches_final"] = (
                scatter_meta.get("quorum3_action_key") == _action_key(final_action)
            )
        scatter_meta["straggler_tokens_deferred"] = straggler_tokens_deferred
        decision_meta.update(scatter_meta)
        self.last_decision = decision_meta
        log_meta = dict(deliberate_meta)
        for key in (
            "voi_route",
            "disputed_aspect",
            "read_resolvable",
            "evpi",
            "redundancy_cost",
        ):
            log_meta.setdefault(key, decision_meta.get(key))
        ctx_logger.info(
            "Consensus-routed decision",
            route=route,
            top_share=round(cluster_result.top_share, 3),
            entropy=round(cluster_result.entropy, 3),
            axis=cluster_result.axis,
            clusters=[len(cluster) for cluster in cluster_result.clusters],
            unresolved=len(queue),
            **log_meta,
            **scatter_meta,
            **reroute_meta,
            **verifier_meta,
            **grounded_retry_meta,
            **loop_breaker_meta,
            **announced_meta,
            **post_substitution_meta,
            **mandated_ask_meta,
            **output_guard_meta,
            **scope_shadow_meta,
            **spiral_meta,
            repeat_nudge_injected=repeat_notice is not None,
            repeat_nudge_capped=repeat_capped,
            stall_count=stall_count,
            stall_max_count=stall_state.max_stall_count,
            stall_nudge_injected=stall_notice is not None,
            stall_nudges_injected=stall_state.nudges_injected,
            stall_hard_stop=stall_hard_stop,
            stall_hard_stops=stall_state.hard_stops,
            sequential_calls=tally.sequential_calls,
            total_calls=tally.total_calls,
            call_comp_scatter=len(passes),
            call_comp_candidate_review=int(
                deliberate_meta.get("candidate_reviewed", False)
            ),
            call_comp_sharpen=int(deliberate_meta.get("sharpen_iters") or 0),
            call_comp_verifier=int(verifier_meta.get("verifier_ensemble", False)),
            call_comp_total_llm=tally.total_calls,
            seq_budget_skips=tally.seq_budget_skips,
            completion_notices=completion_notice_fires,
            completion_respond_deferrals=completion_respond_deferrals,
            **completion_negation_meta,
            **_policy_rag_meta(tally),
            action=final_action.get("action"),
        )

        if tally.sequential_calls > SEQUENTIAL_LLM_BUDGET:
            if _env_bool("TRACK2_ASSERT_SEQ_BUDGET", False):
                raise AssertionError(
                    f"sequential LLM budget exceeded: {tally.sequential_calls}"
                )
            ctx_logger.warning(
                "Consensus planner exceeded sequential budget",
                sequential_calls=tally.sequential_calls,
            )

        return self._result(final_action, tally)

    def _loop_breaker_confirmed_action(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        state = self._loop_breaker_state_by_context.setdefault(
            context_id, LoopBreakerState()
        )
        if state.pending_action is None:
            return None
        if not messages or messages[-1].get("role") != "user":
            return None
        if not _is_affirmation_message(_message_text(messages[-1])):
            return None
        action = state.pending_action
        state.pending_action = None
        available = {str(tool.get("function", {}).get("name") or "") for tool in tools}
        if available:
            for call in action.get("tool_calls") or []:
                if str(call.get("tool_name") or "") not in available:
                    return None
        return json.loads(json.dumps(action))

    def _apply_loop_breaker(
        self,
        *,
        context_id: str,
        final_action: dict[str, Any],
        passes: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self._loop_breaker_state_by_context.setdefault(
            context_id, LoopBreakerState()
        )
        transformed, meta = _loop_breaker_transform(
            final_action=final_action,
            passes=passes,
            messages=messages,
            tools=tools,
        )
        if transformed is None:
            return final_action, meta
        if state.fired:
            state.suppressed_repeats += 1
            meta["loop_breaker_fires"] = 0
            meta["loop_breaker_suppressed_repeats"] = state.suppressed_repeats
            ctx_logger.info(
                "Loop breaker suppressed repeat",
                loop_breaker_suppressed_repeats=state.suppressed_repeats,
                loop_breaker_kind=meta.get("loop_breaker_kind"),
            )
            return final_action, meta
        state.fired = True
        state.fired_message_index = meta.get("loop_breaker_message_index")
        state.pending_action = transformed.pop("_pending_confirmed_action", None)
        meta["loop_breaker_fires"] = 1
        ctx_logger.info(
            "Loop breaker fired",
            loop_breaker_kind=meta.get("loop_breaker_kind"),
            loop_breaker_message_index=meta.get("loop_breaker_message_index"),
            loop_breaker_pending_action=bool(state.pending_action),
        )
        return transformed, meta

    def _repeat_notice_for_context(
        self,
        context_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, bool]:
        notice = _repeat_nudge_notice(messages)
        if notice is None:
            return None, False
        state = self._repeat_state_by_context.setdefault(context_id, RepeatNudgeState())
        state.nudges_injected += 1
        if state.nudges_injected <= 3:
            return notice, False
        repeated_tool = _repeated_same_read_result_tool(messages, threshold=3)
        if repeated_tool is None:
            return notice, False
        state.capped += 1
        return REPEAT_CAP_NOTICE.format(tool_name=repeated_tool), True

    def _update_spiral_turns(
        self,
        context_id: str,
        messages: list[dict[str, Any]],
    ) -> SpiralCapState:
        state = self._spiral_state_by_context.setdefault(context_id, SpiralCapState())
        state.assistant_turns = max(state.assistant_turns, _assistant_turn_count(messages))
        self._maybe_engage_spiral_cap(state)
        return state

    def _account_spiral_tokens(
        self,
        state: SpiralCapState,
        tally: _CallTally,
    ) -> None:
        if tally.token_usage is not None:
            state.billed_tokens += _billed_tokens(tally.token_usage)
        self._maybe_engage_spiral_cap(state)

    def _maybe_engage_spiral_cap(self, state: SpiralCapState) -> None:
        if not self.config.spiral_cap or state.engaged:
            return
        token_cap_reached = (
            state.billed_tokens >= self.config.spiral_cap_tokens
            and state.assistant_turns > max(
                self.config.spiral_cap_token_min_turns,
                self.config.spiral_cap_turns,
            )
        )
        turn_cap_reached = state.assistant_turns > self.config.spiral_cap_turns
        if (
            token_cap_reached
            or turn_cap_reached
        ):
            state.engaged = True
            state.engaged_tokens = state.billed_tokens
            state.engaged_turns = state.assistant_turns

    def _spiral_cap_active(self, state: SpiralCapState) -> bool:
        return bool(self.config.spiral_cap and state.engaged)

    def _spiral_meta(self, state: SpiralCapState) -> dict[str, Any]:
        return {
            "spiral_cap_enabled": self.config.spiral_cap,
            "spiral_cap_engaged": self._spiral_cap_active(state),
            "spiral_cap_episode_tokens": state.billed_tokens,
            "spiral_cap_episode_turns": state.assistant_turns,
            "spiral_cap_engaged_tokens": (
                state.engaged_tokens if state.engaged else None
            ),
            "spiral_cap_engaged_turns": state.engaged_turns if state.engaged else None,
        }

    def _drain_pending_stragglers(self, tally: _CallTally) -> int:
        with self._pending_straggler_lock:
            pending = list(self._pending_straggler_usage)
            self._pending_straggler_usage.clear()
        for result in pending:
            tally.add(result, sequential=False)
        return len(pending)

    def _update_stall_state(
        self,
        context_id: str,
        messages: list[dict[str, Any]],
    ) -> StallState:
        state = self._stall_state_by_context.setdefault(context_id, StallState())
        tool_results = _tool_messages_by_call_id(messages)
        start = min(state.processed_messages, len(messages))
        for index in range(start, len(messages)):
            message = messages[index]
            if message.get("role") != "assistant":
                continue
            calls = _assistant_message_tool_calls(message)
            made_progress = False
            for call in calls:
                call_id = call.get("id")
                result = tool_results.get(str(call_id)) if call_id is not None else None
                if result is None or not _tool_result_progress(result):
                    continue
                call_site = _tool_call_site_key(call)
                if call_site in state.successful_call_sites:
                    continue
                state.successful_call_sites.add(call_site)
                made_progress = True
            if made_progress:
                state.stall_count = 0
            else:
                state.stall_count += 1
                state.max_stall_count = max(state.max_stall_count, state.stall_count)
        state.processed_messages = len(messages)
        return state

    def _grounded_retry_correction(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        meta: dict[str, Any] = {
            "grounded_retry_enabled": self.config.enable_grounded_retry,
            "grounded_retry_actionable": False,
            "grounded_retry_triggered": False,
            "grounded_retry_skipped": None,
            "grounded_retry_round": None,
            "grounded_retry_error_kind": None,
        }
        if not self.config.enable_grounded_retry:
            return None, meta

        state = self._grounded_retry_state_by_context.setdefault(
            context_id,
            GroundedRetryState(),
        )
        signal = _latest_grounded_retry_signal(messages)
        if signal is None:
            state.awaiting_corrective_result = False
            return None, meta

        meta["grounded_retry_error_kind"] = signal.reason
        if signal.carveout:
            state.awaiting_corrective_result = False
            meta["grounded_retry_skipped"] = "carveout"
            ctx_logger.info(
                "Grounded retry skipped non-actionable tool failure",
                reason=signal.reason,
                tool_name=signal.tool_name,
            )
            return None, meta

        meta["grounded_retry_actionable"] = True
        if state.awaiting_corrective_result:
            state.awaiting_corrective_result = False
            meta["grounded_retry_skipped"] = "corrected_call_failed"
            ctx_logger.info(
                "Grounded retry not repeated after corrected call failed",
                tool_name=signal.tool_name,
            )
            return None, meta
        if state.total_rounds >= 2:
            meta["grounded_retry_skipped"] = "episode_cap"
            return None, meta
        if signal.call_site_key in state.corrected_call_sites:
            meta["grounded_retry_skipped"] = "call_site_cap"
            return None, meta

        state.total_rounds += 1
        state.corrected_call_sites.add(signal.call_site_key)
        state.awaiting_corrective_result = True
        correction = _grounded_retry_prompt_correction(signal)
        meta.update(
            {
                "grounded_retry_triggered": True,
                "grounded_retry_round": state.total_rounds,
            }
        )
        ctx_logger.info(
            "Grounded retry correction enabled for failed tool observation",
            round=state.total_rounds,
            tool_name=signal.tool_name,
            reason=signal.reason,
        )
        return correction, meta

    def _result(self, action: dict[str, Any], tally: _CallTally) -> AgentInferenceResult:
        return AgentInferenceResult(
            next_action=action,
            elapsed_ms=tally.duration_ms,
            token_usage=tally.token_usage,
            cost=tally.cost,
            internal_calls=max(tally.total_calls, 1),
            quota_wait_ms=tally.quota_wait_ms,
        )

    def _policy_corpus_for_context(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> PolicyCorpus:
        cached = self._policy_corpus_by_context.get(context_id)
        if cached is not None:
            return cached
        system_text = next(
            (
                _message_text(message)
                for message in messages
                if message.get("role") == "system"
            ),
            "",
        )
        corpus = build_policy_corpus(
            system_text,
            tool_names=_available_tool_names(tools),
            embedding_backend=(self.embedding_backend if self.config.policy_rag else None),
        )
        self._policy_corpus_by_context[context_id] = corpus
        return corpus

    def _policy_corpus_for_transcript(
        self, transcript: list[dict[str, Any]]
    ) -> PolicyCorpus:
        del transcript
        return getattr(self._policy_rag_local, "corpus", PolicyCorpus((), ()))

    def _policy_injection_block(
        self,
        *,
        corpus: PolicyCorpus | None,
        transcript: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
        call_count: int = 1,
    ) -> str:
        if self.config.policy_reminders:
            return POLICY_REMINDERS
        if not self.config.policy_rag:
            return ""
        active_corpus = corpus or PolicyCorpus((), ())
        latest_user = _latest_user_text(transcript)
        clauses = _nonpermissive_user_clauses(latest_user)
        subjects: set[str] = set()
        for clause in clauses:
            subjects.update(extract_policy_subjects(clause))
        available_tool_names = getattr(
            self._policy_rag_local, "available_tool_names", frozenset()
        )
        candidate_tool_names = tuple(
            sorted(
                _policy_action_tool_names(
                    actions,
                    available_tool_names=available_tool_names,
                )
            )
        )
        route_options_present = bool(_route_disclosure_legs(transcript))
        decision_presents_or_selects_route = _decision_presents_or_selects_route(
            actions=actions,
            candidate_tool_names=candidate_tool_names,
            messages=transcript,
        )
        navigation_active = _navigation_active_without_delete(transcript)
        tier2_subjects = _tier2_decision_subjects(clauses, subjects)
        tier2_fired = _tier2_fired_levers(
            candidate_tool_names=candidate_tool_names,
            subjects=tier2_subjects,
            available_tool_names=available_tool_names,
            requires_confirmation_tool_names=getattr(
                self._policy_rag_local,
                "requires_confirmation_tool_names",
                frozenset(),
            ),
            policy_conditioned_confirmation_tool_names=getattr(
                self._policy_rag_local,
                "policy_conditioned_confirmation_tool_names",
                frozenset(),
            ),
            route_options_present=route_options_present,
            decision_presents_or_selects_route=decision_presents_or_selects_route,
            navigation_active=navigation_active,
        )
        tier1_text = _policy_inventory_lines(POLICY_TIER1_LINE_NUMBERS)
        tier2_text = _tier2_text(tier2_fired)
        agent_skills_tokens = policy_token_count(
            "\n\n".join(part for part in (tier1_text, tier2_text) if part)
        )
        pinned_sections = (
            _route_presentation_policy_sections(active_corpus)
            if "route_presentation_disclosure" in tier2_fired
            else ()
        )
        retrieval_corpus = _policy_corpus_with_pins(
            active_corpus,
            pinned_sections,
        )
        tier3_budget = max(
            retrieval_corpus.core_token_count,
            POLICY_INJECTION_TOKEN_BUDGET - agent_skills_tokens,
        )
        dense_query = "\n".join((latest_user, *candidate_tool_names))
        retrieval = retrieve_policy_sections(
            retrieval_corpus,
            policy_query_keys(
                tool_names=candidate_tool_names,
                subjects=subjects,
            ),
            max_tokens=tier3_budget,
            query_text=dense_query,
            embedding_backend=self.embedding_backend,
        )
        core_ids = set(active_corpus.always_on_ids)
        core_sections = tuple(
            section for section in retrieval.sections if section.section_id in core_ids
        )
        tail_sections = tuple(
            section for section in retrieval.sections if section.section_id not in core_ids
        )
        tier3_core_tokens = policy_token_count(
            "\n\n".join(section.text for section in core_sections)
        )
        tier3_tail_tokens = policy_token_count(
            "\n\n".join(section.text for section in tail_sections)
        )
        injection = _assemble_policy_injection(
            tier1_text=tier1_text,
            tier2_text=tier2_text,
            tier2_fired=tier2_fired,
            tier3_text=retrieval.text,
            tier3_core_tokens=tier3_core_tokens,
            tier3_tail_tokens=tier3_tail_tokens,
        )
        section_ids = list(retrieval.section_ids)
        tally.record_policy_rag(
            section_ids=section_ids,
            empty=retrieval.empty,
            tokens=retrieval.token_count,
            calls=call_count,
        )
        tally.record_policy_partition(
            tier1_tokens=injection.tier1_tokens,
            tier2_fired=list(injection.tier2_fired),
            tier2_tokens=injection.tier2_tokens,
            tier3_core_tokens=injection.tier3_core_tokens,
            tier3_tail_tokens=injection.tier3_tail_tokens,
            injected_tokens_total=injection.injected_tokens_total,
            calls=call_count,
        )
        ctx_logger.info(
            "Policy RAG retrieval",
            section_ids=section_ids,
            empty=retrieval.empty,
            injected_tokens=retrieval.token_count,
            tier1_tokens=injection.tier1_tokens,
            tier2_fired=list(injection.tier2_fired),
            tier2_tokens=injection.tier2_tokens,
            tier3_core_tokens=injection.tier3_core_tokens,
            tier3_tail_tokens=injection.tier3_tail_tokens,
            injected_tokens_total=injection.injected_tokens_total,
            calls=call_count,
        )
        return injection.text

    def _empty_preflight_meta(self) -> dict[str, int]:
        return {
            "preflight_nav_active_blocks": 0,
            "preflight_navread_blocks": 0,
            "preflight_ungrounded_blocks": 0,
            "preflight_segment_mismatch_blocks": 0,
            "preflight_corrective_reads": 0,
            "preflight_corrective_read_suppressed": 0,
            "preflight_segment_mismatch_regressions": 0,
            "terminal_honesty_escalations": 0,
            "route_substitution_hints": 0,
            "route_arg_substitutions": 0,
            "preflight_capped_template_blocks": 0,
            "preflight_type_blocks": 0,
            "preflight_name_guess_blocks": 0,
        }

    def _apply_announced_call_execution(
        self,
        *,
        context_id: str,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        meta = {"announced_call_executions": 0}
        if not self.config.act_on_approval:
            return action, meta
        state = self._announced_call_state_by_context.setdefault(
            context_id, AnnouncedCallState()
        )
        transformed, announcement = _announced_call_execution(
            action=action,
            messages=messages,
            tools=tools,
            state=state,
        )
        if transformed is None or announcement is None:
            return action, meta
        state.executions += 1
        meta["announced_call_executions"] = 1
        for call in transformed.get("tool_calls") or []:
            self._guard_state_by_context.setdefault(
                context_id, GuardState()
            ).confirmed_call_sites.add(_tool_call_sort_key(_normalized_tool_call(call)))
        ctx_logger.info(
            "Consensus announced call execution",
            announced_call_executions=state.executions,
            announcement=announcement,
            action=_compact_action_summary(transformed),
        )
        return transformed, meta

    def _post_substitution_mutation_meta(
        self,
        *,
        context_id: str,
        action: dict[str, Any],
        ctx_logger: Any,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "post_substitution_extra_nav_mutations": 0,
            "post_substitution_extra_nav_mutation_details": [],
        }
        state = self._preflight_state_by_context.get(context_id)
        if state is None or state.route_arg_substitutions < 1:
            return meta
        count, details = _post_substitution_extra_nav_mutations(state, action)
        if not count:
            return meta
        state.post_substitution_extra_nav_mutations += count
        meta["post_substitution_extra_nav_mutations"] = count
        meta["post_substitution_extra_nav_mutation_details"] = details
        ctx_logger.info(
            "Consensus post-substitution extra navigation mutation",
            post_substitution_extra_nav_mutations=count,
            post_substitution_extra_nav_mutations_episode=(
                state.post_substitution_extra_nav_mutations
            ),
            details=details,
        )
        return meta

    def _apply_nav_detailed_read(
        self,
        action: dict[str, Any],
        *,
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        if not self.config.nav_detailed_read:
            return action, {"nav_detailed_rewrites": 0}
        rewritten, count = _rewrite_nav_detailed_read(action)
        if count:
            ctx_logger.info(
                "Consensus nav detailed read rewrite",
                nav_detailed_rewrites=count,
                action=_compact_action_summary(rewritten),
            )
        return rewritten, {"nav_detailed_rewrites": count}

    def _final_preflight_or_respond(
        self,
        *,
        context_id: str,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        meta = self._empty_preflight_meta()
        if not self.config.nav_preflight:
            return action, meta
        issue = _nav_preflight_issue(action, messages)
        if issue is None:
            return action, meta
        meta[issue["counter"]] += 1
        state = self._preflight_state_by_context.setdefault(context_id, PreflightState())
        block_count = _record_preflight_block(state, issue)
        meta.update(_preflight_tripwire_meta(state, issue, block_count))
        substituted, fingerprint = (
            _approval_gated_route_substitution(
                action=action,
                issue=issue,
                messages=messages,
                block_count=block_count,
                state=state,
            )
            if self.config.act_on_approval
            else (None, None)
        )
        if substituted is not None and fingerprint is not None:
            state.route_arg_substitutions += 1
            state.substitution_tool_call_key = _route_replacement_call_key(substituted)
            state.substitution_emission_pending = True
            meta["route_arg_substitutions"] = 1
            ctx_logger.info(
                "Consensus approval-gated route substitution",
                route_arg_substitutions=state.route_arg_substitutions,
                block_count=block_count,
                fingerprint=fingerprint,
            )
            return substituted, meta
        meta["route_substitution_hints"] += _record_route_substitution_hint(
            state, issue, messages, capped=self.config.act_on_approval
        )
        terminal_action = _terminal_honesty_action(
            state,
            issue,
            messages,
            route_hint_cap_reached=(
                self.config.act_on_approval and state.route_substitution_hints >= 3
            ),
        )
        if terminal_action is not None:
            meta["terminal_honesty_escalations"] += 1
            ctx_logger.info(
                "Consensus terminal honesty escalation",
                signature=list(_preflight_signature(issue)),
                block_count=block_count,
                action=_compact_action_summary(terminal_action),
            )
            return terminal_action, meta
        response_count = _record_preflight_response(state, issue)
        if response_count > 2:
            meta["preflight_capped_template_blocks"] += 1
        ctx_logger.info(
            "Consensus final preflight blocked emission",
            counter=issue["counter"],
            notice=issue["notice"],
            block_count=block_count,
            response_count=response_count,
            action=_compact_action_summary(action),
        )
        return _preflight_block_response(
            issue["notice"],
            response_count=response_count,
            messages=messages,
            issue=issue,
        ), meta

    def _final_preflight_reroute_once(
        self,
        *,
        context_id: str,
        action: dict[str, Any],
        repair_fallback_action: dict[str, Any] | None = None,
        messages: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        queue: list[Proposition],
        tally: _CallTally,
        ctx_logger: Any,
        stall_notice: str | None,
        economy_mode: bool,
    ) -> tuple[dict[str, Any], list[Proposition], dict[str, int], dict[str, Any]]:
        meta = self._empty_preflight_meta()
        issue = _nav_preflight_issue(action, messages)
        if issue is None:
            return action, queue, meta, {}

        meta[issue["counter"]] += 1
        state = self._preflight_state_by_context.setdefault(context_id, PreflightState())
        block_count = _record_preflight_block(state, issue)
        meta.update(_preflight_tripwire_meta(state, issue, block_count))
        substituted, fingerprint = (
            _approval_gated_route_substitution(
                action=action,
                issue=issue,
                messages=messages,
                block_count=block_count,
                state=state,
            )
            if self.config.act_on_approval
            else (None, None)
        )
        if substituted is not None and fingerprint is not None:
            state.route_arg_substitutions += 1
            state.substitution_tool_call_key = _route_replacement_call_key(substituted)
            state.substitution_emission_pending = True
            meta["route_arg_substitutions"] = 1
            ctx_logger.info(
                "Consensus approval-gated route substitution",
                route_arg_substitutions=state.route_arg_substitutions,
                block_count=block_count,
                fingerprint=fingerprint,
            )
            return substituted, queue, meta, {}
        meta["route_substitution_hints"] += _record_route_substitution_hint(
            state, issue, messages, capped=self.config.act_on_approval
        )
        terminal_action = _terminal_honesty_action(
            state,
            issue,
            messages,
            route_hint_cap_reached=(
                self.config.act_on_approval and state.route_substitution_hints >= 3
            ),
        )
        if terminal_action is not None:
            meta["terminal_honesty_escalations"] += 1
            ctx_logger.info(
                "Consensus terminal honesty escalation",
                signature=list(_preflight_signature(issue)),
                block_count=block_count,
                action=_compact_action_summary(terminal_action),
            )
            return terminal_action, queue, meta, {}
        if repair_fallback_action is not None:
            meta["repair_reverts_block"] = 1
            ctx_logger.info(
                "Consensus final preflight reverted blocked repair",
                counter=issue["counter"],
                notice=issue["notice"],
                block_count=block_count,
                repaired_action=_compact_action_summary(action),
                reverted_action=_compact_action_summary(repair_fallback_action),
            )
            return repair_fallback_action, queue, meta, {}

        ctx_logger.info(
            "Consensus final preflight rerouting blocked action",
            counter=issue["counter"],
            notice=issue["notice"],
            block_count=block_count,
            action=_compact_action_summary(action),
        )
        if issue["counter"] == "preflight_navread_blocks":
            args = {"detailed_information": True} if self.config.nav_detailed_read else {}
            return (
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": "get_current_navigation_state",
                            "arguments": args,
                        }
                    ],
                },
                queue,
                meta,
                {},
            )
        if not tally.has_sequential_budget():
            tally.skip_sequential()
            ctx_logger.info(
                "Consensus final preflight skipped deliberate reroute for sequential budget",
                counter=issue["counter"],
                notice=issue["notice"],
                block_count=block_count,
                sequential_calls=tally.sequential_calls,
                seq_budget_skips=tally.seq_budget_skips,
                action=_compact_action_summary(action),
            )
            response_count = _record_preflight_response(state, issue)
            if response_count > 2:
                meta["preflight_capped_template_blocks"] += 1
            return (
                _preflight_block_response(
                    issue["notice"],
                    response_count=response_count,
                    messages=messages,
                    issue=issue,
                ),
                queue,
                meta,
                {},
            )
        rerouted_action, rerouted_queue, rerouted_meta = self._run_deliberate_pipeline(
            transcript=transcript,
            tools=tools,
            passes=passes,
            route="deliberate",
            plan_action=action,
            queue=queue,
            tally=tally,
            ctx_logger=ctx_logger,
            stall_notice=_append_notice(stall_notice, issue["notice"]),
            economy_mode=economy_mode,
        )
        second_issue = _nav_preflight_issue(rerouted_action, messages)
        if second_issue is None:
            return rerouted_action, rerouted_queue, meta, rerouted_meta

        meta[second_issue["counter"]] += 1
        second_block_count = _record_preflight_block(state, second_issue)
        meta.update(_preflight_tripwire_meta(state, second_issue, second_block_count))
        substituted, fingerprint = (
            _approval_gated_route_substitution(
                action=rerouted_action,
                issue=second_issue,
                messages=messages,
                block_count=second_block_count,
                state=state,
            )
            if self.config.act_on_approval
            else (None, None)
        )
        if substituted is not None and fingerprint is not None:
            state.route_arg_substitutions += 1
            state.substitution_tool_call_key = _route_replacement_call_key(substituted)
            state.substitution_emission_pending = True
            meta["route_arg_substitutions"] = 1
            ctx_logger.info(
                "Consensus approval-gated route substitution",
                route_arg_substitutions=state.route_arg_substitutions,
                block_count=second_block_count,
                fingerprint=fingerprint,
            )
            return substituted, rerouted_queue, meta, rerouted_meta
        meta["route_substitution_hints"] += _record_route_substitution_hint(
            state, second_issue, messages, capped=self.config.act_on_approval
        )
        terminal_action = _terminal_honesty_action(
            state,
            second_issue,
            messages,
            route_hint_cap_reached=(
                self.config.act_on_approval and state.route_substitution_hints >= 3
            ),
        )
        if terminal_action is not None:
            meta["terminal_honesty_escalations"] += 1
            ctx_logger.info(
                "Consensus terminal honesty escalation",
                signature=list(_preflight_signature(second_issue)),
                block_count=second_block_count,
                action=_compact_action_summary(terminal_action),
            )
            return terminal_action, rerouted_queue, meta, rerouted_meta
        corrective_read = _corrective_read_for_repeated_segment_mismatch(
            second_issue,
            block_count=second_block_count,
            state=state,
            messages=messages,
        )
        if corrective_read is not None:
            meta["preflight_corrective_reads"] += 1
            state.corrective_reads += 1
            ctx_logger.info(
                "Consensus final preflight corrective read",
                counter=second_issue["counter"],
                notice=second_issue["notice"],
                block_count=second_block_count,
                action=_compact_action_summary(rerouted_action),
                corrective_action=_compact_action_summary(corrective_read),
            )
            return corrective_read, rerouted_queue, meta, rerouted_meta
        if (
            second_issue.get("counter") == "preflight_segment_mismatch_blocks"
            and state.corrective_reads_suppressed
        ):
            meta["preflight_corrective_read_suppressed"] += 1
        response_count = _record_preflight_response(state, second_issue)
        if response_count > 2:
            meta["preflight_capped_template_blocks"] += 1
        ctx_logger.info(
            "Consensus final preflight deterministic stop",
            counter=second_issue["counter"],
            notice=second_issue["notice"],
            block_count=second_block_count,
            response_count=response_count,
            action=_compact_action_summary(rerouted_action),
        )
        return (
            _preflight_block_response(
                second_issue["notice"],
                response_count=response_count,
                messages=messages,
                issue=second_issue,
            ),
            rerouted_queue,
            meta,
            rerouted_meta,
        )

    def _scatter(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
        grounded_retry_correction: dict[str, Any] | None = None,
        stall_notice: str | None = None,
        repeat_notice: str | None = None,
        economy_mode: bool = False,
    ) -> list[dict[str, Any]]:
        width = 1 if economy_mode else max(1, self.config.scatter_width)
        live_quorum = 0
        if width >= 3 and not economy_mode:
            live_quorum = 3 if self.config.quorum_live else 0
            if live_quorum == 0 and self.config.scatter_quorum > 0:
                live_quorum = self.config.scatter_quorum
        initial_width = min(width, live_quorum) if live_quorum else width
        policy_block = self._policy_injection_block(
            corpus=self._policy_corpus_for_transcript(transcript),
            transcript=transcript,
            actions=[],
            tally=tally,
            ctx_logger=ctx_logger,
            call_count=initial_width,
        )
        prompt = _scatter_prompt_with_notices(
            _scatter_prompt(transcript, tools),
            grounded_retry_correction,
            stall_notice,
            repeat_notice,
        )
        results: list[dict[str, Any]] = []
        drops: dict[str, int] = {}
        arrival_start = time.perf_counter()
        arrival_rows: list[dict[str, Any]] = []
        scatter_meta: dict[str, Any] = {
            "quorum_shadow": width >= 3,
            "quorum_live": self.config.quorum_live or self.config.scatter_quorum > 0,
            "spiral_cap_economy": economy_mode,
            "quorum_commit": False,
            "quorum_live_commits": 0,
            "quorum_width": None,
            "quorum_action_mode": None,
            "quorum3_unanimous": False,
            "quorum3_action_key": None,
            "quorum3_matches_final": None,
            "latency_saved_ms": None,
            "tokens_saved_estimate": 0,
            "quorum_live_skipped_scatter_calls": 0,
            "quorum_live_skipped_verifier_calls": 0,
            "prefix_route_k3": None,
            "prefix_route_k4": None,
            "prefix_route_k5": None,
            "prefix_route_k6": None,
        }
        self._last_scatter_meta = scatter_meta

        def _one(idx: int) -> tuple[int, Any | None, float]:
            client = self._client(idx)
            messages = _scatter_messages_for_pass(
                prompt=prompt,
                pass_index=idx,
                diverse=self.config.scatter_diverse,
            )
            if policy_block:
                messages = _with_policy_block(messages, policy_block)
            temperature = _scatter_temperature_for_pass(
                pass_index=idx,
                width=width,
                diverse=self.config.scatter_diverse,
                uniform_temperature=self.config.scatter_temperature,
            )
            try:
                res = client.generate(
                    model=self.model,
                    messages=messages,
                    response_schema=SCATTER_PASS_SCHEMA,
                    response_schema_name="consensus_scatter_pass",
                    max_completion_tokens=self.config.scatter_max_completion_tokens,
                    temperature=temperature,
                    reasoning_effort=self.reasoning_effort,
                )
                return idx, res, (time.perf_counter() - arrival_start) * 1000.0
            except CerebrasTemplateError as exc:
                ctx_logger.warning("Consensus scatter call failed", idx=idx, error=str(exc))
                return idx, None, (time.perf_counter() - arrival_start) * 1000.0

        pool = ThreadPoolExecutor(max_workers=self._pool_size(width))

        try:
            self._consume_scatter_batch(
                [pool.submit(_one, idx) for idx in range(initial_width)],
                tally=tally,
                drops=drops,
                arrival_rows=arrival_rows,
                results=results,
                ctx_logger=ctx_logger,
            )
            if live_quorum and len(arrival_rows) >= live_quorum:
                prefix_rows = arrival_rows[:live_quorum]
                prefix = [row["item"] for row in prefix_rows]
                prefix_cluster, prefix_route = scatter_cluster_drafts_route(
                    prefix,
                    commit_threshold=self.config.commit_threshold,
                    embedding_backend=self.embedding_backend,
                    embedding_cluster_threshold=self.config.embedding_cluster_threshold,
                    token_overlap_cluster_threshold=self.config.token_overlap_cluster_threshold,
                )
                representative = (
                    prefix_cluster.representatives[0]
                    if prefix_cluster.representatives
                    else {}
                )
                action_mode = _action_mode(representative)
                scatter_meta.update(
                    {
                        "prefix_route_k3": prefix_route if live_quorum == 3 else None,
                        "quorum_action_mode": action_mode,
                    }
                )
                if len(prefix_cluster.clusters) == 1 and action_mode != "question":
                    skipped_scatter = max(0, width - live_quorum)
                    skipped_verifier = int(self.config.enable_verifier_ensemble)
                    avg_tokens = _average_billed_tokens(
                        row.get("result") for row in prefix_rows
                    )
                    scatter_meta.update(
                        {
                            "quorum_commit": True,
                            "quorum_live_commits": 1,
                            "quorum_width": live_quorum,
                            "quorum3_unanimous": live_quorum == 3,
                            "quorum3_action_key": (
                                _action_key(representative)
                                if live_quorum == 3
                                else None
                            ),
                            "quorum_live_skipped_scatter_calls": skipped_scatter,
                            "quorum_live_skipped_verifier_calls": skipped_verifier,
                            "tokens_saved_estimate": int(
                                avg_tokens * (skipped_scatter + skipped_verifier)
                            ),
                        }
                    )
                    ordered_prefix = [
                        row["item"] for row in sorted(prefix_rows, key=lambda item: item["idx"])
                    ]
                    self._last_scatter_meta = scatter_meta
                    self._log_scatter_summary(
                        ctx_logger=ctx_logger,
                        width=width,
                        valid=len(ordered_prefix),
                        drops=drops,
                        scatter_meta=scatter_meta,
                    )
                    return ordered_prefix
            if initial_width < width:
                self._policy_injection_block(
                    corpus=self._policy_corpus_for_transcript(transcript),
                    transcript=transcript,
                    actions=[],
                    tally=tally,
                    ctx_logger=ctx_logger,
                    call_count=width - initial_width,
                )
                self._consume_scatter_batch(
                    [pool.submit(_one, idx) for idx in range(initial_width, width)],
                    tally=tally,
                    drops=drops,
                    arrival_rows=arrival_rows,
                    results=results,
                    ctx_logger=ctx_logger,
                )
        finally:
            pool.shutdown(wait=not scatter_meta.get("quorum_commit"))

        ordered_rows = sorted(arrival_rows, key=lambda row: row["idx"])
        results = [row["item"] for row in ordered_rows]
        if width >= 3 or self.config.scatter_quorum > 0:
            self._populate_quorum_shadow_meta(arrival_rows, scatter_meta)
        valid = len(results)
        self._last_scatter_meta = scatter_meta
        self._log_scatter_summary(
            ctx_logger=ctx_logger,
            width=width,
            valid=valid,
            drops=drops,
            scatter_meta=scatter_meta,
        )
        return results

    def _consume_scatter_batch(
        self,
        futures: list[Future],
        *,
        tally: _CallTally,
        drops: dict[str, int],
        arrival_rows: list[dict[str, Any]],
        results: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> None:
        for future in as_completed(futures):
            idx, res, arrival_ms = future.result()
            if res is None:
                drops["call_error"] = drops.get("call_error", 0) + 1
                continue
            tally.add(res, sequential=False)
            if not (res.text or "").strip():
                reason = (
                    "truncated_length"
                    if res.finish_reason == "length"
                    else f"empty_{res.finish_reason or 'unknown'}"
                )
                drops[reason] = drops.get(reason, 0) + 1
                continue
            try:
                parsed = _parse_scatter_pass(res.text)
                row = {
                    "idx": idx,
                    "item": parsed,
                    "arrival_ms": arrival_ms,
                    "result": res,
                }
                arrival_rows.append(row)
                results.append(parsed)
            except (
                MalformedModelResponseError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                drops["malformed"] = drops.get("malformed", 0) + 1
                ctx_logger.warning("Malformed consensus scatter pass", error=str(exc))

    def _log_scatter_summary(
        self,
        *,
        ctx_logger: Any,
        width: int,
        valid: int,
        drops: dict[str, int],
        scatter_meta: dict[str, Any],
    ) -> None:
        reason_str = ",".join(f"{k}={v}" for k, v in drops.items()) or "none"
        log = ctx_logger.warning if valid < (width + 1) // 2 else ctx_logger.info
        log(
            f"Consensus scatter passes valid={valid}/{width} "
            f"dropped={width - valid} reasons={reason_str}",
            width=width,
            valid=valid,
            dropped=width - valid,
            drop_reasons=drops or None,
            **scatter_meta,
        )

    def _populate_quorum_shadow_meta(
        self,
        arrival_rows: list[dict[str, Any]],
        scatter_meta: dict[str, Any],
    ) -> None:
        if not arrival_rows:
            return
        arrival_order = list(arrival_rows)
        last_arrival = max(float(row["arrival_ms"]) for row in arrival_order)
        for k in (3, 4, 5, 6):
            if len(arrival_order) < k:
                continue
            prefix = [row["item"] for row in arrival_order[:k]]
            cluster_result, prefix_route = scatter_cluster_drafts_route(
                prefix,
                commit_threshold=self.config.commit_threshold,
                embedding_backend=self.embedding_backend,
                embedding_cluster_threshold=self.config.embedding_cluster_threshold,
                token_overlap_cluster_threshold=self.config.token_overlap_cluster_threshold,
            )
            scatter_meta[f"prefix_route_k{k}"] = prefix_route
            if k == 3 and len(cluster_result.clusters) == 1:
                representative = cluster_result.representatives[0]
                scatter_meta["quorum3_unanimous"] = True
                scatter_meta["quorum3_action_key"] = _action_key(representative)
                scatter_meta["quorum_action_mode"] = _action_mode(representative)
                scatter_meta["latency_saved_ms"] = round(
                    last_arrival - float(arrival_order[2]["arrival_ms"]),
                    1,
                )

    def _aggregate(
        self, passes: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[Proposition]]:
        scored: dict[str, float] = {}
        rep: dict[str, dict[str, Any]] = {}
        for item in passes:
            action = item["action"]
            key = _action_key(action)
            scored[key] = scored.get(key, 0.0) + float(item.get("confidence", 0.5))
            rep.setdefault(key, action)
        best_key = max(scored, key=scored.get)
        plan_action = rep[best_key]

        props = {kind: Proposition(kind=kind) for kind in PROPOSITION_KINDS}
        for item in passes:
            for kind, disposition in (item.get("dispositions") or {}).items():
                if kind in props and disposition in DISPOSITIONS:
                    props[kind].record(disposition)
        queue = [
            prop
            for prop in props.values()
            if prop.total and prop.ok_ratio() < self.config.resolve_threshold
        ]
        return plan_action, queue

    def _run_deliberate_pipeline(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        route: str,
        plan_action: dict[str, Any],
        queue: list[Proposition],
        tally: _CallTally,
        ctx_logger: Any,
        stall_notice: str | None = None,
        economy_mode: bool = False,
    ) -> tuple[dict[str, Any], list[Proposition], dict[str, Any]]:
        policy_corpus = self._policy_corpus_for_transcript(transcript)
        candidate_reviewed = False
        review_skipped_readonly = False
        candidate_review = None
        if not economy_mode and _is_readonly_get_action(plan_action):
            review_skipped_readonly = True
            ctx_logger.info(
                "Consensus candidate verifier skipped for read-only action",
                route=route,
                action=plan_action.get("action"),
                tool_names=[
                    call.get("tool_name")
                    for call in plan_action.get("tool_calls", [])
                ],
            )
        elif not economy_mode and tally.has_sequential_budget():
            policy_block = self._policy_injection_block(
                corpus=policy_corpus,
                transcript=transcript,
                actions=[plan_action] + [item.get("action") or {} for item in passes],
                tally=tally,
                ctx_logger=ctx_logger,
            )
            candidate_review = self._candidate_review(
                transcript=transcript,
                tools=tools,
                passes=passes,
                aggregate_action=plan_action,
                aggregate_queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
                policy_block=policy_block,
            )
        elif not economy_mode:
            tally.skip_sequential()
            ctx_logger.info(
                "Consensus candidate verifier skipped for sequential budget",
                route=route,
                sequential_calls=tally.sequential_calls,
                seq_budget_skips=tally.seq_budget_skips,
            )
        if candidate_review is not None:
            candidate_reviewed = True
            plan_action = candidate_review["action"]
            queue = candidate_review["queue"]
            ctx_logger.info(
                "Consensus candidate verifier selected action",
                route=route,
                selected_candidate_index=candidate_review.get("selected_index"),
                approved_candidate=candidate_review.get("approved_candidate"),
                issue_category=candidate_review.get("issue_category"),
                unresolved=len(queue),
                action=plan_action.get("action"),
            )

        rescatters = 0
        iters = 0
        while queue and not economy_mode and iters < self.config.max_sharpen_iters:
            if not tally.has_sequential_budget():
                tally.skip_sequential()
                ctx_logger.info(
                    "Consensus sharpen skipped for sequential budget",
                    route=route,
                    sequential_calls=tally.sequential_calls,
                    seq_budget_skips=tally.seq_budget_skips,
                    unresolved=len(queue),
                )
                break
            iters += 1
            plan_action, queue, new_items = self._sharpen_iteration(
                transcript=transcript,
                tools=tools,
                plan_action=plan_action,
                queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
                policy_corpus=policy_corpus,
                stall_notice=stall_notice,
            )
            if new_items and rescatters < self.config.max_rescatters:
                rescatters += 1
                queue.extend(new_items)

        final_action = self._defer_action(queue, plan_action) if queue else plan_action
        return final_action, queue, {
            "branch": "deliberate",
            "candidate_reviewed": candidate_reviewed,
            "review_skipped_readonly": int(review_skipped_readonly),
            "sharpen_iters": iters,
            "rescatters": rescatters,
            "final_reviewed": False,
        }

    def _run_voi_branch(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        cluster_result: ClusterResult,
        tally: _CallTally,
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidates = _structured_voi_candidates(passes)
        aspect = _locate_disputed_aspect(candidates, cluster_result, transcript)
        read_action = _read_action_for_disputed_aspect(
            aspect=aspect,
            passes=passes,
            tools=tools,
            transcript=transcript,
        )
        read_resolvable = read_action is not None
        evpi = _voi_evpi_for_aspect(
            aspect=aspect,
            passes=passes,
            cluster_result=cluster_result,
        )
        redundancy_cost = self.config.voi_redundancy_lambda * _aspect_question_count(
            transcript, aspect
        )

        if cluster_result.top_share >= self.config.commit_threshold:
            action = _select_committed_action(
                cluster_result,
                deterministic=self.config.deterministic_commit,
                route="clarify",
                ctx_logger=ctx_logger,
            )
            voi_route = "act"
            branch = "voi_act"
        elif read_action is not None:
            action = read_action
            voi_route = "read"
            branch = "voi_read"
        elif evpi - redundancy_cost >= self.config.voi_evpi_alpha:
            action = self._run_voi_question_call(
                transcript=transcript,
                tools=tools,
                passes=passes,
                cluster_result=cluster_result,
                aspect=aspect,
                tally=tally,
                ctx_logger=ctx_logger,
            )
            voi_route = "ask"
            branch = "voi_ask"
        else:
            action = _select_best_act_candidate(
                passes,
                cluster_result,
                deterministic=self.config.deterministic_commit,
                route="clarify",
                ctx_logger=ctx_logger,
            )
            voi_route = "act"
            branch = "voi_act"

        meta = _branch_meta(branch, branch_llm=voi_route == "ask")
        meta.update(
            {
                "voi_route": voi_route,
                "disputed_aspect": aspect.name,
                "read_resolvable": read_resolvable,
                "evpi": round(evpi, 3),
                "redundancy_cost": round(redundancy_cost, 3),
            }
        )
        ctx_logger.info(
            "Consensus VOI branch decision",
            voi_route=voi_route,
            disputed_aspect=aspect.name,
            read_resolvable=read_resolvable,
            evpi=round(evpi, 3),
            redundancy_cost=round(redundancy_cost, 3),
            action=action.get("action"),
        )
        return action, meta

    def _run_voi_question_call(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        cluster_result: ClusterResult,
        aspect: VoiAspect,
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any]:
        prompt = _voi_question_prompt(
            transcript=transcript,
            tools=tools,
            passes=passes,
            cluster_result=cluster_result,
            aspect=aspect,
        )
        fallback = _deterministic_voi_question_action(aspect)
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": VOI_CLARIFY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=CLARIFY_SCHEMA,
                response_schema_name="voi_clarify_question",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=(
                    0 if self.config.deterministic_commit else self.config.sharpen_temperature
                ),
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Consensus VOI question failed", error=str(exc))
            return fallback

        tally.add(res, sequential=True)
        try:
            action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning("Malformed VOI question output", error=str(exc))
            return fallback

        if action.get("action") != "respond":
            return fallback
        return _ground_voi_question(action, cluster_result, transcript, tools, aspect)

    def _run_clarify_branch(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        cluster_result: ClusterResult,
        tally: _CallTally,
        ctx_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = _clarify_prompt(
            transcript=transcript,
            tools=tools,
            passes=passes,
            cluster_result=cluster_result,
        )
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLARIFY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=CLARIFY_SCHEMA,
                response_schema_name="clarify_question",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Consensus clarify branch failed", error=str(exc))
            return _deterministic_clarify_action(
                cluster_result,
                transcript=transcript,
                tools=tools,
            ), _branch_meta(
                "clarify",
                branch_llm=False,
            )

        tally.add(res, sequential=True)
        try:
            action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning("Malformed clarify branch output", error=str(exc))
            action = _deterministic_clarify_action(
                cluster_result,
                transcript=transcript,
                tools=tools,
            )

        if action.get("action") != "respond":
            action = _deterministic_clarify_action(
                cluster_result,
                transcript=transcript,
                tools=tools,
            )
        action = _ground_clarify_question(action, cluster_result, transcript, tools)
        return action, _branch_meta("clarify", branch_llm=True)

    def _run_verifier_ensemble(
        self,
        *,
        plan_action: dict[str, Any],
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> VerifierEnsembleResult:
        votes = _deterministic_verifier_votes(
            plan_action=plan_action,
            transcript=transcript,
            tools=tools,
            passes=passes,
        )
        votes.extend(
            self._prompted_verifier_votes(
                plan_action=plan_action,
                transcript=transcript,
                tools=tools,
                passes=passes,
                tally=tally,
                ctx_logger=ctx_logger,
            )
        )
        result = _aggregate_verifier_votes(
            plan_action=plan_action,
            votes=votes,
            threshold=self.config.verifier_veto_threshold,
            weights=self.config.verifier_weights,
        )
        ctx_logger.info(
            "Consensus verifier ensemble decision",
            verifier_score=round(result.score, 3),
            verifier_decision=result.decision,
            verifier_vetoed=result.vetoed,
            verifier_repaired=result.repaired,
            votes=[
                {
                    "name": vote.name,
                    "score": round(vote.score, 3),
                    "veto": vote.veto,
                }
                for vote in votes
            ],
            action=result.action.get("action"),
        )
        return result

    def _run_deterministic_verifier_ensemble(
        self,
        *,
        plan_action: dict[str, Any],
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> VerifierEnsembleResult:
        votes = _deterministic_verifier_votes(
            plan_action=plan_action,
            transcript=transcript,
            tools=tools,
            passes=passes,
        )
        result = _aggregate_verifier_votes(
            plan_action=plan_action,
            votes=votes,
            threshold=self.config.verifier_veto_threshold,
            weights=self.config.verifier_weights,
        )
        ctx_logger.info(
            "Consensus deterministic verifier decision",
            verifier_score=round(result.score, 3),
            verifier_decision=result.decision,
            verifier_vetoed=result.vetoed,
            verifier_repaired=result.repaired,
            votes=[
                {
                    "name": vote.name,
                    "score": round(vote.score, 3),
                    "veto": vote.veto,
                }
                for vote in votes
            ],
            action=result.action.get("action"),
        )
        return result

    def _prompted_verifier_votes(
        self,
        *,
        plan_action: dict[str, Any],
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> list[VerifierVote]:
        if not tally.has_sequential_budget():
            tally.skip_sequential()
            ctx_logger.info(
                "Prompted verifier ensemble skipped for sequential budget",
                sequential_calls=tally.sequential_calls,
                seq_budget_skips=tally.seq_budget_skips,
            )
            return []
        prompt = _prompted_verifier_prompt(
            plan_action=plan_action,
            transcript=transcript,
            tools=tools,
            passes=passes,
        )
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=PROMPTED_VERIFIER_SCHEMA,
                response_schema_name="prompted_verifier_votes",
                max_completion_tokens=self.config.adversarial_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Prompted verifier ensemble failed", error=str(exc))
            return []

        tally.add(res, sequential=True)
        try:
            payload = json.loads(res.text)
        except json.JSONDecodeError as exc:
            ctx_logger.warning("Malformed prompted verifier output", error=str(exc))
            return []

        votes: list[VerifierVote] = []
        for name in PROMPTED_VERIFIERS:
            item = payload.get(name) or {}
            votes.append(
                VerifierVote(
                    name=name,
                    score=_clamp_score(item.get("score", 0.5)),
                    veto=False,
                    repair=None,
                    rationale=str(item.get("rationale", "")),
                )
            )
        return votes

    def _candidate_review(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        passes: list[dict[str, Any]],
        aggregate_action: dict[str, Any],
        aggregate_queue: list[Proposition],
        tally: _CallTally,
        ctx_logger: Any,
        policy_block: str = "",
    ) -> dict[str, Any] | None:
        candidates = _candidate_options(passes)
        if not candidates:
            return None
        prompt = _candidate_review_prompt(
            transcript=transcript,
            tools=tools,
            candidates=candidates,
            aggregate_action=aggregate_action,
            aggregate_queue=aggregate_queue,
        )
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=_with_policy_block([
                    {"role": "system", "content": CANDIDATE_REVIEW_SYSTEM},
                    {"role": "user", "content": prompt},
                ], policy_block),
                response_schema=CANDIDATE_REVIEW_SCHEMA,
                response_schema_name="candidate_review",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Consensus candidate verifier failed", error=str(exc))
            return None

        tally.add(res, sequential=True)
        try:
            payload = json.loads(res.text)
            action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning(
                "Malformed consensus candidate verifier output", error=str(exc)
            )
            return None

        unresolved = payload.get("unresolved_proposition_kinds", []) or []
        return {
            "action": action,
            "queue": _queue_from_proposition_kinds(unresolved),
            "selected_index": payload.get("selected_candidate_index"),
            "approved_candidate": bool(payload.get("approved_candidate")),
            "issue_category": payload.get("issue_category", "other"),
            "explanation": payload.get("explanation", ""),
        }

    def _sharpen_iteration(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        plan_action: dict[str, Any],
        queue: list[Proposition],
        tally: _CallTally,
        ctx_logger: Any,
        policy_corpus: PolicyCorpus | None = None,
        stall_notice: str | None = None,
    ) -> tuple[dict[str, Any], list[Proposition], list[Proposition]]:
        target = queue[0]
        refine_policy_block = self._policy_injection_block(
            corpus=policy_corpus,
            transcript=transcript,
            actions=[plan_action],
            tally=tally,
            ctx_logger=ctx_logger,
        )
        adversarial_policy_block = self._policy_injection_block(
            corpus=policy_corpus,
            transcript=transcript,
            actions=[plan_action],
            tally=tally,
            ctx_logger=ctx_logger,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            adv_future = pool.submit(
                self._adversarial,
                transcript,
                tools,
                plan_action,
                ctx_logger,
                adversarial_policy_block,
            )
            refine = self._refine(
                transcript,
                tools,
                plan_action,
                queue,
                target,
                ctx_logger,
                stall_notice,
                refine_policy_block,
            )
            adv = adv_future.result()

        if refine is not None:
            tally.add(refine["raw"], sequential=True)
        if adv is not None:
            tally.add(adv["raw"], sequential=False)

        new_items: list[Proposition] = []
        if refine is not None:
            plan_action = refine["action"]
            resolved_ids = set(refine.get("resolved", []))
            remaining = [prop for prop in queue if prop.kind not in resolved_ids]
        else:
            remaining = queue

        if adv is not None and adv.get("violation") and adv.get("severity") == "high":
            category = adv.get("category", "policy_compliance")
            kind = category if category in PROPOSITION_KINDS else "policy_compliance"
            reopened = Proposition(kind=kind)
            reopened.record("blocked")
            if not any(prop.kind == kind for prop in remaining):
                new_items.append(reopened)

        return plan_action, remaining, new_items

    def _refine(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        plan_action: dict[str, Any],
        queue: list[Proposition],
        target: Proposition,
        ctx_logger: Any,
        stall_notice: str | None = None,
        policy_block: str = "",
    ) -> dict[str, Any] | None:
        prompt = _prompt_with_stall_notice(
            _sharpen_prompt(transcript, tools, plan_action, queue, target),
            stall_notice,
        )
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=_with_policy_block([
                    {"role": "system", "content": SHARPEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ], policy_block),
                response_schema=SHARPEN_SCHEMA,
                response_schema_name="sharpen",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Consensus sharpen refine failed", error=str(exc))
            return None
        try:
            payload = json.loads(res.text)
            action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning("Malformed consensus sharpen output", error=str(exc))
            return {"raw": res, "action": plan_action, "resolved": []}
        return {
            "raw": res,
            "action": action,
            "resolved": payload.get("resolved_proposition_kinds", []) or [],
            "decision": payload.get("decision", "act"),
        }

    def _adversarial(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        plan_action: dict[str, Any],
        ctx_logger: Any,
        policy_block: str = "",
    ) -> dict[str, Any] | None:
        prompt = _adversarial_prompt(transcript, tools, plan_action)
        try:
            res = self._client(1).generate(
                model=self.model,
                messages=_with_policy_block([
                    {"role": "system", "content": ADVERSARIAL_SYSTEM},
                    {"role": "user", "content": prompt},
                ], policy_block),
                response_schema=ADVERSARIAL_SCHEMA,
                response_schema_name="adversarial",
                max_completion_tokens=self.config.adversarial_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Consensus adversarial pass failed", error=str(exc))
            return None
        try:
            payload = json.loads(res.text)
        except json.JSONDecodeError:
            return {"raw": res, "violation": False}
        return {
            "raw": res,
            "violation": bool(payload.get("violation_found")),
            "severity": payload.get("severity", "none"),
            "category": payload.get("category", "none"),
            "explanation": payload.get("explanation", ""),
        }

    def _defer_action(
        self, queue: list[Proposition], plan_action: dict[str, Any]
    ) -> dict[str, Any]:
        del plan_action
        kinds = {prop.kind for prop in queue}
        if kinds & {"feasibility", "tool_availability"}:
            content = (
                "I can't complete that with the controls available to me right "
                "now, so I don't want to guess. Could you tell me how you'd like "
                "to proceed?"
            )
        else:
            content = (
                "I need the missing tool argument before I can act on that."
            )
        return {"action": "respond", "content": content}

    def _single_action(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
        grounded_retry_correction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": "Choose exactly one next assistant action for this turn.",
            "available_tools": tools,
            "conversation_transcript": transcript,
            "rules": [
                "Use only the supplied tool definitions.",
                "Do not invent tool observations or unavailable capabilities.",
                "If a capability or parameter is unavailable, say so briefly.",
            ],
        }
        if grounded_retry_correction is not None:
            payload["grounded_retry_correction"] = grounded_retry_correction
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        policy_block = self._policy_injection_block(
            corpus=self._policy_corpus_for_transcript(transcript),
            transcript=transcript,
            actions=[],
            tally=tally,
            ctx_logger=ctx_logger,
        )
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _developer_instructions(policy_block),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_schema=NEXT_ACTION_OUTPUT_SCHEMA,
                response_schema_name="next_action",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=None,
                reasoning_effort=self.reasoning_effort,
            )
            tally.add(res, sequential=True)
            return parse_next_action(res.text)
        except (
            CerebrasTemplateError,
            MalformedModelResponseError,
            json.JSONDecodeError,
        ) as exc:
            ctx_logger.warning("Consensus single-action fallback failed", error=str(exc))
            return {
                "action": "respond",
                "content": (
                    "Sorry, I had trouble processing that. Could you say it again?"
                ),
            }

    def _pool_size(self, width: int) -> int:
        return max(1, min(width, self.config.max_parallel_clients))

    def _client(self, idx: int) -> CerebrasCompletionClient:
        size = max(1, self.config.max_parallel_clients)
        while len(self._client_pool) <= (idx % size):
            self._client_pool.append(
                CerebrasCompletionClient(
                    api_base=self.api_base,
                    service_tier=self.service_tier,
                    logger=(
                        self.logger.bind(context="cerebras")
                        if self.logger is not None
                        else None
                    ),
                )
            )
        return self._client_pool[idx % size]


def scatter_cluster_drafts_route(
    passes: list[dict[str, Any]],
    *,
    commit_threshold: float = 0.7,
    embedding_backend: EmbeddingBackend | None = None,
    embedding_cluster_threshold: float = ConsensusPlannerConfig.embedding_cluster_threshold,
    token_overlap_cluster_threshold: float = 0.5,
) -> tuple[ClusterResult, str]:
    cluster_result = _cluster_drafts(
        passes,
        commit_threshold=commit_threshold,
        embedding_backend=embedding_backend,
        embedding_cluster_threshold=embedding_cluster_threshold,
        token_overlap_cluster_threshold=token_overlap_cluster_threshold,
    )
    return cluster_result, _route(cluster_result)


def _cluster_drafts(
    passes: list[dict[str, Any]],
    *,
    commit_threshold: float,
    embedding_backend: EmbeddingBackend | None,
    embedding_cluster_threshold: float,
    token_overlap_cluster_threshold: float,
) -> ClusterResult:
    actions = [
        item["action"]
        for item in passes
        if isinstance(item, dict) and isinstance(item.get("action"), dict)
    ]
    if not actions:
        return ClusterResult(
            clusters=[],
            top_share=0.0,
            entropy=0.0,
            axis="none",
            representatives=[],
        )

    clusters: list[list[dict[str, Any]]] = []
    exact_index: dict[str, int] = {}
    response_items: list[tuple[int, dict[str, Any], str, str]] = []

    for action in actions:
        if action.get("action") == "tool_calls":
            key = _tool_action_key(action)
            idx = exact_index.get(key)
            if idx is None:
                idx = len(clusters)
                exact_index[key] = idx
                clusters.append([])
            clusters[idx].append(action)
            continue

        text = _response_text(action)
        response_items.append(
            (len(response_items), action, text, _response_kind(action))
        )

    response_vectors = _embed_response_texts(
        [item[2] for item in response_items],
        embedding_backend,
    )
    response_cluster_reps: list[tuple[int, int, str]] = []
    for item_pos, action, text, kind in response_items:
        matched_idx: int | None = None
        for cluster_idx, rep_pos, rep_kind in response_cluster_reps:
            if kind != rep_kind:
                continue
            if response_vectors is not None:
                score = _cosine(response_vectors[item_pos], response_vectors[rep_pos])
                if score >= embedding_cluster_threshold:
                    matched_idx = cluster_idx
                    break
            else:
                rep_text = response_items[rep_pos][2]
                if (
                    _jaccard_similarity(text, rep_text)
                    >= token_overlap_cluster_threshold
                ):
                    matched_idx = cluster_idx
                    break
        if matched_idx is None:
            matched_idx = len(clusters)
            clusters.append([])
            response_cluster_reps.append((matched_idx, item_pos, kind))
        clusters[matched_idx].append(action)

    clusters.sort(key=lambda cluster: len(cluster), reverse=True)
    representatives = [cluster[0] for cluster in clusters]
    top_share = len(clusters[0]) / len(actions) if clusters else 0.0
    entropy = _normalized_entropy([len(cluster) for cluster in clusters])
    axis = _axis_for_clusters(
        clusters,
        top_share=top_share,
        commit_threshold=commit_threshold,
    )
    return ClusterResult(
        clusters=clusters,
        top_share=top_share,
        entropy=entropy,
        axis=axis,
        representatives=representatives,
    )


def _route(cluster_result: ClusterResult) -> str:
    if cluster_result.axis == "none":
        return "commit"
    if cluster_result.axis == "intent":
        return "clarify"
    if cluster_result.axis == "claim":
        return "verify"
    return "deliberate"


def _axis_for_clusters(
    clusters: list[list[dict[str, Any]]],
    *,
    top_share: float,
    commit_threshold: float,
) -> str:
    if not clusters or top_share >= commit_threshold:
        return "none"

    reps = [cluster[0] for cluster in clusters]
    kinds = [_action_mode(rep) for rep in reps]
    has_question = "question" in kinds
    has_tool = "tool_calls" in kinds
    has_statement = "statement" in kinds
    if has_question and (has_tool or has_statement):
        return "intent"
    if has_tool and has_statement:
        return "intent"
    if has_question and not has_tool:
        return "intent"

    tool_reps = [rep for rep in reps if rep.get("action") == "tool_calls"]
    if len(tool_reps) == len(reps):
        signatures = {_tool_name_signature(rep) for rep in tool_reps}
        if len(signatures) == 1:
            return "claim"
        return "action"

    if all(kind == "statement" for kind in kinds):
        return "claim"
    return "action"


def _parse_scatter_pass(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    action = parse_next_action(text)
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    dispositions_raw = payload.get("dispositions") or {}
    dispositions = {
        kind: (
            dispositions_raw.get(kind)
            if dispositions_raw.get(kind) in DISPOSITIONS
            else "uncertain"
        )
        for kind in PROPOSITION_KINDS
    }
    recommendation = payload.get("recommendation")
    if recommendation not in RECOMMENDATIONS:
        recommendation = "act"
    return {
        "action": action,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "dispositions": dispositions,
        "recommendation": recommendation,
    }


def _developer_instructions(policy_block: str) -> str:
    if not policy_block:
        return CEREBRAS_DEVELOPER_INSTRUCTIONS
    return f"{CEREBRAS_DEVELOPER_INSTRUCTIONS}\n\n{policy_block}"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _with_policy_block(
    messages: list[dict[str, str]],
    policy_block: str,
) -> list[dict[str, str]]:
    updated = [dict(message) for message in messages]
    if not policy_block:
        return updated
    for message in updated:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content', '')}\n\n{policy_block}"
            break
    return updated


def _with_policy_reminders(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Compatibility wrapper for the static reminder A/B arm."""

    return _with_policy_block(messages, POLICY_REMINDERS)


def _policy_inventory_lines(line_numbers: tuple[int, ...]) -> str:
    return "\n".join(_POLICY_REMINDER_LINES[number - 1] for number in line_numbers)


def _tier2_text(fired_levers: tuple[str, ...]) -> str:
    line_numbers = {
        number
        for lever_id in fired_levers
        for number in POLICY_TIER2_LINE_NUMBERS[lever_id]
    }
    return _policy_inventory_lines(tuple(sorted(line_numbers)))


def _tier2_decision_subjects(
    clauses: list[str], policy_subjects: set[str]
) -> frozenset[str]:
    subjects = set(policy_subjects)
    if any(_TIER2_SCOPE_DEGREE_RE.search(clause) for clause in clauses):
        subjects.add("scope_degree")
    if any(_TIER2_DESTINATION_CHANGE_RE.search(clause) for clause in clauses):
        subjects.add("destination_change")
    if any(_TIER2_EXPLICIT_READ_RE.search(clause) for clause in clauses):
        subjects.add("explicit_read")
    return frozenset(subjects)


def _is_direct_car_state_setter(tool_name: str) -> bool:
    lowered = tool_name.casefold()
    if not lowered.startswith(
        ("set_", "open_", "close_", "open_close_", "turn_", "adjust_")
    ):
        return False
    if any(fragment in lowered for fragment in ("navigation", "route", "message")):
        return False
    return any(
        fragment in lowered
        for fragment in (
            "air_",
            "climate",
            "defrost",
            "door",
            "fan",
            "light",
            "mirror",
            "seat",
            "sunroof",
            "sunshade",
            "temperature",
            "window",
            "wiper",
        )
    )


def _tier2_is_read_tool(tool_name: str) -> bool:
    return tool_name.casefold().startswith(
        ("get_", "read_", "search_", "list_", "calculate_")
    )


def _decision_presents_or_selects_route(
    *,
    actions: list[dict[str, Any]],
    candidate_tool_names: tuple[str, ...],
    messages: list[dict[str, Any]],
) -> bool:
    if any(
        name.casefold() == "set_new_navigation"
        or name.casefold().startswith("navigation_")
        for name in candidate_tool_names
    ):
        return True
    known_route_ids = {
        route_id
        for leg in _route_disclosure_legs(messages)
        for route_id in leg["route_ids"]
    }
    for action in actions:
        if action.get("action") != "respond":
            continue
        content = str(action.get("content") or "")
        if any(
            re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(route_id)}(?![A-Za-z0-9_-])",
                content,
            )
            for route_id in known_route_ids
        ):
            return True
        if re.search(
            r"\b(?:route|routes|routing|fastest|shortest|alternative|toll)\b",
            content,
            re.IGNORECASE,
        ):
            return True
    return False


def _route_presentation_policy_sections(
    corpus: PolicyCorpus,
) -> tuple[PolicySection, ...]:
    return tuple(
        section
        for section in corpus.sections
        if section.section_id == "policy-022"
        and "more information on alternative routes" in section.text.casefold()
    )


def _policy_corpus_with_pins(
    corpus: PolicyCorpus,
    pinned_sections: tuple[PolicySection, ...],
) -> PolicyCorpus:
    if not pinned_sections:
        return corpus
    pinned_ids = {section.section_id for section in pinned_sections}
    always_on_ids = tuple(
        section.section_id
        for section in corpus.sections
        if section.section_id in set(corpus.always_on_ids) | pinned_ids
    )
    core_text = "\n\n".join(
        section.text
        for section in corpus.sections
        if section.section_id in set(always_on_ids)
    )
    return replace(
        corpus,
        always_on_ids=always_on_ids,
        core_token_count=policy_token_count(core_text),
    )


def _tier2_fired_levers(
    *,
    candidate_tool_names: tuple[str, ...],
    subjects: frozenset[str],
    available_tool_names: frozenset[str],
    requires_confirmation_tool_names: frozenset[str],
    policy_conditioned_confirmation_tool_names: frozenset[str],
    route_options_present: bool,
    decision_presents_or_selects_route: bool,
    navigation_active: bool,
) -> tuple[str, ...]:
    candidate_set = frozenset(candidate_tool_names) & available_tool_names
    candidate_tool_names = tuple(sorted(candidate_set))
    fired: set[str] = set()
    if candidate_set & (
        requires_confirmation_tool_names
        | policy_conditioned_confirmation_tool_names
    ):
        fired.add("confirmation_protocol")
    if "scope_degree" in subjects and any(
        _is_direct_car_state_setter(name) for name in candidate_tool_names
    ):
        fired.add("scope_degree_mapping")
    if route_options_present and decision_presents_or_selects_route:
        fired.add("route_presentation_disclosure")
    if "explicit_read" in subjects and any(
        _tier2_is_read_tool(name) for name in candidate_tool_names
    ):
        fired.add("explicit_read_actual_output")
    if "preference" in subjects:
        fired.add("preference_fidelity")
    nav_mutation = {
        name
        for name in candidate_tool_names
        if name.casefold() == "set_new_navigation"
        or name.casefold().startswith("navigation_")
    }
    if "destination_change" in subjects or nav_mutation:
        fired.add("destination_mutation_completion")
    if any(
        name.casefold() == "navigation_replace_final_destination"
        or name.casefold() == "set_new_navigation" and navigation_active
        for name in nav_mutation
    ):
        fired.add("active_navigation_mutation")
    return tuple(
        lever_id for lever_id in POLICY_TIER2_LINE_NUMBERS if lever_id in fired
    )


def _policy_conditioned_confirmation_tool_names(
    corpus: PolicyCorpus,
) -> frozenset[str]:
    names: set[str] = set()
    for section in corpus.sections:
        if "subject:confirmation" not in section.keys:
            continue
        names.update(
            key.removeprefix("tool:")
            for key in section.keys
            if key.startswith("tool:")
        )
    return frozenset(names)


def _assemble_policy_injection(
    *,
    tier1_text: str,
    tier2_text: str,
    tier2_fired: tuple[str, ...],
    tier3_text: str,
    tier3_core_tokens: int,
    tier3_tail_tokens: int,
) -> PolicyInjection:
    text = "\n\n".join(
        part for part in (tier1_text, tier2_text, tier3_text) if part
    )
    return PolicyInjection(
        text=text,
        tier1_tokens=policy_token_count(tier1_text),
        tier2_fired=tier2_fired,
        tier2_tokens=policy_token_count(tier2_text),
        tier3_core_tokens=tier3_core_tokens,
        tier3_tail_tokens=tier3_tail_tokens,
        injected_tokens_total=policy_token_count(text),
    )


def _policy_action_tool_names(
    actions: list[dict[str, Any]],
    *,
    available_tool_names: frozenset[str] = frozenset(),
) -> set[str]:
    names: set[str] = set()
    for action in actions:
        for call in action.get("tool_calls") or []:
            name = call.get("tool_name")
            if not name and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            if name:
                names.add(str(name))
        content = str(action.get("content") or "").casefold()
        for available_name in available_tool_names:
            normalized_name = available_name.casefold()
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(normalized_name)}(?![a-z0-9_])",
                content,
            ):
                names.add(available_name)
    return names


def _policy_rag_meta(tally: _CallTally) -> dict[str, Any]:
    return {
        "policy_rag_retrievals": tally.policy_rag_retrievals,
        "policy_rag_empty": tally.policy_rag_empty,
        "policy_rag_tokens": tally.policy_rag_tokens,
        "policy_rag_section_ids": tally.policy_rag_section_ids,
        "policy_rag_tokens_per_call": tally.policy_rag_tokens_per_call,
        "tier1_tokens": tally.tier1_tokens,
        "tier1_tokens_per_call": tally.tier1_tokens_per_call,
        "tier2_fired": tally.tier2_fired,
        "tier2_fired_per_call": tally.tier2_fired_per_call,
        "tier2_tokens": tally.tier2_tokens,
        "tier2_tokens_per_call": tally.tier2_tokens_per_call,
        "tier3_core_tokens": tally.tier3_core_tokens,
        "tier3_core_tokens_per_call": tally.tier3_core_tokens_per_call,
        "tier3_tail_tokens": tally.tier3_tail_tokens,
        "tier3_tail_tokens_per_call": tally.tier3_tail_tokens_per_call,
        "injected_tokens_total": tally.injected_tokens_total,
        "injected_tokens_total_per_call": tally.injected_tokens_total_per_call,
    }


def _apply_output_guards(
    *,
    final_action: dict[str, Any],
    passes: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    guard_state: GuardState,
    enabled: bool,
    cascade_enabled: bool,
    response_guards_v2: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta: dict[str, Any] = {
        "output_guard_time_fixes": 0,
        "output_guard_confirm_fixes": 0,
        "confirm_intercepts": 0,
        "cascade_expansions": 0,
        "cascade_rules": [],
        "cascade_schema_skips": 0,
        "cascade_schema_skip_details": [],
        "disclosure_toll_fixes": 0,
        "disclosure_fastest_fixes": 0,
        "disclosure_tempdiff_fixes": 0,
    }
    guarded = dict(final_action)

    if enabled and guarded.get("action") == "respond":
        content = str(guarded.get("content", ""))
        content, time_fixes = _normalize_12h_times(content)
        meta["output_guard_time_fixes"] = time_fixes
        if response_guards_v2:
            content, disclosure_meta = _apply_response_disclosure_guards(
                content,
                messages,
                decision_actions=[
                    guarded,
                    *(item.get("action") or {} for item in passes),
                ],
            )
            meta.update(disclosure_meta)
        guarded["content"] = content
        return guarded, meta

    if enabled and guarded.get("action") == "tool_calls":
        intercepted = _intercept_requires_confirmation_call(
            guarded,
            tools=tools,
            messages=messages,
            guard_state=guard_state,
        )
        if intercepted is not None:
            meta["confirm_intercepts"] = 1
            return intercepted, meta
        climate_read = _climate_temperature_preread_action(
            guarded,
            tools=tools,
            messages=messages,
            guard_state=guard_state,
        )
        if climate_read is not None:
            meta["cascade_expansions"] = 1
            meta["cascade_rules"] = ["AUT-POL:012"]
            return climate_read, meta

    if cascade_enabled and guarded.get("action") == "tool_calls":
        guarded, cascade_meta = _expand_autpol_cascade(
            guarded,
            tools=tools,
            messages=messages,
            guard_state=guard_state,
        )
        meta.update(cascade_meta)
    return guarded, meta


def _append_notice(existing: str | None, notice: str) -> str:
    return notice if not existing else f"{existing}\n\n{notice}"


def _should_unknown_proceed_reroute(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bool:
    if action.get("action") != "respond":
        return False
    content = str(action.get("content") or "").lower()
    if not re.search(
        r"\b(can't|cannot|won't|unable|not able|sorry|can't verify|cannot safely)\b",
        content,
    ):
        return False
    return _has_unknown_get_result(messages) and _latest_user_explicit_action(messages)


def _unknown_proceed_transform(
    *,
    final_action: dict[str, Any],
    passes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    guard_state: GuardState,
) -> tuple[dict[str, Any] | None, str | None]:
    if not _should_unknown_proceed_reroute(final_action, messages):
        return None, None
    refusal_key = _unknown_refusal_site_key(final_action, messages)
    prior_count = guard_state.unknown_proceed_refusal_sites.get(refusal_key, 0)
    guard_state.unknown_proceed_refusal_sites[refusal_key] = prior_count + 1

    instructed = _instructed_tool_candidates(messages, tools)
    for item in passes:
        action = item.get("action") or {}
        if action.get("action") != "tool_calls":
            continue
        calls = [_normalized_tool_call(call) for call in action.get("tool_calls") or []]
        if _calls_match_instructed_action(calls, instructed):
            return {"action": "tool_calls", "tool_calls": calls}, "unknown_proceed_draft_override"

    confirm = _unknown_proceed_confirm_action(messages, instructed)
    if confirm is not None:
        return confirm, "unknown_proceed_confirm_ask"
    if prior_count >= 1 and instructed:
        tool_name, args = instructed[0]
        return _unknown_confirm_response(tool_name, args), "unknown_proceed_confirm_ask"
    return None, None


def _unknown_refusal_site_key(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str:
    latest_user = _latest_user_text(messages)
    content = str(action.get("content") or "")
    return json.dumps(
        {
            "user": _normalize_for_overlap(latest_user)[-240:],
            "refusal": _normalize_for_overlap(content)[:240],
        },
        sort_keys=True,
    )


def _instructed_tool_candidates(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    latest = _latest_user_text(messages)
    lowered = latest.lower()
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    available = {str(tool.get("function", {}).get("name") or ""): tool for tool in tools}

    def add(score: int, name: str, args: dict[str, Any]) -> None:
        if available and name not in available:
            return
        candidates.append((score, name, args))

    if re.search(r"\b(high[- ]?beam|high beams?|headlights?)\b", lowered) and re.search(
        r"\b(turn on|enable|switch on|set .*on|proceed)\b", lowered
    ):
        add(100, "set_head_lights_high_beams", {"on": True})
    if re.search(r"\b(fan|blower)\b", lowered):
        match = re.search(r"\b(?:level|speed)\s*(\d+)\b", lowered)
        if match and re.search(r"\b(set|raise|increase|turn|switch|change)\b", lowered):
            add(90, "set_fan_speed", {"level": int(match.group(1))})
    if re.search(r"\b(defrost|defog|windshield|windscreen)\b", lowered) and re.search(
        r"\b(turn on|enable|keep|start|set)\b", lowered
    ):
        add(80, "set_window_defrost", {"defrost_window": "FRONT", "on": True})
    if re.search(r"\b(ac|a/c|air conditioning)\b", lowered) and re.search(
        r"\b(turn on|enable|keep|start|set)\b", lowered
    ):
        add(70, "set_air_conditioning", {"on": True})

    candidates.sort(key=lambda item: -item[0])
    deduped: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for _, name, args in candidates:
        key = json.dumps({"name": name, "args": args}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, args))
    return deduped


def _calls_match_instructed_action(
    calls: list[dict[str, Any]],
    instructed: list[tuple[str, dict[str, Any]]],
) -> bool:
    if not calls or not instructed:
        return False
    for call in calls:
        name = str(call.get("tool_name") or "")
        args = call.get("arguments") or {}
        for expected_name, expected_args in instructed:
            if name != expected_name:
                continue
            if all(args.get(key) == value for key, value in expected_args.items()):
                return True
    return False


def _unknown_proceed_confirm_action(
    messages: list[dict[str, Any]],
    instructed: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    if not instructed:
        return None
    tool_name, args = instructed[0]
    return _unknown_confirm_response(tool_name, args)


def _unknown_confirm_response(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    args_text = json.dumps(args, sort_keys=True)
    return {
        "action": "respond",
        "content": (
            "I still have an unknown value from the readback, so I will not claim it "
            f"is known. I can proceed with your explicit instruction by calling "
            f"{tool_name} with {args_text}. Please confirm."
        ),
    }


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return ""


def _has_unknown_get_result(messages: list[dict[str, Any]]) -> bool:
    get_call_ids: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw_call in _assistant_message_tool_calls(message):
            function = raw_call.get("function") or {}
            name = str(function.get("name") or "").lower()
            if name.startswith("get_"):
                call_id = raw_call.get("id")
                if call_id is not None:
                    get_call_ids.add(str(call_id))
    unknown_pattern = re.compile(
        r"\b(unknown|unavailable|not available|unreadable|unable to read|could not read|null)\b",
        re.IGNORECASE,
    )
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_name = str(message.get("name") or "").lower()
        call_id = str(message.get("tool_call_id") or "")
        if not tool_name.startswith("get_") and call_id not in get_call_ids:
            continue
        if unknown_pattern.search(_message_text(message)):
            return True
        result = _tool_success_result(message)
        if result is not None and _object_contains_unknown(result):
            return True
    return False


def _object_contains_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"unknown", "unavailable", "not available", "null"}
    if isinstance(value, dict):
        return any(_object_contains_unknown(item) for item in value.values())
    if isinstance(value, list):
        return any(_object_contains_unknown(item) for item in value)
    return False


def _latest_user_explicit_action(messages: list[dict[str, Any]]) -> bool:
    texts: list[str] = []
    for message in reversed(messages):
        if message.get("role") == "user":
            texts.append(_message_text(message).lower())
        if len(texts) >= 2:
            break
    text = "\n".join(texts)
    return bool(
        re.search(
            r"\b(turn|set|increase|raise|lower|close|open|start|navigate|replace|"
            r"remove|delete|send|call|enable|disable|switch|change|proceed|do it)\b",
            text,
        )
    )


def _nav_preflight_issue(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if action.get("action") != "tool_calls":
        return None
    result_text = _prior_tool_result_text(messages)
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        name = str(call.get("tool_name") or "")
        if name == "set_new_navigation" and _navigation_active_without_delete(messages):
            return {
                "tool_name": name,
                "counter": "preflight_nav_active_blocks",
                "notice": NAV_ACTIVE_PREFLIGHT_NOTICE,
            }
        if name == "set_new_navigation" and not _has_fresh_inactive_navigation_read(messages):
            return {
                "tool_name": name,
                "counter": "preflight_navread_blocks",
                "notice": NAV_READ_PREFLIGHT_NOTICE,
            }
        name_issue = _name_guess_preflight_issue(name, call.get("arguments") or {}, messages)
        if name_issue is not None:
            return name_issue
        type_issue = _id_type_preflight_issue(name, call.get("arguments") or {})
        if type_issue is not None:
            return type_issue
        for key, value in _iter_id_arguments(call.get("arguments") or {}):
            del key
            if str(value) not in result_text:
                return {
                    "tool_name": name,
                    "counter": "preflight_ungrounded_blocks",
                    "notice": NAV_UNGROUNDED_PREFLIGHT_NOTICE,
                }
        segment_issue = _nav_route_segment_mismatch_issue(
            name, call.get("arguments") or {}, messages
        )
        if segment_issue is not None:
            segment_issue = dict(segment_issue)
            segment_issue["notice"] = _delete_aware_notice(
                str(segment_issue.get("notice") or ""),
                messages,
            )
            return segment_issue
    return None


_ID_PREFIX_TYPES = {
    "loc": "location_or_poi",
    "poi": "location_or_poi",
    "rll": "route",
    "rlp": "route",
    "rpl": "route",
    "route": "route",
    "con": "contact",
    "cal": "calendar",
    "plg": "plug",
}

_TOOL_ID_ARGUMENT_TYPES = {
    "get_routes": {
        "start_id": {"location_or_poi"},
        "destination_id": {"location_or_poi"},
    },
    "get_routes_from_start_to_destination": {
        "start_id": {"location_or_poi"},
        "destination_id": {"location_or_poi"},
    },
    "set_new_navigation": {
        "route_ids": {"route"},
    },
    "navigation_select_route": {
        "route_id": {"route"},
    },
    "navigation_replace_final_destination": {
        "new_destination_id": {"location_or_poi"},
        "route_id_leading_to_new_destination": {"route"},
    },
    "navigation_replace_one_waypoint": {
        "new_waypoint_id": {"location_or_poi"},
        "waypoint_id_to_replace": {"location_or_poi"},
        "route_id_leading_to_new_waypoint": {"route"},
        "route_id_leading_away_from_new_waypoint": {"route"},
    },
    "navigation_delete_waypoint": {
        "waypoint_id": {"location_or_poi"},
    },
    "search_poi_along_the_route": {
        "route_id": {"route"},
    },
    "convert_route_distance_and_time": {
        "route_id": {"route"},
    },
    "search_poi_at_location": {
        "location_id": {"location_or_poi"},
        "location_or_poi_id": {"location_or_poi"},
    },
    "get_weather": {
        "location_or_poi_id": {"location_or_poi"},
    },
}


def _id_type_preflight_issue(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, str] | None:
    expected = _TOOL_ID_ARGUMENT_TYPES.get(tool_name)
    if not expected:
        return None
    for key, types in expected.items():
        if key not in arguments:
            continue
        values = arguments[key] if isinstance(arguments[key], list) else [arguments[key]]
        for value in values:
            actual = _id_prefix_type(value)
            if isinstance(value, str) and value.strip() and actual is None:
                expected_text = "/".join(sorted(types))
                return {
                    "tool_name": tool_name,
                    "counter": "preflight_type_blocks",
                    "notice": (
                        f"ID argument {key} has no recognized ID prefix; expected "
                        f"{expected_text}. Re-read the relevant tool result and use "
                        "the exact ID from that result before acting."
                    ),
                }
            if actual is not None and actual not in types:
                expected_text = "/".join(sorted(types))
                return {
                    "tool_name": tool_name,
                    "counter": "preflight_type_blocks",
                    "notice": (
                        f"ID argument {key} has prefix type {actual}; expected "
                        f"{expected_text}. Re-read the relevant tool result and use "
                        "an ID with the correct prefix type before acting."
                    ),
                }
    if tool_name in {"get_routes", "get_routes_from_start_to_destination"}:
        start_id = arguments.get("start_id")
        destination_id = arguments.get("destination_id")
        if (
            isinstance(start_id, str)
            and isinstance(destination_id, str)
            and start_id
            and start_id == destination_id
        ):
            return {
                "tool_name": tool_name,
                "counter": "preflight_type_blocks",
                "notice": (
                    "Route lookup start_id and destination_id are identical; expected "
                    "two distinct route endpoint IDs. Re-read the request and known "
                    "locations before acting."
                ),
            }
    return None


_NAME_LOOKUP_ARGUMENTS = {
    "get_location_id_by_location_name": ("location",),
}


def _name_guess_preflight_issue(
    tool_name: str,
    arguments: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, str] | None:
    keys = _NAME_LOOKUP_ARGUMENTS.get(tool_name)
    if not keys:
        return None
    grounded_text = _user_and_tool_grounding_text(messages)
    for key in keys:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if _text_contains_phrase(grounded_text, value):
            continue
        return {
            "tool_name": tool_name,
            "counter": "preflight_name_guess_blocks",
            "notice": NAME_GUESS_PREFLIGHT_NOTICE,
        }
    return None


def _id_prefix_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^([a-z]+)_", value)
    if not match:
        return None
    return _ID_PREFIX_TYPES.get(match.group(1))


def _user_and_tool_grounding_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") in {"user", "tool"}:
            parts.append(_message_text(message))
    return "\n".join(parts)


def _text_contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", text).casefold()
    normalized_phrase = re.sub(r"\s+", " ", phrase).strip().casefold()
    return bool(normalized_phrase and normalized_phrase in normalized_text)


def _preflight_signature(issue: dict[str, Any]) -> tuple[str, str]:
    return (str(issue.get("tool_name") or ""), str(issue.get("counter") or ""))


def _record_preflight_block(state: PreflightState, issue: dict[str, Any]) -> int:
    signature = _preflight_signature(issue)
    state.block_signatures[signature] = state.block_signatures.get(signature, 0) + 1
    return state.block_signatures[signature]


def _record_preflight_response(state: PreflightState, issue: dict[str, Any]) -> int:
    signature = _preflight_signature(issue)
    state.response_templates[signature] = state.response_templates.get(signature, 0) + 1
    return state.response_templates[signature]


def _preflight_tripwire_meta(
    state: PreflightState,
    issue: dict[str, Any],
    block_count: int,
) -> dict[str, int]:
    if issue.get("counter") != "preflight_segment_mismatch_blocks":
        return {
            "preflight_segment_mismatch_regressions": 0,
        }
    state.max_segment_mismatch_blocks = max(
        state.max_segment_mismatch_blocks,
        block_count,
    )
    return {
        "preflight_segment_mismatch_regressions": int(
            state.max_segment_mismatch_blocks > 4
        ),
    }


def _corrective_read_for_repeated_segment_mismatch(
    issue: dict[str, Any],
    *,
    block_count: int,
    state: PreflightState,
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if block_count < 2:
        return None
    if issue.get("counter") != "preflight_segment_mismatch_blocks":
        return None
    if issue.get("route_substitution_id"):
        return None
    if state.corrective_reads_suppressed or state.corrective_reads >= 2:
        state.corrective_reads_suppressed = True
        return None
    if issue.get("tool_name") != "navigation_replace_final_destination":
        return None
    if _latest_user_removal_command(messages):
        state.corrective_reads_suppressed = True
        return None
    corrective_args = issue.get("corrective_read_args")
    if not isinstance(corrective_args, dict):
        return None
    start_id = corrective_args.get("start_id")
    destination_id = corrective_args.get("destination_id")
    if not isinstance(start_id, str) or not isinstance(destination_id, str):
        return None
    if not start_id or not destination_id:
        return None
    if start_id == destination_id:
        state.corrective_reads_suppressed = True
        return None
    if _id_prefix_type(start_id) != "location_or_poi":
        state.corrective_reads_suppressed = True
        return None
    if _id_prefix_type(destination_id) != "location_or_poi":
        state.corrective_reads_suppressed = True
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "get_routes_from_start_to_destination",
                "arguments": {
                    "start_id": start_id,
                    "destination_id": destination_id,
                },
            }
        ],
    }


def _latest_user_removal_command(messages: list[dict[str, Any]]) -> bool:
    text = _latest_user_text(messages).lower()
    return bool(
        re.search(
            r"\b(remove\w*|delet\w*|drop\w*|cancel\w*|skip\w*|"
            r"take\s+out|get\s+rid\s+of)\b",
            text,
        )
        and re.search(r"\b(stop|waypoint|destination|route|navigation)\b", text)
    )


def _delete_aware_notice(notice: str, messages: list[dict[str, Any]]) -> str:
    if not _latest_user_removal_command(messages):
        return notice
    if "fetch routes" not in notice and "fetch a route" not in notice:
        return notice
    resolved_tool = _removal_target_delete_tool(messages)
    if resolved_tool is not None:
        return (
            "PREFLIGHT: the latest user command is a route-stop removal. Call "
            f"{resolved_tool} with the target ID from the current navigation "
            "state instead of fetching replacement routes or using replace tools."
        )
    return (
        "PREFLIGHT: the latest user command is a route-stop removal. Prefer "
        "navigation_delete_destination or navigation_delete_waypoint with an "
        "ID from the current navigation state instead of fetching replacement "
        "routes or using replace tools."
    )


def _removal_target_delete_tool(messages: list[dict[str, Any]]) -> str | None:
    state = _latest_navigation_state(messages)
    if state is None:
        return None
    waypoint_ids = state.get("waypoints_id")
    details = state.get("details")
    detailed_waypoints = details.get("waypoints") if isinstance(details, dict) else None
    if not isinstance(waypoint_ids, list):
        waypoint_ids = []
    candidates: list[tuple[str, str]] = []
    if isinstance(detailed_waypoints, list):
        for waypoint in detailed_waypoints:
            if not isinstance(waypoint, dict):
                continue
            waypoint_id = str(waypoint.get("id") or "")
            waypoint_name = str(waypoint.get("name") or "")
            if waypoint_id:
                candidates.append((waypoint_id, waypoint_id))
            if waypoint_name and waypoint_id:
                candidates.append((waypoint_name, waypoint_id))
    user_text = _latest_user_text(messages).casefold()
    matched_id = next(
        (
            waypoint_id
            for label, waypoint_id in sorted(candidates, key=lambda item: -len(item[0]))
            if label.casefold() in user_text
        ),
        None,
    )
    if matched_id is None or matched_id not in waypoint_ids:
        return None
    index = waypoint_ids.index(matched_id)
    if index == len(waypoint_ids) - 1:
        return "navigation_delete_destination"
    if 0 < index < len(waypoint_ids) - 1:
        return "navigation_delete_waypoint"
    return None


def _latest_navigation_state(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if str(message.get("name") or "") != "get_current_navigation_state":
            continue
        result = _tool_success_result(message)
        if isinstance(result, dict):
            return result
    return None


def _terminal_honesty_action(
    state: PreflightState,
    issue: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    route_hint_cap_reached: bool = False,
) -> dict[str, Any] | None:
    if issue.get("counter") != "preflight_segment_mismatch_blocks":
        return None
    signature = _preflight_signature(issue)
    if _refresh_route_substitution(issue, messages) and not route_hint_cap_reached:
        state.terminal_honesty_signatures.discard(signature)
        return None
    total_blocks = sum(
        count
        for candidate, count in state.block_signatures.items()
        if candidate[1] == "preflight_segment_mismatch_blocks"
    )
    if (
        signature not in state.terminal_honesty_signatures
        and state.block_signatures.get(signature, 0) < 4
        and total_blocks < 6
    ):
        return None
    state.terminal_honesty_signatures.add(signature)
    current_state = _current_route_truth(messages)
    content = "I wasn't able to complete that change with the tools available"
    if current_state:
        content += f"; {current_state}"
    return {"action": "respond", "content": content + "."}


def _current_route_truth(messages: list[dict[str, Any]]) -> str:
    state = _latest_navigation_state(messages)
    if state is None:
        return "the requested route change has not been completed"
    waypoint_ids = state.get("waypoints_id")
    details = state.get("details")
    detailed_waypoints = details.get("waypoints") if isinstance(details, dict) else None
    names: list[str] = []
    if isinstance(detailed_waypoints, list):
        names = [
            str(item.get("name") or item.get("id") or "")
            for item in detailed_waypoints
            if isinstance(item, dict)
        ]
    if not names and isinstance(waypoint_ids, list):
        names = [str(item) for item in waypoint_ids]
    names = [name for name in names if name]
    if not state.get("navigation_active"):
        return "navigation is currently inactive"
    if len(names) >= 2:
        return f"the current route still ends at {names[-1]}"
    return "the current navigation state remains unchanged"


def _preflight_block_response(
    notice: str,
    *,
    response_count: int = 1,
    messages: list[dict[str, Any]] | None = None,
    issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if issue and issue.get("route_substitution_id"):
        start_name = _navigation_waypoint_label(
            messages or [], str(issue.get("required_start_id") or "")
        )
        destination_name = _navigation_waypoint_label(
            messages or [], str(issue.get("required_destination_id") or "")
        )
        return {
            "action": "respond",
            "content": (
                f"I found a valid route from {start_name} to {destination_name}, "
                "so the destination change can be made. Would you like me to "
                "proceed with that route?"
            ),
        }
    if response_count > 2:
        current_state = _current_route_truth(messages or [])
        return {
            "action": "respond",
            "content": (
                "I wasn't able to complete that change with the tools available; "
                f"{current_state}."
            ),
        }
    if notice == NAME_GUESS_PREFLIGHT_NOTICE:
        content = (
            "I need the exact place name from you or from a search result before I "
            "can look up that location. Which place should I use?"
        )
    elif "ID argument" in notice or "Route lookup" in notice:
        content = (
            "I need the correct navigation ID from a prior result before I can do "
            "that. Please confirm the destination or let me look it up first."
        )
    else:
        content = (
            "I need to verify the navigation details before I can do that. Please "
            "confirm the destination or route details."
        )
    return {"action": "respond", "content": content}


def _navigation_waypoint_label(messages: list[dict[str, Any]], waypoint_id: str) -> str:
    state = _latest_navigation_state(messages) or {}
    details = state.get("details")
    waypoints = details.get("waypoints") if isinstance(details, dict) else None
    if isinstance(waypoints, list):
        for waypoint in waypoints:
            if not isinstance(waypoint, dict):
                continue
            if str(waypoint.get("id") or "") == waypoint_id:
                return str(waypoint.get("name") or waypoint_id)
    return waypoint_id or "the required endpoint"


def _rewrite_nav_detailed_read(action: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if action.get("action") != "tool_calls":
        return action, 0
    calls = action.get("tool_calls") or []
    rewrites = 0
    new_calls: list[dict[str, Any]] = []
    for raw_call in calls:
        call = dict(raw_call)
        normalized = _normalized_tool_call(call)
        if normalized.get("tool_name") == "get_current_navigation_state":
            args = dict(normalized.get("arguments") or {})
            if args.get("detailed_information") is not True:
                args["detailed_information"] = True
                call["arguments"] = args
                rewrites += 1
        new_calls.append(call)
    if not rewrites:
        return action, 0
    rewritten = dict(action)
    rewritten["tool_calls"] = new_calls
    return rewritten, rewrites


def _compact_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    if action.get("action") != "tool_calls":
        return {
            "action": action.get("action"),
            "content": str(action.get("content", ""))[:240],
        }
    calls = []
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        calls.append(
            {
                "tool_name": call.get("tool_name"),
                "arguments": _truncate_for_log(call.get("arguments") or {}),
            }
        )
    return {"action": "tool_calls", "tool_calls": calls}


def _repair_tool_names_subset(
    pre_repair_action: dict[str, Any],
    repaired_action: dict[str, Any],
) -> bool:
    repaired_tools = _action_tool_name_set(repaired_action)
    if not repaired_tools:
        return True
    return repaired_tools <= _action_tool_name_set(pre_repair_action)


def _action_tool_name_set(action: dict[str, Any]) -> set[str]:
    if action.get("action") != "tool_calls":
        return set()
    names: set[str] = set()
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        name = call.get("tool_name")
        if name:
            names.add(str(name))
    return names


def _truncate_for_log(value: Any, limit: int = 240) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[: limit - 3] + "..."


def _iter_id_arguments(arguments: dict[str, Any]) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for key, value in arguments.items():
        key_text = str(key)
        if not re.search(
            r"(?:^|_)(route|waypoint|destination)_?ids?(?:_|$)",
            key_text,
        ):
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, (str, int, float)) and str(item):
                pairs.append((key_text, item))
    return pairs


def _prior_tool_result_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        if message.get("role") == "tool":
            chunks.append(_message_text(message))
    return "\n".join(chunks)


def _navigation_active_without_delete(messages: list[dict[str, Any]]) -> bool:
    active_seen = False
    for message in messages:
        if message.get("role") == "tool":
            text = _message_text(message)
            if re.search(r'"?navigation_active"?\s*:\s*true', text, re.IGNORECASE):
                active_seen = True
        if message.get("role") != "assistant":
            continue
        for raw_call in _assistant_message_tool_calls(message):
            function = raw_call.get("function") or {}
            name = str(function.get("name") or "")
            if name in {"delete_current_navigation", "stop_navigation", "clear_navigation"}:
                active_seen = False
    return active_seen


_NAV_MUTATING_TOOLS = {
    "set_new_navigation",
    "navigation_select_route",
    "navigation_replace_final_destination",
    "navigation_replace_one_waypoint",
    "navigation_add_one_waypoint",
    "navigation_delete_waypoint",
    "navigation_delete_destination",
    "delete_current_navigation",
    "stop_navigation",
    "clear_navigation",
}


def _has_fresh_inactive_navigation_read(messages: list[dict[str, Any]]) -> bool:
    latest_mutation_index = -1
    latest_nav_state: dict[str, Any] | None = None
    latest_nav_state_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for raw_call in _assistant_message_tool_calls(message):
                function = raw_call.get("function") or {}
                if str(function.get("name") or "") in _NAV_MUTATING_TOOLS:
                    latest_mutation_index = index
                    latest_nav_state = None
                    latest_nav_state_index = -1
        if message.get("role") != "tool" or message.get("name") != "get_current_navigation_state":
            continue
        result = _tool_success_result(message)
        if isinstance(result, dict):
            latest_nav_state = result
            latest_nav_state_index = index
    return (
        latest_nav_state_index > latest_mutation_index
        and latest_nav_state is not None
        and latest_nav_state.get("navigation_active") is False
    )


def _nav_route_segment_mismatch_issue(
    tool_name: str,
    arguments: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    routes = _known_navigation_routes(messages)
    if not routes:
        return None
    waypoints = _latest_navigation_waypoints(messages)
    if tool_name == "navigation_replace_final_destination":
        route_id = str(arguments.get("route_id_leading_to_new_destination") or "")
        new_destination_id = str(arguments.get("new_destination_id") or "")
        route = routes.get(route_id)
        if not route:
            return None
        if new_destination_id and route.get("destination_id") != new_destination_id:
            return {
                "tool_name": tool_name,
                "counter": "preflight_segment_mismatch_blocks",
                "notice": (
                    "PREFLIGHT: route_id_leading_to_new_destination must end at "
                    f"new_destination_id {new_destination_id}; fetch routes to "
                    "that destination and use one of those."
                ),
            }
        if len(waypoints) >= 2 and route.get("start_id") != waypoints[-2]:
            start_id = waypoints[-2]
            notice = (
                "PREFLIGHT: route_id_leading_to_new_destination must start at "
                f"the remaining waypoint {start_id}; fetch routes from {start_id} "
                f"to {new_destination_id} and use one of those."
            )
            issue: dict[str, Any] = {
                "tool_name": tool_name,
                "counter": "preflight_segment_mismatch_blocks",
                "notice": notice,
                "required_start_id": start_id,
                "required_destination_id": new_destination_id,
            }
            if new_destination_id:
                issue["corrective_read_args"] = {
                    "start_id": start_id,
                    "destination_id": new_destination_id,
                }
            return issue
    if tool_name == "navigation_replace_one_waypoint":
        route_to_id = str(arguments.get("route_id_leading_to_new_waypoint") or "")
        route_away_id = str(arguments.get("route_id_leading_away_from_new_waypoint") or "")
        old_waypoint_id = str(arguments.get("waypoint_id_to_replace") or "")
        new_waypoint_id = str(arguments.get("new_waypoint_id") or "")
        if old_waypoint_id not in waypoints:
            return None
        index = waypoints.index(old_waypoint_id)
        previous_waypoint = waypoints[index - 1] if index > 0 else ""
        next_waypoint = waypoints[index + 1] if index + 1 < len(waypoints) else ""
        route_to = routes.get(route_to_id)
        route_away = routes.get(route_away_id)
        if route_to:
            if previous_waypoint and route_to.get("start_id") != previous_waypoint:
                return {
                    "tool_name": tool_name,
                    "counter": "preflight_segment_mismatch_blocks",
                    "notice": (
                        "PREFLIGHT: route_id_leading_to_new_waypoint must start "
                        f"at the prior waypoint {previous_waypoint}; fetch a route "
                        "for that segment and use it."
                    ),
                }
            if new_waypoint_id and route_to.get("destination_id") != new_waypoint_id:
                return {
                    "tool_name": tool_name,
                    "counter": "preflight_segment_mismatch_blocks",
                    "notice": (
                        "PREFLIGHT: route_id_leading_to_new_waypoint must end at "
                        f"new_waypoint_id {new_waypoint_id}; fetch a route for "
                        "that segment and use it."
                    ),
                }
        if route_away:
            if new_waypoint_id and route_away.get("start_id") != new_waypoint_id:
                return {
                    "tool_name": tool_name,
                    "counter": "preflight_segment_mismatch_blocks",
                    "notice": (
                        "PREFLIGHT: route_id_leading_away_from_new_waypoint must "
                        f"start at new_waypoint_id {new_waypoint_id}; fetch a route "
                        "for that segment and use it."
                    ),
                }
            if next_waypoint and route_away.get("destination_id") != next_waypoint:
                return {
                    "tool_name": tool_name,
                    "counter": "preflight_segment_mismatch_blocks",
                    "notice": (
                        "PREFLIGHT: route_id_leading_away_from_new_waypoint must "
                        f"end at the next waypoint {next_waypoint}; fetch a route "
                        "for that segment and use it."
                    ),
                }
    return None


def _refresh_route_substitution(
    issue: dict[str, Any], messages: list[dict[str, Any]]
) -> str | None:
    start_id = str(issue.get("required_start_id") or "")
    destination_id = str(issue.get("required_destination_id") or "")
    if not start_id or not destination_id:
        issue.pop("route_substitution_id", None)
        return None
    routes = _known_navigation_routes(messages)
    replacement_id = next(
        (
            route_id
            for route_id, route in sorted(routes.items())
            if route.get("start_id") == start_id
            and route.get("destination_id") == destination_id
        ),
        None,
    )
    if replacement_id is None:
        issue.pop("route_substitution_id", None)
        return None
    issue["route_substitution_id"] = replacement_id
    base_notice = str(issue.get("notice") or "")
    hint = (
        "A route matching the required segment exists in prior results: use "
        f"route_id_leading_to_new_destination={replacement_id}."
    )
    if hint not in base_notice:
        issue["notice"] = f"{base_notice} {hint}".strip()
    return replacement_id


def _record_route_substitution_hint(
    state: PreflightState,
    issue: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    capped: bool,
) -> int:
    replacement_id = _refresh_route_substitution(issue, messages)
    if replacement_id is None:
        return 0
    if capped and state.route_substitution_hints >= 3:
        hint = (
            "A route matching the required segment exists in prior results: use "
            f"route_id_leading_to_new_destination={replacement_id}."
        )
        issue["notice"] = str(issue.get("notice") or "").replace(hint, "").strip()
        return 0
    state.route_substitution_hints += 1
    return 1


def _unique_route_substitution_id(
    issue: dict[str, Any], messages: list[dict[str, Any]]
) -> str | None:
    start_id = str(issue.get("required_start_id") or "")
    destination_id = str(issue.get("required_destination_id") or "")
    if not start_id or not destination_id:
        return None
    matches: list[tuple[str, set[str]]] = []
    route_tools = (
        "get_current_navigation_state",
        "get_routes",
        "get_routes_from_start_to_destination",
        "get_route_options",
    )
    for payload in _tool_success_payloads(messages, route_tools):
        for route in _walk_dicts(payload):
            route_id = str(route.get("route_id") or route.get("id") or "")
            if not route_id:
                continue
            if str(route.get("start_id") or "") != start_id:
                continue
            if str(route.get("destination_id") or "") != destination_id:
                continue
            aliases = route.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            matches.append((route_id, {str(alias).casefold() for alias in aliases}))
    latest_user = _latest_user_text(messages).casefold()
    if "fastest" in latest_user:
        matches = [item for item in matches if "fastest" in item[1]]
    elif "shortest" in latest_user:
        matches = [item for item in matches if "shortest" in item[1]]
    route_ids = sorted({route_id for route_id, _ in matches})
    return route_ids[0] if len(route_ids) == 1 else None


def _approval_gated_route_substitution(
    *,
    action: dict[str, Any],
    issue: dict[str, Any],
    messages: list[dict[str, Any]],
    block_count: int,
    state: PreflightState,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if state.route_arg_substitutions >= 1 or block_count < 2:
        return None, None
    if issue.get("counter") != "preflight_segment_mismatch_blocks":
        return None, None
    if issue.get("tool_name") != "navigation_replace_final_destination":
        return None, None
    latest_user = _latest_user_text(messages)
    destination_id = str(issue.get("required_destination_id") or "")
    if not _explicit_route_approval(latest_user, destination_id, messages):
        return None, None
    replacement_id = _unique_route_substitution_id(issue, messages)
    if replacement_id is None or action.get("action") != "tool_calls":
        return None, None
    rewritten_calls: list[dict[str, Any]] = []
    fingerprint: dict[str, str] | None = None
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        if call.get("tool_name") != "navigation_replace_final_destination":
            rewritten_calls.append(call)
            continue
        arguments = dict(call.get("arguments") or {})
        if str(arguments.get("new_destination_id") or "") != destination_id:
            return None, None
        blocked_id = str(arguments.get("route_id_leading_to_new_destination") or "")
        arguments["route_id_leading_to_new_destination"] = replacement_id
        rewritten_calls.append({"tool_name": call["tool_name"], "arguments": arguments})
        fingerprint = {"blocked_route_id": blocked_id, "replacement_route_id": replacement_id}
    if fingerprint is None:
        return None, None
    return {"action": "tool_calls", "tool_calls": rewritten_calls}, fingerprint


def _route_replacement_call_key(action: dict[str, Any]) -> str | None:
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        if call.get("tool_name") == "navigation_replace_final_destination":
            return _tool_call_sort_key(call)
    return None


def _post_substitution_extra_nav_mutations(
    state: PreflightState,
    action: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    if action.get("action") != "tool_calls":
        return 0, []
    details: list[dict[str, Any]] = []
    skipped_substitution = False
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        tool_name = str(call.get("tool_name") or "")
        if tool_name not in _NAV_MUTATING_TOOLS:
            continue
        call_key = _tool_call_sort_key(call)
        if (
            state.substitution_emission_pending
            and not skipped_substitution
            and call_key == state.substitution_tool_call_key
        ):
            skipped_substitution = True
            continue
        details.append(
            {
                "tool_name": tool_name,
                "arguments": _truncate_for_log(call.get("arguments") or {}),
            }
        )
    if state.substitution_emission_pending and skipped_substitution:
        state.substitution_emission_pending = False
    return len(details), details


def _explicit_route_approval(
    latest_user: str,
    destination_id: str,
    messages: list[dict[str, Any]],
) -> bool:
    text = latest_user.casefold()
    if not re.search(r"\b(yes|approved?|proceed|go ahead|do it)\b", text):
        return False
    references_change = bool(
        re.search(r"\b(route|destination|navigation|navigate|change|replace)\b", text)
    )
    destination_labels = _labels_for_grounded_id(destination_id, messages)
    references_destination = destination_id.casefold() in text or any(
        label.casefold() in text for label in destination_labels
    )
    if not (references_change or references_destination):
        return False
    for message in messages:
        if message.get("role") != "user":
            continue
        command = _message_text(message).casefold()
        if not re.search(r"\b(change|replace|set|start|navigate|destination|route)\b", command):
            continue
        if destination_id.casefold() in command or any(
            label.casefold() in command for label in destination_labels
        ):
            return True
    return False


def _labels_for_grounded_id(
    identifier: str, messages: list[dict[str, Any]]
) -> set[str]:
    labels: set[str] = set()
    payloads = [
        _tool_success_result(message)
        for message in messages
        if message.get("role") == "tool"
    ]
    for payload in payloads:
        if payload is None:
            continue
        for item in _walk_dicts(payload):
            item_id = str(item.get("id") or item.get("location_id") or "")
            if item_id != identifier:
                continue
            for key in ("name", "location", "destination_name"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    labels.add(value.strip())
    pending_labels: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for raw_call in _assistant_message_tool_calls(message):
                function = raw_call.get("function") or {}
                if function.get("name") != "get_location_id_by_location_name":
                    continue
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                label = arguments.get("location") if isinstance(arguments, dict) else None
                call_id = raw_call.get("id")
                if isinstance(call_id, str) and isinstance(label, str):
                    pending_labels[call_id] = label
        if message.get("role") != "tool":
            continue
        result = _tool_success_result(message)
        if not isinstance(result, dict) or str(result.get("id") or "") != identifier:
            continue
        label = pending_labels.get(str(message.get("tool_call_id") or ""))
        if label:
            labels.add(label)
    return labels


def _known_navigation_routes(messages: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    routes: dict[str, dict[str, str]] = {}
    for payload in _tool_success_payloads(
        messages,
        (
            "get_current_navigation_state",
            "get_routes",
            "get_routes_from_start_to_destination",
            "get_route_options",
        ),
    ):
        for route in _walk_dicts(payload):
            route_id = str(route.get("route_id") or route.get("id") or "")
            start_id = str(route.get("start_id") or "")
            destination_id = str(route.get("destination_id") or "")
            if route_id and start_id and destination_id:
                routes[route_id] = {
                    "start_id": start_id,
                    "destination_id": destination_id,
                }
        for item in _walk_dicts(payload):
            waypoint_ids = item.get("waypoints_id")
            route_ids = item.get("routes_to_final_destination_id")
            if not isinstance(waypoint_ids, list) or not isinstance(route_ids, list):
                continue
            waypoints = [str(value) for value in waypoint_ids if isinstance(value, str) and value]
            route_list = [str(value) for value in route_ids if isinstance(value, str) and value]
            for index, route_id in enumerate(route_list):
                if route_id in routes or index + 1 >= len(waypoints):
                    continue
                routes[route_id] = {
                    "start_id": waypoints[index],
                    "destination_id": waypoints[index + 1],
                }
    return routes


def _latest_navigation_waypoints(messages: list[dict[str, Any]]) -> list[str]:
    for payload in reversed(_tool_success_payloads(messages, "get_current_navigation_state")):
        for item in _walk_dicts(payload):
            raw = item.get("waypoints_id")
            if isinstance(raw, list):
                waypoints = [str(value) for value in raw if isinstance(value, str) and value]
                if waypoints:
                    return waypoints
    return []


def _apply_response_disclosure_guards(
    content: str,
    messages: list[dict[str, Any]],
    decision_actions: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, int]]:
    meta = {
        "disclosure_toll_fixes": 0,
        "disclosure_fastest_fixes": 0,
        "disclosure_tempdiff_fixes": 0,
    }
    additions: list[str] = []
    toll_ids, fastest_ids = _route_disclosure_values(messages)
    normalized = content.lower()
    if toll_ids and "toll" not in normalized:
        ids = ", ".join(toll_ids)
        additions.append(f"Route option(s) {ids} include toll roads.")
        meta["disclosure_toll_fixes"] = 1
    if fastest_ids and "alternative" not in normalized and "more information" not in normalized:
        if "fastest" in normalized:
            additions.append("Ask if you want information on alternative routes.")
            meta["disclosure_fastest_fixes"] = 1
        else:
            selected_ids = _decision_selected_route_ids(
                content,
                messages,
                decision_actions or [],
            )
            selected_fastest_ids = _selected_fastest_route_ids(
                messages,
                selected_ids,
            )
            if selected_fastest_ids:
                ids = ", ".join(selected_fastest_ids)
                additions.append(
                    f"I selected the fastest available route ({ids}); ask if you want "
                    "information on alternative routes."
                )
                meta["disclosure_fastest_fixes"] = 1
    tempdiff = _temperature_difference_disclosure(messages)
    if tempdiff and not _temperature_difference_already_disclosed(normalized):
        additions.append(tempdiff)
        meta["disclosure_tempdiff_fixes"] = 1
    if additions:
        suffix = " ".join(additions)
        content = f"{content.rstrip()} {suffix}" if content.strip() else suffix
    return content, meta


def replay_response_disclosure_guards(
    *,
    response_content: str,
    messages: list[dict[str, Any]],
    decision_actions: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Run response-disclosure detectors without mutating a live decision."""

    _, meta = _apply_response_disclosure_guards(
        response_content,
        messages,
        decision_actions=decision_actions,
    )
    return meta


def replay_nav_preflight(
    *,
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    """Run nav preflight detectors without invoking the planner."""

    meta = {
        "preflight_nav_active_blocks": 0,
        "preflight_navread_blocks": 0,
        "preflight_ungrounded_blocks": 0,
        "preflight_segment_mismatch_blocks": 0,
        "preflight_corrective_reads": 0,
        "preflight_corrective_read_suppressed": 0,
        "preflight_segment_mismatch_regressions": 0,
        "route_substitution_hints": 0,
        "preflight_capped_template_blocks": 0,
        "preflight_type_blocks": 0,
        "preflight_name_guess_blocks": 0,
    }
    issue = _nav_preflight_issue(action, messages)
    if issue is not None:
        meta[issue["counter"]] = 1
        meta["route_substitution_hints"] = int(
            bool(_refresh_route_substitution(issue, messages))
        )
    return meta


def replay_unknown_proceed_transform(
    *,
    final_action: dict[str, Any],
    passes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the unknown-proceed transform with fresh replay state."""

    return _unknown_proceed_transform(
        final_action=final_action,
        passes=passes,
        messages=messages,
        tools=tools or [],
        guard_state=GuardState(),
    )


def replay_loop_breaker_transform(
    *,
    final_action: dict[str, Any],
    passes: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run loop-breaker v2 detectors without mutating live planner state."""

    return _loop_breaker_transform(
        final_action=final_action,
        passes=passes or [],
        messages=messages,
        tools=tools or [],
    )


def _loop_breaker_transform(
    *,
    final_action: dict[str, Any],
    passes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "loop_breaker_fires": 0,
        "loop_breaker_suppressed_repeats": 0,
        "loop_breaker_kind": None,
        "loop_breaker_message_index": None,
        "loop_breaker_agent_similarity": 0.0,
        "loop_breaker_user_similarity": 0.0,
        "loop_breaker_draft_similarity": 0.0,
    }
    signal = _loop_breaker_signal(final_action, messages)
    if signal is None:
        return None, meta
    last_agent = str(signal.pop("last_agent"))
    last_user = str(signal.pop("last_user"))
    meta.update(signal)
    candidate = _loop_breaker_action_candidate(messages, passes, tools)
    subject = _loop_breaker_unknown_subject(last_agent)
    user_words = _loop_breaker_user_words(last_user)
    if candidate is not None:
        confirm = {
            "action": "respond",
            "content": (
                f"The {subject} reads unknown. You asked me to \"{user_words}\". "
                "Shall I proceed anyway?"
            ),
            "_pending_confirmed_action": candidate,
        }
        meta["loop_breaker_kind"] = "confirm"
        return confirm, meta
    meta["loop_breaker_kind"] = "final"
    return {
        "action": "respond",
        "content": (
            f"The {subject} reads unknown. I checked the available readback, "
            "and I cannot determine it from the current data. I will stop here "
            "rather than ask again."
        ),
    }, meta


def _loop_breaker_signal(
    final_action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if final_action.get("action") != "respond":
        return None
    draft = str(final_action.get("content") or "")
    if not draft.strip():
        return None
    agent_items = _prior_assistant_respond_texts(messages)
    user_items = _prior_user_texts(messages)
    if len(agent_items) < 2 or len(user_items) < 2:
        return None
    prev_agent_index, prev_agent = agent_items[-2]
    last_agent_index, last_agent = agent_items[-1]
    prev_user_index, prev_user = user_items[-2]
    last_user_index, last_user = user_items[-1]
    del prev_agent_index, prev_user_index
    agent_similarity = _near_identical_similarity(prev_agent, last_agent)
    user_similarity = _near_identical_similarity(prev_user, last_user)
    draft_similarity = _near_identical_similarity(last_agent, draft)
    if agent_similarity < 0.72 or user_similarity < 0.72 or draft_similarity < 0.72:
        return None
    if _loop_breaker_confirmation_cycle_user(prev_user) or _loop_breaker_confirmation_cycle_user(last_user):
        return None
    if not (_unknown_refusal_text(last_agent) and _unknown_refusal_text(draft)):
        return None
    if _loop_breaker_clean_unavailable_domain(last_agent, last_user):
        return None
    if len(messages) <= 40 and not _loop_breaker_early_id_deadlock(last_agent, last_user):
        return None
    return {
        "loop_breaker_message_index": len(messages),
        "loop_breaker_agent_similarity": round(agent_similarity, 3),
        "loop_breaker_user_similarity": round(user_similarity, 3),
        "loop_breaker_draft_similarity": round(draft_similarity, 3),
        "last_agent": last_agent,
        "last_user": last_user,
        "last_user_index": last_user_index,
    }


def _prior_assistant_respond_texts(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("tool_calls"):
            continue
        text = _message_text(message).strip()
        if text:
            out.append((index, text))
    return out


def _prior_user_texts(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [
        (index, _message_text(message).strip())
        for index, message in enumerate(messages)
        if message.get("role") == "user" and _message_text(message).strip()
    ]


def _near_identical_similarity(left: str, right: str) -> float:
    left_norm = _normalize_for_overlap(left)
    right_norm = _normalize_for_overlap(right)
    if not left_norm or not right_norm:
        return 0.0
    return max(
        _normalized_jaccard(left, right),
        SequenceMatcher(None, left_norm, right_norm).ratio(),
    )


def _unknown_refusal_text(text: str) -> bool:
    lowered = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )
    verify_deadlock = bool(
        re.search(r"\bneed to verify\b", lowered)
        and re.search(r"\bplease confirm\b", lowered)
    )
    unavailable_deadlock = bool(
        re.search(
            r"\b(can't|cannot|unable|unavailable|don't have|do not have|"
            r"can't access|cannot access|can't read|cannot read|can't view|cannot view|"
            r"can't find|cannot find|can't locate|cannot locate|can't see|cannot see)\b",
            lowered,
        )
    )
    asks_for_unavailable_id = bool(
        re.search(r"\bcould you tell me\b", lowered)
        and re.search(r"\b(name|id|waypoint|destination)\b", lowered)
    )
    if (
        "unknown" not in lowered
        and not verify_deadlock
        and not unavailable_deadlock
        and not asks_for_unavailable_id
    ):
        return False
    return bool(
        re.search(
            r"\b(can't|cannot|won't|unable|not able|don't have|do not have|"
            r"can't confirm|cannot confirm|can't verify|cannot verify|"
            r"can't access|cannot access|can't read|cannot read|can't view|cannot view|"
            r"can't find|cannot find|can't locate|cannot locate|can't see|cannot see|"
            r"unavailable|need to verify|please confirm|can't determine|cannot determine|"
            r"could you tell me)\b",
            lowered,
        )
    )


def _loop_breaker_confirmation_cycle_user(text: str) -> bool:
    lowered = _normalize_for_overlap(text)
    if re.match(r"^(yes|yeah|yep|ok|okay|confirming|confirmed)\b", lowered):
        return True
    if re.match(r"^confirm (the )?(route|destination|details)", lowered):
        if re.search(r"\b(route|rll|rlp|loc|poi|monaco|karlsruhe|stuttgart|luxembourg|bonn|brussels|berlin)\b", lowered):
            return True
    return False


def _loop_breaker_clean_unavailable_domain(agent_text: str, user_text: str) -> bool:
    combined = (
        f"{agent_text} {user_text}".lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )
    return bool(
        re.search(r"\b(battery|state of charge|soc|range|charging stop)\b", combined)
        and re.search(r"\b(no tool|don't have a tool|don't have a way|can't read|cannot read|can't determine|cannot determine)\b", combined)
    )


def _loop_breaker_early_id_deadlock(agent_text: str, user_text: str) -> bool:
    agent = _normalize_for_overlap(agent_text)
    user = _normalize_for_overlap(user_text)
    return bool(
        re.search(r"\b(id|waypoint)\b", agent)
        and re.search(r"\b(could you tell me|provide|give me|tell me)\b", agent)
        and re.search(r"\b(look up|access|find|remove|delete|route|waypoint|destination)\b", user)
    )


def _loop_breaker_unknown_subject(text: str) -> str:
    normalized = " ".join(text.replace("‑", "-").split())
    patterns = [
        r"(?:the|my)\s+([A-Za-z0-9 ,/'-]{3,80}?)\s+(?:status|state)\s+(?:is|reads|came back|comes back|reported as)\s+unknown",
        r"([A-Za-z0-9 ,/'-]{3,80}?)\s+(?:is|are)\s+(?:reported as\s+)?unknown",
        r"verify\s+the\s+([A-Za-z0-9 ,/'-]{3,80}?)\s+before",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            subject = _clean_loop_breaker_phrase(match.group(1))
            if subject:
                return subject
    return "requested value"


def _clean_loop_breaker_phrase(text: str) -> str:
    text = re.sub(r"^(the|my|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-")
    if len(text) > 90:
        text = text[:87].rstrip() + "..."
    return text or "requested value"


def _loop_breaker_user_words(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:177].rstrip() + "..."
    return text.replace('"', "'")


def _loop_breaker_action_candidate(
    messages: list[dict[str, Any]],
    passes: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    instructed = _loop_breaker_instructed_tool_candidates(messages, tools)
    for item in passes:
        action = item.get("action") if isinstance(item, dict) else None
        if not isinstance(action, dict) or action.get("action") != "tool_calls":
            continue
        calls = [_normalized_tool_call(call) for call in action.get("tool_calls") or []]
        if _calls_match_instructed_action(calls, instructed):
            return {"action": "tool_calls", "tool_calls": calls}
    if instructed:
        tool_name, args = instructed[0]
        return {
            "action": "tool_calls",
            "tool_calls": [{"tool_name": tool_name, "arguments": args}],
        }
    return None


def _loop_breaker_instructed_tool_candidates(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    candidates = list(_instructed_tool_candidates(messages, tools))
    latest = _latest_user_text(messages).lower()
    available = {str(tool.get("function", {}).get("name") or "") for tool in tools}

    def has_tool(name: str) -> bool:
        return not available or name in available

    if (
        has_tool("navigation_delete_waypoint")
        and re.search(r"\b(remove|delete)\b", latest)
        and re.search(r"\b(intermediate|middle|stop|waypoint|essen)\b", latest)
    ):
        waypoints = _latest_navigation_waypoints(messages)
        if len(waypoints) >= 3:
            candidates.append(("navigation_delete_waypoint", {"waypoint_id": waypoints[1]}))
    deduped: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name, args in candidates:
        key = json.dumps({"name": name, "args": args}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, args))
    return deduped


def _route_disclosure_values(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    toll_ids: set[str] = set()
    fastest_ids: set[str] = set()
    for payload in _tool_success_payloads(
        messages,
        (
            "get_routes",
            "get_routes_from_start_to_destination",
            "get_route_options",
        ),
    ):
        for route in _walk_dicts(payload):
            if "route_id" not in route and "id" not in route:
                continue
            route_id = str(route.get("route_id") or route.get("id") or "")
            if not route_id:
                continue
            if route.get("includes_toll") is True:
                toll_ids.add(route_id)
            alias = route.get("alias") or route.get("aliases") or route.get("label") or ""
            alias_text = " ".join(map(str, alias)) if isinstance(alias, list) else str(alias)
            if (
                "fastest" in alias_text.lower()
                or route.get("fastest") is True
                or route.get("is_fastest") is True
            ):
                fastest_ids.add(route_id)
    return sorted(toll_ids), sorted(fastest_ids)


def _route_disclosure_legs(
    messages: list[dict[str, Any]],
) -> list[dict[str, set[str]]]:
    legs: dict[tuple[str, str], dict[str, set[str]]] = {}
    payloads = _tool_success_payloads(
        messages,
        (
            "get_routes",
            "get_routes_from_start_to_destination",
            "get_route_options",
        ),
    )
    for payload_index, payload in enumerate(payloads):
        for route in _walk_dicts(payload):
            route_id = str(route.get("route_id") or route.get("id") or "")
            if not route_id:
                continue
            start_id = str(route.get("start_id") or "")
            destination_id = str(route.get("destination_id") or "")
            leg_key = (
                (start_id, destination_id)
                if start_id and destination_id
                else (f"payload:{payload_index}", "")
            )
            leg = legs.setdefault(
                leg_key,
                {"route_ids": set(), "fastest_ids": set()},
            )
            leg["route_ids"].add(route_id)
            alias = route.get("alias") or route.get("aliases") or route.get("label") or ""
            alias_text = " ".join(map(str, alias)) if isinstance(alias, list) else str(alias)
            if (
                "fastest" in alias_text.lower()
                or route.get("fastest") is True
                or route.get("is_fastest") is True
            ):
                leg["fastest_ids"].add(route_id)
    return list(legs.values())


def _decision_selected_route_ids(
    content: str,
    messages: list[dict[str, Any]],
    decision_actions: list[dict[str, Any]],
) -> list[str]:
    """Return route IDs selected by text, active execution, or current drafts."""

    known_ids = {
        route_id
        for leg in _route_disclosure_legs(messages)
        for route_id in leg["route_ids"]
    }
    selected = {
        route_id
        for route_id in known_ids
        if re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(route_id)}(?![A-Za-z0-9_-])",
            content,
        )
    }
    selected.update(_most_recent_successful_navigation_route_ids(messages))
    selected.update(_navigation_mutation_route_ids(decision_actions))
    return sorted(selected)


def _navigation_mutation_route_ids(
    actions: list[dict[str, Any]],
) -> set[str]:
    selected: set[str] = set()
    for action in actions:
        if not isinstance(action, dict) or action.get("action") != "tool_calls":
            continue
        for call in action.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            tool_name = str(call.get("tool_name") or function.get("name") or "")
            if tool_name not in _NAV_MUTATING_TOOLS:
                continue
            route_arguments = {
                name
                for name, identifier_types in _TOOL_ID_ARGUMENT_TYPES.get(
                    tool_name, {}
                ).items()
                if "route" in identifier_types
            }
            arguments = _parse_tool_arguments(
                call.get("arguments", function.get("arguments", {}))
            )
            for name in sorted(route_arguments):
                value = arguments.get(name)
                values = value if isinstance(value, list) else [value]
                selected.update(
                    str(item)
                    for item in values
                    if isinstance(item, str) and item
                )
    return selected


def _most_recent_successful_navigation_route_ids(
    messages: list[dict[str, Any]],
) -> set[str]:
    pending: dict[str, dict[str, Any]] = {}
    latest: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                tool_name = str(function.get("name") or call.get("tool_name") or "")
                if tool_name not in _NAV_MUTATING_TOOLS:
                    continue
                call_id = str(call.get("id") or "")
                if call_id:
                    pending[call_id] = {
                        "action": "tool_calls",
                        "tool_calls": [call],
                    }
            continue
        if message.get("role") != "tool" or _tool_success_result(message) is None:
            continue
        action = pending.get(str(message.get("tool_call_id") or ""))
        if action is not None:
            latest = _navigation_mutation_route_ids([action])
    return latest


def _selected_fastest_route_ids(
    messages: list[dict[str, Any]],
    selected_ids: list[str],
) -> list[str]:
    selected = set(selected_ids)
    return sorted(
        {
            route_id
            for leg in _route_disclosure_legs(messages)
            for route_id in leg["fastest_ids"] & selected
        }
    )


def _tool_success_payloads(
    messages: list[dict[str, Any]],
    tool_name: str | tuple[str, ...],
) -> list[Any]:
    names = {tool_name} if isinstance(tool_name, str) else set(tool_name)
    payloads: list[Any] = []
    for message in messages:
        if message.get("role") != "tool" or str(message.get("name") or "") not in names:
            continue
        result = _tool_success_result(message)
        if result is not None:
            payloads.append(result)
    return payloads


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for item in value.values():
            found.extend(_walk_dicts(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_dicts(item))
    return found


def _temperature_difference_disclosure(messages: list[dict[str, Any]]) -> str | None:
    temps: dict[str, float] = {}
    latest_disclosure: str | None = None
    for message in messages:
        if message.get("role") == "tool" and str(message.get("name") or "") in {
            "get_climate_settings",
            "get_temperature_inside_car",
            "set_climate_temperature",
        }:
            result = _tool_success_result(message)
            if result is not None:
                temps.update(_extract_zone_temperatures(result))
        if message.get("role") != "assistant":
            continue
        event_temps = dict(temps)
        set_seen = False
        for raw_call in _assistant_message_tool_calls(message):
            function = raw_call.get("function") or {}
            if str(function.get("name") or "") != "set_climate_temperature":
                continue
            args = _parse_tool_call_arguments(function.get("arguments") or {})
            temp = _argument_temperature(args)
            if temp is None:
                continue
            set_seen = True
            zone = _argument_zone(args)
            if zone in {"ALL", "FRONT", ""}:
                if event_temps:
                    for key in list(event_temps):
                        event_temps[key] = temp
                else:
                    event_temps["ALL"] = temp
            else:
                event_temps[zone] = temp
        if set_seen:
            latest_disclosure = _temperature_difference_text(event_temps) or latest_disclosure
            temps = event_temps
    return latest_disclosure


def _temperature_difference_text(temps: dict[str, float]) -> str | None:
    if len(temps) < 2:
        return None
    values = list(temps.values())
    if max(values) - min(values) <= 3:
        return None
    ordered = ", ".join(f"{key.lower()} {value:g} deg C" for key, value in sorted(temps.items()))
    return f"The climate zones now differ by {max(values) - min(values):g} deg C ({ordered})."


def _temperature_difference_already_disclosed(normalized_content: str) -> bool:
    if re.search(r"\b(differ|different|more than\s+3|greater than\s+3)\b", normalized_content):
        return True
    return "driver" in normalized_content and "passenger" in normalized_content


def _extract_zone_temperatures(value: Any) -> dict[str, float]:
    temps: dict[str, float] = {}
    for item in _walk_dicts(value):
        for key, temp in item.items():
            lowered = str(key).lower()
            if "temperature" not in lowered:
                continue
            parsed = _coerce_float(temp)
            if parsed is None:
                continue
            zone = lowered.replace("temperature", "").replace("_", " ").strip()
            zone = zone.replace("climate", "").strip()
            zone = zone.upper().replace(" ", "_") or "CABIN"
            temps[zone] = parsed
    return temps


def _argument_temperature(arguments: dict[str, Any]) -> float | None:
    for key in ("temperature", "temperature_celsius", "temperature_c"):
        if key in arguments:
            return _coerce_float(arguments.get(key))
    return None


def _argument_zone(arguments: dict[str, Any]) -> str:
    for key in ("zone", "seat_zone", "climate_zone"):
        if key in arguments:
            return str(arguments.get(key) or "").upper()
    return ""


def _climate_temperature_preread_action(
    action: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    guard_state: GuardState,
) -> dict[str, Any] | None:
    if not _tool_exists(tools, "get_temperature_inside_car"):
        return None
    if len(_known_climate_zone_temperatures(messages)) >= 2:
        return None
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        if str(call.get("tool_name") or "") != "set_climate_temperature":
            continue
        args = call.get("arguments") or {}
        zone = _argument_zone(args)
        if zone in {"", "ALL", "FRONT"}:
            continue
        site = f"autpol012:{_tool_call_sort_key(call)}"
        if site in guard_state.emitted_cascade_sites:
            continue
        guard_state.emitted_cascade_sites.add(site)
        return {
            "action": "tool_calls",
            "tool_calls": [{"tool_name": "get_temperature_inside_car", "arguments": {}}],
        }
    return None


def _known_climate_zone_temperatures(messages: list[dict[str, Any]]) -> dict[str, float]:
    temps: dict[str, float] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        if name not in {
            "get_climate_settings",
            "get_temperature_inside_car",
            "set_climate_temperature",
        }:
            continue
        result = _tool_success_result(message)
        if result is not None:
            temps.update(_extract_zone_temperatures(result))
    return temps


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _normalize_12h_times(text: str) -> tuple[str, int]:
    fixes = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal fixes
        hour = int(match.group(1))
        minute_part = match.group(2) or ""
        marker = match.group(3).lower().replace(".", "")
        minute = minute_part[1:] if minute_part else "00"
        if marker == "pm" and hour != 12:
            hour += 12
        elif marker == "am" and hour == 12:
            hour = 0
        fixes += 1
        return f"{hour:02d}:{minute}"

    return OUTPUT_GUARD_TIME_PATTERN.sub(_replace, text), fixes


def _rewrite_settled_reopen_question(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    content = str(action.get("content") or "")
    if "?" not in content:
        return None
    normalized = _normalize_for_overlap(content)
    settled = _successful_set_tool_calls(messages)
    for call in reversed(settled):
        tool_name = str(call.get("tool_name", ""))
        family = _settled_tool_family(tool_name)
        if family is None or not _settled_question_mentions_family(normalized, family):
            continue
        return {
            "action": "respond",
            "content": f"Done: {tool_name} with {json.dumps(call.get('arguments') or {}, sort_keys=True)}.",
        }
    return None


def _force_or_apply_stored_preference(
    action: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _latest_user_requested_stored_preference(messages):
        return None
    if not _action_touches_preference_topic(action):
        return None
    if not _has_successful_tool_result(messages, "get_user_preferences"):
        if _tool_exists(tools, "get_user_preferences"):
            return {
                "action": "tool_calls",
                "tool_calls": [{"tool_name": "get_user_preferences", "arguments": {}}],
            }
        return None
    level = _seat_heating_preference_level(messages)
    if level is None:
        return None
    rewritten_calls: list[dict[str, Any]] = []
    changed = False
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        name = str(call.get("tool_name", ""))
        args = dict(call.get("arguments") or {})
        if name in {"set_seat_heating", "set_steering_wheel_heating"}:
            if args.get("level") != level:
                args["level"] = level
                changed = True
            call["arguments"] = args
        rewritten_calls.append(call)
    if not changed:
        return None
    return {"action": "tool_calls", "tool_calls": rewritten_calls}


def _rewrite_preference_question_to_apply(
    action: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _latest_user_requested_stored_preference(messages):
        return None
    if not _has_successful_tool_result(messages, "get_user_preferences"):
        if _tool_exists(tools, "get_user_preferences"):
            return {
                "action": "tool_calls",
                "tool_calls": [{"tool_name": "get_user_preferences", "arguments": {}}],
            }
        return None
    content = _normalize_for_overlap(str(action.get("content") or ""))
    if not re.search(r"\b(seat|steering|heat|heating|level|preference)\b", content):
        return None
    level = _seat_heating_preference_level(messages)
    if level is None:
        return None
    latest_user = _latest_user_text(messages)
    calls: list[dict[str, Any]] = []
    temperature = _temperature_from_text(latest_user)
    if temperature is not None and _tool_exists(tools, "set_climate_temperature"):
        calls.append(
            {
                "tool_name": "set_climate_temperature",
                "arguments": {"temperature": temperature, "seat_zone": "DRIVER"},
            }
        )
    if _tool_exists(tools, "set_seat_heating"):
        calls.append(
            {
                "tool_name": "set_seat_heating",
                "arguments": {"level": level, "seat_zone": "DRIVER"},
            }
        )
    if _tool_exists(tools, "set_steering_wheel_heating"):
        calls.append(
            {
                "tool_name": "set_steering_wheel_heating",
                "arguments": {"level": level},
            }
        )
    if not calls:
        return None
    return {"action": "tool_calls", "tool_calls": calls}


def _latest_user_requested_stored_preference(messages: list[dict[str, Any]]) -> bool:
    text = _latest_user_text(messages).lower()
    return bool(
        re.search(
            r"\b(stored preference|stored preferences|my preference|my preferences|"
            r"preferred|usual|saved setting|saved settings|my settings)\b",
            text,
        )
    )


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return ""


def _action_touches_preference_topic(action: dict[str, Any]) -> bool:
    if action.get("action") == "respond":
        return bool(
            re.search(
                r"\b(preference|preferred|seat|steering|heat|heating|route|toll|navigation)\b",
                str(action.get("content") or "").lower(),
            )
        )
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name", "")).lower()
        if re.search(r"\b(seat|steering|heat|climate|navigation|route)\b", name.replace("_", " ")):
            return True
    return False


def _has_successful_tool_result(messages: list[dict[str, Any]], name: str) -> bool:
    return any(
        message.get("role") == "tool"
        and str(message.get("name") or "") == name
        and _tool_success_result(message) is not None
        for message in messages
    )


def _tool_exists(tools: list[dict[str, Any]], name: str) -> bool:
    return name in _tools_by_name(tools)


def _seat_heating_preference_level(messages: list[dict[str, Any]]) -> int | None:
    text = _preference_result_text(messages)
    patterns = [
        r"seat\s+heating[^0-9]{0,80}level\s*(\d)",
        r"heating[^0-9]{0,80}level\s*(\d)",
        r"level\s*(\d)[^.;]{0,80}seat\s+heating",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _preference_result_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        if message.get("role") != "tool" or str(message.get("name") or "") != "get_user_preferences":
            continue
        result = _tool_success_result(message)
        if result is not None:
            chunks.append(json.dumps(result, sort_keys=True, default=str))
            chunks.append(str(result))
    return "\n".join(chunks)


def _temperature_from_text(text: str) -> int | None:
    match = re.search(r"(?<!\d)(1[6-9]|2[0-9]|3[0-2])\s*(?:deg|degree|degrees|c|celsius|°)?\b", text.lower())
    if not match:
        return None
    return int(match.group(1))


def _successful_set_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results_by_id = _tool_messages_by_call_id(messages)
    calls: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw_call in _assistant_message_tool_calls(message):
            call_id = raw_call.get("id")
            if call_id is None:
                continue
            result = results_by_id.get(str(call_id))
            if result is None or _tool_success_result(result) is None:
                continue
            function = raw_call.get("function") or {}
            name = str(function.get("name") or "")
            if not _settled_tool_family(name):
                continue
            calls.append(
                {
                    "tool_name": name,
                    "arguments": _parse_tool_call_arguments(function.get("arguments") or {}),
                }
            )
    return calls


def _settled_tool_family(tool_name: str) -> str | None:
    lowered = tool_name.lower()
    if "fan_airflow_direction" in lowered:
        return "airflow"
    if "fan_speed" in lowered:
        return "fan_speed"
    if "window" in lowered:
        return "window"
    if "temperature" in lowered or "climate" in lowered:
        return "climate"
    if "seat" in lowered and "heating" in lowered:
        return "seat_heating"
    if "steering" in lowered and "heating" in lowered:
        return "steering_heating"
    if "navigation" in lowered or "route" in lowered:
        return "navigation"
    if lowered.startswith("set_") or lowered.startswith("open_close_"):
        return lowered
    return None


def _settled_question_mentions_family(normalized: str, family: str) -> bool:
    tokens_by_family = {
        "airflow": ("airflow", "direction", "windshield", "feet", "head", "fan"),
        "fan_speed": ("fan", "speed"),
        "window": ("window", "open", "close"),
        "climate": ("temperature", "climate", "cabin"),
        "seat_heating": ("seat", "heat", "heating", "level"),
        "steering_heating": ("steering", "wheel", "heat", "heating", "level"),
        "navigation": ("route", "navigation", "toll", "destination"),
    }
    tokens = tokens_by_family.get(family, tuple(family.replace("_", " ").split()))
    return any(token in normalized for token in tokens)


def _is_confirmation_request(content: str) -> bool:
    normalized = content.strip().lower()
    if normalized.endswith("?"):
        return bool(
            re.search(
                r"\b(confirm|confirmation|proceed|okay|ok|yes|approve|should i|"
                r"shall i|may i|can i|do you want|would you like)\b",
                normalized,
            )
        )
    return bool(re.search(r"\b(confirm|confirmation)\b", normalized))


def _mandated_ask_completion_meta(
    *,
    action: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    completion_notices: int,
    completion_respond_deferrals: int,
    completion_pivotal_veto: bool,
) -> dict[str, int]:
    mandated = _is_mandated_confirmation_ask(action, messages, tools)
    influenced = mandated and bool(
        completion_notices
        or completion_respond_deferrals
        or completion_pivotal_veto
    )
    return {
        "mandated_ask_decisions": int(mandated),
        "mandated_ask_completion_influence": int(influenced),
    }


def _is_mandated_confirmation_ask(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    if action.get("action") != "respond":
        return False
    content = str(action.get("content") or "")
    if not _is_confirmation_request(content):
        return False
    lowered = content.casefold()
    requires = _requires_confirmation_tool_names(tools)
    if any(name.casefold() in lowered for name in requires):
        return True
    parsed = _parse_confirmed_tool_call(content)
    if parsed is not None and parsed[0] in requires:
        return True
    if re.search(
        r"\b(energy inefficien|weather|rain|snow|hail|thunder|unsafe|safety)\b",
        lowered,
    ):
        return True
    for subject in ("sunroof", "fog"):
        if subject in lowered and _policy_approval_hint(messages, subject):
            return True
    return False


def _select_requires_confirmation_intent(
    passes: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    requires = _requires_confirmation_tool_names(tools)
    if not requires:
        return None
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for item in passes:
        action = item.get("action") or {}
        if not isinstance(action, dict) or action.get("action") != "tool_calls":
            continue
        confidence = float(item.get("confidence", 0.5) or 0.5)
        for call in action.get("tool_calls") or []:
            if str(call.get("tool_name", "")) not in requires:
                continue
            normalized = _normalized_tool_call(call)
            candidates.append((confidence, _tool_call_sort_key(normalized), normalized))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _requires_confirmation_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function", {}) or {}
        description = str(function.get("description", "")).lstrip()
        if not description.upper().startswith("REQUIRES_CONFIRMATION"):
            continue
        name = str(function.get("name", ""))
        if name:
            names.add(name)
    return names


_ANNOUNCED_CALL_RE = re.compile(
    r"\b(?:i['\u2019]ll|i\s+will)\s+call\s+([a-z][a-z0-9_]*)\s+with\b",
    re.IGNORECASE,
)
_ANNOUNCED_EMAIL_RE = re.compile(
    r"\b(?:i['\u2019]ll|i\s+will)\s+send\s+the\s+email\b",
    re.IGNORECASE,
)


def _announced_call_execution(
    *,
    action: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    state: AnnouncedCallState,
    exclude_non_proceed_interrogative: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    if action.get("action") != "respond" or state.executions >= 2:
        return None, None
    content = str(action.get("content") or "")
    parsed = _parse_announced_tool_call(content, tools)
    if parsed is None:
        return None, None
    tool_name, arguments = parsed
    if exclude_non_proceed_interrogative and _announcement_has_non_proceed_question(
        content, tool_name, arguments
    ):
        return None, None
    available = _tools_by_name(tools)
    tool = available.get(tool_name)
    if tool is None:
        return None, None
    if (
        tool_name in _requires_confirmation_tool_names(tools)
        and not _latest_user_approved_announced_subject(tool_name, messages)
    ):
        return None, None
    subject = _completion_subject_for_tool(tool_name)
    if subject and _policy_approval_hint(messages, subject):
        return None, None
    if _validate_json_schema(arguments, _tool_parameters(tool)):
        return None, None
    grounding_text = (
        _user_and_tool_grounding_text(messages)
        + "\n"
        + content
        + "\n"
        + _announced_argument_text(arguments)
    )
    grounded, blocking = _grounded_arguments(
        arguments, _tool_parameters(tool), grounding_text
    )
    if blocking or grounded != arguments:
        return None, None
    call = {"tool_name": tool_name, "arguments": arguments}
    if _tool_call_already_executed(call, messages):
        return None, None
    return {"action": "tool_calls", "tool_calls": [call]}, content


def _announcement_has_non_proceed_question(
    content: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> bool:
    remaining = content
    brace = remaining.find("{")
    if brace >= 0:
        try:
            _, end = json.JSONDecoder().raw_decode(remaining[brace:])
        except json.JSONDecodeError:
            pass
        else:
            remaining = remaining[:brace] + remaining[brace + end :]
    for value in _walk_argument_strings(arguments):
        if value:
            remaining = remaining.replace(value, "")
    questions = re.findall(r"(?:^|[.!\n]\s*)([^?\n]*\?)", remaining)
    for question in questions:
        normalized = re.sub(r"\s+", " ", question).strip().casefold()
        if _proceed_question(normalized, tool_name):
            continue
        return True
    return False


def _walk_argument_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _walk_argument_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _walk_argument_strings(nested)]
    return []


def _proceed_question(question: str, tool_name: str) -> bool:
    action_words = {"proceed", "confirm", "confirmation", "go ahead"}
    if tool_name == "send_email":
        action_words.update({"send it", "send this", "send the email"})
    if any(word in question for word in action_words):
        return not re.search(
            r"\b(which|what|where|when|who|how|include|address|recipient|"
            r"subject|content|message|detail|format|option)\b",
            question,
        )
    return False


def _announced_argument_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(_announced_argument_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_announced_argument_text(item) for item in value)
    return str(value)


def _latest_user_approved_announced_subject(
    tool_name: str, messages: list[dict[str, Any]]
) -> bool:
    latest = _latest_user_text(messages).casefold()
    if not re.search(r"\b(yes|approved?|proceed|go ahead|send it|do it|send .* now)\b", latest):
        return False
    subject = _completion_subject_for_tool(tool_name)
    if subject is None:
        return False
    prior_assistant = "\n".join(
        _message_text(message).casefold()
        for message in messages
        if message.get("role") == "assistant"
    )
    subject_words = {
        "message": ("email", "message", "text", "call"),
        "navigation": ("route", "destination", "navigation"),
        "window": ("window",),
        "climate": ("climate", "temperature", "fan", "air conditioning"),
        "lights": ("light", "beam"),
        "fog": ("fog",),
        "sunroof": ("sunroof", "sunshade"),
    }.get(subject, (subject,))
    return any(word in latest or word in prior_assistant for word in subject_words)


def _parse_announced_tool_call(
    content: str, tools: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]] | None:
    match = _ANNOUNCED_CALL_RE.search(content)
    if match:
        tool_name = match.group(1)
        tool = _tools_by_name(tools).get(tool_name)
        if tool is None:
            return None
        tail = content[match.end() :].lstrip(" :-\u2014")
        arguments = _parse_json_object_prefix(tail)
        if arguments is None:
            arguments = _parse_named_json_arguments(
                tail, _tool_parameters(tool)
            )
        if arguments is not None:
            return tool_name, arguments
    if _ANNOUNCED_EMAIL_RE.search(content):
        arguments = _parse_announced_email(content)
        if arguments is not None and "send_email" in _tools_by_name(tools):
            return "send_email", arguments
    return None


def _parse_json_object_prefix(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_named_json_arguments(
    text: str, schema: dict[str, Any]
) -> dict[str, Any] | None:
    required = list(schema.get("required") or [])
    if not required:
        return None
    decoder = json.JSONDecoder()
    arguments: dict[str, Any] = {}
    for key in required:
        match = re.search(rf"\b{re.escape(str(key))}\s*:\s*", text)
        if match is None:
            return None
        try:
            value, _ = decoder.raw_decode(text[match.end() :])
        except json.JSONDecodeError:
            return None
        arguments[str(key)] = value
    return arguments


def _parse_announced_email(content: str) -> dict[str, Any] | None:
    recipient = re.search(
        r"(?:^|\n)\s*-?\s*To:\s*([^\s,;]+@[^\s,;]+)",
        content,
        re.IGNORECASE,
    )
    if recipient is None:
        recipient = re.search(
            r"\bsend\s+the\s+email\s+to\s+([^\s,;]+@[^\s,;]+)",
            content,
            re.IGNORECASE,
        )
    body = re.search(
        r"(?:^|\n)\s*-?\s*(?:Message|Content):\s*\n?(.*)$",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if body is None:
        body = re.search(
            r"\bwith\s+the\s+following\s+content:\s*\n+(.*)$",
            content,
            re.IGNORECASE | re.DOTALL,
        )
    if recipient is None or body is None:
        return None
    message = re.sub(
        r"\n+\s*(?:Please\s+confirm|Shall\s+I|Would\s+you\s+like\s+me)[\s\S]*$",
        "",
        body.group(1),
        flags=re.IGNORECASE,
    ).strip()
    address = recipient.group(1).rstrip(".>")
    if not address or not message:
        return None
    return {"email_addresses": [address], "content_message": message}


def _completion_subject_for_tool(tool_name: str) -> str | None:
    for subject, names in _COMPLETION_SUBJECT_TOOLS.items():
        if tool_name in names:
            return subject
    return None


def _tool_call_already_executed(
    call: dict[str, Any], messages: list[dict[str, Any]]
) -> bool:
    fingerprint = _tool_call_sort_key(_normalized_tool_call(call))
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw_call in _assistant_message_tool_calls(message):
            function = raw_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            prior = {
                "tool_name": function.get("name"),
                "arguments": arguments,
            }
            if _tool_call_sort_key(_normalized_tool_call(prior)) == fingerprint:
                return True
    return False


def _response_mentions_call_arguments(content: str, call: dict[str, Any]) -> bool:
    normalized = _normalize_for_overlap(content)
    arguments = call.get("arguments") or {}
    if not arguments:
        return str(call.get("tool_name", "")).replace("_", " ").lower() in normalized
    for key, value in sorted(arguments.items()):
        key_text = str(key).replace("_", " ").lower()
        value_text = _argument_value_text(value)
        if key_text not in normalized:
            return False
        if value_text and value_text not in normalized:
            return False
    return True


def _format_tool_call_details(call: dict[str, Any]) -> str:
    tool_name = str(call.get("tool_name", ""))
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict) or not arguments:
        return tool_name
    arg_text = ", ".join(
        f"{key}: {json.dumps(value, sort_keys=True)}"
        for key, value in sorted(arguments.items())
    )
    return f"{tool_name} with {arg_text}"


def _intercept_requires_confirmation_call(
    action: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    guard_state: GuardState,
) -> dict[str, Any] | None:
    requires = _requires_confirmation_tool_names(tools)
    if not requires:
        return None
    for call in action.get("tool_calls") or []:
        normalized = _normalized_tool_call(call)
        if str(normalized.get("tool_name", "")) not in requires:
            continue
        call_site = _tool_call_sort_key(normalized)
        if call_site in guard_state.confirmed_call_sites:
            return None
        if _assistant_already_confirmed_call_details(messages, normalized):
            return None
        guard_state.confirmed_call_sites.add(call_site)
        return {
            "action": "respond",
            "content": (
                "To confirm: I'll call "
                f"{normalized.get('tool_name')} with "
                f"{json.dumps(normalized.get('arguments') or {}, sort_keys=True)}. "
                "Shall I proceed?"
            ),
        }
    return None


def _assistant_already_confirmed_call_details(
    messages: list[dict[str, Any]],
    call: dict[str, Any],
) -> bool:
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        if not _is_affirmation_message(_message_text(message)):
            start = index + 1
            break
    tool_name = str(call.get("tool_name", ""))
    arguments = call.get("arguments") or {}
    for message in messages[start:]:
        if message.get("role") != "assistant":
            continue
        content = _message_text(message)
        normalized = _normalize_for_overlap(content)
        if tool_name and tool_name.lower() in content.lower():
            return True
        if arguments and _content_mentions_scalar_values(normalized, arguments):
            return True
    return False


def _confirmed_tool_call_from_latest_affirmation(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    guard_state: GuardState,
) -> dict[str, Any] | None:
    if not messages or messages[-1].get("role") != "user":
        return None
    latest_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            latest_user_index = index
            break
    if latest_user_index is None:
        return None
    if not _is_affirmation_message(_message_text(messages[latest_user_index])):
        return None

    available = {str(tool.get("function", {}).get("name") or "") for tool in tools}
    for message in reversed(messages[:latest_user_index]):
        if message.get("role") == "user":
            break
        if message.get("role") != "assistant":
            continue
        content = _message_text(message)
        if not re.search(r"\b(confirm|confirmed|proceed|shall i|please confirm)\b", content, re.I):
            continue
        parsed = _parse_confirmed_tool_call(content)
        if parsed is None:
            continue
        tool_name, args = parsed
        if available and tool_name not in available:
            continue
        normalized = {"tool_name": tool_name, "arguments": args}
        guard_state.confirmed_call_sites.add(_tool_call_sort_key(normalized))
        return {"action": "tool_calls", "tool_calls": [normalized]}
    return None


def _parse_confirmed_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    match = re.search(r"\bcall(?:ing)?\s+([A-Za-z_][A-Za-z0-9_]*)\s+with\s+", content)
    if match is None:
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+with\s+", content)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    json_start = content.find("{", match.end())
    if json_start < 0:
        return None
    try:
        args, _ = decoder.raw_decode(content[json_start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return match.group(1), args


def _is_affirmation_message(text: str) -> bool:
    normalized = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    words = normalized.split()
    if not words or len(words) > 4:
        return False
    return bool(
        re.fullmatch(
            r"(yes|yeah|yep|confirm|confirmed|sure|ok|okay|please|do|proceed|go|ahead|"
            r"please do|go ahead|do it|that works)(\s+(yes|yeah|yep|confirm|sure|ok|okay|"
            r"please|do|proceed|go|ahead|it|works))*",
            normalized,
        )
    )


def _content_mentions_scalar_values(normalized_content: str, arguments: dict[str, Any]) -> bool:
    scalar_values = [
        _argument_value_text(value)
        for value in arguments.values()
        if isinstance(value, (str, int, float, bool))
    ]
    scalar_values = [value for value in scalar_values if value]
    return bool(scalar_values) and all(value in normalized_content for value in scalar_values)


def _expand_autpol_cascade(
    action: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    guard_state: GuardState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _observed_autpol_state(messages)
    calls = [_normalized_tool_call(call) for call in action.get("tool_calls") or []]
    expanded = [dict(call) for call in calls]
    rules: list[str] = []
    schema_skips: list[dict[str, str]] = []
    poisoned_tools = _removed_marker_tool_names(messages)

    def add_call(
        rule: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        subject: str | None = None,
    ) -> None:
        del subject
        call = {"tool_name": tool_name, "arguments": arguments}
        if _has_tool_call(expanded, tool_name, arguments):
            return
        schema_issue = _cascade_schema_issue(
            tools=tools,
            tool_name=tool_name,
            arguments=arguments,
            poisoned_tools=poisoned_tools,
        )
        if schema_issue is not None:
            schema_skips.append({"rule": rule, "tool": tool_name, "reason": schema_issue})
            return
        expanded.append(call)
        rules.append(rule)

    def add_or_replace_call(
        rule: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        subject: str | None = None,
    ) -> None:
        del subject
        schema_issue = _cascade_schema_issue(
            tools=tools,
            tool_name=tool_name,
            arguments=arguments,
            poisoned_tools=poisoned_tools,
        )
        if schema_issue is not None:
            schema_skips.append({"rule": rule, "tool": tool_name, "reason": schema_issue})
            return
        for existing in expanded:
            if str(existing.get("tool_name", "")) != tool_name:
                continue
            existing["arguments"] = dict(arguments)
            rules.append(rule)
            return
        add_call(rule, tool_name, arguments, subject=subject)

    pending_autpol010 = "autpol010:pending_airflow_read"
    done_autpol010 = "autpol010:airflow_done"
    if (
        pending_autpol010 in guard_state.emitted_cascade_sites
        and done_autpol010 not in guard_state.emitted_cascade_sites
        and state["climate_read"]
    ):
        airflow_direction = _autpol_windshield_airflow_direction(
            state.get("fan_airflow_direction"),
            tools,
        )
        if airflow_direction:
            add_or_replace_call(
                "AUT-POL:010",
                "set_fan_airflow_direction",
                {"direction": airflow_direction},
                subject="airflow",
            )
            guard_state.emitted_cascade_sites.add(done_autpol010)

    original_and_appended_index = 0
    while original_and_appended_index < len(expanded):
        call = expanded[original_and_appended_index]
        original_and_appended_index += 1
        tool_name = str(call.get("tool_name", ""))
        arguments = call.get("arguments") or {}
        site = f"{tool_name}:{_tool_call_sort_key(call)}"

        if (
            tool_name == "set_window_defrost"
            and bool(arguments.get("on")) is True
            and str(arguments.get("defrost_window", "")).upper() in {"FRONT", "ALL"}
            and f"autpol010:{site}" not in guard_state.emitted_cascade_sites
        ):
            guard_state.emitted_cascade_sites.add(f"autpol010:{site}")
            if not state["climate_read"]:
                add_call("AUT-POL:010", "get_climate_settings", {})
            if not _known_number_at_least(state.get("fan_speed"), 2):
                add_call("AUT-POL:010", "set_fan_speed", {"level": 2}, subject="fan speed")
            if state["climate_read"]:
                airflow_direction = _autpol_windshield_airflow_direction(
                    state.get("fan_airflow_direction"),
                    tools,
                )
                if airflow_direction:
                    add_or_replace_call(
                        "AUT-POL:010",
                        "set_fan_airflow_direction",
                        {"direction": airflow_direction},
                        subject="airflow",
                    )
                    guard_state.emitted_cascade_sites.add(done_autpol010)
            else:
                guard_state.emitted_cascade_sites.add(pending_autpol010)
            if state.get("air_conditioning") is not True:
                add_call(
                    "AUT-POL:010",
                    "set_air_conditioning",
                    {"on": True},
                    subject="air conditioning",
                )

        if (
            tool_name == "set_air_conditioning"
            and bool(arguments.get("on")) is True
            and f"autpol011:{site}" not in guard_state.emitted_cascade_sites
        ):
            guard_state.emitted_cascade_sites.add(f"autpol011:{site}")
            if not state["windows_read"]:
                add_call("AUT-POL:011", "get_vehicle_window_positions", {})
            if _windows_need_close(state):
                add_call(
                    "AUT-POL:011",
                    "open_close_window",
                    {"window": "ALL", "percentage": 0},
                    subject="window",
                )
            if not _has_tool_call(expanded, "set_fan_speed", {"level": 2}) and _known_number_equal(state.get("fan_speed"), 0):
                add_call("AUT-POL:011", "set_fan_speed", {"level": 1}, subject="fan speed")

        if (
            tool_name == "set_fog_lights"
            and bool(arguments.get("on")) is True
            and f"autpol013:{site}" not in guard_state.emitted_cascade_sites
        ):
            guard_state.emitted_cascade_sites.add(f"autpol013:{site}")
            if not state["exterior_read"]:
                add_call("AUT-POL:013", "get_exterior_lights_status", {})
            if state.get("head_lights_low_beams") is not True:
                add_call("AUT-POL:013", "set_head_lights_low_beams", {"on": True})
            if state.get("head_lights_high_beams") is not False:
                add_call("AUT-POL:013", "set_head_lights_high_beams", {"on": False})

    return (
        {"action": "tool_calls", "tool_calls": expanded},
        {
            "cascade_expansions": max(0, len(expanded) - len(calls)),
            "cascade_rules": rules,
            "cascade_schema_skips": len(schema_skips),
            "cascade_schema_skip_details": schema_skips,
        },
    )


def _observed_autpol_state(messages: list[dict[str, Any]]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "climate_read": False,
        "windows_read": False,
        "exterior_read": False,
        "fan_speed": None,
        "fan_airflow_direction": None,
        "air_conditioning": None,
        "window_positions": [],
        "head_lights_low_beams": None,
        "head_lights_high_beams": None,
    }
    for message in messages:
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        payload = _tool_success_result(message)
        if not isinstance(payload, dict):
            continue
        if name == "get_climate_settings":
            state["climate_read"] = True
            state["fan_speed"] = payload.get("fan_speed", state["fan_speed"])
            state["fan_airflow_direction"] = payload.get(
                "fan_airflow_direction",
                state["fan_airflow_direction"],
            )
            state["air_conditioning"] = payload.get("air_conditioning", state["air_conditioning"])
        elif name == "get_vehicle_window_positions":
            state["windows_read"] = True
            state["window_positions"] = [
                payload.get("window_driver_position"),
                payload.get("window_passenger_position"),
                payload.get("window_driver_rear_position"),
                payload.get("window_passenger_rear_position"),
            ]
        elif name == "get_exterior_lights_status":
            state["exterior_read"] = True
            state["head_lights_low_beams"] = payload.get(
                "head_lights_low_beams",
                state["head_lights_low_beams"],
            )
            state["head_lights_high_beams"] = payload.get(
                "head_lights_high_beams",
                state["head_lights_high_beams"],
            )
    return state


def _tool_success_result(message: dict[str, Any]) -> Any:
    content = message.get("content")
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or str(payload.get("status", "")).upper() != "SUCCESS":
        return None
    return payload.get("result")


def _removed_marker_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    poisoned: set[str] = set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = _message_text(message).lower()
        if "removed" not in content:
            continue
        name = str(message.get("name") or "")
        if name:
            poisoned.add(name)
    return poisoned


def _cascade_schema_issue(
    *,
    tools: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
    poisoned_tools: set[str],
) -> str | None:
    if tool_name in poisoned_tools:
        return "removed_marker_poisoned"
    schema = _tool_parameter_schema(tools, tool_name)
    if schema is None:
        return "tool_missing"
    properties = _schema_properties(schema)
    for key in arguments:
        if key not in properties:
            return f"argument_missing:{key}"
    return None


def _tool_parameter_schema(tools: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function", {}) or {}
        if str(function.get("name", "")) != tool_name:
            continue
        parameters = function.get("parameters")
        return parameters if isinstance(parameters, dict) else {}
    return None


def _schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _autpol_windshield_airflow_direction(
    current_direction: Any,
    tools: list[dict[str, Any]],
) -> str | None:
    current = str(current_direction or "").upper().strip()
    if "WINDSHIELD" in current:
        return None
    valid = _valid_tool_enum_values(tools, "set_fan_airflow_direction", "direction")
    if not valid:
        valid = {
            "HEAD",
            "FEET",
            "HEAD_FEET",
            "WINDSHIELD",
            "WINDSHIELD_FEET",
            "WINDSHIELD_HEAD",
            "WINDSHIELD_HEAD_FEET",
        }
    current_parts = {part for part in current.split("_") if part}
    required = set(current_parts)
    required.add("WINDSHIELD")
    candidates = [
        value
        for value in valid
        if required.issubset({part for part in str(value).upper().split("_") if part})
    ]
    if candidates:
        return sorted(candidates, key=lambda value: (len(str(value).split("_")), str(value)))[0]
    return "WINDSHIELD" if "WINDSHIELD" in valid else None


def _valid_tool_enum_values(
    tools: list[dict[str, Any]],
    tool_name: str,
    argument_name: str,
) -> set[str]:
    for tool in tools:
        function = tool.get("function", {}) or {}
        if str(function.get("name", "")) != tool_name:
            continue
        parameters = function.get("parameters") or {}
        values = _enum_values_for_schema_key(parameters, argument_name)
        return {str(value).upper() for value in values}
    return set()


def _enum_values_for_schema_key(schema: Any, key: str) -> list[Any]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(properties.get(key), dict):
        enum_values = properties[key].get("enum")
        if isinstance(enum_values, list):
            return enum_values
    for value in schema.values():
        if isinstance(value, dict):
            found = _enum_values_for_schema_key(value, key)
            if found:
                return found
        elif isinstance(value, list):
            for item in value:
                found = _enum_values_for_schema_key(item, key)
                if found:
                    return found
    return []


def _has_tool_call(calls: list[dict[str, Any]], tool_name: str, arguments: dict[str, Any]) -> bool:
    expected = _normalize_json(arguments)
    return any(
        str(call.get("tool_name", "")) == tool_name
        and _normalize_json(call.get("arguments") or {}) == expected
        for call in calls
    )


def _known_number_at_least(value: Any, threshold: float) -> bool:
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def _known_number_equal(value: Any, expected: float) -> bool:
    try:
        return float(value) == expected
    except (TypeError, ValueError):
        return False


def _windows_need_close(state: dict[str, Any]) -> bool:
    if not state.get("windows_read"):
        return True
    positions = state.get("window_positions") or []
    if not positions:
        return True
    for value in positions:
        try:
            if float(value) > 20:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _argument_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).lower()
    if isinstance(value, str):
        return value.replace("_", " ").lower()
    return json.dumps(value, sort_keys=True).replace("_", " ").lower()


def _tool_call_sort_key(call: dict[str, Any]) -> str:
    return json.dumps(call, sort_keys=True, separators=(",", ":"))


def _repeat_nudge_notice(messages: list[dict[str, Any]]) -> str | None:
    latest_user_index = None
    latest_user = ""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user":
            latest_user_index = index
            latest_user = _message_text(message)
            break
    if latest_user_index is None or not latest_user.strip():
        return None

    previous_assistant = ""
    previous_user = ""
    for message in reversed(messages[:latest_user_index]):
        role = message.get("role")
        if role == "assistant" and not previous_assistant:
            previous_assistant = _message_text(message)
        elif role == "user" and not previous_user:
            previous_user = _message_text(message)
        if previous_assistant and previous_user:
            break
    if not previous_assistant.strip():
        return None

    repeated_user = (
        _normalized_jaccard(latest_user, previous_user) >= 0.55
        if previous_user.strip()
        else False
    )
    explicit_retry = bool(
        re.search(r"\b(again|still|now|retry|repeat|please|check|get|read|run)\b", latest_user.lower())
    )
    repeated_generic_clarify = (
        _is_generic_confirm_detail_response(previous_assistant)
        and (
            explicit_retry
            or _normalized_jaccard(latest_user, previous_user) >= 0.35
            or bool(
                re.search(
                    r"\b(stored preference|stored setting|preference|setting|won't specify|will not specify)\b",
                    latest_user.lower(),
                )
            )
        )
    )
    refusal = bool(
        re.search(
            r"\b(can't|cannot|unable|not able|not available|do not have|don't have|"
            r"no access|unknown|sorry)\b",
            previous_assistant.lower(),
        )
    )
    if repeated_generic_clarify or (refusal and (repeated_user or explicit_retry)):
        return REPEAT_NOTICE
    return None


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _repeated_same_read_result_tool(
    messages: list[dict[str, Any]],
    *,
    threshold: int,
) -> str | None:
    counts: dict[tuple[str, str], int] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        if not name or not _read_like_tool_name(name):
            continue
        result_key = _normalized_tool_result_key(message)
        key = (name, result_key)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= threshold:
            return name
    return None


def _read_like_tool_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(("get_", "read_", "check_", "list_", "search_", "query_", "lookup_"))


def _normalized_tool_result_key(message: dict[str, Any]) -> str:
    content = message.get("content")
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        payload = content
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(part for part in parts if part)
    return str(content)


def _normalized_jaccard(left: str, right: str) -> float:
    left_tokens = set(_normalize_for_overlap(left).split())
    right_tokens = set(_normalize_for_overlap(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _normalize_for_overlap(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def _scatter_prompt_with_notices(
    prompt: str,
    correction: dict[str, Any] | None,
    stall_notice: str | None,
    repeat_notice: str | None,
) -> str:
    if correction is None and stall_notice is None and repeat_notice is None:
        return prompt
    payload = json.loads(prompt)
    base_instructions = str(payload.get("instructions") or "")
    notices: list[str] = []
    if correction is not None:
        notices.append(str(correction["directive"]))
    if stall_notice is not None:
        notices.append(stall_notice)
    if repeat_notice is not None:
        notices.append(repeat_notice)
    notice_text = "\n\n".join(notices)
    payload["instructions"] = (
        f"{notice_text}\n\n{base_instructions}" if base_instructions else notice_text
    )
    if correction is not None:
        payload["grounded_retry_correction"] = correction
    if stall_notice is not None:
        payload["stall_notice"] = stall_notice
    if repeat_notice is not None:
        payload["repeat_notice"] = repeat_notice
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _prompt_with_stall_notice(prompt: str, stall_notice: str | None) -> str:
    if stall_notice is None:
        return prompt
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return f"{stall_notice}\n\n{prompt}"
    base_instructions = str(payload.get("instructions") or "")
    payload["instructions"] = (
        f"{stall_notice}\n\n{base_instructions}"
        if base_instructions
        else stall_notice
    )
    payload["stall_notice"] = stall_notice
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _acknowledge_limit_majority(passes: list[dict[str, Any]]) -> bool:
    if not passes:
        return False
    count = sum(1 for item in passes if item.get("recommendation") == "acknowledge_limit")
    return count > len(passes) / 2


def _select_acknowledge_limit_action(passes: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        item for item in passes if item.get("recommendation") == "acknowledge_limit"
    ]
    respond = [
        item for item in candidates if (item.get("action") or {}).get("action") == "respond"
    ]
    pool = respond or candidates or passes
    best = max(pool, key=lambda item: float(item.get("confidence", 0.5)))
    return dict(best.get("action") or {"action": "respond", "content": ""})


def _tool_messages_by_call_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if call_id is None:
            continue
        results[str(call_id)] = message
    return results


def _assistant_message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        calls.append(tool_call)
    return calls


def _tool_call_site_key(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    return json.dumps(
        {
            "name": function.get("name", ""),
            "arguments": _parse_tool_call_arguments(function.get("arguments", {})),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_result_progress(message: dict[str, Any]) -> bool:
    content = message.get("content")
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status") or "").upper() != "SUCCESS":
        return False
    return not _empty_tool_result(payload.get("result"))


def _empty_tool_result(result: Any) -> bool:
    if result is None or result == "" or result == [] or result == {}:
        return True
    if isinstance(result, dict):
        return all(_empty_tool_result(value) for value in result.values())
    return False


def _latest_grounded_retry_signal(
    messages: list[dict[str, Any]],
) -> GroundedRetrySignal | None:
    tool_messages: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        tool_messages.append(message)
    if not tool_messages:
        return None

    calls_by_id = _assistant_tool_calls_by_id(messages)
    for message in tool_messages:
        failure = _tool_failure_from_message(message)
        if failure is None:
            continue
        error_text, reason = failure
        tool_call_id = (
            str(message.get("tool_call_id"))
            if message.get("tool_call_id") is not None
            else None
        )
        failed_call = calls_by_id.get(tool_call_id or "")
        tool_name = (
            str(message.get("name"))
            if message.get("name")
            else (
                str(failed_call.get("tool_name"))
                if isinstance(failed_call, dict) and failed_call.get("tool_name")
                else None
            )
        )
        carveout_reason = _grounded_retry_carveout_reason(error_text)
        call_site_key = _grounded_retry_call_site_key(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            failed_call=failed_call,
            error_text=error_text,
        )
        return GroundedRetrySignal(
            error_text=error_text,
            carveout=carveout_reason is not None,
            reason=carveout_reason or reason,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            failed_call=failed_call,
            call_site_key=call_site_key,
        )
    return None


def _tool_failure_from_message(message: dict[str, Any]) -> tuple[str, str] | None:
    content = message.get("content")
    raw = _stringify_tool_content(content)
    stripped = raw.strip()
    if stripped.startswith("Error:"):
        return raw, "error_string"
    try:
        payload = json.loads(stripped) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status") or "").upper()
    if status == "FAILURE":
        return raw, "status_failure"
    if "errors" in payload:
        return raw, "errors_key"
    return None


def _grounded_retry_carveout_reason(error_text: str) -> str | None:
    lowered = error_text.lower()
    if "currently removed" in lowered or "argument is currently removed" in lowered:
        return "removed_tool_or_argument"
    if "list index out of range" in lowered or "string index out of range" in lowered:
        return "generic_python_crash"
    return None


def _grounded_retry_prompt_correction(
    signal: GroundedRetrySignal,
) -> dict[str, Any]:
    return {
        "directive": (
            "CORRECTION: your previous tool call FAILED. The environment "
            "returned the error shown in grounded_retry_correction.environment_error. "
            "This external tool observation explains what went wrong. Usually it "
            "means: use an EDIT/REPLACE/DELETE tool instead of a create/set-new "
            "one; or first READ current state with a get_* / *_status / positions "
            "tool to obtain a valid id; or fix the named argument. Choose a "
            "corrected next action. Do NOT repeat the identical failing call."
        ),
        "environment_error": signal.error_text,
        "failed_tool_call": signal.failed_call,
        "failed_tool_name": signal.tool_name,
        "rules": [
            "Use only the supplied transcript, policy text, tool observations, and tool schemas.",
            "Treat intentionally removed tools or arguments as unavailable; do not retry them.",
            "Do not repeat the identical failing tool call.",
        ],
    }


def _assistant_tool_calls_by_id(
    messages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            tool_call_id = tool_call.get("id")
            function = tool_call.get("function") or {}
            name = function.get("name")
            if not tool_call_id or not name:
                continue
            calls[str(tool_call_id)] = {
                "tool_name": str(name),
                "arguments": _parse_tool_call_arguments(function.get("arguments", {})),
            }
    return calls


def _parse_tool_call_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _grounded_retry_call_site_key(
    *,
    tool_call_id: str | None,
    tool_name: str | None,
    failed_call: dict[str, Any] | None,
    error_text: str,
) -> str:
    if isinstance(failed_call, dict):
        return json.dumps(failed_call, sort_keys=True, separators=(",", ":"))
    return json.dumps(
        {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "error": error_text,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _stringify_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _candidate_options(passes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in passes:
        action = item["action"]
        key = _candidate_action_key(action)
        group = grouped.setdefault(
            key,
            {
                "action": action,
                "support_weight": 0.0,
                "vote_count": 0,
                "disposition_counts": {
                    kind: {disp: 0 for disp in DISPOSITIONS}
                    for kind in PROPOSITION_KINDS
                },
            },
        )
        group["support_weight"] += float(item.get("confidence", 0.0))
        group["vote_count"] += 1
        dispositions = item.get("dispositions", {}) or {}
        for kind in PROPOSITION_KINDS:
            disposition = dispositions.get(kind)
            if disposition in DISPOSITIONS:
                group["disposition_counts"][kind][disposition] += 1

    options = sorted(
        grouped.values(),
        key=lambda option: (-option["support_weight"], -option["vote_count"]),
    )
    for index, option in enumerate(options):
        option["candidate_index"] = index
        option["support_weight"] = round(option["support_weight"], 3)
    return options


def _candidate_action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _structured_voi_candidates(passes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    total = 0
    for item in passes:
        action = item.get("action") if isinstance(item, dict) else None
        if not isinstance(action, dict):
            continue
        total += 1
        key = _voi_candidate_key(action)
        group = groups.setdefault(
            key,
            {
                "key": key,
                "action": action,
                "vote_count": 0,
                "support_weight": 0.0,
                "param_values": {},
                "tokens": set(),
            },
        )
        group["vote_count"] += 1
        group["support_weight"] += float(item.get("confidence", 0.5))
        group["tokens"] |= _action_semantic_tokens(action)
        for name, values in _action_param_values(action).items():
            group["param_values"].setdefault(name, set()).update(values)

    all_params = {
        param
        for group in groups.values()
        for param in group.get("param_values", {})
    }
    for group in groups.values():
        for param in all_params:
            if param not in group["param_values"]:
                group["param_values"][param] = {"<UNK>"}
        group["share"] = group["vote_count"] / total if total else 0.0

    return sorted(
        groups.values(),
        key=lambda group: (
            -float(group.get("share", 0.0)),
            _canonical_action_sort_key(group.get("action", {})),
        ),
    )


def _voi_candidate_key(action: dict[str, Any]) -> str:
    if action.get("action") != "tool_calls":
        return json.dumps(
            {"action": "respond", "kind": _response_kind(action)},
            sort_keys=True,
            separators=(",", ":"),
        )
    signatures = []
    for call in action.get("tool_calls", []) or []:
        arguments = call.get("arguments") or {}
        signatures.append(
            {
                "tool_name": str(call.get("tool_name", "")),
                "specified_param_keys": sorted(str(key) for key in arguments),
            }
        )
    return json.dumps(signatures, sort_keys=True, separators=(",", ":"))


def _action_param_values(action: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    if action.get("action") != "tool_calls":
        return values
    for call in action.get("tool_calls", []) or []:
        tool_name = str(call.get("tool_name", ""))
        for key, value in (call.get("arguments") or {}).items():
            rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
            values.setdefault(str(key), set()).add(rendered)
            values.setdefault(f"{tool_name}.{key}", set()).add(rendered)
    return values


def _locate_disputed_aspect(
    candidates: list[dict[str, Any]],
    cluster_result: ClusterResult,
    transcript: list[dict[str, Any]],
) -> VoiAspect:
    param_scores: list[tuple[int, str, set[str]]] = []
    all_params = {
        param
        for candidate in candidates
        for param in candidate.get("param_values", {})
    }
    for param in all_params:
        values: set[str] = set()
        related_tokens: set[str] = set()
        for candidate in candidates:
            values |= set(candidate.get("param_values", {}).get(param, {"<UNK>"}))
            if param in candidate.get("param_values", {}):
                related_tokens |= set(candidate.get("tokens", set()))
        if len(values) > 1:
            bare_param = param.split(".")[-1]
            tokens = (
                _meaningful_tokens(bare_param.replace("_", " "))
                | related_tokens
                | _meaningful_tokens(_latest_text_by_role(transcript, "user"))
            )
            name = bare_param
            if not _meaningful_tokens(bare_param.replace("_", " ")) and related_tokens:
                name = _aspect_name_from_tokens(
                    related_tokens
                    | _meaningful_tokens(_latest_text_by_role(transcript, "user"))
                )
            param_scores.append((len(values), name, tokens))
    if param_scores:
        _, name, tokens = max(param_scores, key=lambda item: (item[0], item[1]))
        return VoiAspect(name=name, tokens=frozenset(tokens or {name}), kind="parameter")

    reps = cluster_result.representatives
    question_tokens: set[str] = set()
    for rep in reps:
        if rep.get("action") != "respond":
            continue
        text = _response_text(rep)
        if _is_question_text(text):
            question_tokens |= _question_aspect_tokens(text)

    tool_tokens: set[str] = set()
    for rep in reps:
        if rep.get("action") == "tool_calls":
            tool_tokens |= _action_semantic_tokens(rep)

    latest_user_tokens = _meaningful_tokens(_latest_text_by_role(transcript, "user"))
    tokens = question_tokens | tool_tokens
    if latest_user_tokens:
        tokens |= latest_user_tokens
    if tokens:
        name = _aspect_name_from_tokens(tokens)
        return VoiAspect(name=name, tokens=frozenset(tokens), kind="intent")

    return VoiAspect(
        name="requested action",
        tokens=frozenset({"requested", "action"}),
        kind="intent",
    )


def _question_aspect_tokens(text: str) -> set[str]:
    tokens = _meaningful_tokens(text)
    generic = {
        "ask",
        "confirm",
        "could",
        "should",
        "would",
        "whether",
        "want",
        "need",
        "which",
        "what",
    }
    return {token for token in tokens if token not in generic}


def _aspect_name_from_tokens(tokens: set[str]) -> str:
    ordered = sorted(token.replace("_", " ") for token in tokens if token)
    if not ordered:
        return "requested action"
    return " ".join(ordered[:4])


def _read_action_for_disputed_aspect(
    *,
    aspect: VoiAspect,
    passes: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> dict[str, Any] | None:
    available = _tools_by_name(tools)
    trace_text = _transcript_text(transcript)
    candidates: list[tuple[int, str, dict[str, Any]]] = []

    for item in passes:
        action = item.get("action") if isinstance(item, dict) else None
        if not isinstance(action, dict) or action.get("action") != "tool_calls":
            continue
        for call in action.get("tool_calls", []) or []:
            name = str(call.get("tool_name", ""))
            tool = available.get(name)
            if tool is None or not _is_read_only_getter_tool(tool):
                continue
            arguments = call.get("arguments") or {}
            if _validate_json_schema(arguments, _tool_parameters(tool)):
                continue
            score = _getter_coverage_score(tool, aspect.tokens)
            if score <= 0:
                continue
            candidates.append(
                (
                    score + 2,
                    _canonical_action_sort_key(
                        {"action": "tool_calls", "tool_calls": [call]}
                    ),
                    {"action": "tool_calls", "tool_calls": [_normalized_tool_call(call)]},
                )
            )

    for tool in tools:
        if not _is_read_only_getter_tool(tool):
            continue
        score = _getter_coverage_score(tool, aspect.tokens)
        if score <= 0:
            continue
        arguments = _arguments_for_read_tool(_tool_parameters(tool), trace_text)
        if arguments is None:
            continue
        name = str(tool.get("function", {}).get("name", ""))
        action = {
            "action": "tool_calls",
            "tool_calls": [{"tool_name": name, "arguments": arguments}],
        }
        candidates.append((score, _canonical_action_sort_key(action), action))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _is_read_only_getter_tool(tool: dict[str, Any]) -> bool:
    function = tool.get("function", {}) or {}
    name = str(function.get("name", "")).lower()
    description = str(function.get("description", "")).lower()
    normalized_name = name.replace("_", " ")
    is_getter = (
        name.startswith("get_")
        or name.startswith("check_")
        or name.endswith("_status")
        or " status" in normalized_name
    )
    if not is_getter:
        return False
    mutation_pattern = (
        r"\b(set|turn on|turn off|enable|disable|activate|deactivate|open|close|"
        r"send|call|create|delete|update|modify|change|write)\b"
    )
    return not re.search(mutation_pattern, description)


def _getter_coverage_score(tool: dict[str, Any], aspect_tokens: frozenset[str]) -> int:
    if not aspect_tokens:
        return 0
    tool_tokens = _tool_search_tokens(tool)
    return len(set(aspect_tokens) & tool_tokens)


def _voi_evpi_for_aspect(
    *,
    aspect: VoiAspect,
    passes: list[dict[str, Any]],
    cluster_result: ClusterResult,
) -> float:
    counts: dict[str, int] = {}
    for item in passes:
        action = item.get("action") if isinstance(item, dict) else None
        if not isinstance(action, dict):
            continue
        key = _aspect_value_key(action, aspect)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return 1.0 - (max(counts.values()) / total)


def _aspect_value_key(action: dict[str, Any], aspect: VoiAspect) -> str:
    if aspect.kind == "parameter" and action.get("action") == "tool_calls":
        values: list[str] = []
        for call in action.get("tool_calls", []) or []:
            args = call.get("arguments") or {}
            if aspect.name in args:
                values.append(json.dumps(args[aspect.name], sort_keys=True))
        return "|".join(sorted(values)) if values else "<UNK>"
    if action.get("action") == "tool_calls":
        return _tool_name_signature(action)
    return f"respond:{_response_kind(action)}"


def _aspect_question_count(
    transcript: list[dict[str, Any]],
    aspect: VoiAspect,
) -> int:
    count = 0
    for item in transcript:
        if item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, str) or not _is_question_text(content):
            continue
        question_tokens = _meaningful_tokens(content)
        if question_tokens & set(aspect.tokens):
            count += 1
    return count


def _select_committed_action(
    cluster_result: ClusterResult,
    *,
    deterministic: bool,
    route: str | None = None,
    ctx_logger: Any | None = None,
) -> dict[str, Any]:
    if not cluster_result.clusters:
        return {"action": "respond", "content": "Could you say that again?"}
    if not deterministic:
        return cluster_result.representatives[0]
    baseline = cluster_result.representatives[0]
    ranked = [
        (
            -len(cluster),
            _canonical_action_sort_key(cluster[0]),
            cluster[0],
        )
        for cluster in cluster_result.clusters
        if cluster
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = ranked[0][2]
    _log_deterministic_divergence(
        ctx_logger=ctx_logger,
        route=route,
        selected=selected,
        baseline=baseline,
        source="commit",
    )
    return selected


def _select_best_act_candidate(
    passes: list[dict[str, Any]],
    cluster_result: ClusterResult,
    *,
    deterministic: bool,
    route: str | None = None,
    ctx_logger: Any | None = None,
) -> dict[str, Any]:
    actions = [
        item.get("action")
        for item in passes
        if isinstance(item, dict) and isinstance(item.get("action"), dict)
    ]
    tool_actions = [action for action in actions if action.get("action") == "tool_calls"]
    if not tool_actions:
        return _select_committed_action(
            cluster_result,
            deterministic=deterministic,
            route=route,
            ctx_logger=ctx_logger,
        )
    counts: dict[str, int] = {}
    reps: dict[str, dict[str, Any]] = {}
    for action in tool_actions:
        key = _candidate_action_key(action)
        counts[key] = counts.get(key, 0) + 1
        reps.setdefault(key, action)
    ranked = [
        (-count, _canonical_action_sort_key(reps[key]), reps[key])
        for key, count in counts.items()
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    if deterministic:
        selected = ranked[0][2]
        _log_deterministic_divergence(
            ctx_logger=ctx_logger,
            route=route,
            selected=selected,
            baseline=tool_actions[0],
            source="act",
        )
        return selected
    return tool_actions[0]


def _log_deterministic_divergence(
    *,
    ctx_logger: Any | None,
    route: str | None,
    selected: dict[str, Any],
    baseline: dict[str, Any],
    source: str,
) -> None:
    if _canonical_action_sort_key(selected) == _canonical_action_sort_key(baseline):
        return
    if ctx_logger is None:
        return
    ctx_logger.info(
        "DETERMINISTIC divergence "
        f"route={route or 'unknown'} source={source} "
        f"selected_action={selected.get('action')} "
        f"baseline_action={baseline.get('action')} "
        f"selected_key={_canonical_action_sort_key(selected)} "
        f"baseline_key={_canonical_action_sort_key(baseline)}"
    )


def _canonical_action_sort_key(action: dict[str, Any]) -> str:
    return json.dumps(_normalize_json(action), sort_keys=True, separators=(",", ":"))


def _action_semantic_tokens(action: dict[str, Any]) -> set[str]:
    if action.get("action") == "respond":
        return _meaningful_tokens(str(action.get("content", "")))
    tokens: set[str] = set()
    for call_item in action.get("tool_calls", []) or []:
        tokens |= _meaningful_tokens(str(call_item.get("tool_name", "")).replace("_", " "))
        for key, value in (call_item.get("arguments") or {}).items():
            tokens |= _meaningful_tokens(str(key).replace("_", " "))
            if isinstance(value, str):
                tokens |= _meaningful_tokens(value.replace("_", " "))
    return tokens


def _voi_question_prompt(
    *,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    passes: list[dict[str, Any]],
    cluster_result: ClusterResult,
    aspect: VoiAspect,
) -> str:
    return json.dumps(
        {
            "task": "Ask one targeted clarification question.",
            "disputed_aspect": aspect.name,
            "available_tools": tools,
            "conversation_transcript": transcript,
            "draft_action_summary": [
                {
                    "action": item.get("action"),
                    "confidence": item.get("confidence", 0.0),
                }
                for item in passes
            ],
            "disagreement": {
                "axis": cluster_result.axis,
                "top_share": cluster_result.top_share,
            },
            "rules": [
                "Return a respond action only.",
                "Ask exactly one question about disputed_aspect.",
                "Do not list candidate actions, internal options, tool names, or draft text.",
                "Do not use 'Which should I do'.",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _ground_voi_question(
    action: dict[str, Any],
    cluster_result: ClusterResult,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    aspect: VoiAspect,
) -> dict[str, Any]:
    fallback = _deterministic_voi_question_action(aspect)
    if _contains_candidate_leak(action):
        return fallback
    grounded = _ground_clarify_question(action, cluster_result, transcript, tools)
    if _contains_candidate_leak(grounded):
        return fallback
    return grounded


def _deterministic_voi_question_action(aspect: VoiAspect) -> dict[str, Any]:
    label = aspect.name.strip() or "specific detail"
    label = re.sub(r"\s+", " ", label)
    return {
        "action": "respond",
        "content": f"Could you confirm the {label} you want?",
    }


def _sanitize_candidate_leak_response(
    action: dict[str, Any],
    *,
    disputed_aspect: Any,
    route: str | None = None,
    ctx_logger: Any | None = None,
) -> dict[str, Any]:
    marker = _candidate_leak_marker(action)
    if marker is None:
        return action
    if ctx_logger is not None:
        ctx_logger.info(
            "VOI sanitizer FIRED "
            f"route={route or 'unknown'} matched={marker} "
            f"was_action={action.get('action')}"
        )
    aspect_name = str(disputed_aspect or "specific action")
    return _deterministic_voi_question_action(
        VoiAspect(
            name=aspect_name,
            tokens=frozenset(_meaningful_tokens(aspect_name)),
            kind="intent",
        )
    )


def _contains_candidate_leak(action: dict[str, Any]) -> bool:
    return _candidate_leak_marker(action) is not None


def _candidate_leak_marker(action: dict[str, Any]) -> str | None:
    if action.get("action") != "respond":
        return None
    content = str(action.get("content", ""))
    lowered = content.lower()
    leak_markers = (
        ("which_should_i_do", "which should i do"),
        ("or_planning_tool", " or planning tool"),
        ("planning_tool", "planning tool ("),
        ("candidate", "candidate"),
        ("draft_action", "draft action"),
    )
    for label, marker in leak_markers:
        if marker in lowered:
            return label
    return None


def _queue_from_proposition_kinds(kinds: list[Any]) -> list[Proposition]:
    queue: list[Proposition] = []
    seen: set[str] = set()
    for raw_kind in kinds:
        kind = str(raw_kind)
        if kind not in PROPOSITION_KINDS or kind in seen:
            continue
        seen.add(kind)
        prop = Proposition(kind=kind)
        prop.record("uncertain")
        queue.append(prop)
    return queue


def _action_key(action: dict[str, Any]) -> str:
    if action.get("action") == "respond":
        return "respond"
    names = sorted(call.get("tool_name", "") for call in action.get("tool_calls", []))
    return "tool_calls:" + ",".join(names)


def _is_readonly_get_action(action: dict[str, Any]) -> bool:
    if action.get("action") != "tool_calls":
        return False
    calls = action.get("tool_calls") or []
    if not calls:
        return False
    return all(str(call.get("tool_name") or "").startswith("get_") for call in calls)


def _average_billed_tokens(results: Any) -> float:
    totals: list[int] = []
    for result in results:
        usage = getattr(result, "token_usage", None)
        if usage is None:
            continue
        total = (
            int(getattr(usage, "input_tokens", 0) or 0)
            + int(getattr(usage, "output_tokens", 0) or 0)
            + int(getattr(usage, "reasoning_output_tokens", 0) or 0)
        )
        if total:
            totals.append(total)
    return sum(totals) / len(totals) if totals else 0.0


def _branch_meta(branch: str, *, branch_llm: bool = False) -> dict[str, Any]:
    return {
        "branch": branch,
        "branch_llm": branch_llm,
        "candidate_reviewed": False,
        "review_skipped_readonly": 0,
        "sharpen_iters": 0,
        "rescatters": 0,
        "final_reviewed": False,
    }


def _clarify_prompt(
    *,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    passes: list[dict[str, Any]],
    cluster_result: ClusterResult,
) -> str:
    return json.dumps(
        {
            "task": "Ask one short clarifying question for the user's current request.",
            "available_tools": tools,
            "conversation_transcript": transcript,
            "disagreement": {
                "axis": cluster_result.axis,
                "top_share": cluster_result.top_share,
                "representatives": cluster_result.representatives,
            },
            "draft_actions": [
                {
                    "action": item.get("action"),
                    "confidence": item.get("confidence", 0.0),
                    "recommendation": item.get("recommendation", "act"),
                }
                for item in passes
            ],
            "rules": [
                "Return a respond action only.",
                "Ask exactly one question.",
                "Only mention options, tools, values, or objects present in the transcript, representative draft actions, or supplied tool schemas.",
                "Do not act or claim a tool result.",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _ground_clarify_question(
    action: dict[str, Any],
    cluster_result: ClusterResult,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    content = str(action.get("content", "")).strip()
    if not content.endswith("?"):
        content = content.rstrip(".") + "?"
    allowed = (
        _meaningful_tokens(_transcript_text(transcript))
        | _meaningful_tokens(json.dumps(cluster_result.representatives, sort_keys=True))
        | _meaningful_tokens(json.dumps(tools, sort_keys=True))
    )
    question_tokens = _meaningful_tokens(content)
    if question_tokens and not (question_tokens - allowed):
        return {"action": "respond", "content": content}
    return _deterministic_clarify_action(
        cluster_result,
        transcript=transcript,
        tools=tools,
    )


def _deterministic_clarify_action(
    cluster_result: ClusterResult,
    *,
    transcript: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reps = cluster_result.representatives[:3]
    read_action = _select_safe_read_representative(reps, tools or [])
    if read_action is not None:
        return read_action
    tool_action = _select_tool_over_refusal_representative(reps)
    if tool_action is not None:
        return tool_action
    if reps and all(rep.get("action") == "respond" for rep in reps):
        candidates = list(reps)
        if transcript is not None:
            candidates = [
                rep
                for rep in candidates
                if not _assistant_already_said(transcript, str(rep.get("content", "")))
            ] or list(reps)
        candidates = [
            rep
            for rep in candidates
            if not _is_generic_confirm_detail_response(str(rep.get("content", "")))
        ] or candidates
        best = max(
            candidates,
            key=lambda rep: (
                len(str(rep.get("content", "")).strip()),
                _canonical_action_sort_key(rep),
            ),
        )
        return dict(best)
    labels = [_action_option_label(rep) for rep in cluster_result.representatives[:3]]
    labels = [label for index, label in enumerate(labels) if label not in labels[:index]]
    if len(labels) >= 2:
        content = "Could you confirm which option you want?"
    elif labels:
        content = f"Could you confirm whether you want me to {labels[0]}?"
    else:
        content = "Could you confirm the action you want?"
    return {"action": "respond", "content": content}


def _select_safe_read_representative(
    reps: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for rep in reps:
        if rep.get("action") != "tool_calls":
            continue
        calls = rep.get("tool_calls") or []
        if not calls:
            continue
        if all(_is_read_tool_call(call, tools) for call in calls):
            candidates.append(rep)
    if not candidates:
        return None
    return dict(min(candidates, key=_canonical_action_sort_key))


def _select_tool_over_refusal_representative(
    reps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not reps:
        return None
    tool_reps = [rep for rep in reps if rep.get("action") == "tool_calls"]
    response_reps = [rep for rep in reps if rep.get("action") == "respond"]
    if not tool_reps or not response_reps:
        return None
    if any(_is_refusal_response(str(rep.get("content", ""))) for rep in response_reps):
        return dict(min(tool_reps, key=_canonical_action_sort_key))
    return None


def _is_read_tool_call(call: dict[str, Any], tools: list[dict[str, Any]]) -> bool:
    name = str(call.get("tool_name", ""))
    lowered = name.lower()
    if lowered.startswith(("get_", "read_", "check_", "list_", "search_", "query_", "lookup_")):
        return True
    for tool in tools:
        function = tool.get("function", {}) or {}
        if str(function.get("name", "")) != name:
            continue
        return _is_read_tool(tool)
    return False


def _is_refusal_response(content: str) -> bool:
    lowered = content.lower()
    return bool(
        re.search(
            r"\b(can't|cannot|unable|not able|not available|do not have|don't have|"
            r"no access|sorry|impossible)\b",
            lowered,
        )
    )


def _is_generic_confirm_detail_response(content: str) -> bool:
    normalized = _normalize_for_overlap(content)
    return bool(
        re.search(
            r"\b(confirm|clarify).*\b(specific detail|specific action|detail you want|value you want)\b",
            normalized,
        )
    )


def _assistant_already_said(transcript: list[dict[str, Any]], content: str) -> bool:
    if not content.strip():
        return False
    normalized = _normalize_for_overlap(content)
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        previous = _normalize_for_overlap(_message_text(message))
        if previous and _jaccard_similarity(normalized, previous) >= 0.9:
            return True
    return False


def _action_option_label(action: dict[str, Any]) -> str:
    if action.get("action") == "respond":
        return "respond"
    calls = action.get("tool_calls") or []
    if not calls:
        return "use a tool"
    call = calls[0]
    name = str(call.get("tool_name", ""))
    arguments = call.get("arguments") or {}
    values = [
        f"{key}={value}"
        for key, value in sorted(arguments.items())
        if isinstance(value, (str, int, float, bool))
    ]
    label = f"{name}({', '.join(values[:2])})" if values else name
    return _truncate_label(label, limit=36)


def _truncate_label(label: str, limit: int = 60) -> str:
    clean = re.sub(r"\s+", " ", label).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _is_read_tool(tool: dict[str, Any]) -> bool:
    function = tool.get("function", {}) or {}
    name = str(function.get("name", "")).lower()
    description = str(function.get("description", "")).lower()
    text = f"{name} {description}"
    return bool(
        re.search(
            r"\b(get|read|check|query|lookup|list|status|current|retrieve|measure)\b",
            text.replace("_", " "),
        )
    )


def _tool_search_tokens(tool: dict[str, Any]) -> set[str]:
    function = tool.get("function", {}) or {}
    return _meaningful_tokens(
        " ".join(
            [
                str(function.get("name", "")).replace("_", " "),
                str(function.get("description", "")),
                json.dumps(function.get("parameters") or {}, sort_keys=True),
            ]
        )
    )


def _arguments_for_read_tool(
    schema: dict[str, Any],
    trace_text: str,
) -> dict[str, Any] | None:
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    arguments: dict[str, Any] = {}
    for name in required:
        prop_schema = properties.get(name) or {}
        value = _infer_argument_from_trace(name, prop_schema, trace_text)
        if value is None:
            return None
        arguments[name] = value
    return arguments


def _infer_argument_from_trace(
    name: str,
    schema: dict[str, Any],
    trace_text: str,
) -> Any | None:
    if "default" in schema:
        return schema["default"]
    enum = schema.get("enum") or []
    normalized_trace = _normalize_text(trace_text)
    for value in enum:
        if _normalize_text(str(value)) in normalized_trace:
            return value
    expected = schema.get("type")
    if expected in {"number", "integer"}:
        match = re.search(r"(?<![a-z0-9])(-?\d+(?:\.\d+)?)(?![a-z0-9])", trace_text.lower())
        if not match:
            return None
        number = float(match.group(1))
        return int(number) if expected == "integer" else number
    if expected == "boolean":
        if re.search(r"\b(on|enable|enabled|yes|true|open)\b", trace_text.lower()):
            return True
        if re.search(r"\b(off|disable|disabled|no|false|closed?)\b", trace_text.lower()):
            return False
        return None
    if expected == "string":
        name_tokens = _meaningful_tokens(name.replace("_", " "))
        trace_tokens = list(re.findall(r"[a-z0-9_]+", trace_text.lower()))
        for index, token in enumerate(trace_tokens):
            if token in name_tokens and index + 1 < len(trace_tokens):
                return trace_tokens[index + 1]
        return None
    return None


def _deterministic_verifier_votes(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    passes: list[dict[str, Any]],
) -> list[VerifierVote]:
    del passes
    return [
        _verify_tool_exists(plan_action, tools),
        _verify_schema_valid(plan_action, tools),
        _verify_args_grounded_in_trace(plan_action, transcript, tools),
        _verify_not_user_rejected(plan_action, transcript),
        _verify_completion_check(plan_action, transcript, tools),
    ]


_COMPLETION_SUBJECT_TOOLS = {
    "navigation": {
        "set_new_navigation",
        "navigation_select_route",
        "navigation_replace_final_destination",
        "navigation_replace_one_waypoint",
        "navigation_delete_destination",
        "navigation_delete_waypoint",
        "navigation_add_one_waypoint",
    },
    "window": {"open_close_window", "set_window_defrost"},
    "climate": {
        "set_air_conditioning",
        "set_air_circulation",
        "set_fan_speed",
        "set_temperature",
        "set_climate_temperature",
        "set_airflow_direction",
        "set_fan_airflow_direction",
    },
    "lights": {"set_head_lights_high_beams", "set_head_lights"},
    "fog": {"set_fog_lights"},
    "wipers": {"set_windshield_wipers"},
    "seat": {"set_seat_heating", "set_seat_ventilation"},
    "sunroof": {"open_close_sunroof", "open_close_sunshade"},
    "ambient": {"set_ambient_light_color"},
    "radio": {"set_radio_station"},
    "message": {"send_text", "send_email", "make_phone_call"},
}
_COMPLETION_COMMAND_RE = re.compile(
    r"\b(set|start|change|replace|navigate|go|take|choose|pick|open|"
    r"close|turn|switch|enable|disable|make|send|call|defrost|heat|cool|"
    r"adjust|use|remove|delete|drop|take out)\b",
    re.IGNORECASE,
)
_COMPLETION_COMMAND_VARIANT = (
    r"(?:set(?:ting)?|start(?:ing)?|chang(?:e|ing)|replac(?:e|ing)|"
    r"navigat(?:e|ing)|go(?:ing)?|tak(?:e|ing)|choos(?:e|ing)|"
    r"pick(?:ing)?|open(?:ing)?|clos(?:e|ing)|turn(?:ing)?|"
    r"switch(?:ing)?|enabl(?:e|ing)|disabl(?:e|ing)|mak(?:e|ing)|"
    r"send(?:ing)?|call(?:ing)?|defrost(?:ing)?|heat(?:ing)?|"
    r"cool(?:ing)?|adjust(?:ing)?|us(?:e|ing)|remov(?:e|ing)|"
    r"delet(?:e|ing)|drop(?:ping)?|taking\s+out)"
)
_COMPLETION_COMMAND_GERUND = (
    r"(?:setting|starting|changing|replacing|navigating|going|taking|"
    r"choosing|picking|opening|closing|turning|switching|enabling|"
    r"disabling|making|sending|calling|defrosting|heating|cooling|"
    r"adjusting|using|removing|deleting|dropping)"
)
_COMPLETION_COMMAND_VARIANT_RE = re.compile(
    rf"\b{_COMPLETION_COMMAND_VARIANT}\b", re.IGNORECASE
)
_NEGATION_MARKER_RE = re.compile(
    r"\b(?:don['’]t|do\s+not|no\s+need\s+to|not\s+yet|"
    r"hold\s+off(?:\s+on)?|wait\s+(?:before|until)|skip\b|without\b)",
    re.IGNORECASE,
)
_NEGATED_COMPLETION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b(?:don['’]t|do\s+not|no\s+need\s+to)\b[^,.;!?]{{0,48}}\b{_COMPLETION_COMMAND_VARIANT}\b",
        rf"\bnot\s+yet\b[^,.;!?]{{0,48}}\b{_COMPLETION_COMMAND_VARIANT}\b",
        rf"\b{_COMPLETION_COMMAND_VARIANT}\b[^,.;!?]{{0,100}}\bnot\s+yet\b",
        r"\bhold\s+off(?:\s+on)?\b",
        r"\bwait\s+(?:before|until)\b",
        rf"\bskip\s+{_COMPLETION_COMMAND_GERUND}\b[^,.;!?]{{0,80}}\bfor\s+now\b",
        rf"\bwithout\s+{_COMPLETION_COMMAND_GERUND}\b",
        rf"\b{_COMPLETION_COMMAND_VARIANT}\b[^,.;!?]{{0,100}}\byet\b",
    )
)


def _verify_completion_check(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> VerifierVote:
    signal = _completion_signal(transcript, tools)
    fired = plan_action.get("action") == "respond" and signal is not None
    return VerifierVote(
        name="completion_check",
        score=0.0 if fired else 1.0,
        veto=False,
        repair=None,
        rationale=(
            f"completion_detector_fired candidates={signal['candidates']}"
            if fired
            else "completion_detector_clear"
        ),
    )


def _completion_notice(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    state: CompletionState,
    *,
    force_escalated: bool = False,
) -> tuple[str | None, int]:
    signal = _completion_signal(messages, tools)
    if signal is None:
        return None, 0
    command_key = _completion_command_key(signal["user_text"])
    fire_count = state.notice_fires.get(command_key, 0) + 1
    state.notice_fires[command_key] = fire_count
    if force_escalated or fire_count >= 2:
        candidate_text = ", ".join(signal["candidates"])
        return (
            "COMPLETION: Do not draft or describe the action. Call "
            f"{candidate_text} now if the arguments are known, or state "
            "specifically why you cannot.",
            1,
        )
    user_text = _quote_for_notice(signal["user_text"], 180)
    return (
        "COMPLETION: the user's command "
        f'"{user_text}" has not been executed; if its arguments are known, '
        "execute it now instead of describing or asking.",
        1,
    )


def _completion_command_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


_PERMISSIVE_CLAUSE_RE = re.compile(
    r"(?:\b(?:i(?:'m| am)?|we(?:'re| are)?)\s+(?:fine|ok|okay)\s+(?:with|if)\b|"
    r"\bit(?:'s| is)\s+(?:fine|ok|okay)\s+if\b|"
    r"\bfeel\s+free\b|\bi\s+do(?:n't| not)\s+mind\b|"
    r"\bno\s+problem\s+if\b|\bif\s+(?:needed|necessary)\b|"
    r"\bany\b[^.;!?]{0,80}\b(?:the\s+car|it)\s+makes?\b)",
    re.IGNORECASE,
)


def _user_command_clauses(text: str) -> list[str]:
    raw_clauses = re.split(
        r"[.;!?]+|,\s*(?=(?:and|but|while|then|i(?:'m| am)|"
        r"it(?:'s| is)|feel\s+free|if\s+(?:needed|necessary))\b)",
        text,
        flags=re.IGNORECASE,
    )
    clauses: list[str] = []
    negation_boundary = re.compile(
        r"\s+(?=(?:(?:but|and|then)\s+)(?:don['’]?t|do\s+not|"
        r"no\s+need\s+to|hold\s+off|wait\s+(?:before|until)|skip\b)|"
        r"without\b)",
        re.IGNORECASE,
    )
    positive_contrast_boundary = re.compile(
        rf"(?:\s+(?=(?:but|and|then)\s+(?:(?:please|just|only)\s+)?"
        rf"(?:{_COMPLETION_COMMAND_VARIANT}|add|ask|increase|decrease|look|"
        r"include|gather|draft|show|confirm)\b)|"
        rf",\s*(?=(?:just|only|please)\s+(?:{_COMPLETION_COMMAND_VARIANT}|"
        r"add|ask|increase|decrease|look|include|gather|draft|show|confirm)\b))",
        re.IGNORECASE,
    )
    for raw_clause in raw_clauses:
        parenthetical_negations: list[str] = []

        def remove_negated_parenthetical(match: re.Match[str]) -> str:
            parenthetical = match.group(0)
            if not _NEGATION_MARKER_RE.search(parenthetical):
                return parenthetical
            parenthetical_negations.append(parenthetical[1:-1])
            return " "

        without_negated_parentheticals = re.sub(
            r"\([^()]+\)", remove_negated_parenthetical, raw_clause
        )
        pieces = negation_boundary.split(without_negated_parentheticals)
        pieces.extend(parenthetical_negations)
        refined: list[str] = []
        for piece in pieces:
            if _NEGATION_MARKER_RE.search(piece):
                for dash_piece in re.split(r"\s*[—–]\s*|\s+-\s+", piece):
                    refined.extend(positive_contrast_boundary.split(dash_piece))
            else:
                refined.append(piece)
        for piece in refined:
            clause = piece.strip()
            if not clause:
                continue
            if re.fullmatch(r"(?:but\s+)?(?:not\s+)?yet", clause, re.IGNORECASE) and clauses:
                clauses[-1] = f"{clauses[-1]} {clause}"
            else:
                clauses.append(clause)
    return clauses


def _negated_completion_clause(clause: str) -> bool:
    if _PERMISSIVE_CLAUSE_RE.search(clause):
        return False
    if not _completion_subjects(clause):
        return False
    if any(pattern.search(clause) for pattern in _NEGATED_COMPLETION_PATTERNS):
        return True
    return bool(
        re.search(
            r"\b(?:don['’]t|do\s+not|no\s+need\s+to)\b"
            r"[^,.;!?]{0,48}\b[a-z]+(?:ing|ed)?\b",
            clause,
            re.IGNORECASE,
        )
    )


def _nonpermissive_user_clauses(text: str) -> list[str]:
    return [
        clause
        for clause in _user_command_clauses(text)
        if not _PERMISSIVE_CLAUSE_RE.search(clause)
        and not _negated_completion_clause(clause)
    ]


def _legacy_completion_subjects(text: str) -> set[str]:
    clauses = re.split(
        r"[.;!?]+|,\s*(?=(?:and|but|while|then|i(?:'m| am)|"
        r"it(?:'s| is)|feel\s+free|if\s+(?:needed|necessary))\b)",
        text,
        flags=re.IGNORECASE,
    )
    subjects: set[str] = set()
    for clause in clauses:
        if _PERMISSIVE_CLAUSE_RE.search(clause):
            continue
        if _COMPLETION_COMMAND_RE.search(clause):
            subjects.update(_completion_subjects(clause))
    return subjects


def _active_completion_hold_operations(
    messages: list[dict[str, Any]],
) -> list[set[str]]:
    active: list[set[str]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        additions: list[set[str]] = []
        for clause in _user_command_clauses(_message_text(message)):
            if _PERMISSIVE_CLAUSE_RE.search(clause):
                continue
            if _negated_completion_clause(clause):
                held_tools = _completion_operation_tools(clause, negated=True)
                if held_tools:
                    additions.append(held_tools)
                continue
            positive_tools = _completion_operation_tools(clause, negated=False)
            if positive_tools:
                active = [held - positive_tools for held in active]
                active = [held for held in active if held]
        for addition in additions:
            if addition not in active:
                active.append(addition)
    return active


def _active_completion_holds(messages: list[dict[str, Any]]) -> set[str]:
    return {
        tool_name
        for operation in _active_completion_hold_operations(messages)
        for tool_name in operation
    }


def _completion_command_verbs(clause: str) -> set[str]:
    bases = {
        "setting": "set",
        "starting": "start",
        "changing": "change",
        "replacing": "replace",
        "navigating": "navigate",
        "going": "go",
        "taking": "take",
        "taking out": "take out",
        "choosing": "choose",
        "picking": "pick",
        "opening": "open",
        "closing": "close",
        "turning": "turn",
        "switching": "switch",
        "enabling": "enable",
        "disabling": "disable",
        "making": "make",
        "sending": "send",
        "calling": "call",
        "defrosting": "defrost",
        "heating": "heat",
        "cooling": "cool",
        "adjusting": "adjust",
        "using": "use",
        "removing": "remove",
        "deleting": "delete",
        "dropping": "drop",
    }
    return {
        bases.get(match.group(0).lower(), match.group(0).lower())
        for match in _COMPLETION_COMMAND_VARIANT_RE.finditer(clause)
    }


def _negated_completion_verbs(clause: str) -> set[str]:
    marker_match = re.search(
        r"\b(?:don['’]t|do\s+not|no\s+need\s+to)\b"
        r"\s+(?:(?:actually|please|just)\s+)?([a-z]+)",
        clause,
        re.IGNORECASE,
    )
    bases = {
        "adding": "add",
        "asking": "ask",
        "closing": "close",
        "changing": "change",
        "deleting": "delete",
        "opening": "open",
        "sending": "send",
        "setting": "set",
        "starting": "start",
        "stopping": "stop",
    }
    if marker_match:
        governed = bases.get(
            marker_match.group(1).lower(), marker_match.group(1).lower()
        )
        tail = clause[marker_match.end() :]
        if governed in {"want", "wish", "prefer"}:
            return _completion_command_verbs(tail)
        verbs = {governed}
        coordinated = re.compile(
            rf"\b(?:or|nor)\s+({_COMPLETION_COMMAND_VARIANT}|add|stop)\b",
            re.IGNORECASE,
        )
        for match in coordinated.finditer(tail):
            verb = match.group(1).lower()
            verbs.add(bases.get(verb, verb))
        return verbs
    governed_match = re.search(
        r"\b(?:hold\s+off(?:\s+on)?|wait\s+(?:before|until)|skip|without)"
        r"\s+([a-z]+)",
        clause,
        re.IGNORECASE,
    )
    if governed_match:
        verb = governed_match.group(1).lower()
        return {bases.get(verb, verb)}
    return _completion_command_verbs(clause)


def _completion_operation_tools(clause: str, *, negated: bool) -> set[str]:
    lowered = clause.lower()
    verbs = (
        _negated_completion_verbs(clause)
        if negated
        else _completion_command_verbs(clause)
    )
    tools: set[str] = set()
    has_navigation_object = bool(
        re.search(r"\b(?:navigation|navigate|route|destination|waypoint|stop)\b", lowered)
    )
    has_climate_object = bool(
        re.search(
            r"\b(?:climate|temperature|fan|airflow|vent|recirculation|"
            r"air\s+conditioning|a/c|ac|heating|cooling)\b",
            lowered,
        )
    )

    if has_navigation_object:
        if verbs & {"start", "navigate", "go"}:
            tools.add("set_new_navigation")
        if "add" in verbs:
            tools.add("navigation_add_one_waypoint")
        if verbs & {"remove", "delete", "drop", "take", "take out"}:
            if re.search(r"\bfinal\s+(?:destination|stop)|destination\b", lowered):
                tools.add("navigation_delete_destination")
            if re.search(r"\b(?:waypoint|stop)\b", lowered) or not tools:
                tools.add("navigation_delete_waypoint")
        if verbs & {"replace", "change"}:
            if re.search(r"\b(?:final\s+)?destination\b", lowered):
                tools.add("navigation_replace_final_destination")
            elif re.search(r"\b(?:waypoint|stop)\b", lowered):
                tools.add("navigation_replace_one_waypoint")
        if verbs & {"choose", "pick", "take", "use"} and "route" in lowered:
            tools.add("navigation_select_route")
        if "make" in verbs and "destination" in lowered:
            tools.add("navigation_replace_final_destination")
        if "set" in verbs:
            if re.search(r"\b(?:waypoint|stop|charger|charging\s+station)\b", lowered):
                tools.add("navigation_add_one_waypoint")
            elif "route" in lowered:
                tools.add("set_new_navigation")
            else:
                tools.update(_COMPLETION_SUBJECT_TOOLS["navigation"])

    if re.search(r"\b(?:email|text|message|call)\b", lowered):
        if "send" in verbs:
            if "email" in lowered:
                tools.add("send_email")
            if re.search(r"\btext\b", lowered):
                tools.add("send_text")
            if "message" in lowered and not tools:
                tools.add("send_text")
        if verbs & {"call", "make"} and re.search(r"\bcall\b", lowered):
            tools.add("make_phone_call")

    if re.search(r"\b(?:windows?|windshield)\b", lowered):
        if verbs & {"open", "close"}:
            tools.add("open_close_window")
        if "defrost" in verbs or "defrost" in lowered:
            tools.add("set_window_defrost")

    if has_climate_object and verbs & {
        "set", "start", "change", "turn", "switch", "enable", "disable",
        "adjust", "heat", "cool", "increase", "decrease",
    }:
        if "circulation" in lowered or "recirculation" in lowered:
            tools.add("set_air_circulation")
        if re.search(r"\b(?:fan\s+speed|fan\s+level)\b", lowered):
            tools.add("set_fan_speed")
        if "airflow" in lowered or re.search(r"\bvent\s+mode\b", lowered):
            tools.update({"set_airflow_direction", "set_fan_airflow_direction"})
        if "temperature" in lowered:
            tools.update({"set_temperature", "set_climate_temperature"})
        if re.search(r"\b(?:air conditioning|a/c|ac\s+mode)\b", lowered):
            tools.add("set_air_conditioning")
        if not tools:
            tools.update(_COMPLETION_SUBJECT_TOOLS["climate"])

    subjects = _completion_subjects(clause)
    for subject in subjects & {"lights", "fog", "wipers", "seat", "sunroof", "ambient"}:
        if verbs & {
            "set", "start", "change", "turn", "switch", "enable", "disable",
            "adjust", "open", "close", "heat", "cool",
        }:
            tools.update(_COMPLETION_SUBJECT_TOOLS[subject])
    if "radio" in lowered and verbs & {"set", "change", "choose", "pick", "use"}:
        tools.add("set_radio_station")
    return tools


def _completion_negation_meta(messages: list[dict[str, Any]]) -> dict[str, int]:
    latest_user = _latest_user_text(messages)
    exclusions = sum(
        1
        for clause in _user_command_clauses(latest_user)
        if _negated_completion_clause(clause)
    )
    return {
        "completion_negation_exclusions": exclusions,
        "completion_negation_holds": len(
            _active_completion_hold_operations(messages)
        ),
    }


def _completion_signal(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    user_text = _latest_user_text(messages)
    if not user_text:
        return None
    if _confirmation_only_text(user_text):
        return None
    available = _available_tool_names(tools)
    requires = _requires_confirmation_tool_names(tools)
    held_tools = _active_completion_holds(messages)
    subjects: set[str] = set()
    user_clauses = _user_command_clauses(user_text)
    nonpermissive_clauses = [
        clause
        for clause in user_clauses
        if not _PERMISSIVE_CLAUSE_RE.search(clause)
        and not _negated_completion_clause(clause)
    ]
    for clause in nonpermissive_clauses:
        if not _COMPLETION_COMMAND_RE.search(clause):
            continue
        subjects.update(_completion_subjects(clause))
        if not _completion_subjects(clause):
            subjects.update(_completion_subjects(user_text))
    if any(_negated_completion_clause(clause) for clause in user_clauses):
        positive_subjects: set[str] = set()
        for clause in nonpermissive_clauses:
            positive_subjects.update(_completion_subjects(clause))
        subjects.update(positive_subjects & _legacy_completion_subjects(user_text))
    if not subjects:
        return None
    executed = _executed_tool_names(messages)
    candidates: list[str] = []
    for subject in subjects:
        for tool_name in sorted(_COMPLETION_SUBJECT_TOOLS.get(subject, set())):
            if available and tool_name not in available:
                continue
            if tool_name in requires:
                continue
            if tool_name in executed:
                continue
            if tool_name in held_tools:
                continue
            if _policy_approval_hint(messages, subject):
                continue
            candidates.append(tool_name)
    if not candidates:
        return None
    if not _has_grounding_payload(messages) and not _simple_arg_subject(subjects):
        return None
    return {"user_text": user_text, "candidates": candidates[:4]}


def _completion_subjects(text: str) -> set[str]:
    lowered = text.lower()
    aliases = {
        "navigation": ("navigation", "navigate", "route", "destination", "waypoint", "stop"),
        "window": ("window", "windshield", "defrost"),
        "climate": ("climate", "temperature", "fan", "air conditioning", "a/c", "ac", "fresh air", "recirculation"),
        "lights": ("headlight", "high beam", "beam"),
        "fog": ("fog light",),
        "wipers": ("wiper",),
        "seat": ("seat heat", "seat ventilation", "seat heater"),
        "sunroof": ("sunroof", "sunshade"),
        "ambient": ("ambient",),
        "radio": ("radio", "station"),
        "message": ("email", "text", "message", "call"),
    }
    return {
        subject
        for subject, words in aliases.items()
        if any(word in lowered for word in words)
    }


def _confirmation_only_text(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    tokens = normalized.split()
    if not tokens:
        return False
    confirm_words = {"yes", "yep", "yeah", "ok", "okay", "confirm", "confirmed", "proceed", "approve", "approved", "do", "it"}
    return all(token in confirm_words for token in tokens[:8])


def _available_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str((tool.get("function") or {}).get("name") or "")
        for tool in tools
        if (tool.get("function") or {}).get("name")
    }


def _executed_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in _assistant_message_tool_calls(message):
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if name:
                names.add(name)
    return names


def _has_grounding_payload(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "tool" for message in messages)


def _simple_arg_subject(subjects: set[str]) -> bool:
    return bool(subjects & {"window", "climate", "lights", "fog", "wipers", "seat", "sunroof", "ambient", "radio"})


def _policy_approval_hint(messages: list[dict[str, Any]], subject: str) -> bool:
    text = "\n".join(_message_text(message).lower() for message in messages)
    return bool(
        subject in {"sunroof", "fog"}
        and re.search(r"\b(confirm|approval|required|requires|shall i|should i)\b", text)
    )


def _completion_vote_zero_count(result: VerifierEnsembleResult) -> int:
    return sum(1 for vote in result.votes if vote.name == "completion_check" and vote.score <= 0.0)


def _scope_shadow_meta(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    if action.get("action") != "tool_calls":
        return {
            "scope_shadow_flags": 0,
            "scope_shadow_details": [],
            "scope_shadow_inverse_flags": 0,
            "scope_shadow_inverse_details": [],
        }
    user_text = _latest_user_text(messages)
    clauses = _nonpermissive_user_clauses(user_text)
    if not clauses:
        return {
            "scope_shadow_flags": 0,
            "scope_shadow_details": [],
            "scope_shadow_inverse_flags": 0,
            "scope_shadow_inverse_details": [],
        }
    details: list[dict[str, Any]] = []
    inverse_details: list[dict[str, Any]] = []
    for raw_call in action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        tool_name = str(call.get("tool_name") or "")
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        for key, value in args.items():
            if not re.search(r"(window|zone|seat|defrost|airflow)", str(key), re.IGNORECASE):
                continue
            for clause in clauses:
                if not _scope_clause_matches(tool_name, str(key), clause):
                    continue
                scopes = _specific_scope_words(clause)
                universal_phrases = _universal_scope_phrases(clause)
                value_text = str(value)
                if value_text in {"ALL", "ALL_ZONES"} and scopes:
                    details.append(
                        {
                            "tool_name": call.get("tool_name"),
                            "argument": key,
                            "chosen": value,
                            "user_scope": sorted(scopes),
                            "user_phrase": _quote_for_notice(clause, 180),
                            "reward_passed": None,
                        }
                    )
                elif universal_phrases and _specific_scope_argument_value(value_text):
                    inverse_details.append(
                        {
                            "tool_name": call.get("tool_name"),
                            "argument": key,
                            "chosen": value,
                            "user_scope": universal_phrases,
                            "user_phrase": _quote_for_notice(clause, 180),
                            "reward_passed": None,
                        }
                    )
    return {
        "scope_shadow_flags": len(details),
        "scope_shadow_details": details,
        "scope_shadow_inverse_flags": len(inverse_details),
        "scope_shadow_inverse_details": inverse_details,
    }


def _scope_clause_matches(tool_name: str, argument: str, clause: str) -> bool:
    lowered = clause.lower()
    tool = tool_name.lower()
    key = argument.lower()
    if tool == "open_close_window":
        return "window" in lowered and not re.search(r"\bdefrost|windshield\b", lowered)
    if "defrost" in tool or "defrost" in key:
        return bool(re.search(r"\b(defrost|windshield)\b", lowered))
    if "seat" in tool or "seat" in key:
        return "seat" in lowered
    if "window" in key:
        return "window" in lowered
    if "airflow" in tool or "airflow" in key:
        return "airflow" in lowered
    if "zone" in key:
        return bool(
            re.search(
                r"\b(zone|driver|passenger|front|rear|back|left|right|seat|temperature)\b",
                lowered,
            )
        )
    return False


def _universal_scope_phrases(text: str) -> list[str]:
    return sorted(
        {
            match.group(0).lower()
            for match in re.finditer(
                r"\b(?:all|every)\s+(?:the\s+)?(?:windows?|zones?|seats?)\b",
                text,
                re.IGNORECASE,
            )
        }
    )


def _specific_scope_argument_value(value: str) -> bool:
    return value.upper() in {
        "DRIVER",
        "PASSENGER",
        "DRIVER_FRONT",
        "DRIVER_REAR",
        "PASSENGER_FRONT",
        "PASSENGER_REAR",
        "FRONT",
        "REAR",
        "LEFT",
        "RIGHT",
    }


def _specific_scope_words(text: str) -> set[str]:
    lowered = text.lower()
    words = {
        "driver": ("driver",),
        "passenger": ("passenger",),
        "rear": ("rear", "back"),
        "front": ("front", "windshield"),
        "left": ("left",),
        "right": ("right",),
    }
    return {
        scope
        for scope, aliases in words.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases)
    }


def _quote_for_notice(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) > limit:
        clean = clean[: limit - 3] + "..."
    return clean.replace('"', "'")


def _assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def _billed_tokens(token_usage: Any) -> int:
    return int(getattr(token_usage, "input_tokens", 0) or 0) + int(
        getattr(token_usage, "output_tokens", 0) or 0
    ) + int(getattr(token_usage, "reasoning_output_tokens", 0) or 0)


def _verifier_grounding_repair_rerouted(result: VerifierEnsembleResult) -> bool:
    if result.decision != "repair" or not result.repaired:
        return False
    for vote in result.votes:
        if vote.name != "args_grounded_in_trace" or not vote.veto or vote.repair is None:
            continue
        if vote.repair.get("action") == "respond":
            continue
        if result.action == vote.repair:
            return True
    return False


def _verifier_grounding_repair_counters(
    result: VerifierEnsembleResult,
) -> dict[str, int]:
    counters = {
        "pref_read_commits": 0,
        "pref_arg_rewrites": 0,
        "grounding_fallback_commits": 0,
    }
    if result.decision != "repair" or not result.repaired:
        return counters
    for vote in result.votes:
        if vote.name != "args_grounded_in_trace" or not vote.veto:
            continue
        reason = vote.rationale or ""
        if "repair_kind=pref_read" in reason:
            counters["pref_read_commits"] = 1
        elif "repair_kind=pref_arg_rewrite" in reason:
            counters["pref_arg_rewrites"] = 1
        elif "repair_kind=grounding_fallback_commit" in reason:
            counters["grounding_fallback_commits"] = 1
    return counters


def _aggregate_verifier_votes(
    *,
    plan_action: dict[str, Any],
    votes: list[VerifierVote],
    threshold: float,
    weights: dict[str, float] | None = None,
) -> VerifierEnsembleResult:
    active_weights = dict(DEFAULT_VERIFIER_WEIGHTS)
    if weights:
        active_weights.update(
            {
                str(name): float(weight)
                for name, weight in weights.items()
                if isinstance(weight, (int, float)) and weight > 0
            }
        )
    total_weight = 0.0
    weighted_sum = 0.0
    for vote in votes:
        weight = active_weights.get(vote.name, 1.0)
        total_weight += weight
        weighted_sum += weight * _clamp_score(vote.score)
    score = weighted_sum / total_weight if total_weight else 1.0

    vetoes = [vote for vote in votes if vote.veto]
    repair = next((vote.repair for vote in vetoes if vote.repair), None)
    if vetoes:
        if repair is not None:
            return VerifierEnsembleResult(
                action=repair,
                votes=votes,
                score=score,
                decision="repair",
                vetoed=True,
                repaired=True,
            )
        return VerifierEnsembleResult(
            action=_verifier_defer_action(votes),
            votes=votes,
            score=score,
            decision="defer",
            vetoed=True,
            repaired=False,
        )

    repair = next((vote.repair for vote in votes if vote.repair), None)
    if score < threshold:
        if repair is not None:
            return VerifierEnsembleResult(
                action=repair,
                votes=votes,
                score=score,
                decision="repair",
                vetoed=False,
                repaired=True,
            )
        return VerifierEnsembleResult(
            action=_verifier_defer_action(votes),
            votes=votes,
            score=score,
            decision="defer",
            vetoed=False,
            repaired=False,
        )

    return VerifierEnsembleResult(
        action=plan_action,
        votes=votes,
        score=score,
        decision="act",
        vetoed=False,
        repaired=False,
    )


def _verify_tool_exists(
    plan_action: dict[str, Any],
    tools: list[dict[str, Any]],
) -> VerifierVote:
    if plan_action.get("action") != "tool_calls":
        return VerifierVote("tool_exists", 1.0, False, None, "No tool call.")
    available = _tools_by_name(tools)
    missing = [
        call.get("tool_name", "")
        for call in plan_action.get("tool_calls", []) or []
        if call.get("tool_name", "") not in available
    ]
    if not missing:
        return VerifierVote(
            "tool_exists", 1.0, False, None, "All called tools are supplied."
        )
    kept = [
        _normalized_tool_call(call)
        for call in plan_action.get("tool_calls", []) or []
        if call.get("tool_name", "") in available
    ]
    repair = (
        {"action": "tool_calls", "tool_calls": kept}
        if kept
        else _unavailable_tool_response(missing)
    )
    return VerifierVote(
        "tool_exists",
        0.0,
        True,
        repair,
        "Unavailable tool call(s): " + ", ".join(sorted(set(missing))),
    )


def _verify_schema_valid(
    plan_action: dict[str, Any],
    tools: list[dict[str, Any]],
) -> VerifierVote:
    if plan_action.get("action") != "tool_calls":
        return VerifierVote("schema_valid", 1.0, False, None, "No tool call.")
    available = _tools_by_name(tools)
    invalid: list[str] = []
    kept: list[dict[str, Any]] = []
    for call in plan_action.get("tool_calls", []) or []:
        name = call.get("tool_name", "")
        tool = available.get(name)
        if tool is None:
            invalid.append(f"{name}: unavailable")
            continue
        arguments = call.get("arguments") or {}
        errors = _validate_json_schema(arguments, _tool_parameters(tool))
        if errors:
            invalid.extend(f"{name}: {error}" for error in errors)
            continue
        kept.append(_normalized_tool_call(call))
    if not invalid:
        return VerifierVote(
            "schema_valid", 1.0, False, None, "Tool arguments match schemas."
        )
    total = len(plan_action.get("tool_calls", []) or []) or 1
    repair = (
        {"action": "tool_calls", "tool_calls": kept}
        if kept
        else {
            "action": "respond",
            "content": (
                "I need a valid set of tool arguments before I can act on that."
            ),
        }
    )
    return VerifierVote(
        "schema_valid",
        len(kept) / total,
        True,
        repair,
        "; ".join(invalid[:4]),
    )


def _verify_args_grounded_in_trace(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> VerifierVote:
    if plan_action.get("action") != "tool_calls":
        return VerifierVote(
            "args_grounded_in_trace", 1.0, False, None, "No tool arguments."
        )
    trace_text = _transcript_text(transcript)
    available = _tools_by_name(tools)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    blocked_calls: list[tuple[dict[str, Any], list[str]]] = []
    for call in plan_action.get("tool_calls", []) or []:
        name = call.get("tool_name", "")
        tool = available.get(name)
        arguments = call.get("arguments") or {}
        grounded_args, blocking = _grounded_arguments(
            arguments,
            _tool_parameters(tool) if tool is not None else {},
            trace_text,
        )
        if blocking:
            dropped.append(f"{name}: " + ", ".join(blocking))
            blocked_calls.append((_normalized_tool_call(call), blocking))
            continue
        kept.append({"tool_name": name, "arguments": grounded_args})

    if not dropped:
        return VerifierVote(
            "args_grounded_in_trace",
            1.0,
            False,
            None,
            "All supplied argument values are grounded in the trace.",
        )
    total = len(plan_action.get("tool_calls", []) or []) or 1
    repair, repair_kind = _grounding_repair_action(
        plan_action=plan_action,
        transcript=transcript,
        tools=tools,
        kept=kept,
        blocked_calls=blocked_calls,
    )
    reason = "Ungrounded required argument(s): " + "; ".join(dropped[:4])
    if repair_kind:
        reason += f" | repair_kind={repair_kind}"
    return VerifierVote(
        "args_grounded_in_trace",
        len(kept) / total,
        True,
        repair,
        reason,
    )


def _grounding_repair_action(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    blocked_calls: list[tuple[dict[str, Any], list[str]]],
) -> tuple[dict[str, Any], str]:
    pref_repair = _preference_grounding_repair(plan_action, transcript, tools, blocked_calls)
    if pref_repair is not None:
        return pref_repair
    if kept:
        return {"action": "tool_calls", "tool_calls": kept}, "partial_commit"
    read_repair = _grounding_read_repair(plan_action, transcript, tools)
    if read_repair is not None:
        return read_repair, "read_reroute"
    return _original_tool_calls_action(plan_action), "grounding_fallback_commit"


def _preference_grounding_repair(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    blocked_calls: list[tuple[dict[str, Any], list[str]]],
) -> tuple[dict[str, Any], str] | None:
    if not any(str(call.get("tool_name", "")).startswith("set_") for call, _ in blocked_calls):
        return None
    if not _has_successful_tool_result(transcript, "get_user_preferences"):
        read = _preference_read_action(tools, transcript)
        if read is not None:
            return read, "pref_read"
        return None

    rewritten = _rewrite_calls_from_preferences(plan_action, transcript, blocked_calls)
    if rewritten is not None:
        return rewritten, "pref_arg_rewrite"
    return _original_tool_calls_action(plan_action), "grounding_fallback_commit"


def _preference_read_action(
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> dict[str, Any] | None:
    tool = _tools_by_name(tools).get("get_user_preferences")
    if tool is None:
        return None
    arguments = _arguments_for_read_tool(_tool_parameters(tool), _transcript_text(transcript))
    if arguments is None:
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [{"tool_name": "get_user_preferences", "arguments": arguments}],
    }


def _rewrite_calls_from_preferences(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    blocked_calls: list[tuple[dict[str, Any], list[str]]],
) -> dict[str, Any] | None:
    blocked_by_name: dict[tuple[str, str], list[str]] = {}
    for call, blocking in blocked_calls:
        blocked_by_name[
            (str(call.get("tool_name", "")), _tool_call_sort_key(call))
        ] = list(blocking)
    changed = False
    rewritten_calls: list[dict[str, Any]] = []
    for raw_call in plan_action.get("tool_calls") or []:
        call = _normalized_tool_call(raw_call)
        name = str(call.get("tool_name", ""))
        args = dict(call.get("arguments") or {})
        blocking = blocked_by_name.get((name, _tool_call_sort_key(call)), [])
        if name.startswith("set_"):
            for key in blocking:
                value = _preference_value_for_arg(name, key, transcript)
                if value is None:
                    continue
                if args.get(key) != value:
                    args[key] = value
                    changed = True
        call["arguments"] = args
        rewritten_calls.append(call)
    if not changed:
        return None
    return {"action": "tool_calls", "tool_calls": rewritten_calls}


def _original_tool_calls_action(plan_action: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "tool_calls",
        "tool_calls": [
            _normalized_tool_call(call) for call in plan_action.get("tool_calls") or []
        ],
    }


def _preference_value_for_arg(
    tool_name: str,
    arg_name: str,
    transcript: list[dict[str, Any]],
) -> Any | None:
    text = _preference_result_text(transcript)
    lowered_tool = tool_name.lower()
    lowered_arg = arg_name.lower()
    if lowered_arg == "level" and (
        "seat" in lowered_tool or "heating" in lowered_tool or "steering" in lowered_tool
    ):
        return _seat_heating_preference_level(transcript)
    if lowered_arg in {"temperature", "target_temperature"}:
        return _temperature_from_text(text)
    if lowered_arg in {"route_selection", "route", "route_id"}:
        return _preference_string_value(text, ("route", "selection", "preferred"))
    return None


def _preference_string_value(text: str, keywords: tuple[str, ...]) -> str | None:
    if not text:
        return None
    normalized = text.strip()
    for keyword in keywords:
        match = re.search(
            rf"{re.escape(keyword)}[^A-Za-z0-9_]+([A-Za-z0-9_:-]+)",
            normalized,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
    return None


def _grounding_read_repair(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    trace_text = _transcript_text(transcript)
    planned_text = json.dumps(plan_action.get("tool_calls") or [], sort_keys=True)
    planned_tokens = _meaningful_tokens(planned_text.replace("_", " "))
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for tool in tools:
        if not _is_read_only_getter_tool(tool):
            continue
        name = str(tool.get("function", {}).get("name", ""))
        arguments = _arguments_for_read_tool(_tool_parameters(tool), trace_text)
        if arguments is None:
            continue
        score = _getter_coverage_score(tool, planned_tokens)
        score = max(score, _grounding_domain_read_score(name, planned_tokens))
        if score <= 0:
            continue
        action = {
            "action": "tool_calls",
            "tool_calls": [{"tool_name": name, "arguments": arguments}],
        }
        candidates.append((score, _canonical_action_sort_key(action), action))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _grounding_domain_read_score(name: str, planned_tokens: set[str]) -> float:
    domains = {
        "get_climate_settings": {
            "climate",
            "temperature",
            "fan",
            "airflow",
            "defrost",
            "conditioning",
            "seat",
            "heating",
            "steering",
        },
        "get_vehicle_window_positions": {"window", "windows", "defrost"},
        "get_current_location": {"location", "route", "navigation", "destination"},
        "get_user_preferences": {"preference", "preferences", "preferred", "usual"},
    }
    for tool_name, tokens in domains.items():
        if name != tool_name:
            continue
        return 1.0 if planned_tokens & tokens else 0.0
    return 0.0


def _verify_not_user_rejected(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> VerifierVote:
    if plan_action.get("action") != "tool_calls":
        return VerifierVote(
            "not_user_rejected", 1.0, False, None, "No tool call to reject."
        )
    latest_user = _latest_text_by_role(transcript, "user")
    kept: list[dict[str, Any]] = []
    rejected: list[str] = []
    for call in plan_action.get("tool_calls", []) or []:
        if _call_rejected_by_user(call, latest_user):
            rejected.append(call.get("tool_name", ""))
        else:
            kept.append(_normalized_tool_call(call))
    if not rejected:
        return VerifierVote(
            "not_user_rejected", 1.0, False, None, "No rejected action detected."
        )
    total = len(plan_action.get("tool_calls", []) or []) or 1
    repair = (
        {"action": "tool_calls", "tool_calls": kept}
        if kept
        else {
            "action": "respond",
            "content": "I won't do the action you rejected. What should I do instead?",
        }
    )
    return VerifierVote(
        "not_user_rejected",
        len(kept) / total,
        True,
        repair,
        "Rejected by latest user request: " + ", ".join(sorted(set(rejected))),
    )


def _prompted_verifier_prompt(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    passes: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "task": "Score three independent verifier heads for the proposed action.",
            "allowed_inputs": [
                "conversation_transcript",
                "available_tools",
                "proposed_action",
                "draft_actions",
            ],
            "available_tools": tools,
            "conversation_transcript": transcript,
            "proposed_action": plan_action,
            "draft_actions": [
                {
                    "action": item.get("action"),
                    "confidence": item.get("confidence", 0.0),
                    "recommendation": item.get("recommendation", "act"),
                }
                for item in passes
            ],
            "heads": {
                "policy_consistent": (
                    "Score whether the action follows visible policy and user "
                    "constraints in the transcript."
                ),
                "read_before_act": (
                    "Score whether the action can safely proceed now, or whether "
                    "a supplied read/state tool is needed first."
                ),
                "no_unsupported_claim": (
                    "Score whether any user-facing response avoids unsupported "
                    "claims not grounded in the trace or tool schemas."
                ),
            },
            "scoring": "Use 1 for clearly pass, 0 for clearly fail, 0.5 if unsure.",
        },
        ensure_ascii=False,
        indent=2,
    )


def _verifier_defer_action(votes: list[VerifierVote]) -> dict[str, Any]:
    failing = [vote.name for vote in votes if vote.veto or vote.score < 0.5]
    if "tool_exists" in failing or "schema_valid" in failing:
        content = (
            "I can't complete that with the available controls as stated. "
            "Could you tell me how you'd like to proceed?"
        )
    else:
        content = (
            "I need the missing tool argument before I can act on that."
        )
    return {"action": "respond", "content": content}


def _unavailable_tool_response(missing: list[str]) -> dict[str, Any]:
    del missing
    return {
        "action": "respond",
        "content": "I can't do that with the controls available to me here.",
    }


def _verifier_weights_from_env() -> dict[str, float] | None:
    raw = os.getenv("TRACK2_VERIFIER_WEIGHTS")
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    weights: dict[str, float] = {}
    for name, weight in payload.items():
        if isinstance(weight, (int, float)) and weight > 0:
            weights[str(name)] = float(weight)
    return weights or None


def _tools_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = tool.get("function", {}).get("name")
        if isinstance(name, str) and name:
            result[name] = tool
    return result


def _tool_parameters(tool: dict[str, Any] | None) -> dict[str, Any]:
    if not tool:
        return {}
    parameters = tool.get("function", {}).get("parameters") or {}
    return parameters if isinstance(parameters, dict) else {}


def _validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected == "object" or "properties" in schema:
        if not isinstance(value, dict):
            return [f"{path} must be object"]
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")
        for key, item in value.items():
            prop_schema = properties.get(key)
            if isinstance(prop_schema, dict):
                errors.extend(_validate_json_schema(item, prop_schema, path=f"{path}.{key}"))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"{path} must be array"]
        item_schema = schema.get("items") or {}
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    _validate_json_schema(item, item_schema, path=f"{path}[{index}]")
                )
        return errors
    if expected == "string" and not isinstance(value, str):
        errors.append(f"{path} must be string")
    elif expected == "boolean" and not isinstance(value, bool):
        errors.append(f"{path} must be boolean")
    elif expected == "integer" and not (
        isinstance(value, int) and not isinstance(value, bool)
    ):
        errors.append(f"{path} must be integer")
    elif expected == "number" and not (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        errors.append(f"{path} must be number")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path} must be one of {enum}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    return errors


def _grounded_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
    trace_text: str,
) -> tuple[dict[str, Any], list[str]]:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    grounded: dict[str, Any] = {}
    blocking: list[str] = []
    for key, value in arguments.items():
        if _arg_value_grounded(value, trace_text):
            grounded[key] = value
            continue
        if key in required:
            blocking.append(key)
        elif isinstance(properties.get(key), dict):
            continue
    return grounded, blocking


def _arg_value_grounded(value: Any, trace_text: str) -> bool:
    text = trace_text.lower()
    if value is None:
        return True
    if isinstance(value, bool):
        if value:
            return bool(
                re.search(r"\b(on|enable|enabled|activate|start|open|yes|true)\b", text)
            )
        return bool(
            re.search(r"\b(off|disable|disabled|deactivate|stop|close|no|false)\b", text)
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rendered = str(value).lower()
        variants = {rendered}
        if isinstance(value, float) and value.is_integer():
            variants.add(str(int(value)))
        return any(re.search(rf"(?<![a-z0-9]){re.escape(item)}(?![a-z0-9])", text) for item in variants)
    if isinstance(value, str):
        normalized = _normalize_text(value)
        if not normalized:
            return True
        if normalized in _normalize_text(trace_text):
            return True
        value_tokens = _meaningful_tokens(value.replace("_", " "))
        trace_tokens = _meaningful_tokens(trace_text)
        return bool(value_tokens) and value_tokens <= trace_tokens
    if isinstance(value, list):
        return all(_arg_value_grounded(item, trace_text) for item in value)
    if isinstance(value, dict):
        return all(_arg_value_grounded(item, trace_text) for item in value.values())
    return False


def _call_rejected_by_user(call: dict[str, Any], latest_user: str) -> bool:
    key_tokens = _rejection_key_tokens(call)
    fallback_tokens = _meaningful_tokens(_tool_call_text(call))
    if not key_tokens and not fallback_tokens:
        return False
    words = re.findall(r"[a-z0-9_]+", latest_user.lower().replace("don't", "do not"))
    if not words:
        return False
    negators = {"not", "no", "never", "without", "avoid", "except"}
    for index, word in enumerate(words):
        if word not in negators:
            continue
        window = set(words[index + 1 : index + 9])
        if _rejection_tokens_match(key_tokens, fallback_tokens, window):
            return True
    if "instead" in words and "of" in words:
        of_index = words.index("of")
        window = set(words[of_index + 1 : of_index + 9])
        if _rejection_tokens_match(key_tokens, fallback_tokens, window):
            return True
    return False


def _rejection_tokens_match(
    key_tokens: set[str],
    fallback_tokens: set[str],
    window: set[str],
) -> bool:
    if key_tokens:
        if len(key_tokens) == 1:
            return bool(key_tokens & window)
        return key_tokens <= window
    return bool(fallback_tokens & window)


def _rejection_key_tokens(call: dict[str, Any]) -> set[str]:
    generic = {
        "set",
        "get",
        "enable",
        "disable",
        "turn",
        "open",
        "close",
        "activate",
        "deactivate",
    }
    name_tokens = _meaningful_tokens(str(call.get("tool_name", "")).replace("_", " "))
    name_tokens = {token for token in name_tokens if token not in generic}
    arg_tokens: set[str] = set()
    for value in (call.get("arguments") or {}).values():
        if isinstance(value, str):
            arg_tokens |= _meaningful_tokens(value.replace("_", " "))
    return name_tokens | arg_tokens


def _tool_call_text(call: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": call.get("tool_name", ""),
            "arguments": call.get("arguments") or {},
        },
        sort_keys=True,
    ).replace("_", " ")


def _normalized_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": call.get("tool_name", ""),
        "arguments": call.get("arguments") or {},
    }


def _transcript_text(transcript: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in transcript:
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
        if item.get("tool_calls"):
            chunks.append(json.dumps(item.get("tool_calls"), sort_keys=True))
    return "\n".join(chunks)


def _latest_text_by_role(transcript: list[dict[str, Any]], role: str) -> str:
    for item in reversed(transcript):
        if item.get("role") == role and isinstance(item.get("content"), str):
            return item["content"]
    return ""


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def _clamp_score(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.5
    return float(max(0.0, min(1.0, value)))


def _embed_response_texts(
    texts: list[str],
    embedding_backend: EmbeddingBackend | None,
) -> list[list[float]] | None:
    if not texts or embedding_backend is None:
        return None
    vectors = embedding_backend.embed(texts)
    if not vectors or len(vectors) != len(texts):
        return None
    return vectors


def _tool_action_key(action: dict[str, Any]) -> str:
    calls = action.get("tool_calls") or []
    normalized = [
        {
            "tool_name": call.get("tool_name", ""),
            "arguments": _normalize_json(call.get("arguments") or {}),
        }
        for call in calls
    ]
    return json.dumps(
        {"action": "tool_calls", "tool_calls": normalized},
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_name_signature(action: dict[str, Any]) -> str:
    return json.dumps(
        [
            call.get("tool_name", "")
            for call in action.get("tool_calls") or []
        ],
        separators=(",", ":"),
    )


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    return value


def _response_text(action: dict[str, Any]) -> str:
    if action.get("action") == "respond":
        return str(action.get("content", ""))
    return json.dumps(action, sort_keys=True)


def _response_kind(action: dict[str, Any]) -> str:
    return "question" if _is_question_text(_response_text(action)) else "statement"


def _action_mode(action: dict[str, Any]) -> str:
    if action.get("action") == "tool_calls":
        return "tool_calls"
    return _response_kind(action)


def _is_question_text(text: str) -> bool:
    stripped = text.strip().lower()
    if stripped.endswith("?"):
        return True
    return bool(
        re.match(
            r"^(which|what|when|where|who|how|do|does|did|should|would|could|can)\b",
            stripped,
        )
    )


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _meaningful_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "do",
        "does",
        "for",
        "i",
        "it",
        "of",
        "on",
        "or",
        "please",
        "should",
        "the",
        "to",
        "want",
        "what",
        "which",
        "would",
        "you",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 1 and token not in stop
    }


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalized_vector(vector: Any) -> list[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if not norm:
        return values
    return [item / norm for item in values]


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counts))


def build_planner_from_env(
    *,
    model: str,
    api_base: str,
    service_tier: str | None,
    reasoning_effort: str | None,
    logger: Any | None = None,
) -> ConsensusPlanner:
    return ConsensusPlanner(
        model=model,
        api_base=api_base,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
        config=ConsensusPlannerConfig.from_env(),
        logger=logger,
    )
