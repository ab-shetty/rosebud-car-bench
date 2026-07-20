"""Scatter-then-Sharpen planning harness for the Track 2 Cerebras agent.

This is the v0 scaffold of the design in PLAN.md (Track 2). It replaces the
baseline single-call next-action decision with:

  1. SCATTER (1 sequential step, N parallel passes): each pass proposes a
     next action and judges atomic propositions about the turn
     (feasibility, tool availability, parameter determinacy, policy
     compliance). Parallel calls are free under the Track 2 rule and do not
     count toward the 5-sequential-call budget.
  2. VOTE -> work queue: propositions the passes disagree on (or flag as
     uncertain/blocked) become a queue of exactly what is unresolved.
  3. SHARPEN (sequential, <= max_iters): each iteration targets the queue
     with one sequential "refine" call plus a *parallel* adversarial pass
     that red-teams the current plan for a policy / correctness / tool-schema
     violation. Resolved propositions collapse; new ones can re-scatter
     (bounded).
  4. TERMINATE: queue empty -> ACT on the plan. Budget hit with the queue
     non-empty -> DEFER (ask a clarification or acknowledge a limit) rather
     than act on unresolved uncertainty.

Audit note (Track 2): sequential depth per next-action decision is
1 (scatter) + #sharpen-iterations. With the default max_iters=3 that is <= 4,
inside the 5-call ceiling. Adversarial passes are parallel and do not count.

Compliance guardrails baked in (PLAN.md sec.3 red lines):
  * Tool availability is judged ONLY from the per-task tool list supplied by
    the evaluator. We never diff against a bundled catalog to infer that a
    tool was removed, and never classify task type.
  * The adversarial pass checks the plan against the system prompt/policies,
    the interaction trace, and the tool schemas. It does NOT reconstruct
    evaluator subscores or any evaluation internals.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

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
        TokenUsage,
        add_token_usage,
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
        TokenUsage,
        add_token_usage,
    )


HARNESS_NAME = "scatter_sharpen"
SEQUENTIAL_LLM_BUDGET = 5

PROPOSITION_KINDS = (
    "feasibility",
    "tool_availability",
    "parameter_determinacy",
    "policy_compliance",
)
DISPOSITIONS = ("ok", "uncertain", "blocked")
RECOMMENDATIONS = ("act", "clarify", "acknowledge_limit")

SCATTER_PERSONAS = (
    (
        "policy-compliance-first: focus on visible policy, safety "
        "requirements, confirmations, preconditions, and avoiding unrequested "
        "side effects."
    ),
    (
        "user-literal-request: preserve the user's requested operation exactly "
        "and avoid substituting a related convenience action unless the request "
        "or supplied tools require it."
    ),
    (
        "under-specified-parameters: identify missing or ambiguous required "
        "arguments and prefer supplied preference, state, or context read tools "
        "over guessing."
    ),
    (
        "tool-availability-and-limits: verify every tool and argument against "
        "only the supplied tool schemas; if the needed capability is absent, "
        "acknowledge that limit."
    ),
    (
        "minimal-action-bias: choose the smallest complete action that satisfies "
        "the current user request and avoid extra state changes."
    ),
    (
        "proactive-internal-resolution: before asking the user, look for "
        "supplied read tools that can resolve preferences, current state, or "
        "context needed for the next action."
    ),
)

_DIVERSE_SCATTER_TEMPERATURE_MIN = 0.4
_DIVERSE_SCATTER_TEMPERATURE_MAX = 1.0
_DIVERSE_SCATTER_TEMPERATURE_SINGLE = 0.7


@dataclass
class ScatterSharpenConfig:
    """Tunable knobs. All overridable via TRACK2_SCATTER_* env vars."""

    scatter_width: int = 6
    # Sequential budget audit with the default pipeline:
    # scatter + candidate review + <=2 sharpen iterations + final review = <=5.
    max_sharpen_iters: int = 2
    # supermajority of passes that must judge a proposition "ok" for it to be
    # considered resolved at scatter time (1.0 == unanimity, brittle).
    resolve_threshold: float = 0.8
    max_rescatters: int = 2
    scatter_temperature: float | None = 0.7
    sharpen_temperature: float | None = 0.2
    # Headroom matters: gpt-oss reasoning tokens are charged against this budget
    # before the structured body is emitted. Too low -> empty/truncated output.
    scatter_max_completion_tokens: int = 2048
    sharpen_max_completion_tokens: int = 2048
    adversarial_max_completion_tokens: int = 1024
    # cap on truly-parallel clients (each holds its own SDK client + lock)
    max_parallel_clients: int = 8
    scatter_diverse: bool = True
    enable_candidate_review: bool = True
    enable_protected_read: bool = True
    enable_final_review: bool = True
    enable_mechanical_guard: bool = True

    @classmethod
    def from_env(cls) -> "ScatterSharpenConfig":
        def _int(name: str, default: int) -> int:
            v = os.getenv(name)
            return int(v) if v and v.strip() else default

        def _clamped_int(name: str, default: int, low: int, high: int) -> int:
            return max(low, min(high, _int(name, default)))

        def _float(name: str, default: float) -> float:
            v = os.getenv(name)
            return float(v) if v and v.strip() else default

        def _opt_float(name: str, default: float | None) -> float | None:
            v = os.getenv(name)
            if v is None or not v.strip():
                return default
            return float(v)

        def _bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None or not v.strip():
                return default
            return v.strip().lower() not in {"0", "false", "no", "off"}

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
            enable_candidate_review=_bool(
                "TRACK2_ENABLE_CANDIDATE_REVIEW", cls.enable_candidate_review
            ),
            enable_protected_read=_bool(
                "TRACK2_ENABLE_PROTECTED_READ", cls.enable_protected_read
            ),
            enable_final_review=_bool(
                "TRACK2_ENABLE_FINAL_REVIEW", cls.enable_final_review
            ),
            enable_mechanical_guard=_bool(
                "TRACK2_ENABLE_MECHANICAL_GUARD", cls.enable_mechanical_guard
            ),
        )


@dataclass
class _CallTally:
    """Aggregates token usage / counts across every internal LLM call."""

    token_usage: TokenUsage | None = None
    cost: float = 0.0
    quota_wait_ms: float = 0.0
    total_calls: int = 0
    sequential_calls: int = 0
    seq_budget_skips: int = 0
    duration_ms: float = 0.0
    policy_rag_retrievals: int = 0
    policy_rag_empty: int = 0
    policy_rag_tokens: int = 0
    policy_rag_section_ids: list[list[str]] = field(default_factory=list)
    policy_rag_tokens_per_call: list[int] = field(default_factory=list)
    tier1_tokens: int = 0
    tier1_tokens_per_call: list[int] = field(default_factory=list)
    tier2_fired: list[str] = field(default_factory=list)
    tier2_fired_per_call: list[list[str]] = field(default_factory=list)
    tier2_tokens: int = 0
    tier2_tokens_per_call: list[int] = field(default_factory=list)
    tier3_core_tokens: int = 0
    tier3_core_tokens_per_call: list[int] = field(default_factory=list)
    tier3_tail_tokens: int = 0
    tier3_tail_tokens_per_call: list[int] = field(default_factory=list)
    injected_tokens_total: int = 0
    injected_tokens_total_per_call: list[int] = field(default_factory=list)

    def add(self, result: Any, *, sequential: bool) -> None:
        self.token_usage = add_token_usage(self.token_usage, result.token_usage)
        self.cost += result.cost
        self.quota_wait_ms += result.quota_wait_ms
        self.duration_ms += result.duration_ms
        self.total_calls += 1
        if sequential:
            self.sequential_calls += 1

    def has_sequential_budget(self) -> bool:
        return self.sequential_calls < SEQUENTIAL_LLM_BUDGET

    def skip_sequential(self) -> None:
        self.seq_budget_skips += 1

    def record_policy_rag(
        self,
        *,
        section_ids: list[str],
        empty: bool,
        tokens: int,
        calls: int = 1,
    ) -> None:
        count = max(0, calls)
        self.policy_rag_retrievals += count
        self.policy_rag_empty += int(empty) * count
        self.policy_rag_tokens += tokens * count
        self.policy_rag_section_ids.extend([list(section_ids) for _ in range(count)])
        self.policy_rag_tokens_per_call.extend([tokens] * count)

    def record_policy_partition(
        self,
        *,
        tier1_tokens: int,
        tier2_fired: list[str],
        tier2_tokens: int,
        tier3_core_tokens: int,
        tier3_tail_tokens: int,
        injected_tokens_total: int,
        calls: int = 1,
    ) -> None:
        count = max(0, calls)
        self.tier1_tokens += tier1_tokens * count
        self.tier1_tokens_per_call.extend([tier1_tokens] * count)
        for lever_id in tier2_fired:
            if lever_id not in self.tier2_fired:
                self.tier2_fired.append(lever_id)
        self.tier2_fired_per_call.extend(
            [list(tier2_fired) for _ in range(count)]
        )
        self.tier2_tokens += tier2_tokens * count
        self.tier2_tokens_per_call.extend([tier2_tokens] * count)
        self.tier3_core_tokens += tier3_core_tokens * count
        self.tier3_core_tokens_per_call.extend([tier3_core_tokens] * count)
        self.tier3_tail_tokens += tier3_tail_tokens * count
        self.tier3_tail_tokens_per_call.extend([tier3_tail_tokens] * count)
        self.injected_tokens_total += injected_tokens_total * count
        self.injected_tokens_total_per_call.extend(
            [injected_tokens_total] * count
        )


@dataclass
class Proposition:
    kind: str
    votes: dict[str, int] = field(default_factory=dict)

    def record(self, disposition: str) -> None:
        self.votes[disposition] = self.votes.get(disposition, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.votes.values())

    def ok_ratio(self) -> float:
        return self.votes.get("ok", 0) / self.total if self.total else 0.0

    def dominant_unresolved(self) -> str:
        blocked = self.votes.get("blocked", 0)
        uncertain = self.votes.get("uncertain", 0)
        return "blocked" if blocked >= uncertain else "uncertain"


class ScatterSharpenPlanner:
    """Produces one CAR-bench next action via Scatter-then-Sharpen."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str = DEFAULT_CEREBRAS_API_BASE,
        service_tier: str | None = None,
        reasoning_effort: str | None = None,
        config: ScatterSharpenConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort
        self.config = config or ScatterSharpenConfig()
        self.logger = logger
        self._client_pool: list[CerebrasCompletionClient] = []

    # ----- public API ---------------------------------------------------

    def plan(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> AgentInferenceResult:
        tally = _CallTally()
        transcript = _messages_for_prompt(messages)

        # --- SCATTER (parallel; counts as 1 sequential step) -------------
        passes = self._scatter(transcript, tools, tally, ctx_logger)
        tally.sequential_calls += 1  # the fan-out is one sequential step
        if not passes:
            # Whole-turn scatter failure (e.g. provider-wide truncation). Degrade
            # to one plain next-action call rather than erroring the turn. This is
            # logged loudly by _scatter; treat it as a harness health signal.
            ctx_logger.warning("Scatter empty; using single-action fallback")
            action = self._single_action(transcript, tools, tally, ctx_logger)
            return AgentInferenceResult(
                next_action=action,
                elapsed_ms=tally.duration_ms,
                token_usage=tally.token_usage,
                cost=tally.cost,
                internal_calls=max(tally.total_calls, 1),
                quota_wait_ms=tally.quota_wait_ms,
            )

        plan_action, queue = self._aggregate(passes)
        candidate_reviewed = False
        candidate_review = None
        if self.config.enable_candidate_review:
            candidate_review = self._candidate_review(
                transcript=transcript,
                tools=tools,
                passes=passes,
                aggregate_action=plan_action,
                aggregate_queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
            )
        if candidate_review is not None:
            candidate_reviewed = True
            plan_action = candidate_review["action"]
            queue = candidate_review["queue"]
            ctx_logger.info(
                "Candidate verifier selected action",
                selected_candidate_index=candidate_review.get("selected_index"),
                approved_candidate=candidate_review.get("approved_candidate"),
                issue_category=candidate_review.get("issue_category"),
                unresolved=len(queue),
                action=plan_action.get("action"),
            )

        protected_read = False
        if self.config.enable_protected_read:
            plan_action, queue, protected_read = _protect_read_only_plan(
                plan_action=plan_action,
                queue=queue,
                transcript=transcript,
                tools=tools,
            )
        if protected_read:
            ctx_logger.info(
                "Protected modal read",
                tool_names=[
                    tc.get("tool_name", "")
                    for tc in plan_action.get("tool_calls", [])
                ],
            )

        # --- SHARPEN (sequential; <= max_iters, +parallel adversarial) ---
        rescatters = 0
        iters = 0
        while queue and iters < self.config.max_sharpen_iters:
            iters += 1
            plan_action, queue, new_items = self._sharpen_iteration(
                transcript=transcript,
                tools=tools,
                plan_action=plan_action,
                queue=queue,
                tally=tally,
                ctx_logger=ctx_logger,
            )
            if new_items and rescatters < self.config.max_rescatters:
                rescatters += 1
                queue.extend(new_items)

        final_reviewed = False
        if not queue:
            if (
                self.config.enable_final_review
                and not (protected_read and _is_read_only_action(plan_action))
            ):
                review = self._final_review(
                    transcript=transcript,
                    tools=tools,
                    plan_action=plan_action,
                    tally=tally,
                    ctx_logger=ctx_logger,
                )
                if review is not None:
                    final_reviewed = True
                    plan_action = review
            mechanical_guard = None
            if self.config.enable_mechanical_guard:
                mechanical_guard = _mechanical_guard_action(
                    plan_action=plan_action,
                    passes=passes,
                    transcript=transcript,
                    tools=tools,
                    protected_read=protected_read,
                )
            if mechanical_guard is not None:
                guard_name, guarded_action = mechanical_guard
                plan_action = guarded_action
                ctx_logger.info(
                    "Mechanical guard revised action",
                    guard=guard_name,
                    action=plan_action.get("action"),
                )

        # --- TERMINATE: act if queue is empty, else defer ----------------
        if queue:
            final_action = self._defer_action(queue, plan_action)
            decision = "defer"
        else:
            final_action = plan_action
            decision = "act"

        ctx_logger.info(
            "Scatter-then-Sharpen decision",
            decision=decision,
            scatter_width=len(passes),
            sharpen_iters=iters,
            rescatters=rescatters,
            unresolved=len(queue),
            candidate_reviewed=candidate_reviewed,
            final_reviewed=final_reviewed,
            protected_read=protected_read,
            candidate_review_enabled=self.config.enable_candidate_review,
            protected_read_enabled=self.config.enable_protected_read,
            final_review_enabled=self.config.enable_final_review,
            mechanical_guard_enabled=self.config.enable_mechanical_guard,
            sequential_calls=tally.sequential_calls,
            total_calls=tally.total_calls,
            action=final_action.get("action"),
        )

        return AgentInferenceResult(
            next_action=final_action,
            elapsed_ms=tally.duration_ms,
            token_usage=tally.token_usage,
            cost=tally.cost,
            internal_calls=max(tally.total_calls, 1),
            quota_wait_ms=tally.quota_wait_ms,
        )

    # ----- phases -------------------------------------------------------

    def _scatter(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> list[dict[str, Any]]:
        width = max(1, self.config.scatter_width)
        prompt = _scatter_prompt(transcript, tools)
        results: list[dict[str, Any]] = []
        drops: dict[str, int] = {}  # reason -> count, so degradation is visible

        def _one(idx: int) -> Any | None:
            client = self._client(idx)
            messages = _scatter_messages_for_pass(
                prompt=prompt,
                pass_index=idx,
                diverse=self.config.scatter_diverse,
            )
            temperature = _scatter_temperature_for_pass(
                pass_index=idx,
                width=width,
                diverse=self.config.scatter_diverse,
                uniform_temperature=self.config.scatter_temperature,
            )
            try:
                return client.generate(
                    model=self.model,
                    messages=messages,
                    response_schema=SCATTER_PASS_SCHEMA,
                    response_schema_name="scatter_pass",
                    max_completion_tokens=self.config.scatter_max_completion_tokens,
                    temperature=temperature,
                    reasoning_effort=self.reasoning_effort,
                )
            except CerebrasTemplateError as exc:
                ctx_logger.warning("Scatter pass call failed", idx=idx, error=str(exc))
                return None

        with ThreadPoolExecutor(max_workers=self._pool_size(width)) as pool:
            for res in pool.map(_one, range(width)):
                if res is None:
                    drops["call_error"] = drops.get("call_error", 0) + 1
                    continue
                # token cost counts even for an unusable pass (we paid for it).
                # Done on the main thread; the per-call lock makes generate() safe
                # to fan out, but tally mutation stays single-threaded here.
                tally.add(res, sequential=False)
                if not (res.text or "").strip():
                    # empty body: almost always reasoning-budget truncation.
                    reason = (
                        "truncated_length"
                        if res.finish_reason == "length"
                        else f"empty_{res.finish_reason or 'unknown'}"
                    )
                    drops[reason] = drops.get(reason, 0) + 1
                    continue
                try:
                    results.append(_parse_scatter_pass(res.text))
                except (
                    MalformedModelResponseError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    drops["malformed"] = drops.get("malformed", 0) + 1
                    ctx_logger.warning("Malformed scatter pass", error=str(exc))

        valid = len(results)
        # Surface the drop rate. If most passes fail, the ensemble is degraded
        # and the A/B is not measuring the intended width -- warn loudly so this
        # cannot hide behind the graceful fallback.
        log = ctx_logger.warning if valid < (width + 1) // 2 else ctx_logger.info
        # Counts live in the message text (not just kwargs) so they are visible
        # regardless of log level/format -- this is an audit-relevant health
        # signal and must not hide behind DEBUG-only extras. NB: loguru treats
        # "{" / "}" in the message as format placeholders, so the message must
        # contain NO braces (no dict repr, no "{}" literal) or it raises.
        reason_str = ",".join(f"{k}={v}" for k, v in drops.items()) or "none"
        log(
            f"Scatter passes valid={valid}/{width} "
            f"dropped={width - valid} reasons={reason_str}",
            width=width,
            valid=valid,
            dropped=width - valid,
            drop_reasons=drops or None,
        )
        return results

    def _aggregate(
        self, passes: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[Proposition]]:
        # working plan: modal action among passes weighted by confidence.
        scored: dict[str, float] = {}
        rep: dict[str, dict[str, Any]] = {}
        for p in passes:
            key = _action_key(p["action"])
            scored[key] = scored.get(key, 0.0) + p["confidence"]
            rep.setdefault(key, p["action"])
        best_key = max(scored, key=scored.get)
        plan_action = rep[best_key]

        # propositions: tally dispositions per kind, queue the unresolved.
        props = {kind: Proposition(kind=kind) for kind in PROPOSITION_KINDS}
        for p in passes:
            for kind, disp in p["dispositions"].items():
                if kind in props and disp in DISPOSITIONS:
                    props[kind].record(disp)
        queue = [
            prop
            for prop in props.values()
            if prop.total and prop.ok_ratio() < self.config.resolve_threshold
        ]
        return plan_action, queue

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
                messages=[
                    {"role": "system", "content": CANDIDATE_REVIEW_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=CANDIDATE_REVIEW_SCHEMA,
                response_schema_name="candidate_review",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Candidate verifier failed", error=str(exc))
            return None

        tally.add(res, sequential=True)
        try:
            payload = json.loads(res.text)
            action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning("Malformed candidate verifier output", error=str(exc))
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
    ) -> tuple[dict[str, Any], list[Proposition], list[Proposition]]:
        target = queue[0]  # one queued proposition per iteration
        # adversarial pass runs in parallel with the refine call (free).
        with ThreadPoolExecutor(max_workers=2) as pool:
            adv_future = pool.submit(
                self._adversarial, transcript, tools, plan_action, ctx_logger
            )
            refine = self._refine(
                transcript, tools, plan_action, queue, target, ctx_logger
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
            remaining = [p for p in queue if p.kind not in resolved_ids]
        else:
            remaining = queue

        # an adversarial high-severity hit re-opens / adds a proposition.
        if adv is not None and adv.get("violation") and adv.get("severity") == "high":
            cat = adv.get("category", "policy_compliance")
            kind = cat if cat in PROPOSITION_KINDS else "policy_compliance"
            reopened = Proposition(kind=kind)
            reopened.record("blocked")
            if not any(p.kind == kind for p in remaining):
                new_items.append(reopened)

        return plan_action, remaining, new_items

    # ----- single LLM calls --------------------------------------------

    def _single_action(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any]:
        """Baseline-style single next-action call (graceful fallback)."""
        prompt = json.dumps(
            {
                "task": "Choose exactly one next assistant action for this turn.",
                "available_tools": tools,
                "conversation_transcript": transcript,
                "rules": [
                    "Use only the supplied tool definitions.",
                    "Do not invent tool observations or unavailable capabilities.",
                    "If a capability/parameter is unavailable, say so transparently.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": CEREBRAS_DEVELOPER_INSTRUCTIONS},
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
            # Last-resort: never hard-error the turn. Defer to the user.
            ctx_logger.warning("Single-action fallback failed", error=str(exc))
            return {
                "action": "respond",
                "content": (
                    "Sorry, I had trouble processing that. Could you say it again?"
                ),
            }

    def _refine(
        self,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        plan_action: dict[str, Any],
        queue: list[Proposition],
        target: Proposition,
        ctx_logger: Any,
    ) -> dict[str, Any] | None:
        prompt = _sharpen_prompt(transcript, tools, plan_action, queue, target)
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": SHARPEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=SHARPEN_SCHEMA,
                response_schema_name="sharpen",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Sharpen refine failed", error=str(exc))
            return None
        try:
            payload = json.loads(res.text)
            action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning("Malformed sharpen output", error=str(exc))
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
    ) -> dict[str, Any] | None:
        prompt = _adversarial_prompt(transcript, tools, plan_action)
        try:
            res = self._client(1).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": ADVERSARIAL_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=ADVERSARIAL_SCHEMA,
                response_schema_name="adversarial",
                max_completion_tokens=self.config.adversarial_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Adversarial pass failed", error=str(exc))
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

    def _final_review(
        self,
        *,
        transcript: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        plan_action: dict[str, Any],
        tally: _CallTally,
        ctx_logger: Any,
    ) -> dict[str, Any] | None:
        prompt = _final_review_prompt(transcript, tools, plan_action)
        try:
            res = self._client(0).generate(
                model=self.model,
                messages=[
                    {"role": "system", "content": FINAL_REVIEW_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_schema=FINAL_REVIEW_SCHEMA,
                response_schema_name="final_review",
                max_completion_tokens=self.config.sharpen_max_completion_tokens,
                temperature=self.config.sharpen_temperature,
                reasoning_effort=self.reasoning_effort,
            )
        except (CerebrasTemplateError, json.JSONDecodeError) as exc:
            ctx_logger.warning("Final review failed", error=str(exc))
            return None

        tally.add(res, sequential=True)
        try:
            payload = json.loads(res.text)
            reviewed_action = parse_next_action(res.text)
        except (json.JSONDecodeError, MalformedModelResponseError) as exc:
            ctx_logger.warning("Malformed final review output", error=str(exc))
            return None

        approved = bool(payload.get("approved"))
        if approved:
            return plan_action

        ctx_logger.info(
            "Final review revised action",
            issue_category=payload.get("issue_category", "other"),
            explanation=payload.get("explanation", ""),
            action=reviewed_action.get("action"),
        )
        return reviewed_action

    # ----- termination --------------------------------------------------

    def _defer_action(
        self, queue: list[Proposition], plan_action: dict[str, Any]
    ) -> dict[str, Any]:
        """Budget exhausted with unresolved uncertainty -> defer, don't guess.

        feasibility/tool_availability blocks -> acknowledge a limit.
        parameter/policy uncertainty -> ask one clarifying question.
        """
        kinds = {p.kind for p in queue}
        limit_kinds = {"feasibility", "tool_availability"}
        if kinds & limit_kinds:
            content = (
                "I can't complete that with the controls available to me right "
                "now, so I don't want to guess. Could you tell me how you'd like "
                "to proceed?"
            )
        else:
            content = (
                "I want to get this right before acting. Could you confirm the "
                "specific detail you'd like, so I don't assume the wrong thing?"
            )
        return {"action": "respond", "content": content}

    # ----- client pool --------------------------------------------------

    def _pool_size(self, width: int) -> int:
        return max(1, min(width, self.config.max_parallel_clients))

    def _client(self, idx: int) -> CerebrasCompletionClient:
        size = self.config.max_parallel_clients
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


# ----- parsing helpers --------------------------------------------------


def _scatter_messages_for_pass(
    *, prompt: str, pass_index: int, diverse: bool
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _scatter_system_for_pass(pass_index, diverse=diverse),
        },
        {"role": "user", "content": prompt},
    ]


def _scatter_system_for_pass(pass_index: int, *, diverse: bool) -> str:
    if not diverse:
        return SCATTER_SYSTEM
    persona = SCATTER_PERSONAS[pass_index % len(SCATTER_PERSONAS)]
    return (
        SCATTER_SYSTEM
        + "\n\nDIVERSITY LENS FOR THIS PASS: "
        + persona
        + "\nApply this lens as an emphasis only. Use only the supplied "
        "system prompt/policies, interaction trace, and tool schemas. Do not "
        "compare tools against any outside catalog, infer scenario category, "
        "or optimize for any external score."
    )


def _scatter_temperature_for_pass(
    *,
    pass_index: int,
    width: int,
    diverse: bool,
    uniform_temperature: float | None,
) -> float | None:
    if not diverse:
        return uniform_temperature
    if width <= 1:
        return _DIVERSE_SCATTER_TEMPERATURE_SINGLE
    span = _DIVERSE_SCATTER_TEMPERATURE_MAX - _DIVERSE_SCATTER_TEMPERATURE_MIN
    ratio = pass_index / (width - 1)
    return round(_DIVERSE_SCATTER_TEMPERATURE_MIN + span * ratio, 3)


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
    names = sorted(tc.get("tool_name", "") for tc in action.get("tool_calls", []))
    return "tool_calls:" + ",".join(names)


def _protect_read_only_plan(
    *,
    plan_action: dict[str, Any],
    queue: list[Proposition],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[Proposition], bool]:
    if not _is_read_only_action(plan_action):
        return plan_action, queue, False
    fixed_action = _fill_missing_read_arguments(
        plan_action, tools, transcript, plan_action
    )
    relevant_action = _filter_relevant_read_only_action(
        fixed_action, tools, transcript
    )
    if relevant_action is None:
        return plan_action, queue, False
    remaining = [p for p in queue if p.kind != "parameter_determinacy"]
    return relevant_action, remaining, True


def _filter_relevant_read_only_action(
    action: dict[str, Any],
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> dict[str, Any] | None:
    tools_by_name = _tool_by_name(tools)
    latest_user = _latest_text_by_role(transcript, "user")
    query_tokens = _meaningful_tokens(latest_user)
    read_seen_after_latest_user = _any_read_tool_after_latest_user(transcript)
    kept_calls: list[dict[str, Any]] = []

    for tool_call in action.get("tool_calls", []) or []:
        name = tool_call.get("tool_name", "")
        tool = tools_by_name.get(name)
        if tool is None or not _is_read_tool_name(name):
            continue

        if _is_preference_read_tool(tool):
            if read_seen_after_latest_user and not _preference_reference_present(
                transcript, action
            ):
                continue
            if not _preference_reference_present(transcript, action):
                continue
            kept_calls.append(
                {"tool_name": name, "arguments": tool_call.get("arguments") or {}}
            )
            continue

        arguments = tool_call.get("arguments") or {}
        if _required_string_arguments_reuse_value(tool, arguments):
            continue
        if _strong_token_overlap_count(query_tokens, _tool_search_tokens(tool)) <= 0:
            continue
        if read_seen_after_latest_user and not _direct_context_read_relevance(
            tool=tool,
            query_tokens=query_tokens,
            action_tokens=set(),
        ):
            continue
        score = _context_read_score(
            tool=tool,
            arguments=arguments,
            transcript=transcript,
            query_text=latest_user,
            latest_user_tokens=query_tokens,
            action_tokens=set(),
            query_tokens=query_tokens,
            requires_action_relevance=False,
            allow_argument_grounding=False,
        )
        if score <= 0:
            continue
        kept_calls.append(
            {"tool_name": name, "arguments": arguments}
        )

    if not kept_calls:
        return None
    return {"action": "tool_calls", "content": "", "tool_calls": kept_calls}


def _final_read_only_relevance_repair(
    *,
    plan_action: dict[str, Any],
    passes: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_read_only_action(plan_action):
        return None
    relevant_action = _filter_relevant_read_only_action(
        plan_action, tools, transcript
    )
    if relevant_action is not None:
        if relevant_action.get("tool_calls") == plan_action.get("tool_calls"):
            return None
        return relevant_action

    tool_calls = _best_state_change_calls_from_passes(passes, tools)
    if not tool_calls:
        tool_calls = _user_requested_action_calls(transcript=transcript, tools=tools)
    tool_calls = _numeric_repaired_tool_calls(tool_calls, transcript, tools)
    tool_calls = _filter_explicitly_rejected_tool_calls(
        tool_calls,
        transcript,
        tools,
    )
    if tool_calls:
        repaired_action = {
            "action": "tool_calls",
            "content": "",
            "tool_calls": tool_calls,
        }
        forced_read = _forced_read_action(
            passes=passes,
            queue=[],
            plan_action=repaired_action,
            transcript=transcript,
            tools=tools,
        )
        if forced_read is not None:
            return forced_read[1]
        if _scatter_suggests_confirmation(passes):
            confirmation = _confirmation_response_for_tool_calls(tool_calls, tools)
            if confirmation is not None:
                return confirmation
        return repaired_action

    return {
        "action": "respond",
        "content": "Could you clarify the specific setting you want?",
    }


def _mechanical_guard_action(
    *,
    plan_action: dict[str, Any],
    passes: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    protected_read: bool,
) -> tuple[str, dict[str, Any]] | None:
    """Apply narrow non-LLM guards after the compute-heavy decision steps."""

    unavailable_tool_repair = _unavailable_tool_call_repair(
        plan_action=plan_action,
        transcript=transcript,
        tools=tools,
    )
    if unavailable_tool_repair is not None:
        return "unavailable_tool", unavailable_tool_repair

    guard_checks = (
        (
            "ungrounded_action",
            lambda: _prune_ungrounded_action_calls(
                plan_action=plan_action,
                transcript=transcript,
                tools=tools,
            ),
        ),
        (
            "explicit_rejection",
            lambda: _prune_rejected_action_calls(
                plan_action=plan_action,
                transcript=transcript,
                tools=tools,
            ),
        ),
        (
            "confirmation_detail",
            lambda: _schema_detailed_confirmation_repair(
                plan_action=plan_action,
                passes=passes,
                transcript=transcript,
                tools=tools,
            ),
        ),
    )
    for name, check in guard_checks:
        repaired = check()
        if repaired is not None:
            return name, repaired

    if _is_read_only_action(plan_action) and not protected_read:
        relevant_action = _filter_relevant_read_only_action(
            plan_action, tools, transcript
        )
        if relevant_action is None:
            return (
                "read_only_relevance",
                {
                    "action": "respond",
                    "content": "Could you clarify the specific setting you want?",
                },
            )
        if relevant_action.get("tool_calls") != plan_action.get("tool_calls"):
            return "read_only_relevance", relevant_action

    post_action_repair = _post_action_completion_guard(
        plan_action=plan_action,
        transcript=transcript,
        tools=tools,
    )
    if post_action_repair is not None:
        return "post_action_completion", post_action_repair
    return None


def _post_action_completion_guard(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _successful_action_tool_after_latest_user(transcript, tools):
        return None

    if plan_action.get("action") == "tool_calls":
        raw_calls = plan_action.get("tool_calls", []) or []
        remaining_calls = _remaining_after_successful_action_calls(
            requested_calls=raw_calls,
            transcript=transcript,
            tools=tools,
        )
        if len(remaining_calls) == len(raw_calls):
            return None
        if remaining_calls:
            repaired = dict(plan_action)
            repaired["tool_calls"] = remaining_calls
            return repaired
        return _post_action_success_response(transcript, tools)

    if _is_read_only_action(plan_action):
        return _post_action_success_response(transcript, tools)

    if plan_action.get("action") != "respond":
        return None
    content = plan_action.get("content") or ""
    if _looks_like_parameter_question(content) or _looks_like_confirmation_question(
        content
    ):
        return _post_action_success_response(transcript, tools)
    return None


def _post_action_completion_repair(
    *,
    plan_action: dict[str, Any],
    passes: list[dict[str, Any]] | None = None,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _successful_action_tool_after_latest_user(transcript, tools):
        return None
    confirmed_remaining = _remaining_confirmed_sequence_action_calls(
        transcript=transcript,
        tools=tools,
    )
    if confirmed_remaining is not None:
        if confirmed_remaining:
            remaining_action = {
                "action": "tool_calls",
                "content": "",
                "tool_calls": confirmed_remaining,
            }
            forced_read = _forced_disambiguation_read(
                passes=passes or [],
                queue=[],
                plan_action=remaining_action,
                transcript=transcript,
                tools=tools,
            )
            if forced_read is not None:
                return forced_read
            return remaining_action
        return _post_action_success_response(transcript, tools)
    remaining_calls = _remaining_user_requested_action_calls(
        transcript=transcript,
        tools=tools,
    )
    if remaining_calls:
        remaining_action = {
            "action": "tool_calls",
            "content": "",
            "tool_calls": remaining_calls,
        }
        forced_read = _forced_disambiguation_read(
            passes=passes or [],
            queue=[],
            plan_action=remaining_action,
            transcript=transcript,
            tools=tools,
        )
        if forced_read is not None:
            return forced_read
        return remaining_action
    if _is_read_only_action(plan_action):
        return _post_action_success_response(transcript, tools)
    if plan_action.get("action") != "respond":
        return None

    content = plan_action.get("content") or ""
    if _looks_like_parameter_question(content) or _looks_like_confirmation_question(
        content
    ):
        return _post_action_success_response(transcript, tools)
    return None


def _post_action_success_response(
    transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    details = _completed_tool_call_detail_strings(
        _successful_action_tool_calls_after_latest_user(transcript, tools),
        tools,
    )
    if not details:
        return {"action": "respond", "content": "Done."}
    return {
        "action": "respond",
        "content": f"Done. I completed {'; then '.join(details)}.",
    }


def _completed_tool_call_detail_strings(
    tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[str]:
    details = []
    for detail in _tool_call_detail_strings(tool_calls, tools):
        if detail.startswith("call "):
            details.append(detail[5:])
        else:
            details.append(detail)
    return details


def _remaining_confirmed_sequence_action_calls(
    *, transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    prior_confirmation = _prior_assistant_confirmation_before_latest_user(transcript)
    if not prior_confirmation or not _latest_user_affirms(transcript):
        return None
    requested = _infer_confirmation_tool_calls(
        prior_confirmation,
        transcript,
        tools,
        include_latest_user=False,
    )
    if not requested:
        return None
    requested = _filter_explicitly_rejected_tool_calls(
        requested,
        transcript,
        tools,
    )
    return _remaining_after_successful_action_calls(
        requested_calls=requested,
        transcript=transcript,
        tools=tools,
    )


def _filter_indirect_remaining_calls(
    tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not tool_calls:
        return tool_calls

    latest_user = _latest_text_by_role(transcript, "user")
    tools_by_name = _tool_by_name(tools)
    action_tools = [
        tool for tool in tools if not _is_read_tool_name(_tool_name(tool))
    ]
    unique_tokens_by_name = _unique_tool_reference_tokens(action_tools)
    kept_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = str(tool_call.get("tool_name") or "")
        tool = tools_by_name.get(name)
        if tool is None:
            kept_calls.append(tool_call)
            continue
        grounding_tokens = unique_tokens_by_name.get(name) or _tool_reference_tokens(
            tool
        )
        if _direct_request_segments_for_tokens(latest_user, grounding_tokens):
            kept_calls.append(tool_call)
    return kept_calls


def _direct_request_segments_for_tokens(text: str, tokens: set[str]) -> list[str]:
    if not tokens:
        return []
    matching_segments: list[str] = []
    for segment in _request_segments(text):
        if not _segment_has_directive_marker(segment):
            continue
        if _strong_token_overlap_count(_meaningful_tokens(segment), tokens) > 0:
            matching_segments.append(segment)
    return matching_segments


def _request_segments(text: str) -> list[str]:
    parts = re.split(
        r"[.;!?\n]+|\s+(?:-|\u2013|\u2014)\s+|\bso(?:\s+that)?\b|\bin\s+order\s+to\b",
        text,
        flags=re.I,
    )
    return [part.strip() for part in parts if part.strip()]


def _segment_has_directive_marker(segment: str) -> bool:
    lowered = segment.lower()
    if "please" in lowered or "go ahead" in lowered or "proceed" in lowered:
        return True
    return bool(
        re.search(
            r"\b(?:set|change|adjust|move|open|close|enable|disable|turn|switch|activate|deactivate|start|stop|call|make|put)\b",
            lowered,
        )
    )


def _remaining_user_requested_action_calls(
    *, transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    requested = _confirmed_action_calls_from_latest_user(
        transcript=transcript,
        tools=tools,
    )
    if not requested:
        requested = _user_requested_action_calls(transcript=transcript, tools=tools)
    requested = _numeric_repaired_tool_calls(requested, transcript, tools)
    requested = _filter_explicitly_rejected_tool_calls(
        requested,
        transcript,
        tools,
    )
    requested = [
        tool_call
        for tool_call in requested
        if _tool_call_identity_grounded_in_recent_request(
            tool_call,
            transcript,
            tools,
        )
    ]
    if not requested:
        return []

    remaining = _remaining_after_successful_action_calls(
        requested_calls=requested,
        transcript=transcript,
        tools=tools,
    )
    if _successful_action_tool_calls_after_latest_user(transcript, tools):
        remaining = _filter_indirect_remaining_calls(
            remaining,
            transcript,
            tools,
        )
    return remaining


def _remaining_after_successful_action_calls(
    *,
    requested_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    successful = _successful_action_tool_calls_after_latest_user(transcript, tools)
    if not successful:
        return requested_calls

    consumed: set[int] = set()
    remaining: list[dict[str, Any]] = []
    for requested_call in requested_calls:
        match_index = next(
            (
                index
                for index, successful_call in enumerate(successful)
                if index not in consumed
                and _tool_calls_match_for_completion(
                    successful_call,
                    requested_call,
                    tools,
                )
            ),
            None,
        )
        if match_index is None:
            remaining.append(requested_call)
        else:
            consumed.add(match_index)
    return remaining


def _tool_call_identity_grounded_in_recent_request(
    tool_call: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    tool = _tool_by_name(tools).get(str(tool_call.get("tool_name") or ""))
    if tool is None:
        return False
    text = "\n".join(
        part
        for part in (
            _latest_text_by_role(transcript, "user"),
            _prior_assistant_confirmation_before_latest_user(transcript),
        )
        if part
    )
    text_tokens = _meaningful_tokens(text)
    identity_tokens = _meaningful_tokens(f"{_tool_name(tool)} {_tool_description(tool)}")
    return _strong_token_overlap_count(text_tokens, identity_tokens) > 0


def _tool_calls_match_for_completion(
    successful_call: dict[str, Any],
    requested_call: dict[str, Any],
    tools: list[dict[str, Any]],
) -> bool:
    if _tool_calls_match_for_confirmation(successful_call, requested_call, tools):
        return True
    if successful_call.get("tool_name") != requested_call.get("tool_name"):
        return False
    return not (successful_call.get("arguments") or {})


def _unavailable_tool_call_repair(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "tool_calls":
        return None

    raw_calls = plan_action.get("tool_calls", []) or []
    if not raw_calls:
        return None

    tools_by_name = _tool_by_name(tools)
    has_unavailable = any(
        str(tool_call.get("tool_name") or "") not in tools_by_name
        for tool_call in raw_calls
    )
    if not has_unavailable:
        return None

    if _successful_action_tool_after_latest_user(transcript, tools):
        return {"action": "respond", "content": "Done."}

    inferred = _numeric_repaired_tool_calls(
        _user_requested_action_calls(transcript=transcript, tools=tools),
        transcript,
        tools,
    )
    if inferred:
        return {"action": "tool_calls", "content": "", "tool_calls": inferred}

    available_calls = _numeric_repaired_tool_calls(
        _actionable_tool_calls(raw_calls, tools),
        transcript,
        tools,
    )
    if available_calls:
        return {"action": "tool_calls", "content": "", "tool_calls": available_calls}

    return {
        "action": "respond",
        "content": "I cannot do that with the available controls.",
    }


def _successful_action_tool_after_latest_user(
    transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> bool:
    return bool(_successful_action_tool_calls_after_latest_user(transcript, tools))


def _successful_action_tool_calls_after_latest_user(
    transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    latest_user_index = -1
    for index, message in enumerate(transcript):
        if message.get("role") == "user":
            latest_user_index = index
    if latest_user_index < 0:
        return []

    tools_by_name = _tool_by_name(tools)
    pending_calls: list[dict[str, Any]] = []
    successful_calls: list[dict[str, Any]] = []
    for message in transcript[latest_user_index + 1 :]:
        if message.get("role") == "assistant":
            pending_calls.extend(
                _actionable_tool_calls(message.get("tool_calls", []) or [], tools)
            )
            continue
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        tool = tools_by_name.get(name)
        if tool is None or _is_read_tool_name(name):
            continue
        if not _tool_message_succeeded(message):
            continue
        match_index = next(
            (
                index
                for index, tool_call in enumerate(pending_calls)
                if tool_call.get("tool_name") == name
            ),
            None,
        )
        if match_index is None:
            successful_calls.append({"tool_name": name, "arguments": {}})
            continue
        successful_calls.append(pending_calls.pop(match_index))
    return successful_calls


def _tool_message_succeeded(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, str):
        return True
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return True
    if not isinstance(payload, dict):
        return True
    status = str(payload.get("status") or "").lower()
    return not status or status == "success"


def _forced_read_action(
    *,
    passes: list[dict[str, Any]],
    queue: list[Proposition],
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    forced_read = _forced_disambiguation_read(
        passes=passes,
        queue=queue,
        plan_action=plan_action,
        transcript=transcript,
        tools=tools,
    )
    if forced_read is not None:
        return "disambiguation", forced_read

    forced_context_read = _forced_context_read(
        passes=passes,
        queue=queue,
        plan_action=plan_action,
        transcript=transcript,
        tools=tools,
    )
    if forced_context_read is not None:
        return "context", forced_context_read
    return None


def _forced_disambiguation_read(
    *,
    passes: list[dict[str, Any]],
    queue: list[Proposition],
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Replace a noisy clarify/guess vote with a deterministic read action.

    The rule uses only the per-task tool list and the visible transcript. It
    does not infer missing tools or task type from any external catalog.
    """
    preference_tools = _preference_read_tools(tools)
    if not preference_tools:
        return None

    tools_by_name = _tool_by_name(tools)
    available_read_names = {
        name for tool in preference_tools if (name := _tool_name(tool))
    }
    read_seen_after_latest_user = _any_read_tool_after_latest_user(transcript)
    already_read_names = _already_read_tool_names(transcript, available_read_names)
    if available_read_names <= already_read_names:
        return None
    preference_reference_present = _preference_reference_present(
        transcript, plan_action
    )
    preference_resolvable_question = _preference_resolvable_parameter_question(
        plan_action, transcript, tools
    )
    preference_resolvable_action = _preference_resolvable_action_arguments(
        plan_action, transcript, tools
    )
    preference_resolvable_confirmation = _preference_resolvable_confirmation_action(
        plan_action, passes, transcript, tools
    )
    if (
        read_seen_after_latest_user
        and not preference_reference_present
        and not preference_resolvable_question
        and not preference_resolvable_action
        and not preference_resolvable_confirmation
    ):
        return None

    if _is_read_only_action(plan_action):
        return None

    scatter_read = _best_scatter_read_action(
        passes=passes,
        available_read_names=available_read_names,
        already_read_names=already_read_names,
    )
    if scatter_read is not None:
        scatter_read = _fill_missing_read_arguments(
            scatter_read, tools, transcript, plan_action
        )
    if scatter_read is not None:
        return scatter_read

    if not (
        preference_reference_present
        or _scatter_indicates_parameter_gap(passes, queue, plan_action)
        or preference_resolvable_question
        or preference_resolvable_action
        or preference_resolvable_confirmation
    ):
        return None

    read_name = next(
        name for name in available_read_names if name not in already_read_names
    )
    return {
        "action": "tool_calls",
        "content": "",
        "tool_calls": [
            {
                "tool_name": read_name,
                "arguments": _preference_read_arguments(
                    tools_by_name[read_name], transcript, plan_action
                ),
            }
        ],
    }


def _forced_context_read(
    *,
    passes: list[dict[str, Any]],
    queue: list[Proposition],
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if _is_read_only_action(plan_action):
        return None
    if (
        plan_action.get("action") == "tool_calls"
        and _latest_user_confirmed_prior_confirmation(transcript)
    ):
        return None

    content = plan_action.get("content") or ""
    action_tool_calls: list[dict[str, Any]] = []
    if plan_action.get("action") == "respond":
        if _looks_like_confirmation_question(content):
            action_tool_calls = _actionable_tool_calls(
                _infer_confirmation_tool_calls(content, transcript, tools),
                tools,
            )
        elif _looks_like_parameter_question(content):
            action_tool_calls = _user_requested_action_calls(
                transcript=transcript,
                tools=tools,
            )
        policy_read_scores = _policy_required_read_scores(
            action_tool_calls=action_tool_calls,
            transcript=transcript,
            tools=tools,
        )
        ambiguous_action_family = False
        should_read = (
            _looks_like_parameter_question(content)
            or _looks_like_confirmation_question(content)
            or _scatter_indicates_parameter_gap(passes, queue, plan_action)
        )
        should_read = should_read or bool(policy_read_scores)
        action_text = "\n".join(
            part
            for part in (content, _tool_call_reference_text(action_tool_calls))
            if part
        )
    elif plan_action.get("action") == "tool_calls":
        action_tool_calls = _actionable_tool_calls(
            plan_action.get("tool_calls", []) or [], tools
        )
        policy_read_scores = _policy_required_read_scores(
            action_tool_calls=action_tool_calls,
            transcript=transcript,
            tools=tools,
        )
        ambiguous_action_family = _ambiguous_action_family(
            action_tool_calls=action_tool_calls,
            latest_user_tokens=_meaningful_tokens(
                _latest_text_by_role(transcript, "user")
            ),
            tools=tools,
        )
        should_read = bool(action_tool_calls) and _scatter_indicates_parameter_gap(
            passes, queue, plan_action
        )
        should_read = should_read or ambiguous_action_family
        should_read = should_read or bool(policy_read_scores)
        action_text = _tool_call_reference_text(action_tool_calls)
    else:
        policy_read_scores = {}
        ambiguous_action_family = False
        should_read = False
        action_text = ""

    if not should_read:
        return None

    tools_by_name = _tool_by_name(tools)
    available_read_names = {
        name
        for name, tool in tools_by_name.items()
        if _is_context_read_tool(tool)
    }
    if not available_read_names:
        return None

    read_seen_after_latest_user = _any_read_tool_after_latest_user(transcript)
    already_read_names = _already_read_tool_names(transcript, available_read_names)
    if available_read_names <= already_read_names:
        return None
    query_text = "\n".join(
        part
        for part in (_latest_text_by_role(transcript, "user"), action_text)
        if part
    )
    latest_user_tokens = _meaningful_tokens(_latest_text_by_role(transcript, "user"))
    action_tokens = _meaningful_tokens(action_text)
    query_tokens = _meaningful_tokens(query_text)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for name in available_read_names - already_read_names:
        tool = tools_by_name[name]
        policy_read_score = policy_read_scores.get(name, 0)
        if policy_read_scores and policy_read_score <= 0:
            continue
        arguments = _read_tool_arguments(tool, transcript, query_text)
        if arguments is None:
            continue
        if _required_string_arguments_reuse_value(tool, arguments):
            continue
        score = _context_read_score(
            tool=tool,
            arguments=arguments,
            transcript=transcript,
            query_text=query_text,
            latest_user_tokens=latest_user_tokens,
            action_tokens=action_tokens,
            query_tokens=query_tokens,
            requires_action_relevance=bool(action_tool_calls)
            and not ambiguous_action_family
            and policy_read_score <= 0,
            allow_argument_grounding=ambiguous_action_family
            or (
                plan_action.get("action") == "respond"
                and _looks_like_parameter_question(content)
            )
            or policy_read_score > 0,
        )
        if policy_read_score > 0:
            score = max(score, 100 + policy_read_score)
        if (
            read_seen_after_latest_user
            and policy_read_score <= 0
            and not _direct_context_read_relevance(
                tool=tool,
                query_tokens=query_tokens,
                action_tokens=action_tokens,
            )
        ):
            continue
        if score <= 0:
            continue
        candidates.append((score, name, arguments))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return {
        "action": "tool_calls",
        "content": "",
        "tool_calls": [
            {"tool_name": name, "arguments": arguments}
            for _, name, arguments in candidates[:2]
        ],
    }


def _policy_required_read_scores(
    *,
    action_tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, int]:
    if not action_tool_calls:
        return {}

    tools_by_name = _tool_by_name(tools)
    action_tokens: set[str] = set()
    for tool_call in action_tool_calls:
        tool = tools_by_name.get(tool_call.get("tool_name", ""))
        if tool is None:
            continue
        action_tokens.update(_tool_call_grounding_tokens(tool_call, tool))
    if not action_tokens:
        return {}

    scores: dict[str, int] = {}
    for segment in _policy_dependency_segments(transcript):
        if not _policy_segment_needs_read_before_action(segment):
            continue
        segment_tokens = _meaningful_tokens(segment)
        action_overlap = _strong_token_overlap_count(action_tokens, segment_tokens)
        if action_overlap <= 0:
            continue
        for tool in tools:
            name = _tool_name(tool)
            if not name or not _is_context_read_tool(tool):
                continue
            read_overlap = _strong_token_overlap_count(
                _tool_search_tokens(tool), segment_tokens
            )
            if read_overlap < 2:
                continue
            scores[name] = max(scores.get(name, 0), action_overlap + read_overlap)
    return scores


def _policy_segment_needs_read_before_action(segment: str) -> bool:
    lowered = segment.lower()
    obligation = (
        "must" in lowered
        or "should" in lowered
        or "need" in lowered
        or "needs" in lowered
        or "needed" in lowered
        or "necessary" in lowered
        or "require" in lowered
        or "required" in lowered
        or "has to" in lowered
        or "have to" in lowered
        or "can only" in lowered
        or "only if" in lowered
    )
    information = (
        "check" in lowered
        or "checked" in lowered
        or "retrieve" in lowered
        or "retrieved" in lowered
        or "read" in lowered
        or "available after" in lowered
    )
    ordering = "before" in lowered or "first" in lowered or "already" in lowered
    return obligation and information and ordering


def _preference_resolvable_parameter_question(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    if plan_action.get("action") != "respond":
        return False
    content = plan_action.get("content") or ""
    if not _looks_like_parameter_question(content):
        return False

    text = "\n".join(
        part for part in (_latest_text_by_role(transcript, "user"), content) if part
    )
    text_tokens = _meaningful_tokens(text)
    for tool in tools:
        name = _tool_name(tool)
        if not name or _is_read_tool_name(name):
            continue
        if not (text_tokens & _tool_search_tokens(tool)):
            continue
        if _tool_has_unresolved_nonboolean_required_argument(tool, text):
            return True
    return False


def _preference_resolvable_action_arguments(
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    if plan_action.get("action") != "tool_calls":
        return False

    latest_user = _latest_text_by_role(transcript, "user")
    if not latest_user.strip():
        return False

    latest_numeric_segments = _numeric_segments_from_text(latest_user)
    lowered_user = latest_user.lower()
    tools_by_name = _tool_by_name(tools)
    for tool_call in _actionable_tool_calls(
        plan_action.get("tool_calls", []) or [],
        tools,
    ):
        tool = tools_by_name.get(str(tool_call.get("tool_name") or ""))
        if tool is None:
            continue
        arguments = tool_call.get("arguments") or {}
        properties = _tool_parameters(tool).get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for arg_name in _required_argument_names(tool):
            schema = properties.get(arg_name)
            if not isinstance(schema, dict):
                continue
            if schema.get("type") == "boolean":
                continue
            if arg_name not in arguments:
                return True
            value = arguments[arg_name]
            if _is_numeric_schema(schema):
                if (
                    _numeric_value_mentioned_for_schema(
                        value,
                        schema,
                        latest_numeric_segments,
                    )
                    and _numeric_value_anchored_to_tool(
                        value,
                        tool,
                        latest_numeric_segments,
                    )
                ):
                    continue
                return True
            if _argument_detail_present(lowered_user, arg_name, value, schema):
                continue
            return True
    return False


def _preference_resolvable_confirmation_action(
    plan_action: dict[str, Any],
    passes: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    if plan_action.get("action") != "respond":
        return False
    content = plan_action.get("content") or ""
    if not _looks_like_confirmation_question(content):
        return False

    tool_calls = _actionable_tool_calls(
        _infer_confirmation_tool_calls(content, transcript, tools),
        tools,
    )
    if not tool_calls:
        tool_calls = _best_state_change_calls_from_passes(passes, tools)
    if not tool_calls:
        return False
    return _preference_resolvable_action_arguments(
        {"action": "tool_calls", "content": "", "tool_calls": tool_calls},
        transcript,
        tools,
    )


def _tool_has_unresolved_nonboolean_required_argument(
    tool: dict[str, Any], text: str
) -> bool:
    properties = _tool_parameters(tool).get("properties", {})
    if not isinstance(properties, dict):
        return False
    for name in _required_argument_names(tool):
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict):
            continue
        if prop_schema.get("type") == "boolean":
            continue
        if _infer_schema_value_from_text(prop_schema, text) is None:
            return True
    return False


def _scatter_indicates_parameter_gap(
    passes: list[dict[str, Any]],
    queue: list[Proposition],
    plan_action: dict[str, Any],
) -> bool:
    if not passes:
        return False

    total = len(passes)
    param_unresolved = sum(
        1
        for p in passes
        if p.get("dispositions", {}).get("parameter_determinacy") != "ok"
    )
    value_questions = sum(
        1
        for p in passes
        if p.get("action", {}).get("action") == "respond"
        and _looks_like_parameter_question(p.get("action", {}).get("content") or "")
    )
    confirmation_questions = sum(
        1
        for p in passes
        if p.get("action", {}).get("action") == "respond"
        and _looks_like_confirmation_question(
            p.get("action", {}).get("content") or ""
        )
    )

    if confirmation_questions / total >= 1 / 2 and value_questions == 0:
        return False

    return (
        any(p.kind == "parameter_determinacy" for p in queue)
        or param_unresolved / total >= 1 / 3
        or (
            plan_action.get("action") == "respond"
            and _looks_like_parameter_question(plan_action.get("content") or "")
            and value_questions / total >= 1 / 2
        )
    )


def _looks_like_parameter_question(content: str) -> bool:
    lowered = content.lower()
    if "?" not in lowered:
        return False
    value_markers = (
        "which",
        "what",
        "how many",
        "how much",
        "amount",
        "detail",
        "level",
        "option",
        "parameter",
        "setting",
        "specific",
        "value",
    )
    return any(marker in lowered for marker in value_markers)


def _looks_like_confirmation_question(content: str) -> bool:
    lowered = content.lower()
    confirmation_markers = (
        "confirm",
        "confirmation",
        "do you want",
        "do you want me to",
        "would you like me to",
        "should i",
        "shall i",
        "may i",
        "are you sure",
        "is that okay",
        "is this ok",
        "is this okay",
    )
    imperative_markers = (
        "please say yes",
        "say yes",
        "reply yes",
        "respond yes",
    )
    if any(marker in lowered for marker in imperative_markers):
        return True
    if "?" not in lowered and not any(
        marker in lowered for marker in ("confirm", "confirmation")
    ):
        return False
    return any(marker in lowered for marker in confirmation_markers)


def _numeric_argument_scope_repair(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "tool_calls":
        return None

    segments = _numeric_reference_segments(transcript)

    tools_by_name = _tool_by_name(tools)
    changed = False
    repaired_calls: list[dict[str, Any]] = []
    for tool_call in plan_action.get("tool_calls", []) or []:
        name = tool_call.get("tool_name", "")
        tool = tools_by_name.get(name)
        arguments = dict(tool_call.get("arguments") or {})
        if tool is None or not arguments or _is_read_tool_name(name):
            repaired_calls.append({"tool_name": name, "arguments": arguments})
            continue

        properties = _tool_parameters(tool).get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        for arg_name, value in list(arguments.items()):
            if not _is_number(value):
                continue
            prop_schema = properties.get(arg_name)
            if not isinstance(prop_schema, dict) or not _is_numeric_schema(prop_schema):
                continue
            if not _numeric_value_mentioned_for_schema(value, prop_schema, segments):
                replacement = _schema_neutral_numeric_value(prop_schema)
                if replacement is not None and not _numbers_close(
                    float(replacement), float(value)
                ):
                    arguments[arg_name] = replacement
                    changed = True
                continue
            owners = _numeric_value_owner_names(value, tools, segments)
            current_name = _tool_name(tool)
            if current_name in owners:
                continue
            if not any(owner != current_name for owner in owners):
                continue

            replacement = _numeric_schema_value_anchored_to_tool(
                tool=tool,
                tools=tools,
                schema=prop_schema,
                segments=segments,
            )
            if replacement is None:
                replacement = _schema_neutral_numeric_value(prop_schema)
            if replacement is None or _numbers_close(float(replacement), float(value)):
                continue
            arguments[arg_name] = replacement
            changed = True

        repaired_calls.append({"tool_name": name, "arguments": arguments})

    if not changed:
        return None

    repaired = dict(plan_action)
    repaired["tool_calls"] = repaired_calls
    return repaired


def _numeric_reference_segments(transcript: list[dict[str, Any]]) -> list[str]:
    return _numeric_segments_from_text(
        _all_text_by_role(transcript, "user")
    )


def _numeric_segments_from_text(text: str) -> list[str]:
    segments: list[str] = []
    for segment in re.split(
        r"[.;!?]|\band\s+then\b|\bthen\b|\bafter\s+that\b",
        text,
        flags=re.IGNORECASE,
    ):
        if _numbers_in_text(segment) or _qualitative_numeric_markers(segment):
            segments.append(segment)
    return segments


def _numeric_value_mentioned_for_schema(
    value: Any, schema: dict[str, Any], segments: list[str]
) -> bool:
    return any(
        _numbers_close(number, float(value))
        for segment in segments
        for number, _start, _end in _schema_numeric_value_matches_in_text(
            segment, schema
        )
    )


def _numeric_value_owner_names(
    value: Any,
    tools: list[dict[str, Any]],
    segments: list[str],
) -> set[str]:
    if not _is_number(value):
        return set()

    token_map = _discriminating_tool_reference_tokens(tools)
    owners: set[str] = set()
    for segment in segments:
        for number, start, _end in _number_matches_in_text(segment):
            if not _numbers_close(number, float(value)):
                continue
            distances: list[tuple[int, str]] = []
            for tool in tools:
                name = _tool_name(tool)
                if not name or _is_read_tool_name(name):
                    continue
                positions = _token_positions(segment, token_map.get(name, set()))
                if positions:
                    distances.append((min(abs(pos - start) for pos in positions), name))
            if distances:
                best_distance = min(distance for distance, _ in distances)
                owners.update(
                    name for distance, name in distances if distance == best_distance
                )
        for marker, start, _end in _qualitative_numeric_markers(segment):
            distances = []
            for tool in tools:
                name = _tool_name(tool)
                if not name or _is_read_tool_name(name):
                    continue
                if not _tool_qualitative_marker_matches_value(tool, value, marker):
                    continue
                positions = _token_positions(segment, token_map.get(name, set()))
                if positions:
                    distances.append((min(abs(pos - start) for pos in positions), name))
            if distances:
                best_distance = min(distance for distance, _ in distances)
                owners.update(
                    name for distance, name in distances if distance == best_distance
                )
    return owners


def _numeric_schema_value_anchored_to_tool(
    *,
    tool: dict[str, Any],
    tools: list[dict[str, Any]],
    schema: dict[str, Any],
    segments: list[str],
) -> int | float | None:
    current_name = _tool_name(tool)
    if not current_name:
        return None

    candidates: list[float] = []
    for segment in segments:
        for number, _start, _end in _schema_numeric_value_matches_in_text(
            segment, schema
        ):
            owners = _numeric_value_owner_names(number, tools, [segment])
            if owners == {current_name}:
                candidates.append(number)

    unique: list[float] = []
    for candidate in candidates:
        if not any(_numbers_close(candidate, existing) for existing in unique):
            unique.append(candidate)
    if len(unique) != 1:
        return None
    return _coerce_numeric_for_schema(unique[0], schema)


def _numeric_value_anchored_to_tool(
    value: Any, tool: dict[str, Any], segments: list[str]
) -> bool:
    return _tool_name(tool) in _numeric_value_owner_names(value, [tool], segments)


def _numeric_value_anchored_to_other_tool(
    value: Any,
    current_tool: dict[str, Any],
    tools: list[dict[str, Any]],
    segments: list[str],
) -> bool:
    current_name = _tool_name(current_tool)
    return any(
        name != current_name
        for name in _numeric_value_owner_names(value, tools, segments)
    )


def _numbers_in_text(text: str) -> list[float]:
    return [number for number, _start, _end in _number_matches_in_text(text)]


def _number_matches_in_text(text: str) -> list[tuple[float, int, int]]:
    return [
        (float(match.group(0)), match.start(), match.end())
        for match in re.finditer(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?", text)
    ]


def _schema_numeric_value_matches_in_text(
    text: str, schema: dict[str, Any]
) -> list[tuple[float, int, int]]:
    matches = [
        (number, start, end)
        for number, start, end in _number_matches_in_text(text)
        if _number_in_schema_range(number, schema)
    ]
    for marker, start, end in _qualitative_numeric_markers(text):
        value = _qualitative_numeric_value_for_schema(marker, schema)
        if value is not None:
            matches.append((float(value), start, end))
    return matches


def _qualitative_numeric_markers(text: str) -> list[tuple[str, int, int]]:
    patterns = (
        ("midpoint", r"\bhalfway\b|\bhalf\s+way\b|\bhalf\b"),
        ("maximum", r"\ball\s+the\s+way\b|\bfully\b|\bfull\b|\bcompletely\b|\bmaximum\b|\bmax\b"),
        ("minimum", r"\bminimum\b|\bmin\b|\bzero\b|\bclosed\b"),
    )
    markers: list[tuple[str, int, int]] = []
    for marker, pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            if _qualitative_marker_is_negated(text, match.start()):
                continue
            markers.append((marker, match.start(), match.end()))
    markers.sort(key=lambda item: item[1])
    return markers


def _qualitative_marker_is_negated(text: str, marker_start: int) -> bool:
    preceding = text[max(0, marker_start - 24) : marker_start].lower()
    return bool(
        re.search(r"\b(?:not|never|no|dont|don't|do\s+not)\s+(?:\w+\s+){0,2}$", preceding)
    )


def _qualitative_numeric_value_for_schema(
    marker: str, schema: dict[str, Any]
) -> int | float | None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if not (_is_number(minimum) and _is_number(maximum)):
        return None
    if marker == "maximum":
        value = float(maximum)
    elif marker == "minimum":
        value = float(minimum)
    elif marker == "midpoint":
        value = (float(minimum) + float(maximum)) / 2
    else:
        return None
    return _coerce_numeric_for_schema(value, schema)


def _tool_qualitative_marker_matches_value(
    tool: dict[str, Any], value: Any, marker: str
) -> bool:
    if not _is_number(value):
        return False
    for schema in _tool_numeric_argument_schemas(tool):
        mapped = _qualitative_numeric_value_for_schema(marker, schema)
        if mapped is not None and _numbers_close(float(mapped), float(value)):
            return True
    return False


def _tool_numeric_argument_schemas(tool: dict[str, Any]) -> list[dict[str, Any]]:
    properties = _tool_parameters(tool).get("properties", {})
    if not isinstance(properties, dict):
        return []
    return [
        schema
        for schema in properties.values()
        if isinstance(schema, dict) and _is_numeric_schema(schema)
    ]


def _infer_numeric_schema_value_from_anchored_text(
    *, name: str, schema: dict[str, Any], text: str
) -> int | float | None:
    anchors = _meaningful_tokens(f"{name} {schema.get('description') or ''}")
    if not anchors:
        return None

    for value, start, end in _schema_numeric_value_matches_in_text(text, schema):
        nearby_text = text[max(0, start - 48) : min(len(text), end + 48)]
        if _token_overlap_count(_meaningful_tokens(nearby_text), anchors) > 0:
            return _coerce_numeric_for_schema(value, schema)
    return None


def _tool_reference_tokens(tool: dict[str, Any]) -> set[str]:
    tokens = _meaningful_tokens(_tool_name(tool))
    return tokens if tokens else _meaningful_tokens(_tool_description(tool))


def _discriminating_tool_reference_tokens(
    tools: list[dict[str, Any]]
) -> dict[str, set[str]]:
    raw: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for tool in tools:
        name = _tool_name(tool)
        if not name or _is_read_tool_name(name):
            continue
        tokens = _tool_reference_tokens(tool)
        raw[name] = tokens
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return {
        name: {token for token in tokens if counts.get(token) == 1} or tokens
        for name, tokens in raw.items()
    }


def _token_positions(text: str, tokens: set[str]) -> list[int]:
    if not tokens:
        return []
    lowered = text.lower().replace("_", " ")
    positions: list[int] = []
    for token in tokens:
        pattern = re.compile(rf"\b{re.escape(token.lower())}\b")
        positions.extend(match.start() for match in pattern.finditer(lowered))
    return positions


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_schema(schema: dict[str, Any]) -> bool:
    return schema.get("type") in {"integer", "number"}


def _schema_neutral_numeric_value(schema: dict[str, Any]) -> int | float | None:
    if "default" in schema and _is_number(schema["default"]):
        return _coerce_numeric_for_schema(float(schema["default"]), schema)

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if not (_is_number(minimum) and _is_number(maximum)):
        return None
    midpoint = (float(minimum) + float(maximum)) / 2
    return _coerce_numeric_for_schema(midpoint, schema)


def _coerce_numeric_for_schema(value: float, schema: dict[str, Any]) -> int | float:
    if schema.get("type") == "integer":
        return int(round(value))
    return value


def _numbers_close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9


def _schema_grounded_clarification_repair(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "respond":
        return None
    content = plan_action.get("content") or ""
    if not _looks_like_parameter_question(content):
        return None

    tool_calls = _user_requested_action_calls(transcript=transcript, tools=tools)
    if not tool_calls:
        return None
    return {"action": "tool_calls", "content": "", "tool_calls": tool_calls}


def _context_grounded_choice_repair(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "respond":
        return None
    content = plan_action.get("content") or ""
    if "?" not in content:
        return None
    lowered_content = content.lower()
    if " or " not in lowered_content and not _looks_like_parameter_question(content):
        return None
    if not _any_read_tool_after_latest_user(transcript):
        return None

    candidates = _context_choice_action_candidates(
        content=content,
        transcript=transcript,
        tools=tools,
    )
    if len(candidates) < 2:
        return None

    tools_by_name = _tool_by_name(tools)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for call in candidates:
        name = str(call.get("tool_name") or "")
        tool = tools_by_name.get(name)
        if tool is None:
            continue
        score = _context_choice_candidate_score(
            tool=tool,
            tool_call=call,
            content=content,
            transcript=transcript,
        )
        if score <= 0:
            continue
        scored.append((score, name, call))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    if len(scored) > 1 and scored[0][0] <= scored[1][0] + 2:
        return None
    tool_calls = [scored[0][2]]
    if _policy_requires_confirmation_for_actions(
        action_tool_calls=tool_calls,
        transcript=transcript,
        tools=tools,
    ):
        return _policy_context_confirmation_response_for_tool_calls(
            tool_calls,
            tools,
            transcript,
        )
    return {"action": "tool_calls", "content": "", "tool_calls": tool_calls}


def _context_choice_action_candidates(
    *,
    content: str,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_user = _latest_text_by_role(transcript, "user")
    text = "\n".join(part for part in (latest_user, content) if part)
    text_tokens = _meaningful_tokens(text)
    content_tokens = _meaningful_tokens(content)
    numeric_segments = _numeric_segments_from_text(text)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name or _is_read_tool_name(name):
            continue
        tool_tokens = _tool_reference_tokens(tool)
        if _token_overlap_count(content_tokens, tool_tokens) <= 0:
            continue
        if _token_overlap_count(text_tokens, tool_tokens) <= 0:
            continue
        arguments = _infer_required_arguments_for_confirmed_action(
            tool=tool,
            tools=tools,
            numeric_segments=numeric_segments,
            text=text,
        )
        if arguments is None:
            continue
        position = _tool_request_position(content, tool_tokens)
        candidates.append((position, name, {"tool_name": name, "arguments": arguments}))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [call for _position, _name, call in candidates]


def _context_choice_candidate_score(
    *,
    tool: dict[str, Any],
    tool_call: dict[str, Any],
    content: str,
    transcript: list[dict[str, Any]],
) -> int:
    tool_tokens = _tool_reference_tokens(tool)
    latest_user_tokens = _meaningful_tokens(_latest_text_by_role(transcript, "user"))
    content_tokens = _meaningful_tokens(content)
    score = 2 * _strong_token_overlap_count(content_tokens, tool_tokens)
    score += _strong_token_overlap_count(latest_user_tokens, tool_tokens)

    state_delta = _context_choice_boolean_state_delta(
        tool=tool,
        tool_call=tool_call,
        transcript=transcript,
    )
    if state_delta == "already":
        return -1
    if state_delta == "changes":
        score += 5

    score += _context_choice_policy_score(tool=tool, transcript=transcript)
    return score


def _context_choice_policy_score(
    *, tool: dict[str, Any], transcript: list[dict[str, Any]]
) -> int:
    tool_tokens = _tool_reference_tokens(tool)
    latest_user_tokens = _meaningful_tokens(_latest_text_by_role(transcript, "user"))
    if not tool_tokens or not latest_user_tokens:
        return 0

    score = 0
    for segment in _policy_dependency_segments(transcript):
        segment_tokens = _meaningful_tokens(segment)
        action_overlap = _strong_token_overlap_count(tool_tokens, segment_tokens)
        if action_overlap <= 0:
            continue
        evidence_score = _policy_read_evidence_score(
            segment_tokens=segment_tokens,
            latest_user_tokens=latest_user_tokens,
            transcript=transcript,
        )
        if evidence_score <= 0:
            continue
        score += action_overlap * 10 + evidence_score
        if _policy_segment_mentions_confirmation(segment):
            score += 2
    return score


def _policy_read_evidence_score(
    *,
    segment_tokens: set[str],
    latest_user_tokens: set[str],
    transcript: list[dict[str, Any]],
) -> int:
    score = 0
    for message in _messages_after_latest_user(transcript):
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        if not _is_read_tool_name(name):
            continue
        name_overlap = _strong_token_overlap_count(
            _meaningful_tokens(name),
            segment_tokens,
        )
        if name_overlap <= 0:
            continue
        value_overlap = _strong_token_overlap_count(
            _meaningful_tokens(" ".join(_tool_message_scalar_summary_values(message))),
            latest_user_tokens,
        )
        if value_overlap <= 0:
            continue
        score += name_overlap * 8 + value_overlap * 4
    return score


def _policy_segment_mentions_confirmation(segment: str) -> bool:
    return bool(
        re.search(
            r"\b(confirm|confirmation|approve|approval|permission|explicit)\b",
            segment,
            flags=re.I,
        )
    )


def _recent_read_context_value_tokens(transcript: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for message in _messages_after_latest_user(transcript):
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        if not _is_read_tool_name(name):
            continue
        tokens.update(_meaningful_tokens(name))
        tokens.update(_meaningful_tokens(" ".join(_tool_message_scalar_summary_values(message))))
    return tokens


def _context_choice_boolean_state_delta(
    *,
    tool: dict[str, Any],
    tool_call: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> str:
    target_value = _single_boolean_argument_value(tool, tool_call.get("arguments") or {})
    if target_value is None:
        return ""

    current_value = _best_boolean_read_value_for_tool(
        tool=tool,
        transcript=transcript,
    )
    if current_value is None:
        return ""
    return "already" if current_value == target_value else "changes"


def _single_boolean_argument_value(
    tool: dict[str, Any], arguments: dict[str, Any]
) -> bool | None:
    properties = _tool_parameters(tool).get("properties", {})
    if not isinstance(properties, dict):
        return None
    values: list[bool] = []
    for name, schema in properties.items():
        if not isinstance(schema, dict) or schema.get("type") != "boolean":
            continue
        value = arguments.get(name)
        if isinstance(value, bool):
            values.append(value)
    if len(values) != 1:
        return None
    return values[0]


def _best_boolean_read_value_for_tool(
    *, tool: dict[str, Any], transcript: list[dict[str, Any]]
) -> bool | None:
    tool_tokens = _tool_reference_tokens(tool)
    best: tuple[int, int, bool] | None = None
    for index, message in enumerate(_messages_after_latest_user(transcript)):
        if message.get("role") != "tool" or not _is_read_tool_name(
            str(message.get("name") or "")
        ):
            continue
        for key, value in _boolean_items_from_tool_message(message):
            key_tokens = _meaningful_tokens(key)
            overlap = _strong_token_overlap_count(tool_tokens, key_tokens)
            if overlap <= 0:
                continue
            candidate = (overlap, index, value)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return None if best is None else best[2]


def _boolean_items_from_tool_message(message: dict[str, Any]) -> list[tuple[str, bool]]:
    content = message.get("content")
    if not isinstance(content, str):
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    result = payload.get("result") if isinstance(payload, dict) else payload
    return _boolean_items_from_value(result)


def _boolean_items_from_value(value: Any, prefix: str = "") -> list[tuple[str, bool]]:
    if isinstance(value, dict):
        items: list[tuple[str, bool]] = []
        for key, child in value.items():
            key_text = f"{prefix} {key}".strip()
            items.extend(_boolean_items_from_value(child, key_text))
        return items
    if isinstance(value, list):
        items = []
        for child in value:
            items.extend(_boolean_items_from_value(child, prefix))
        return items
    if isinstance(value, bool) and prefix:
        return [(prefix, value)]
    return []


def _messages_after_latest_user(
    transcript: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    latest_user_index = -1
    for index, message in enumerate(transcript):
        if message.get("role") == "user":
            latest_user_index = index
    if latest_user_index < 0:
        return []
    return transcript[latest_user_index + 1 :]


def _schema_detailed_confirmation_repair(
    *,
    plan_action: dict[str, Any],
    passes: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") == "respond":
        content = plan_action.get("content") or ""
        if not _looks_like_confirmation_question(content):
            return None
        if _latest_user_confirmed_prior_confirmation(transcript):
            confirmed_calls = _confirmed_action_calls_from_latest_user(
                transcript=transcript,
                tools=tools,
            )
            confirmed_calls = _filter_explicitly_rejected_tool_calls(
                confirmed_calls,
                transcript,
                tools,
            )
            if confirmed_calls:
                return {
                    "action": "tool_calls",
                    "content": "",
                    "tool_calls": confirmed_calls,
                }
        if _latest_user_affirms(transcript):
            preconfirmed_calls = _user_requested_action_calls(
                transcript=transcript,
                tools=tools,
            )
            preconfirmed_calls = _numeric_repaired_tool_calls(
                preconfirmed_calls,
                transcript,
                tools,
            )
            preconfirmed_calls = _filter_explicitly_rejected_tool_calls(
                preconfirmed_calls,
                transcript,
                tools,
            )
            if preconfirmed_calls and not _policy_requires_confirmation_for_actions(
                action_tool_calls=preconfirmed_calls,
                transcript=transcript,
                tools=tools,
            ):
                return {
                    "action": "tool_calls",
                    "content": "",
                    "tool_calls": preconfirmed_calls,
                }
        tool_calls = _best_state_change_calls_from_passes(passes, tools)
        inferred_tool_calls = _infer_confirmation_tool_calls(content, transcript, tools)
        if tool_calls and inferred_tool_calls and _tool_call_sequence_covers(
            candidate_calls=inferred_tool_calls,
            required_calls=tool_calls,
            tools=tools,
        ) and _tool_calls_grounded_in_text(inferred_tool_calls, content, tools):
            tool_calls = inferred_tool_calls
        elif not tool_calls:
            tool_calls = inferred_tool_calls
        tool_calls = _actionable_tool_calls(tool_calls, tools)
        tool_calls = _numeric_and_preference_repaired_tool_calls(
            tool_calls,
            transcript,
            tools,
        )
        tool_calls = _filter_explicitly_rejected_tool_calls(
            tool_calls,
            transcript,
            tools,
        )
        if not tool_calls:
            return None
        if _policy_requires_confirmation_for_actions(
            action_tool_calls=tool_calls,
            transcript=transcript,
            tools=tools,
        ):
            return _policy_context_confirmation_response_for_tool_calls(
                tool_calls,
                tools,
                transcript,
            )
        if _confirmation_has_required_details(content, tool_calls, tools):
            return None
        return _confirmation_response_for_tool_calls(tool_calls, tools)

    if plan_action.get("action") != "tool_calls":
        return None
    prior_confirmation = _prior_assistant_confirmation_before_latest_user(transcript)
    tool_calls = _actionable_tool_calls(plan_action.get("tool_calls", []) or [], tools)
    tool_calls = _numeric_and_preference_repaired_tool_calls(
        tool_calls,
        transcript,
        tools,
    )
    tool_calls = _filter_explicitly_rejected_tool_calls(
        tool_calls,
        transcript,
        tools,
    )
    if not tool_calls:
        return None
    if not prior_confirmation:
        needs_policy_confirmation = _policy_requires_confirmation_for_actions(
            action_tool_calls=tool_calls,
            transcript=transcript,
            tools=tools,
        )
        if needs_policy_confirmation:
            return _policy_context_confirmation_response_for_tool_calls(
                tool_calls,
                tools,
                transcript,
            )
        if _scatter_suggests_confirmation(passes) and not (
            _tool_calls_fully_grounded_in_latest_user(
                tool_calls=tool_calls,
                transcript=transcript,
                tools=tools,
            )
        ):
            return _confirmation_response_for_tool_calls(tool_calls, tools)
        return None
    if not _latest_user_affirms(transcript):
        return None
    if _confirmation_has_required_details(prior_confirmation, tool_calls, tools):
        return None
    return _confirmation_response_for_tool_calls(tool_calls, tools)


def _numeric_repaired_tool_calls(
    tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    repaired_action = _numeric_argument_scope_repair(
        plan_action={"action": "tool_calls", "content": "", "tool_calls": tool_calls},
        transcript=transcript,
        tools=tools,
    )
    if repaired_action is None:
        return tool_calls
    return repaired_action.get("tool_calls", []) or tool_calls


def _numeric_and_preference_repaired_tool_calls(
    tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    repaired_calls = _numeric_repaired_tool_calls(tool_calls, transcript, tools)
    preference_repair = _preference_numeric_argument_repair(
        plan_action={
            "action": "tool_calls",
            "content": "",
            "tool_calls": repaired_calls,
        },
        transcript=transcript,
        tools=tools,
    )
    if preference_repair is None:
        return repaired_calls
    return preference_repair.get("tool_calls", []) or repaired_calls


def _preference_numeric_argument_repair(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "tool_calls":
        return None

    preference_names = {
        name
        for tool in _preference_read_tools(tools)
        if (name := _tool_name(tool))
    }
    if not preference_names:
        return None

    preference_segments = _preference_numeric_segments(
        transcript, preference_names
    )
    if not preference_segments:
        return None

    latest_segments = _numeric_reference_segments(transcript)
    tools_by_name = _tool_by_name(tools)
    changed = False
    repaired_calls: list[dict[str, Any]] = []
    for tool_call in plan_action.get("tool_calls", []) or []:
        name = tool_call.get("tool_name", "")
        tool = tools_by_name.get(name)
        arguments = dict(tool_call.get("arguments") or {})
        if tool is None or not arguments:
            repaired_calls.append({"tool_name": name, "arguments": arguments})
            continue

        properties = _tool_parameters(tool).get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for arg_name, value in list(arguments.items()):
            if not _is_number(value):
                continue
            prop_schema = properties.get(arg_name)
            if not isinstance(prop_schema, dict) or not _is_numeric_schema(prop_schema):
                continue
            if latest_segments and _numeric_value_anchored_to_tool(
                value, tool, latest_segments
            ):
                continue

            preferred = _preference_numeric_value_for_tool_argument(
                tool=tool,
                argument_name=arg_name,
                schema=prop_schema,
                preference_segments=preference_segments,
            )
            if preferred is None or _numbers_close(float(preferred), float(value)):
                continue
            arguments[arg_name] = preferred
            changed = True

        repaired_calls.append({"tool_name": name, "arguments": arguments})

    if not changed:
        return None
    repaired = dict(plan_action)
    repaired["tool_calls"] = repaired_calls
    return repaired


def _preference_numeric_segments(
    transcript: list[dict[str, Any]], preference_names: set[str]
) -> list[str]:
    segments: list[str] = []
    for message in transcript:
        if message.get("role") != "tool" or message.get("name") not in preference_names:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        segments.extend(_numeric_segments_from_text(content))
    return segments


def _preference_numeric_value_for_tool_argument(
    *,
    tool: dict[str, Any],
    argument_name: str,
    schema: dict[str, Any],
    preference_segments: list[str],
) -> int | float | None:
    tool_tokens = _tool_search_tokens(tool)
    argument_tokens = _meaningful_tokens(argument_name)
    best: tuple[int, float] | None = None
    for segment in preference_segments:
        segment_tokens = _meaningful_tokens(segment)
        relevance = _token_overlap_count(segment_tokens, tool_tokens)
        if argument_tokens:
            relevance += _token_overlap_count(segment_tokens, argument_tokens)
        if relevance <= 0:
            continue
        for number in _numbers_in_text(segment):
            if not _number_in_schema_range(number, schema):
                continue
            if best is None or relevance > best[0]:
                best = (relevance, number)
    if best is None:
        return None
    return _coerce_numeric_for_schema(best[1], schema)


def _prune_ungrounded_action_calls(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "tool_calls":
        return None

    raw_calls = plan_action.get("tool_calls", []) or []
    tools_by_name = _tool_by_name(tools)
    call_infos: list[dict[str, Any]] = []
    for tool_call in raw_calls:
        if not _is_actionable_tool_call(tool_call, tools_by_name):
            return None
        name = tool_call.get("tool_name", "")
        tool = tools_by_name.get(name)
        if tool is None:
            return None
        normalized = {
            "tool_name": name,
            "arguments": tool_call.get("arguments") or {},
        }
        call_infos.append(
            {
                "call": normalized,
                "tool": tool,
                "tokens": _tool_call_grounding_tokens(normalized, tool),
            }
        )

    if len(call_infos) <= 1:
        return None

    user_text = _all_text_by_role(transcript, "user")
    if not user_text.strip():
        return None

    for info in call_infos:
        info["score"] = _tool_call_user_grounding_score(
            user_text=user_text,
            tool_call=info["call"],
            tool=info["tool"],
        )

    best_score = max(int(info["score"]) for info in call_infos)
    if best_score < 2:
        return None

    protected_names: set[str] = set()
    for info in call_infos:
        if int(info["score"]) <= 0:
            continue
        protected_names.update(
            _policy_dependent_action_names(
                trigger_info=info,
                call_infos=call_infos,
                transcript=transcript,
            )
        )

    kept_calls = [
        info["call"]
        for info in call_infos
        if int(info["score"]) > 0
        or info["call"].get("tool_name", "") in protected_names
    ]
    if not kept_calls or len(kept_calls) == len(call_infos):
        return None

    repaired = dict(plan_action)
    repaired["tool_calls"] = kept_calls
    return repaired


@dataclass(frozen=True)
class _RejectionSpan:
    text: str
    start: int
    end: int


def _prune_rejected_action_calls(
    *,
    plan_action: dict[str, Any],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan_action.get("action") != "tool_calls":
        return None

    raw_calls = plan_action.get("tool_calls", []) or []
    if not raw_calls:
        return None

    filtered_calls = _filter_explicitly_rejected_tool_calls(
        raw_calls,
        transcript,
        tools,
    )
    if len(filtered_calls) == len(raw_calls):
        return None

    if filtered_calls:
        repaired = dict(plan_action)
        repaired["tool_calls"] = filtered_calls
        return repaired

    fallback_calls = _non_rejected_user_requested_action_calls(
        transcript=transcript,
        tools=tools,
    )
    if fallback_calls:
        return {"action": "tool_calls", "content": "", "tool_calls": fallback_calls}

    if _successful_action_tool_after_latest_user(transcript, tools):
        return _post_action_success_response(transcript, tools)
    return {"action": "respond", "content": "Understood. I will not change that."}


def _non_rejected_user_requested_action_calls(
    *, transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    requested = _confirmed_action_calls_from_latest_user(
        transcript=transcript,
        tools=tools,
    )
    if not requested:
        requested = _user_requested_action_calls(transcript=transcript, tools=tools)
    requested = _numeric_repaired_tool_calls(requested, transcript, tools)
    requested = _filter_explicitly_rejected_tool_calls(
        requested,
        transcript,
        tools,
    )
    requested = [
        tool_call
        for tool_call in requested
        if _tool_call_identity_grounded_in_recent_request(
            tool_call,
            transcript,
            tools,
        )
    ]
    if not requested:
        return []
    return _remaining_after_successful_action_calls(
        requested_calls=requested,
        transcript=transcript,
        tools=tools,
    )


def _filter_explicitly_rejected_tool_calls(
    tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    user_text = _all_text_by_role(transcript, "user")
    rejection_spans = _explicit_rejection_spans(user_text)
    if not rejection_spans:
        return tool_calls

    active_text = _text_without_spans(user_text, rejection_spans)
    tools_by_name = _tool_by_name(tools)
    kept_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = str(tool_call.get("tool_name") or "")
        tool = tools_by_name.get(name)
        if tool is None or not _is_actionable_tool_call(tool_call, tools_by_name):
            kept_calls.append(tool_call)
            continue
        normalized = {"tool_name": name, "arguments": tool_call.get("arguments") or {}}
        if _tool_call_rejected_by_user_text(
            tool_call=normalized,
            tool=tool,
            user_text=user_text,
            active_text=active_text,
            rejection_spans=rejection_spans,
        ):
            continue
        kept_calls.append(normalized)
    return kept_calls


def _tool_call_rejected_by_user_text(
    *,
    tool_call: dict[str, Any],
    tool: dict[str, Any],
    user_text: str,
    active_text: str,
    rejection_spans: list[_RejectionSpan],
) -> bool:
    call_tokens = _tool_call_grounding_tokens(tool_call, tool)
    for span in rejection_spans:
        if _rejection_overridden_after_span(
            span=span,
            user_text=user_text,
            call_tokens=call_tokens,
        ):
            continue
        if _strong_token_overlap_count(_meaningful_tokens(span.text), call_tokens) <= 0:
            continue
        if _rejection_span_polarity_allows_call(span.text, tool_call):
            continue
        return True

    return _boolean_value_only_supported_by_rejected_text(
        tool_call=tool_call,
        user_text=user_text,
        active_text=active_text,
    )


def _explicit_rejection_spans(text: str) -> list[_RejectionSpan]:
    spans: list[_RejectionSpan] = []
    search_text = _normalized_apostrophes(text)
    patterns = (
        r"\b(?:do\s+not|don't|dont|never)\s+(?P<body>[^.;!?\n]+)",
        r"\b(?:without|avoid|skip|exclude|excluding)\s+(?P<body>[^.;!?\n]+)",
        r"\bno\s+(?P<body>[^.;!?\n]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, search_text, flags=re.I):
            start = match.start("body")
            end = _trim_rejection_end(match.group("body"), start)
            if end <= start:
                continue
            spans.append(_RejectionSpan(text=text[start:end], start=start, end=end))
    spans.sort(key=lambda span: (span.start, span.end))
    return spans


def _normalized_apostrophes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")


def _trim_rejection_end(body: str, body_start: int) -> int:
    end = len(body)
    split = re.search(r"\b(?:but|however|though)\b", body, flags=re.I)
    if split is not None:
        end = split.start()
    return body_start + len(body[:end].rstrip(" ,"))


def _text_without_spans(text: str, spans: list[_RejectionSpan]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(text[cursor:span.start])
        cursor = max(cursor, span.end)
    parts.append(text[cursor:])
    return "".join(parts)


def _rejection_overridden_after_span(
    *,
    span: _RejectionSpan,
    user_text: str,
    call_tokens: set[str],
) -> bool:
    later_text = user_text[span.end :]
    for positive in _positive_activation_spans(later_text):
        if _strong_token_overlap_count(_meaningful_tokens(positive.text), call_tokens) > 0:
            return True
    return False


def _positive_activation_spans(text: str) -> list[_RejectionSpan]:
    spans: list[_RejectionSpan] = []
    patterns = (
        r"\b(?:turn|switch|power)\s+on\s+(?P<body>[^.;!?\n]+)",
        r"\b(?:enable|activate|start)\s+(?P<body>[^.;!?\n]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            start = match.start("body")
            end = _trim_rejection_end(match.group("body"), start)
            if end > start:
                spans.append(_RejectionSpan(text=text[start:end], start=start, end=end))
    return spans


def _rejection_span_polarity_allows_call(
    span_text: str, tool_call: dict[str, Any]
) -> bool:
    polarity = _activation_polarity(span_text)
    if polarity is None:
        return False
    boolean_values = _tool_call_boolean_values(tool_call)
    if not boolean_values:
        return False
    if polarity is True and all(value is False for value in boolean_values):
        return True
    if polarity is False and all(value is True for value in boolean_values):
        return True
    return False


def _boolean_value_only_supported_by_rejected_text(
    *,
    tool_call: dict[str, Any],
    user_text: str,
    active_text: str,
) -> bool:
    boolean_values = _tool_call_boolean_values(tool_call)
    if not boolean_values:
        return False
    schema = {"type": "boolean"}
    for value in set(boolean_values):
        if _infer_schema_value_from_text(schema, user_text) is not value:
            continue
        if _infer_schema_value_from_text(schema, active_text) is value:
            continue
        return True
    return False


def _tool_call_boolean_values(tool_call: dict[str, Any]) -> list[bool]:
    values: list[bool] = []

    def collect(value: Any) -> None:
        if isinstance(value, bool):
            values.append(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
            return
        if isinstance(value, list):
            for child in value:
                collect(child)

    collect(tool_call.get("arguments") or {})
    return values


def _activation_polarity(text: str) -> bool | None:
    lowered = text.lower()
    if re.search(r"\b(?:turn|switch|power)\s+on\b", lowered) or re.search(
        r"\b(?:enable|enabled|activate|start)\b",
        lowered,
    ):
        return True
    if re.search(r"\b(?:turn|switch|power)\s+off\b", lowered) or re.search(
        r"\b(?:disable|disabled|deactivate|stop)\b",
        lowered,
    ):
        return False
    return None


def _tool_call_user_grounding_score(
    *, user_text: str, tool_call: dict[str, Any], tool: dict[str, Any]
) -> int:
    user_tokens = _meaningful_tokens(user_text)
    call_tokens = _tool_call_grounding_tokens(tool_call, tool)
    score = _strong_token_overlap_count(user_tokens, call_tokens)
    reference_text = _tool_call_reference_source_text(tool_call, tool)
    for token in user_tokens:
        if 2 <= len(token) <= 4 and re.search(
            rf"\b{re.escape(token.upper())}\b", reference_text
        ):
            score += 1
    return score


def _tool_call_grounding_tokens(
    tool_call: dict[str, Any], tool: dict[str, Any]
) -> set[str]:
    return _meaningful_tokens(_tool_call_reference_source_text(tool_call, tool))


def _tool_call_reference_source_text(
    tool_call: dict[str, Any], tool: dict[str, Any]
) -> str:
    arguments = tool_call.get("arguments") or {}
    argument_parts = [
        f"{name} {_argument_value_text(value)}"
        for name, value in arguments.items()
    ]
    return " ".join(
        [
            _tool_name(tool),
            _tool_description(tool),
            *argument_parts,
        ]
    )


def _strong_token_overlap_count(left: set[str], right: set[str]) -> int:
    return sum(
        1
        for token in (_token_family_set(left) & _token_family_set(right))
        if len(token) >= 4
    )


def _policy_dependent_action_names(
    *,
    trigger_info: dict[str, Any],
    call_infos: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> set[str]:
    trigger_tokens = trigger_info["tokens"]
    protected: set[str] = set()
    for segment in _policy_dependency_segments(transcript):
        split = _split_policy_dependency_segment(segment)
        if split is None:
            continue
        before, after = split
        if _strong_token_overlap_count(
            trigger_tokens, _meaningful_tokens(before)
        ) <= 0:
            continue
        after_tokens = _meaningful_tokens(after)
        for info in call_infos:
            name = info["call"].get("tool_name", "")
            if name == trigger_info["call"].get("tool_name", ""):
                continue
            if _strong_token_overlap_count(info["tokens"], after_tokens) > 0:
                protected.add(name)
    return protected


def _policy_dependency_segments(transcript: list[dict[str, Any]]) -> list[str]:
    segments: list[str] = []
    for message in transcript:
        if message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        current: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                if current:
                    segments.append(" ".join(current))
                    current = []
                continue
            starts_new = bool(
                re.match(
                    r"^[-*\d. ]*(?:[A-Z]+-[A-Z]+:)?\s*(when|if)\b",
                    stripped,
                    re.I,
                )
                or re.search(r"\b[A-Z]+-[A-Z]+:", stripped)
            )
            if starts_new and current:
                segments.append(" ".join(current))
                current = []
            current.append(stripped)
        if current:
            segments.append(" ".join(current))
    return segments


def _split_policy_dependency_segment(segment: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(must|should|require|requires|required|automatically|ensure)\b",
        segment,
        flags=re.I,
    )
    if match is None:
        return None
    return segment[: match.start()], segment[match.start() :]


def _number_in_schema_range(value: float, schema: dict[str, Any]) -> bool:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if _is_number(minimum) and value < float(minimum):
        return False
    if _is_number(maximum) and value > float(maximum):
        return False
    return True


def _best_state_change_calls_from_passes(
    passes: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    best_score = -1.0
    best_calls: list[dict[str, Any]] = []
    tools_by_name = _tool_by_name(tools)
    for p in passes:
        action = p.get("action", {})
        if action.get("action") != "tool_calls":
            continue
        calls = [
            {
                "tool_name": tc.get("tool_name", ""),
                "arguments": tc.get("arguments") or {},
            }
            for tc in action.get("tool_calls", []) or []
            if _is_actionable_tool_call(tc, tools_by_name)
            and _tool_call_has_required_arguments(tc, tools_by_name)
        ]
        if not calls:
            continue
        score = float(p.get("confidence", 0.5))
        if score > best_score:
            best_score = score
            best_calls = calls
    return best_calls


def _actionable_tool_calls(
    tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tools_by_name = _tool_by_name(tools)
    return [
        {"tool_name": tc.get("tool_name", ""), "arguments": tc.get("arguments") or {}}
        for tc in tool_calls
        if _is_actionable_tool_call(tc, tools_by_name)
    ]


def _is_actionable_tool_call(
    tool_call: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> bool:
    name = tool_call.get("tool_name", "")
    if _is_read_tool_name(name):
        return False
    tool = tools_by_name.get(name)
    if tool is None:
        return False
    arguments = tool_call.get("arguments") or {}
    if _arguments_request_information_only(arguments):
        return False
    return True


def _arguments_request_information_only(arguments: dict[str, Any]) -> bool:
    read_values = {
        "check",
        "find",
        "get",
        "list",
        "lookup",
        "query",
        "read",
        "retrieve",
        "search",
        "status",
    }
    for value in arguments.values():
        if isinstance(value, str) and value.strip().lower() in read_values:
            return True
    return False


def _scatter_suggests_confirmation(passes: list[dict[str, Any]]) -> bool:
    if not passes:
        return False
    confirmation_questions = sum(
        1
        for p in passes
        if p.get("action", {}).get("action") == "respond"
        and _looks_like_confirmation_question(
            p.get("action", {}).get("content") or ""
        )
    )
    if confirmation_questions:
        return True
    policy_uncertain = sum(
        1
        for p in passes
        if p.get("dispositions", {}).get("policy_compliance") != "ok"
    )
    return policy_uncertain / len(passes) >= 0.5


def _infer_confirmation_tool_calls(
    confirmation_content: str,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    include_latest_user: bool = True,
) -> list[dict[str, Any]]:
    latest_user = _latest_text_by_role(transcript, "user") if include_latest_user else ""
    text = "\n".join(part for part in (latest_user, confirmation_content) if part)
    text_tokens = _meaningful_tokens(text)
    numeric_segments = _numeric_segments_from_text(text)
    action_tools = [
        tool
        for tool in tools
        if (name := _tool_name(tool)) and not _is_read_tool_name(name)
    ]
    unique_tokens_by_name = _unique_tool_reference_tokens(action_tools)
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []

    for tool in action_tools:
        name = _tool_name(tool)
        if not name:
            continue
        unique_tokens = unique_tokens_by_name.get(name, set())
        if unique_tokens and _token_overlap_count(text_tokens, unique_tokens) <= 0:
            continue
        if not unique_tokens and len(action_tools) > 1:
            continue
        arguments = _infer_required_arguments_for_confirmed_action(
            tool=tool,
            tools=tools,
            numeric_segments=numeric_segments,
            text=text,
        )
        if arguments is None:
            continue
        tool_tokens = _meaningful_tokens(
            f"{name} {_tool_description(tool)}"
        )
        score = len(text_tokens & tool_tokens)
        if score <= 0:
            continue
        positions = _token_positions(text, unique_tokens or tool_tokens)
        position = min(positions) if positions else len(text)
        candidates.append(
            (position, -score, name, {"tool_name": name, "arguments": arguments})
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [call for _position, _score, _name, call in candidates]


def _confirmed_action_calls_from_latest_user(
    *, transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    latest_user = _latest_text_by_role(transcript, "user")
    prior_confirmation = _prior_assistant_confirmation_before_latest_user(transcript)
    text = "\n".join(part for part in (latest_user, prior_confirmation) if part)
    if not text.strip():
        return []

    text_tokens = _meaningful_tokens(text)
    segments = [segment for segment in (latest_user, prior_confirmation) if segment]
    user_text = _all_text_by_role(transcript, "user")
    grounded_numeric_parts: list[str] = []
    for message in transcript:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "user":
            grounded_numeric_parts.append(content)
            continue
        lowered_content = content.lower()
        if message.get("role") == "tool" and any(
            marker in lowered_content
            for marker in ("preference", "prefers", "preferred", "default", "usual")
        ):
            grounded_numeric_parts.append(content)
    grounded_numeric_text = "\n".join(grounded_numeric_parts)
    numeric_segments = _numeric_segments_from_text(grounded_numeric_text)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name or _is_read_tool_name(name):
            continue
        tool_tokens = _tool_reference_tokens(tool)
        if _token_overlap_count(text_tokens, tool_tokens) <= 0:
            continue
        arguments = _infer_required_arguments_for_confirmed_action(
            tool=tool,
            tools=tools,
            numeric_segments=numeric_segments,
            text=user_text or latest_user,
        )
        if arguments is None:
            continue
        positions = _token_positions(text, tool_tokens)
        position = min(positions) if positions else len(text)
        candidates.append((position, name, {"tool_name": name, "arguments": arguments}))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [call for _position, _name, call in candidates]


def _user_requested_action_calls(
    *, transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    latest_user = _latest_text_by_role(transcript, "user")
    if not latest_user.strip():
        return []

    action_tools = [
        tool
        for tool in tools
        if (name := _tool_name(tool)) and not _is_read_tool_name(name)
    ]
    if not action_tools:
        return []

    text_tokens = _meaningful_tokens(latest_user)
    unique_tokens_by_name = _unique_tool_reference_tokens(action_tools)
    numeric_segments = _numeric_segments_from_text(latest_user)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for tool in action_tools:
        name = _tool_name(tool)
        if not name:
            continue
        tool_tokens = _tool_reference_tokens(tool)
        unique_tokens = unique_tokens_by_name.get(name, set())
        if unique_tokens:
            grounding_tokens = unique_tokens
        elif len(action_tools) == 1:
            grounding_tokens = tool_tokens
        else:
            continue
        if _token_overlap_count(text_tokens, grounding_tokens) <= 0:
            continue

        arguments = _infer_required_arguments_for_confirmed_action(
            tool=tool,
            tools=tools,
            numeric_segments=numeric_segments,
            text=latest_user,
        )
        if arguments is None:
            continue
        position = _tool_request_position(latest_user, grounding_tokens)
        candidates.append((position, name, {"tool_name": name, "arguments": arguments}))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [call for _position, _name, call in candidates]


def _unique_tool_reference_tokens(tools: list[dict[str, Any]]) -> dict[str, set[str]]:
    raw: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for tool in tools:
        name = _tool_name(tool)
        if not name:
            continue
        tokens = _tool_reference_tokens(tool)
        raw[name] = tokens
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return {
        name: {token for token in tokens if counts.get(token) == 1}
        for name, tokens in raw.items()
    }


def _tool_request_position(text: str, grounding_tokens: set[str]) -> int:
    positions = _token_positions(text, grounding_tokens)
    if not positions:
        return len(text)
    earliest = min(positions)
    first_positions = [
        position for position in positions if _has_local_first_marker(text, position)
    ]
    if first_positions:
        return -len(text) + min(first_positions)
    return earliest


def _has_local_first_marker(text: str, position: int) -> bool:
    lowered = text.lower()
    after = lowered[position : min(len(lowered), position + 64)]
    before = lowered[max(0, position - 32) : position]
    for match in re.finditer(r"\bfirst\b", after):
        between = after[: match.start()]
        if not re.search(r"[.;!?]", between):
            return True
    for match in re.finditer(r"\bfirst\b", before):
        between = before[match.end() :]
        if not re.search(r"[.;!?]", between):
            return True
    return False


def _infer_required_arguments_for_confirmed_action(
    *,
    tool: dict[str, Any],
    tools: list[dict[str, Any]],
    numeric_segments: list[str],
    text: str,
) -> dict[str, Any] | None:
    schema = _tool_parameters(tool)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    inferred: dict[str, Any] = {}
    for name in _required_argument_names(tool):
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict):
            return None
        if _is_numeric_schema(prop_schema):
            value = _numeric_schema_value_anchored_to_tool(
                tool=tool,
                tools=tools,
                schema=prop_schema,
                segments=numeric_segments,
            )
            number_owned_by_other_tool = any(
                _numeric_value_anchored_to_other_tool(
                    number, tool, tools, numeric_segments
                )
                for number in _numbers_in_text("\n".join(numeric_segments))
            )
            if value is None and not number_owned_by_other_tool:
                value = _infer_numeric_schema_value_from_anchored_text(
                    name=name,
                    schema=prop_schema,
                    text=text,
                )
            if value is None:
                value = _schema_neutral_numeric_value(prop_schema)
        else:
            value = _infer_schema_value_from_text(prop_schema, text)
        if value is None:
            return None
        inferred[name] = value
    return inferred


def _tool_call_has_required_arguments(
    tool_call: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]
) -> bool:
    tool = tools_by_name.get(tool_call.get("tool_name", ""))
    if tool is None:
        return False
    arguments = tool_call.get("arguments") or {}
    return all(name in arguments for name in _required_argument_names(tool))


def _infer_required_arguments_from_text(
    tool: dict[str, Any], text: str
) -> dict[str, Any] | None:
    schema = _tool_parameters(tool)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    inferred: dict[str, Any] = {}
    for name in _required_argument_names(tool):
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict):
            return None
        value = _infer_schema_value_from_text(prop_schema, text)
        if value is None:
            return None
        inferred[name] = value
    return inferred


def _infer_schema_value_from_text(schema: dict[str, Any], text: str) -> Any:
    lowered = text.lower()
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        directional_value = _enum_value_from_directional_text(enum_values, text)
        if directional_value is not None:
            return directional_value
        text_tokens = _meaningful_tokens(text)
        for value in enum_values:
            value_text = str(value)
            if value_text.lower() in lowered:
                return value
            if _meaningful_tokens(value_text) & text_tokens:
                return value
        return None

    schema_type = schema.get("type")
    if schema_type == "boolean":
        if any(word in lowered for word in ("true", " on", "enable", "enabled")):
            return True
        if any(word in lowered for word in ("false", " off", "disable", "disabled")):
            return False
        return None
    if schema_type in {"integer", "number"}:
        matches = _schema_numeric_value_matches_in_text(text, schema)
        if not matches:
            return None
        value = matches[0][0]
        return int(value) if schema_type == "integer" else value
    if "default" in schema:
        return schema["default"]
    return None


def _enum_value_from_directional_text(enum_values: list[Any], text: str) -> Any | None:
    candidates: list[tuple[int, int, str, Any]] = []
    for marker_match in re.finditer(
        r"\b(?:to|towards?|into)\b\s+(?P<body>[^.;!?\n]+)",
        text,
        flags=re.I,
    ):
        body = marker_match.group("body")
        body_tokens = _meaningful_tokens(body)
        for value in enum_values:
            value_text = str(value)
            value_tokens = _meaningful_tokens(value_text)
            if not value_tokens:
                continue
            lowered_body = body.lower()
            literal_index = lowered_body.find(value_text.lower())
            overlap = _token_overlap_count(body_tokens, value_tokens)
            if literal_index < 0 and overlap <= 0:
                continue
            token_positions = _token_positions(body, value_tokens)
            relative_position = (
                literal_index
                if literal_index >= 0
                else min(token_positions) if token_positions else 0
            )
            candidates.append(
                (
                    marker_match.start("body") + relative_position,
                    -overlap,
                    value_text,
                    value,
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _confirmation_has_required_details(
    content: str, tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> bool:
    if not tool_calls:
        return False
    tools_by_name = _tool_by_name(tools)
    lowered = content.lower()
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call.get("tool_name", ""))
        if tool is None:
            return False
        arguments = tool_call.get("arguments") or {}
        properties = _tool_parameters(tool).get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for name in _required_argument_names(tool):
            if name not in arguments:
                return False
            prop_schema = properties.get(name)
            if not isinstance(prop_schema, dict):
                return False
            if not _requires_explicit_confirmation_detail(prop_schema):
                continue
            if not _argument_detail_present(
                lowered, name, arguments[name], prop_schema
            ):
                return False
    return True


def _tool_calls_fully_grounded_in_latest_user(
    *,
    tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    latest_user = _latest_text_by_role(transcript, "user")
    if not latest_user.strip() or not tool_calls:
        return False

    user_tokens = _meaningful_tokens(latest_user)
    tools_by_name = _tool_by_name(tools)
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call.get("tool_name", ""))
        if tool is None:
            return False
        if _token_overlap_count(user_tokens, _tool_reference_tokens(tool)) <= 0:
            return False
    return _confirmation_has_required_details(latest_user, tool_calls, tools)


def _policy_requires_confirmation_for_actions(
    *,
    action_tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    if not action_tool_calls:
        return False

    tools_by_name = _tool_by_name(tools)
    action_tokens: set[str] = set()
    for tool_call in action_tool_calls:
        tool = tools_by_name.get(tool_call.get("tool_name", ""))
        if tool is None:
            continue
        if _tool_description_declares_confirmation_required(tool):
            return True
        action_tokens.update(_tool_call_grounding_tokens(tool_call, tool))
    if not action_tokens:
        return False

    for segment in _policy_dependency_segments(transcript):
        lowered = segment.lower()
        if not any(
            marker in lowered
            for marker in ("confirm", "confirmation", "ask", "permission")
        ):
            continue
        if _strong_token_overlap_count(action_tokens, _meaningful_tokens(segment)) > 0:
            return True
    return False


def _tool_description_declares_confirmation_required(tool: dict[str, Any]) -> bool:
    leading_description = _tool_description(tool)[:96].lower().replace("_", " ")
    if "confirm" not in leading_description:
        return False
    return any(
        marker in leading_description
        for marker in ("require", "required", "requires", "must", "explicit")
    )


def _requires_explicit_confirmation_detail(schema: dict[str, Any]) -> bool:
    enum_values = schema.get("enum")
    return (
        isinstance(enum_values, list)
        and bool(enum_values)
        or _is_numeric_schema(schema)
    )


def _argument_detail_present(
    lowered_content: str, name: str, value: Any, schema: dict[str, Any]
) -> bool:
    value_text = _argument_value_text(value).lower()
    name_present = _schema_name_present(lowered_content, name)
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return name_present and value_text in lowered_content
    schema_type = schema.get("type")
    if schema_type in {"integer", "number"}:
        compact_value = value_text[:-2] if value_text.endswith(".0") else value_text
        return value_text in lowered_content or compact_value in lowered_content
    if schema_type == "boolean":
        if isinstance(value, bool):
            words = ("true", "on", "enable", "enabled") if value else (
                "false",
                "off",
                "disable",
                "disabled",
            )
            return name_present or any(word in lowered_content for word in words)
    return value_text in lowered_content


def _confirmation_response_for_tool_calls(
    tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any] | None:
    details = _tool_call_detail_strings(tool_calls, tools)
    if not details:
        return None
    joined = "; then ".join(details)
    return {
        "action": "respond",
        "content": (
            f"Please confirm: I will {joined}. Should I proceed?"
        ),
    }


def _policy_context_confirmation_response_for_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> dict[str, Any] | None:
    tool_calls = _complete_policy_confirmation_tool_calls(
        tool_calls=tool_calls,
        tools=tools,
        transcript=transcript,
    )
    response = _confirmation_response_for_tool_calls(tool_calls, tools)
    if response is None:
        return None
    summary = _policy_read_result_summary(
        action_tool_calls=tool_calls,
        transcript=transcript,
        tools=tools,
    ) or _relevant_read_result_summary(
        action_tool_calls=tool_calls,
        transcript=transcript,
        tools=tools,
    ) or _recent_read_result_summary(transcript)
    response = dict(response)
    prefix_parts: list[str] = []
    if summary:
        prefix_parts.append(f"The latest context check returned {summary}.")
    prefix_parts.append("I need you to say yes before I can proceed.")
    response["content"] = (
        " ".join(prefix_parts)
        + f" {response['content']}"
    )
    return response


def _complete_policy_confirmation_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actionable = _numeric_and_preference_repaired_tool_calls(
        _actionable_tool_calls(tool_calls, tools),
        transcript,
        tools,
    )
    if not actionable:
        return tool_calls

    inferred = _numeric_and_preference_repaired_tool_calls(
        _user_requested_action_calls(transcript=transcript, tools=tools),
        transcript,
        tools,
    )
    if not inferred:
        return actionable

    if not _tool_call_sequence_covers(
        candidate_calls=inferred,
        required_calls=actionable,
        tools=tools,
    ):
        merged = _merge_user_grounded_tool_call_arguments(
            candidate_calls=inferred,
            proposed_calls=actionable,
            transcript=transcript,
            tools=tools,
        )
        if merged is not None:
            return merged
        return actionable

    return inferred


def _merge_user_grounded_tool_call_arguments(
    *,
    candidate_calls: list[dict[str, Any]],
    proposed_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if len(candidate_calls) != len(proposed_calls):
        return None
    candidate_by_name: dict[str, dict[str, Any]] = {}
    for call in candidate_calls:
        name = str(call.get("tool_name") or "")
        if not name or name in candidate_by_name:
            return None
        candidate_by_name[name] = call
    proposed_by_name: dict[str, dict[str, Any]] = {}
    for call in proposed_calls:
        name = str(call.get("tool_name") or "")
        if not name or name in proposed_by_name:
            return None
        proposed_by_name[name] = call
    proposed_names = list(proposed_by_name)
    if len(set(proposed_names)) != len(proposed_names):
        return None
    if set(candidate_by_name) != set(proposed_names):
        return None

    latest_user = _latest_text_by_role(transcript, "user")
    if not latest_user.strip():
        return None
    numeric_segments = _numeric_segments_from_text(latest_user)
    lowered_user = latest_user.lower()
    tools_by_name = _tool_by_name(tools)
    changed = False
    merged_calls: list[dict[str, Any]] = []
    for candidate_call in candidate_calls:
        name = str(candidate_call.get("tool_name") or "")
        proposed_call = proposed_by_name[name]
        tool = tools_by_name.get(name)
        if tool is None:
            return None
        proposed_arguments = dict(proposed_call.get("arguments") or {})
        candidate_arguments = candidate_call.get("arguments") or {}
        properties = _tool_parameters(tool).get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for argument_name, candidate_value in candidate_arguments.items():
            proposed_value = proposed_arguments.get(argument_name)
            if _argument_values_match(candidate_value, proposed_value):
                continue
            schema = properties.get(argument_name)
            if not isinstance(schema, dict):
                continue
            if _is_numeric_schema(schema):
                grounded = _numeric_value_anchored_to_tool(
                    candidate_value,
                    tool,
                    numeric_segments,
                )
            else:
                grounded = _argument_detail_present(
                    lowered_user,
                    argument_name,
                    candidate_value,
                    schema,
                )
            if not grounded:
                continue
            proposed_arguments[argument_name] = candidate_value
            changed = True
        merged_calls.append({"tool_name": name, "arguments": proposed_arguments})

    if not changed:
        return None
    return merged_calls


def _tool_call_sequence_covers(
    *,
    candidate_calls: list[dict[str, Any]],
    required_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    remaining = list(candidate_calls)
    for required_call in required_calls:
        match_index = next(
            (
                index
                for index, candidate_call in enumerate(remaining)
                if _tool_calls_match_for_confirmation(
                    candidate_call,
                    required_call,
                    tools,
                )
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _tool_calls_match_for_confirmation(
    left: dict[str, Any],
    right: dict[str, Any],
    tools: list[dict[str, Any]],
) -> bool:
    if left.get("tool_name") != right.get("tool_name"):
        return False

    tool = _tool_by_name(tools).get(str(left.get("tool_name") or ""))
    if tool is None:
        return False
    required_names = _required_argument_names(tool)
    left_arguments = left.get("arguments") or {}
    right_arguments = right.get("arguments") or {}
    for name in required_names:
        if name not in left_arguments or name not in right_arguments:
            return False
        if not _argument_values_match(left_arguments[name], right_arguments[name]):
            return False
    return True


def _tool_calls_grounded_in_text(
    tool_calls: list[dict[str, Any]],
    text: str,
    tools: list[dict[str, Any]],
) -> bool:
    text_tokens = _meaningful_tokens(text)
    action_tools = [
        tool
        for tool in tools
        if (name := _tool_name(tool)) and not _is_read_tool_name(name)
    ]
    unique_tokens_by_name = _unique_tool_reference_tokens(action_tools)
    tools_by_name = _tool_by_name(tools)
    for tool_call in tool_calls:
        name = str(tool_call.get("tool_name") or "")
        tool = tools_by_name.get(name)
        if tool is None:
            return False
        grounding_tokens = unique_tokens_by_name.get(name, set())
        if not grounding_tokens and len(action_tools) == 1:
            grounding_tokens = _tool_reference_tokens(tool)
        if not grounding_tokens:
            return False
        if _token_overlap_count(text_tokens, grounding_tokens) <= 0:
            return False
    return True


def _argument_values_match(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return _numbers_close(float(left), float(right))
    return left == right


def _policy_read_result_summary(
    *,
    action_tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    read_scores = _policy_required_read_scores(
        action_tool_calls=action_tool_calls,
        transcript=transcript,
        tools=tools,
    )
    if not read_scores:
        return ""

    read_names = set(read_scores)
    tools_by_name = _tool_by_name(tools)
    latest_user_tokens = _meaningful_tokens(_latest_text_by_role(transcript, "user"))
    action_tokens = _action_tool_call_relevance_tokens(action_tool_calls, tools)
    policy_tokens = _policy_relevance_tokens_for_actions(
        action_tool_calls=action_tool_calls,
        transcript=transcript,
        tools=tools,
    )
    candidates: list[tuple[int, int, int, int, list[str]]] = []
    for index, message in enumerate(transcript):
        if message.get("role") != "tool" or message.get("name") not in read_names:
            continue
        values = _tool_message_scalar_summary_values(message)
        if not values:
            continue
        name = str(message.get("name") or "")
        tool = tools_by_name.get(name)
        relevance_score = (
            _read_result_relevance_score(
                tool=tool,
                values=values,
                latest_user_tokens=latest_user_tokens,
                action_tokens=action_tokens,
            )
            if tool is not None
            else 0
        )
        if tool is not None and policy_tokens:
            relevance_score += _read_result_relevance_score(
                tool=tool,
                values=values,
                latest_user_tokens=policy_tokens,
                action_tokens=set(),
            )
        policy_score = read_scores.get(name, 0)
        candidates.append(
            (
                1 if relevance_score > 0 else 0,
                policy_score + relevance_score,
                policy_score,
                index,
                values,
            )
        )
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3]))
    return ", ".join(candidates[0][4][:3])


def _policy_relevance_tokens_for_actions(
    *,
    action_tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> set[str]:
    action_tokens = _action_tool_call_relevance_tokens(action_tool_calls, tools)
    if not action_tokens:
        return set()
    tokens: set[str] = set()
    for segment in _policy_dependency_segments(transcript):
        segment_tokens = _meaningful_tokens(segment)
        if _strong_token_overlap_count(action_tokens, segment_tokens) <= 0:
            continue
        tokens.update(segment_tokens)
    return tokens


def _action_tool_call_relevance_tokens(
    action_tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> set[str]:
    tools_by_name = _tool_by_name(tools)
    action_tokens: set[str] = set()
    for tool_call in action_tool_calls:
        tool = tools_by_name.get(str(tool_call.get("tool_name") or ""))
        if tool is not None:
            action_tokens.update(_tool_call_grounding_tokens(tool_call, tool))
    return action_tokens


def _read_result_relevance_score(
    *,
    tool: dict[str, Any],
    values: list[str],
    latest_user_tokens: set[str],
    action_tokens: set[str],
) -> int:
    read_tokens = _tool_search_tokens(tool)
    value_tokens = _meaningful_tokens(" ".join(values))
    score = 2 * _strong_token_overlap_count(read_tokens, latest_user_tokens)
    score += _strong_token_overlap_count(read_tokens, action_tokens)
    score += 3 * _strong_token_overlap_count(value_tokens, latest_user_tokens)
    score += _strong_token_overlap_count(value_tokens, action_tokens)
    return score


def _relevant_read_result_summary(
    *,
    action_tool_calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    tools_by_name = _tool_by_name(tools)
    latest_user_tokens = _meaningful_tokens(_latest_text_by_role(transcript, "user"))
    action_tokens = _action_tool_call_relevance_tokens(action_tool_calls, tools)

    candidates: list[tuple[int, int, list[str]]] = []
    for index, message in enumerate(transcript):
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        tool = tools_by_name.get(name)
        if tool is None or not _is_read_tool_name(name):
            continue
        values = _tool_message_scalar_summary_values(message)
        if not values:
            continue
        score = _read_result_relevance_score(
            tool=tool,
            values=values,
            latest_user_tokens=latest_user_tokens,
            action_tokens=action_tokens,
        )
        if score <= 0:
            continue
        candidates.append((score, index, values))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return ", ".join(candidates[0][2][:3])


def _recent_read_result_summary(transcript: list[dict[str, Any]]) -> str:
    for message in reversed(transcript):
        if message.get("role") != "tool" or not _is_read_tool_name(
            str(message.get("name") or "")
        ):
            continue
        values = _tool_message_scalar_summary_values(message)
        if values:
            return ", ".join(values[:3])
    return ""


def _tool_message_scalar_summary_values(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if not isinstance(content, str):
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    result = payload.get("result") if isinstance(payload, dict) else payload
    return _scalar_summary_values(result)


def _scalar_summary_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for key, child in value.items():
            if str(key).lower() in {"status", "description"}:
                continue
            values.extend(_scalar_summary_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_scalar_summary_values(child))
        return values
    if isinstance(value, str):
        normalized = value.replace("_", " ").strip()
        if not normalized or re.fullmatch(r"\d{1,2}:\d{2}", normalized):
            return []
        return [normalized]
    return []


def _tool_call_detail_strings(
    tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[str]:
    tools_by_name = _tool_by_name(tools)
    details: list[str] = []
    for tool_call in tool_calls:
        name = tool_call.get("tool_name", "")
        tool = tools_by_name.get(name)
        if tool is None:
            continue
        arguments = tool_call.get("arguments") or {}
        required_names = _required_argument_names(tool)
        if required_names:
            arg_details = [
                f"{arg_name}: {_argument_value_text(arguments[arg_name])}"
                for arg_name in required_names
                if arg_name in arguments
            ]
            if len(arg_details) != len(required_names):
                continue
            details.append(f"call {name} with {', '.join(arg_details)}")
        else:
            details.append(f"call {name}")
    return details


def _required_argument_names(tool: dict[str, Any]) -> list[str]:
    required = _tool_parameters(tool).get("required", [])
    return [name for name in required if isinstance(name, str)] if isinstance(required, list) else []


def _argument_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _schema_name_present(lowered_content: str, name: str) -> bool:
    lowered_name = name.lower()
    spaced_name = lowered_name.replace("_", " ")
    return lowered_name in lowered_content or spaced_name in lowered_content


def _prior_assistant_confirmation_before_latest_user(
    transcript: list[dict[str, Any]]
) -> str:
    seen_latest_user = False
    for message in reversed(transcript):
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str) and not seen_latest_user:
            seen_latest_user = True
            continue
        if not seen_latest_user:
            continue
        if role == "assistant" and isinstance(content, str):
            return content if _looks_like_confirmation_question(content) else ""
    return ""


def _latest_user_affirms(transcript: list[dict[str, Any]]) -> bool:
    content = _latest_text_by_role(transcript, "user").lower()
    if not content:
        return False
    affirmative_markers = (
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "proceed",
        "confirm",
        "confirmed",
        "do it",
    )
    return any(marker in content for marker in affirmative_markers)


def _latest_user_confirmed_prior_confirmation(
    transcript: list[dict[str, Any]]
) -> bool:
    return bool(
        _prior_assistant_confirmation_before_latest_user(transcript)
        and _latest_user_affirms(transcript)
    )


def _latest_text_by_role(transcript: list[dict[str, Any]], role: str) -> str:
    for message in reversed(transcript):
        if message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _all_text_by_role(transcript: list[dict[str, Any]], role: str) -> str:
    return "\n".join(
        message["content"]
        for message in transcript
        if message.get("role") == role and isinstance(message.get("content"), str)
    )


def _tool_description(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        return str(function.get("description") or "")
    return str(tool.get("description") or "") if isinstance(tool, dict) else ""


def _meaningful_tokens(text: str) -> set[str]:
    generic = {
        "a",
        "an",
        "and",
        "are",
        "additionalproperties",
        "both",
        "boolean",
        "by",
        "call",
        "can",
        "close",
        "current",
        "currently",
        "default",
        "description",
        "details",
        "do",
        "either",
        "false",
        "for",
        "from",
        "get",
        "gets",
        "i",
        "including",
        "info",
        "information",
        "inside",
        "is",
        "it",
        "me",
        "mode",
        "now",
        "number",
        "object",
        "of",
        "on",
        "open",
        "parameters",
        "please",
        "properties",
        "required",
        "returns",
        "search",
        "set",
        "should",
        "specified",
        "state",
        "status",
        "string",
        "system",
        "the",
        "to",
        "tool",
        "turn",
        "true",
        "type",
        "vehicle",
        "want",
        "with",
        "you",
        "car",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        if len(token) > 1 and token not in generic
    }


def _is_read_only_action(action: dict[str, Any]) -> bool:
    if action.get("action") != "tool_calls":
        return False
    tool_calls = action.get("tool_calls", []) or []
    if not tool_calls:
        return False
    return all(_is_read_tool_name(tc.get("tool_name", "")) for tc in tool_calls)


def _is_read_tool_name(name: str) -> bool:
    return name.startswith("get_")


def _is_context_read_tool(tool: dict[str, Any]) -> bool:
    name = _tool_name(tool)
    return bool(name) and _is_read_tool_name(name) and not _is_preference_read_tool(tool)


def _read_tool_arguments(
    tool: dict[str, Any],
    transcript: list[dict[str, Any]],
    query_text: str,
) -> dict[str, Any] | None:
    safe_arguments = _safe_read_tool_arguments(tool)
    if safe_arguments is not None:
        return safe_arguments

    schema = _tool_parameters(tool)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None

    arguments: dict[str, Any] = {}
    for name, prop_schema in properties.items():
        if not isinstance(name, str) or not isinstance(prop_schema, dict):
            continue
        value = _infer_read_argument_value(name, prop_schema, transcript, query_text)
        if value is not None:
            arguments[name] = value
        elif "default" in prop_schema:
            arguments[name] = prop_schema["default"]

    for name in _required_argument_names(tool):
        if name not in arguments:
            return None
    if not _required_argument_names(tool) and properties and not arguments:
        return None
    return arguments


def _safe_read_tool_arguments(tool: dict[str, Any]) -> dict[str, Any] | None:
    required = _required_argument_names(tool)
    schema = _tool_parameters(tool)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    if not required:
        if not properties:
            return {}
        if all(
            isinstance(prop_schema, dict) and "default" in prop_schema
            for prop_schema in properties.values()
        ):
            return {}
        return None

    arguments: dict[str, Any] = {}
    for name in required:
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict) or "default" not in prop_schema:
            return None
        arguments[name] = prop_schema["default"]
    return arguments


def _infer_read_argument_value(
    name: str,
    schema: dict[str, Any],
    transcript: list[dict[str, Any]],
    query_text: str,
) -> Any:
    combined_text = _transcript_text(transcript)
    latest_user = _latest_text_by_role(transcript, "user")
    name_tokens = _meaningful_tokens(name)
    description_tokens = _meaningful_tokens(str(schema.get("description") or ""))
    all_name_tokens = name_tokens | description_tokens

    schema_type = schema.get("type")
    if schema_type in {"integer", "number"}:
        user_time = _time_from_text(latest_user)
        current_time = _current_datetime_values(combined_text)
        if _tokens_contain_family(all_name_tokens, "hour") and user_time is not None:
            return _coerce_numeric_for_schema(float(user_time[0]), schema)
        if _tokens_contain_family(all_name_tokens, "minute") and user_time is not None:
            return _coerce_numeric_for_schema(float(user_time[1]), schema)
        for key in ("month", "day", "hour", "minute"):
            if _tokens_contain_family(all_name_tokens, key) and key in current_time:
                return _coerce_numeric_for_schema(current_time[key], schema)
        return _infer_numeric_schema_value_from_anchored_text(
            name=name,
            schema=schema,
            text=query_text,
        )

    if schema_type == "string":
        if "id" in all_name_tokens:
            value = _id_value_from_transcript(
                transcript=transcript,
                query_text=query_text,
                wants_current=bool(
                    {"location", "poi", "point", "place"} & all_name_tokens
                ),
            )
            if value:
                return value
        return _infer_schema_value_from_text(schema, query_text)

    return _infer_schema_value_from_text(schema, query_text)


def _tokens_contain_family(tokens: set[str], root: str) -> bool:
    return any(token == root or token == f"{root}s" for token in tokens)


def _time_from_text(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"\b([0-2]?\d)(?::([0-5]\d))?\s*([AaPp]\.?[Mm]\.?)?\b", text
    )
    if match is None:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = (match.group(3) or "").lower().replace(".", "")
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return hour, minute


def _current_datetime_values(text: str) -> dict[str, float]:
    match = re.search(r'"current_datetime"\s*:\s*\{(?P<body>[^}]+)\}', text)
    body = match.group("body") if match else text
    values: dict[str, float] = {}
    for key in ("year", "month", "day", "hour", "minute"):
        key_match = re.search(rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)', body)
        if key_match:
            values[key] = float(key_match.group(1))
    return values


def _id_value_from_transcript(
    *,
    transcript: list[dict[str, Any]],
    query_text: str,
    wants_current: bool,
) -> str | None:
    combined_text = _transcript_text(transcript)
    query_lower = query_text.lower()
    for item in _named_id_values(combined_text):
        if item["name"].lower() in query_lower:
            return item["id"]

    if wants_current:
        current_match = re.search(
            r'"current_[^"]*"\s*:\s*\{(?P<body>[^}]+)\}', combined_text
        )
        if current_match is None:
            current_match = re.search(
                r"\bCURRENT_[A-Z0-9_]*\s*=\s*\{(?P<body>[^}]+)\}",
                combined_text,
            )
        if current_match:
            id_match = re.search(r'"id"\s*:\s*"([^"]+)"', current_match.group("body"))
            if id_match:
                return id_match.group(1)
    return None


def _named_id_values(text: str) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    patterns = (
        re.compile(
            r'"id"\s*:\s*"(?P<id>[^"]+)".{0,240}?"name"\s*:\s*"(?P<name>[^"]+)"',
            re.DOTALL,
        ),
        re.compile(
            r'"name"\s*:\s*"(?P<name>[^"]+)".{0,240}?"id"\s*:\s*"(?P<id>[^"]+)"',
            re.DOTALL,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            values.append({"id": match.group("id"), "name": match.group("name")})
    return values


def _transcript_text(transcript: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in transcript:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        if message.get("tool_calls"):
            parts.append(json.dumps(message.get("tool_calls"), ensure_ascii=False))
    return "\n".join(parts)


def _tool_call_reference_text(tool_calls: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{tc.get('tool_name', '')} {json.dumps(tc.get('arguments') or {})}"
        for tc in tool_calls
    )


def _ambiguous_action_family(
    *,
    action_tool_calls: list[dict[str, Any]],
    latest_user_tokens: set[str],
    tools: list[dict[str, Any]],
) -> bool:
    if not action_tool_calls or not latest_user_tokens:
        return False

    tools_by_name = _tool_by_name(tools)
    action_tools = [
        tool
        for tool in tools
        if (name := _tool_name(tool))
        and not _is_read_tool_name(name)
        and name in tools_by_name
    ]
    all_action_tokens = {
        _tool_name(tool): _tool_reference_tokens(tool) for tool in action_tools
    }
    for tool_call in action_tool_calls:
        current_tool = tools_by_name.get(tool_call.get("tool_name", ""))
        if current_tool is None:
            continue
        current_name = _tool_name(current_tool)
        current_tokens = all_action_tokens.get(current_name, set())
        if _token_overlap_count(latest_user_tokens, current_tokens) <= 0:
            continue

        siblings: list[tuple[str, set[str]]] = []
        for other_name, other_tokens in all_action_tokens.items():
            if other_name == current_name:
                continue
            if _token_overlap_count(current_tokens, other_tokens) <= 0:
                continue
            if _token_overlap_count(latest_user_tokens, other_tokens) <= 0:
                continue
            siblings.append((other_name, other_tokens))
        if not siblings:
            continue

        sibling_tokens = set().union(*(tokens for _, tokens in siblings))
        current_unique_tokens = current_tokens - sibling_tokens
        if _token_overlap_count(latest_user_tokens, current_unique_tokens) <= 0:
            return True
    return False


def _tool_search_tokens(tool: dict[str, Any]) -> set[str]:
    return _meaningful_tokens(
        f"{_tool_name(tool)} {_tool_description(tool)} "
        f"{json.dumps(_tool_parameters(tool), ensure_ascii=False)}"
    )


def _context_read_score(
    *,
    tool: dict[str, Any],
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]],
    query_text: str,
    latest_user_tokens: set[str],
    action_tokens: set[str],
    query_tokens: set[str],
    requires_action_relevance: bool,
    allow_argument_grounding: bool,
) -> int:
    read_tokens = _tool_search_tokens(tool)
    query_overlap = _strong_token_overlap_count(query_tokens, read_tokens)
    grounded_required_bonus = (
        2
        if _required_argument_names(tool)
        and arguments
        and not _arguments_match_schema_defaults(tool, arguments)
        else 0
    )
    argument_grounding = _argument_values_grounded_in_query(
        arguments=arguments,
        transcript=transcript,
        query_text=query_text,
    )
    if query_overlap <= 0:
        if (
            not allow_argument_grounding
            or grounded_required_bonus <= 0
            or argument_grounding <= 0
        ):
            return 0
        query_overlap = argument_grounding

    required_bonus = 2 if _required_argument_names(tool) else 0
    if not requires_action_relevance:
        return query_overlap + required_bonus + grounded_required_bonus

    user_overlap = _strong_token_overlap_count(latest_user_tokens, read_tokens)
    action_overlap = _strong_token_overlap_count(action_tokens, read_tokens)
    if user_overlap <= 0 and grounded_required_bonus <= 0:
        return 0
    if action_overlap >= 2:
        return (
            100
            + (action_overlap * 10)
            + user_overlap
            + required_bonus
            + grounded_required_bonus
        )
    if _required_argument_names(tool) and arguments:
        return 50 + (user_overlap * 10) + required_bonus + grounded_required_bonus
    return 0


def _argument_values_grounded_in_query(
    *,
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]],
    query_text: str,
) -> int:
    lowered_query = query_text.lower()
    named_ids = _named_id_values(_transcript_text(transcript))
    score = 0
    for value in arguments.values():
        if _is_number(value):
            if any(
                _numbers_close(number, float(value))
                for number in _numbers_in_text(query_text)
            ):
                score += 1
            continue
        if isinstance(value, str):
            lowered_value = value.lower()
            if lowered_value and lowered_value in lowered_query:
                score += 1
                continue
            for item in named_ids:
                if item["id"] != value:
                    continue
                if item["name"].lower() in lowered_query:
                    score += 1
                    break
    return score


def _arguments_match_schema_defaults(
    tool: dict[str, Any], arguments: dict[str, Any]
) -> bool:
    properties = _tool_parameters(tool).get("properties", {})
    if not isinstance(properties, dict):
        return False
    saw_default = False
    for name, value in arguments.items():
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict) or "default" not in prop_schema:
            return False
        saw_default = True
        if prop_schema["default"] != value:
            return False
    return saw_default


def _required_string_arguments_reuse_value(
    tool: dict[str, Any], arguments: dict[str, Any]
) -> bool:
    properties = _tool_parameters(tool).get("properties", {})
    if not isinstance(properties, dict):
        return False
    seen: dict[str, str] = {}
    for name, value in arguments.items():
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict) or prop_schema.get("type") != "string":
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip().lower()
        previous_name = seen.get(normalized)
        if previous_name is not None and _meaningful_tokens(
            previous_name
        ) != _meaningful_tokens(name):
            return True
        seen[normalized] = name
    return False


def _token_overlap_count(left: set[str], right: set[str]) -> int:
    return len(_token_family_set(left) & _token_family_set(right))


def _token_family_set(tokens: set[str]) -> set[str]:
    families: set[str] = set()
    for token in tokens:
        families.add(token)
        if len(token) > 5 and token.endswith("ing"):
            families.add(token[:-3])
        if len(token) > 4 and token.endswith("y"):
            families.add(token[:-1])
        if len(token) > 3 and token.endswith("ies"):
            families.add(f"{token[:-3]}y")
        if len(token) > 3 and token.endswith("s"):
            families.add(token[:-1])
    return families


def _direct_context_read_relevance(
    *, tool: dict[str, Any], query_tokens: set[str], action_tokens: set[str]
) -> bool:
    read_tokens = _tool_search_tokens(tool)
    return (
        _strong_token_overlap_count(query_tokens, read_tokens) > 0
        or _strong_token_overlap_count(action_tokens, read_tokens) >= 2
    )


def _best_scatter_read_action(
    *,
    passes: list[dict[str, Any]],
    available_read_names: set[str],
    already_read_names: set[str],
) -> dict[str, Any] | None:
    best_score = -1.0
    best_calls: list[dict[str, Any]] | None = None

    for p in passes:
        action = p.get("action", {})
        if action.get("action") != "tool_calls":
            continue
        read_calls = []
        for tool_call in action.get("tool_calls", []) or []:
            name = tool_call.get("tool_name")
            if name not in available_read_names or name in already_read_names:
                continue
            read_calls.append(
                {
                    "tool_name": name,
                    "arguments": tool_call.get("arguments") or {},
                }
            )
        if not read_calls:
            continue

        score = float(p.get("confidence", 0.5))
        if score > best_score:
            best_score = score
            best_calls = read_calls

    if best_calls is None:
        return None
    return {"action": "tool_calls", "content": "", "tool_calls": best_calls}


def _already_read_tool_names(
    transcript: list[dict[str, Any]], available_read_names: set[str]
) -> set[str]:
    seen: set[str] = set()
    for message in transcript:
        if message.get("role") == "tool":
            name = message.get("name")
            if name in available_read_names:
                seen.add(name)
        for tool_call in message.get("tool_calls", []) or []:
            name = tool_call.get("tool_name")
            if name in available_read_names:
                seen.add(name)
    return seen


def _read_tool_after_latest_user(
    transcript: list[dict[str, Any]], available_read_names: set[str]
) -> bool:
    latest_user_index = -1
    for index, message in enumerate(transcript):
        if message.get("role") == "user":
            latest_user_index = index
    if latest_user_index < 0:
        return False

    for message in transcript[latest_user_index + 1 :]:
        if message.get("role") == "tool" and message.get("name") in available_read_names:
            return True
        for tool_call in message.get("tool_calls", []) or []:
            if tool_call.get("tool_name") in available_read_names:
                return True
    return False


def _any_read_tool_after_latest_user(transcript: list[dict[str, Any]]) -> bool:
    latest_user_index = -1
    for index, message in enumerate(transcript):
        if message.get("role") == "user":
            latest_user_index = index
    if latest_user_index < 0:
        return False

    for message in transcript[latest_user_index + 1 :]:
        if message.get("role") == "tool" and _is_read_tool_name(
            str(message.get("name") or "")
        ):
            return True
        for tool_call in message.get("tool_calls", []) or []:
            if _is_read_tool_name(str(tool_call.get("tool_name") or "")):
                return True
    return False


def _fill_missing_read_arguments(
    action: dict[str, Any],
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    plan_action: dict[str, Any],
) -> dict[str, Any]:
    tools_by_name = _tool_by_name(tools)
    fixed_calls = []
    for tool_call in action.get("tool_calls", []) or []:
        name = tool_call.get("tool_name", "")
        arguments = tool_call.get("arguments") or {}
        if (
            name in tools_by_name
            and _is_preference_read_tool(tools_by_name[name])
            and _schema_arguments_need_repair(
                arguments, _tool_parameters(tools_by_name[name])
            )
        ):
            arguments = _preference_read_arguments(
                tools_by_name[name], transcript, plan_action
            )
        fixed_calls.append({"tool_name": name, "arguments": arguments})
    return {"action": "tool_calls", "content": "", "tool_calls": fixed_calls}


def _tool_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: tool
        for tool in tools
        if (name := _tool_name(tool))
    }


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name") if isinstance(tool, dict) else None
    return name if isinstance(name, str) else ""


def _tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        return function["parameters"]
    params = tool.get("parameters") if isinstance(tool, dict) else None
    return params if isinstance(params, dict) else {}


def _preference_read_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool for tool in tools if _is_preference_read_tool(tool)]


def _preference_reference_present(
    transcript: list[dict[str, Any]], plan_action: dict[str, Any]
) -> bool:
    text = " ".join(
        part
        for part in (
            _latest_text_by_role(transcript, "user"),
            plan_action.get("content") if plan_action.get("action") == "respond" else "",
        )
        if isinstance(part, str)
    ).lower()
    markers = (
        "default",
        "favorite",
        "favourite",
        "preference",
        "preferences",
        "prefer",
        "preferred",
        "profile",
        "saved",
        "usual",
    )
    return any(marker in text for marker in markers)


def _is_preference_read_tool(tool: dict[str, Any]) -> bool:
    name = _tool_name(tool).lower()
    function = tool.get("function") if isinstance(tool, dict) else None
    description = ""
    if isinstance(function, dict):
        description = str(function.get("description") or "")
    elif isinstance(tool, dict):
        description = str(tool.get("description") or "")
    params = _tool_parameters(tool)
    schema_text = json.dumps(params, ensure_ascii=False).lower()
    return (
        name.startswith("get_")
        and ("preference" in name or "preference" in description.lower())
    ) or "preference" in schema_text


def _preference_read_arguments(
    preference_tool: dict[str, Any],
    transcript: list[dict[str, Any]],
    plan_action: dict[str, Any],
) -> dict[str, Any]:
    return _schema_default_arguments(_tool_parameters(preference_tool))


def _schema_arguments_need_repair(arguments: dict[str, Any], schema: dict[str, Any]) -> bool:
    if not arguments:
        return True
    required = schema.get("required", [])
    if isinstance(required, list) and any(name not in arguments for name in required):
        return True
    return False


def _schema_default_arguments(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return {}
    return {
        name: _schema_default_value(prop_schema)
        for name, prop_schema in properties.items()
        if isinstance(name, str) and isinstance(prop_schema, dict)
    }


def _schema_default_value(schema: dict[str, Any]) -> Any:
    if "default" in schema:
        return schema["default"]
    schema_type = schema.get("type")
    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        return _schema_default_arguments(schema)
    if schema_type == "boolean":
        return True
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    if schema_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        return minimum if isinstance(minimum, (int, float)) else 0
    if schema_type == "array":
        return []
    if schema_type == "string":
        return ""
    return None


# ----- prompts & schemas ------------------------------------------------

_META_PROPS = {
    "confidence": {
        "type": "number",
        "description": "0-1 confidence in the proposed action being correct.",
    },
    "dispositions": {
        "type": "object",
        "required": list(PROPOSITION_KINDS),
        "additionalProperties": False,
        "properties": {
            kind: {"type": "string", "enum": list(DISPOSITIONS)}
            for kind in PROPOSITION_KINDS
        },
    },
    "recommendation": {"type": "string", "enum": list(RECOMMENDATIONS)},
}


def _action_schema_with(extra: dict[str, Any], required_extra: list[str]) -> dict:
    schema = json.loads(json.dumps(NEXT_ACTION_OUTPUT_SCHEMA))  # deep copy
    schema["properties"].update(extra)
    schema["required"] = schema["required"] + required_extra
    return schema


SCATTER_PASS_SCHEMA = _action_schema_with(
    _META_PROPS, ["confidence", "dispositions", "recommendation"]
)

SHARPEN_SCHEMA = _action_schema_with(
    {
        "resolved_proposition_kinds": {
            "type": "array",
            "items": {"type": "string", "enum": list(PROPOSITION_KINDS)},
            "description": "Which queued propositions this revision resolves.",
        },
        "decision": {"type": "string", "enum": list(RECOMMENDATIONS)},
    },
    ["resolved_proposition_kinds", "decision"],
)

CANDIDATE_REVIEW_SCHEMA = _action_schema_with(
    {
        "selected_candidate_index": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Index of the candidate that best supports the returned action. "
                "Use the nearest candidate if the returned action is a revision."
            ),
        },
        "approved_candidate": {
            "type": "boolean",
            "description": (
                "True only when the returned action exactly approves one supplied "
                "candidate. False when the returned action revises or replaces it."
            ),
        },
        "unresolved_proposition_kinds": {
            "type": "array",
            "items": {"type": "string", "enum": list(PROPOSITION_KINDS)},
            "description": (
                "Any remaining uncertainty that should be sharpened before "
                "emitting the action."
            ),
        },
        "issue_category": {
            "type": "string",
            "enum": [
                "none",
                "tool_availability",
                "tool_schema",
                "missing_information",
                "missing_confirmation",
                "policy_compliance",
                "unrequested_side_effect",
                "completed_action",
                "other",
            ],
        },
        "explanation": {
            "type": "string",
            "description": "Short reason for the selected or revised action.",
        },
    },
    [
        "selected_candidate_index",
        "approved_candidate",
        "unresolved_proposition_kinds",
        "issue_category",
        "explanation",
    ],
)

ADVERSARIAL_SCHEMA = {
    "type": "object",
    "required": ["violation_found", "severity", "category", "explanation"],
    "additionalProperties": False,
    "properties": {
        "violation_found": {"type": "boolean"},
        "severity": {"type": "string", "enum": ["none", "low", "high"]},
        "category": {
            "type": "string",
            "enum": list(PROPOSITION_KINDS) + ["correctness", "none"],
        },
        "explanation": {"type": "string"},
    },
}

FINAL_REVIEW_SCHEMA = _action_schema_with(
    {
        "approved": {
            "type": "boolean",
            "description": (
                "True when draft_action is safe, grounded, schema-valid, and "
                "ready to send unchanged. False when the output action is a "
                "corrected safer next action."
            ),
        },
        "issue_category": {
            "type": "string",
            "enum": [
                "none",
                "tool_availability",
                "tool_schema",
                "missing_information",
                "missing_confirmation",
                "policy_compliance",
                "unrequested_side_effect",
                "other",
            ],
        },
        "explanation": {
            "type": "string",
            "description": "Short reason for approval or revision.",
        },
    },
    ["approved", "issue_category", "explanation"],
)


SCATTER_SYSTEM = (
    CEREBRAS_DEVELOPER_INSTRUCTIONS
    + "\n\nYou are one independent voter in a panel. Decide the single best "
    "next action AND judge these atomic propositions about THIS turn, using "
    "only the supplied tool list, the transcript, and the policies:\n"
    "- feasibility: is the user's request achievable with the available tools?\n"
    "- tool_availability: are the exact tools you need present in the supplied "
    "list? Judge ONLY from that list; do not assume tools exist that are not "
    "listed, and do not speculate about why something might be missing.\n"
    "- parameter_determinacy: are all required parameter values uniquely "
    "determined by the request, stored preferences, or context?\n"
    "- policy_compliance: does the action comply with every policy?\n"
    "Mark each ok / uncertain / blocked. Recommend act, clarify, or "
    "acknowledge_limit.\n\n"
    "MISSING-INFORMATION RULE: when a required parameter is "
    "under-specified, do NOT ask the user yet. The values you need are usually "
    "retrievable, not already in the transcript -- so the correct next action "
    "is an INFORMATION-GATHERING TOOL CALL: read stored user preferences "
    "through the supplied preference-read tool and relevant state/context "
    "(get_* tools) to resolve the value. Recommend 'clarify' "
    "only after such tools genuinely cannot determine it. Asking the user when "
    "the value was internally resolvable is a FAILURE, and so is acting on a "
    "guessed value. Before a state-changing action on an ambiguous request, "
    "gather the relevant get_* state first. If multiple tool options seem "
    "possible and an unread get_* status/context tool overlaps those options, "
    "call that read tool before asking the user to choose. Numeric values are "
    "scoped to the operation they modify; do not copy a number from one "
    "operation to another tool argument merely because the argument name is the "
    "same.\n\n"
    "EXISTING-RESOURCE RULE: before creating or setting a new resource that "
    "may already exist, such as active navigation, an existing plan, or an "
    "already-set mode, first establish its current state from the transcript or "
    "a get_* read. Then use the tool that edits, replaces, or updates it rather "
    "than one that creates a new one. A create or set-new call on an already "
    "active resource fails.\n\n"
    "READ-BEFORE-SET RULE: before any state-changing tool call (set_*, "
    "open_close_*, activate/deactivate) on a vehicle subsystem, first call that "
    "subsystem's read tool (get_* / *_status / positions) if one is supplied "
    "and you have not already read it this turn -- e.g. read climate settings "
    "before changing climate, read window positions before moving windows. "
    "These reads are required context, not optional; skipping the relevant read "
    "is a failure even when you are confident of the value.\n\n"
    "CONFIRMATION RULE: a yes/no confirmation required by policy, safety, or "
    "a tool description is policy_compliance, not parameter_determinacy. A "
    "confirmation response must state the concrete intended operation and its "
    "required tool argument names and values before the user confirms. When "
    "the next step after confirmation would be a tool call, include the planned "
    "tool name and each required argument as a name/value detail from the "
    "supplied schema. Vague wording like asking whether to proceed or whether "
    "to do the requested operation is insufficient and does not authorize a "
    "later tool call."
)

CANDIDATE_REVIEW_SYSTEM = (
    CEREBRAS_DEVELOPER_INSTRUCTIONS
    + "\n\nYou are a verifier and reranker for a panel of independently "
    "generated next-action candidates. Use inference-time compute to compare "
    "the candidates against the visible transcript, policy text, tool results, "
    "and supplied tool schemas. Do not infer task type, hidden evaluator "
    "state, missing tools, or scoring rules.\n"
    "Select the candidate that is best grounded, policy-compliant, complete, "
    "and minimal. If every candidate has a fixable defect, return a corrected "
    "next action instead of blindly voting with the majority. If information "
    "needed for a required argument is missing and an available read-only tool "
    "can obtain it, revise to that information-gathering call before asking "
    "the user or guessing. If a policy or tool description requires user "
    "confirmation, return a short confirmation response that names the planned "
    "operation and each required argument name/value. If an action has already "
    "succeeded after the latest user request, do not repeat it. If uncertainty "
    "remains after your review, list the proposition kinds that still need "
    "sharpening."
)

SHARPEN_SYSTEM = (
    CEREBRAS_DEVELOPER_INSTRUCTIONS
    + "\n\nYou are refining a draft action to resolve a specific flagged "
    "uncertainty. Produce the improved next action and list which proposition "
    "kinds you have now resolved.\n"
    "When the flagged uncertainty is an under-specified parameter "
    "(parameter_determinacy), prefer an action that RESOLVES it via an "
    "information-gathering tool -- typically a supplied preference-read tool "
    "or the relevant get_* state tool -- over asking the user. Treat asking the user "
    "as a last resort, used only after those tools cannot determine the value. "
    "When a draft question asks the user to choose among possible tools, first "
    "use any unread relevant get_* status/context tool if its schema arguments "
    "are already known. Numeric values only ground the operation they are "
    "textually attached to; use schema defaults/ranges instead of copying a "
    "number across unrelated operations. "
    "If the uncertainty truly cannot be resolved from the tools, transcript, "
    "and policies, then prefer clarify or acknowledge_limit over guessing.\n"
    "If the remaining uncertainty is a required yes/no confirmation, the next "
    "assistant response must name the planned tool operation and each required "
    "argument name/value from the supplied schema before asking for "
    "confirmation. If prior confirmation did not expose those name/value "
    "details, revise to a detailed confirmation response instead of treating "
    "it as satisfied."
)

ADVERSARIAL_SYSTEM = (
    "You are a strict red-team reviewer for an in-car assistant. Your job is to "
    "find at least one way the proposed action violates a policy, is factually "
    "or operationally incorrect, calls a tool not in the supplied list, or "
    "supplies an unjustified parameter. Use ONLY the system prompt/policies, "
    "the transcript, and the supplied tool schemas. Do NOT try to reconstruct "
    "any grading rubric or evaluation scores. If after genuine effort you find "
    "no violation, say so."
)

FINAL_REVIEW_SYSTEM = (
    CEREBRAS_DEVELOPER_INSTRUCTIONS
    + "\n\nYou are the final reviewer for one in-car assistant next action. "
    "Use ONLY the supplied transcript, policies/system text present in that "
    "transcript, and supplied tool schemas/descriptions. Do NOT reconstruct or "
    "optimize against evaluator scoring internals.\n"
    "Approve the draft only if it is the correct next action now: every tool is "
    "available, every argument is schema-valid and grounded in the transcript, "
    "tool results, policy text, or tool schema, required confirmations and "
    "preconditions visible in policy/tool descriptions are satisfied, and the "
    "action adds no unrequested side effects. If information is missing and a "
    "supplied read-only tool can gather it, revise to that information-gathering "
    "tool call before asking the user to choose among possible tools. Ensure "
    "numeric arguments are grounded in the same operation text or in the tool "
    "schema; do not propagate a number mentioned for a different operation just "
    "because another tool has the same argument name. If user confirmation or "
    "clarification is required, revise to a "
    "short user-facing response. A user yes only satisfies confirmation if the "
    "prior assistant message identified the planned tool operation and each "
    "required argument name/value from the supplied schema; otherwise revise to "
    "that detailed confirmation response. Do not approve a vague confirmation "
    "question that only asks whether to proceed or whether to do the requested "
    "operation. "
    "If a draft tool call includes extra unsupported "
    "or unrequested operations, revise to the minimal justified next action. "
    "When approved=true, repeat the draft action exactly. When approved=false, "
    "the action fields must contain the corrected next action."
)


def _scatter_prompt(
    transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> str:
    return json.dumps(
        {
            "available_tools": tools,
            "conversation_transcript": transcript,
            "instructions": (
                "Choose one next action and judge the four propositions."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def _candidate_review_prompt(
    *,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    aggregate_action: dict[str, Any],
    aggregate_queue: list[Proposition],
) -> str:
    return json.dumps(
        {
            "available_tools": tools,
            "conversation_transcript": transcript,
            "candidate_actions": candidates,
            "weighted_vote_winner": aggregate_action,
            "aggregate_unresolved_proposition_kinds": [
                p.kind for p in aggregate_queue
            ],
            "instructions": (
                "Rerank candidate_actions using only the transcript, visible "
                "policies, tool results, and supplied schemas. Return the "
                "selected or corrected next action plus any unresolved "
                "proposition kinds that still need sharpening."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def _sharpen_prompt(
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    plan_action: dict[str, Any],
    queue: list[Proposition],
    target: Proposition,
) -> str:
    return json.dumps(
        {
            "available_tools": tools,
            "conversation_transcript": transcript,
            "draft_action": plan_action,
            "open_uncertainties": [p.kind for p in queue],
            "focus_uncertainty": {
                "kind": target.kind,
                "status": target.dominant_unresolved(),
            },
            "instructions": (
                "Resolve focus_uncertainty if the transcript/tools/policies "
                "support it; otherwise clarify or acknowledge the limit."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def _adversarial_prompt(
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    plan_action: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "available_tools": tools,
            "conversation_transcript": transcript,
            "proposed_action": plan_action,
            "instructions": "Hunt for one concrete violation in proposed_action.",
        },
        ensure_ascii=False,
        indent=2,
    )


def _final_review_prompt(
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    plan_action: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "available_tools": tools,
            "conversation_transcript": transcript,
            "draft_action": plan_action,
            "instructions": (
                "Approve draft_action only if it is fully supported. Otherwise "
                "return one corrected next action."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_planner_from_env(
    *,
    model: str,
    api_base: str,
    service_tier: str | None,
    reasoning_effort: str | None,
    logger: Any | None = None,
) -> ScatterSharpenPlanner:
    return ScatterSharpenPlanner(
        model=model,
        api_base=api_base,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
        config=ScatterSharpenConfig.from_env(),
        logger=logger,
    )
