"""Deterministic dense retrieval over a small synthetic behavior library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

if __package__:
    from .policy_rag import policy_token_count
else:  # pragma: no cover - loose-script compatibility
    from policy_rag import policy_token_count


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


@dataclass(frozen=True)
class SyntheticExample:
    example_id: str
    retrieval_text: str
    episode: str


@dataclass(frozen=True)
class FewShotSelection:
    example_id: str
    text: str
    token_count: int
    score: float


SYNTHETIC_EXAMPLES: tuple[SyntheticExample, ...] = (
    SyntheticExample(
        "unavailable_observable",
        "unknown unavailable absent sensor observable cannot read finish available work terminal",
        """SYNTHETIC EXAMPLE — unavailable observable:
User: Enable the invented cabin scent and tell me whether its imaginary saturation sensor is clear.
Assistant: calls get_scent_state, then enable_scent.
Tools: scent enabled; saturation = unknown.
Assistant: The scent is enabled. Saturation is unavailable from the car, so I cannot verify that part. I will not query it again.
User: Can you still finish what is possible?
Assistant: Yes. The requested activation is complete; only the unavailable sensor check remains unverified. Check the cabin directly if that observation matters.
Pattern: complete the doable action, name the unavailable observable exactly once, offer the nearest practical alternative, and close instead of retrying or asking for the missing value.""",
    ),
    SyntheticExample(
        "approval_execute",
        "yes approve confirm affirmation proposed action execute immediately do not restate",
        """SYNTHETIC EXAMPLE — approval means execute:
Assistant: I can send the prepared maintenance note to the selected recipient. Send it now?
User: Yes.
Assistant: immediately calls send_note with the already confirmed recipient and content.
Tool: sent = true.
Assistant: The note was sent.
User: Did you use the approved draft?
Assistant: Yes. I executed the already-confirmed recipient and content without changing either.
Pattern: a clear affirmation binds to the immediately preceding concrete proposal. Execute it at once, preserve its confirmed arguments, verify the tool result, and give one terminal acknowledgement. Do not show the same draft again or request a second confirmation.""",
    ),
    SyntheticExample(
        "degree_mapping",
        "preference preferred level degree exact fully all every setting apply numeric",
        """SYNTHETIC EXAMPLE — exact degree and preference:
User: Set the invented seat pulse to my preferred level and match the wheel pulse.
Assistant: calls get_preferences.
Tool: preferred pulse level = 2.
Assistant: calls set_seat_pulse(level=2) and set_wheel_pulse(level=2).
Tools: both succeed.
Assistant: Seat and wheel pulse are both level 2.
User: Did you approximate my preference?
Assistant: No. I used the exact stored level and applied that same value to both requested controls.
Pattern: retrieve the relevant preference, use its exact runtime value, map degree words to canonical arguments, and apply the same resolved level wherever the user explicitly requests a match. Never silently weaken a requested degree.""",
    ),
    SyntheticExample(
        "target_time",
        "condition weather arrival destination target time later when applies evaluate branch",
        """SYNTHETIC EXAMPLE — evaluate at the target:
User: If it will be cold when I arrive at 21:00, enable the invented warmer.
Assistant: calls get_weather(destination, time=21:00), not the current time.
Tool: target-time temperature is below the stated threshold.
Assistant: calls enable_warmer.
Tool: success.
Assistant: The 21:00 condition is met, so the warmer is enabled.
User: Why did you not use the weather right now?
Assistant: Because the condition governs arrival, not departure; current weather would answer the wrong question.
Pattern: identify the condition's target location, date, and time before reading data. Query those target parameters, state which target was evaluated, then take exactly the branch supported by that result.""",
    ),
    SyntheticExample(
        "minimal_diff",
        "only change keep unchanged minimal diff duplicate satisfied setting edit preserve other fields",
        """SYNTHETIC EXAMPLE — minimal state diff:
User: Raise only the invented vent rhythm by one; keep scent mode unchanged.
Assistant: calls get_vent_state.
Tool: rhythm=1, scent=FRESH.
Assistant: calls set_vent_rhythm(level=2) only.
Tool: success.
Assistant: Vent rhythm is level 2; scent mode was not changed.
User: Did you restart the whole vent configuration?
Assistant: No. I changed only the requested rhythm field and preserved every other observed field.
Pattern: read the current resource, compute the smallest requested delta, avoid unrelated mutations, and consult the episode mutation log before executing. If that exact delta already succeeded, do not issue it again; simply report completion.""",
    ),
    SyntheticExample(
        "modify_active",
        "active navigation route exists modify replace edit do not recreate destination waypoint",
        """SYNTHETIC EXAMPLE — modify an active resource:
User: Change the active route to include the invented Museum Stop.
Assistant: calls get_navigation_state.
Tool: an active route exists.
Assistant: calls navigation_replace_waypoints with the retained route data plus Museum Stop; it does not create a new route.
Tool: success.
Assistant: The active route now includes Museum Stop.
User: Did you create another route?
Assistant: No. I preserved the active route and edited its waypoint list.
Pattern: establish whether a resource is active before choosing a mutation family. When active, use a catalog edit, update, or replace operation and retain unaffected fields. Use creation only when the read establishes that no matching active resource exists.""",
    ),
    SyntheticExample(
        "grounded_claim",
        "state value claim read verify grounded tool result range temperature status tell me current",
        """SYNTHETIC EXAMPLE — ground state claims:
User: Tell me the invented reservoir level.
Assistant: calls get_reservoir_level.
Tool: level=43 percent.
Assistant: The reservoir level is 43 percent.
User: Are you estimating that number?
Assistant: No. It is the exact value returned by the read.
Pattern: every quoted state value must have a matching observable in a tool result from this episode. If the result is unknown, absent, or an error, state that the value is unavailable. Never convert a requested target, assumption, or nearby value into a claim about current state. End with one concise, grounded answer.""",
    ),
)


SYNTHETIC_EXAMPLES_V12R: tuple[SyntheticExample, ...] = (
    SyntheticExample(
        "unavailable_observable",
        SYNTHETIC_EXAMPLES[0].retrieval_text,
        """SYNTHETIC EXAMPLE — unavailable observable:
User: Enable Cedar cabin scent and report the pollen sensor.
Assistant: calls get_cedar_scent_state, then enable_cedar_scent.
Tools: scent enabled; pollen sensor is unknown.
Assistant: Cedar scent is enabled. The pollen reading is unavailable, so I cannot verify it and will not query it again.
User: Finish whatever is possible.
Assistant: The activation is complete. For pollen, check the blue dashboard indicator instead.
Pattern: complete doable work, state an unavailable observable once, offer a concrete available alternative, and close.""",
    ),
    SyntheticExample(
        "approval_execute",
        SYNTHETIC_EXAMPLES[1].retrieval_text,
        """SYNTHETIC EXAMPLE — approval means execute:
Assistant: I can send the prepared tire note to Dana Reed. Send it now?
User: Yes.
Assistant: immediately calls send_tire_note for Dana Reed with the approved text.
Tool: sent is true.
Assistant: The tire note was sent to Dana Reed.
User: Did you use the approved note?
Assistant: Yes. I sent the confirmed recipient and wording unchanged.
Pattern: bind a clear affirmation to the concrete proposal, execute immediately with its confirmed arguments, verify success, and acknowledge once.""",
    ),
    SyntheticExample(
        "degree_mapping",
        SYNTHETIC_EXAMPLES[2].retrieval_text,
        """SYNTHETIC EXAMPLE — exact degree and preference:
User: Set Cedar seat pulse to my preferred level and match the wheel pulse.
Assistant: calls get_cedar_preferences.
Tool: preferred pulse level is 2.
Assistant: calls set_cedar_seat_pulse with level 2 and set_cedar_wheel_pulse with level 2.
Tools: both succeed.
Assistant: Seat and wheel pulse are both level 2.
Pattern: read the relevant preference, use its exact value, map degree words to canonical arguments, and apply the resolved level to every explicitly matched control.""",
    ),
    SyntheticExample(
        "target_time",
        SYNTHETIC_EXAMPLES[3].retrieval_text,
        """SYNTHETIC EXAMPLE — evaluate at the target:
User: If Harbor Point will be below 6 degrees when I arrive at 21:00, enable Cedar warmer.
Assistant: calls get_harbor_weather for Harbor Point at 21:00.
Tool: temperature is 4 degrees.
Assistant: calls enable_cedar_warmer.
Tool: success.
Assistant: Harbor Point is forecast at 4 degrees for 21:00, so Cedar warmer is enabled.
Pattern: identify the condition's target place and time, query those concrete parameters, and take exactly the supported branch.""",
    ),
    SyntheticExample(
        "minimal_diff",
        SYNTHETIC_EXAMPLES[4].retrieval_text,
        """SYNTHETIC EXAMPLE — minimal state diff:
User: Raise only Cedar vent rhythm by one and keep scent mode unchanged.
Assistant: calls get_cedar_vent_state.
Tool: rhythm is 1 and scent is FRESH.
Assistant: calls set_cedar_vent_rhythm with level 2 only.
Tool: success.
Assistant: Cedar vent rhythm is level 2; scent remains FRESH.
Pattern: read current state, compute the smallest requested delta, preserve unrelated fields, and do not repeat a mutation already recorded as successful.""",
    ),
    SyntheticExample(
        "modify_active",
        SYNTHETIC_EXAMPLES[5].retrieval_text,
        """SYNTHETIC EXAMPLE — modify an active resource:
User: Add Harbor Museum to my active Cedar route.
Assistant: calls get_cedar_route_state.
Tool: route cedar_17 is active from Oak Plaza to River Hall.
Assistant: calls cedar_replace_waypoints for route cedar_17 with Harbor Museum inserted.
Tool: success.
Assistant: Active route cedar_17 now includes Harbor Museum.
Pattern: read whether the resource is active, then edit or replace that active resource while retaining unaffected fields; create only when no matching resource exists.""",
    ),
    SyntheticExample(
        "grounded_claim",
        SYNTHETIC_EXAMPLES[6].retrieval_text,
        """SYNTHETIC EXAMPLE — ground state claims:
User: Tell me the Cedar reservoir level.
Assistant: calls get_cedar_reservoir_level.
Tool: level is 43 percent.
Assistant: The Cedar reservoir level is 43 percent.
User: Is that an estimate?
Assistant: No. It is the exact value returned by the read.
Pattern: support every state value with a tool result from this episode. If a result is unknown, absent, or errored, state that it is unavailable instead of substituting another value.""",
    ),
)


# Event injection reuses the frozen W5 placeholder-free library.  The only
# derived wording is E1's published CAR-bench objective: terminal limitation
# acknowledgement.  Keeping this as a new tuple preserves historical W5 runs.
EVENT_TRIGGERED_EXAMPLES: tuple[SyntheticExample, ...] = tuple(
    SyntheticExample(
        example.example_id,
        example.retrieval_text,
        (
            example.episode
            + "\nRequired terminal acknowledgment: I acknowledge this limitation: "
            "the requested reading is unavailable and cannot be verified with "
            "the available tools."
            if example.example_id == "unavailable_observable"
            else example.episode
        ),
    )
    for example in SYNTHETIC_EXAMPLES_V12R
)


def selection_for_example(
    example_id: str,
    *,
    examples: tuple[SyntheticExample, ...] = EVENT_TRIGGERED_EXAMPLES,
) -> FewShotSelection:
    """Return a deterministic transient selection without similarity retrieval."""

    example = next((item for item in examples if item.example_id == example_id), None)
    if example is None:
        raise KeyError(example_id)
    text = f"[event-triggered synthetic worked episode: {example.example_id}]\n{example.episode}"
    return FewShotSelection(
        example_id=example.example_id,
        text=text,
        token_count=policy_token_count(text),
        score=1.0,
    )


class FewShotRetriever:
    """Top-1 cosine retrieval with stable library-order tie-breaking."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        examples: tuple[SyntheticExample, ...] = SYNTHETIC_EXAMPLES,
    ) -> None:
        self.embedding_backend = embedding_backend
        self.examples = examples
        self._vectors = embedding_backend.embed(
            [example.retrieval_text for example in examples]
        )
        if self._vectors is None or len(self._vectors) != len(examples):
            raise RuntimeError("Model2Vec could not embed the synthetic example library")

    def select(self, user_text: str) -> FewShotSelection:
        rows = self.embedding_backend.embed([user_text])
        if rows is None or len(rows) != 1:
            raise RuntimeError("Model2Vec could not embed the user turn")
        query = rows[0]
        scores = [sum(a * b for a, b in zip(query, vector)) for vector in self._vectors]
        index = max(range(len(scores)), key=lambda item: (scores[item], -item))
        example = self.examples[index]
        text = f"[retrieved synthetic worked episode: {example.example_id}]\n{example.episode}"
        return FewShotSelection(
            example_id=example.example_id,
            text=text,
            token_count=policy_token_count(text),
            score=scores[index],
        )


def append_selection_to_latest_user(
    messages: list[dict], selection: FewShotSelection
) -> list[dict]:
    """Return a copy with the example appended only to the latest user message."""

    copied = [dict(message) for message in messages]
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") != "user":
            continue
        content = str(copied[index].get("content") or "")
        copied[index]["content"] = f"{content}\n\n{selection.text}"
        break
    return copied
