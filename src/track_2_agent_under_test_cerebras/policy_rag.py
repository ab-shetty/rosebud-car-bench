"""Deterministic per-decision retrieval over the supplied in-car policy."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol


POLICY_HEADING = "# In-Car Assistant agent policy"
POLICY_RAG_TOP_K = 2
POLICY_SUBRULE_TARGET_TOKENS = 120
POLICY_CORE_MIN_IMPLICATIONS = 2
POLICY_CORE_TOKEN_CAP = 125

# Generic subject -> tool-family fragments.  These are applied only to the
# current task's supplied tool names; they are not a bundled tool catalog.
POLICY_SUBJECT_TOOL_FAMILIES: dict[str, tuple[str, ...]] = {
    "toll": ("route", "navigation"),
    "charging": ("charging", "charger"),
    "soc": ("soc", "charging", "battery", "range"),
    "battery": ("battery", "charging", "soc", "range"),
    "window": ("window", "sunroof", "sunshade"),
    "sunroof": ("sunroof", "sunshade"),
    "defrost": ("defrost", "fan", "airflow", "conditioning"),
    "climate": ("climate", "temperature", "fan", "airflow", "conditioning"),
    "email_message": ("email", "text", "message", "contact", "phone_call"),
    "navigation": ("navigation", "route", "waypoint", "location", "poi"),
    "lights": ("light", "beam"),
    "wipers": ("wiper",),
    "preference": ("get_user_preferences", "preference"),
    "weather": ("weather", "fog"),
    "route_selection": ("route", "navigation"),
    "broad_scope": (
        "window",
        "light",
        "wiper",
        "door",
        "seat",
        "mirror",
        "sunroof",
        "sunshade",
    ),
    "confirmation": (),
    "units": ("datetime", "distance", "temperature", "weather", "route"),
}

_SUBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (subject, re.compile(pattern, re.IGNORECASE))
    for subject, pattern in (
        ("toll", r"\btolls?\b"),
        ("charging", r"\bcharg(?:e|ed|er|ers|ing|ing station|ing stations)\b"),
        ("soc", r"\b(?:soc|state of charge)\b"),
        ("battery", r"\b(?:battery|range)\b"),
        ("window", r"\b(?:windows?|windshield|sunroof|sunshade)\b"),
        ("sunroof", r"\b(?:sunroofs?|sunshades?)\b"),
        ("defrost", r"\bdefrost(?:er|ing)?\b"),
        (
            "climate",
            r"\b(?:climate|temperature|fan|airflow|air conditioning|a/c|hvac)\b",
        ),
        ("email_message", r"\b(?:emails?|messages?|texts?|contacts?|phone calls?)\b"),
        (
            "navigation",
            r"\b(?:navigation|navigate|routes?|destinations?|waypoints?|stops?|poi)\b",
        ),
        ("lights", r"\b(?:lights?|headlights?|beams?|fog lights?)\b"),
        ("wipers", r"\bwipers?\b"),
        ("preference", r"\b(?:preference|preferences|preferred)\b"),
        ("weather", r"\b(?:weather|fog)\b"),
        (
            "route_selection",
            r"\b(?:fastest|shortest|alternatives?|route selection)\b",
        ),
        (
            "broad_scope",
            r"\b(?:all|every|entire)\b(?:\W+\w+){0,4}\W+"
            r"(?:windows?|lights?|wipers?|doors?|seats?|mirrors?|sunroofs?|sunshades?|"
            r"devices?|controls?|tools?|actions?|options?)\b",
        ),
        ("confirmation", r"\b(?:confirmation|confirm|confirmed|approval|approve|yes)\b"),
        (
            "units",
            r"\b(?:units?|format|24h|24-hour|datetime|times?|kilometers?|kilometres?|meters?|metres?|celsius)\b",
        ),
    )
)

_RULE_START_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+]\s+|\d+[.)]\s+)")
_NUMBERED_RULE_START_RE = re.compile(
    r"^[ \t]*(?:\d+[.)]\s+|[-*+]\s+(?:(?:LLM-POL:|AUT-POL:)?\d{3}:))",
    re.IGNORECASE,
)
_POLICY_NUMBER_RE = re.compile(
    r"(?:LLM-POL:|AUT-POL:)?(?P<number>\d{3}):", re.IGNORECASE
)


@dataclass(frozen=True)
class PolicyCoreImplication:
    section_id: str
    violated_trials: int = 0
    cure_sites: int = 0

    @property
    def implication_count(self) -> int:
        return self.violated_trials + self.cure_sites


# Full p3i17-p3i22 + final_v2 + frontier_sweep replay census. Membership uses
# only policy-section identifiers and aggregate implication counts.
POLICY_CORE_CENSUS: tuple[PolicyCoreImplication, ...] = (
    PolicyCoreImplication("policy-cde4b554d59a", cure_sites=39),
    PolicyCoreImplication("policy-93060b7231dc", cure_sites=36),
    PolicyCoreImplication("policy-ba61361a1da4", cure_sites=15),
    PolicyCoreImplication("policy-024", violated_trials=8),
    PolicyCoreImplication("policy-022", violated_trials=5),
    PolicyCoreImplication("policy-002", violated_trials=1, cure_sites=3),
    PolicyCoreImplication("policy-008", violated_trials=4),
    PolicyCoreImplication("policy-021", violated_trials=4),
)


@dataclass(frozen=True)
class PolicySection:
    section_id: str
    text: str
    order: int
    keys: frozenset[str]
    own_keys: frozenset[str] = frozenset()
    inherited_keys: frozenset[str] = frozenset()
    parent_id: str = ""
    parent_order: int = -1
    header: str = ""
    own_text: str = ""
    dense_vector: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.own_keys and not self.inherited_keys and self.keys:
            object.__setattr__(self, "own_keys", self.keys)
        if not self.parent_id:
            object.__setattr__(self, "parent_id", self.section_id)
        if self.parent_order < 0:
            object.__setattr__(self, "parent_order", self.order)
        if not self.own_text:
            object.__setattr__(self, "own_text", self.text)


@dataclass(frozen=True)
class PolicyCorpus:
    sections: tuple[PolicySection, ...]
    always_on_ids: tuple[str, ...]
    split_parent_count: int = 0
    split_child_count: int = 0
    parent_key_document_frequency: tuple[tuple[str, int], ...] = ()
    core_token_count: int = 0
    rejected_core_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyRank:
    section: PolicySection
    lexical_rank: int | None
    dense_rank: int | None
    lexical_score: float
    dense_score: float | None
    rrf_score: float


@dataclass(frozen=True)
class PolicyRetrieval:
    sections: tuple[PolicySection, ...]
    matched_sections: tuple[PolicySection, ...]
    empty: bool
    text: str
    token_count: int
    ranking: tuple[PolicyRank, ...] = ()

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(section.section_id for section in self.sections)


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return one deterministic vector per text, or None if unavailable."""


def policy_token_count(text: str) -> int:
    """Return a deterministic, conservative word-token accounting unit."""

    return len(re.findall(r"\S+", text))


def segment_policy_text(system_text: str) -> tuple[str, ...]:
    """Split the supplied policy into verbatim top-level rule sections."""

    start = system_text.find(POLICY_HEADING)
    if start < 0:
        return ()
    policy_text = system_text[start:]
    lines = policy_text.splitlines()
    sections: list[str] = []
    current: list[str] = []
    current_is_numbered_rule = False
    previous_was_nested_item = False

    def flush() -> None:
        if not current:
            return
        text = "\n".join(current).rstrip()
        if text:
            sections.append(text)
        current.clear()

    for line in lines:
        marker = _RULE_START_RE.match(line)
        is_top_rule = marker is not None and len(marker.group("indent").expandtabs(4)) == 0
        is_numbered_rule = bool(is_top_rule and _NUMBERED_RULE_START_RE.match(line))
        is_header = line.startswith("#")
        continues_numbered_rule = (
            is_top_rule
            and not is_numbered_rule
            and current_is_numbered_rule
            and previous_was_nested_item
        )
        if (is_top_rule and not continues_numbered_rule) or is_header:
            flush()
            current_is_numbered_rule = is_numbered_rule
        if current or is_top_rule or is_header:
            current.append(line)
        previous_was_nested_item = bool(
            line.strip()
            and marker is not None
            and len(marker.group("indent").expandtabs(4)) > 0
        )
    flush()
    return tuple(section for section in sections if section != POLICY_HEADING)


def distilled_wiki_entries(system_text: str) -> tuple[str, ...]:
    """Mirror the benchmark's numbered LLM/AUT policy-entry distillation."""

    return tuple(
        section
        for section in segment_policy_text(system_text)
        if _POLICY_NUMBER_RE.search(section)
    )


def split_policy_subrules(
    section_text: str,
    *,
    max_tokens: int = POLICY_SUBRULE_TARGET_TOKENS,
) -> tuple[str, ...]:
    """Split a long rule at structural boundaries and repeat its header verbatim."""

    text = section_text.rstrip()
    lines = text.splitlines()
    if (
        policy_token_count(text) <= max_tokens
        or len(lines) < 2
        or _RULE_START_RE.match(lines[0]) is None
    ):
        return (text,)

    header = lines[0]
    units = _subrule_units(lines[1:])
    if not units:
        return (text,)

    payload_limit = max(1, max_tokens - policy_token_count(header))
    expanded: list[str] = []
    for unit in units:
        expanded.extend(_split_subclauses(unit, payload_limit))

    children = ["\n".join((header, unit)).rstrip() for unit in expanded]
    return tuple(children) if len(children) > 1 else (text,)


def extract_policy_subjects(text: str) -> frozenset[str]:
    return frozenset(
        subject for subject, pattern in _SUBJECT_PATTERNS if pattern.search(text)
    )


def policy_query_keys(
    *, tool_names: Iterable[str], subjects: Iterable[str]
) -> frozenset[str]:
    return frozenset(
        {f"tool:{name.casefold()}" for name in tool_names if name}
        | {f"subject:{subject}" for subject in subjects if subject}
    )


def build_policy_corpus(
    system_text: str,
    *,
    tool_names: Iterable[str] = (),
    distilled_entries: Iterable[str] | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    core_implications: Iterable[PolicyCoreImplication] = POLICY_CORE_CENSUS,
    core_token_cap: int = POLICY_CORE_TOKEN_CAP,
) -> PolicyCorpus:
    """Build a stable, deduplicated policy corpus and deterministic key index."""

    parsed = list(segment_policy_text(system_text))
    distilled = list(
        distilled_wiki_entries(system_text)
        if distilled_entries is None
        else distilled_entries
    )
    parent_sections: list[str] = []
    seen: set[str] = set()
    for text in parsed + distilled:
        normalized = _dedupe_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parent_sections.append(text.rstrip())

    raw_sections: list[tuple[str, str, str, int, str]] = []
    split_parent_count = 0
    split_child_count = 0
    available_tools = tuple(sorted({name for name in tool_names if name}))
    parent_key_sets: list[frozenset[str]] = []
    parent_ids: list[str] = []
    used_parent_ids: set[str] = set()
    for parent_order, text in enumerate(parent_sections):
        parent_id = _stable_section_id(text, used_parent_ids)
        used_parent_ids.add(parent_id)
        parent_ids.append(parent_id)
        parent_key_sets.append(_section_keys(text, available_tools))
        children = split_policy_subrules(text)
        if len(children) == 1:
            raw_sections.append((children[0], "", children[0], parent_order, parent_id))
        else:
            header = text.splitlines()[0]
            for child in children:
                own_text = "\n".join(child.splitlines()[1:]).rstrip()
                raw_sections.append((child, header, own_text, parent_order, parent_id))
        if len(children) > 1:
            split_parent_count += 1
            split_child_count += len(children)

    vectors: list[tuple[float, ...]] = [()] * len(raw_sections)
    if embedding_backend is not None and raw_sections:
        embedded = embedding_backend.embed([text for text, _, _, _, _ in raw_sections])
        if embedded is not None and len(embedded) == len(raw_sections):
            vectors = [tuple(map(float, vector)) for vector in embedded]

    sections: list[PolicySection] = []
    used_ids: set[str] = set()
    for order, ((text, header, own_text, parent_order, parent_id), dense_vector) in enumerate(
        zip(raw_sections, vectors)
    ):
        section_id = _stable_section_id(text, used_ids)
        used_ids.add(section_id)
        own_keys = _section_keys(own_text, available_tools)
        inherited_keys = _section_keys(header, available_tools) - own_keys
        keys = own_keys | inherited_keys
        sections.append(
            PolicySection(
                section_id=section_id,
                text=text,
                order=order,
                keys=frozenset(keys),
                own_keys=frozenset(own_keys),
                inherited_keys=frozenset(inherited_keys),
                parent_id=parent_id,
                parent_order=parent_order,
                header=header,
                own_text=own_text,
                dense_vector=dense_vector,
            )
        )

    core_selection = select_policy_core(
        sections,
        core_implications,
        min_implications=POLICY_CORE_MIN_IMPLICATIONS,
        max_tokens=core_token_cap,
    )
    return PolicyCorpus(
        tuple(sections),
        tuple(section.section_id for section, _ in core_selection.admitted),
        split_parent_count=split_parent_count,
        split_child_count=split_child_count,
        parent_key_document_frequency=tuple(
            sorted(Counter(key for keys in parent_key_sets for key in keys).items())
        ),
        core_token_count=core_selection.token_count,
        rejected_core_ids=tuple(
            section.section_id for section, _ in core_selection.rejected
        ),
    )


@dataclass(frozen=True)
class PolicyCoreSelection:
    admitted: tuple[tuple[PolicySection, PolicyCoreImplication], ...]
    rejected: tuple[tuple[PolicySection, PolicyCoreImplication], ...]
    token_count: int


def select_policy_core(
    sections: Iterable[PolicySection],
    implications: Iterable[PolicyCoreImplication],
    *,
    min_implications: int = POLICY_CORE_MIN_IMPLICATIONS,
    max_tokens: int = POLICY_CORE_TOKEN_CAP,
) -> PolicyCoreSelection:
    """Select the implication-ranked eligible prefix under the core token cap."""

    section_by_id = {section.section_id: section for section in sections}
    candidates = [
        (section_by_id[item.section_id], item)
        for item in implications
        if item.section_id in section_by_id
        and (
            item.violated_trials >= min_implications
            or item.cure_sites >= min_implications
        )
    ]
    candidates.sort(key=lambda row: (-row[1].implication_count, row[0].order))
    admitted: list[tuple[PolicySection, PolicyCoreImplication]] = []
    rejected: list[tuple[PolicySection, PolicyCoreImplication]] = []
    stopped = False
    for candidate in candidates:
        if stopped:
            rejected.append(candidate)
            continue
        proposed = admitted + [candidate]
        proposed_text = "\n\n".join(section.text for section, _ in proposed)
        if policy_token_count(proposed_text) > max_tokens:
            stopped = True
            rejected.append(candidate)
            continue
        admitted.append(candidate)
    text = "\n\n".join(section.text for section, _ in admitted)
    return PolicyCoreSelection(
        admitted=tuple(admitted),
        rejected=tuple(rejected),
        token_count=policy_token_count(text),
    )


def retrieve_policy_sections(
    corpus: PolicyCorpus,
    query_keys: Iterable[str],
    *,
    top_k: int = POLICY_RAG_TOP_K,
    max_tokens: int | None = None,
    query_text: str = "",
    embedding_backend: EmbeddingBackend | None = None,
) -> PolicyRetrieval:
    """Retrieve by parent-IDF lexical/dense RRF with stable order ties."""

    query = frozenset(query_keys)
    key_document_frequency = _parent_document_frequency(corpus)
    lexical = sorted(
        (
            (
                sum(
                    1.0 / key_document_frequency[key]
                    for key in section.own_keys & query
                    if key_document_frequency[key]
                )
                / math.sqrt(len(section.own_keys))
                if section.own_keys
                else 0.0,
                section.order,
                section,
            )
            for section in corpus.sections
            if section.keys & query
        ),
        key=lambda item: (-item[0], item[1]),
    )
    core_ids = set(corpus.always_on_ids)
    core = [section for section in corpus.sections if section.section_id in core_ids]
    lexical = [item for item in lexical if item[2].section_id not in core_ids]
    lexical_ranks = {
        section.section_id: rank
        for rank, (_, _, section) in enumerate(lexical, start=1)
    }
    lexical_scores = {section.section_id: score for score, _, section in lexical}

    dense: list[tuple[float, int, PolicySection]] = []
    query_vector = _query_vector(query_text, embedding_backend)
    if query_vector:
        dense = sorted(
            (
                (_cosine(query_vector, section.dense_vector), section.order, section)
                for section in corpus.sections
                if section.section_id not in core_ids and section.dense_vector
            ),
            key=lambda item: (-item[0], item[1]),
        )
    dense_ranks = {
        section.section_id: rank
        for rank, (_, _, section) in enumerate(dense, start=1)
    }
    dense_scores = {section.section_id: score for score, _, section in dense}

    candidate_ids = set(lexical_ranks) | set(dense_ranks)
    ranking = sorted(
        (
            PolicyRank(
                section=section,
                lexical_rank=lexical_ranks.get(section.section_id),
                dense_rank=dense_ranks.get(section.section_id),
                lexical_score=lexical_scores.get(section.section_id, 0.0),
                dense_score=dense_scores.get(section.section_id),
                rrf_score=(
                    (1.0 / (60 + lexical_ranks[section.section_id]) if section.section_id in lexical_ranks else 0.0)
                    + (1.0 / (60 + dense_ranks[section.section_id]) if section.section_id in dense_ranks else 0.0)
                ),
            )
            for section in corpus.sections
            if section.section_id in candidate_ids
        ),
        key=lambda item: (-item.rrf_score, item.section.order),
    )
    matched = [item.section for item in ranking[: max(0, top_k)]]

    selected = list(core)
    injected_matched: list[PolicySection] = []
    for section in matched:
        proposed = selected + [section]
        proposed_text = "\n\n".join(item.text for item in proposed)
        if max_tokens is not None and policy_token_count(proposed_text) > max_tokens:
            break
        selected.append(section)
        injected_matched.append(section)
    selected.sort(key=lambda section: section.order)
    text = "\n\n".join(section.text for section in selected)
    return PolicyRetrieval(
        sections=tuple(selected),
        matched_sections=tuple(injected_matched),
        empty=not ranking,
        text=text,
        token_count=policy_token_count(text),
        ranking=tuple(ranking),
    )


def _section_keys(text: str, available_tools: tuple[str, ...]) -> frozenset[str]:
    subjects = extract_policy_subjects(text)
    keys = {f"subject:{subject}" for subject in subjects}
    lowered = text.casefold()
    for tool_name in available_tools:
        normalized_tool = tool_name.casefold()
        if re.search(
            rf"(?<![a-z0-9_]){re.escape(normalized_tool)}(?![a-z0-9_])", lowered
        ):
            keys.add(f"tool:{normalized_tool}")
            continue
        if any(
            fragment in normalized_tool
            for subject in subjects
            for fragment in POLICY_SUBJECT_TOOL_FAMILIES.get(subject, ())
        ):
            keys.add(f"tool:{normalized_tool}")
    return frozenset(keys)


def _parent_document_frequency(corpus: PolicyCorpus) -> Counter[str]:
    if corpus.parent_key_document_frequency:
        return Counter(dict(corpus.parent_key_document_frequency))
    parent_keys: dict[str, set[str]] = {}
    for section in corpus.sections:
        parent_keys.setdefault(section.parent_id, set()).update(section.own_keys)
    return Counter(key for keys in parent_keys.values() for key in keys)


def _query_vector(
    query_text: str, embedding_backend: EmbeddingBackend | None
) -> tuple[float, ...]:
    if not query_text or embedding_backend is None:
        return ()
    vectors = embedding_backend.embed([query_text])
    if vectors is None or len(vectors) != 1:
        return ()
    return tuple(map(float, vectors[0]))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return numerator / (left_norm * right_norm)


def _dedupe_text(text: str) -> str:
    return re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", text.strip())


def _subrule_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for line in lines:
        if _RULE_START_RE.match(line) is not None and current:
            units.append("\n".join(current).rstrip())
            current = []
        current.append(line)
    if current:
        units.append("\n".join(current).rstrip())
    return [unit for unit in units if unit.strip()]


def _split_subclauses(text: str, max_tokens: int) -> tuple[str, ...]:
    if policy_token_count(text) <= max_tokens:
        return (text,)
    boundaries = tuple(
        match.start()
        for match in re.finditer(
            r"(?<=[.;:])(?=\s+)|(?<=,)(?=\s+(?:and|or|but|if|when|then|specifically)\b)",
            text,
            re.IGNORECASE,
        )
    )
    if not boundaries:
        return (text,)
    fragments: list[str] = []
    start = 0
    for boundary in (*boundaries, len(text)):
        fragment = text[start:boundary]
        if not fragment:
            continue
        if fragments and policy_token_count(fragments[-1] + fragment) <= max_tokens:
            fragments[-1] += fragment
        else:
            fragments.append(fragment)
        start = boundary
    return tuple(fragment for fragment in fragments if fragment.strip()) or (text,)


def _stable_section_id(text: str, used_ids: set[str]) -> str:
    number_match = _POLICY_NUMBER_RE.search(text)
    base = (
        f"policy-{number_match.group('number')}"
        if number_match is not None
        else "policy-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    )
    if base not in used_ids:
        return base
    suffix = 2
    while f"{base}-{suffix}" in used_ids:
        suffix += 1
    return f"{base}-{suffix}"
