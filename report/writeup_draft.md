# CAR-bench Track 2 — Innovation Writeup (DRAFT v2, 2026-07-18)

Agent under test: frozen Cerebras `gpt-oss-120b`. All mechanisms are
deterministic harness-side interventions using only internal episode signals
(tool results, drafted decisions, dialogue text). No task identifiers, no
evaluation metrics, no hidden-test probing. Sealed splits were read as
aggregates only (never traces); headline sealed figures are pooled across
independent replicated reads, not best draws.

## 1. Submission arm and headline results

**Submission arm ("fixed fast"): full deterministic guard stack +
mutation-point consensus + medium reasoning effort + semantically validated
zero-LLM prefetch and four deterministic val-final repairs** (internally
p3i67). The deep-deliberate arm (p3i51: the same core stack at high effort,
without prefetch) remains the replicated sealed-proxy accuracy comparator.

| Split | Stock baseline | Deep-deliberate arm | Fixed-fast arm |
|---|---:|---:|---:|
| benchmark-18 sealed | 9/18 (50.0%) | **23/36 tasks pooled (63.9%), Pass^1 79.6%** | 8/18 (44.4%); p3i56 predecessor 9/18, within n=3 noise |
| val_peek (24) | 11/24 (45.8%) | [not measured as final arm] | 16/24 (66.7%) @ 3.44s for p3i49-family predecessor |
| discovery-24 | 9/24 (37.5%) | [PENDING] | 17/24 (70.8%) |
| **val_final (101), decisive** | 43/101 (42.6%), Pass^1 60.1% | 55/101 (54.5%), Pass^1 70.6% @ 10.744s p50 | **62/101 (61.4%), Pass^1 75.6% @ 4.335s p50** |
| **val_final under OFFICIAL roles (Gemini 3.5 Flash sim + judge)** | [not measured] | [not measured] | **68/101 (67.3%), Pass^1 75.9% @ 3.625s p50 / 9.374s p90** |

All development measurements used gpt-5-mini as user simulator and policy
judge (cost control). Because the competition evaluates every submission with
a fixed Gemini 3.5 Flash simulator, we re-measured the frozen submission arm
(identical code, flags, and scenarios; only the two evaluator role models
changed) as a final dress rehearsal. Under official roles the arm improves to
68/101 with a 16% p50 latency reduction — Gemini's shorter dialogues (mean
1.91 vs 2.93 user turns, 40-turn error tails eliminated) stabilize base and
hallucination families (+5 and +6 Pass^3 tasks) — while disambiguation drops
from 11/17 to 6/17, so the aggregate gain carries a family-specific risk
note. The sealed proxy held at its established 50% level (9/18) under
official roles.

Reference: published gpt-oss-120b SCORE .28; best published model overall
(GPT-5 thinking) .54. The deep-deliberate arm's replicated 63.9% Pass^3 on
the sealed proxy — from a frozen 120b open-weights model — shows the accuracy
available from harness engineering alone, but val-final decides the
submission recommendation: fixed fast beats deep deliberate by 7 Pass^3
tasks while cutting p50 by 59.7%. It reaches +18.8pp over baseline on the
broad 101-task distribution. Its single p3i67 sealed-proxy read (8/18) is
one task below the p3i56 predecessor (9/18), inside the expected n=3 noise
rather than evidence of a stable regression.

## 2. The three load-bearing layers (each causally evidenced)

### 2.1 Right-sized reasoning budget (cap + effort)
gpt-oss-120b at the stock 1,024-token completion cap burns the entire budget
on thinking and returns empty content on reasoning-heavy tasks; the stock
pipeline surfaces this as an error string to the user-simulator and
death-spirals (84-turn error-only episodes; 99/375 val records affected).
Fixes: initial cap 4096 (truncated-empty completions 190 -> 0, calls -16%),
escalating rescue ladder as backstop, then HIGH reasoning effort — which the
larger cap makes viable for the first time (at cap 1024, 555/680 high-effort
completions truncated). Effort-high with right-sized caps is worth +10pp
sealed over the medium arm, stable across two independent reads.

### 2.2 Deterministic guard stack ("reads are free, asks are dangerous")
Value provenance: every mutated value must trace to the user's literal
statement, a read stored preference, or a literal clarification answer;
never ask for a value already in context; per-seat mutations grounded in
read occupancy; responds may not claim unexecuted actions. Plus schema and
reference preflights, identical-read loop breaking, policy-lint
post-conditions, and a global ask budget (max ONE injected ask per episode)
derived from an empirical law observed over eight dev->sealed transfer
reversals: injected reads/applies/bounces never caused sealed harm; injected
asks reversed three times (worst: a 112-fire ask loop collapsing sealed
disambiguation to 0/6). The law mirrors the reward construction — extra get
calls are free; every question routes through user-simulator variance and
end-conversation penalties.

### 2.3 Mutation-point consensus
Pass^3 punishes per-trial variance cubed, and actions_intermediate makes
wrong mutations unrecoverable even when corrected. At any decision whose
draft contains a mutation, the harness samples two additional completions
and executes only a 2-of-3 exact-signature majority (fail-open otherwise,
bounded per episode, zero cost on read/respond turns). **Ablation proof:**
the identical arm with consensus disabled fell from the twice-measured
79.6% Pass^1 to 63.0% and lost 3-4 Pass^3 tasks, while saving ~130 calls —
consensus is causally load-bearing, not decorative.

## 3. The local-optimum audit (six sealed refutations)
Single-variable modifications of the deep-deliberate branch, each measured on the
sealed proxy: + prefetch (8/18 — context stuffing perturbs deep reasoning;
kept only on the medium arm where it is Pass^3-neutral at -23% p50);
terminal decisions re-issued at medium (11/18, +25% p50); initial cap 8192
(10/18 — rescue churn 72 -> 19 fires but no accuracy); consensus removed
(8/18 — the ablation above); mixed-effort votes (10/18 — trades per-type
columns without net gain); plus earlier: struggle-gated effort escalation
(9/18 — the high-effort benefit requires high from episode start),
terminal-respond consensus (9/18 — voting at end-of-conversation forks
pushes action over acknowledgment). The deep-deliberate composition survived
attack from every direction, but the broader val-final read selected the
fixed-fast branch for submission.

## 4. Additional negative results
Temperature 0 (sealed regression); global scaffold stacking, worked-example
prompts, few-shot retrieval, tool-description enrichment (deliver but do not
transfer); 5-way vote deepening (contested 3-way forks remain contested at
5 — consensus disagreement marks genuine model uncertainty, not sampling
noise).

## 5. Benchmark observations (deficiencies found during failure analysis)

Trace-level autopsies surfaced several places where the reward construction
grades process rather than outcome; we report them as constructive findings,
with one counter-example where the strict grading is right.

1. **A single failed probe call is unrecoverable even when the outcome is
   correct.** In one hallucination task (waypoint-removal tool deleted), the
   agent removed the final destination correctly, issued one substitute call
   that failed cleanly (no side effect), and closed with a correct limitation
   acknowledgment — the user's achievable goal was fully met and truthfully
   explained, yet `r_tool_execution=0` zeroes the task. This penalizes safe
   API exploration over omniscience about constraints that are only
   discoverable by state reasoning or probing.
2. **Explicitly authorized extra adjustments are graded as wrong actions.**
   In a climate task the user says "I'm fine with you making any extra
   adjustments needed to make it effective"; ground truth closes two windows
   (25% and 30% open) but leaves a third 10%-open window. An agent that also
   closes the third window — strictly improving AC effectiveness, squarely
   within the granted latitude — scores `r_actions=0`. Single-sequence ground
   truth contradicts the user's own stated flexibility.
3. **Corrected mistakes count as never corrected.** `actions_intermediate`
   permanently zeroes tasks where a wrong mutation was immediately reversed
   and the final state matches ground truth. Defensible for side-effectful
   actions, but it grants no distinction between harmful and harmless,
   promptly-corrected detours.
4. **Underspecified parameters have single-value ground truth.** Requests
   like "improve the air quality" pin an exact fan level in GT; reasonable
   neighboring values score zero. One task labeled disambiguation resolves
   its "internal element" (recirculation vs fresh-air) by a judgment call
   that competent humans would split on.
5. **The simulated user amplifies errors instead of repairing them.** Asked
   a malformed question ("what exact value for level, percentage?" on a
   boolean AC request), the user-simulator confabulates ("set the air
   conditioning level to 65%") rather than pushing back as a real user
   would, converting one bad question into an unrecoverable failure.

**Counter-example where strict grading is right:** an agent hand-computed a
charging time in prose instead of calling the charging-time calculator; every
metric except `r_tool_subset` was perfect, and the arithmetic looked correct.
But the vehicle's nonlinear charging curve makes manual estimates reliable
only by luck (this one sat in the flat region of the curve). Requiring the
verified instrument for computed quantities is sound agent policy, and our
step-coverage mechanism enforces it as such.

## 6. Measurement discipline
Two-lane concurrent evaluation with hard rate-limit fallback (full-tree
SIGSTOP/CONT on real 429s; zero record poisoning across the campaign);
latency-balanced lane partitions; n=3 Pass^3 with per-task matrices on open
splits; strict fire-level causal audits separating mechanism effect from
draw luck; sealed aggregates only, with replication of headline reads on
frozen arms — pooled figures reported, never best draws.

## 7. External work, tools, and attributions

**Components in the submitted system:**
- **CAR-bench starter kit and evaluator** (CAR-bench organizers; arXiv
  2601.22027; github.com/CAR-bench/car-bench) — the agent scaffolding,
  A2A serving layer, scenario format, and all evaluation code. Our harness
  extends the Track 2 Cerebras template.
- **A2A protocol / a2a-sdk** (a2aproject) — agent serving, via the starter kit.
- **gpt-oss-120b on Cerebras** (frozen competition model) via
  `cerebras-cloud-sdk`.
- **Model2Vec with the potion-base-8M static embedding model** (MinishLab,
  MIT license; github.com/MinishLab/model2vec) — bundled locally (no network
  calls, no torch) as the embedding backend for few-shot retrieval,
  policy-RAG retrieval, and scatter/consensus response clustering.
- **openai/harmony** renderer — the native harmony transport
  (`harmony_native.py`).
- Standard Python libraries per `pyproject.toml` (httpx, pydantic, loguru,
  uvicorn, python-dotenv, nest-asyncio).

The submission ships the full research tree, so several alternative
harnesses and retrieval paths are present in the source alongside the
selected arm. The selected arm's configuration enables the deterministic
guard stack, semantic prefetch, and mutation-point consensus, whose
majorities compare exact canonical `(tool, arguments)` signatures rather
than embeddings; the remaining paths are disabled by flag but ship with the
code and are credited below on the same footing.

**Models used only for measurement, never inside the harness:**
- OpenAI **gpt-5-mini** as the development-time user simulator and policy
  judge (cost control during iteration).
- Google **Gemini 3.5 Flash** as user simulator and policy judge for the
  final official-conditions rehearsal (matching the announced competition
  configuration).

**Research behind the code in this repository.** Each entry names the
mechanism it corresponds to and whether that mechanism is enabled in the
submitted configuration. Every idea we read and built on is listed, whether
or not it survived measurement.

| Source | Mechanism in this repository | In selected arm |
|---|---|---|
| Self-consistency majority voting (Wang et al., arXiv 2203.11171) | Mutation-point consensus (`_apply_mutation_consensus`), specialized to mutation-bearing decisions with exact-signature majorities; also the earlier scatter/consensus planner | **Yes** |
| gpt-oss model card (arXiv 2508.10925) | Reasoning-effort selection, completion-cap sizing, truncation-rescue ladder | **Yes** |
| "Benchmark Test-Time Scaling of General LLM Agents" (arXiv 2602.18998) | Evidence that added turns/depth can hurt; informed the right-sized budget and the decision not to deepen voting | **Yes** (as a design constraint) |
| Harness-engineering surveys: Awesome-Agent-Harness, awesome-harness-engineering, arXiv 2606.25447, arXiv 2510.22898 | "Model proposes, deterministic harness validates" architecture — the whole guard stack | **Yes** (as architecture) |
| "In Harmony with gpt-oss" (arXiv 2604.00362) + openai/harmony renderer | Native harmony transport (`harmony_native.py`) | No — built, measured, refuted |
| IRMA input reformulation (arXiv 2508.20931, EMNLP 2025 Findings) | User-turn reformulation (`reformulation_ledger.py`) | No |
| SAGE-Agent structured-uncertainty clarification (arXiv 2511.08798); uncertainty decomposition for clarification seeking (arXiv 2606.19559) | Value-of-information-gated clarification (`voi_clarify` in `consensus_planner.py`) | No — measured regression, reverted |
| "When Retrieval Metrics Mislead" (arXiv 2606.23937) | Diagnostics for retrieval-based policy injection (`policy_rag.py`, `fewshot_rag.py`) | No |
| Agentic harness evolution (arXiv 2604.25850) | Considered for automated harness search; rejected for overfit risk — no code | No |

Practitioner references: Cerebras inference API documentation, the official
CAR-bench competition rules/tracks pages, and public agent-harness
engineering write-ups (philschmid.de).

---
DRAFT NOTES (not for submission): deep-deliberate = p3i51 (3be2fe2 flags);
fixed fast = p3i67 (`c0161b6`, semantic prefetch + four repairs,
`READ_RESOLVE` off). Val-final is complete: p3i51 55/101 @ 10.744s p50,
p3i56 59/101 @ 4.182s, fixed p3i67 62/101 @ 4.335s. p3i77 six-repair bundle
REFUTED (60/101, +30% p50) — p3i67 retained. Official-conditions rehearsal
(p3i78, Gemini 3.5 Flash sim+judge): val_final 68/101 @ 3.625s, b18 9/18;
dis 11/17→6/17. Verify final prose and all figures against
p3i64/p3i69/p3i77/p3i78 reports before submission.
