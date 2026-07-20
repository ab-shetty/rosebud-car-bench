# p3i77 val-final repair round 2

## Result

The frozen p3i77 arm completed **303/303** val-final trials with **0 `info.error`**. It scored **60/101 Pass^3 (59.4%)** and **229/303 Pass^1 (75.6%)**, at **5.627s p50 / 22.964s p90**.

This is **-2 Pass^3 tasks vs p3i69** (62/101) with identical Pass^1 (229/303). It is **+29.8% p50 latency vs p3i69** (4.335s). The bundle did not beat the decisive p3i69 fixed-fast arm on Pass^3 and is **not the submission recommendation** from this read.

| Arm | Pass^3 | Pass^1 | p50 | p90 |
|---|---:|---:|---:|---:|
| Raw baseline | 43/101 (42.6%) | 182/303 (60.1%) | 3.153s | 11.156s |
| Wave-5 reference | 57/101 (56.4%) | 215/303 (71.0%) | 3.535s | 22.367s |
| p3i56 fast | 59/101 (58.4%) | 216/303 (71.3%) | 4.182s | 11.699s |
| p3i69 / p3i67 fixed fast | **62/101 (61.4%)** | **229/303 (75.6%)** | **4.335s** | **13.270s** |
| p3i77 repair bundle | **60/101 (59.4%)** | **229/303 (75.6%)** | **5.627s** | **22.964s** |

## Per-type

| Type | Pass^3 | Pass^1 | p50 | p90 |
|---|---:|---:|---:|---:|
| base | 27/42 (64.3%) | 96/126 (76.2%) | 5.784s | 14.441s |
| hallucination | 25/42 (59.5%) | 97/126 (77.0%) | 5.055s | 23.858s |
| disambiguation | 8/17 (47.1%) | 36/51 (70.6%) | 6.107s | 36.931s |

## Replay gate

Before live measurement, the archive-backed replay gate passed every named fork: route budget on `base_49`, `base_77`, `disambiguation_23`, `disambiguation_31`; navigation intent on `base_95`, `hallucination_69`, `hallucination_81`, `hallucination_87`, and opened-b18 `hallucination_78`; step coverage on `base_23`, `base_39`, `base_89`, opened-b18 `base_98`, and policy-mandated-weather `base_10`; P3 ask gate v2 on `base_93` and opened-b18 `disambiguation_51`; textcall re-decision on `base_59` and `disambiguation_37`; and the limitation trigger on `hallucination_23`, `hallucination_13`, and `hallucination_21`. Focused tests passed **57/57**. The repository-wide suite passed 542/545; its three unrelated current failures were A2A exception-accounting (2) and historical scenario-matrix enumeration (1), outside this source change.

The burned-15 smoke then completed **15/15**, `info.error=0`, 15 episode contexts, exact flags, status 0. No mechanism/source edits occurred after replay or smoke.

## Named targets and strict causal reading

| Target | Family | p3i69 | p3i77 | Target-flag touched trials | Flag count | Reading |
|---|---|---:|---:|---:|---:|---|
| `base_49` | route-loop 0/3 | 0/3 | **0/3** | 3/3 | 124 | fired, no pass-count movement |
| `base_77` | route-loop 0/3 | 0/3 | **0/3** | 3/3 | 211 | fired, no pass-count movement |
| `disambiguation_23` | route-loop 0/3 | 0/3 | **0/3** | 3/3 | 200 | fired, no pass-count movement |
| `disambiguation_31` | route-loop 0/3 | 0/3 | **0/3** | 3/3 | 110 | fired, no pass-count movement |
| `base_95` | navigation-intent 0/3 | 0/3 | **3/3** | 3/3 | 3 | fire-touched improvement; compatible with causality, n=3 noisy |
| `hallucination_69` | navigation-intent 0/3 | 0/3 | **1/3** | 3/3 | 3 | fire-touched improvement; compatible with causality, n=3 noisy |
| `hallucination_81` | navigation-intent 0/3 | 0/3 | **1/3** | 3/3 | 3 | fire-touched improvement; compatible with causality, n=3 noisy |
| `hallucination_87` | navigation-intent 0/3 | 0/3 | **0/3** | 3/3 | 3 | fired, no pass-count movement |
| `base_39` | omission 0/3 | 0/3 | **0/3** | 3/3 | 3 | fired, no pass-count movement |
| `base_89` | omission 0/3 | 0/3 | **3/3** | 2/3 | 2 | fire-touched improvement; compatible with causality, n=3 noisy |
| `base_23` | dependent-read coverage | 1/3 | **1/3** | 3/3 | 3 | fired, no pass-count movement |
| `base_59` | textcall safety | 1/3 | **1/3** | 0/3 | 0 | no target fire and no pass-count movement |
| `disambiguation_37` | textcall safety | 1/3 | **3/3** | 0/3 | 0 | changed with no target fire: wobble only |
| `base_93` | non-numeric ask guard | 1/3 | **1/3** | 3/3 | 4 | fired, no pass-count movement |
| `hallucination_23` | limitation 0/3 | 0/3 | **1/3** | 3/3 | 3 | fire-touched improvement; compatible with causality, n=3 noisy |
| `hallucination_13` | limitation wobble | 2/3 | **3/3** | 2/3 | 2 | fire-touched improvement; compatible with causality, n=3 noisy |
| `hallucination_21` | limitation wobble | 2/3 | **3/3** | 0/3 | 0 | changed with no target fire: wobble only |

The ten predeclared deterministic val targets were the first ten rows above. Recoveries were `base_95` 0/3→3/3 and `base_89` 0/3→3/3, both fire-touched. The four route-loop targets all fired but remained 0/3. Live counters also show a boundedness defect in the new route path: 384 cumulative fires, 466 blocked reads, and 366 terminal-limitation decisions, concentrated in seven tasks. The run was frozen, so this was reported rather than fixed mid-gate.

`disambiguation_37` improved 1/3→3/3 with step-coverage fires but **no textcall fire**, so TEXTCALL cannot claim the recovery. `hallucination_21` changed without a limitation-classifier fire and is wobble. Every such no-fire change is treated as stochastic variation.

Known opened-b18 residuals: `disambiguation_28` (recirculation vs fresh-air judgment) and `base_46` (fan-level choice) are consistently held wrong value judgments, not variance; no p3i77 repair targets them. No benchmark-18 measurement was run or accessed in this order.

## Every pass-count change vs p3i69

| Task | p3i69 | p3i77 | Any p3i77 repair touched changed trials? | Attribution |
|---|---:|---:|---|---|
| `base_5` | 2/3 | 0/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_17` | 3/3 | 2/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_35` | 3/3 | 2/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_37` | 2/3 | 3/3 | 1/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_45` | 1/3 | 0/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_47` | 3/3 | 1/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_51` | 1/3 | 2/3 | 0/3 | no repair fire: stochastic wobble |
| `base_53` | 2/3 | 1/3 | 1/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_55` | 2/3 | 3/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_61` | 3/3 | 2/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_63` | 2/3 | 3/3 | 0/3 | no repair fire: stochastic wobble |
| `base_89` | 0/3 | 3/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_91` | 3/3 | 2/3 | 0/3 | no repair fire: stochastic wobble |
| `base_95` | 0/3 | 3/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `base_97` | 2/3 | 3/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `disambiguation_5` | 3/3 | 2/3 | 0/3 | no repair fire: stochastic wobble |
| `disambiguation_17` | 3/3 | 2/3 | 1/3 | fire-touched; direction is compatible but not uniquely attributable |
| `disambiguation_19` | 3/3 | 2/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `disambiguation_27` | 3/3 | 2/3 | 0/3 | no repair fire: stochastic wobble |
| `disambiguation_33` | 2/3 | 3/3 | 0/3 | no repair fire: stochastic wobble |
| `disambiguation_37` | 1/3 | 3/3 | 1/3 | fire-touched; direction is compatible but not uniquely attributable |
| `disambiguation_39` | 3/3 | 2/3 | 0/3 | no repair fire: stochastic wobble |
| `disambiguation_47` | 1/3 | 2/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_13` | 2/3 | 3/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_21` | 2/3 | 3/3 | 0/3 | no repair fire: stochastic wobble |
| `hallucination_23` | 0/3 | 1/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_25` | 1/3 | 2/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_33` | 2/3 | 3/3 | 0/3 | no repair fire: stochastic wobble |
| `hallucination_37` | 3/3 | 2/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_51` | 3/3 | 1/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_53` | 2/3 | 3/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_55` | 2/3 | 1/3 | 1/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_59` | 3/3 | 2/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_61` | 3/3 | 2/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_67` | 3/3 | 2/3 | 2/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_69` | 0/3 | 1/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_81` | 0/3 | 1/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_83` | 1/3 | 0/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_91` | 1/3 | 0/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |
| `hallucination_96` | 2/3 | 3/3 | 3/3 | fire-touched; direction is compatible but not uniquely attributable |

## Episode-final mechanism counters

| Mechanism | Episode-final cumulative count |
|---|---:|
| Route-budget fires | 384 |
| Route-budget blocked reads | 466 |
| Route-budget terminal limitations | 366 |
| Navigation-intent bounces | 29 |
| Navigation-intent bounded pass-throughs | 112 |
| Step-coverage fires | 133 |
| Step-coverage re-decisions | 133 |
| P3 ask-gate-v2 suppressions | 38 |
| Textcall-guard fires | 8 |
| Textcall direct executes (must be zero) | 0 |
| Textcall re-decisions | 8 |
| Limitation-classifier calls | 30 |
| Classifier unavailable labels | 14 |
| Classifier available labels | 11 |
| Classifier errors | 0 |
| Classifier timeouts | 0 |
| Classifier malformed outputs | 5 |
| Classifier added latency (ms) | 13917.1 |
| Prefetch reads | 3630 |
| Semantic-prefetch valid calls emitted | 3630 |
| Semantic-prefetch invalid calls suppressed | 616 |
| Truncation rescues | 42 |
| Malformed-argument rescues | 41 |
| Placeholder-guard fires | 6 |
| Schema-preflight bounces | 17 |
| P1 context-value applies | 3 |
| P2 binding bounces | 0 |
| P3 preference reads | 23 |
| P3 fallback asks | 0 |
| P4 navigation redirects | 51 |
| P5 occupancy reads | 0 |
| P6 claim revises | 28 |
| 24-hour-format revises | 2 |
| Injected asks | 1 |
| Ask-budget suppressions | 0 |
| Repeated-read blocks | 84 |
| Route-reference bounces | 18 |
| Policy-lint revises | 19 |
| Mutation-consensus invocations | 369 |
| Consensus agreements | 239 |
| Consensus overrides | 36 |
| Consensus no-majority fallbacks | 130 |
| Consensus extra LLM calls | 767 |

Transport bookkeeping: 3261 LLM requests, 3261 responses, 0 real `queue_exceeded` events, and 0 other SDK-error events.

## 101-task matrix

| Task | Type | Lane | Baseline | p3i51 | p3i56 | p3i69 | p3i77 |
|---|---|:---:|---:|---:|---:|---:|---:|
| `base_3` | base | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_5` | base | A | 1/3 | 0/3 | 2/3 | 2/3 | **0/3** |
| `base_7` | base | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_9` | base | B | 2/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_11` | base | A | 1/3 | 3/3 | 2/3 | 3/3 | **3/3** |
| `base_13` | base | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_17` | base | A | 2/3 | 3/3 | 2/3 | 3/3 | **2/3** |
| `base_19` | base | A | 3/3 | 3/3 | 0/3 | 3/3 | **3/3** |
| `base_21` | base | A | 3/3 | 3/3 | 0/3 | 3/3 | **3/3** |
| `base_23` | base | B | 2/3 | 1/3 | 0/3 | 1/3 | **1/3** |
| `base_25` | base | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_27` | base | B | 3/3 | 2/3 | 3/3 | 3/3 | **3/3** |
| `base_31` | base | B | 2/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_33` | base | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_35` | base | B | 3/3 | 2/3 | 3/3 | 3/3 | **2/3** |
| `base_37` | base | A | 0/3 | 2/3 | 3/3 | 2/3 | **3/3** |
| `base_39` | base | A | 1/3 | 0/3 | 1/3 | 0/3 | **0/3** |
| `base_41` | base | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_45` | base | A | 0/3 | 0/3 | 3/3 | 1/3 | **0/3** |
| `base_47` | base | B | 0/3 | 0/3 | 1/3 | 3/3 | **1/3** |
| `base_49` | base | B | 3/3 | 0/3 | 0/3 | 0/3 | **0/3** |
| `base_51` | base | A | 0/3 | 3/3 | 2/3 | 1/3 | **2/3** |
| `base_53` | base | B | 0/3 | 0/3 | 2/3 | 2/3 | **1/3** |
| `base_55` | base | A | 1/3 | 3/3 | 2/3 | 2/3 | **3/3** |
| `base_59` | base | A | 2/3 | 1/3 | 0/3 | 1/3 | **1/3** |
| `base_61` | base | B | 0/3 | 1/3 | 1/3 | 3/3 | **2/3** |
| `base_63` | base | B | 3/3 | 3/3 | 3/3 | 2/3 | **3/3** |
| `base_65` | base | A | 2/3 | 1/3 | 3/3 | 3/3 | **3/3** |
| `base_67` | base | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_69` | base | B | 2/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_73` | base | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_75` | base | A | 3/3 | 2/3 | 0/3 | 3/3 | **3/3** |
| `base_77` | base | A | 2/3 | 0/3 | 0/3 | 0/3 | **0/3** |
| `base_79` | base | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_81` | base | B | 2/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_83` | base | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_87` | base | B | 2/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_89` | base | B | 0/3 | 0/3 | 0/3 | 0/3 | **3/3** |
| `base_91` | base | B | 3/3 | 3/3 | 3/3 | 3/3 | **2/3** |
| `base_93` | base | B | 3/3 | 0/3 | 0/3 | 1/3 | **1/3** |
| `base_95` | base | B | 2/3 | 2/3 | 1/3 | 0/3 | **3/3** |
| `base_97` | base | A | 1/3 | 2/3 | 2/3 | 2/3 | **3/3** |
| `disambiguation_3` | disambiguation | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_5` | disambiguation | A | 3/3 | 2/3 | 3/3 | 3/3 | **2/3** |
| `disambiguation_9` | disambiguation | B | 0/3 | 2/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_11` | disambiguation | A | 2/3 | 3/3 | 2/3 | 3/3 | **3/3** |
| `disambiguation_13` | disambiguation | B | 0/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_17` | disambiguation | B | 3/3 | 3/3 | 3/3 | 3/3 | **2/3** |
| `disambiguation_19` | disambiguation | A | 2/3 | 3/3 | 3/3 | 3/3 | **2/3** |
| `disambiguation_23` | disambiguation | B | 3/3 | 0/3 | 0/3 | 0/3 | **0/3** |
| `disambiguation_25` | disambiguation | A | 0/3 | 0/3 | 0/3 | 0/3 | **0/3** |
| `disambiguation_27` | disambiguation | A | 0/3 | 3/3 | 2/3 | 3/3 | **2/3** |
| `disambiguation_31` | disambiguation | A | 0/3 | 0/3 | 0/3 | 0/3 | **0/3** |
| `disambiguation_33` | disambiguation | A | 1/3 | 3/3 | 3/3 | 2/3 | **3/3** |
| `disambiguation_37` | disambiguation | B | 3/3 | 2/3 | 0/3 | 1/3 | **3/3** |
| `disambiguation_39` | disambiguation | A | 1/3 | 3/3 | 2/3 | 3/3 | **2/3** |
| `disambiguation_41` | disambiguation | B | 1/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_45` | disambiguation | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_47` | disambiguation | B | 1/3 | 2/3 | 1/3 | 1/3 | **2/3** |
| `hallucination_3` | hallucination | A | 3/3 | 2/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_5` | hallucination | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_7` | hallucination | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_9` | hallucination | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_11` | hallucination | A | 1/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_13` | hallucination | A | 1/3 | 3/3 | 3/3 | 2/3 | **3/3** |
| `hallucination_17` | hallucination | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_19` | hallucination | A | 1/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_21` | hallucination | A | 2/3 | 3/3 | 3/3 | 2/3 | **3/3** |
| `hallucination_23` | hallucination | B | 0/3 | 0/3 | 0/3 | 0/3 | **1/3** |
| `hallucination_25` | hallucination | B | 0/3 | 0/3 | 1/3 | 1/3 | **2/3** |
| `hallucination_27` | hallucination | B | 0/3 | 3/3 | 1/3 | 2/3 | **2/3** |
| `hallucination_31` | hallucination | B | 3/3 | 3/3 | 3/3 | 2/3 | **2/3** |
| `hallucination_33` | hallucination | B | 2/3 | 1/3 | 3/3 | 2/3 | **3/3** |
| `hallucination_35` | hallucination | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_37` | hallucination | A | 0/3 | 1/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_39` | hallucination | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_41` | hallucination | B | 0/3 | 2/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_45` | hallucination | A | 0/3 | 2/3 | 0/3 | 3/3 | **3/3** |
| `hallucination_47` | hallucination | B | 1/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_49` | hallucination | B | 1/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_51` | hallucination | B | 0/3 | 2/3 | 1/3 | 3/3 | **1/3** |
| `hallucination_53` | hallucination | A | 3/3 | 3/3 | 3/3 | 2/3 | **3/3** |
| `hallucination_55` | hallucination | A | 1/3 | 0/3 | 1/3 | 2/3 | **1/3** |
| `hallucination_59` | hallucination | B | 0/3 | 3/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_61` | hallucination | A | 0/3 | 2/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_63` | hallucination | B | 3/3 | 2/3 | 2/3 | 1/3 | **1/3** |
| `hallucination_65` | hallucination | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_67` | hallucination | B | 2/3 | 2/3 | 2/3 | 3/3 | **2/3** |
| `hallucination_69` | hallucination | A | 1/3 | 0/3 | 1/3 | 0/3 | **1/3** |
| `hallucination_73` | hallucination | B | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_75` | hallucination | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_77` | hallucination | A | 3/3 | 2/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_79` | hallucination | A | 2/3 | 2/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_81` | hallucination | A | 2/3 | 1/3 | 1/3 | 0/3 | **1/3** |
| `hallucination_83` | hallucination | B | 0/3 | 1/3 | 1/3 | 1/3 | **0/3** |
| `hallucination_87` | hallucination | B | 0/3 | 0/3 | 0/3 | 0/3 | **0/3** |
| `hallucination_89` | hallucination | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_91` | hallucination | A | 3/3 | 2/3 | 1/3 | 1/3 | **0/3** |
| `hallucination_93` | hallucination | B | 3/3 | 3/3 | 3/3 | 2/3 | **2/3** |
| `hallucination_95` | hallucination | A | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_96` | hallucination | B | 1/3 | 1/3 | 2/3 | 2/3 | **3/3** |

## Integrity and frozen identities

| Lane | Tasks | Records | info.error | p50 | p90 | Scenario SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| A | 51 | 153/153 | 0 | 5.716s | 21.194s | `9e41bf2e7c056067e08c58dfbc269c60b0518ae2118773ba8f4ad1796b5bffef` |
| B | 50 | 150/150 | 0 | 5.624s | 24.304s | `60efb023e72d619e4e6bdfaa75948ece3e0a341ef3cb85f7fb255534e273aa31` |

| Check | Result |
|---|---|
| Records/tasks | 303/303; 101/101; exactly 3 trials each |
| `info.error` | 0 |
| Episode-final counter contexts | 303/303 |
| Lane statuses | A=0, B=0 |
| Rerolls | 0 |
| Real `429/queue_exceeded` | 0 |
| Full-tree SIGSTOP/SIGCONT | not triggered |
| Measurement suffixes | `_p77_vf_A_r3`, `_p77_vf_B_r3` |

Two zero-record startup aborts preceded the measured suffix: the first correctly rejected a historical p3i56 server identity, and the second lacked venv site-packages in the launch shim. Both produced no checkpoints and are preserved under `output/runs/`; only launch plumbing changed. The measured source, flags, manifest, scenarios, temperature, prompt, and partition never changed.

The first archive pass copied both lanes into one B-named combined directory due Bash local-variable expansion. That combined archive is preserved unchanged. Clean A/B adopted archives were created by copying the already-complete, separately named checkpoint/log files; no record was edited or rerun.

- build commit: `c936feacb9012e6ff5049f9bef7f06ffc57675b2`
- adaptive-minimal SHA-256: `3dcf4303e4390f575e67309f446a87f3d572f754672c630fa5b91ba11a2bdc29`
- server SHA-256: `f4d055b322a197bacd5febc3a7bf85f58b1c5fb7fbe6fa553354496917863337`
- manifest SHA-256: `ffe6f6445df76dd52732e3dd564b40c9f9e6adbf4c30ecfe02a04b4a3658c048`
- lane A scenario SHA-256: `9e41bf2e7c056067e08c58dfbc269c60b0518ae2118773ba8f4ad1796b5bffef`
- lane B scenario SHA-256: `60efb023e72d619e4e6bdfaa75948ece3e0a341ef3cb85f7fb255534e273aa31`
- lane A adopted archive: `output/runs/p3i77_val_final_A_r3_20260719-134253_adopted`
- lane B adopted archive: `output/runs/p3i77_val_final_B_r3_20260719-134253_adopted`
- combined historical archive: `output/runs/p3i77_val_final_B_r3_20260719-134253`
- startup-abort archives: `output/runs/p3i77_val_final_startup_abort_20260719-120506`, `output/runs/p3i77_val_final_r2_startup_abort_20260719-120735`
