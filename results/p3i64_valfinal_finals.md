# p3i64 val-final final-arm program

## Result and submission recommendation

The sequential two-arm program completed **606/606 records with `info.error=0`**. The fast p3i56 arm wins val-final accuracy: **59/101 Pass^3** versus **55/101** for p3i51, a **4-task / 4.0 percentage-point advantage**. It also wins latency by a wide margin: **4.182s p50** versus **10.744s** (61.1% lower), and **11.699s p90** versus **40.746s** (71.3% lower).

Under the stated rubric—Pass^3 first, latency second—the implied submission recommendation is **p3i56 (p3i49 mechanisms + prefetch, medium effort)**. It leads on the primary metric and is substantially faster. Relative to the raw baseline, p3i56 gains **+15.8 pp Pass^3**; p3i51 gains **+11.9 pp**.

| Arm | Pass^3 | Pass^1 | Baseline delta | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| Raw baseline | 43/101 (42.6%) | 182/303 (60.1%) | — | 3.153s | 11.156s |
| Wave-5 reference | 57/101 (56.4%) | 215/303 (71.0%) | +13.9 pp | 3.535s | 22.367s |
| **p3i51 deep-deliberate** | **55/101 (54.5%)** | **214/303 (70.6%)** | **+11.9 pp** | **10.744s** | **40.746s** |
| **p3i56 fast + prefetch** | **59/101 (58.4%)** | **216/303 (71.3%)** | **+15.8 pp** | **4.182s** | **11.699s** |

## Per-type results

| Arm / type | Pass^3 | Pass^1 | p50 | p90 |
|---|---:|---:|---:|---:|
| p3i51 deep-deliberate / base | 23/42 (54.8%) | 85/126 (67.5%) | 11.281s | 33.511s |
| p3i51 deep-deliberate / hallucination | 22/42 (52.4%) | 91/126 (72.2%) | 11.188s | 79.948s |
| p3i51 deep-deliberate / disambiguation | 10/17 (58.8%) | 38/51 (74.5%) | 8.384s | 23.847s |
| p3i56 fast + prefetch / base | 22/42 (52.4%) | 84/126 (66.7%) | 4.709s | 10.074s |
| p3i56 fast + prefetch / hallucination | 28/42 (66.7%) | 98/126 (77.8%) | 3.724s | 17.782s |
| p3i56 fast + prefetch / disambiguation | 9/17 (52.9%) | 34/51 (66.7%) | 3.778s | 9.738s |

p3i56's head-to-head advantage is concentrated in hallucination (+6 Pass^3 tasks and +7 Pass^1 trials versus p3i51). p3i51 leads base by one Pass^3 task and one Pass^1 trial, and disambiguation by one Pass^3 task and four Pass^1 trials. The aggregate p3i56 advantage is therefore +4 Pass^3 and +2 Pass^1.

## Mechanism counters

Counters are aggregate structured-log totals across 303 episode contexts per arm. They are not keyed to outcomes or used for tuning.

| Aggregate counter | p3i51 | p3i56 |
|---|---:|---:|
| Consensus agreements | 265 | 261 |
| Consensus extra calls | 927 | 732 |
| Consensus invocations | 375 | 359 |
| Consensus no-majority | 110 | 98 |
| Consensus overrides | 51 | 49 |
| Episode contexts | 303 | 303 |
| LLM requests | 3923 | 2958 |
| LLM responses | 3923 | 2957 |
| Malformed-argument rescues | 3 | 27 |
| Prefetch episodes | 0 | 303 |
| Prefetch reads | 0 | 4245 |
| Truncation rescues | 525 | 29 |
| 24-hour revises | 2 | 7 |
| Ask-budget suppressions | 14 | 20 |
| Consensus added latency ms | 2053619 | 652693 |
| Exact confirmation re-asks | 27 | 31 |
| Injected asks | 20 | 22 |
| L1 repeated-read blocks | 176 | 105 |
| L2 route-reference bounces | 16 | 17 |
| L3 policy-lint revises | 16 | 10 |
| Other transport errors | 0 | 1 |
| P1 known-value applies | 38 | 3 |
| P2 binding bounces | 7 | 5 |
| P3 fallback asks | 20 | 22 |
| P3 preference reads | 24 | 25 |
| P4 navigation redirects | 67 | 4 |
| P4 repeat blocks | 58 | 0 |
| P5 occupancy reads | 76 | 0 |
| P6 claim revises | 40 | 34 |
| Placeholder guards | 0 | 5 |
| Route-lineage corrections | 584 | 549 |
| Schema-preflight bounces | 10 | 11 |
| Unavailable acknowledgments | 242 | 313 |

Prefetch composed cleanly: p3i56 emitted its deterministic start-of-episode reads in every episode, while the p3i51 arm had prefetch disabled. Both arms retained mutation consensus and the frozen guard stack.

## Latency-balanced partition

The 101-task val-final pool was partitioned 51/50 by snake-drafting within type on six-record per-task latency medians from the two existing p3i37 val-final arms. Only operational task identity/type and `total_llm_induced_latency_ms` were read for balancing. The split preserves A = 21 base + 21 hallucination + 9 disambiguation and B = 21 + 21 + 8, with projected median-latency sums **292.324s A** and **280.724s B**. Both arms used the identical partition.

| Arm / lane | Tasks | Records | Scenario SHA-256 | Observed p50 | Observed p90 |
|---|---:|---:|---|---:|---:|
| p3i51 A | 51 | 153/153 | `b65a923ea392361efff2e629a0cd39c2539940ca113fb0efbe8c3b28df0f5da1` | 11.563s | 74.870s |
| p3i51 B | 50 | 150/150 | `c65dfa799787a3d1ed7a60a37060d4f14bbfdcfafd10a096a3278bde08796621` | 9.996s | 31.735s |
| p3i56 A | 51 | 153/153 | `9e41bf2e7c056067e08c58dfbc269c60b0518ae2118773ba8f4ad1796b5bffef` | 3.856s | 11.299s |
| p3i56 B | 50 | 150/150 | `60efb023e72d619e4e6bdfaa75948ece3e0a341ef3cb85f7fb255534e273aa31` | 4.601s | 14.700s |

## 101-task pass-count matrix

| Task | Type | Lane | Baseline | p3i51 | p3i56 |
|---|---|:---:|---:|---:|---:|
| `base_3` | base | B | 3/3 | 3/3 | 3/3 |
| `base_5` | base | A | 1/3 | 0/3 | 2/3 |
| `base_7` | base | A | 3/3 | 3/3 | 3/3 |
| `base_9` | base | B | 2/3 | 3/3 | 3/3 |
| `base_11` | base | A | 1/3 | 3/3 | 2/3 |
| `base_13` | base | B | 3/3 | 3/3 | 3/3 |
| `base_17` | base | A | 2/3 | 3/3 | 2/3 |
| `base_19` | base | A | 3/3 | 3/3 | 0/3 |
| `base_21` | base | A | 3/3 | 3/3 | 0/3 |
| `base_23` | base | B | 2/3 | 1/3 | 0/3 |
| `base_25` | base | B | 3/3 | 3/3 | 3/3 |
| `base_27` | base | B | 3/3 | 2/3 | 3/3 |
| `base_31` | base | B | 2/3 | 3/3 | 3/3 |
| `base_33` | base | B | 3/3 | 3/3 | 3/3 |
| `base_35` | base | B | 3/3 | 2/3 | 3/3 |
| `base_37` | base | A | 0/3 | 2/3 | 3/3 |
| `base_39` | base | A | 1/3 | 0/3 | 1/3 |
| `base_41` | base | A | 3/3 | 3/3 | 3/3 |
| `base_45` | base | A | 0/3 | 0/3 | 3/3 |
| `base_47` | base | B | 0/3 | 0/3 | 1/3 |
| `base_49` | base | B | 3/3 | 0/3 | 0/3 |
| `base_51` | base | A | 0/3 | 3/3 | 2/3 |
| `base_53` | base | B | 0/3 | 0/3 | 2/3 |
| `base_55` | base | A | 1/3 | 3/3 | 2/3 |
| `base_59` | base | A | 2/3 | 1/3 | 0/3 |
| `base_61` | base | B | 0/3 | 1/3 | 1/3 |
| `base_63` | base | B | 3/3 | 3/3 | 3/3 |
| `base_65` | base | A | 2/3 | 1/3 | 3/3 |
| `base_67` | base | A | 3/3 | 3/3 | 3/3 |
| `base_69` | base | B | 2/3 | 3/3 | 3/3 |
| `base_73` | base | A | 3/3 | 3/3 | 3/3 |
| `base_75` | base | A | 3/3 | 2/3 | 0/3 |
| `base_77` | base | A | 2/3 | 0/3 | 0/3 |
| `base_79` | base | A | 3/3 | 3/3 | 3/3 |
| `base_81` | base | B | 2/3 | 3/3 | 3/3 |
| `base_83` | base | A | 3/3 | 3/3 | 3/3 |
| `base_87` | base | B | 2/3 | 3/3 | 3/3 |
| `base_89` | base | B | 0/3 | 0/3 | 0/3 |
| `base_91` | base | B | 3/3 | 3/3 | 3/3 |
| `base_93` | base | B | 3/3 | 0/3 | 0/3 |
| `base_95` | base | B | 2/3 | 2/3 | 1/3 |
| `base_97` | base | A | 1/3 | 2/3 | 2/3 |
| `disambiguation_3` | disambiguation | A | 3/3 | 3/3 | 3/3 |
| `disambiguation_5` | disambiguation | A | 3/3 | 2/3 | 3/3 |
| `disambiguation_9` | disambiguation | B | 0/3 | 2/3 | 3/3 |
| `disambiguation_11` | disambiguation | A | 2/3 | 3/3 | 2/3 |
| `disambiguation_13` | disambiguation | B | 0/3 | 3/3 | 3/3 |
| `disambiguation_17` | disambiguation | B | 3/3 | 3/3 | 3/3 |
| `disambiguation_19` | disambiguation | A | 2/3 | 3/3 | 3/3 |
| `disambiguation_23` | disambiguation | B | 3/3 | 0/3 | 0/3 |
| `disambiguation_25` | disambiguation | A | 0/3 | 0/3 | 0/3 |
| `disambiguation_27` | disambiguation | A | 0/3 | 3/3 | 2/3 |
| `disambiguation_31` | disambiguation | A | 0/3 | 0/3 | 0/3 |
| `disambiguation_33` | disambiguation | A | 1/3 | 3/3 | 3/3 |
| `disambiguation_37` | disambiguation | B | 3/3 | 2/3 | 0/3 |
| `disambiguation_39` | disambiguation | A | 1/3 | 3/3 | 2/3 |
| `disambiguation_41` | disambiguation | B | 1/3 | 3/3 | 3/3 |
| `disambiguation_45` | disambiguation | B | 3/3 | 3/3 | 3/3 |
| `disambiguation_47` | disambiguation | B | 1/3 | 2/3 | 1/3 |
| `hallucination_3` | hallucination | A | 3/3 | 2/3 | 3/3 |
| `hallucination_5` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_7` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_9` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_11` | hallucination | A | 1/3 | 3/3 | 3/3 |
| `hallucination_13` | hallucination | A | 1/3 | 3/3 | 3/3 |
| `hallucination_17` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_19` | hallucination | A | 1/3 | 3/3 | 3/3 |
| `hallucination_21` | hallucination | A | 2/3 | 3/3 | 3/3 |
| `hallucination_23` | hallucination | B | 0/3 | 0/3 | 0/3 |
| `hallucination_25` | hallucination | B | 0/3 | 0/3 | 1/3 |
| `hallucination_27` | hallucination | B | 0/3 | 3/3 | 1/3 |
| `hallucination_31` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_33` | hallucination | B | 2/3 | 1/3 | 3/3 |
| `hallucination_35` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_37` | hallucination | A | 0/3 | 1/3 | 3/3 |
| `hallucination_39` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_41` | hallucination | B | 0/3 | 2/3 | 3/3 |
| `hallucination_45` | hallucination | A | 0/3 | 2/3 | 0/3 |
| `hallucination_47` | hallucination | B | 1/3 | 3/3 | 3/3 |
| `hallucination_49` | hallucination | B | 1/3 | 3/3 | 3/3 |
| `hallucination_51` | hallucination | B | 0/3 | 2/3 | 1/3 |
| `hallucination_53` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_55` | hallucination | A | 1/3 | 0/3 | 1/3 |
| `hallucination_59` | hallucination | B | 0/3 | 3/3 | 3/3 |
| `hallucination_61` | hallucination | A | 0/3 | 2/3 | 3/3 |
| `hallucination_63` | hallucination | B | 3/3 | 2/3 | 2/3 |
| `hallucination_65` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_67` | hallucination | B | 2/3 | 2/3 | 2/3 |
| `hallucination_69` | hallucination | A | 1/3 | 0/3 | 1/3 |
| `hallucination_73` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_75` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_77` | hallucination | A | 3/3 | 2/3 | 3/3 |
| `hallucination_79` | hallucination | A | 2/3 | 2/3 | 3/3 |
| `hallucination_81` | hallucination | A | 2/3 | 1/3 | 1/3 |
| `hallucination_83` | hallucination | B | 0/3 | 1/3 | 1/3 |
| `hallucination_87` | hallucination | B | 0/3 | 0/3 | 0/3 |
| `hallucination_89` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_91` | hallucination | A | 3/3 | 2/3 | 1/3 |
| `hallucination_93` | hallucination | B | 3/3 | 3/3 | 3/3 |
| `hallucination_95` | hallucination | A | 3/3 | 3/3 | 3/3 |
| `hallucination_96` | hallucination | B | 1/3 | 1/3 | 2/3 |

## Integrity and frozen-arm verification

| Check | p3i51 | p3i56 |
|---|---:|---:|
| Expected/completed records | 303/303 | 303/303 |
| Expected/completed tasks | 101/101 | 101/101 |
| Trial balance | every task exactly 3 | every task exactly 3 |
| `info.error` | 0 | 0 |
| Lane statuses | A=0, B=0 | A=0, B=0 |
| Real `429/queue_exceeded` | 0 | 0 |
| Other transport-error events | 0 | 1, recovered in place |
| SIGSTOP/SIGCONT fallback | 0 | 0 |
| Resume suffixes | 0 | 0 |
| Rerolls | 0 | 0 |
| Measured wall time | 7,548s | 5,103s |

p3i51 used mechanism source `3be2fe25ea0cf93153324b359a5764204cf10577`, adaptive SHA-256 `2b92223b0f0a46aa99dadce7c7b716f6c71b0c596377184bdddd30019a67e7dc`, exact high effort, initial cap 4096, consensus, no prefetch, and none of the p3i52–54/p3i60–63 flags. p3i56 used source `53fb8929e33a5576574bc8ca8f91d14414fcc720`, adaptive SHA-256 `db6df0a56ad7b3ab4d29beac714f81e4ce880883211d23674ebd70de5c01adea`, medium effort, initial cap 4096, consensus, and prefetch, with the same later flags excluded.

The first p3i56 launcher attempt stopped before any trial or checkpoint because the older frozen source omitted a disabled startup field that the new runner initially expected as explicit `false`. Its logs were preserved in `output/runs/p3i64_p56_val_final_A_startup_abort_20260719-012650` and lane B's matching archive with status 97. Only the assertion was corrected; the frozen arm, scenarios, suffixes, and mechanisms were unchanged. This was not a reroll. The measured invocation then completed exactly once.

The measured p3i56 log contains one non-429 Cerebras SDK error. The runner retained the same episode in place and completed 303/303 with zero `info.error`; no record, suffix, or lane was restarted.

Final archives:

- `output/runs/p3i64_p51_val_final_A_20260719-012035`
- `output/runs/p3i64_p51_val_final_B_20260719-012035`
- `output/runs/p3i64_p56_val_final_A_20260719-025225`
- `output/runs/p3i64_p56_val_final_B_20260719-025225`

Val traces were user-opened, but this order performed no trace autopsy. The report reads checkpoint task IDs/types, trials, top-level rewards, `info.error`, and latency plus aggregate structured-log counters and startup/coordination records.
