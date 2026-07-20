# p3i69 val-final fixed-fast confirm

## Result

The exact frozen p3i67 fixed-fast arm completed **303/303** val-final trials with **0 `info.error`**. It scored **62/101 Pass^3 (61.4%)** and **229/303 Pass^1 (75.6%)**, with **4.335s p50 / 13.270s p90** latency.

Relative to raw baseline, the fixed arm is **+18.8 percentage points** on Pass^3. Relative to p3i56, it is **+3 Pass^3 tasks**.

| Arm | Pass^3 | Pass^1 | p50 | p90 |
|---|---:|---:|---:|---:|
| Raw baseline | 43/101 (42.6%) | 182/303 (60.1%) | 3.153s | 11.156s |
| Wave-5 reference | 57/101 (56.4%) | 215/303 (71.0%) | 3.535s | 22.367s |
| p3i51 deep-deliberate | 55/101 (54.5%) | 214/303 (70.6%) | 10.744s | 40.746s |
| p3i56 fast | 59/101 (58.4%) | 216/303 (71.3%) | 4.182s | 11.699s |
| **p3i67 fixed fast** | **62/101 (61.4%)** | **229/303 (75.6%)** | **4.335s** | **13.270s** |

## Per-type results

| Type | Pass^3 | Pass^1 | p50 | p90 |
|---|---:|---:|---:|---:|
| base | 26/42 (61.9%) | 95/126 (75.4%) | 4.649s | 13.566s |
| hallucination | 25/42 (59.5%) | 97/126 (77.0%) | 4.092s | 18.843s |
| disambiguation | 11/17 (64.7%) | 37/51 (72.5%) | 3.837s | 11.389s |

## Projected deterministic recoveries

These are measurement-only pass counts for the three p3i56 0/3 losses named before the run; there is no trace autopsy or post-hoc mechanism change.

| Projected recovery | p3i56 | p3i67 fixed fast |
|---|---:|---:|
| `base_19` | 0/3 | 3/3 |
| `base_21` | 0/3 | 3/3 |
| `disambiguation_37` | 0/3 | 1/3 |

## Episode-final mechanism counters

Counters are aggregate structured-log totals across all 303 final episode contexts. They were not keyed to outcomes.

| Mechanism | Episode-final cumulative count |
|---|---:|
| Prefetch reads | 3630 |
| Semantic-prefetch valid calls emitted | 3630 |
| Semantic-prefetch invalid calls suppressed | 615 |
| Prefetch error results dropped | 0 |
| Route-resolver fires | 0 |
| Route-resolver blocked reads | 0 |
| Ask-type-gate suppressions | 0 |
| Textcall-guard fires | 6 |
| Textcall direct executes | 1 |
| Textcall re-decisions | 5 |
| Argument-lint fires | 10 |
| Argument-lint payload bounces | 0 |
| Argument-lint disclosure revises | 10 |
| Read-resolve redirects (must be zero) | 0 |
| Truncation rescues | 25 |
| Malformed-argument rescues | 24 |
| Placeholder-guard fires | 6 |
| Schema-preflight bounces | 4 |
| P1 context-value applies | 3 |
| P1 ask suppressions | 0 |
| P2 binding bounces | 8 |
| P3 preference reads | 26 |
| P3 fallback asks | 22 |
| P4 navigation redirects | 36 |
| P5 occupancy reads | 0 |
| P6 claim revises | 26 |
| 24-hour-format revises | 2 |
| Injected asks | 22 |
| Ask-budget suppressions | 14 |
| Repeated-read blocks | 57 |
| Route-reference bounces | 15 |
| Policy-lint revises | 3 |
| Mutation-consensus invocations | 356 |
| Consensus agreements | 268 |
| Consensus overrides | 56 |
| Consensus no-majority fallbacks | 88 |
| Consensus extra LLM calls | 723 |

Transport bookkeeping: 2973 LLM requests, 2969 responses, 0 real `queue_exceeded` events, and 4 other SDK-error events.

## 101-task pass-count matrix

| Task | Type | Lane | Baseline | p3i51 | p3i56 | p3i67 fixed fast |
|---|---|:---:|---:|---:|---:|---:|
| `base_3` | base | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_5` | base | A | 1/3 | 0/3 | 2/3 | **2/3** |
| `base_7` | base | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_9` | base | B | 2/3 | 3/3 | 3/3 | **3/3** |
| `base_11` | base | A | 1/3 | 3/3 | 2/3 | **3/3** |
| `base_13` | base | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_17` | base | A | 2/3 | 3/3 | 2/3 | **3/3** |
| `base_19` | base | A | 3/3 | 3/3 | 0/3 | **3/3** |
| `base_21` | base | A | 3/3 | 3/3 | 0/3 | **3/3** |
| `base_23` | base | B | 2/3 | 1/3 | 0/3 | **1/3** |
| `base_25` | base | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_27` | base | B | 3/3 | 2/3 | 3/3 | **3/3** |
| `base_31` | base | B | 2/3 | 3/3 | 3/3 | **3/3** |
| `base_33` | base | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_35` | base | B | 3/3 | 2/3 | 3/3 | **3/3** |
| `base_37` | base | A | 0/3 | 2/3 | 3/3 | **2/3** |
| `base_39` | base | A | 1/3 | 0/3 | 1/3 | **0/3** |
| `base_41` | base | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_45` | base | A | 0/3 | 0/3 | 3/3 | **1/3** |
| `base_47` | base | B | 0/3 | 0/3 | 1/3 | **3/3** |
| `base_49` | base | B | 3/3 | 0/3 | 0/3 | **0/3** |
| `base_51` | base | A | 0/3 | 3/3 | 2/3 | **1/3** |
| `base_53` | base | B | 0/3 | 0/3 | 2/3 | **2/3** |
| `base_55` | base | A | 1/3 | 3/3 | 2/3 | **2/3** |
| `base_59` | base | A | 2/3 | 1/3 | 0/3 | **1/3** |
| `base_61` | base | B | 0/3 | 1/3 | 1/3 | **3/3** |
| `base_63` | base | B | 3/3 | 3/3 | 3/3 | **2/3** |
| `base_65` | base | A | 2/3 | 1/3 | 3/3 | **3/3** |
| `base_67` | base | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_69` | base | B | 2/3 | 3/3 | 3/3 | **3/3** |
| `base_73` | base | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_75` | base | A | 3/3 | 2/3 | 0/3 | **3/3** |
| `base_77` | base | A | 2/3 | 0/3 | 0/3 | **0/3** |
| `base_79` | base | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_81` | base | B | 2/3 | 3/3 | 3/3 | **3/3** |
| `base_83` | base | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_87` | base | B | 2/3 | 3/3 | 3/3 | **3/3** |
| `base_89` | base | B | 0/3 | 0/3 | 0/3 | **0/3** |
| `base_91` | base | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `base_93` | base | B | 3/3 | 0/3 | 0/3 | **1/3** |
| `base_95` | base | B | 2/3 | 2/3 | 1/3 | **0/3** |
| `base_97` | base | A | 1/3 | 2/3 | 2/3 | **2/3** |
| `disambiguation_3` | disambiguation | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_5` | disambiguation | A | 3/3 | 2/3 | 3/3 | **3/3** |
| `disambiguation_9` | disambiguation | B | 0/3 | 2/3 | 3/3 | **3/3** |
| `disambiguation_11` | disambiguation | A | 2/3 | 3/3 | 2/3 | **3/3** |
| `disambiguation_13` | disambiguation | B | 0/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_17` | disambiguation | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_19` | disambiguation | A | 2/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_23` | disambiguation | B | 3/3 | 0/3 | 0/3 | **0/3** |
| `disambiguation_25` | disambiguation | A | 0/3 | 0/3 | 0/3 | **0/3** |
| `disambiguation_27` | disambiguation | A | 0/3 | 3/3 | 2/3 | **3/3** |
| `disambiguation_31` | disambiguation | A | 0/3 | 0/3 | 0/3 | **0/3** |
| `disambiguation_33` | disambiguation | A | 1/3 | 3/3 | 3/3 | **2/3** |
| `disambiguation_37` | disambiguation | B | 3/3 | 2/3 | 0/3 | **1/3** |
| `disambiguation_39` | disambiguation | A | 1/3 | 3/3 | 2/3 | **3/3** |
| `disambiguation_41` | disambiguation | B | 1/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_45` | disambiguation | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `disambiguation_47` | disambiguation | B | 1/3 | 2/3 | 1/3 | **1/3** |
| `hallucination_3` | hallucination | A | 3/3 | 2/3 | 3/3 | **3/3** |
| `hallucination_5` | hallucination | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_7` | hallucination | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_9` | hallucination | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_11` | hallucination | A | 1/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_13` | hallucination | A | 1/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_17` | hallucination | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_19` | hallucination | A | 1/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_21` | hallucination | A | 2/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_23` | hallucination | B | 0/3 | 0/3 | 0/3 | **0/3** |
| `hallucination_25` | hallucination | B | 0/3 | 0/3 | 1/3 | **1/3** |
| `hallucination_27` | hallucination | B | 0/3 | 3/3 | 1/3 | **2/3** |
| `hallucination_31` | hallucination | B | 3/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_33` | hallucination | B | 2/3 | 1/3 | 3/3 | **2/3** |
| `hallucination_35` | hallucination | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_37` | hallucination | A | 0/3 | 1/3 | 3/3 | **3/3** |
| `hallucination_39` | hallucination | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_41` | hallucination | B | 0/3 | 2/3 | 3/3 | **3/3** |
| `hallucination_45` | hallucination | A | 0/3 | 2/3 | 0/3 | **3/3** |
| `hallucination_47` | hallucination | B | 1/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_49` | hallucination | B | 1/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_51` | hallucination | B | 0/3 | 2/3 | 1/3 | **3/3** |
| `hallucination_53` | hallucination | A | 3/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_55` | hallucination | A | 1/3 | 0/3 | 1/3 | **2/3** |
| `hallucination_59` | hallucination | B | 0/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_61` | hallucination | A | 0/3 | 2/3 | 3/3 | **3/3** |
| `hallucination_63` | hallucination | B | 3/3 | 2/3 | 2/3 | **1/3** |
| `hallucination_65` | hallucination | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_67` | hallucination | B | 2/3 | 2/3 | 2/3 | **3/3** |
| `hallucination_69` | hallucination | A | 1/3 | 0/3 | 1/3 | **0/3** |
| `hallucination_73` | hallucination | B | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_75` | hallucination | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_77` | hallucination | A | 3/3 | 2/3 | 3/3 | **3/3** |
| `hallucination_79` | hallucination | A | 2/3 | 2/3 | 3/3 | **3/3** |
| `hallucination_81` | hallucination | A | 2/3 | 1/3 | 1/3 | **0/3** |
| `hallucination_83` | hallucination | B | 0/3 | 1/3 | 1/3 | **1/3** |
| `hallucination_87` | hallucination | B | 0/3 | 0/3 | 0/3 | **0/3** |
| `hallucination_89` | hallucination | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_91` | hallucination | A | 3/3 | 2/3 | 1/3 | **1/3** |
| `hallucination_93` | hallucination | B | 3/3 | 3/3 | 3/3 | **2/3** |
| `hallucination_95` | hallucination | A | 3/3 | 3/3 | 3/3 | **3/3** |
| `hallucination_96` | hallucination | B | 1/3 | 1/3 | 2/3 | **2/3** |

## Integrity and frozen-arm verification

| Lane | Tasks | Records | info.error | p50 | p90 | Scenario SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| A | 51 | 153/153 | 0 | 4.205s | 14.923s | `9e41bf2e7c056067e08c58dfbc269c60b0518ae2118773ba8f4ad1796b5bffef` |
| B | 50 | 150/150 | 0 | 4.426s | 12.347s | `60efb023e72d619e4e6bdfaa75948ece3e0a341ef3cb85f7fb255534e273aa31` |

| Check | Result |
|---|---|
| Expected/completed records | 303/303 |
| Expected/completed tasks | 101/101, exactly 3 trials each |
| `info.error` | 0 |
| Episode-final counter contexts | 303/303 |
| Lane statuses | A=0, B=0 |
| Rerolls | 0 |
| Real `429/queue_exceeded` | 0 |
| Full-tree SIGSTOP/SIGCONT fallback | not triggered |
| Measurement policy | val traces open; no autopsy performed |

The measured invocation used fresh `_r2` suffixes after two pre-trial launcher aborts, both with zero records and no checkpoints. The first assertion correctly detected that the unchanged p3i64 scenario's absolute agent path had selected historical p3i56 rather than p3i67. The second routing attempt invoked the shim-directory Python but lacked venv site-packages and exited before either server started. Both aborts are archived with status 97; the only corrections were runner routing/import plumbing. The measured arm, flags, temperature, prompt, partition, and scenario bytes never changed.

The p3i64 scenario files are byte-identical to their p3i56 run. A narrow Python shim redirects only their historical absolute agent-server path to the frozen p3i67 worktree; the evaluator and every other Python command pass through unchanged.

- frozen source commit: `c0161b63ede4f3b9a7f3712d7e770f26a9b8ba56`
- adaptive-minimal SHA-256: `0122d79062bc9dacedceb8ecf5e1db28a6d6ea8b170017d3f7a461e785b07b28`
- server SHA-256: `ff7cc4d8401f647c9ee7c4ccf8bffdea6d9b4ae4f62f976b086ef09640d4383d`
- manifest SHA-256: `c64973d47c8b45effb2a5ad6e85e5ca6cb65144df8220bd5692301942040c83c`
- lane A scenario SHA-256: `9e41bf2e7c056067e08c58dfbc269c60b0518ae2118773ba8f4ad1796b5bffef`
- lane B scenario SHA-256: `60efb023e72d619e4e6bdfaa75948ece3e0a341ef3cb85f7fb255534e273aa31`
- lane A measured archive: `output/runs/p3i69_p67_val_final_A_r2_20260719-064114`
- lane B measured archive: `output/runs/p3i69_p67_val_final_B_r2_20260719-064114`
- pre-trial abort archives: `output/runs/p3i69_p67_val_final_A_startup_abort_20260719-051936`, matching lane B, and `output/runs/p3i69_p67_val_final_A_r1_startup_abort_20260719-052251`, matching lane B

No benchmark-18 artifact was accessed in this order.
