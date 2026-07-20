# p3i78 official-conditions dress rehearsal

## Arm selection and frozen identity

Order 77 scored **60/101 Pass^3**, below its `>= 62/101` selection bar. This rehearsal therefore used the exact frozen **p3i67 fixed-fast** arm from commit `c0161b63ede4f3b9a7f3712d7e770f26a9b8ba56`, not the order-77 repair arm.

- adaptive-minimal SHA-256: `0122d79062bc9dacedceb8ecf5e1db28a6d6ea8b170017d3f7a461e785b07b28`
- server SHA-256: `ff7cc4d8401f647c9ee7c4ccf8bffdea6d9b4ae4f62f976b086ef09640d4383d`
- manifest SHA-256: `c64973d47c8b45effb2a5ad6e85e5ca6cb65144df8220bd5692301942040c83c`
- arm: p3i49 guard stack plus mutation consensus and prefetch, medium executor effort, initial cap 4096, plus semantic prefetch, route resolver, ask-type gate, textcall guard, and argument lint; `READ_RESOLVE` off
- user simulator: `gemini/gemini-3.5-flash` through provider `gemini`
- policy evaluator: `gemini/gemini-3.5-flash` through provider `gemini`

The evaluator startup record on both lanes explicitly logged all four Gemini role fields before scored work proceeded. The evaluator-only logging addition records role identities and does not alter evaluation behavior.

## Headline results

The official-role rehearsal improved val-final Pass^3 from **62/101 to 68/101 (+5.9 percentage points)** while leaving Pass^1 nearly unchanged at **230/303 versus 229/303**. The gain is not uniform: base and hallucination improved, while disambiguation fell sharply. On benchmark-18, the rehearsal scored **9/18**, exactly the p3i67 family's prior average of 27/54 task-draws.

| Split / role configuration | Pass^3 | Pass^1 | Base P^3 / P^1 | Hallucination P^3 / P^1 | Disambiguation P^3 / P^1 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_final, GPT-5-mini sim + judge (p3i69) | 62/101 (61.4%) | 229/303 (75.6%) | 26/42 / 95/126 | 25/42 / 97/126 | 11/17 / 37/51 | 4.336s | 13.270s |
| **val_final, Gemini sim + judge (p3i78)** | **68/101 (67.3%)** | **230/303 (75.9%)** | **31/42 / 100/126** | **31/42 / 102/126** | **6/17 / 28/51** | **3.625s** | **9.374s** |
| benchmark-18, GPT-5-mini p3i67 pool | 27/54 task-draws (50.0%) | 116/162 (71.6%) | 8/18 / 35/54 | 12/18 / 46/54 | 7/18 / 35/54 | 5.302s | 12.488s |
| **benchmark-18, Gemini sim + judge (p3i78)** | **9/18 (50.0%)** | **40/54 (74.1%)** | **3/6 / 13/18** | **5/6 / 15/18** | **1/6 / 12/18** | **4.646s** | **12.168s** |

Val-final type deltas are especially informative. Relative to the identical arm under GPT-5-mini roles, Gemini gained five base Pass^3 tasks and six hallucination Pass^3 tasks, but lost five disambiguation Pass^3 tasks. Pass^1 moved by +5 base, +5 hallucination, and -9 disambiguation trials. Thus the +6-task headline is real on this read, but it comes with a clear disambiguation-family risk rather than a uniform lift.

## Policy-judge shift

`r_policy` is not evaluated on every record, so rates use records with a non-null policy result as the denominator.

| Split | GPT-5-mini policy failures | Gemini policy failures | Delta |
|---|---:|---:|---:|
| val_final | 4/164 (2.4%) | 0/164 (0.0%) | -2.4 pp |
| benchmark-18 | 9/108 (8.3%), pooled | 4/36 (11.1%) | +2.8 pp |

The direction is mixed across splits. Because this rehearsal changed the simulator and policy judge together, `r_policy` deltas versus prior runs mix simulator-induced dialogue changes with judge-model changes and must not be attributed to either role alone.

## Simulator-facing termination and dialogue metrics

Termination/control-word and dialogue-length metrics are the cleaner view of the simulator shift. User-turn counts include the initial request and exclude the terminal `###STOP###` marker.

| Split / simulator | Hallucination error | Disambiguation error | Out of scope | Null/unscored termination | User turns mean / p50 / p90 / max |
|---|---:|---:|---:|---:|---:|
| val_final, GPT-5-mini | 10/303 (3.3%) | 2/303 (0.7%) | 1/303 (0.3%) | 20/303 (6.6%) | 2.93 / 2 / 4 / 40 |
| **val_final, Gemini** | **9/303 (3.0%)** | **2/303 (0.7%)** | **2/303 (0.7%)** | **16/303 (5.3%)** | **1.91 / 2 / 3 / 6** |
| benchmark-18, GPT-5-mini p3i67 pool | 2/162 (1.2%) | 2/162 (1.2%) | 5/162 (3.1%) | 4/162 (2.5%) | 2.98 / 2 / 3 / 40 |
| **benchmark-18, Gemini** | **2/54 (3.7%)** | **1/54 (1.9%)** | **0/54** | **0/54** | **1.89 / 2 / 3 / 4** |

Gemini produced substantially shorter dialogues and eliminated the extreme 40-turn tail in both reads. On val-final this coincided with fewer null/unscored terminations and better base/hallucination stability, but not better disambiguation: disambiguation Pass^3 fell from 11/17 to 6/17 even though the explicit `DISAMBIGUATION_ERROR` count remained 2/303. This indicates that most of the disambiguation damage is in task resolution or action choice, not merely the terminal control word.

There were no evaluator/API errors and no distinct malformed-simulator failure class. Behavioral confabulation or malformed-ask outcomes that reached evaluation are reflected in the hallucination/disambiguation control-word counts above.

## Predeclared benchmark-18 killer tasks

Only the three user-specified task outcomes are reported here.

| Task | Gemini pass count |
|---|---:|
| `disambiguation_51` | 2/3 |
| `base_98` | 1/3 |
| `hallucination_78` | 0/3 |

The official simulator did not cure this set: the three tasks total 3/9 successful trials, and `hallucination_78` remains a hard zero.

## Integrity and run history

| Run | Lane A | Lane B | Combined | `info.error` | Archive timestamp |
|---|---:|---:|---:|---:|---|
| val_final | 153/153 | 150/150 | 303/303 | 0 | `20260719-152704` |
| benchmark-18 | 27/27, status 0 | 27/27, status 0 | 54/54 | 0 | `20260719-153458` |

Val-final used the unchanged p3i64/p3i69 latency-balanced 51/50 task partition. Benchmark-18 used the p3i57 latency-balanced 9/9 partition. New official-role scenario identities:

| Scenario | SHA-256 |
|---|---|
| val_final lane A | `77561c49e6d8a95da3bbc96b2a9a25f0e7a183ff2114c588da2c5d7bdd9cc644` |
| val_final lane B | `e46e81a9f859ce6f443484b7dcd05d5b8121b57f0cd5b6645a3a9a99d272bb48` |
| benchmark-18 lane A | `41a6fea52fb46d8f5a2e1c23445a56c0db51688f0547705285e3833982dc3f27` |
| benchmark-18 lane B | `7ea265e2177a05064202b91bfa94daa1bfe0f0d26103688e88db6601fe5b5cb5` |

The first val-final monitor pause was an operational false positive: a location identifier ending in `429` and a normal Cerebras `rate_limit_headers` field matched an initially over-broad detector. Both full process trees were SIGSTOP-preserved and resumed in place; no checkpoint was removed and no trial was rerolled. At 17 clean records per lane, lane A then encountered a real **Cerebras** `token_quota_exceeded` 429 with a 59-second retry instruction. The normal full-tree serial fallback kept lane B stopped, finished lane A alone, and then resumed lane B alone. Gemini emitted no quota/rate-limit error. Val-final therefore ran 17 records per lane concurrently, followed by 136 lane-A records and 133 lane-B records serially. Benchmark-18 completed concurrently without a 429.

Archives:

- `output/runs/p3i78_gemini_val_final_A_20260719-152704`
- `output/runs/p3i78_gemini_val_final_B_20260719-152704`
- `output/runs/p3i78_gemini_benchmark18_A_20260719-153458`
- `output/runs/p3i78_gemini_benchmark18_B_20260719-153458`

## Submission-impact verdict

**The official simulator moves this arm materially on val-final, positively overall but negatively for disambiguation.** The frozen p3i67 arm rises from 62/101 to 68/101 Pass^3, improves p50 by 16.4% and p90 by 29.4%, and holds benchmark-18 at its established 50% Pass^3 family level. That supports keeping p3i67 as the submission arm under the stated val-like hidden-test assumption.

The caveat is important: Gemini changes the family profile. Base and hallucination become stronger, while val-final disambiguation falls by 29.4 percentage points (11/17 to 6/17), and the benchmark read has only 1/6 disambiguation Pass^3. Submission expectations should therefore use the **68/101 aggregate with a specific disambiguation-risk note**, not assume that the simulator swap is a uniform improvement. The policy comparison is not separately causal because both official roles changed together.
