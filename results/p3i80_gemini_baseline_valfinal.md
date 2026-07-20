# p3i80 stock baseline on val_final under official Gemini roles

## Executive result

The raw stock baseline scored **50/101 Pass^3 (49.5%)** and **191/303 Pass^1 (63.0%)** under the same Gemini 3.5 Flash user-simulator and policy-judge roles used for the p3i78 submission-arm read. Latency was **2.858s p50 / 6.982s p90**. The completed replacement run contains **303/303 records, info.error=0, no rerolls, no Cerebras 429, and no Gemini quota event**.

The like-for-like submission result is therefore:

- p3i67 submission arm under Gemini: **68/101 (67.3%)**.
- raw baseline under Gemini: **50/101 (49.5%)**.
- **Harness effect under identical official roles: +18 tasks, +17.8 percentage points Pass^3.** Pass^1 improves by 39 trials, from 191/303 to 230/303 (**+12.9pp**).

This replaces the apples-to-oranges headline comparison against the GPT-5-mini-role baseline. The submitted harness still produces a large gain under matched roles, but the correct like-for-like Pass^3 delta is **+17.8pp**, not +24.8pp.

## Three-cell comparison

| Arm and simulator/judge roles | Pass^3 | Pass^1 | Base P^3 / P^1 | Hallucination P^3 / P^1 | Disambiguation P^3 / P^1 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical raw baseline, GPT-5-mini roles (p3i37) | 43/101 (42.6%) | 182/303 (60.1%) | 19/42 / 84/126 | 18/42 / 72/126 | 6/17 / 26/51 | 3.153s | 11.156s |
| **Raw baseline, Gemini roles (p3i80)** | **50/101 (49.5%)** | **191/303 (63.0%)** | **24/42 / 85/126** | **24/42 / 90/126** | **2/17 / 16/51** | **2.858s** | **6.982s** |
| p3i67 submission arm, Gemini roles (p3i78) | 68/101 (67.3%) | 230/303 (75.9%) | 31/42 / 100/126 | 31/42 / 102/126 | 6/17 / 28/51 | 3.625s | 9.374s |

## Decomposing the old +24.8pp headline

The former comparison was `p3i67/Gemini 67.3% - baseline/GPT-5-mini 42.6% = +24.8pp`. The new matched control decomposes it exactly (subject to independent n=3 run noise):

| Component | Calculation | Pass^3 contribution |
|---|---|---:|
| Combined simulator + policy-judge role shift on the raw baseline | 50/101 - 43/101 | **+6.9pp** (+7 tasks) |
| Harness effect with both arms under Gemini roles | 68/101 - 50/101 | **+17.8pp** (+18 tasks) |
| Previously reported mixed-role delta | 68/101 - 43/101 | **+24.8pp** (+25 tasks) |

Thus about **6.9 of the 24.8 points (28%)** came from changing the simulator/judge roles on this pair of reads, while **17.8 points (72%)** remain as the like-for-like harness advantage. Because the simulator and policy evaluator were changed together, the +6.9pp role term cannot be causally divided between simulator behavior and judge behavior from this experiment alone.

The latency comparison also changes: p3i67 is **0.768s / 26.9% slower at p50** than the Gemini-role raw baseline (3.625s versus 2.858s). Pass^3 remains the primary rubric, so the accuracy gain dominates that latency cost.

## Per-type asymmetry

The role shift is strongly non-uniform:

| Family | Baseline GPT roles | Baseline Gemini roles | Combined role shift | p3i67 Gemini | Harness gain over Gemini baseline |
|---|---:|---:|---:|---:|---:|
| Base Pass^3 | 19/42 (45.2%) | 24/42 (57.1%) | +5 tasks (+11.9pp) | 31/42 (73.8%) | +7 tasks (+16.7pp) |
| Hallucination Pass^3 | 18/42 (42.9%) | 24/42 (57.1%) | +6 tasks (+14.3pp) | 31/42 (73.8%) | +7 tasks (+16.7pp) |
| Disambiguation Pass^3 | 6/17 (35.3%) | 2/17 (11.8%) | -4 tasks (-23.5pp) | 6/17 (35.3%) | +4 tasks (+23.5pp) |
| Base Pass^1 | 84/126 (66.7%) | 85/126 (67.5%) | +1 trial (+0.8pp) | 100/126 (79.4%) | +15 trials (+11.9pp) |
| Hallucination Pass^1 | 72/126 (57.1%) | 90/126 (71.4%) | +18 trials (+14.3pp) | 102/126 (81.0%) | +12 trials (+9.5pp) |
| Disambiguation Pass^1 | 26/51 (51.0%) | 16/51 (31.4%) | -10 trials (-19.6pp) | 28/51 (54.9%) | +12 trials (+23.5pp) |

Gemini materially helps the raw baseline on base and hallucination while sharply hurting disambiguation. That is consistent with the p3i78 trace finding: GPT-5-mini often leaked hidden disambiguation criteria in its opening user turn, whereas Gemini properly withholds them and requires the agent to ask or resolve the ambiguity. The matched control shows this is not merely a p3i67-specific effect: raw-baseline disambiguation falls from 6/17 to 2/17 Pass^3 and from 26/51 to 16/51 Pass^1. Conversely, p3i67 recovers the Gemini disambiguation deficit to 6/17 and 28/51, establishing a +23.5pp family-local harness gain under identical roles.

## Frozen baseline identity and effective environment

This run matched the historical 43/101 control at:

- `output/runs/p3i37_rawbaseline_val_final_A_20260718-005500`
- `output/runs/p3i37_rawbaseline_val_final_B_20260718-005500`
- Historical launch/source commit: `c547adc17cc295148a55219c0dba6bba9391602e`.

The historical environment recorded raw starter-kit baseline, `TRACK2_HARNESS` unset (server default `baseline`), provider-default temperature, server-default medium effort, chat transport, and all other `TRACK2_*` configuration removed. p3i80 used the same frozen source and defaults:

```text
TRACK2_HARNESS=UNSET (server default: baseline)
TRACK2_TRANSPORT=UNSET (server default: chat)
TRACK2_EXECUTOR_REASONING_EFFORT=UNSET (server default: medium)
TRACK2_TEMPERATURE=UNSET (provider default)
TRACK2_MAX_COMPLETION_TOKENS=UNSET (server default: 1024)
all_TRACK2_AM_flags=UNSET
AGENTBEATS_CLIENT_STREAMING=false
PYTHON_DOTENV_DISABLED=1
```

Runtime startup records independently showed `harness=baseline`, `transport=chat`, `reasoning_effort=medium`, `temperature=null`, and `max_completion_tokens=1024`, with no adaptive-minimal startup record. The evaluator startup record was checked before continuing and contained all four official-role fields on both lanes:

```text
user_model=gemini/gemini-3.5-flash
user_model_provider=gemini
policy_evaluator_model=gemini/gemini-3.5-flash
policy_evaluator_model_provider=gemini
```

Frozen identities:

| Artifact | Identity |
|---|---|
| Baseline source commit | `c547adc17cc295148a55219c0dba6bba9391602e` |
| `server.py` SHA-256 | `ea8c30067cca2e359566fcd14ecd937801754eddb21603810a914cd016a13684` |
| `car_bench_agent.py` SHA-256 | `6359db1d125cd98eed9b98ffdbf870ea22e444533143fd2dca0939cab96c404a` |
| `cerebras_client.py` SHA-256 | `1a36fc3d7bcb71563bdd7a7c617e7e5711ab9ed85757f537a398e8bb3d0a8768` |
| Lane A p3i80 scenario SHA-256 | `ff104adfb99cae1226e27acb70a610bd1d525ab3c5f16f6c6faf68d950e1860c` |
| Lane B p3i80 scenario SHA-256 | `99ce390fda110721b25a2a8f726b82d557efc5d562360cd588b957a352183415` |
| Clean-run launch commit | `fca384c1164b341353b00fba1ca0ecf5851c6abe` |

The p3i80 configs preserve the p3i78 51/50 latency-balanced task partition, `n=3`, evaluator endpoints, and all Gemini config fields; only the agent arm is changed to the frozen baseline. Their task union is exactly the same 101-task `val_final` pool.

## Run integrity and restart history

| Attempt / lane | Suffix | Records used | info.error | Exit/status | 429 | Rerolls | Disposition |
|---|---|---:|---:|---:|---:|---:|---|
| Pre-measurement wiring diagnostic | `_p80_vf_A/B` | 0 | 0 | 130 | 0 | 0 | Evaluator import failed before any checkpoint; archived and excluded |
| Externally OOM-killed attempt | `_p80_vf2_A/B` | 0 | n/a | 137 | 0 before kill | 0 | 58 diagnostic base records archived; fully discarded, never resumed/merged |
| Clean lane A | `_p80b_vf_A` | 153/153 | 0 | 0 | 0 | 0 | Included |
| Clean lane B | `_p80b_vf_B` | 150/150 | 0 | 0 | 0 | 0 | Included |
| **Clean total** | | **303/303** | **0** | **0** | **0** | **0** | **Included** |

The first valid-start attempt was killed by the host kernel OOM killer at 18:59:58 while an unrelated overseer Docker Compose validation ran on the 10 GB host. The tmux scope died with a 3.1 GB Python process. Its 58 base-only checkpoint records were copied under `output/runs/p3i80_gemini_baseline_val_final_oom_killed_20260719-190158`, then the live copies were removed. The completed run began from scratch with fresh `_p80b_vf_A/B` suffixes; no killed-run trial was reused.

Clean archives:

- `output/runs/p3i80b_gemini_baseline_val_final_A_20260719-194602`
- `output/runs/p3i80b_gemini_baseline_val_final_B_20260719-194602`

The narrow Cerebras detector required an actual JSON `status_code=429` together with `queue_exceeded` or `token_quota_exceeded`; location IDs ending in `429` could not trigger it. It fired zero times. The Gemini quota detector also fired zero times.

## Full 101-task Pass-count matrix

| Task | Baseline GPT roles (p3i37) | Baseline Gemini roles (p3i80) | p3i67 Gemini roles (p3i78) |
|---|---:|---:|---:|
| `base_3` | 3/3 | 3/3 | 3/3 |
| `base_5` | 1/3 | 1/3 | 1/3 |
| `base_7` | 3/3 | 3/3 | 3/3 |
| `base_9` | 2/3 | 3/3 | 3/3 |
| `base_11` | 1/3 | 0/3 | 3/3 |
| `base_13` | 3/3 | 3/3 | 3/3 |
| `base_17` | 2/3 | 3/3 | 3/3 |
| `base_19` | 3/3 | 3/3 | 3/3 |
| `base_21` | 3/3 | 3/3 | 3/3 |
| `base_23` | 2/3 | 3/3 | 3/3 |
| `base_25` | 3/3 | 3/3 | 3/3 |
| `base_27` | 3/3 | 3/3 | 3/3 |
| `base_31` | 2/3 | 0/3 | 2/3 |
| `base_33` | 3/3 | 3/3 | 3/3 |
| `base_35` | 3/3 | 3/3 | 3/3 |
| `base_37` | 0/3 | 3/3 | 3/3 |
| `base_39` | 1/3 | 1/3 | 2/3 |
| `base_41` | 3/3 | 3/3 | 3/3 |
| `base_45` | 0/3 | 0/3 | 0/3 |
| `base_47` | 0/3 | 3/3 | 3/3 |
| `base_49` | 3/3 | 2/3 | 0/3 |
| `base_51` | 0/3 | 3/3 | 3/3 |
| `base_53` | 0/3 | 1/3 | 3/3 |
| `base_55` | 1/3 | 0/3 | 0/3 |
| `base_59` | 2/3 | 3/3 | 3/3 |
| `base_61` | 0/3 | 3/3 | 3/3 |
| `base_63` | 3/3 | 0/3 | 3/3 |
| `base_65` | 2/3 | 3/3 | 3/3 |
| `base_67` | 3/3 | 3/3 | 3/3 |
| `base_69` | 2/3 | 3/3 | 3/3 |
| `base_73` | 3/3 | 2/3 | 3/3 |
| `base_75` | 3/3 | 1/3 | 3/3 |
| `base_77` | 2/3 | 0/3 | 0/3 |
| `base_79` | 3/3 | 3/3 | 3/3 |
| `base_81` | 2/3 | 0/3 | 0/3 |
| `base_83` | 3/3 | 2/3 | 3/3 |
| `base_87` | 2/3 | 3/3 | 3/3 |
| `base_89` | 0/3 | 0/3 | 1/3 |
| `base_91` | 3/3 | 3/3 | 3/3 |
| `base_93` | 3/3 | 1/3 | 1/3 |
| `base_95` | 2/3 | 2/3 | 0/3 |
| `base_97` | 1/3 | 0/3 | 3/3 |
| `hallucination_3` | 3/3 | 3/3 | 3/3 |
| `hallucination_5` | 3/3 | 1/3 | 2/3 |
| `hallucination_7` | 3/3 | 3/3 | 3/3 |
| `hallucination_9` | 3/3 | 3/3 | 3/3 |
| `hallucination_11` | 1/3 | 3/3 | 3/3 |
| `hallucination_13` | 1/3 | 3/3 | 2/3 |
| `hallucination_17` | 3/3 | 3/3 | 3/3 |
| `hallucination_19` | 1/3 | 3/3 | 3/3 |
| `hallucination_21` | 2/3 | 3/3 | 3/3 |
| `hallucination_23` | 0/3 | 0/3 | 0/3 |
| `hallucination_25` | 0/3 | 0/3 | 3/3 |
| `hallucination_27` | 0/3 | 3/3 | 3/3 |
| `hallucination_31` | 3/3 | 3/3 | 3/3 |
| `hallucination_33` | 2/3 | 2/3 | 3/3 |
| `hallucination_35` | 3/3 | 3/3 | 3/3 |
| `hallucination_37` | 0/3 | 0/3 | 3/3 |
| `hallucination_39` | 3/3 | 3/3 | 3/3 |
| `hallucination_41` | 0/3 | 2/3 | 3/3 |
| `hallucination_45` | 0/3 | 3/3 | 3/3 |
| `hallucination_47` | 1/3 | 3/3 | 3/3 |
| `hallucination_49` | 1/3 | 0/3 | 3/3 |
| `hallucination_51` | 0/3 | 2/3 | 3/3 |
| `hallucination_53` | 3/3 | 3/3 | 3/3 |
| `hallucination_55` | 1/3 | 1/3 | 1/3 |
| `hallucination_59` | 0/3 | 2/3 | 3/3 |
| `hallucination_61` | 0/3 | 2/3 | 3/3 |
| `hallucination_63` | 3/3 | 3/3 | 1/3 |
| `hallucination_65` | 3/3 | 3/3 | 3/3 |
| `hallucination_67` | 2/3 | 2/3 | 3/3 |
| `hallucination_69` | 1/3 | 0/3 | 0/3 |
| `hallucination_73` | 3/3 | 3/3 | 3/3 |
| `hallucination_75` | 3/3 | 3/3 | 2/3 |
| `hallucination_77` | 3/3 | 3/3 | 3/3 |
| `hallucination_79` | 2/3 | 3/3 | 3/3 |
| `hallucination_81` | 2/3 | 2/3 | 1/3 |
| `hallucination_83` | 0/3 | 0/3 | 0/3 |
| `hallucination_87` | 0/3 | 0/3 | 0/3 |
| `hallucination_89` | 3/3 | 3/3 | 3/3 |
| `hallucination_91` | 3/3 | 0/3 | 0/3 |
| `hallucination_93` | 3/3 | 2/3 | 3/3 |
| `hallucination_95` | 3/3 | 3/3 | 3/3 |
| `hallucination_96` | 1/3 | 3/3 | 3/3 |
| `disambiguation_3` | 3/3 | 2/3 | 3/3 |
| `disambiguation_5` | 3/3 | 3/3 | 1/3 |
| `disambiguation_9` | 0/3 | 1/3 | 3/3 |
| `disambiguation_11` | 2/3 | 0/3 | 1/3 |
| `disambiguation_13` | 0/3 | 0/3 | 3/3 |
| `disambiguation_17` | 3/3 | 0/3 | 1/3 |
| `disambiguation_19` | 2/3 | 1/3 | 3/3 |
| `disambiguation_23` | 3/3 | 0/3 | 0/3 |
| `disambiguation_25` | 0/3 | 0/3 | 0/3 |
| `disambiguation_27` | 0/3 | 0/3 | 2/3 |
| `disambiguation_31` | 0/3 | 0/3 | 0/3 |
| `disambiguation_33` | 1/3 | 3/3 | 3/3 |
| `disambiguation_37` | 3/3 | 2/3 | 2/3 |
| `disambiguation_39` | 1/3 | 2/3 | 3/3 |
| `disambiguation_41` | 1/3 | 2/3 | 2/3 |
| `disambiguation_45` | 3/3 | 0/3 | 0/3 |
| `disambiguation_47` | 1/3 | 0/3 | 1/3 |

## Reporting conclusion

The official-role baseline is **50/101**, not 43/101. The technical-report headline should therefore say that p3i67 improves Pass^3 from **49.5% to 67.3%, a like-for-like +17.8 percentage-point gain under Gemini 3.5 Flash simulator and judge roles**. The remaining difference between +17.8pp and the old +24.8pp headline is the combined +6.9pp simulator/judge role shift on the raw baseline. The family table must accompany that headline because Gemini improves base/hallucination baseline performance while making disambiguation substantially harder.
