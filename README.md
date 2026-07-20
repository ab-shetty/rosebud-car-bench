# rosebud-car-bench — CAR-bench Track 2 submission

Team **Rosebud's** entry for the CAR-bench Challenge @ IJCAI-ECAI 2026, Track 2
(Cerebras fast-reasoning).

The agent under test is the **frozen** Cerebras-hosted `gpt-oss-120b`.

## Headline result

Measured under the official evaluation configuration (Gemini 3.5 Flash as both
the fixed user simulator and the policy judge), on a 101-task validation
split, n=3, Pass^3:

| Arm | Pass^3 | Pass^1 | p50 latency | p90 latency |
|---|---:|---:|---:|---:|
| Stock baseline | 50/101 (49.5%) | 63.0% | 2.858s | 6.982s |
| **This submission** | **68/101 (67.3%)** | **75.9%** | **3.625s** | **9.374s** |

Both rows are measured under identical official roles, so the **+17.8 point
Pass^3 gain (+18 tasks)** is a like-for-like harness effect. We buy that
accuracy with latency: the harness is **26.8% slower at p50 and 34.3% slower
at p90** than the stock baseline, the cost of prefetch reads and of sampling
two extra completions at mutation points.

Full per-task matrices and the measurement reports are in [`results/`](results/).

## What the harness does

Three load-bearing layers, each with causal evidence in `results/`:

1. **Right-sized reasoning budget.** At the stock 1,024-token cap, gpt-oss-120b
   spends the whole budget thinking and returns empty content on hard tasks,
   which the stock pipeline surfaces as an error and death-spirals on. Raising
   the initial cap and adding an escalating rescue ladder removes this failure
   class.
2. **Deterministic guard stack.** Value provenance (every numeric mutated value must
   trace to something the user said, a stored preference that was read, or a
   literal clarification answer), schema and reference preflights, loop
   breaking, policy-lint post-conditions, and a global one-ask-per-episode
   budget. Guards are pure functions over the episode transcript.
3. **Mutation-point consensus.** At any decision whose draft mutates vehicle
   state, the harness samples two additional completions and executes only an
   exact 2-of-3 `(tool, arguments)` signature majority, failing open otherwise.
   Read and respond turns cost nothing extra.

## Compute profile

Measured over 303 trials of the official-conditions run: **1.78 calls per
decision** on average. Consensus contributes 2 extra calls only at mutation
points; guard re-decisions are bounded per episode.

## Reproducing

Requires Docker and a Cerebras API key (agent) plus a Gemini API key
(evaluator's user simulator and judge).

```bash
# 1. Fetch the organizers' car-bench package (not vendored here)
git clone https://github.com/CAR-bench/car-bench.git third_party/car-bench

# 2. Provide credentials
cp .env.example .env    # fill in CEREBRAS_API_KEY and GEMINI_API_KEY

# 3. Build the agent image
docker buildx build --platform linux/amd64 \
  -f src/track_2_agent_under_test_cerebras/Dockerfile.track-2-agent-under-test-cerebras \
  -t rosebud-car-bench-agent:local --load .

# 4. Run a smoke scenario against the official evaluator image
uv run python generate_compose.py \
  --scenario scenarios/track_2_agent_under_test_cerebras/local_docker_smoke.toml
docker compose --env-file .env \
  -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml \
  up --abort-on-container-exit
```

[`scenario.toml`](scenario.toml) is the submitted scenario: the official
evaluator image, our digest-pinned GHCR agent image, and the arm's
configuration exposed entirely through environment variables. See
[`ENVIRONMENT.md`](ENVIRONMENT.md) for the variable contract.

## Repository contents

- `src/track_2_agent_under_test_cerebras/` — the harness. `adaptive_minimal.py`
  is the submitted arm; `consensus_planner.py`, `harmony_native.py`,
  `policy_rag.py`, `fewshot_rag.py`, and `reformulation_ledger.py` are
  alternative approaches that were built and measured but are disabled in the
  submitted configuration. They ship because the technical report reports them
  as negative results.
- `scenario.toml`, `scenarios/` — submission and development scenarios.
- `results/` — measurement reports with per-task matrices, including
  [`optionc_validation.md`](results/optionc_validation.md) (the GHCR image
  validation record) and the arms we measured and **rejected**.

The 4-page technical report is submitted separately and is not included here.
The evaluator is the organizers' official published image; we neither modify
nor ship one.

This repository builds on the organizers' starter kit
([CAR-bench/car-bench-ijcai](https://github.com/CAR-bench/car-bench-ijcai));
attributions for all external work appear in the technical report.
