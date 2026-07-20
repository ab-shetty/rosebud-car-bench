# README option C — GHCR image validation

Run 2026-07-20 01:06–01:08 UTC, `scenarios/track_2_agent_under_test_cerebras/ghcr_validation.toml`.

## Configuration validated

| Component | Value |
|---|---|
| Evaluator | `ghcr.io/car-bench/car-bench-evaluator:latest` (official, unmodified) |
| Agent under test | `ghcr.io/ab-shetty/rosebud-car-bench-agent@sha256:faaf0857d6a5510b4bd89061c36dfe8e3b39ef29be98742eae39b681d6099536` (digest-pinned, pulled from GHCR) |
| Split | `train`, 1 base + 1 hallucination + 1 disambiguation, 1 trial |

The agent image is the **exact digest** named in the submission scenario; this
validates the published artifact, not a local build.

## Result

`docker compose ... --abort-on-container-exit` exited **0**.

- Tasks 3, Pass^1 **66.7%** (base 1/1 pass, hallucination 1/1 pass,
  disambiguation 0/1), wall time 90.6s.
- Results file:
  `output/track_2_agent_under_test_cerebras/20260720-010830__..._ghcr_validation__train-trials1-base1-hall1-dis1__gpt-oss-120b__medium.json`

## Submission-checkbox evidence

| Claim | Evidence |
|---|---|
| Image is `linux/amd64` | Single-platform manifest (`application/vnd.docker.distribution.manifest.v2+json`), built `--platform linux/amd64` |
| Runs with README option C | Compose run above, exit 0, against the official evaluator image |
| Emits `turn_metrics` token fields | `prompt_tokens`, `completion_tokens`, `thinking_tokens` each appear 74x in the results file (sample turn: prompt 16663, completion 576, thinking 509) |
| Submitted arm actually runs | Container logs show `Adaptive-minimal startup checks passed` and 13x `Adaptive-minimal decision`; compose-resolved env shows `TRACK2_HARNESS=adaptive_minimal`, `TRACK2_AM_MUTATION_CONSENSUS=true`, `TRACK2_AM_PREFETCH=true` |
| Guard stack is live, not inert | Per-turn `num_llm_calls` distribution 1/2/3/4 (18 turns at 4 calls) — prefetch, guard re-decisions and mutation consensus all consume extra calls; a bare baseline emits exactly 1 per turn. The agent also read weather and requested confirmation before opening the sunroof in rain (policy path) |
| No secrets in image | `.dockerignore` excludes `.env`; keys supplied only at runtime via `--env-file` |

## Known, benign difference from development runs

Development measurements ran the evaluator from local source and recorded our
`adaptive_minimal_*` harness counters inside `turn_metrics` (220k occurrences in
a val-final results file). The **official evaluator image does not propagate
these custom keys**, so they are absent from this run (0 occurrences).

This affects only our own instrumentation, never scoring: rewards, the required
token fields, and per-turn call counts are all present and identical in kind.
Our repository's evaluator-side diffs against upstream `main` are limited to a
`--no-stream` transport option in the A2A client and removal of an unused
import — no scoring logic is modified.
