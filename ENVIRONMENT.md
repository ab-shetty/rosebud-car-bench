# Environment variable contract

Variable **names only** — no secret values appear in this repository, in
`scenario.toml`, or in the published image.

## Required

| Variable | Consumed by | Purpose |
|---|---|---|
| `CEREBRAS_API_KEY` | agent | Cerebras Cloud API key for the `gpt-oss-120b` agent model |
| `GEMINI_API_KEY` | evaluator | Gemini API key for the fixed user simulator and policy judge |

## Optional — inference routing and model selection

All model names, routes, API bases, service tiers, and reasoning-effort
selectors are configurable, per the submission requirements.

| Variable | Default | Purpose |
|---|---|---|
| `TRACK2_EXECUTOR_MODEL` | `gpt-oss-120b` | Agent model name |
| `TRACK2_CEREBRAS_API_BASE` | *(unset)* | Override the Cerebras API base URL |
| `TRACK2_CEREBRAS_SERVICE_TIER` | *(unset)* | Cerebras service tier |
| `TRACK2_EXECUTOR_REASONING_EFFORT` | `medium` | Reasoning-effort selector |
| `TRACK2_MAX_COMPLETION_TOKENS` | `1024` | Base completion cap |
| `TRACK2_TEMPERATURE` | *(unset)* | Sampling temperature |
| `TRACK2_LLM_MALFORMED_RETRIES` | `1` | Retries on malformed completions |

## Optional — harness selection

| Variable | Default | Purpose |
|---|---|---|
| `TRACK2_HARNESS` | `adaptive_minimal` *(in `scenario.toml`)* | Harness implementation. **The agent's own default is `baseline`**, so the submitted scenario sets this explicitly; omitting it runs the stock baseline. |
| `TRACK2_TRANSPORT` | `chat` | Transport; `harmony_native` selects the alternative native-harmony transport (not used in the submission). |

## Optional — submitted arm configuration

Each is a boolean (`true`/`false`), defaulting to `true` in `scenario.toml`.
Disabling any one runs an ablation of the submitted arm.

`TRACK2_AM_PREFETCH`, `TRACK2_AM_PREFETCH_SEMANTIC`,
`TRACK2_AM_ARGUMENT_BINDING_GUARD`, `TRACK2_AM_DISCLOSURE_GUARD`,
`TRACK2_AM_TRUNCATION_RESCUE`, `TRACK2_AM_PLACEHOLDER_GUARD`,
`TRACK2_AM_VAGUE_DEGREE_CLARIFY`, `TRACK2_AM_SCHEMA_PREFLIGHT`,
`TRACK2_AM_VALUE_PROVENANCE`, `TRACK2_AM_TIME_FORMAT_REVISE`,
`TRACK2_AM_ASK_BUDGET`, `TRACK2_AM_REPEATED_READ_BREAKER`,
`TRACK2_AM_ROUTE_REFERENCE_PREFLIGHT`, `TRACK2_AM_POLICY_LINT`,
`TRACK2_AM_RESCUE_QUALITY`, `TRACK2_AM_INITIAL_CAP_4096`,
`TRACK2_AM_MUTATION_CONSENSUS`, `TRACK2_AM_ROUTE_RESOLVER`,
`TRACK2_AM_ASK_TYPE_GATE`, `TRACK2_AM_TEXTCALL_GUARD`, `TRACK2_AM_ARG_LINT`

## Optional — rate-limit handling

`TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS`,
`TRACK2_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO`,
`TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS`,
`TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS`,
`TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS`,
`TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS`,
`TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS`,
`CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR`

## Logging

| Variable | Default | Purpose |
|---|---|---|
| `LOGURU_LEVEL` | `INFO` | Log verbosity for both services |
