# Scenario TOML Run Configs

Scenario files are the run configs for local and Docker CAR-bench evaluations.
They define which evaluator and agent-under-test to run, which task subset to
evaluate, how many trials to execute, and which environment variables or Docker
images/builds are used.

## Directory Layout

Scenario directories mirror the reference agent package names under `src/`.
Participant agents should follow the same pattern when adding their own
scenarios.

| Agent Package | Scenario Directory |
| --- | --- |
| `src/track_1_agent_under_test/` | `scenarios/track_1_agent_under_test/` |
| `src/track_2_agent_under_test_cerebras/` | `scenarios/track_2_agent_under_test_cerebras/` |
| `src/track_2_agent_under_test_cerebras_planner/` | `scenarios/track_2_agent_under_test_cerebras_planner/` |

Each directory contains the same six-file matrix:

| Scenario File | Mode | Task Selection |
| --- | --- | --- |
| `local_smoke.toml` | Local Python | Train split, one task from each task type, one trial |
| `local_test_set.toml` | Local Python | Public CAR-bench test split, all tasks from each task type, three trials |
| `local_docker_smoke.toml` | Official evaluator image plus local agent build | Train split, one task from each task type, one trial |
| `local_docker_test_set.toml` | Official evaluator image plus local agent build | Public CAR-bench test split, all tasks from each task type, three trials |
| `ghcr_smoke.toml` | Official evaluator image plus published agent image | Train split, one task from each task type, one trial |
| `ghcr_test_set.toml` | Official evaluator image plus published agent image | Public CAR-bench test split, all tasks from each task type, three trials |

The public test-set scenarios are development validation only. Official final
evaluation is run by the organizers on a hidden test set.

## Submission Scenarios

Final submission can reuse the same scenario TOML shape. Start from the
matching `ghcr_test_set.toml`, replace the agent-under-test image with your
public digest-pinned GHCR image, keep the official evaluator image, and change
the config to the hidden split:

```toml
[config]
num_trials = 3
task_split = "hidden"
tasks_base_num_tasks = -1
tasks_hallucination_num_tasks = -1
tasks_disambiguation_num_tasks = -1
max_steps = 50
```

The submitted scenario should list env var names through Compose-style
interpolation, but must not contain secret values. Organizers provide the
official evaluator runtime and secrets; participants submit only the
agent-under-test image/config needed to run their agent through the official
evaluator boundary.

## TOML Structure

Every scenario has three main tables:

```toml
[evaluator]
# local Python: endpoint + cmd
# Docker: official evaluator image, env, optional command_args

[agent_under_test]
# local Python: endpoint + cmd
# Docker: build or image, env, volumes, optional command_args
# optional result labels: name, result_label, result_model, result_reasoning_effort

[config]
# CAR-bench task/trial selection
```

### `[evaluator]`

The evaluator wraps CAR-bench and owns the simulated user, tools, environment,
and scoring.

Docker scenarios use the official organizer-published evaluator image,
`ghcr.io/car-bench/car-bench-evaluator:latest`. Participants should not build,
self-host, modify, or submit evaluator images; only the agent-under-test image
is participant-controlled.

For local Python scenarios, provide:

```toml
[evaluator]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/evaluator/server.py --host 127.0.0.1 --port 8081"
```

For Docker scenarios, use the official evaluator image:

```toml
[evaluator]
image = "ghcr.io/car-bench/car-bench-evaluator:latest"
env = { GEMINI_API_KEY = "${GEMINI_API_KEY:?Set GEMINI_API_KEY in .env}" }
```

### `[agent_under_test]`

This is the participant or reference agent being evaluated.

For local Python scenarios:

```toml
[agent_under_test]
endpoint = "http://127.0.0.1:8080"
cmd = "python src/track_1_agent_under_test/server.py --host 127.0.0.1 --port 8080"
```

For Docker local-build scenarios:

```toml
[agent_under_test]
build = { context = ".", dockerfile = "src/track_1_agent_under_test/Dockerfile.track-1-agent-under-test" }
env = { AGENT_LLM = "${AGENT_LLM:-gemini/gemini-2.5-flash}" }
```

For GHCR scenarios:

```toml
[agent_under_test]
image = "ghcr.io/yourusername/your-agent:latest"
env = { AGENT_LLM = "${AGENT_LLM:-gemini/gemini-2.5-flash}" }
```

Participant GHCR agent images must be public, or explicitly accessible to the
organizers if a private package is approved. After the first push, check the
package page under either
`https://github.com/users/yourusername/packages/container/package/your-agent`
or `https://github.com/orgs/your-org/packages/container/package/your-agent` and
set **Package visibility** to **Public**. The scenario `image` value must match
the pushed GHCR image.

Optional result-label fields help make output filenames and metadata easier to
read when your harness routes through multiple models:

```toml
[agent_under_test]
name = "my-agent"
result_model = "my-model-or-harness-label"
result_reasoning_effort = "medium"
```

### `[config]`

`[config]` maps to CAR-bench evaluation options:

```toml
[config]
num_trials = 3
task_split = "test"         # "train" or "test"
max_steps = 50

tasks_base_num_tasks = -1
tasks_hallucination_num_tasks = -1
tasks_disambiguation_num_tasks = -1

# Optional exact task filters:
# tasks_base_task_id_filter = ["base_0"]
# tasks_hallucination_task_id_filter = ["hallucination_0"]
# tasks_disambiguation_task_id_filter = ["disambiguation_0"]
```

Use `-1` for all tasks in a task type. Smoke scenarios use `1` for each task
type so you can quickly validate the full loop.

## Environment Variables

Docker scenario env values use Compose-style interpolation:

| Syntax | Meaning |
| --- | --- |
| `${VAR}` | Substitute `VAR` from `.env` or the shell. |
| `${VAR:-default}` | Use `default` when `VAR` is unset or blank. |
| `${VAR:?message}` | Fail early with `message` when `VAR` is missing. |

Keep secret values in `.env` or your deployment secret manager. Do not commit
real API keys.

## Docker Compose Generation

Generate Docker Compose from any `local_docker_*.toml` or `ghcr_*.toml` file:

```bash
uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit
```

The generated Compose file pulls the official evaluator image and then either
builds the local agent-under-test image or pulls the participant-published
agent-under-test image, depending on the selected scenario.

`generate_compose.py` writes two ignored files into the selected scenario
folder:

- `docker-compose.yml`: starts evaluator, agent-under-test, and the A2A client.
- `a2a-scenario.toml`: internal scenario consumed by the A2A client inside
  Docker.

Results are written under `output/<agent-name>/`.
