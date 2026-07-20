"""Server entry point for the Track 2 Cerebras CAR-bench agent."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

if __package__:
    from .adaptive_minimal import (
        HARNESS_NAME as ADAPTIVE_MINIMAL_HARNESS_NAME,
        build_planner_from_env as build_adaptive_minimal_planner,
    )
    from .car_bench_agent import CARBenchAgentExecutor
    from .cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
    )
    from .consensus_planner import build_planner_from_env as build_consensus_planner
    from .scatter_sharpen import (
        HARNESS_NAME as SCATTER_SHARPEN_HARNESS_NAME,
        build_planner_from_env as build_scatter_planner,
    )
else:
    from adaptive_minimal import (
        HARNESS_NAME as ADAPTIVE_MINIMAL_HARNESS_NAME,
        build_planner_from_env as build_adaptive_minimal_planner,
    )
    from car_bench_agent import CARBenchAgentExecutor
    from cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
    )
    from consensus_planner import build_planner_from_env as build_consensus_planner
    from scatter_sharpen import (
        HARNESS_NAME as SCATTER_SHARPEN_HARNESS_NAME,
        build_planner_from_env as build_scatter_planner,
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="server")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_float(name: str, default: float | None = None) -> float | None:
    value = _env_or_default(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = _env_or_default(name)
    if value is None:
        return default
    return int(value)


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the Cerebras agent under test."""

    card = AgentCard(
        name="car_bench_agent_cerebras",
        description=(
            "In-car voice assistant agent for CAR-bench using direct "
            "Cerebras SDK inference"
        ),
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    iface = card.supported_interfaces.add()
    iface.url = url
    iface.protocol_binding = "JSONRPC"
    iface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant (Cerebras)"
    skill.description = "Returns CAR-bench text responses or tool calls through A2A"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant", "cerebras"])

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench Track 2 Cerebras agent under test."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--service-tier", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--executor-reasoning-effort", type=str, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--malformed-retries", type=int, default=None)
    args = parser.parse_args()

    if not _env_or_default("CEREBRAS_API_KEY"):
        raise SystemExit("CEREBRAS_API_KEY must be set for Track 2 Cerebras runs.")

    executor_model = (
        args.executor_model
        if args.executor_model is not None
        else _env_or_default("TRACK2_EXECUTOR_MODEL", DEFAULT_EXECUTOR_MODEL)
    )
    service_tier = (
        args.service_tier
        if args.service_tier is not None
        else _env_or_default("TRACK2_CEREBRAS_SERVICE_TIER")
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else _env_float("TRACK2_TEMPERATURE")
    )
    reasoning_effort = (
        args.executor_reasoning_effort
        if args.executor_reasoning_effort is not None
        else (
            args.reasoning_effort
            if args.reasoning_effort is not None
            else _env_or_default(
                "TRACK2_EXECUTOR_REASONING_EFFORT",
                DEFAULT_EXECUTOR_REASONING_EFFORT,
            )
        )
    )
    max_completion_tokens = (
        args.max_completion_tokens
        if args.max_completion_tokens is not None
        else _env_int("TRACK2_MAX_COMPLETION_TOKENS", 1024)
    )
    malformed_retries = (
        args.malformed_retries
        if args.malformed_retries is not None
        else _env_int("TRACK2_LLM_MALFORMED_RETRIES", 1)
    )

    harness = (_env_or_default("TRACK2_HARNESS", "baseline") or "baseline").strip()
    transport = (
        _env_or_default("TRACK2_TRANSPORT", "chat") or "chat"
    ).strip().casefold()
    planner_kind = (
        _env_or_default("TRACK2_PLANNER", "scatter") or "scatter"
    ).strip().lower()

    logger.info(
        "Starting CAR-bench agent (Cerebras)",
        harness=harness,
        transport=transport,
        planner=planner_kind,
        executor_model=executor_model,
        service_tier=service_tier,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        malformed_retries=malformed_retries,
        host=args.host,
        port=args.port,
    )

    planner = None
    if harness == SCATTER_SHARPEN_HARNESS_NAME:
        if planner_kind == "scatter":
            planner_factory = build_scatter_planner
        elif planner_kind == "consensus":
            planner_factory = build_consensus_planner
        else:
            raise SystemExit(
                "TRACK2_PLANNER must be one of: scatter, consensus "
                f"(got {planner_kind!r})."
            )
        planner = planner_factory(
            model=executor_model or DEFAULT_EXECUTOR_MODEL,
            api_base=DEFAULT_CEREBRAS_API_BASE,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
            logger=logger,
        )
    elif harness == ADAPTIVE_MINIMAL_HARNESS_NAME:
        if transport not in {"chat", "harmony_native"}:
            raise SystemExit(
                "TRACK2_TRANSPORT must be chat or harmony_native for "
                f"adaptive_minimal (got {transport!r})."
            )
        adaptive_planner = build_adaptive_minimal_planner(
            model=executor_model or DEFAULT_EXECUTOR_MODEL,
            api_base=DEFAULT_CEREBRAS_API_BASE,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            transport=transport,
            logger=logger,
        )
        planner = adaptive_planner
        csp_startup = {}
        if adaptive_planner.csp_enabled:
            csp_startup = {
                "csp_brief": adaptive_planner.config.csp_brief,
                "csp_afford": adaptive_planner.config.csp_afford,
            }
        logger.info(
            "Adaptive-minimal startup checks passed",
            micro_prompt_tokens=adaptive_planner.micro_prompt_token_count,
            transport=transport,
            prefetch=adaptive_planner.config.prefetch,
            prefetch_semantic=adaptive_planner.config.prefetch_semantic,
            turn_guard=adaptive_planner.config.turn_guard,
            procedures=adaptive_planner.config.procedures,
            autopsy_fixes=adaptive_planner.config.autopsy_fixes,
            event_exemplars=adaptive_planner.config.event_exemplars,
            terminal_readback=adaptive_planner.config.terminal_readback,
            phase_gate=adaptive_planner.config.phase_gate,
            terminal_effort_high=adaptive_planner.config.terminal_effort_high,
            argument_binding_guard=(
                adaptive_planner.config.argument_binding_guard
            ),
            disclosure_guard=adaptive_planner.config.disclosure_guard,
            truncation_rescue=adaptive_planner.config.truncation_rescue,
            placeholder_guard=adaptive_planner.config.placeholder_guard,
            vague_degree_clarify=(
                adaptive_planner.config.vague_degree_clarify
            ),
            schema_preflight=adaptive_planner.config.schema_preflight,
            value_provenance=adaptive_planner.config.value_provenance,
            time_format_revise=adaptive_planner.config.time_format_revise,
            ask_budget=adaptive_planner.config.ask_budget,
            repeated_read_breaker=(
                adaptive_planner.config.repeated_read_breaker
            ),
            route_reference_preflight=(
                adaptive_planner.config.route_reference_preflight
            ),
            policy_lint=adaptive_planner.config.policy_lint,
            rescue_quality=adaptive_planner.config.rescue_quality,
            initial_cap_4096=adaptive_planner.config.initial_cap_4096,
            initial_cap_8192=adaptive_planner.config.initial_cap_8192,
            mutation_consensus=adaptive_planner.config.mutation_consensus,
            consensus_mixed_effort=(
                adaptive_planner.config.consensus_mixed_effort
            ),
            executor_effort_high=(
                adaptive_planner.config.executor_effort_high
            ),
            struggle_effort=adaptive_planner.config.struggle_effort,
            terminal_consensus=adaptive_planner.config.terminal_consensus,
            consensus_deepen=adaptive_planner.config.consensus_deepen,
            terminal_medium=adaptive_planner.config.terminal_medium,
            route_resolver=adaptive_planner.config.route_resolver,
            route_budget=adaptive_planner.config.route_budget,
            route_budget_limit=adaptive_planner.config.route_budget_limit,
            nav_intent_preflight=(
                adaptive_planner.config.nav_intent_preflight
            ),
            step_coverage=adaptive_planner.config.step_coverage,
            p3_ask_gate_v2=adaptive_planner.config.p3_ask_gate_v2,
            ask_type_gate=adaptive_planner.config.ask_type_gate,
            textcall_guard=adaptive_planner.config.textcall_guard,
            arg_lint=adaptive_planner.config.arg_lint,
            read_resolve=adaptive_planner.config.read_resolve,
            grounded_ask=adaptive_planner.config.grounded_ask,
            ask_content_consensus=(
                adaptive_planner.config.ask_content_consensus
            ),
            llm_consensus_judge=(
                adaptive_planner.config.llm_consensus_judge
            ),
            llm_ask_triage=adaptive_planner.config.llm_ask_triage,
            llm_limitation_classifier=(
                adaptive_planner.config.llm_limitation_classifier
            ),
            executor_reasoning_effort=(
                adaptive_planner.executor_reasoning_effort
            ),
            repetition_guard=adaptive_planner.repetition_guard_enabled,
            max_completion_tokens=adaptive_planner.max_completion_tokens,
            **csp_startup,
        )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=CARBenchAgentExecutor(
            model=executor_model or DEFAULT_EXECUTOR_MODEL,
            api_base=DEFAULT_CEREBRAS_API_BASE,
            service_tier=service_tier,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            malformed_retries=malformed_retries,
            planner=planner,
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True)
    card_routes = create_agent_card_routes(card)
    app = Starlette(routes=routes + card_routes)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
