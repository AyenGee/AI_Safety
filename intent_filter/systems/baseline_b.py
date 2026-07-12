"""System 2: Baseline B - Multi-Agent Planner-Critic.

A Planner LLM proposes candidate interpretation(s); a Critic LLM reviews the
top one for semantic/safety issues and returns accept/reject, or the Critic
short-circuits to "clarify" via margin-based ambiguity detection on the
Planner's own confidence scores (no LLM call spent on that case). No formal
verification - critique is still entirely probabilistic. See
docs/architecture.md for the data-flow diagram.
"""

from __future__ import annotations

import time

from intent_filter.agents.critic import review
from intent_filter.agents.planner import plan
from intent_filter.decision import Decision, PipelineResult, StageLog, SystemContext
from intent_filter.environment.state import WorldState

_DECISION_MAP: dict[str, Decision] = {"accept": "Accept", "reject": "Reject", "clarify": "Clarify"}


def run(instruction: str, state: WorldState, ctx: SystemContext) -> PipelineResult:
    planner_start = time.perf_counter()
    planner_output = plan(ctx.client, ctx.models.planner, instruction, state, ctx.ontology)
    planner_latency = time.perf_counter() - planner_start

    planner_stage = StageLog(
        stage="planner",
        detail={
            "interpretations": [
                {"description": i.description, "confidence": i.confidence}
                for i in planner_output.interpretations
            ]
        },
        latency_seconds=planner_latency,
    )

    critic_start = time.perf_counter()
    critic_output = review(
        ctx.client,
        ctx.models.critic,
        instruction,
        planner_output,
        state,
        ctx.ontology,
        ctx.rule_base,
        ctx.ambiguity_margin,
    )
    critic_latency = time.perf_counter() - critic_start

    critic_stage = StageLog(
        stage="critic",
        detail={
            "decision": critic_output.decision,
            "rationale": critic_output.rationale,
            "ambiguity_detected": critic_output.ambiguity_detected,
            "margin": critic_output.margin,
        },
        latency_seconds=critic_latency,
    )

    return PipelineResult(
        decision=_DECISION_MAP[critic_output.decision],
        rationale=critic_output.rationale,
        stages=(planner_stage, critic_stage),
        total_latency_seconds=planner_latency + critic_latency,
    )
