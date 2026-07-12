"""System 4: Multi-Agent + LTL (the proposed / primary system).

Planner -> Critic (with margin-based ambiguity short-circuit) -> [always]
NL->LTL Translator (logged, not decision-gating - see decision.py) ->
deterministic Verifier against the fixed safety rule base. On UNSAT, the
Critic explains the violation in natural language and the Planner gets one
more attempt, bounded by `ctx.max_refinement_attempts`, before defaulting to
Reject. See docs/architecture.md for the full data-flow diagram.

Simplification: when multiple rules are violated at once, the reprompting
loop's feedback is built from the first violated rule found (rule-base
order), not necessarily the most severe - noted here rather than silently
assumed, since it's a defensible but real simplification.
"""

from __future__ import annotations

import time

from intent_filter.agents.critic import explain_violation, review
from intent_filter.agents.planner import plan
from intent_filter.agents.translator import translate
from intent_filter.decision import (
    Decision,
    PipelineResult,
    StageLog,
    SystemContext,
    build_trajectory,
    summarize_violations,
)
from intent_filter.environment.state import WorldState
from intent_filter.verifier import VerificationResult, check_rule_base, overall_result

_DECISION_MAP: dict[str, Decision] = {"accept": "Accept", "reject": "Reject", "clarify": "Clarify"}


def run(instruction: str, state: WorldState, ctx: SystemContext) -> PipelineResult:
    stages: list[StageLog] = []
    total_latency = 0.0

    planner_start = time.perf_counter()
    planner_output = plan(ctx.client, ctx.models.planner, instruction, state, ctx.ontology)
    planner_latency = time.perf_counter() - planner_start
    total_latency += planner_latency
    stages.append(
        StageLog(
            stage="planner",
            detail={
                "attempt": 0,
                "interpretations": [
                    {"description": i.description, "confidence": i.confidence}
                    for i in planner_output.interpretations
                ],
            },
            latency_seconds=planner_latency,
        )
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
    total_latency += critic_latency
    stages.append(
        StageLog(
            stage="critic",
            detail={
                "decision": critic_output.decision,
                "rationale": critic_output.rationale,
                "ambiguity_detected": critic_output.ambiguity_detected,
                "margin": critic_output.margin,
            },
            latency_seconds=critic_latency,
        )
    )

    # Always translate once for logging/accuracy tracking, regardless of the
    # Critic's decision (see decision.py's module docstring).
    translate_start = time.perf_counter()
    translation = translate(
        ctx.client,
        ctx.models.translator,
        instruction,
        ctx.ontology,
        max_retries=ctx.translation_max_retries,
    )
    translate_latency = time.perf_counter() - translate_start
    total_latency += translate_latency
    stages.append(
        StageLog(
            stage="translator",
            detail={
                "ltl_formula": translation.ltl_formula,
                "success": translation.success,
                "used_fallback": translation.used_fallback,
                "attempts": translation.attempts,
            },
            latency_seconds=translate_latency,
        )
    )

    if critic_output.decision in ("reject", "clarify"):
        return PipelineResult(
            decision=_DECISION_MAP[critic_output.decision],
            rationale=critic_output.rationale,
            stages=tuple(stages),
            total_latency_seconds=total_latency,
        )

    # Critic accepted -> verify its chosen interpretation, with a bounded
    # reprompting loop on UNSAT.
    current_actions = critic_output.chosen_interpretation.actions
    refinement_attempts = 0

    while True:
        verify_start = time.perf_counter()
        trajectory = build_trajectory(state, current_actions, ctx.ontology)
        if trajectory is None:
            verify_latency = time.perf_counter() - verify_start
            total_latency += verify_latency
            stages.append(
                StageLog(
                    stage="verifier",
                    detail={
                        "attempt": refinement_attempts,
                        "result": "UNKNOWN",
                        "reason": "proposed action sequence is physically invalid",
                    },
                    latency_seconds=verify_latency,
                )
            )
            return PipelineResult(
                decision="Reject",
                rationale="Rejected: the proposed action sequence violates an environment precondition.",
                stages=tuple(stages),
                total_latency_seconds=total_latency,
                refinement_attempts=refinement_attempts,
            )

        outcomes = check_rule_base(ctx.rule_base, trajectory, ctx.ontology)
        verify_latency = time.perf_counter() - verify_start
        total_latency += verify_latency
        result = overall_result(outcomes)
        stages.append(
            StageLog(
                stage="verifier",
                detail={
                    "attempt": refinement_attempts,
                    "result": result.value,
                    "violated_rules": [
                        rid for rid, o in outcomes.items() if o.result is VerificationResult.UNSAT
                    ],
                },
                latency_seconds=verify_latency,
            )
        )

        if result is VerificationResult.SAT:
            return PipelineResult(
                decision="Accept",
                rationale=critic_output.rationale,
                stages=tuple(stages),
                total_latency_seconds=total_latency,
                refinement_attempts=refinement_attempts,
            )

        if refinement_attempts >= ctx.max_refinement_attempts:
            return PipelineResult(
                decision="Reject",
                rationale=summarize_violations(outcomes),
                stages=tuple(stages),
                total_latency_seconds=total_latency,
                refinement_attempts=refinement_attempts,
            )

        # Reprompting loop: Critic explains the (first) violation, Planner
        # gets one more attempt.
        first_violation = next(
            o for o in outcomes.values() if o.result is VerificationResult.UNSAT
        )
        explain_start = time.perf_counter()
        feedback = explain_violation(
            ctx.client, ctx.models.critic, instruction, first_violation
        )
        explain_latency = time.perf_counter() - explain_start
        total_latency += explain_latency
        stages.append(
            StageLog(
                stage="critic_explain",
                detail={"attempt": refinement_attempts, "feedback": feedback},
                latency_seconds=explain_latency,
            )
        )

        refinement_attempts += 1

        replan_start = time.perf_counter()
        planner_output = plan(
            ctx.client, ctx.models.planner, instruction, state, ctx.ontology, feedback=feedback
        )
        replan_latency = time.perf_counter() - replan_start
        total_latency += replan_latency
        stages.append(
            StageLog(
                stage="planner",
                detail={
                    "attempt": refinement_attempts,
                    "interpretations": [
                        {"description": i.description, "confidence": i.confidence}
                        for i in planner_output.interpretations
                    ],
                },
                latency_seconds=replan_latency,
            )
        )
        current_actions = planner_output.top.actions
