"""System 5 (experimental): Planner + Verifier only, strict one-shot.

A Planner LLM proposes a single candidate action sequence for the
instruction; that plan is checked exactly once against the fixed safety
rule base by the deterministic verifier. Accept on SAT, Reject on UNSAT -
no Critic call, no ambiguity short-circuit (ambiguity detection lives
inside critic.review, same coupling noted for the remove_critic ablation -
removing the Critic removes clarification too), and critically, no
reprompting loop: unlike remove_critic (multi_agent_ltl.run(...,
use_critic=False), which still gives the Planner one bounded retry with the
verifier's feedback on UNSAT), this system's Planner gets exactly one
attempt. That retry loop is exactly where remove_critic's one documented
quirk lives - a revised plan can satisfy the letter of the rule base while
no longer resembling the original instruction, with no Critic to notice the
drift (see docs/methodology.md). This system can't exhibit that quirk,
because there is no second attempt: the LLM never offers any opinion on
whether the instruction is safe, and the verifier's verdict on the first
and only plan is final.

Kept as its own module rather than a flag on multi_agent_ltl.run(), unlike
the three Phase 6 ablations, since it isn't a component-removal ablation of
the reported system - it's a structurally different pipeline (no retry
loop at all) being evaluated as a system in its own right. Not part of the
reported four systems or Phase 8 - see docs/methodology.md "Planner +
Verifier only (strict one-shot)".
"""

from __future__ import annotations

import time

from intent_filter.agents.planner import plan
from intent_filter.agents.translator import translate
from intent_filter.decision import (
    Decision,
    PipelineResult,
    StageLog,
    SystemContext,
    build_trajectory,
    guard_against_actionless_accept,
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
                "interpretations": [
                    {"description": i.description, "confidence": i.confidence}
                    for i in planner_output.interpretations
                ]
            },
            latency_seconds=planner_latency,
        )
    )

    current_actions = planner_output.top.actions
    decision_so_far, rationale_so_far = guard_against_actionless_accept(
        "accept",
        "No semantic review - proceeding directly to formal verification.",
        current_actions,
    )

    # Logged for every run, like the other LTL-augmented systems, even
    # though the decision is gated on the fixed rule base, not this formula.
    translate_start = time.perf_counter()
    translation = translate(
        ctx.client, ctx.models.translator, instruction, ctx.ontology,
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

    if decision_so_far == "clarify":
        return PipelineResult(
            decision="Clarify",
            rationale=rationale_so_far,
            stages=tuple(stages),
            total_latency_seconds=total_latency,
            chosen_actions=current_actions,
        )

    verify_start = time.perf_counter()
    trajectory = build_trajectory(state, current_actions, ctx.ontology)
    if trajectory is None:
        verify_latency = time.perf_counter() - verify_start
        total_latency += verify_latency
        stages.append(
            StageLog(
                stage="verifier",
                detail={"result": "UNKNOWN", "reason": "proposed action sequence is physically invalid"},
                latency_seconds=verify_latency,
            )
        )
        return PipelineResult(
            decision="Reject",
            rationale="Rejected: the proposed action sequence violates an environment precondition.",
            stages=tuple(stages),
            total_latency_seconds=total_latency,
            chosen_actions=current_actions,
        )

    outcomes = check_rule_base(ctx.rule_base, trajectory, ctx.ontology)
    verify_latency = time.perf_counter() - verify_start
    total_latency += verify_latency
    result = overall_result(outcomes)
    stages.append(
        StageLog(
            stage="verifier",
            detail={
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
            rationale=rationale_so_far,
            stages=tuple(stages),
            total_latency_seconds=total_latency,
            chosen_actions=current_actions,
        )

    return PipelineResult(
        decision="Reject",
        rationale=summarize_violations(outcomes),
        stages=tuple(stages),
        total_latency_seconds=total_latency,
        chosen_actions=current_actions,
    )
