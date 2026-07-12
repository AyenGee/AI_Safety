"""System 3: Single-LLM + LTL.

The same single-LLM call as Baseline A plans and adjudicates the command;
the NL->LTL Translator always runs too (so its output/accuracy is logged for
every instruction, not just accepted ones), and when the LLM's own decision
is "accept", a deterministic verifier checks the LLM's proposed action
trajectory against the *fixed* safety rule base before the final decision.

Design choice (confirmed with the researcher, see docs/methodology.md): the
verifier gates on the fixed rule base (config/safety_rules.yaml), not on the
Translator's per-instruction formula - the Translator's formula is logged
only, and its own accuracy is a separate Phase 6 ablation metric. The
verifier can only make the decision *stricter* than the LLM's own judgement
(catch an unsafe "accept"); it never overrides an LLM "reject" back to
"accept" - a Reject from the LLM's own judgement may reflect a semantic
issue outside the fixed rule base's coverage.
"""

from __future__ import annotations

import time

from intent_filter.agents.single_llm import run as run_single_llm
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
    llm_start = time.perf_counter()
    llm_output = run_single_llm(
        ctx.client, ctx.models.single_llm, instruction, state, ctx.ontology, ctx.rule_base
    )
    llm_latency = time.perf_counter() - llm_start
    llm_stage = StageLog(
        stage="single_llm",
        detail={
            "decision": llm_output.decision,
            "rationale": llm_output.rationale,
            "description": llm_output.description,
            "actions": [repr(a) for a in llm_output.actions],
        },
        latency_seconds=llm_latency,
    )

    translate_start = time.perf_counter()
    translation = translate(
        ctx.client,
        ctx.models.translator,
        instruction,
        ctx.ontology,
        max_retries=ctx.translation_max_retries,
    )
    translate_latency = time.perf_counter() - translate_start
    translate_stage = StageLog(
        stage="translator",
        detail={
            "ltl_formula": translation.ltl_formula,
            "success": translation.success,
            "used_fallback": translation.used_fallback,
            "attempts": translation.attempts,
        },
        latency_seconds=translate_latency,
    )

    stages = [llm_stage, translate_stage]
    total_latency = llm_latency + translate_latency

    # A Reject/Clarify from the LLM's own judgement is not overridden by the
    # verifier - the verifier can only add stricter Rejects, never soften one.
    if llm_output.decision in ("reject", "clarify"):
        return PipelineResult(
            decision=_DECISION_MAP[llm_output.decision],
            rationale=llm_output.rationale,
            stages=tuple(stages),
            total_latency_seconds=total_latency,
        )

    verify_start = time.perf_counter()
    trajectory = build_trajectory(state, llm_output.actions, ctx.ontology)
    if trajectory is None:
        verify_latency = time.perf_counter() - verify_start
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
            total_latency_seconds=total_latency + verify_latency,
        )

    outcomes = check_rule_base(ctx.rule_base, trajectory, ctx.ontology)
    verify_latency = time.perf_counter() - verify_start
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
    total_latency += verify_latency

    if result is VerificationResult.SAT:
        return PipelineResult(
            decision="Accept",
            rationale=llm_output.rationale,
            stages=tuple(stages),
            total_latency_seconds=total_latency,
        )

    return PipelineResult(
        decision="Reject",
        rationale=summarize_violations(outcomes),
        stages=tuple(stages),
        total_latency_seconds=total_latency,
    )
