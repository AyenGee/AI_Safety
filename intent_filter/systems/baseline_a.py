"""System 1: Baseline A - Single-LLM Intent Filter.

One LLM call performs both task planning and safety adjudication via
prompting alone. No external verifier. See docs/architecture.md for the
data-flow diagram.
"""

from __future__ import annotations

import time

from intent_filter.agents.single_llm import run as run_single_llm
from intent_filter.decision import Decision, PipelineResult, StageLog, SystemContext
from intent_filter.environment.state import WorldState

_DECISION_MAP: dict[str, Decision] = {"accept": "Accept", "reject": "Reject", "clarify": "Clarify"}


def run(instruction: str, state: WorldState, ctx: SystemContext) -> PipelineResult:
    start = time.perf_counter()
    output = run_single_llm(
        ctx.client, ctx.models.single_llm, instruction, state, ctx.ontology, ctx.rule_base
    )
    latency = time.perf_counter() - start

    stage = StageLog(
        stage="single_llm",
        detail={
            "decision": output.decision,
            "rationale": output.rationale,
            "description": output.description,
            "actions": [repr(a) for a in output.actions],
        },
        latency_seconds=latency,
    )

    return PipelineResult(
        decision=_DECISION_MAP[output.decision],
        rationale=output.rationale,
        stages=(stage,),
        total_latency_seconds=latency,
    )
