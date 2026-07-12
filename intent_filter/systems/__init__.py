"""The four intent-filtering pipeline systems under comparison.

Each system module exposes a single `run(instruction, state, ctx) ->
PipelineResult` function with an identical signature (intent_filter.decision.SystemContext,
PipelineResult), so scripts/run_single_instruction.py and the Phase 6
evaluation harness can treat all four uniformly via the `SYSTEMS` registry.
"""

from __future__ import annotations

from typing import Callable

from intent_filter.decision import PipelineResult, SystemContext
from intent_filter.environment.state import WorldState
from intent_filter.systems import baseline_a, baseline_b, multi_agent_ltl, single_llm_ltl

SystemRunFn = Callable[[str, WorldState, SystemContext], PipelineResult]

SYSTEMS: dict[str, SystemRunFn] = {
    "single_llm": baseline_a.run,
    "multi_agent": baseline_b.run,
    "single_llm_ltl": single_llm_ltl.run,
    "multi_agent_ltl": multi_agent_ltl.run,
}

__all__ = ["SYSTEMS", "SystemRunFn", "baseline_a", "baseline_b", "single_llm_ltl", "multi_agent_ltl"]
