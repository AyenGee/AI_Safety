"""Shared types and helpers for the decision layer.

Every one of the four intent-filtering systems (intent_filter/systems/)
returns a `PipelineResult` built from `StageLog` entries, so results are
directly comparable across systems in the Phase 6 evaluation harness -
same decision vocabulary, same latency breakdown shape, same trace format
for auditability.

Design note: for the two LTL-augmented systems, the verifier's
decision-relevant check is the candidate trajectory against the *fixed*
safety rule base (config/safety_rules.yaml) only - not the NL->LTL
Translator's per-instruction formula. The Translator still runs and its
output is logged on every LTL-augmented run; its accuracy becomes its own
ablation metric in Phase 6 (compare its formula against the rule(s) a
dataset example was designed to exercise) rather than gating the decision
itself. This keeps "does formal verification help" cleanly isolated from
"is the translator good", and keeps the decision grounded in the same fixed
rule base the dataset's gold labels are themselves built from. See
docs/methodology.md for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intent_filter.agents.client import LLMClient
from intent_filter.config import ModelsConfig
from intent_filter.environment.actions import Action, InvalidActionError
from intent_filter.environment.actions import apply_sequence as _apply_sequence
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.rules import SafetyRuleBase
from intent_filter.environment.state import WorldState
from intent_filter.verifier import VerificationOutcome, violated_rules

Decision = Literal["Accept", "Reject", "Clarify"]


@dataclass(frozen=True)
class StageLog:
    """One stage of a pipeline run (an agent call or a verifier check).

    `detail` is a plain, JSON-serializable dict so a full run trace can be
    logged/replayed for the research write-up without special-casing each
    stage type.
    """

    stage: str
    detail: dict
    latency_seconds: float


@dataclass(frozen=True)
class PipelineResult:
    decision: Decision
    rationale: str
    stages: tuple[StageLog, ...]
    total_latency_seconds: float
    refinement_attempts: int = 0

    def latency_by_stage(self) -> dict[str, float]:
        """Sum latency per stage name (a name can repeat across reprompting attempts)."""
        totals: dict[str, float] = {}
        for stage in self.stages:
            totals[stage.stage] = totals.get(stage.stage, 0.0) + stage.latency_seconds
        return totals


@dataclass(frozen=True)
class SystemContext:
    """Everything a pipeline system needs beyond the instruction and starting state."""

    client: LLMClient
    models: ModelsConfig
    ontology: Ontology
    rule_base: SafetyRuleBase
    ambiguity_margin: float = 0.15
    max_refinement_attempts: int = 2
    translation_max_retries: int = 3


def build_trajectory(
    state: WorldState, actions: tuple[Action, ...], ontology: Ontology
) -> list[WorldState] | None:
    """Execute `actions` from `state`, returning the resulting trajectory.

    Returns None (rather than raising) if the proposed action sequence
    violates an environment precondition (e.g. picking up an object not in
    the agent's current room) - an LLM-proposed plan can be physically
    incoherent, and callers should treat that the same as a safety failure
    (Reject), since there is nothing sensible left to verify.
    """
    try:
        return _apply_sequence(state, list(actions), ontology)
    except InvalidActionError:
        return None


def summarize_violations(outcomes: dict[str, VerificationOutcome]) -> str:
    """Build a human-readable rationale from check_rule_base() outcomes.

    Used by the LTL-augmented systems (single_llm_ltl, multi_agent_ltl) as
    the PipelineResult rationale when the verifier rejects a plan, and by
    the Critic's reprompting-loop feedback (multi_agent_ltl).
    """
    bad_ids = violated_rules(outcomes)
    if not bad_ids:
        return "No safety rule violations found."
    parts = [f"{rule_id}: {outcomes[rule_id].explanation}" for rule_id in bad_ids]
    return "Rejected by formal verification - violated rule(s): " + "; ".join(parts)
