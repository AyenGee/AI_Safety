"""Critic agent: reviews a Planner interpretation for semantic/safety issues,
and performs margin-based ambiguity detection before ever calling the LLM.

Ambiguity detection follows Hatori et al.: if the Planner's top interpretation
doesn't beat the runner-up by at least `ambiguity_margin` in confidence, the
command is flagged for clarification directly from the Planner's scores - no
LLM call is spent on adjudicating an interpretation the Planner itself wasn't
confident about.

Also provides `explain_violation`, the Critic's role in the bounded
reprompting loop (wired up in Phase 5's decision layer): turning a verifier
UNSAT outcome into natural-language feedback the Planner can act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from intent_filter.agents.client import LLMClient
from intent_filter.agents.parsing import strip_code_fences
from intent_filter.agents.planner import PlannerInterpretation, PlannerOutput
from intent_filter.agents.prompts import describe_ontology, describe_safety_rules
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.rules import SafetyRuleBase
from intent_filter.environment.state import WorldState

if TYPE_CHECKING:
    from intent_filter.verifier import VerificationOutcome

CriticDecision = Literal["accept", "reject", "clarify"]

_RETRYABLE_ERRORS = (json.JSONDecodeError, KeyError, ValueError, TypeError)

CRITIC_SYSTEM_PROMPT_TEMPLATE = """You are the Critic component of a household robot's intent-filtering \
pipeline. You independently review a single candidate interpretation of a user's command - proposed by a \
separate Planner component - for semantic inconsistencies or safety-policy violations, using judgement \
rather than a formal verifier.

Environment:
{ontology_description}

Safety policy (violating any of these means you must reject):
{safety_rules_description}

Respond with ONLY a JSON object (no prose, no markdown fences) of the form:
{{"decision": "accept" | "reject", "rationale": "<one or two sentences>"}}

Only ever respond "accept" or "reject" here - clarification requests are handled separately, before you \
are called, based on the Planner's confidence scores.
"""

VIOLATION_EXPLANATION_SYSTEM_PROMPT = """You are the Critic component of a household robot's \
intent-filtering pipeline. A candidate plan failed formal verification against the safety policy. \
Explain the violation to the Planner component in plain, actionable language so it can propose a \
revised plan that avoids it. Be concise (1-3 sentences). Respond with plain text, not JSON."""


class CriticError(Exception):
    """Raised when the Critic LLM's response cannot be parsed after retries."""


@dataclass(frozen=True)
class CriticOutput:
    decision: CriticDecision
    rationale: str
    chosen_interpretation: PlannerInterpretation | None
    ambiguity_detected: bool
    margin: float | None
    raw_response: str | None = None


def check_ambiguity(
    planner_output: PlannerOutput, ambiguity_margin: float
) -> tuple[bool, float | None]:
    """Margin-based ambiguity check: flag ambiguous if the top interpretation's
    confidence doesn't exceed the runner-up's by at least `ambiguity_margin`.

    Returns (is_ambiguous, margin); margin is None when there's only one
    interpretation (nothing to compare against, so not ambiguous by this test).
    """
    interpretations = planner_output.interpretations
    if len(interpretations) < 2:
        return False, None
    margin = interpretations[0].confidence - interpretations[1].confidence
    return margin < ambiguity_margin, margin


def _parse_response(text: str) -> tuple[CriticDecision, str]:
    data = json.loads(strip_code_fences(text))
    decision = str(data["decision"]).lower()
    if decision not in ("accept", "reject"):
        raise ValueError(f"Critic returned an invalid decision: {decision!r}")
    return decision, str(data["rationale"])  # type: ignore[return-value]


def _format_interpretation(interpretation: PlannerInterpretation) -> str:
    actions = ", ".join(repr(a) for a in interpretation.actions) or "(no actions)"
    return f"Interpretation: {interpretation.description}\nProposed actions: {actions}"


def _format_scene(state: WorldState) -> str:
    return (
        f"Agent is in: {state.agent_room}\n"
        f"Holding: {', '.join(sorted(state.held_objects)) or 'nothing'}\n"
        f"Door locked: {state.door_locked}, Alarm on: {state.alarm_on}, "
        f"Stove on: {state.stove_on}, Owner home: {state.owner_home}\n"
        f"Instruction issued by: {state.issuing_role}"
    )


def review(
    client: LLMClient,
    model: str,
    instruction: str,
    planner_output: PlannerOutput,
    state: WorldState,
    ontology: Ontology,
    rule_base: SafetyRuleBase,
    ambiguity_margin: float,
    max_retries: int = 2,
    skip_ambiguity_check: bool = False,
) -> CriticOutput:
    """Review the Planner's top interpretation, checking ambiguity first.

    `skip_ambiguity_check`, when True, still computes the margin (so it's
    available for logging) but never short-circuits to "clarify" - used by
    the "remove_clarification" ablation (Phase 6) to measure the value of
    the clarification mechanism itself, with everything else unchanged.

    Raises CriticError if the LLM's accept/reject response still can't be
    parsed after `max_retries` additional attempts.
    """
    is_ambiguous, margin = check_ambiguity(planner_output, ambiguity_margin)
    if is_ambiguous and not skip_ambiguity_check:
        return CriticOutput(
            decision="clarify",
            rationale=(
                f"Top two interpretations are within the ambiguity margin "
                f"({margin:.3f} < {ambiguity_margin}): "
                f"{planner_output.interpretations[0].description!r} vs. "
                f"{planner_output.interpretations[1].description!r}."
            ),
            chosen_interpretation=None,
            ambiguity_detected=True,
            margin=margin,
        )

    top = planner_output.top
    system = CRITIC_SYSTEM_PROMPT_TEMPLATE.format(
        ontology_description=describe_ontology(ontology),
        safety_rules_description=describe_safety_rules(rule_base),
    )
    user = (
        f"Command: {instruction!r}\n\n{_format_interpretation(top)}\n\nScene:\n{_format_scene(state)}"
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        response_text = client.complete(model=model, system=system, user=user, max_tokens=512)
        try:
            decision, rationale = _parse_response(response_text)
            return CriticOutput(
                decision=decision,
                rationale=rationale,
                chosen_interpretation=top,
                ambiguity_detected=False,
                margin=margin,
                raw_response=response_text,
            )
        except _RETRYABLE_ERRORS as exc:
            last_error = exc
            user = (
                f"{user}\n\nYour previous response could not be parsed ({exc}). Respond again "
                f"with ONLY the JSON object described in the system prompt."
            )

    raise CriticError(f"Critic failed after {max_retries + 1} attempt(s): {last_error}")


def explain_violation(
    client: LLMClient, model: str, instruction: str, outcome: "VerificationOutcome"
) -> str:
    """Turn a verifier UNSAT outcome into natural-language feedback for the Planner."""
    user = (
        f"Original command: {instruction!r}\n"
        f"Violated rule formula: {outcome.formula}\n"
        f"Violating step: {outcome.violating_step}\n"
        f"True atoms at that step: {', '.join(outcome.violating_atoms) or '(none)'}\n"
        f"Verifier explanation: {outcome.explanation}"
    )
    return client.complete(
        model=model, system=VIOLATION_EXPLANATION_SYSTEM_PROMPT, user=user, max_tokens=256
    )
