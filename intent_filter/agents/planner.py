"""Planner agent: interprets a natural-language command against the
environment ontology and current scene, proposing one or more ranked
candidate interpretations (grounded description + confidence + symbolic
action sequence).

Proposing *multiple* interpretations with confidence scores - rather than
just one - is deliberate: it is what lets the Critic's margin-based
ambiguity check (Hatori et al.-style, see agents/critic.py) detect ambiguity
without a separate ambiguity-classification model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from intent_filter.agents.client import LLMClient
from intent_filter.agents.parsing import strip_code_fences
from intent_filter.agents.prompts import describe_action_schema, describe_ontology
from intent_filter.environment.actions import Action, ActionType
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.state import WorldState

_VALID_ACTION_TYPES = {t.name for t in ActionType}

_RETRYABLE_ERRORS = (json.JSONDecodeError, KeyError, ValueError, TypeError)

PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are the Planner component of a household robot's intent-filtering \
pipeline. Given a natural-language command and the current scene, propose one or more candidate \
interpretations of what the user wants the robot to do.

Environment:
{ontology_description}

{action_schema}

Respond with ONLY a JSON object (no prose, no markdown fences) of the form:
{{"interpretations": [{{"description": "<grounded restatement of what this interpretation does>", \
"confidence": <float between 0 and 1>, "actions": [{{"type": "<ACTION_TYPE>", \
"argument": "<room_or_object_name_or_null>"}}]}}]}}

Rules:
- Order interpretations by descending confidence.
- If the command is clear and unambiguous given the scene, return exactly one interpretation with \
confidence >= 0.9.
- If the command is genuinely ambiguous (the target object, room, or action cannot be determined from \
the command and scene alone), return two or more plausible interpretations whose confidence scores are \
close together, so the ambiguity is visible in the score gap. Do not artificially force high confidence \
on a guess.
- Only use rooms, objects, and action types from the schema above - never invent new ones.
- Each action sequence should be the concrete steps to carry out that interpretation (e.g. MOVE to a \
room before PICK_UP-ing an object located there).
"""


class PlannerError(Exception):
    """Raised when the Planner LLM's response cannot be parsed after retries."""


@dataclass(frozen=True)
class PlannerInterpretation:
    description: str
    confidence: float
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class PlannerOutput:
    interpretations: tuple[PlannerInterpretation, ...]
    raw_response: str

    @property
    def top(self) -> PlannerInterpretation:
        return self.interpretations[0]


def _parse_action(raw: dict) -> Action:
    type_name = str(raw.get("type", "")).upper()
    if type_name not in _VALID_ACTION_TYPES:
        raise ValueError(f"Unknown action type: {raw.get('type')!r}")
    return Action(action_type=ActionType[type_name], argument=raw.get("argument"))


def _parse_response(text: str) -> PlannerOutput:
    data = json.loads(strip_code_fences(text))
    raw_interpretations = data["interpretations"]
    if not raw_interpretations:
        raise ValueError("Planner returned zero interpretations")

    interpretations = [
        PlannerInterpretation(
            description=str(item["description"]),
            confidence=float(item["confidence"]),
            actions=tuple(_parse_action(a) for a in item.get("actions", [])),
        )
        for item in raw_interpretations
    ]
    interpretations.sort(key=lambda i: i.confidence, reverse=True)
    return PlannerOutput(interpretations=tuple(interpretations), raw_response=text)


def _format_scene(state: WorldState) -> str:
    return (
        f"Agent is in: {state.agent_room}\n"
        f"Holding: {', '.join(sorted(state.held_objects)) or 'nothing'}\n"
        f"Object locations: {state.object_locations}\n"
        f"Door locked: {state.door_locked}, Alarm on: {state.alarm_on}, "
        f"Stove on: {state.stove_on}, Owner home: {state.owner_home}\n"
        f"Instruction issued by: {state.issuing_role}"
    )


def plan(
    client: LLMClient,
    model: str,
    instruction: str,
    state: WorldState,
    ontology: Ontology,
    max_retries: int = 2,
) -> PlannerOutput:
    """Ask the Planner LLM for one or more candidate interpretations of `instruction`.

    Raises PlannerError if the response still can't be parsed as valid JSON
    matching the expected schema after `max_retries` additional attempts.
    """
    system = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        ontology_description=describe_ontology(ontology),
        action_schema=describe_action_schema(),
    )
    user = f"Command: {instruction!r}\n\nScene:\n{_format_scene(state)}"

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        response_text = client.complete(model=model, system=system, user=user, max_tokens=1024)
        try:
            return _parse_response(response_text)
        except _RETRYABLE_ERRORS as exc:
            last_error = exc
            user = (
                f"Command: {instruction!r}\n\nScene:\n{_format_scene(state)}\n\n"
                f"Your previous response could not be parsed ({exc}). Respond again with ONLY "
                f"the JSON object described in the system prompt."
            )

    raise PlannerError(f"Planner failed after {max_retries + 1} attempt(s): {last_error}")
