"""The planning problem tuple `P = <O, Pr, A, S, T, I, G, tau>` from the methodology.

Bundles a concrete problem instance (one instruction, one initial scene) for
a pipeline system to consume, without prescribing how any system solves it.
"""

from __future__ import annotations

from dataclasses import dataclass

from intent_filter.environment.actions import ActionType
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.state import WorldState


@dataclass(frozen=True)
class PlanningProblem:
    """One instance of the planning problem tuple.

    - O, Pr  -> `ontology` (objects and their properties; rooms and their tags)
    - A      -> `available_actions` (the fixed action schema, see ActionType)
    - S      -> implicit: the WorldState type (not enumerated explicitly)
    - T      -> intent_filter.environment.actions.transition
    - I      -> `initial_state`
    - G      -> `goal`, optional free-form description; not required by the
                intent-filtering pipelines (they adjudicate Accept/Reject/
                Clarify, not full task planning), kept for completeness and
                possible future planning-quality experiments
    - tau    -> `instruction`, the natural-language command this problem was
                built for
    """

    ontology: Ontology
    initial_state: WorldState
    instruction: str
    goal: str | None = None

    @property
    def available_actions(self) -> tuple[ActionType, ...]:
        return tuple(ActionType)
