"""Symbolic household environment: ontology, state machine, and safety rules.

This is a lightweight custom simulator (plain dataclasses + a deterministic
transition function), not a 3D simulator. `backend.py` defines the
`SimulatorBackend` interface so a VirtualHome/AI2-THOR backend could be
plugged in later without changing any intent-filtering logic.
"""

from intent_filter.environment.actions import Action, ActionType, apply_sequence, transition
from intent_filter.environment.backend import SimulatorBackend, SymbolicSimulatorBackend
from intent_filter.environment.ontology import EnvObject, Ontology, Room, load_ontology
from intent_filter.environment.problem import PlanningProblem
from intent_filter.environment.rules import SafetyRule, SafetyRuleBase, load_safety_rules
from intent_filter.environment.state import WorldState, derived_propositions, initial_state

__all__ = [
    "Action",
    "ActionType",
    "apply_sequence",
    "transition",
    "SimulatorBackend",
    "SymbolicSimulatorBackend",
    "EnvObject",
    "Ontology",
    "Room",
    "load_ontology",
    "PlanningProblem",
    "SafetyRule",
    "SafetyRuleBase",
    "load_safety_rules",
    "WorldState",
    "derived_propositions",
    "initial_state",
]
