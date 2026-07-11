"""Shared prompt-construction helpers for the Planner, Critic, and Translator agents.

Kept separate from the agents themselves so the exact text sent to the LLM
is easy to inspect and quote in the research write-up (viva). All three
agents describe the environment dynamically from the ontology/rule base
rather than hardcoding room/object names, so the prompts stay correct if
config/environment_ontology.yaml or config/safety_rules.yaml change.
"""

from __future__ import annotations

from intent_filter.environment.actions import ActionType
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.rules import SafetyRuleBase
from intent_filter.environment.state import GENERIC_PROPOSITIONS

# Which argument kind (if any) each action type takes, for describing the
# schema to the Planner without hardcoding a second copy of ActionType.
_ROOM_ARG_ACTIONS = {ActionType.MOVE}
_OBJECT_ARG_ACTIONS = {ActionType.PICK_UP, ActionType.PUT_DOWN}


def describe_ontology(ontology: Ontology) -> str:
    """Render the environment ontology as compact text for an agent's system prompt."""
    lines = ["Rooms:"]
    for room in ontology.rooms.values():
        tags = ", ".join(sorted(room.tags)) or "none"
        lines.append(f"  - {room.name} (tags: {tags})")

    lines.append("Objects:")
    for obj in ontology.objects.values():
        props = ", ".join(sorted(obj.properties)) or "none"
        lines.append(f"  - {obj.name} (properties: {props}, default room: {obj.default_room})")

    lines.append(f"Roles: {', '.join(ontology.roles)}")
    lines.append(f"World variables: {', '.join(ontology.world_variables)}")
    return "\n".join(lines)


def describe_action_schema() -> str:
    """Render the fixed action schema (ActionType) for the Planner's system prompt."""
    lines = ["Available actions (JSON \"type\" field -> required \"argument\"):"]
    for action_type in ActionType:
        if action_type in _ROOM_ARG_ACTIONS:
            arg_desc = "argument: a room name"
        elif action_type in _OBJECT_ARG_ACTIONS:
            arg_desc = "argument: an object name"
        else:
            arg_desc = "argument: null"
        lines.append(f"  - {action_type.name} ({arg_desc})")
    return "\n".join(lines)


def describe_safety_rules(rule_base: SafetyRuleBase) -> str:
    """Render human-readable rule descriptions (no LTL) for the Critic's system prompt.

    Baseline B (Multi-Agent Planner-Critic, no verifier) relies on the Critic
    LLM's judgement of these descriptions alone; the LTL-augmented systems
    additionally check the formal `ltl` field via the deterministic verifier.
    """
    lines = []
    for rule in rule_base:
        lines.append(f"  - [{rule.severity}] {rule.description}")
    return "\n".join(lines)


def describe_atom_vocabulary(ontology: Ontology) -> str:
    """Render the full atomic-proposition vocabulary for the Translator's system prompt.

    Mirrors exactly what intent_filter.environment.state.derived_propositions
    produces for a given ontology, so a formula using only these names is
    guaranteed to be evaluable by the verifier.
    """
    lines = ["Grounded atoms:"]
    for room_name in ontology.rooms:
        lines.append(f"  - agent_at({room_name})")
    for obj_name in ontology.objects:
        lines.append(f"  - has_object({obj_name})")
    for role_name in ontology.roles:
        lines.append(f"  - issued_by({role_name})")

    lines.append("Generic atoms:")
    for prop in GENERIC_PROPOSITIONS:
        lines.append(f"  - {prop}")

    return "\n".join(lines)
