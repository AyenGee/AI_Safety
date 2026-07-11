"""World state and the derived atomic-proposition (AP) layer.

`WorldState` is the concrete state space `S` in the planning problem tuple.
`derived_propositions` computes the full AP vocabulary referenced by
config/safety_rules.yaml (both room/object-grounded atoms like
`agent_at(kitchen)` and category-level derived atoms like `holds_sharp_item`)
from a state + the ontology. The LTL verifier (Phase 2) evaluates formulas
against this AP dict rather than against raw state fields, so rules stay
declarative and don't need to know about dataclass internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from intent_filter.environment.ontology import Ontology


@dataclass(frozen=True)
class WorldState:
    agent_room: str
    held_objects: frozenset[str] = field(default_factory=frozenset)
    object_locations: dict[str, str] = field(default_factory=dict)
    door_locked: bool = True
    alarm_on: bool = False
    stove_on: bool = False
    owner_home: bool = True
    issuing_role: str = "owner"

    def with_updates(self, **changes) -> "WorldState":
        """Return a new WorldState with the given fields replaced (state is immutable)."""
        return replace(self, **changes)


def initial_state(ontology: Ontology, issuing_role: str = "owner") -> WorldState:
    """Build the default initial state: agent in the kitchen, objects at their default rooms."""
    object_locations = {
        obj.name: obj.default_room
        for obj in ontology.objects.values()
        if obj.default_room is not None
    }
    return WorldState(
        agent_room=next(iter(ontology.rooms), "kitchen"),
        held_objects=frozenset(),
        object_locations=object_locations,
        door_locked=True,
        alarm_on=False,
        stove_on=False,
        owner_home=True,
        issuing_role=issuing_role,
    )


def derived_propositions(state: WorldState, ontology: Ontology) -> dict[str, bool]:
    """Compute the full AP dict for `state`, keyed by the names used in safety_rules.yaml."""
    aps: dict[str, bool] = {}

    for room_name in ontology.rooms:
        aps[f"agent_at({room_name})"] = state.agent_room == room_name

    for obj_name in ontology.objects:
        aps[f"has_object({obj_name})"] = obj_name in state.held_objects

    for role_name in ontology.roles:
        aps[f"issued_by({role_name})"] = state.issuing_role == role_name

    held_objs = [ontology.obj(name) for name in state.held_objects]
    aps["holds_sharp_item"] = any(o.has_property("sharp") for o in held_objs)
    aps["holds_dangerous_item"] = any(o.has_property("dangerous") for o in held_objs)
    aps["holds_private_item"] = any(o.has_property("private_item") for o in held_objs)
    aps["holds_fragile_item"] = any(o.has_property("fragile") for o in held_objs)
    aps["holds_heavy_item"] = any(o.has_property("heavy") for o in held_objs)

    current_room = ontology.room(state.agent_room)
    aps["at_child_zone"] = current_room.has_tag("child_zone")
    aps["at_restricted_room"] = current_room.has_tag("restricted")
    aps["at_private_room"] = current_room.has_tag("private")

    aps["door_locked"] = state.door_locked
    aps["alarm_on"] = state.alarm_on
    aps["stove_on"] = state.stove_on
    aps["owner_home"] = state.owner_home

    return aps
