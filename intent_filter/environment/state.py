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

# The generic (non-grounded) atomic propositions derived_propositions() always
# emits, i.e. every AP name it produces that ISN'T agent_at(<room>),
# has_object(<object>), or issued_by(<role>). Exposed as a constant so the
# NL->LTL translator's prompt (intent_filter/agents/prompts.py) can list the
# full atom vocabulary without duplicating this list by hand.
GENERIC_PROPOSITIONS: tuple[str, ...] = (
    "holds_sharp_item",
    "holds_dangerous_item",
    "holds_private_item",
    "holds_fragile_item",
    "holds_heavy_item",
    "at_child_zone",
    "at_restricted_room",
    "at_private_room",
    "door_locked",
    "alarm_on",
    "stove_on",
    "owner_home",
    "child_gate_locked",
    "supervisor_present",
    "medicine_cabinet_locked",
    "caretaker_present",
    "window_locked",
)


@dataclass(frozen=True)
class WorldState:
    agent_room: str
    held_objects: frozenset[str] = field(default_factory=frozenset)
    object_locations: dict[str, str] = field(default_factory=dict)
    door_locked: bool = True
    alarm_on: bool = False
    stove_on: bool = False
    owner_home: bool = True
    # Added for the child-gate misdirection experiment (see
    # docs/methodology.md "Older-model comparison" follow-up / the
    # conversation record): supervisor_present mirrors owner_home's role as
    # a claim about a *person's* status (not a directly-observable device
    # state), paired with child_gate_locked (mirrors door_locked - a
    # device state that's safe by default) via the new
    # child_gate_locked_when_unsupervised rule. Not used by any of the
    # original 8 rules or the Phase 8 dataset/results - purely additive,
    # defaults preserve every existing trajectory's semantics unchanged.
    child_gate_locked: bool = True
    supervisor_present: bool = False
    # Added for the large-scale generalization run (docs/methodology.md
    # "Large-scale temporal-inconsistency generalization"): two more
    # device-state/person-status pairs with the same shape, for object
    # diversity beyond door_locked/owner_home and
    # child_gate_locked/supervisor_present. medicine_cabinet_locked and
    # window_locked mirror door_locked (safe by default); caretaker_present
    # mirrors owner_home/supervisor_present (a person-status claim).
    # supervisor_present is reused for window_locked's rule rather than
    # adding a third person-status variable. Purely additive, same
    # defaults-preserve-existing-semantics guarantee as above.
    medicine_cabinet_locked: bool = True
    caretaker_present: bool = False
    window_locked: bool = True
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
        child_gate_locked=True,
        supervisor_present=False,
        medicine_cabinet_locked=True,
        caretaker_present=False,
        window_locked=True,
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
    aps["child_gate_locked"] = state.child_gate_locked
    aps["supervisor_present"] = state.supervisor_present
    aps["medicine_cabinet_locked"] = state.medicine_cabinet_locked
    aps["caretaker_present"] = state.caretaker_present
    aps["window_locked"] = state.window_locked

    return aps
