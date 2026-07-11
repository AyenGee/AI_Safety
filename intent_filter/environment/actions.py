"""Actions and the deterministic transition function `T: State x Action -> State`.

Kept intentionally small and explicit (one branch per action type) rather than
a generic STRIPS-style precondition/effect engine, since the domain is small
and a research audience (viva) needs to be able to read this top to bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from intent_filter.environment.ontology import Ontology
from intent_filter.environment.state import WorldState


class ActionType(Enum):
    MOVE = auto()
    PICK_UP = auto()
    PUT_DOWN = auto()
    LOCK_DOOR = auto()
    UNLOCK_DOOR = auto()
    TURN_ON_STOVE = auto()
    TURN_OFF_STOVE = auto()
    ARM_ALARM = auto()
    DISARM_ALARM = auto()
    LEAVE_HOME = auto()
    RETURN_HOME = auto()


# Action types that take a room name as their argument.
_ROOM_ARGUMENT_ACTIONS = {ActionType.MOVE}
# Action types that take an object name as their argument.
_OBJECT_ARGUMENT_ACTIONS = {ActionType.PICK_UP, ActionType.PUT_DOWN}


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    argument: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        if self.argument is not None:
            return f"{self.action_type.name}({self.argument})"
        return self.action_type.name


class InvalidActionError(Exception):
    """Raised when an action violates a transition-function precondition."""


def transition(state: WorldState, action: Action, ontology: Ontology) -> WorldState:
    """Apply `action` to `state`, returning the resulting state.

    Raises InvalidActionError if `action`'s preconditions are not met given
    `state` and `ontology`. This function is deliberately unaware of the
    safety rule base: precondition checks here are physical/logical
    plausibility only (e.g. can't pick up an object that isn't in the room),
    not safety policy, which is the verifier's job.
    """
    action_type = action.action_type

    if action_type in _ROOM_ARGUMENT_ACTIONS:
        room = action.argument
        if room not in ontology.rooms:
            raise InvalidActionError(f"Cannot move to unknown room {room!r}")
        new_locations = dict(state.object_locations)
        for obj_name in state.held_objects:
            new_locations[obj_name] = room
        return state.with_updates(agent_room=room, object_locations=new_locations)

    if action_type in _OBJECT_ARGUMENT_ACTIONS:
        obj = action.argument
        if obj not in ontology.objects:
            raise InvalidActionError(f"Unknown object {obj!r}")

        if action_type is ActionType.PICK_UP:
            if obj in state.held_objects:
                raise InvalidActionError(f"Already holding {obj!r}")
            if state.object_locations.get(obj) != state.agent_room:
                raise InvalidActionError(
                    f"Cannot pick up {obj!r}: not present in current room {state.agent_room!r}"
                )
            return state.with_updates(held_objects=state.held_objects | {obj})

        # PUT_DOWN
        if obj not in state.held_objects:
            raise InvalidActionError(f"Cannot put down {obj!r}: not currently held")
        new_locations = dict(state.object_locations)
        new_locations[obj] = state.agent_room
        return state.with_updates(
            held_objects=state.held_objects - {obj}, object_locations=new_locations
        )

    if action_type is ActionType.LOCK_DOOR:
        return state.with_updates(door_locked=True)
    if action_type is ActionType.UNLOCK_DOOR:
        return state.with_updates(door_locked=False)
    if action_type is ActionType.TURN_ON_STOVE:
        return state.with_updates(stove_on=True)
    if action_type is ActionType.TURN_OFF_STOVE:
        return state.with_updates(stove_on=False)
    if action_type is ActionType.ARM_ALARM:
        return state.with_updates(alarm_on=True)
    if action_type is ActionType.DISARM_ALARM:
        return state.with_updates(alarm_on=False)
    if action_type is ActionType.LEAVE_HOME:
        return state.with_updates(owner_home=False)
    if action_type is ActionType.RETURN_HOME:
        return state.with_updates(owner_home=True)

    raise InvalidActionError(f"Unhandled action type: {action_type}")  # pragma: no cover


def apply_sequence(
    state: WorldState, actions: list[Action], ontology: Ontology
) -> list[WorldState]:
    """Apply a sequence of actions, returning the full trajectory including `state`.

    The returned list has length len(actions) + 1: trajectory[0] is the
    initial state and trajectory[i+1] is the state after actions[i]. This is
    the shape the LTL verifier (Phase 2) consumes when checking a candidate
    plan's resulting state trajectory against a safety formula.
    """
    trajectory = [state]
    current = state
    for action in actions:
        current = transition(current, action, ontology)
        trajectory.append(current)
    return trajectory
