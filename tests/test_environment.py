"""Phase 1 unit tests: ontology loading, transition function, derived
propositions, and safety rule base schema/coverage. No LLM calls, no LTL
semantics (that's Phase 2) - pure symbolic environment.
"""

from pathlib import Path

import pytest

from intent_filter.environment import (
    Action,
    ActionType,
    apply_sequence,
    derived_propositions,
    initial_state,
    load_ontology,
    load_safety_rules,
    transition,
)
from intent_filter.environment.actions import InvalidActionError

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = REPO_ROOT / "config" / "environment_ontology.yaml"
SAFETY_RULES_PATH = REPO_ROOT / "config" / "safety_rules.yaml"

REQUIRED_RULE_CATEGORIES = {"sharp", "dangerous", "private_item", "child_zone"}


@pytest.fixture(scope="module")
def ontology():
    return load_ontology(ONTOLOGY_PATH)


# --- Ontology -----------------------------------------------------------------


def test_ontology_loads_expected_rooms_and_objects(ontology):
    assert set(ontology.rooms) == {"kitchen", "bedroom", "child_room", "bathroom", "garage"}
    assert set(ontology.objects) == {"knife", "medication", "laptop", "toy", "heavy_box"}
    assert set(ontology.roles) == {"owner", "child", "guest"}


def test_room_tags(ontology):
    assert ontology.room("child_room").has_tag("child_zone")
    assert ontology.room("child_room").has_tag("private")
    assert ontology.room("garage").has_tag("restricted")
    assert not ontology.room("kitchen").tags


def test_object_properties(ontology):
    assert ontology.obj("knife").has_property("sharp")
    assert ontology.obj("knife").has_property("dangerous")
    assert ontology.obj("medication").has_property("dangerous")
    assert ontology.obj("laptop").has_property("private_item")
    assert not ontology.obj("toy").properties


def test_unknown_room_and_object_raise(ontology):
    with pytest.raises(KeyError):
        ontology.room("attic")
    with pytest.raises(KeyError):
        ontology.obj("sword")


# --- Transition function --------------------------------------------------------


def test_move_updates_agent_room(ontology):
    state = initial_state(ontology)
    new_state = transition(state, Action(ActionType.MOVE, "child_room"), ontology)
    assert new_state.agent_room == "child_room"
    assert state.agent_room != "child_room"  # original state untouched (immutability)


def test_move_to_unknown_room_raises(ontology):
    state = initial_state(ontology)
    with pytest.raises(InvalidActionError):
        transition(state, Action(ActionType.MOVE, "attic"), ontology)


def test_pick_up_requires_object_in_current_room(ontology):
    state = initial_state(ontology)  # agent starts in kitchen; knife defaults to kitchen
    assert state.object_locations["knife"] == "kitchen"

    # medication defaults to bathroom, agent is in kitchen -> cannot pick up
    with pytest.raises(InvalidActionError):
        transition(state, Action(ActionType.PICK_UP, "medication"), ontology)

    picked_up = transition(state, Action(ActionType.PICK_UP, "knife"), ontology)
    assert "knife" in picked_up.held_objects


def test_pick_up_twice_raises(ontology):
    state = initial_state(ontology)
    state = transition(state, Action(ActionType.PICK_UP, "knife"), ontology)
    with pytest.raises(InvalidActionError):
        transition(state, Action(ActionType.PICK_UP, "knife"), ontology)


def test_held_object_travels_with_agent(ontology):
    state = initial_state(ontology)
    state = transition(state, Action(ActionType.PICK_UP, "knife"), ontology)
    state = transition(state, Action(ActionType.MOVE, "child_room"), ontology)
    assert state.object_locations["knife"] == "child_room"


def test_put_down_requires_holding_object(ontology):
    state = initial_state(ontology)
    with pytest.raises(InvalidActionError):
        transition(state, Action(ActionType.PUT_DOWN, "knife"), ontology)

    state = transition(state, Action(ActionType.PICK_UP, "knife"), ontology)
    state = transition(state, Action(ActionType.PUT_DOWN, "knife"), ontology)
    assert "knife" not in state.held_objects
    assert state.object_locations["knife"] == state.agent_room


def test_door_stove_alarm_toggles(ontology):
    state = initial_state(ontology)
    state = transition(state, Action(ActionType.UNLOCK_DOOR), ontology)
    assert state.door_locked is False
    state = transition(state, Action(ActionType.LOCK_DOOR), ontology)
    assert state.door_locked is True
    state = transition(state, Action(ActionType.TURN_ON_STOVE), ontology)
    assert state.stove_on is True
    state = transition(state, Action(ActionType.ARM_ALARM), ontology)
    assert state.alarm_on is True
    state = transition(state, Action(ActionType.LEAVE_HOME), ontology)
    assert state.owner_home is False


def test_apply_sequence_returns_full_trajectory(ontology):
    state = initial_state(ontology)
    actions = [
        Action(ActionType.PICK_UP, "knife"),
        Action(ActionType.MOVE, "child_room"),
    ]
    trajectory = apply_sequence(state, actions, ontology)
    assert len(trajectory) == 3
    assert trajectory[0] == state
    assert trajectory[-1].agent_room == "child_room"
    assert "knife" in trajectory[-1].held_objects


# --- Derived propositions --------------------------------------------------------


def test_derived_propositions_knife_in_child_room(ontology):
    state = initial_state(ontology)
    state = transition(state, Action(ActionType.PICK_UP, "knife"), ontology)
    state = transition(state, Action(ActionType.MOVE, "child_room"), ontology)

    aps = derived_propositions(state, ontology)
    assert aps["agent_at(child_room)"] is True
    assert aps["has_object(knife)"] is True
    assert aps["holds_sharp_item"] is True
    assert aps["holds_dangerous_item"] is True
    assert aps["at_child_zone"] is True
    assert aps["holds_private_item"] is False


def test_derived_propositions_issued_by_role(ontology):
    state = initial_state(ontology, issuing_role="guest")
    aps = derived_propositions(state, ontology)
    assert aps["issued_by(guest)"] is True
    assert aps["issued_by(owner)"] is False
    assert aps["issued_by(child)"] is False


def test_derived_propositions_restricted_room(ontology):
    state = initial_state(ontology)
    state = transition(state, Action(ActionType.MOVE, "garage"), ontology)
    aps = derived_propositions(state, ontology)
    assert aps["at_restricted_room"] is True
    assert aps["at_child_zone"] is False


# --- Safety rule base -------------------------------------------------------------


def test_safety_rules_load_and_have_required_fields():
    rule_base = load_safety_rules(SAFETY_RULES_PATH)
    assert len(rule_base) >= 8
    for rule in rule_base:
        assert rule.id
        assert rule.description
        assert rule.severity in {"critical", "high", "medium", "low"}
        assert rule.ltl.strip()


def test_safety_rule_ids_are_unique():
    rule_base = load_safety_rules(SAFETY_RULES_PATH)
    ids = [rule.id for rule in rule_base]
    assert len(ids) == len(set(ids))


def test_safety_rules_cover_required_categories():
    rule_base = load_safety_rules(SAFETY_RULES_PATH)
    assert REQUIRED_RULE_CATEGORIES.issubset(rule_base.categories)


def test_safety_rules_have_worked_examples():
    rule_base = load_safety_rules(SAFETY_RULES_PATH)
    for rule in rule_base:
        assert rule.violating_example, f"{rule.id} missing a violating_example"
        assert rule.safe_example, f"{rule.id} missing a safe_example"
