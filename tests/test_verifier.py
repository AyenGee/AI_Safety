"""Phase 2 unit tests for the deterministic LTLf verifier.

All formulas/traces here are hand-written or built from the real
environment + real safety rule base - no LLM calls, matching the Phase 2
scope (verifier only; NL->LTL translation is Phase 4).
"""

from pathlib import Path

import pytest

from intent_filter.environment import (
    Action,
    ActionType,
    apply_sequence,
    initial_state,
    load_ontology,
    load_safety_rules,
)
from intent_filter.verifier import (
    LTLFormulaError,
    VerificationResult,
    build_atom_map,
    check_rule_base,
    overall_result,
    parse_formula,
    sanitize_formula,
    verify_state_trajectory,
    verify_trace,
    violated_rules,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = REPO_ROOT / "config" / "environment_ontology.yaml"
SAFETY_RULES_PATH = REPO_ROOT / "config" / "safety_rules.yaml"


@pytest.fixture(scope="module")
def ontology():
    return load_ontology(ONTOLOGY_PATH)


@pytest.fixture(scope="module")
def rule_base():
    return load_safety_rules(SAFETY_RULES_PATH)


@pytest.fixture(scope="module")
def atom_map(ontology):
    return build_atom_map(ontology)


# --- Formula parsing --------------------------------------------------------------


def test_parse_formula_valid():
    formula = parse_formula("G(!(a & b))")
    assert formula is not None


def test_parse_formula_raises_on_malformed():
    with pytest.raises(LTLFormulaError):
        parse_formula("G(&&&broken")


def test_all_safety_rules_are_syntactically_valid_ltlf(rule_base, atom_map):
    """Every rule in the real rule base must sanitize + parse cleanly."""
    for rule in rule_base:
        sanitized = sanitize_formula(rule.ltl, atom_map)
        formula = parse_formula(sanitized)  # raises LTLFormulaError on failure
        assert formula is not None


def test_atom_sanitization_leaves_generic_atoms_untouched(atom_map):
    # Generic derived atoms (no parens) must not appear as sanitization keys.
    assert "holds_sharp_item" not in atom_map
    assert "door_locked" not in atom_map
    assert sanitize_formula("G(!(holds_sharp_item & at_child_zone))", atom_map) == (
        "G(!(holds_sharp_item & at_child_zone))"
    )


# --- verify_trace on hand-written formulas/traces ----------------------------------


def test_verify_trace_sat():
    formula = "G(!(a & b))"
    trace = [{"a": True, "b": False}, {"a": False, "b": True}]
    outcome = verify_trace(formula, trace)
    assert outcome.result is VerificationResult.SAT


def test_verify_trace_unsat_with_violation_details():
    formula = "G(!(a & b))"
    trace = [{"a": True, "b": False}, {"a": True, "b": True}]
    outcome = verify_trace(formula, trace)
    assert outcome.result is VerificationResult.UNSAT
    assert outcome.violating_step == 1
    assert outcome.violating_atoms == ("a", "b")
    assert outcome.explanation is not None


def test_verify_trace_unknown_on_malformed_formula_does_not_raise():
    outcome = verify_trace("G(&&&broken", [{"a": True}])
    assert outcome.result is VerificationResult.UNKNOWN
    assert outcome.explanation is not None


def test_verify_trace_vacuously_sat_on_empty_trace():
    outcome = verify_trace("G(!(a & b))", [])
    assert outcome.result is VerificationResult.SAT


# --- verify_state_trajectory / check_rule_base on the real environment ------------


def test_knife_in_child_room_violates_multiple_rules(ontology, rule_base):
    state = initial_state(ontology)
    trajectory = apply_sequence(
        state,
        [Action(ActionType.PICK_UP, "knife"), Action(ActionType.MOVE, "child_room")],
        ontology,
    )

    outcomes = check_rule_base(rule_base, trajectory, ontology)
    assert overall_result(outcomes) is VerificationResult.UNSAT

    # The specific rule, plus the generic sharp/dangerous-in-child-zone rules,
    # should all fire - the knife is both sharp and dangerous.
    assert set(violated_rules(outcomes)) == {
        "no_knife_in_child_room",
        "no_sharp_items_in_child_zone",
        "no_dangerous_items_in_child_zone",
    }

    knife_outcome = outcomes["no_knife_in_child_room"]
    assert knife_outcome.violating_step == 2  # index 0=initial, 1=after pickup, 2=after move
    assert "has_object(knife)" in knife_outcome.violating_atoms
    assert "agent_at(child_room)" in knife_outcome.violating_atoms


def test_knife_kept_in_kitchen_is_safe(ontology, rule_base):
    state = initial_state(ontology)
    trajectory = apply_sequence(state, [Action(ActionType.PICK_UP, "knife")], ontology)

    outcomes = check_rule_base(rule_base, trajectory, ontology)
    assert overall_result(outcomes) is VerificationResult.SAT
    assert violated_rules(outcomes) == ()


def test_unlocking_door_while_owner_away_violates_rule(ontology, rule_base):
    state = initial_state(ontology)
    trajectory = apply_sequence(
        state,
        [Action(ActionType.LEAVE_HOME), Action(ActionType.UNLOCK_DOOR)],
        ontology,
    )
    rule = rule_base.by_id("lock_door_when_owner_away")
    outcome = verify_state_trajectory(rule.ltl, trajectory, ontology, rule_id=rule.id)
    assert outcome.result is VerificationResult.UNSAT


def test_locking_door_while_owner_away_is_safe(ontology, rule_base):
    state = initial_state(ontology)
    trajectory = apply_sequence(
        state,
        [Action(ActionType.LEAVE_HOME), Action(ActionType.LOCK_DOOR)],
        ontology,
    )
    rule = rule_base.by_id("lock_door_when_owner_away")
    outcome = verify_state_trajectory(rule.ltl, trajectory, ontology, rule_id=rule.id)
    assert outcome.result is VerificationResult.SAT


def test_guest_fetching_laptop_violates_private_item_rule(ontology, rule_base):
    state = initial_state(ontology, issuing_role="guest")
    trajectory = apply_sequence(
        state,
        [Action(ActionType.MOVE, "bedroom"), Action(ActionType.PICK_UP, "laptop")],
        ontology,
    )
    rule = rule_base.by_id("no_private_item_access_by_guest")
    outcome = verify_state_trajectory(rule.ltl, trajectory, ontology, rule_id=rule.id)
    assert outcome.result is VerificationResult.UNSAT


def test_owner_fetching_laptop_is_safe(ontology, rule_base):
    state = initial_state(ontology, issuing_role="owner")
    trajectory = apply_sequence(
        state,
        [Action(ActionType.MOVE, "bedroom"), Action(ActionType.PICK_UP, "laptop")],
        ontology,
    )
    rule = rule_base.by_id("no_private_item_access_by_guest")
    outcome = verify_state_trajectory(rule.ltl, trajectory, ontology, rule_id=rule.id)
    assert outcome.result is VerificationResult.SAT
