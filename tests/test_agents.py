"""Phase 4 unit tests for the Planner, Critic, and Translator agents.

Every test uses ScriptedLLMClient (intent_filter.agents.client) - no network
calls, no real Anthropic API access needed to run this suite.
"""

import json
from pathlib import Path

import pytest

from intent_filter.agents import (
    CriticError,
    PlannerError,
    ScriptedLLMClient,
    check_ambiguity,
    explain_violation,
    plan,
    review,
    strip_code_fences,
    template_translate,
    translate,
)
from intent_filter.agents.planner import PlannerInterpretation, PlannerOutput
from intent_filter.environment import (
    Action,
    ActionType,
    initial_state,
    load_ontology,
    load_safety_rules,
)
from intent_filter.verifier import VerificationOutcome, VerificationResult, parse_formula
from intent_filter.verifier.atoms import build_atom_map, sanitize_formula

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = REPO_ROOT / "config" / "environment_ontology.yaml"
SAFETY_RULES_PATH = REPO_ROOT / "config" / "safety_rules.yaml"

MODEL = "claude-sonnet-5"


@pytest.fixture(scope="module")
def ontology():
    return load_ontology(ONTOLOGY_PATH)


@pytest.fixture(scope="module")
def rule_base():
    return load_safety_rules(SAFETY_RULES_PATH)


def _planner_json(interpretations: list[dict]) -> str:
    return json.dumps({"interpretations": interpretations})


# --- Planner -------------------------------------------------------------------------


def test_plan_parses_well_formed_response(ontology):
    response = _planner_json(
        [
            {
                "description": "Bring the toy to the child's room",
                "confidence": 0.95,
                "actions": [
                    {"type": "MOVE", "argument": "child_room"},
                    {"type": "PICK_UP", "argument": "toy"},
                ],
            }
        ]
    )
    client = ScriptedLLMClient(responses=[response])
    state = initial_state(ontology)

    output = plan(client, MODEL, "Bring the toy to the child's room", state, ontology)

    assert len(output.interpretations) == 1
    assert output.top.confidence == 0.95
    assert output.top.actions == (
        Action(ActionType.MOVE, "child_room"),
        Action(ActionType.PICK_UP, "toy"),
    )
    assert len(client.calls) == 1


def test_plan_sorts_interpretations_by_confidence_descending(ontology):
    response = _planner_json(
        [
            {"description": "low", "confidence": 0.4, "actions": []},
            {"description": "high", "confidence": 0.6, "actions": []},
        ]
    )
    client = ScriptedLLMClient(responses=[response])
    state = initial_state(ontology)

    output = plan(client, MODEL, "Go get it", state, ontology)

    assert output.top.description == "high"
    assert output.interpretations[1].description == "low"


def test_plan_retries_on_malformed_json_then_succeeds(ontology):
    good_response = _planner_json([{"description": "ok", "confidence": 0.9, "actions": []}])
    client = ScriptedLLMClient(responses=["not json", good_response])
    state = initial_state(ontology)

    output = plan(client, MODEL, "Lock the door", state, ontology, max_retries=2)

    assert output.top.description == "ok"
    assert len(client.calls) == 2
    assert "could not be parsed" in client.calls[1]["user"]


def test_plan_raises_planner_error_after_exhausting_retries(ontology):
    client = ScriptedLLMClient(responses=["garbage", "still garbage"])
    state = initial_state(ontology)

    with pytest.raises(PlannerError):
        plan(client, MODEL, "Lock the door", state, ontology, max_retries=1)


def test_plan_rejects_unknown_action_type(ontology):
    bad_response = _planner_json(
        [{"description": "x", "confidence": 0.9, "actions": [{"type": "FLY", "argument": None}]}]
    )
    good_response = _planner_json([{"description": "y", "confidence": 0.9, "actions": []}])
    client = ScriptedLLMClient(responses=[bad_response, good_response])
    state = initial_state(ontology)

    output = plan(client, MODEL, "Do something", state, ontology, max_retries=1)
    assert output.top.description == "y"


# --- Critic: ambiguity detection ------------------------------------------------------


def _interp(description: str, confidence: float) -> PlannerInterpretation:
    return PlannerInterpretation(description=description, confidence=confidence, actions=())


def test_check_ambiguity_flags_close_confidences():
    output = PlannerOutput(
        interpretations=(_interp("a", 0.55), _interp("b", 0.50)), raw_response=""
    )
    is_ambiguous, margin = check_ambiguity(output, ambiguity_margin=0.15)
    assert is_ambiguous is True
    assert margin == pytest.approx(0.05)


def test_check_ambiguity_allows_clear_gap():
    output = PlannerOutput(
        interpretations=(_interp("a", 0.95), _interp("b", 0.30)), raw_response=""
    )
    is_ambiguous, margin = check_ambiguity(output, ambiguity_margin=0.15)
    assert is_ambiguous is False
    assert margin == pytest.approx(0.65)


def test_check_ambiguity_single_interpretation_not_ambiguous():
    output = PlannerOutput(interpretations=(_interp("a", 0.5),), raw_response="")
    is_ambiguous, margin = check_ambiguity(output, ambiguity_margin=0.15)
    assert is_ambiguous is False
    assert margin is None


# --- Critic: review ------------------------------------------------------------------


def test_review_returns_clarify_without_calling_llm_when_ambiguous(ontology, rule_base):
    output = PlannerOutput(
        interpretations=(_interp("a", 0.55), _interp("b", 0.52)), raw_response=""
    )
    client = ScriptedLLMClient(responses=[])  # would raise if called
    state = initial_state(ontology)

    result = review(
        client, MODEL, "Go get it", output, state, ontology, rule_base, ambiguity_margin=0.15
    )

    assert result.decision == "clarify"
    assert result.ambiguity_detected is True
    assert client.calls == []


def test_review_accepts_when_unambiguous_and_llm_accepts(ontology, rule_base):
    planner_output = PlannerOutput(interpretations=(_interp("safe plan", 0.95),), raw_response="")
    client = ScriptedLLMClient(
        responses=[json.dumps({"decision": "accept", "rationale": "Looks fine."})]
    )
    state = initial_state(ontology)

    result = review(
        client, MODEL, "Turn off the stove", planner_output, state, ontology, rule_base,
        ambiguity_margin=0.15,
    )

    assert result.decision == "accept"
    assert result.rationale == "Looks fine."
    assert result.ambiguity_detected is False


def test_review_retries_on_malformed_response(ontology, rule_base):
    planner_output = PlannerOutput(interpretations=(_interp("plan", 0.95),), raw_response="")
    client = ScriptedLLMClient(
        responses=["not json", json.dumps({"decision": "reject", "rationale": "Unsafe."})]
    )
    state = initial_state(ontology)

    result = review(
        client, MODEL, "Bring the knife to the child's room", planner_output, state, ontology,
        rule_base, ambiguity_margin=0.15, max_retries=1,
    )

    assert result.decision == "reject"
    assert len(client.calls) == 2


def test_review_raises_critic_error_after_exhausting_retries(ontology, rule_base):
    planner_output = PlannerOutput(interpretations=(_interp("plan", 0.95),), raw_response="")
    client = ScriptedLLMClient(responses=["garbage", "still garbage"])
    state = initial_state(ontology)

    with pytest.raises(CriticError):
        review(
            client, MODEL, "Do something", planner_output, state, ontology, rule_base,
            ambiguity_margin=0.15, max_retries=1,
        )


def test_explain_violation_sends_verifier_details_and_returns_text():
    outcome = VerificationOutcome(
        result=VerificationResult.UNSAT,
        formula="G(!(has_object(knife) & agent_at(child_room)))",
        rule_id="no_knife_in_child_room",
        violating_step=2,
        violating_atoms=("has_object(knife)", "agent_at(child_room)"),
        explanation="Violated at step 2: has_object(knife)=True, agent_at(child_room)=True",
    )
    client = ScriptedLLMClient(responses=["Don't bring the knife into the child's room."])

    feedback = explain_violation(client, MODEL, "Bring the knife to the child's room", outcome)

    assert feedback == "Don't bring the knife into the child's room."
    assert "has_object(knife)" in client.calls[0]["user"]
    assert "no_knife_in_child_room" not in client.calls[0]["user"]  # rule_id isn't echoed, formula is
    assert outcome.formula in client.calls[0]["user"]


# --- Translator ------------------------------------------------------------------------


def test_translate_succeeds_on_first_valid_response(ontology):
    client = ScriptedLLMClient(
        responses=[json.dumps({"ltl": "G(!(has_object(knife) & agent_at(child_room)))"})]
    )

    result = translate(client, MODEL, "Bring the knife to the child's room.", ontology)

    assert result.success is True
    assert result.attempts == 1
    assert result.used_fallback is False
    # Returned formula must itself be valid once sanitized, matching what
    # the verifier would do with a safety_rules.yaml rule.
    atom_map = build_atom_map(ontology)
    parse_formula(sanitize_formula(result.ltl_formula, atom_map))


def test_translate_retries_on_invalid_ltl_then_succeeds(ontology):
    client = ScriptedLLMClient(
        responses=[
            json.dumps({"ltl": "G(&&&broken"}),
            json.dumps({"ltl": "G(door_locked | owner_home)"}),
        ]
    )

    result = translate(client, MODEL, "Leave the door unlocked while I'm out.", ontology, max_retries=2)

    assert result.success is True
    assert result.attempts == 2
    assert result.used_fallback is False
    assert "invalid" in client.calls[1]["user"].lower()


def test_translate_falls_back_to_template_when_llm_exhausts_retries(ontology):
    client = ScriptedLLMClient(responses=["garbage", "still garbage"])

    result = translate(client, MODEL, "Bring the knife to the child's room.", ontology, max_retries=1)

    assert result.success is True
    assert result.used_fallback is True
    assert result.ltl_formula == "G(!(has_object(knife) & agent_at(child_room)))"


def test_translate_fails_when_llm_and_template_both_fail(ontology):
    client = ScriptedLLMClient(responses=["garbage", "still garbage"])

    result = translate(client, MODEL, "Go handle it.", ontology, max_retries=1)

    assert result.success is False
    assert result.ltl_formula is None


def test_template_translate_knife_child_room(ontology):
    formula = template_translate("Bring the knife to the child's room.", ontology)
    assert formula == "G(!(has_object(knife) & agent_at(child_room)))"


def test_template_translate_stove_bathroom(ontology):
    formula = template_translate("Turn on the stove while in the bathroom.", ontology)
    assert formula == "G(!(stove_on & agent_at(bathroom)))"


def test_template_translate_door_unlock(ontology):
    formula = template_translate("Leave the front door unlocked while I'm out.", ontology)
    assert formula == "G(door_locked | owner_home)"


def test_template_translate_returns_none_for_unmatched_text(ontology):
    assert template_translate("Go handle it.", ontology) is None


# --- Markdown code fence handling ------------------------------------------------------
# Regression coverage for a real failure observed against the live Anthropic API: despite
# every system prompt saying "no markdown fences", the model sometimes wraps its JSON
# response in ```json ... ``` anyway.


def test_strip_code_fences_removes_json_fence():
    fenced = '```json\n{"decision": "reject"}\n```'
    assert strip_code_fences(fenced) == '{"decision": "reject"}'


def test_strip_code_fences_removes_bare_fence():
    fenced = '```\n{"decision": "reject"}\n```'
    assert strip_code_fences(fenced) == '{"decision": "reject"}'


def test_strip_code_fences_leaves_unfenced_text_unchanged():
    assert strip_code_fences('{"decision": "reject"}') == '{"decision": "reject"}'


def test_plan_handles_fenced_response(ontology):
    fenced = "```json\n" + _planner_json([{"description": "ok", "confidence": 0.9, "actions": []}]) + "\n```"
    client = ScriptedLLMClient(responses=[fenced])
    state = initial_state(ontology)

    output = plan(client, MODEL, "Lock the door", state, ontology)

    assert output.top.description == "ok"
    assert len(client.calls) == 1  # parsed on the first attempt, no retry needed


def test_review_handles_fenced_response(ontology, rule_base):
    planner_output = PlannerOutput(interpretations=(_interp("plan", 0.95),), raw_response="")
    fenced = '```json\n{"decision": "reject", "rationale": "Unsafe."}\n```'
    client = ScriptedLLMClient(responses=[fenced])
    state = initial_state(ontology)

    result = review(
        client, MODEL, "Bring the knife to the child's room", planner_output, state, ontology,
        rule_base, ambiguity_margin=0.15,
    )

    assert result.decision == "reject"
    assert len(client.calls) == 1


def test_translate_handles_fenced_response(ontology):
    fenced = '```json\n{"ltl": "G(door_locked | owner_home)"}\n```'
    client = ScriptedLLMClient(responses=[fenced])

    result = translate(client, MODEL, "Leave the door unlocked while I'm out.", ontology)

    assert result.success is True
    assert result.attempts == 1
    assert result.ltl_formula == "G(door_locked | owner_home)"
