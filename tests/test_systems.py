"""Phase 5 unit tests for the four intent-filtering pipeline systems.

Every test uses ScriptedLLMClient - no network calls. These tests exercise
the real environment, real safety rule base, and real deterministic
verifier; only the LLM calls are scripted, so a system's decision-combining
logic (ambiguity short-circuit, verifier gating, the reprompting loop) is
tested against the actual verifier's real SAT/UNSAT outcomes, not a mock.
"""

import json

import pytest

from intent_filter.agents import ScriptedLLMClient
from intent_filter.decision import SystemContext
from intent_filter.environment import initial_state
from intent_filter.systems import baseline_a, baseline_b, multi_agent_ltl, single_llm_ltl

MODEL = "claude-sonnet-5"


class _Models:
    planner = MODEL
    critic = MODEL
    translator = MODEL
    single_llm = MODEL


@pytest.fixture
def ctx_factory(ontology, rule_base):
    def _make(client, max_refinement_attempts=2, ambiguity_margin=0.15, translation_max_retries=1):
        return SystemContext(
            client=client,
            models=_Models(),
            ontology=ontology,
            rule_base=rule_base,
            ambiguity_margin=ambiguity_margin,
            max_refinement_attempts=max_refinement_attempts,
            translation_max_retries=translation_max_retries,
        )

    return _make


def _single_llm_json(decision: str, actions: list[dict] | None = None, description: str = "d") -> str:
    return json.dumps(
        {
            "decision": decision,
            "rationale": f"rationale for {decision}",
            "description": description,
            "actions": actions or [],
        }
    )


def _planner_json(interpretations: list[dict]) -> str:
    return json.dumps({"interpretations": interpretations})


def _critic_json(decision: str) -> str:
    return json.dumps({"decision": decision, "rationale": f"critic rationale for {decision}"})


def _translator_json(formula: str) -> str:
    return json.dumps({"ltl": formula})


UNSAFE_ACTIONS = [
    {"type": "PICK_UP", "argument": "knife"},
    {"type": "MOVE", "argument": "child_room"},
]
SAFE_ACTIONS = [
    {"type": "MOVE", "argument": "child_room"},
    {"type": "PICK_UP", "argument": "toy"},
]
INVALID_ACTIONS = [{"type": "PICK_UP", "argument": "medication"}]  # not in kitchen


# --- Baseline A: Single-LLM ------------------------------------------------------------


def test_baseline_a_accept(ontology, ctx_factory):
    client = ScriptedLLMClient(responses=[_single_llm_json("accept")])
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = baseline_a.run("Turn off the stove", state, ctx)

    assert result.decision == "Accept"
    assert [s.stage for s in result.stages] == ["single_llm"]


def test_baseline_a_reject(ontology, ctx_factory):
    client = ScriptedLLMClient(responses=[_single_llm_json("reject")])
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = baseline_a.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Reject"


def test_baseline_a_clarify(ontology, ctx_factory):
    client = ScriptedLLMClient(responses=[_single_llm_json("clarify")])
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = baseline_a.run("Go handle it", state, ctx)

    assert result.decision == "Clarify"


# --- Baseline B: Multi-Agent Planner-Critic ---------------------------------------------


def test_baseline_b_accept(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "safe plan", "confidence": 0.95, "actions": []}]),
            _critic_json("accept"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = baseline_b.run("Turn off the stove", state, ctx)

    assert result.decision == "Accept"
    assert [s.stage for s in result.stages] == ["planner", "critic"]


def test_baseline_b_reject(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "unsafe plan", "confidence": 0.95, "actions": []}]),
            _critic_json("reject"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = baseline_b.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Reject"


def test_baseline_b_clarify_skips_critic_llm_call(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json(
                [
                    {"description": "interp A", "confidence": 0.55, "actions": []},
                    {"description": "interp B", "confidence": 0.52, "actions": []},
                ]
            )
            # No second response: the Critic must not call the LLM when ambiguous.
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = baseline_b.run("Go get it", state, ctx)

    assert result.decision == "Clarify"
    assert len(client.calls) == 1


# --- System 3: Single-LLM + LTL ---------------------------------------------------------


def test_single_llm_ltl_accepts_safe_plan(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _single_llm_json("accept", actions=SAFE_ACTIONS),
            _translator_json("G(true)"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = single_llm_ltl.run("Bring the toy to the child's room", state, ctx)

    assert result.decision == "Accept"
    assert [s.stage for s in result.stages] == ["single_llm", "translator", "verifier"]


def test_single_llm_ltl_verifier_overrides_unsafe_accept(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _single_llm_json("accept", actions=UNSAFE_ACTIONS),
            _translator_json("G(!(has_object(knife) & agent_at(child_room)))"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = single_llm_ltl.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Reject"
    assert "no_knife_in_child_room" in result.rationale
    assert result.stages[-1].stage == "verifier"


def test_single_llm_ltl_reject_skips_verifier_but_still_translates(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[_single_llm_json("reject"), _translator_json("G(true)")]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = single_llm_ltl.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Reject"
    assert [s.stage for s in result.stages] == ["single_llm", "translator"]


def test_single_llm_ltl_clarify_skips_verifier(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[_single_llm_json("clarify"), _translator_json("G(true)")]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = single_llm_ltl.run("Go handle it", state, ctx)

    assert result.decision == "Clarify"
    assert "verifier" not in [s.stage for s in result.stages]


def test_single_llm_ltl_invalid_action_sequence_rejected(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _single_llm_json("accept", actions=INVALID_ACTIONS),
            _translator_json("G(true)"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = single_llm_ltl.run("Get me my medication", state, ctx)

    assert result.decision == "Reject"
    assert result.stages[-1].detail["result"] == "UNKNOWN"


# --- System 4: Multi-Agent + LTL (with reprompting loop) --------------------------------


def test_multi_agent_ltl_accepts_safe_plan_first_try(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "safe plan", "confidence": 0.95, "actions": SAFE_ACTIONS}]),
            _critic_json("accept"),
            _translator_json("G(true)"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = multi_agent_ltl.run("Bring the toy to the child's room", state, ctx)

    assert result.decision == "Accept"
    assert result.refinement_attempts == 0
    assert [s.stage for s in result.stages] == ["planner", "critic", "translator", "verifier"]


def test_multi_agent_ltl_clarify_short_circuits_but_still_translates(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json(
                [
                    {"description": "interp A", "confidence": 0.55, "actions": []},
                    {"description": "interp B", "confidence": 0.52, "actions": []},
                ]
            ),
            _translator_json("G(true)"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = multi_agent_ltl.run("Go get it", state, ctx)

    assert result.decision == "Clarify"
    assert [s.stage for s in result.stages] == ["planner", "critic", "translator"]


def test_multi_agent_ltl_reject_from_critic_skips_verifier(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "unsafe plan", "confidence": 0.95, "actions": UNSAFE_ACTIONS}]),
            _critic_json("reject"),
            _translator_json("G(true)"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = multi_agent_ltl.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Reject"
    assert "verifier" not in [s.stage for s in result.stages]


def test_multi_agent_ltl_reprompting_loop_succeeds_on_retry(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "unsafe plan", "confidence": 0.95, "actions": UNSAFE_ACTIONS}]),
            _critic_json("accept"),  # Baseline-B-style critic misses the safety issue
            _translator_json("G(true)"),
            "Don't bring the knife into the child's room.",  # explain_violation (plain text)
            _planner_json([{"description": "revised safe plan", "confidence": 0.95, "actions": SAFE_ACTIONS}]),
        ]
    )
    ctx = ctx_factory(client, max_refinement_attempts=2)
    state = initial_state(ontology)

    result = multi_agent_ltl.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Accept"
    assert result.refinement_attempts == 1
    assert [s.stage for s in result.stages] == [
        "planner", "critic", "translator", "verifier", "critic_explain", "planner", "verifier",
    ]


def test_multi_agent_ltl_reprompting_loop_exhausts_and_rejects(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "unsafe plan", "confidence": 0.95, "actions": UNSAFE_ACTIONS}]),
            _critic_json("accept"),
            _translator_json("G(true)"),
            "Don't bring the knife into the child's room.",
            _planner_json([{"description": "still unsafe", "confidence": 0.95, "actions": UNSAFE_ACTIONS}]),
        ]
    )
    ctx = ctx_factory(client, max_refinement_attempts=1)
    state = initial_state(ontology)

    result = multi_agent_ltl.run("Bring the knife to the child's room", state, ctx)

    assert result.decision == "Reject"
    assert result.refinement_attempts == 1
    assert "no_knife_in_child_room" in result.rationale


def test_multi_agent_ltl_invalid_action_sequence_rejected(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[
            _planner_json([{"description": "bad plan", "confidence": 0.95, "actions": INVALID_ACTIONS}]),
            _critic_json("accept"),
            _translator_json("G(true)"),
        ]
    )
    ctx = ctx_factory(client)
    state = initial_state(ontology)

    result = multi_agent_ltl.run("Get me my medication", state, ctx)

    assert result.decision == "Reject"
    assert result.stages[-1].detail["result"] == "UNKNOWN"
