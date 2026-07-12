"""Phase 5 unit tests for the shared decision-layer helpers (intent_filter/decision.py)."""

from intent_filter.decision import PipelineResult, StageLog, build_trajectory, summarize_violations
from intent_filter.environment import Action, ActionType, apply_sequence, initial_state
from intent_filter.verifier import check_rule_base


def test_build_trajectory_returns_states_for_valid_actions(ontology):
    state = initial_state(ontology)
    actions = (Action(ActionType.PICK_UP, "knife"), Action(ActionType.MOVE, "child_room"))

    trajectory = build_trajectory(state, actions, ontology)

    assert trajectory is not None
    assert len(trajectory) == 3
    assert trajectory[-1].agent_room == "child_room"


def test_build_trajectory_returns_none_for_invalid_actions(ontology):
    state = initial_state(ontology)  # agent starts in kitchen; medication defaults to bathroom
    actions = (Action(ActionType.PICK_UP, "medication"),)

    trajectory = build_trajectory(state, actions, ontology)

    assert trajectory is None


def test_summarize_violations_reports_unsat_rules(ontology, rule_base):
    state = initial_state(ontology)
    trajectory = apply_sequence(
        state,
        [Action(ActionType.PICK_UP, "knife"), Action(ActionType.MOVE, "child_room")],
        ontology,
    )
    outcomes = check_rule_base(rule_base, trajectory, ontology)

    summary = summarize_violations(outcomes)

    assert "no_knife_in_child_room" in summary
    assert "Rejected by formal verification" in summary


def test_summarize_violations_reports_no_violations(ontology, rule_base):
    state = initial_state(ontology)
    trajectory = apply_sequence(state, [Action(ActionType.PICK_UP, "knife")], ontology)
    outcomes = check_rule_base(rule_base, trajectory, ontology)

    summary = summarize_violations(outcomes)

    assert summary == "No safety rule violations found."


def test_pipeline_result_latency_by_stage_sums_repeated_stage_names():
    result = PipelineResult(
        decision="Reject",
        rationale="x",
        stages=(
            StageLog(stage="planner", detail={}, latency_seconds=0.1),
            StageLog(stage="verifier", detail={}, latency_seconds=0.2),
            StageLog(stage="planner", detail={}, latency_seconds=0.3),
        ),
        total_latency_seconds=0.6,
        refinement_attempts=1,
    )

    totals = result.latency_by_stage()

    assert totals["planner"] == 0.1 + 0.3
    assert totals["verifier"] == 0.2
