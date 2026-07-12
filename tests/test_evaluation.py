"""Phase 6 unit tests for the evaluation harness: metrics, statistics, and
the run orchestrator. No network calls - the orchestrator tests use
ScriptedLLMClient and tiny synthetic dataset examples, not the full seed
dataset or the real API.
"""

import json

import pytest

from intent_filter.agents import ScriptedLLMClient
from intent_filter.dataset import InstructionExample, SceneContext
from intent_filter.decision import SystemContext
from intent_filter.environment import initial_state
from intent_filter.evaluation import (
    build_latency_comparison,
    build_pairwise_mcnemar,
    build_system_report,
    compare_latencies,
    compute_confusion_counts,
    compute_system_metrics,
    confusion_matrix,
    mcnemar_test,
    mean_confidence_interval,
    run_evaluation,
    run_example,
)
from intent_filter.evaluation.metrics import (
    category_breakdown,
    clarification_accuracy,
    error_rate,
    latency_summary,
    overall_accuracy,
)
from intent_filter.evaluation.types import RunRecord
from intent_filter.systems import baseline_a, baseline_b

MODEL = "claude-sonnet-5"


class _Models:
    planner = MODEL
    critic = MODEL
    translator = MODEL
    single_llm = MODEL


def _rec(system="s", example_id="e1", repeat_index=0, category="legitimate", gold="Accept", predicted="Accept", error=None, latency=1.0, stages=None):
    correct = (predicted == "Clarify") if category == "ambiguous" else (predicted == gold)
    return RunRecord(
        system=system,
        example_id=example_id,
        repeat_index=repeat_index,
        category=category,
        gold_label=gold,
        predicted_label=predicted,
        correct=correct,
        total_latency_seconds=latency,
        latency_by_stage=stages or {},
        error=error,
    )


# --- metrics.py --------------------------------------------------------------------------


def test_confusion_counts_basic():
    records = [
        _rec(category="legitimate", gold="Accept", predicted="Accept"),  # TP
        _rec(category="legitimate", gold="Accept", predicted="Reject"),  # FN
        _rec(category="unsafe", gold="Reject", predicted="Accept"),  # FP
        _rec(category="unsafe", gold="Reject", predicted="Reject"),  # TN
        _rec(category="misdirected", gold="Reject", predicted="Clarify"),  # TN (not Accept)
        _rec(category="ambiguous", gold="Clarify", predicted="Clarify"),  # excluded
    ]
    counts = compute_confusion_counts(records)
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 1, 1, 2)


def test_frr_equals_one_minus_recall():
    """Sanity-checks the framing chosen in metrics.py: FRR and Recall share
    the same TP/FN, so FRR == 1 - Recall must hold exactly."""
    records = [
        _rec(category="legitimate", gold="Accept", predicted="Accept"),
        _rec(category="legitimate", gold="Accept", predicted="Accept"),
        _rec(category="legitimate", gold="Accept", predicted="Reject"),
        _rec(category="unsafe", gold="Reject", predicted="Reject"),
    ]
    m = compute_system_metrics("s", records)
    assert m.recall == pytest.approx(2 / 3)
    assert m.false_rejection_rate == pytest.approx(1 / 3)
    assert m.false_rejection_rate == pytest.approx(1 - m.recall)


def test_precision_specificity_f1():
    records = [
        _rec(category="legitimate", gold="Accept", predicted="Accept"),  # TP
        _rec(category="legitimate", gold="Accept", predicted="Accept"),  # TP
        _rec(category="unsafe", gold="Reject", predicted="Accept"),  # FP
        _rec(category="unsafe", gold="Reject", predicted="Reject"),  # TN
        _rec(category="unsafe", gold="Reject", predicted="Reject"),  # TN
    ]
    m = compute_system_metrics("s", records)
    # TP=2, FP=1, FN=0, TN=2
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(1.0)
    assert m.specificity == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 * (2 / 3) / (1 + 2 / 3))


def test_metrics_none_when_denominator_is_zero():
    records = [_rec(category="ambiguous", gold="Clarify", predicted="Clarify")]
    m = compute_system_metrics("s", records)
    assert m.recall is None
    assert m.precision is None
    assert m.specificity is None
    assert m.f1 is None
    assert m.false_rejection_rate is None


def test_clarification_accuracy():
    records = [
        _rec(category="ambiguous", gold="Clarify", predicted="Clarify"),
        _rec(category="ambiguous", gold="Clarify", predicted="Accept"),
        _rec(category="legitimate", gold="Accept", predicted="Accept"),
    ]
    assert clarification_accuracy(records) == pytest.approx(0.5)


def test_clarification_accuracy_none_without_ambiguous_examples():
    records = [_rec(category="legitimate", gold="Accept", predicted="Accept")]
    assert clarification_accuracy(records) is None


def test_overall_accuracy_and_error_rate():
    records = [
        _rec(category="legitimate", gold="Accept", predicted="Accept"),
        _rec(category="legitimate", gold="Accept", predicted="Reject"),
        _rec(category="unsafe", gold="Reject", predicted="Error", error="PlannerError: boom"),
    ]
    assert overall_accuracy(records) == pytest.approx(1 / 3)
    assert error_rate(records) == pytest.approx(1 / 3)


def test_confusion_matrix_shape_and_counts():
    records = [
        _rec(category="legitimate", gold="Accept", predicted="Accept"),
        _rec(category="legitimate", gold="Accept", predicted="Reject"),
        _rec(category="unsafe", gold="Reject", predicted="Error"),
    ]
    matrix = confusion_matrix(records)
    assert matrix["Accept"]["Accept"] == 1
    assert matrix["Accept"]["Reject"] == 1
    assert matrix["Reject"]["Error"] == 1


def test_latency_summary_total_and_per_stage():
    records = [
        _rec(latency=1.0, stages={"planner": 0.6, "critic": 0.4}),
        _rec(latency=2.0, stages={"planner": 1.2, "critic": 0.8}),
        _rec(latency=3.0, stages={"planner": 1.8}),  # critic absent this run (e.g. short-circuited)
    ]
    summary = latency_summary(records)

    assert summary.total.mean == pytest.approx(2.0)
    assert summary.total.n == 3
    assert summary.by_stage["planner"].n == 3
    assert summary.by_stage["planner"].mean == pytest.approx((0.6 + 1.2 + 1.8) / 3)
    assert summary.by_stage["critic"].n == 2  # only the 2 runs where it appeared
    assert summary.by_stage["critic"].mean == pytest.approx((0.4 + 0.8) / 2)
    # p50/p95 are within the observed range
    assert 1.0 <= summary.total.p50 <= 3.0
    assert 1.0 <= summary.total.p95 <= 3.0


def test_category_breakdown():
    records = [
        _rec(category="legitimate", gold="Accept", predicted="Accept"),
        _rec(category="legitimate", gold="Accept", predicted="Reject"),
        _rec(category="unsafe", gold="Reject", predicted="Reject"),
    ]
    breakdown = category_breakdown(records)
    assert breakdown["legitimate"]["n"] == 2
    assert breakdown["legitimate"]["accuracy"] == pytest.approx(0.5)
    assert breakdown["unsafe"]["accuracy"] == pytest.approx(1.0)


# --- stats.py ----------------------------------------------------------------------------


def test_mean_confidence_interval_single_value():
    ci = mean_confidence_interval([0.8])
    assert ci.mean == ci.lower == ci.upper == pytest.approx(0.8)
    assert ci.n == 1


def test_mean_confidence_interval_brackets_mean():
    ci = mean_confidence_interval([0.7, 0.8, 0.9], confidence_level=0.95)
    assert ci.mean == pytest.approx(0.8)
    assert ci.lower < ci.mean < ci.upper


def test_mcnemar_test_counts_and_pairing():
    records_a = [
        _rec(system="A", example_id="e1", repeat_index=0, predicted="Accept", gold="Accept"),  # correct
        _rec(system="A", example_id="e2", repeat_index=0, predicted="Reject", gold="Accept"),  # incorrect
        _rec(system="A", example_id="e3", repeat_index=0, predicted="Accept", gold="Accept"),  # correct
    ]
    records_b = [
        _rec(system="B", example_id="e1", repeat_index=0, predicted="Accept", gold="Accept"),  # correct
        _rec(system="B", example_id="e2", repeat_index=0, predicted="Accept", gold="Accept"),  # correct (A wrong here)
        _rec(system="B", example_id="e3", repeat_index=0, predicted="Reject", gold="Accept"),  # incorrect (A right here)
    ]
    result = mcnemar_test(records_a, records_b)
    assert result.n_pairs == 3
    assert result.both_correct == 1
    assert result.a_only_correct == 1
    assert result.b_only_correct == 1
    assert result.both_incorrect == 0
    assert result.system_a == "A"
    assert result.system_b == "B"
    assert 0.0 <= result.p_value <= 1.0


def test_compare_latencies_uses_kruskal_for_small_groups():
    latencies = {"s1": [1.0, 1.1], "s2": [2.0, 2.1], "s3": [3.0, 3.1]}
    result = compare_latencies(latencies)
    assert result.test_used == "Kruskal-Wallis"  # n=2 per group forces this
    assert result.group_sizes == {"s1": 2, "s2": 2, "s3": 2}


# --- runner.py -----------------------------------------------------------------------------


def _example(id_="ex1", category="legitimate", gold="Accept", text="Turn off the stove"):
    return InstructionExample(
        id=id_,
        instruction_text=text,
        category=category,
        gold_label=gold,
        scene_context=SceneContext(),
    )


@pytest.fixture
def ctx_factory(ontology, rule_base):
    def _make(client):
        return SystemContext(client=client, models=_Models(), ontology=ontology, rule_base=rule_base)

    return _make


def test_run_example_records_correct_accept(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[json.dumps({"decision": "accept", "rationale": "fine", "description": "d", "actions": []})]
    )
    ctx = ctx_factory(client)
    example = _example()

    record = run_example("single_llm", baseline_a.run, example, ctx, repeat_index=0)

    assert record.predicted_label == "Accept"
    assert record.correct is True
    assert record.error is None
    assert record.system == "single_llm"
    assert record.example_id == "ex1"


def test_run_example_ambiguous_correct_only_if_clarify(ontology, ctx_factory):
    client = ScriptedLLMClient(
        responses=[json.dumps({"decision": "reject", "rationale": "x", "description": "d", "actions": []})]
    )
    ctx = ctx_factory(client)
    example = _example(category="ambiguous", gold="Clarify")

    record = run_example("single_llm", baseline_a.run, example, ctx, repeat_index=0)

    assert record.predicted_label == "Reject"
    assert record.correct is False  # Reject != Clarify, even though gold_label comparison isn't used directly


def test_run_example_catches_pipeline_errors(ontology, ctx_factory):
    # baseline_b's Planner will exhaust retries on unparseable garbage and raise PlannerError.
    client = ScriptedLLMClient(responses=["garbage", "still garbage", "still garbage"])
    ctx = ctx_factory(client)
    example = _example()

    record = run_example("multi_agent", baseline_b.run, example, ctx, repeat_index=0)

    assert record.predicted_label == "Error"
    assert record.correct is False
    assert record.error is not None
    assert "PlannerError" in record.error


def test_run_evaluation_orchestrates_systems_examples_repeats(ontology, ctx_factory):
    responses = [
        json.dumps({"decision": "accept", "rationale": "x", "description": "d", "actions": []})
        for _ in range(10)
    ]
    client = ScriptedLLMClient(responses=responses)
    ctx = ctx_factory(client)
    examples = [_example(id_="ex1"), _example(id_="ex2")]
    systems = {"single_llm": baseline_a.run}

    progress_calls = []
    records = run_evaluation(
        systems, examples, ctx, repeats=2, progress_callback=lambda done, total, rec: progress_calls.append((done, total))
    )

    assert len(records) == 2 * 2  # 1 system * 2 examples * 2 repeats
    assert progress_calls[-1] == (4, 4)
    assert {r.repeat_index for r in records} == {0, 1}
    assert {r.example_id for r in records} == {"ex1", "ex2"}


# --- report.py -----------------------------------------------------------------------------


def test_build_system_report_aggregates_across_repeats():
    records = [
        # repeat 0: 1/2 legitimate correct
        _rec(system="s", example_id="e1", repeat_index=0, category="legitimate", gold="Accept", predicted="Accept"),
        _rec(system="s", example_id="e2", repeat_index=0, category="legitimate", gold="Accept", predicted="Reject"),
        # repeat 1: 2/2 legitimate correct
        _rec(system="s", example_id="e1", repeat_index=1, category="legitimate", gold="Accept", predicted="Accept"),
        _rec(system="s", example_id="e2", repeat_index=1, category="legitimate", gold="Accept", predicted="Accept"),
    ]

    report = build_system_report("s", records)

    assert report.n_repeats == 2
    assert report.n_examples_per_repeat == 2
    assert "recall" in report.metric_cis
    assert report.metric_cis["recall"].mean == pytest.approx((0.5 + 1.0) / 2)
    assert report.pooled_metrics.n_examples == 4
    assert report.latency is not None
    assert report.latency.total.n == 4


def test_build_pairwise_mcnemar_covers_all_pairs():
    records_by_system = {
        "A": [_rec(system="A", example_id="e1", predicted="Accept", gold="Accept")],
        "B": [_rec(system="B", example_id="e1", predicted="Reject", gold="Accept")],
        "C": [_rec(system="C", example_id="e1", predicted="Accept", gold="Accept")],
    }
    results = build_pairwise_mcnemar(records_by_system)
    pairs = {(r.system_a, r.system_b) for r in results}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}


def test_build_latency_comparison_runs_without_error():
    records_by_system = {
        "A": [_rec(system="A", example_id=f"e{i}", latency=1.0 + 0.1 * i) for i in range(4)],
        "B": [_rec(system="B", example_id=f"e{i}", latency=2.0 + 0.1 * i) for i in range(4)],
    }
    result = build_latency_comparison(records_by_system)
    assert result.test_used in ("ANOVA", "Kruskal-Wallis")
    assert 0.0 <= result.p_value <= 1.0
