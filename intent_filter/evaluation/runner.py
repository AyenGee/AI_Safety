"""Orchestrates running the dataset through the systems (and ablations),
repeated multiple times, collecting RunRecord rows.

No metrics/statistics logic here - that's evaluation/metrics.py and
evaluation/stats.py; this module only executes pipelines and captures their
outcomes, including failures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from intent_filter.dataset import InstructionExample
from intent_filter.decision import PipelineResult, SystemContext
from intent_filter.evaluation.types import RunRecord
from intent_filter.systems import SystemRunFn

ProgressCallback = Callable[[int, int, RunRecord], None]


def run_example(
    system_name: str,
    run_fn: SystemRunFn,
    example: InstructionExample,
    ctx: SystemContext,
    repeat_index: int,
) -> RunRecord:
    """Run one dataset example through one system once.

    Catches *any* exception the pipeline raises (PlannerError/CriticError/
    SingleLLMError from malformed-response retries exhausted, but also
    transient API errors - rate limits, timeouts, network issues - from the
    underlying Anthropic client) rather than letting one bad instruction
    abort a batch of hundreds of runs. The failure is recorded, not
    swallowed: `predicted_label` becomes "Error" (which can never match a
    gold label, so it always counts against accuracy) and the exception's
    type and message are kept in `error` for later inspection.
    """
    state = example.scene_context.to_world_state(ctx.ontology)

    try:
        result: PipelineResult = run_fn(example.instruction_text, state, ctx)
        predicted_label = result.decision
        error = None
        total_latency = result.total_latency_seconds
        latency_by_stage = result.latency_by_stage()
        refinement_attempts = result.refinement_attempts
        rationale = result.rationale
    except Exception as exc:  # noqa: BLE001 - see docstring: a batch harness must not abort on one failure
        predicted_label = "Error"
        error = f"{type(exc).__name__}: {exc}"
        total_latency = 0.0
        latency_by_stage = {}
        refinement_attempts = 0
        rationale = ""

    correct = (
        predicted_label == "Clarify"
        if example.category == "ambiguous"
        else predicted_label == example.gold_label
    )

    return RunRecord(
        system=system_name,
        example_id=example.id,
        repeat_index=repeat_index,
        category=example.category,
        gold_label=example.gold_label,
        predicted_label=predicted_label,
        correct=correct,
        total_latency_seconds=total_latency,
        latency_by_stage=latency_by_stage,
        refinement_attempts=refinement_attempts,
        rationale=rationale,
        error=error,
    )


def run_evaluation(
    systems: dict[str, SystemRunFn],
    examples: Iterable[InstructionExample],
    ctx: SystemContext,
    repeats: int,
    progress_callback: ProgressCallback | None = None,
) -> list[RunRecord]:
    """Run every (system, example, repeat) combination, returning all RunRecords.

    Order is system -> repeat -> example, so a `progress_callback` (e.g. a
    tqdm-style printer) reports steady progress through one whole system's
    repeat before moving to the next, rather than interleaving.
    """
    examples = list(examples)
    records: list[RunRecord] = []
    total = len(systems) * len(examples) * repeats
    done = 0

    for system_name, run_fn in systems.items():
        for repeat_index in range(repeats):
            for example in examples:
                record = run_example(system_name, run_fn, example, ctx, repeat_index)
                records.append(record)
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, record)

    return records
