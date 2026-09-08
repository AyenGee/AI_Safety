"""Orchestrates running the dataset through the systems (and ablations),
repeated multiple times, collecting RunRecord rows.

No metrics/statistics logic here - that's evaluation/metrics.py and
evaluation/stats.py; this module only executes pipelines and captures their
outcomes, including failures.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from intent_filter.dataset import InstructionExample
from intent_filter.decision import PipelineResult, SystemContext
from intent_filter.evaluation.types import RunRecord
from intent_filter.systems import SystemRunFn

ProgressCallback = Callable[[int, int, RunRecord], None]
RecordSink = Callable[[RunRecord], None]
RunKey = tuple[str, str, int]  # (system, example_id, repeat_index)


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
        related_rule_ids=tuple(example.related_rule_ids),
        rationale=rationale,
        error=error,
    )


def run_evaluation(
    systems: dict[str, SystemRunFn],
    examples: Iterable[InstructionExample],
    ctx: SystemContext,
    repeats: int,
    progress_callback: ProgressCallback | None = None,
    on_record: RecordSink | None = None,
    skip: set[RunKey] | None = None,
) -> list[RunRecord]:
    """Run every (system, example, repeat) combination, returning all RunRecords.

    Order is system -> repeat -> example, so a `progress_callback` (e.g. a
    tqdm-style printer) reports steady progress through one whole system's
    repeat before moving to the next, rather than interleaving.

    `skip` is a set of (system, example_id, repeat_index) keys to treat as
    already completed - e.g. when resuming a run that was interrupted by a
    crash, a closed laptop, or a lost network connection. Those combos are
    neither re-run nor included in the returned list; the caller is expected
    to already have them (loaded from a previous checkpoint file).

    `on_record`, if given, is called immediately after each *new* RunRecord
    is produced, before moving on to the next combo. This is what makes a
    long batch resumable: the caller can persist each record to disk as it
    happens (see scripts/run_evaluation.py) instead of only writing results
    after the entire run finishes, which would lose everything on a crash.
    """
    examples = list(examples)
    skip = skip or set()
    records: list[RunRecord] = []
    total = len(systems) * len(examples) * repeats
    done = len(skip)

    for system_name, run_fn in systems.items():
        for repeat_index in range(repeats):
            for example in examples:
                if (system_name, example.id, repeat_index) in skip:
                    continue
                record = run_example(system_name, run_fn, example, ctx, repeat_index)
                records.append(record)
                done += 1
                if on_record is not None:
                    on_record(record)
                if progress_callback is not None:
                    progress_callback(done, total, record)

    return records


def record_to_json_line(record: RunRecord) -> str:
    return json.dumps(dataclasses.asdict(record), default=str)


def record_from_dict(data: dict) -> RunRecord:
    data = dict(data)
    data["related_rule_ids"] = tuple(data.get("related_rule_ids", ()))
    data["latency_by_stage"] = dict(data.get("latency_by_stage", {}))
    return RunRecord(**data)


def load_raw_results(path: Path) -> list[RunRecord]:
    """Load previously saved RunRecords from a raw_results.jsonl checkpoint file.

    Used both to resume an interrupted run (build the `skip` set of
    already-completed combos) and, after a run finishes, as the single
    source of truth for aggregation - whether the file was written in one
    uninterrupted process or stitched together across a crash and a
    `--resume`. Returns an empty list if the file doesn't exist yet.

    A hard power-cut (e.g. force-rebooting a frozen machine) can catch a
    write to this file mid-flight, leaving one trailing line truncated
    before the OS finished committing it to disk. Rather than let that one
    unparseable line crash the whole resume, it's skipped (with a warning) -
    the run it belonged to never gets marked complete, so it's simply
    re-run, exactly as if it had never started.
    """
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(record_from_dict(json.loads(line)))
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping unreadable line {line_number} in {path} "
                    "(likely a partial write left by an interrupted run) - it will be re-run."
                )
    return records
