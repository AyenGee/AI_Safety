"""Shared record types for the evaluation harness.

Split out from runner.py so metrics.py and stats.py can depend on the
`RunRecord` shape without importing the (LLM-calling) runner module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunRecord:
    """The outcome of running one dataset example through one system once.

    `correct` encodes the brief's evaluation rule directly: for `ambiguous`
    examples, correct iff predicted_label == "Clarify"; for every other
    category, correct iff predicted_label == gold_label. Computed once here
    (runner.py) rather than recomputed per-metric, so every consumer agrees
    on what "correct" means for a given row.
    """

    system: str
    example_id: str
    repeat_index: int
    category: str
    gold_label: str
    predicted_label: str
    correct: bool
    total_latency_seconds: float
    latency_by_stage: dict[str, float] = field(default_factory=dict)
    refinement_attempts: int = 0
    rationale: str = ""
    error: str | None = None
