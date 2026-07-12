"""Combines metrics.py + stats.py into the final structured evaluation report:
per-system metrics with confidence intervals across repeats, pairwise
McNemar comparisons, and the latency comparison across systems. This is
what scripts/run_evaluation.py serializes to results/<timestamp>/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from intent_filter.evaluation.metrics import (
    LatencySummary,
    SystemMetrics,
    compute_system_metrics,
    latency_summary,
)
from intent_filter.evaluation.stats import (
    ConfidenceInterval,
    LatencyComparisonResult,
    McNemarResult,
    compare_latencies,
    mcnemar_test,
    mean_confidence_interval,
)
from intent_filter.evaluation.types import RunRecord

# Metric fields aggregated (mean + CI) across repeats. Confusion counts
# themselves are not repeat-aggregated - the confusion matrix plot pools all
# repeats' records directly, since it's a count, not a rate.
METRIC_FIELDS = (
    "recall",
    "precision",
    "specificity",
    "f1",
    "false_rejection_rate",
    "clarification_accuracy",
    "overall_accuracy",
    "error_rate",
)


@dataclass(frozen=True)
class SystemReport:
    system: str
    n_repeats: int
    n_examples_per_repeat: int
    metric_cis: dict[str, ConfidenceInterval] = field(default_factory=dict)
    pooled_metrics: SystemMetrics | None = None
    latency: LatencySummary | None = None


def _group_by_repeat(records: list[RunRecord]) -> dict[int, list[RunRecord]]:
    by_repeat: dict[int, list[RunRecord]] = {}
    for r in records:
        by_repeat.setdefault(r.repeat_index, []).append(r)
    return by_repeat


def build_system_report(
    system: str, records: list[RunRecord], confidence_level: float = 0.95
) -> SystemReport:
    """Aggregate one system's records into per-metric mean +/- CI across repeats.

    Each repeat's full pass over the dataset yields one value per metric;
    `mean_confidence_interval` is then computed across those per-repeat
    values, matching the brief's "runs each experiment multiple times ...
    reports mean +/- confidence interval per metric".
    """
    by_repeat = _group_by_repeat(records)
    per_repeat_metrics = [
        compute_system_metrics(system, rows) for rows in by_repeat.values()
    ]

    metric_cis: dict[str, ConfidenceInterval] = {}
    for metric_name in METRIC_FIELDS:
        values = [
            getattr(m, metric_name)
            for m in per_repeat_metrics
            if getattr(m, metric_name) is not None
        ]
        if values:
            metric_cis[metric_name] = mean_confidence_interval(values, confidence_level)

    n_examples_per_repeat = len(records) // len(by_repeat) if by_repeat else 0
    return SystemReport(
        system=system,
        n_repeats=len(by_repeat),
        n_examples_per_repeat=n_examples_per_repeat,
        metric_cis=metric_cis,
        pooled_metrics=compute_system_metrics(system, records),
        latency=latency_summary(records),
    )


def build_pairwise_mcnemar(records_by_system: dict[str, list[RunRecord]]) -> list[McNemarResult]:
    """McNemar's test for every pair of systems, over their pooled (all-repeats) records."""
    systems = list(records_by_system)
    results = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            a, b = systems[i], systems[j]
            results.append(mcnemar_test(records_by_system[a], records_by_system[b]))
    return results


def build_latency_comparison(records_by_system: dict[str, list[RunRecord]]) -> LatencyComparisonResult:
    """ANOVA/Kruskal-Wallis comparison of per-run total latency across systems."""
    latencies = {
        system: [r.total_latency_seconds for r in records]
        for system, records in records_by_system.items()
    }
    return compare_latencies(latencies)
