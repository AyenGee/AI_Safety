"""Combines metrics.py + stats.py into the final structured evaluation report:
per-system metrics with confidence intervals across repeats, pairwise
McNemar comparisons, and the latency comparison across systems. This is
what scripts/run_evaluation.py serializes to results/<timestamp>/.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from intent_filter.environment.rules import SafetyRuleBase
from intent_filter.evaluation.metrics import (
    LatencySummary,
    SystemMetrics,
    UnsafetyTypeStats,
    compute_system_metrics,
    latency_summary,
    unsafety_type_breakdown,
)
from intent_filter.evaluation.plots import (
    plot_confusion_matrices,
    plot_latency_breakdown,
    plot_recall_frr_tradeoff,
    plot_unsafety_type_breakdown,
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


def build_unsafety_breakdown_report(
    records_by_system: dict[str, list[RunRecord]], rule_base: SafetyRuleBase, by: str = "category"
) -> dict[str, dict[str, UnsafetyTypeStats]]:
    """Per-system, per-unsafety-type catch rate (see metrics.unsafety_type_breakdown).

    Pooled across all repeats (like the confusion matrix, unlike the
    accuracy-metric CIs) - the point is to compare *which types* of unsafe
    command each system misses, not to put a confidence interval on it.
    """
    return {
        system: unsafety_type_breakdown(records, rule_base, by=by)
        for system, records in records_by_system.items()
    }


@dataclass(frozen=True)
class FullReport:
    reports: dict[str, SystemReport]
    latency_comparison: LatencyComparisonResult
    unsafety_breakdown: dict[str, dict[str, UnsafetyTypeStats]]
    unsafety_breakdown_by_rule: dict[str, dict[str, UnsafetyTypeStats]]


def write_full_report(
    records_by_system: dict[str, list[RunRecord]],
    rule_base: SafetyRuleBase,
    confidence_level: float,
    output_dir: Path,
) -> FullReport:
    """Aggregate + write every report artifact (metrics/stats/unsafety-breakdown/
    plots) that a full evaluation run produces, given only the records grouped
    by system - independent of how those records were obtained (a single live
    run, a --resume, or several sharded runs merged together, see
    scripts/merge_shards.py). `raw_results.jsonl` itself is not written here -
    each caller owns that (a live run writes it incrementally; merge_shards.py
    concatenates existing shard files) - this only produces everything derived
    from it.

    Factored out of scripts/run_evaluation.py and scripts/regenerate_report.py,
    which previously each carried their own copy of this ~80-line tail.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        system: build_system_report(system, rows, confidence_level)
        for system, rows in records_by_system.items()
    }
    mcnemar_results = build_pairwise_mcnemar(records_by_system)
    latency_comparison = build_latency_comparison(records_by_system)
    unsafety_breakdown = build_unsafety_breakdown_report(records_by_system, rule_base, by="category")
    unsafety_breakdown_by_rule = build_unsafety_breakdown_report(records_by_system, rule_base, by="rule")

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({s: dataclasses.asdict(r) for s, r in reports.items()}, f, indent=2, default=str)

    with open(output_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "metric", "mean", "ci_lower", "ci_upper", "n_repeats"])
        for system, report in reports.items():
            for metric_name, ci in report.metric_cis.items():
                writer.writerow([system, metric_name, ci.mean, ci.lower, ci.upper, ci.n])

    with open(output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mcnemar_pairwise": [dataclasses.asdict(m) for m in mcnemar_results],
                "latency_comparison": dataclasses.asdict(latency_comparison),
            },
            f,
            indent=2,
            default=str,
        )

    with open(output_dir / "unsafety_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "by_category": {
                    s: {k: dataclasses.asdict(v) for k, v in stats.items()}
                    for s, stats in unsafety_breakdown.items()
                },
                "by_rule": {
                    s: {k: dataclasses.asdict(v) for k, v in stats.items()}
                    for s, stats in unsafety_breakdown_by_rule.items()
                },
            },
            f,
            indent=2,
            default=str,
        )

    with open(output_dir / "unsafety_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "granularity", "unsafety_type", "n_examples", "n_caught", "catch_rate"])
        for granularity, breakdown in (("category", unsafety_breakdown), ("rule", unsafety_breakdown_by_rule)):
            for system, stats in breakdown.items():
                for key, s in stats.items():
                    writer.writerow([system, granularity, key, s.n_examples, s.n_caught, s.catch_rate])

    pooled_metrics = {s: r.pooled_metrics for s, r in reports.items()}
    plot_recall_frr_tradeoff(pooled_metrics, plots_dir / "recall_frr_tradeoff.png")
    plot_latency_breakdown(records_by_system, plots_dir / "latency_breakdown.png")
    plot_confusion_matrices(records_by_system, plots_dir / "confusion_matrices.png")
    plot_unsafety_type_breakdown(unsafety_breakdown, plots_dir / "unsafety_type_breakdown.png")
    plot_unsafety_type_breakdown(
        unsafety_breakdown_by_rule,
        plots_dir / "unsafety_type_breakdown_by_rule.png",
        title="Catch rate by individual rule, per system",
    )

    return FullReport(
        reports=reports,
        latency_comparison=latency_comparison,
        unsafety_breakdown=unsafety_breakdown,
        unsafety_breakdown_by_rule=unsafety_breakdown_by_rule,
    )
