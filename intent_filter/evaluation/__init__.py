"""Evaluation harness: run the four systems (+ ablations) over the dataset,
compute metrics, run statistical tests, and produce plots.

See docs/methodology.md ("Metrics") for how Recall/Precision/Specificity/F1/
FRR are operationalized for a 3-class (Accept/Reject/Clarify) decision, and
scripts/run_evaluation.py for the CLI entry point.
"""

from intent_filter.evaluation.metrics import (
    ConfusionCounts,
    LatencyStats,
    LatencySummary,
    SystemMetrics,
    UnsafetyTypeStats,
    category_breakdown,
    clarification_accuracy,
    compute_confusion_counts,
    compute_system_metrics,
    confusion_matrix,
    error_rate,
    latency_summary,
    overall_accuracy,
    unsafety_type_breakdown,
)
from intent_filter.evaluation.plots import (
    plot_confusion_matrices,
    plot_latency_breakdown,
    plot_recall_frr_tradeoff,
    plot_unsafety_type_breakdown,
)
from intent_filter.evaluation.report import (
    SystemReport,
    build_latency_comparison,
    build_pairwise_mcnemar,
    build_system_report,
    build_unsafety_breakdown_report,
)
from intent_filter.evaluation.runner import run_evaluation, run_example
from intent_filter.evaluation.stats import (
    ConfidenceInterval,
    LatencyComparisonResult,
    McNemarResult,
    compare_latencies,
    mcnemar_test,
    mean_confidence_interval,
)
from intent_filter.evaluation.types import RunRecord

__all__ = [
    "ConfusionCounts",
    "SystemMetrics",
    "category_breakdown",
    "clarification_accuracy",
    "compute_confusion_counts",
    "compute_system_metrics",
    "confusion_matrix",
    "error_rate",
    "latency_summary",
    "overall_accuracy",
    "LatencyStats",
    "LatencySummary",
    "UnsafetyTypeStats",
    "unsafety_type_breakdown",
    "plot_confusion_matrices",
    "plot_latency_breakdown",
    "plot_recall_frr_tradeoff",
    "plot_unsafety_type_breakdown",
    "SystemReport",
    "build_latency_comparison",
    "build_pairwise_mcnemar",
    "build_system_report",
    "build_unsafety_breakdown_report",
    "run_evaluation",
    "run_example",
    "ConfidenceInterval",
    "LatencyComparisonResult",
    "McNemarResult",
    "compare_latencies",
    "mcnemar_test",
    "mean_confidence_interval",
    "RunRecord",
]
