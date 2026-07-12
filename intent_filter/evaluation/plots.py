"""Matplotlib figure builders for the evaluation report.

Uses the non-interactive "Agg" backend so this works headlessly (no
display needed, e.g. running the evaluation over SSH or in CI).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use())

from intent_filter.evaluation.metrics import SystemMetrics, confusion_matrix
from intent_filter.evaluation.types import RunRecord


def plot_recall_frr_tradeoff(metrics_by_system: dict[str, SystemMetrics], output_path: Path) -> None:
    """Recall (y) vs. False Rejection Rate (x), one point per system.

    The ideal point is top-left (high recall, low FRR - i.e. FRR = 1 -
    Recall by construction here, so points fall on the anti-diagonal; the
    plot exists to compare *where* on that line each system sits, and how
    the LTL-augmented systems shift relative to their baselines).
    """
    # A legend (rather than inline text annotations) is used deliberately:
    # systems frequently land on or near the same point (e.g. all scoring
    # Recall=1.0, FRR=0.0 on an easy subset), which makes inline labels
    # overlap into unreadable text - a legend has no such failure mode.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for system, m in metrics_by_system.items():
        if m.recall is None or m.false_rejection_rate is None:
            continue
        ax.scatter(m.false_rejection_rate, m.recall, s=90, zorder=3, label=system)
    ax.set_xlabel("False Rejection Rate (FRR)")
    ax.set_ylabel("Recall (legitimate commands correctly accepted)")
    ax.set_title("Recall vs. False Rejection Rate by system")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latency_breakdown(records_by_system: dict[str, list[RunRecord]], output_path: Path) -> None:
    """Stacked bar chart: mean per-stage latency, one bar per system."""
    systems = list(records_by_system)
    stage_names = sorted(
        {stage for records in records_by_system.values() for r in records for stage in r.latency_by_stage}
    )

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(systems) + 2), 5))
    bottoms = [0.0] * len(systems)
    for stage in stage_names:
        means = []
        for system in systems:
            records = records_by_system[system]
            values = [r.latency_by_stage.get(stage, 0.0) for r in records]
            means.append(sum(values) / len(values) if values else 0.0)
        ax.bar(systems, means, bottom=bottoms, label=stage)
        bottoms = [b + m for b, m in zip(bottoms, means, strict=True)]

    ax.set_ylabel("Mean latency (seconds)")
    ax.set_title("Latency breakdown by stage, per system")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(records_by_system: dict[str, list[RunRecord]], output_path: Path) -> None:
    """One gold-label x predicted-label heatmap per system, side by side."""
    systems = list(records_by_system)
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)

    for ax, system in zip(axes[0], systems, strict=True):
        matrix = confusion_matrix(records_by_system[system])
        gold_labels = list(matrix.keys())
        pred_labels = sorted({pred for row in matrix.values() for pred in row})
        data = [[matrix[g].get(p, 0) for p in pred_labels] for g in gold_labels]

        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(gold_labels)))
        ax.set_yticklabels(gold_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(system)
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(j, i, str(val), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
