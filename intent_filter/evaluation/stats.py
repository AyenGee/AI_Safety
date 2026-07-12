"""Statistical testing and repeat aggregation for the evaluation harness.

- `mean_confidence_interval`: t-distribution CI for a metric computed once
  per repeat (repeats are few - 3 to 5 - so a t-interval is more
  appropriate than a normal-approximation z-interval).
- `mcnemar_test`: paired McNemar's test comparing two systems' correctness
  on matched (example_id, repeat_index) pairs, per the brief.
- `compare_latencies`: ANOVA or Kruskal-Wallis across systems' per-run
  latencies, choosing via a per-group Shapiro-Wilk normality check - the
  brief explicitly asks to "check and choose appropriately" rather than
  assuming one test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar

from intent_filter.evaluation.types import RunRecord


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    n: int
    confidence_level: float


def mean_confidence_interval(
    values: list[float], confidence_level: float = 0.95
) -> ConfidenceInterval:
    """t-distribution confidence interval for the mean of `values` (one per repeat).

    With n < 2 there's no variance to estimate a CI from, so lower==upper==mean.
    """
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    if n < 2:
        return ConfidenceInterval(
            mean=mean, lower=mean, upper=mean, n=n, confidence_level=confidence_level
        )
    sem = float(scipy_stats.sem(values))
    half_width = sem * float(scipy_stats.t.ppf((1 + confidence_level) / 2, df=n - 1))
    return ConfidenceInterval(
        mean=mean,
        lower=mean - half_width,
        upper=mean + half_width,
        n=n,
        confidence_level=confidence_level,
    )


@dataclass(frozen=True)
class McNemarResult:
    system_a: str
    system_b: str
    statistic: float
    p_value: float
    n_pairs: int
    both_correct: int
    a_only_correct: int
    b_only_correct: int
    both_incorrect: int


def mcnemar_test(records_a: list[RunRecord], records_b: list[RunRecord]) -> McNemarResult:
    """Paired McNemar's test comparing two systems' correctness.

    Pairs records by (example_id, repeat_index) - both systems are run over
    the same dataset for the same number of repeats, so this pairing is
    exact and deterministic. Uses the exact binomial variant when the
    discordant-pair count is small (<25, the usual rule of thumb) and the
    chi-squared approximation with continuity correction otherwise.
    """
    by_key_a = {(r.example_id, r.repeat_index): r.correct for r in records_a}
    by_key_b = {(r.example_id, r.repeat_index): r.correct for r in records_b}
    shared_keys = sorted(set(by_key_a) & set(by_key_b))

    both_correct = a_only = b_only = both_incorrect = 0
    for key in shared_keys:
        a_correct, b_correct = by_key_a[key], by_key_b[key]
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif b_correct and not a_correct:
            b_only += 1
        else:
            both_incorrect += 1

    table = [[both_correct, a_only], [b_only, both_incorrect]]
    result = mcnemar(table, exact=(a_only + b_only) < 25, correction=True)

    return McNemarResult(
        system_a=records_a[0].system if records_a else "A",
        system_b=records_b[0].system if records_b else "B",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_pairs=len(shared_keys),
        both_correct=both_correct,
        a_only_correct=a_only,
        b_only_correct=b_only,
        both_incorrect=both_incorrect,
    )


@dataclass(frozen=True)
class LatencyComparisonResult:
    test_used: str  # "ANOVA" or "Kruskal-Wallis"
    statistic: float
    p_value: float
    normality_p_values: dict[str, float]
    group_sizes: dict[str, int]


def compare_latencies(
    latencies_by_system: dict[str, list[float]], alpha: float = 0.05
) -> LatencyComparisonResult:
    """Compare per-run total latency across systems.

    Runs a Shapiro-Wilk normality check per system's latency sample; only
    uses one-way ANOVA if every group passes (p >= alpha) and has enough
    samples to test (Shapiro-Wilk needs n >= 3). Falls back to the
    non-parametric Kruskal-Wallis test otherwise, matching the brief's
    instruction to check normality and choose the appropriate test rather
    than assuming ANOVA is always valid.
    """
    normality_p_values: dict[str, float] = {}
    all_normal = True
    for system, values in latencies_by_system.items():
        if len(values) < 3:
            normality_p_values[system] = float("nan")
            all_normal = False
            continue
        _, p = scipy_stats.shapiro(values)
        normality_p_values[system] = float(p)
        if p < alpha:
            all_normal = False

    groups = list(latencies_by_system.values())
    if all_normal:
        statistic, p_value = scipy_stats.f_oneway(*groups)
        test_used = "ANOVA"
    else:
        statistic, p_value = scipy_stats.kruskal(*groups)
        test_used = "Kruskal-Wallis"

    return LatencyComparisonResult(
        test_used=test_used,
        statistic=float(statistic),
        p_value=float(p_value),
        normality_p_values=normality_p_values,
        group_sizes={system: len(values) for system, values in latencies_by_system.items()},
    )
