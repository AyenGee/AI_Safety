#!/usr/bin/env python
"""Evaluation harness: run the four intent-filtering systems (and the
Multi-Agent+LTL ablations) over the labeled instruction dataset, compute
metrics with confidence intervals across repeats, run statistical
significance tests, and save a results table + plots.

Usage:
    # Small smoke test against the real API (a handful of instructions, 1 repeat):
    python scripts/run_evaluation.py --limit 8 --repeats 1

    # Full evaluation (all systems + ablations, config-driven repeat count):
    python scripts/run_evaluation.py

    # Only the two baselines, skip ablations:
    python scripts/run_evaluation.py --systems single_llm,multi_agent --no-ablations

Results are written to results/<timestamp>/: raw_results.jsonl (every run),
metrics_summary.json/.csv (per-system metrics with CIs), statistical_tests.json
(McNemar + latency comparison), plots/*.png, and config_used.json.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import AnthropicLLMClient  # noqa: E402
from intent_filter.config import load_config, load_secrets  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.evaluation import (  # noqa: E402
    build_latency_comparison,
    build_pairwise_mcnemar,
    build_system_report,
    plot_confusion_matrices,
    plot_latency_breakdown,
    plot_recall_frr_tradeoff,
    run_evaluation,
)
from intent_filter.systems import ABLATIONS, SYSTEMS  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml. Default: config/config.yaml, falling back to config.example.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N dataset examples (for smoke testing).")
    parser.add_argument("--repeats", type=int, default=None, help="Override config.evaluation.repeats.")
    parser.add_argument("--systems", default=None, help="Comma-separated subset of: " + ", ".join(SYSTEMS) + ". Default: all four.")
    parser.add_argument("--no-ablations", action="store_true", help="Skip the Multi-Agent+LTL ablation runs.")
    parser.add_argument("--output-dir", default=None, help="Override config.evaluation.results_dir.")
    return parser


def _print_progress(done: int, total: int, record) -> None:
    if done % 5 == 0 or done == total:
        print(f"  [{done}/{total}] {record.system:16s} {record.example_id:12s} "
              f"repeat={record.repeat_index} -> {record.predicted_label}"
              + (f" (ERROR: {record.error})" if record.error else ""))


def main() -> int:
    args = build_arg_parser().parse_args()

    config = load_config(args.config)
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    examples = load_dataset(config.dataset.path)
    if args.limit:
        examples = examples[: args.limit]

    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)
    ctx = SystemContext(
        client=client,
        models=config.models,
        ontology=ontology,
        rule_base=rule_base,
        ambiguity_margin=config.agent.ambiguity_margin,
        max_refinement_attempts=config.agent.max_refinement_attempts,
        translation_max_retries=config.agent.translation_max_retries,
    )

    systems_to_run = dict(SYSTEMS)
    if args.systems:
        requested = {name.strip() for name in args.systems.split(",")}
        systems_to_run = {name: fn for name, fn in systems_to_run.items() if name in requested}
    if not args.no_ablations:
        systems_to_run.update(ABLATIONS)

    repeats = args.repeats or config.evaluation.repeats

    print(f"Evaluating {len(systems_to_run)} system(s) over {len(examples)} example(s), "
          f"{repeats} repeat(s) each ({len(systems_to_run) * len(examples) * repeats} total runs).")
    print(f"Systems: {', '.join(systems_to_run)}")

    records = run_evaluation(systems_to_run, examples, ctx, repeats, progress_callback=_print_progress)

    # --- Aggregate ---------------------------------------------------------------
    records_by_system: dict[str, list] = {}
    for r in records:
        records_by_system.setdefault(r.system, []).append(r)

    reports = {
        system: build_system_report(system, rows, config.evaluation.confidence_level)
        for system, rows in records_by_system.items()
    }
    mcnemar_results = build_pairwise_mcnemar(records_by_system)
    latency_comparison = build_latency_comparison(records_by_system)

    # --- Save ---------------------------------------------------------------------
    results_root = Path(args.output_dir or config.evaluation.results_dir)
    run_dir = results_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "raw_results.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r), default=str) + "\n")

    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({s: dataclasses.asdict(r) for s, r in reports.items()}, f, indent=2, default=str)

    with open(run_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "metric", "mean", "ci_lower", "ci_upper", "n_repeats"])
        for system, report in reports.items():
            for metric_name, ci in report.metric_cis.items():
                writer.writerow([system, metric_name, ci.mean, ci.lower, ci.upper, ci.n])

    with open(run_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mcnemar_pairwise": [dataclasses.asdict(m) for m in mcnemar_results],
                "latency_comparison": dataclasses.asdict(latency_comparison),
            },
            f,
            indent=2,
            default=str,
        )

    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.model_dump(mode="json"),
                "repeats": repeats,
                "limit": args.limit,
                "systems": list(systems_to_run),
            },
            f,
            indent=2,
            default=str,
        )

    pooled_metrics = {s: r.pooled_metrics for s, r in reports.items()}
    plot_recall_frr_tradeoff(pooled_metrics, plots_dir / "recall_frr_tradeoff.png")
    plot_latency_breakdown(records_by_system, plots_dir / "latency_breakdown.png")
    plot_confusion_matrices(records_by_system, plots_dir / "confusion_matrices.png")

    # --- Report to stdout -----------------------------------------------------------
    print(f"\nResults written to {run_dir}")
    print(f"\n{'System':<20}{'Recall':>10}{'Precision':>12}{'Specificity':>13}{'F1':>8}{'FRR':>8}{'ClarifyAcc':>12}")
    for system, report in reports.items():
        def fmt(name: str) -> str:
            ci = report.metric_cis.get(name)
            return f"{ci.mean:.2f}" if ci else "n/a"

        print(
            f"{system:<20}{fmt('recall'):>10}{fmt('precision'):>12}{fmt('specificity'):>13}"
            f"{fmt('f1'):>8}{fmt('false_rejection_rate'):>8}{fmt('clarification_accuracy'):>12}"
        )

    print(f"\n{'System':<20}{'Mean (s)':>10}{'p50 (s)':>10}{'p95 (s)':>10}")
    for system, report in reports.items():
        lat = report.latency
        if lat:
            print(f"{system:<20}{lat.total.mean:>10.2f}{lat.total.p50:>10.2f}{lat.total.p95:>10.2f}")

    print(f"\nLatency comparison: {latency_comparison.test_used} "
          f"(statistic={latency_comparison.statistic:.3f}, p={latency_comparison.p_value:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
