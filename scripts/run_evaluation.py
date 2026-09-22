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

    # Resume a run interrupted by a crash, closed laptop, or lost network
    # connection - re-run the exact same command, adding --resume with the
    # run directory that was printed at the start of the interrupted run.
    # Already-completed (system, example, repeat) combos are skipped.
    python scripts/run_evaluation.py --resume results/20260904_120000

    # Cluster: split the dataset across 20 parallel Slurm array tasks (see
    # cluster/run_experiment.slurm) instead of running it on one node:
    python scripts/run_evaluation.py --shard-index $SLURM_ARRAY_TASK_ID --shard-count 20
    # Each shard writes its own run dir and skips full-dataset aggregation
    # (a shard alone isn't representative); once every shard finishes:
    python scripts/merge_shards.py --shard-dirs results/2026*_shard*of20 --output results/merged

Every individual run is written to raw_results.jsonl the moment it completes
(not batched up for the end), so an interrupted run never loses more than the
one call that was in flight - see --resume above to continue it.

Results are written to results/<timestamp>/: raw_results.jsonl (every run),
metrics_summary.json/.csv (per-system metrics with CIs), statistical_tests.json
(McNemar + latency comparison), plots/*.png, and config_used.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import OllamaLLMClient  # noqa: E402
from intent_filter.config import load_config  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.evaluation import (  # noqa: E402
    load_raw_results,
    record_to_json_line,
    run_evaluation,
    write_full_report,
)
from intent_filter.sharding import shard_slice  # noqa: E402
from intent_filter.systems import ABLATIONS, SYSTEMS  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml. Default: config/config.yaml, falling back to config.example.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N dataset examples (for smoke testing).")
    parser.add_argument("--repeats", type=int, default=None, help="Override config.evaluation.repeats.")
    parser.add_argument("--systems", default=None, help="Comma-separated subset of: " + ", ".join(SYSTEMS) + ". Default: all four.")
    parser.add_argument("--no-ablations", action="store_true", help="Skip the Multi-Agent+LTL ablation runs.")
    parser.add_argument("--output-dir", default=None, help="Override config.evaluation.results_dir.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing results/<timestamp> directory to resume an interrupted run. "
        "Re-run with the same other flags (--limit/--repeats/--systems/--no-ablations) you used "
        "originally; already-completed runs found in its raw_results.jsonl are skipped.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="This shard's index (0-based) when splitting the dataset across several parallel "
        "Slurm jobs - see cluster/run_experiment.slurm. Requires --shard-count. Each shard gets "
        "a round-robin slice (intent_filter.sharding.shard_slice) and writes its own run dir; "
        "merge with scripts/merge_shards.py once every shard finishes.",
    )
    parser.add_argument("--shard-count", type=int, default=None, help="Total number of shards - see --shard-index.")
    return parser


def _print_progress(done: int, total: int, record) -> None:
    if done % 5 == 0 or done == total:
        print(f"  [{done}/{total}] {record.system:16s} {record.example_id:12s} "
              f"repeat={record.repeat_index} -> {record.predicted_label}"
              + (f" (ERROR: {record.error})" if record.error else ""))


def main() -> int:
    args = build_arg_parser().parse_args()

    config = load_config(args.config)
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    examples = load_dataset(config.dataset.path)
    if args.limit:
        examples = examples[: args.limit]
    sharded = args.shard_index is not None or args.shard_count is not None
    if sharded:
        if args.shard_index is None or args.shard_count is None:
            print("--shard-index and --shard-count must be given together.", file=sys.stderr)
            return 1
        examples = shard_slice(examples, args.shard_index, args.shard_count)

    client = OllamaLLMClient(base_url=config.ollama.base_url, timeout=config.ollama.timeout, max_retries=config.ollama.max_retries)
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

    # --- Resolve run directory + resume state ---------------------------------
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            print(f"--resume directory not found: {run_dir}", file=sys.stderr)
            return 1
        raw_results_path = run_dir / "raw_results.jsonl"
        existing_records = load_raw_results(raw_results_path)
        skip = {(r.system, r.example_id, r.repeat_index) for r in existing_records}
        print(f"Resuming {run_dir}: {len(skip)} run(s) already completed, will be skipped.")
    else:
        results_root = Path(args.output_dir or config.evaluation.results_dir)
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if sharded:
            # Shard index is included unconditionally (not just appended to a
            # shared timestamp) so two array tasks starting in the same
            # second never collide on run_dir, even without microsecond
            # precision - see cluster/run_experiment.slurm.
            run_name += f"_shard{args.shard_index}of{args.shard_count}"
        run_dir = results_root / run_name
        raw_results_path = run_dir / "raw_results.jsonl"
        existing_records = []
        skip = set()

    run_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(systems_to_run) * len(examples) * repeats
    print(f"Evaluating {len(systems_to_run)} system(s) over {len(examples)} example(s), "
          f"{repeats} repeat(s) each ({total_runs} total runs"
          + (f", {len(skip)} already done" if skip else "") + ").")
    print(f"Systems: {', '.join(systems_to_run)}")
    print(f"Run directory: {run_dir}"
          + ("" if args.resume else " (pass --resume with this path to continue if interrupted)"))

    # Each record is appended, flushed, and fsync'd to raw_results.jsonl the
    # moment it's produced, not batched up and written only after the whole
    # run finishes. flush() alone only pushes data out of Python's buffer -
    # the OS can still hold it before physically committing it to disk, which
    # a hard power-cut (e.g. force-rebooting a frozen machine) can lose;
    # fsync() forces that commit. Net effect: an interruption loses at most
    # the one call that was in flight, and the run can be continued with
    # --resume instead of restarted (and re-paid-for) from scratch.
    with open(raw_results_path, "a" if args.resume else "w", encoding="utf-8") as raw_results_file:

        def _persist_record(record) -> None:
            raw_results_file.write(record_to_json_line(record) + "\n")
            raw_results_file.flush()
            os.fsync(raw_results_file.fileno())

        new_records = run_evaluation(
            systems_to_run,
            examples,
            ctx,
            repeats,
            progress_callback=_print_progress,
            on_record=_persist_record,
            skip=skip,
        )

    # raw_results.jsonl is the single source of truth for aggregation below,
    # whether it was written in one uninterrupted process or stitched
    # together across a crash and a --resume.
    records = existing_records + new_records

    # --- Aggregate + save (metrics/stats/unsafety-breakdown/plots) ---------------
    records_by_system: dict[str, list] = {}
    for r in records:
        records_by_system.setdefault(r.system, []).append(r)

    if sharded:
        # A single shard only has a slice of the dataset - per-metric CIs,
        # McNemar, and the unsafety-type breakdown would all be computed
        # over a non-representative subset if generated here. Skip the full
        # report per shard; run scripts/merge_shards.py once every shard
        # finishes, which concatenates every shard's raw_results.jsonl and
        # produces one report over the complete dataset.
        print(f"\nShard {args.shard_index}/{args.shard_count} done: "
              f"{len(records)} record(s) written to {raw_results_path}.")
        print("Run scripts/merge_shards.py once every shard has finished to produce the full report.")
    else:
        full_report = write_full_report(records_by_system, rule_base, config.evaluation.confidence_level, run_dir)
        reports = full_report.reports

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

        # --- Report to stdout -------------------------------------------------------
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

        lc = full_report.latency_comparison
        print(f"\nLatency comparison: {lc.test_used} (statistic={lc.statistic:.3f}, p={lc.p_value:.4f})")

        unsafety_breakdown = full_report.unsafety_breakdown
        unsafety_types = sorted({t for stats in unsafety_breakdown.values() for t in stats})
        if unsafety_types:
            header = f"\n{'System':<20}" + "".join(f"{t:>16}" for t in unsafety_types)
            print(header)
            for system, stats in unsafety_breakdown.items():
                row = f"{system:<20}"
                for t in unsafety_types:
                    row += f"{stats[t].catch_rate:>15.0%} " if t in stats else f"{'n/a':>16}"
                print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
