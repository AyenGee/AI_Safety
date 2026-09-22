#!/usr/bin/env python
"""Merge several sharded runs (see scripts/run_evaluation.py --shard-index/
--shard-count, and cluster/run_experiment.slurm's job array) back into one
combined raw_results.jsonl and one full report - the aggregation step each
shard itself deliberately skips, since one shard alone is only a
non-representative slice of the dataset (round-robin, see
intent_filter/sharding.py) and per-metric CIs/McNemar/unsafety-breakdown
computed over it would be misleading.

Usage:
    # Merge every shard dir matching the glob into one combined report:
    python scripts/merge_shards.py --shard-dirs "results/20260922_*_shard*of20" --output results/20260922_merged

    # Or list shard dirs explicitly:
    python scripts/merge_shards.py --shard-dirs results/run_shard0of4 results/run_shard1of4 --output results/merged
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.config import load_config  # noqa: E402
from intent_filter.environment import load_safety_rules  # noqa: E402
from intent_filter.evaluation import load_raw_results, record_to_json_line, write_full_report  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shard-dirs", nargs="+", required=True,
        help="One or more results/<run> directories to merge, or glob pattern(s) matching several "
        "(e.g. 'results/20260922_*_shard*of20'). Each must contain a raw_results.jsonl.",
    )
    parser.add_argument("--output", required=True, help="Directory to write the merged raw_results.jsonl + full report to.")
    parser.add_argument("--config", default=None, help="Path to config.yaml. Default: config/config.yaml, falling back to config.example.yaml.")
    return parser


def _resolve_shard_dirs(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            if Path(pattern).exists():
                matches = [pattern]
            else:
                print(f"Warning: no directories matched {pattern!r}", file=sys.stderr)
                continue
        resolved.extend(Path(m) for m in matches)
    return resolved


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    rule_base = load_safety_rules(config.environment.safety_rules_path)

    shard_dirs = _resolve_shard_dirs(args.shard_dirs)
    if not shard_dirs:
        print("No shard directories found - nothing to merge.", file=sys.stderr)
        return 1

    all_records = []
    seen_combos: set[tuple[str, str, int]] = set()
    for shard_dir in shard_dirs:
        raw_path = shard_dir / "raw_results.jsonl"
        if not raw_path.exists():
            print(f"Warning: {raw_path} not found, skipping {shard_dir}.", file=sys.stderr)
            continue
        records = load_raw_results(raw_path)
        # Round-robin sharding (intent_filter.sharding.shard_slice) guarantees
        # disjoint example sets per shard, so a duplicate combo here means
        # the same shard was merged twice, or --shard-count didn't match
        # across a set of runs - worth failing loudly rather than silently
        # double-counting a record in the merged report.
        for r in records:
            combo = (r.system, r.example_id, r.repeat_index)
            if combo in seen_combos:
                print(f"Error: duplicate run {combo} found across shard dirs - refusing to merge "
                      f"(check --shard-dirs covers each shard exactly once, with a consistent "
                      f"--shard-count).", file=sys.stderr)
                return 1
            seen_combos.add(combo)
        print(f"  {shard_dir}: {len(records)} record(s)")
        all_records.extend(records)

    print(f"Merged {len(all_records)} record(s) from {len(shard_dirs)} shard(s).")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "raw_results.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(record_to_json_line(r) + "\n")

    records_by_system: dict[str, list] = {}
    for r in all_records:
        records_by_system.setdefault(r.system, []).append(r)

    full_report = write_full_report(records_by_system, rule_base, config.evaluation.confidence_level, output_dir)

    print(f"\nMerged report written to {output_dir}")
    print(f"\n{'System':<20}{'Recall':>10}{'Precision':>12}{'Specificity':>13}{'F1':>8}{'FRR':>8}{'ClarifyAcc':>12}")
    for system, report in full_report.reports.items():
        def fmt(name: str) -> str:
            ci = report.metric_cis.get(name)
            return f"{ci.mean:.3f}" if ci else "n/a"

        print(
            f"{system:<20}{fmt('recall'):>10}{fmt('precision'):>12}{fmt('specificity'):>13}"
            f"{fmt('f1'):>8}{fmt('false_rejection_rate'):>8}{fmt('clarification_accuracy'):>12}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
