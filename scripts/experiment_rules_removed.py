#!/usr/bin/env python
"""Rules-removed ablation: how do the four systems perform on a subset of
the existing legitimate/unsafe/misdirected dataset rows when the safety
rule base is completely empty?

With no rules loaded, `describe_safety_rules()` renders an empty policy
section in the Critic's and single_llm's system prompts - so any refusal
they still produce comes from the model's own general training, not from
any stated policy. For the two LTL-augmented systems, the verifier has
nothing to check a trajectory against, so it returns SAT unconditionally -
this isolates exactly what the deterministic verifier was contributing on
top of its non-LTL counterpart (single_llm_ltl vs single_llm, multi_agent_ltl
vs multi_agent) when rules exist: with none, that delta should vanish.

Scaled to 100 examples (34 legitimate / 33 unsafe / 33 misdirected -
ambiguous still excluded, since "safe/unsafe/misdirecting" gold labels are
what's being tested here) with 3 repeats, in the open-weight-model scale-up
round - up from the original 24 examples, single-repeat (cost-constrained
smoke-test size). 100 instructions x 4 systems x 3 repeats = 1200 total runs.

Usage:
    python scripts/experiment_rules_removed.py
    python scripts/experiment_rules_removed.py --shard-index 0 --shard-count 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import OllamaLLMClient  # noqa: E402
from intent_filter.config import load_config  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.environment import load_ontology  # noqa: E402
from intent_filter.environment.rules import SafetyRuleBase  # noqa: E402
from intent_filter.sharding import shard_slice  # noqa: E402
from intent_filter.systems import SYSTEMS  # noqa: E402

SYSTEM_NAMES = ["single_llm", "multi_agent", "single_llm_ltl", "multi_agent_ltl"]
N_PER_CATEGORY = {"legitimate": 34, "unsafe": 33, "misdirected": 33}
CATEGORIES = list(N_PER_CATEGORY)
REPEATS = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config()
    ontology = load_ontology(config.environment.ontology_path)
    empty_rule_base = SafetyRuleBase(rules=[])
    client = OllamaLLMClient(base_url=config.ollama.base_url, timeout=config.ollama.timeout, max_retries=config.ollama.max_retries)

    ctx = SystemContext(
        client=client,
        models=config.models,
        ontology=ontology,
        rule_base=empty_rule_base,
        ambiguity_margin=config.agent.ambiguity_margin,
        max_refinement_attempts=config.agent.max_refinement_attempts,
        translation_max_retries=config.agent.translation_max_retries,
    )

    dataset = load_dataset(config.dataset.path)
    subset = []
    for category in CATEGORIES:
        rows = sorted((r for r in dataset if r.category == category), key=lambda r: r.id)
        subset.extend(rows[: N_PER_CATEGORY[category]])

    shard_suffix = ""
    if args.shard_index is not None:
        subset = shard_slice(subset, args.shard_index, args.shard_count)
        shard_suffix = f"_shard{args.shard_index}of{args.shard_count}"

    print(f"Running {len(subset)} instructions x {args.repeats} repeat(s) with an EMPTY safety rule base.\n")

    tally = {s: {"correct": 0, "incorrect": 0} for s in SYSTEM_NAMES}
    per_category_tally = {
        c: {s: {"correct": 0, "incorrect": 0} for s in SYSTEM_NAMES} for c in CATEGORIES
    }
    results = []
    for repeat_index in range(args.repeats):
        for example in subset:
            state = example.scene_context.to_world_state(ontology)
            row = {
                "id": example.id,
                "repeat_index": repeat_index,
                "category": example.category,
                "gold_label": example.gold_label,
                "instruction": example.instruction_text,
            }
            for system_name in SYSTEM_NAMES:
                try:
                    result = SYSTEMS[system_name](example.instruction_text, state, ctx)
                    row[system_name] = {"decision": result.decision, "rationale": result.rationale}
                    bucket = "correct" if result.decision == example.gold_label else "incorrect"
                except Exception as exc:  # noqa: BLE001
                    row[system_name] = {"error": f"{type(exc).__name__}: {exc}"}
                    bucket = "incorrect"
                tally[system_name][bucket] += 1
                per_category_tally[example.category][system_name][bucket] += 1
            results.append(row)
            marks = "  ".join(f"{s}={row[s].get('decision', row[s].get('error', '?'))}" for s in SYSTEM_NAMES)
            print(f"[{example.id} / {example.category} / gold={example.gold_label} / repeat={repeat_index}] {marks}")

    print("\n=== Overall accuracy (rules removed, pooled across repeats) ===")
    for s in SYSTEM_NAMES:
        t = tally[s]
        total = t["correct"] + t["incorrect"]
        print(f"  {s:16s} {t['correct']}/{total} correct")

    print("\n=== Per-category accuracy ===")
    for c in CATEGORIES:
        print(f"  {c}:")
        for s in SYSTEM_NAMES:
            t = per_category_tally[c][s]
            total = t["correct"] + t["incorrect"]
            print(f"    {s:16s} {t['correct']}/{total} correct")

    output_path = Path("results") / f"rules_removed_experiment{shard_suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tally": tally, "per_category_tally": per_category_tally, "results": results}, f, indent=2)
    print(f"\nFull results written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
