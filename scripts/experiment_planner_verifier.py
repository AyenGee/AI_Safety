#!/usr/bin/env python
"""Evaluate the new planner_verifier system (Planner + Verifier only, strict
one-shot - see intent_filter/systems/planner_verifier.py) against the SAME
24-instruction subset used by scripts/experiment_rules_removed.py (first 8
by sorted id of each of legitimate/unsafe/misdirected), but with the real,
fixed safety rule base this time - the point here is to see how a system
with no LLM safety opinion at all performs when the verifier actually has
something to check against.

Using the identical subset means this is directly comparable to the
rules-removed run, and - since these are the same 200-dataset ids the
reported Phase 8 systems were evaluated on - directly comparable to the
existing results/20260908_085406/raw_results.jsonl too (pulled separately,
no re-running needed, since those numbers already exist).

24 instructions x 1 system x 1 repeat = 24 calls.

Usage:
    python scripts/experiment_planner_verifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import AnthropicLLMClient  # noqa: E402
from intent_filter.config import load_config, load_secrets  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.systems import planner_verifier  # noqa: E402

CATEGORIES = ["legitimate", "unsafe", "misdirected"]
N_PER_CATEGORY = 8


def main() -> int:
    config = load_config()
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
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

    dataset = load_dataset(config.dataset.path)
    subset = []
    for category in CATEGORIES:
        rows = sorted((r for r in dataset if r.category == category), key=lambda r: r.id)
        subset.extend(rows[:N_PER_CATEGORY])
    print(f"Running planner_verifier on {len(subset)} instructions "
          f"({N_PER_CATEGORY} each of {CATEGORIES}) with the real rule base.\n")

    tally = {"correct": 0, "incorrect": 0, "error": 0}
    per_category_tally = {c: {"correct": 0, "incorrect": 0, "error": 0} for c in CATEGORIES}
    results = []
    for example in subset:
        state = example.scene_context.to_world_state(ontology)
        try:
            result = planner_verifier.run(example.instruction_text, state, ctx)
        except Exception as exc:  # noqa: BLE001
            tally["error"] += 1
            per_category_tally[example.category]["error"] += 1
            results.append(
                {
                    "id": example.id,
                    "category": example.category,
                    "gold_label": example.gold_label,
                    "instruction": example.instruction_text,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{example.id} / {example.category} / gold={example.gold_label}] -> ERROR: {exc}")
            continue
        bucket = "correct" if result.decision == example.gold_label else "incorrect"
        tally[bucket] += 1
        per_category_tally[example.category][bucket] += 1
        results.append(
            {
                "id": example.id,
                "category": example.category,
                "gold_label": example.gold_label,
                "instruction": example.instruction_text,
                "decision": result.decision,
                "rationale": result.rationale,
                "chosen_actions": [repr(a) for a in result.chosen_actions],
            }
        )
        print(f"[{example.id} / {example.category} / gold={example.gold_label}] -> {result.decision}")

    print("\n=== Overall accuracy ===")
    total = tally["correct"] + tally["incorrect"] + tally["error"]
    print(f"  planner_verifier  {tally['correct']}/{total} correct  ({tally['error']} errors)")

    print("\n=== Per-category accuracy ===")
    for c in CATEGORIES:
        t = per_category_tally[c]
        ctotal = t["correct"] + t["incorrect"] + t["error"]
        print(f"  {c:12s} {t['correct']}/{ctotal} correct  ({t['error']} errors)")

    output_path = Path("results") / "planner_verifier_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tally": tally, "per_category_tally": per_category_tally, "results": results}, f, indent=2)
    print(f"\nFull results written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
