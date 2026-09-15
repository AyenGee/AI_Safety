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

8 examples per category (legitimate/unsafe/misdirected - ambiguous excluded,
since "safe/unsafe/misdirecting" gold labels are what's being tested here),
taken as the first 8 by sorted id per category for a deterministic, unique
subset. 24 instructions x 4 systems = 96 calls, run once each.

Usage:
    python scripts/experiment_rules_removed.py
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
from intent_filter.environment import load_ontology  # noqa: E402
from intent_filter.environment.rules import SafetyRuleBase  # noqa: E402
from intent_filter.systems import SYSTEMS  # noqa: E402

SYSTEM_NAMES = ["single_llm", "multi_agent", "single_llm_ltl", "multi_agent_ltl"]
CATEGORIES = ["legitimate", "unsafe", "misdirected"]
N_PER_CATEGORY = 8


def main() -> int:
    config = load_config()
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    empty_rule_base = SafetyRuleBase(rules=[])
    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)

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
        subset.extend(rows[:N_PER_CATEGORY])
    print(f"Running {len(subset)} instructions ({N_PER_CATEGORY} each of {CATEGORIES}) "
          f"with an EMPTY safety rule base.\n")

    tally = {s: {"correct": 0, "incorrect": 0} for s in SYSTEM_NAMES}
    per_category_tally = {
        c: {s: {"correct": 0, "incorrect": 0} for s in SYSTEM_NAMES} for c in CATEGORIES
    }
    results = []
    for example in subset:
        state = example.scene_context.to_world_state(ontology)
        row = {
            "id": example.id,
            "category": example.category,
            "gold_label": example.gold_label,
            "instruction": example.instruction_text,
        }
        for system_name in SYSTEM_NAMES:
            result = SYSTEMS[system_name](example.instruction_text, state, ctx)
            row[system_name] = {"decision": result.decision, "rationale": result.rationale}
            bucket = "correct" if result.decision == example.gold_label else "incorrect"
            tally[system_name][bucket] += 1
            per_category_tally[example.category][system_name][bucket] += 1
        results.append(row)
        marks = "  ".join(f"{s}={row[s]['decision']}" for s in SYSTEM_NAMES)
        print(f"[{example.id} / {example.category} / gold={example.gold_label}] {marks}")

    print("\n=== Overall accuracy (rules removed) ===")
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

    output_path = Path("results") / "rules_removed_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tally": tally, "per_category_tally": per_category_tally, "results": results}, f, indent=2)
    print(f"\nFull results written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
