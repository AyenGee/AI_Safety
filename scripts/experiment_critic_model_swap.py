#!/usr/bin/env python
"""Small-scale experiment: does using a stronger model (gemma4:e4b, the
larger of the two available open-weight models) for the Critic - in place
of the faster qwen3.5:4b the reported systems use - fix the
property-hallucination false rejects that two prompt-engineering attempts
(the grounding and fact-injection experiments, both negative results - see
docs/methodology.md) could not?

Not part of the reported Phase 8 systems - config.models.critic stays
qwen3.5:4b there; this experiment only swaps the model string passed to
critic.review() for a comparison run, no pipeline code changes (review()
already takes `model` as a parameter).

Same curated set (extended to ~100, see build_sample() below) as the
grounding/fact-injection experiments, for direct comparability across all
three attempts. For each example the Planner is called ONCE and the Critic
called TWICE on that identical Planner output - once with the reported
qwen3.5:4b, once with gemma4:e4b - isolating any difference to the model
swap rather than Planner variance.

Usage:
    python scripts/experiment_critic_model_swap.py
    python scripts/experiment_critic_model_swap.py --shard-index 0 --shard-count 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import OllamaLLMClient  # noqa: E402
from intent_filter.agents.critic import review  # noqa: E402
from intent_filter.agents.planner import plan  # noqa: E402
from intent_filter.config import load_config  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.sharding import shard_slice  # noqa: E402

CURATED_IDS = [
    "legit_007",  # book/guest - known hallucination (wrongly called private)
    "legit_059",  # remote_control/guest - known hallucination (wrongly called private)
    "legit_008",  # toy/guest - known hallucination/false-reject
    "legit_001",  # knife/kitchen - control: clearly safe
    "unsafe_012",  # medication/guest - control: real private+dangerous item
    "unsafe_064",  # wallet/guest - control: real private-only item (disentangled)
    "unsafe_001",  # knife/child_room - control: real sharp+dangerous item
    "unsafe_022",  # scissors/child_room - control: real sharp-only item (disentangled)
    "unsafe_035",  # cleaning_spray/child_room - control: real dangerous-only item (disentangled)
]

DISENTANGLED_OBJECTS = (
    "book", "remote control", "wallet", "scissors", "cleaning spray",
    "matches", "space heater", "toy",
)
TARGET_TOTAL = 100

BASELINE_MODEL = "qwen3.5:4b"
STRONGER_MODEL = "gemma4:e4b"


def build_sample(all_examples: list) -> list[str]:
    sample = list(CURATED_IDS)
    seen = set(sample)
    for e in sorted(all_examples, key=lambda e: e.id):
        if e.id in seen:
            continue
        if any(obj in e.instruction_text.lower() for obj in DISENTANGLED_OBJECTS):
            sample.append(e.id)
            seen.add(e.id)
    remaining_by_category: dict[str, list[str]] = {}
    for e in sorted(all_examples, key=lambda e: e.id):
        if e.id not in seen:
            remaining_by_category.setdefault(e.category, []).append(e.id)
    need = TARGET_TOTAL - len(sample)
    categories = list(remaining_by_category)
    while need > 0 and any(remaining_by_category.values()):
        for category in categories:
            ids = remaining_by_category.get(category, [])
            if not ids or need <= 0:
                continue
            sample.append(ids.pop(0))
            need -= 1
    return sample


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    all_examples = load_dataset(config.dataset.path)
    examples_by_id = {e.id: e for e in all_examples}
    client = OllamaLLMClient(base_url=config.ollama.base_url, timeout=config.ollama.timeout, max_retries=config.ollama.max_retries)

    sample_ids = build_sample(all_examples)
    shard_suffix = ""
    if args.shard_index is not None:
        sample_ids = shard_slice(sample_ids, args.shard_index, args.shard_count)
        shard_suffix = f"_shard{args.shard_index}of{args.shard_count}"
    print(f"Sample size: {len(sample_ids)} unique instructions, 1 run each.")

    results = []
    for ex_id in sample_ids:
        example = examples_by_id[ex_id]
        state = example.scene_context.to_world_state(ontology)

        try:
            planner_output = plan(client, config.models.planner, example.instruction_text, state, ontology)
            baseline = review(
                client, BASELINE_MODEL, example.instruction_text, planner_output,
                state, ontology, rule_base, config.agent.ambiguity_margin,
            )
            stronger = review(
                client, STRONGER_MODEL, example.instruction_text, planner_output,
                state, ontology, rule_base, config.agent.ambiguity_margin,
            )
        except Exception as exc:  # noqa: BLE001 - one bad example shouldn't abort the batch
            print(f"[{ex_id}] ERROR: {type(exc).__name__}: {exc}")
            results.append({"id": ex_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        expected = "accept" if example.gold_label == "Accept" else "reject"
        b_mark = "OK" if baseline.decision == expected else ("CLARIFY" if baseline.decision == "clarify" else "WRONG")
        s_mark = "OK" if stronger.decision == expected else ("CLARIFY" if stronger.decision == "clarify" else "WRONG")
        changed = baseline.decision != stronger.decision

        results.append({
            "id": ex_id,
            "instruction": example.instruction_text,
            "gold_label": example.gold_label,
            "planner_top_actions": [repr(a) for a in planner_output.top.actions],
            "baseline_model_decision": baseline.decision,
            "baseline_model_rationale": baseline.rationale,
            "stronger_model_decision": stronger.decision,
            "stronger_model_rationale": stronger.rationale,
            "changed": changed,
        })
        print(f"[{ex_id}] gold={example.gold_label:8s} {BASELINE_MODEL}={baseline.decision:7s}({b_mark})  "
              f"{STRONGER_MODEL}={stronger.decision:7s}({s_mark})" + ("  <-- CHANGED" if changed else ""))

    output_path = Path("results") / f"critic_model_swap_experiment{shard_suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
