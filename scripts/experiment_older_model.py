#!/usr/bin/env python
"""Does the model behind Planner/Translator/single_llm change the safety
findings? Compares the current default (gemma4:e4b) against the smaller,
faster alternative (qwen3.5:4b) for the Planner/Translator/single_llm
roles, live on both sides - re-runs a stratified subset of the dataset
through all 4 core systems with each model config.

Originally this compared a live run against an archived claude-sonnet-4-5
vs. claude-sonnet-5 (one generation apart) result set - not meaningful once
the project moved off Anthropic models entirely (see
docs/methodology.md "Scaling to open-weight models"). Re-scoped to the
model-capability axis actually available now: gemma4:e4b (9.6GB, the
larger of the two Ollama models) vs. qwen3.5:4b (3.4GB) for these 3 roles.
The Critic stays on config.models.critic (qwen3.5:4b) unchanged in both
arms, matching the original design's "isolate one role's model choice."

Scoped to legitimate/unsafe/misdirected only (no ambiguous), matching this
research's safety focus. Scaled to ~100 examples (up from 16) in the
open-weight-model scale-up round, run-once (no repeats) - both arms are
live LLM calls now, so this is twice the call volume of a single-arm
100-example run.

Usage:
    python scripts/experiment_older_model.py
    python scripts/experiment_older_model.py --shard-index 0 --shard-count 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import OllamaLLMClient  # noqa: E402
from intent_filter.config import ModelsConfig, load_config  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.sharding import shard_slice  # noqa: E402
from intent_filter.systems import SYSTEMS  # noqa: E402

# known-interesting cases from prior experiments, always included
SEED_IDS = ["legit_007", "legit_059", "legit_008", "misd_029"]
CATEGORIES = ("legitimate", "unsafe", "misdirected")
TARGET_TOTAL = 100

CURRENT_MODEL = "gemma4:e4b"
LEANER_MODEL = "qwen3.5:4b"
SYSTEM_NAMES = ["single_llm", "multi_agent", "single_llm_ltl", "multi_agent_ltl"]


def build_sample(all_examples: list) -> list[str]:
    sample = list(SEED_IDS)
    seen = set(sample)
    by_category: dict[str, list[str]] = {}
    for e in sorted(all_examples, key=lambda e: e.id):
        if e.category in CATEGORIES and e.id not in seen:
            by_category.setdefault(e.category, []).append(e.id)

    need = TARGET_TOTAL - len(sample)
    while need > 0 and any(by_category.values()):
        for category in CATEGORIES:
            ids = by_category.get(category, [])
            if not ids or need <= 0:
                continue
            sample.append(ids.pop(0))
            seen.add(sample[-1])
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
    print(f"Sample size: {len(sample_ids)} instructions, 1 run each per arm.")

    current_models = ModelsConfig(
        planner=CURRENT_MODEL, critic=config.models.critic, translator=CURRENT_MODEL, single_llm=CURRENT_MODEL,
    )
    leaner_models = ModelsConfig(
        planner=LEANER_MODEL, critic=config.models.critic, translator=LEANER_MODEL, single_llm=LEANER_MODEL,
    )
    ctx_current = SystemContext(
        client=client, models=current_models, ontology=ontology, rule_base=rule_base,
        ambiguity_margin=config.agent.ambiguity_margin,
        max_refinement_attempts=config.agent.max_refinement_attempts,
        translation_max_retries=config.agent.translation_max_retries,
    )
    ctx_leaner = SystemContext(
        client=client, models=leaner_models, ontology=ontology, rule_base=rule_base,
        ambiguity_margin=config.agent.ambiguity_margin,
        max_refinement_attempts=config.agent.max_refinement_attempts,
        translation_max_retries=config.agent.translation_max_retries,
    )

    results = []
    n_diff = 0
    n_total = 0
    for ex_id in sample_ids:
        example = examples_by_id[ex_id]
        state = example.scene_context.to_world_state(ontology)
        print(f"[{ex_id}] gold={example.gold_label:8s} '{example.instruction_text[:55]}'")

        row = {"id": ex_id, "instruction": example.instruction_text, "gold_label": example.gold_label}
        for system_name in SYSTEM_NAMES:
            try:
                current_result = SYSTEMS[system_name](example.instruction_text, state, ctx_current)
                current_decision = current_result.decision
            except Exception as exc:  # noqa: BLE001
                current_decision = f"ERROR:{type(exc).__name__}"

            try:
                leaner_result = SYSTEMS[system_name](example.instruction_text, state, ctx_leaner)
                leaner_decision = leaner_result.decision
                row[system_name] = {
                    f"{CURRENT_MODEL}_decision": current_decision,
                    f"{LEANER_MODEL}_decision": leaner_decision,
                    f"{LEANER_MODEL}_rationale": leaner_result.rationale,
                }
            except Exception as exc:  # noqa: BLE001
                leaner_decision = f"ERROR:{type(exc).__name__}"
                row[system_name] = {f"{CURRENT_MODEL}_decision": current_decision, f"{LEANER_MODEL}_decision": leaner_decision}

            n_total += 1
            diff = leaner_decision != current_decision
            if diff:
                n_diff += 1
            print(f"    {system_name:16s} {CURRENT_MODEL}={current_decision:8s} {LEANER_MODEL}={leaner_decision:8s}" +
                  ("  <-- DIFFERENT" if diff else ""))
        results.append(row)
        print()

    print(f"=== Summary: {n_diff}/{n_total} (system, example) pairs differ between "
          f"{CURRENT_MODEL} and {LEANER_MODEL} ===")

    output_path = Path("results") / f"older_model_experiment{shard_suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"n_diff": n_diff, "n_total": n_total}, "results": results}, f, indent=2)
    print(f"Full comparison written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
