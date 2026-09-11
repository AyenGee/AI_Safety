#!/usr/bin/env python
"""Does the model generation behind Planner/Translator/single_llm change
the safety findings? Re-runs a small, stratified subset of the existing
200-example dataset (legitimate/unsafe/misdirected only, matching this
research's safety scope - no ambiguous) through all 4 core systems with
claude-sonnet-4-5 (one full generation older than claude-sonnet-5) in place
of claude-sonnet-5 for the Planner/Translator/single_llm roles. The Critic
stays on claude-haiku-4-5 unchanged - there's no good "older Haiku" option
currently active (claude-haiku-3 is deprecated with an April 2026 retirement
date already passed as of this writing; everything before it is retired).

Compares each live older-model prediction directly against the existing,
already-collected claude-sonnet-5 prediction for the exact same example at
repeat_index=0 (results/20260908_085406_corrected/raw_results.jsonl) - no
need to re-run the current-model side, it's already on disk.

16-example sample: the 4 known-interesting cases from prior experiments
(legit_007/legit_059/legit_008 - property hallucination; misd_029 - the
temporal-inconsistency pattern) plus 4 deterministically-sampled additional
examples per category (legitimate/unsafe/misdirected), for a small, cheap,
run-once (no repeats) comparison.

Usage:
    python scripts/experiment_older_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import AnthropicLLMClient  # noqa: E402
from intent_filter.config import ModelsConfig, load_config, load_secrets  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.systems import SYSTEMS  # noqa: E402

SAMPLE_IDS = [
    # known-interesting cases from prior experiments
    "legit_007", "legit_059", "legit_008", "misd_029",
    # stratified additional sample, legitimate/unsafe/misdirected only (no ambiguous)
    "legit_001", "legit_014", "legit_035", "legit_057",
    "unsafe_001", "unsafe_021", "unsafe_041", "unsafe_062",
    "misd_001", "misd_013", "misd_025", "misd_037",
]

OLDER_MODEL = "claude-sonnet-4-5"
CURRENT_RESULTS = Path("results/20260908_085406_corrected/raw_results.jsonl")
SYSTEM_NAMES = ["single_llm", "multi_agent", "single_llm_ltl", "multi_agent_ltl"]


def load_current_predictions() -> dict:
    """(system, example_id) -> RunRecord dict, repeat_index == 0 only."""
    preds = {}
    with open(CURRENT_RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["repeat_index"] == 0:
                preds[(r["system"], r["example_id"])] = r
    return preds


def main() -> int:
    config = load_config()
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    examples_by_id = {e.id: e for e in load_dataset(config.dataset.path)}
    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)
    current_preds = load_current_predictions()

    older_models = ModelsConfig(
        planner=OLDER_MODEL,
        critic=config.models.critic,  # unchanged - claude-haiku-4-5
        translator=OLDER_MODEL,
        single_llm=OLDER_MODEL,
    )
    ctx = SystemContext(
        client=client,
        models=older_models,
        ontology=ontology,
        rule_base=rule_base,
        ambiguity_margin=config.agent.ambiguity_margin,
        max_refinement_attempts=config.agent.max_refinement_attempts,
        translation_max_retries=config.agent.translation_max_retries,
    )

    results = []
    n_diff = 0
    n_total = 0
    for ex_id in SAMPLE_IDS:
        example = examples_by_id[ex_id]
        state = example.scene_context.to_world_state(ontology)
        print(f"[{ex_id}] gold={example.gold_label:8s} '{example.instruction_text[:55]}'")

        row = {"id": ex_id, "instruction": example.instruction_text, "gold_label": example.gold_label}
        for system_name in SYSTEM_NAMES:
            current = current_preds.get((system_name, ex_id))
            current_decision = current["predicted_label"] if current else "n/a"

            try:
                result = SYSTEMS[system_name](example.instruction_text, state, ctx)
                older_decision = result.decision
                row[system_name] = {
                    "current_sonnet5": current_decision,
                    "older_sonnet4_5": older_decision,
                    "older_rationale": result.rationale,
                }
            except Exception as exc:  # noqa: BLE001
                older_decision = f"ERROR:{type(exc).__name__}"
                row[system_name] = {"current_sonnet5": current_decision, "older_sonnet4_5": older_decision}

            n_total += 1
            diff = older_decision != current_decision
            if diff:
                n_diff += 1
            print(f"    {system_name:16s} current={current_decision:8s} older={older_decision:8s}" +
                  ("  <-- DIFFERENT" if diff else ""))
        results.append(row)
        print()

    print(f"=== Summary: {n_diff}/{n_total} (system, example) pairs differ between "
          f"claude-sonnet-5 and claude-sonnet-4-5 ===")

    output_path = Path("results") / "older_model_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"n_diff": n_diff, "n_total": n_total}, "results": results}, f, indent=2)
    print(f"Full comparison written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
