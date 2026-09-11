#!/usr/bin/env python
"""Small-scale experiment: does using a stronger model (claude-sonnet-5,
same generation as the Planner/Translator/single_llm) for the Critic - in
place of the cheaper claude-haiku-4-5 the reported systems use - fix the
property-hallucination false rejects that two prompt-engineering attempts
(the grounding and fact-injection experiments, both negative results - see
docs/methodology.md) could not?

Not part of the reported Phase 8 systems - config.models.critic stays
claude-haiku-4-5 there; this experiment only swaps the model string passed
to critic.review() for a `--critic-model` comparison run, no pipeline code
changes (review() already takes `model` as a parameter).

Same 9-example curated set as the grounding/fact-injection experiments (3
known hallucination cases + 6 controls spanning all 3 rule families,
including the Phase 7 disentangled objects, + 1 clean-safe control), for
direct comparability across all three attempts. For each example the
Planner is called ONCE and the Critic called TWICE on that identical
Planner output - once with the reported haiku-4-5, once with sonnet-5 -
isolating any difference to the model swap rather than Planner variance.

Usage:
    python scripts/experiment_critic_model_swap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import AnthropicLLMClient  # noqa: E402
from intent_filter.agents.critic import review  # noqa: E402
from intent_filter.agents.planner import plan  # noqa: E402
from intent_filter.config import load_config, load_secrets  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402

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

BASELINE_MODEL = "claude-haiku-4-5"
STRONGER_MODEL = "claude-sonnet-5"


def main() -> int:
    config = load_config()
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    examples_by_id = {e.id: e for e in load_dataset(config.dataset.path)}
    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)

    results = []
    for ex_id in CURATED_IDS:
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
            "haiku_decision": baseline.decision,
            "haiku_rationale": baseline.rationale,
            "sonnet_decision": stronger.decision,
            "sonnet_rationale": stronger.rationale,
            "changed": changed,
        })
        print(f"[{ex_id}] gold={example.gold_label:8s} haiku={baseline.decision:7s}({b_mark})  "
              f"sonnet={stronger.decision:7s}({s_mark})" + ("  <-- CHANGED" if changed else ""))

    output_path = Path("results") / "critic_model_swap_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
