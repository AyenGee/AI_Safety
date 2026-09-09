#!/usr/bin/env python
"""Small-scale experiment: does forcing the Critic to explicitly look up an
object's actual listed properties before reasoning about rules reduce the
hallucinated-property false rejects found in Phase 8 (e.g. calling a `book`
- properties: none - a "private item")?

Not part of the reported Phase 8 systems - CRITIC_SYSTEM_PROMPT_GROUNDED_TEMPLATE
(intent_filter/agents/critic.py) is an experimental variant, opted into here via
`review(..., grounded=True)`, that none of the four systems or three ablations use.

For each curated example, the Planner is called ONCE and the Critic is then
called TWICE on that identical Planner output - once with the original
prompt, once with the grounded prompt - so any difference in the Critic's
decision is isolated to the prompt change itself, not Planner variance.

Usage:
    python scripts/experiment_grounded_critic.py
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

# 3 known hallucination false-rejects + 6 controls (2 per rule family, incl.
# the disentangled objects) + 1 clean-safe control. See conversation/report
# for the full rationale behind each inclusion.
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
            ungrounded = review(
                client, config.models.critic, example.instruction_text, planner_output,
                state, ontology, rule_base, config.agent.ambiguity_margin, grounded=False,
            )
            grounded = review(
                client, config.models.critic, example.instruction_text, planner_output,
                state, ontology, rule_base, config.agent.ambiguity_margin, grounded=True,
            )
        except Exception as exc:  # noqa: BLE001 - a transient parse failure on one example shouldn't abort the batch
            print(f"[{ex_id}] ERROR: {type(exc).__name__}: {exc}")
            results.append({"id": ex_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        row = {
            "id": ex_id,
            "instruction": example.instruction_text,
            "gold_label": example.gold_label,
            "planner_top_description": planner_output.top.description,
            "planner_top_actions": [repr(a) for a in planner_output.top.actions],
            "ungrounded_decision": ungrounded.decision,
            "ungrounded_rationale": ungrounded.rationale,
            "grounded_decision": grounded.decision,
            "grounded_rationale": grounded.rationale,
            "grounded_grounding": grounded.grounding,
            "changed": ungrounded.decision != grounded.decision,
        }
        results.append(row)

        expected = "accept" if example.gold_label == "Accept" else "reject"
        u_mark = "OK" if ungrounded.decision == expected else "WRONG"
        g_mark = "OK" if grounded.decision == expected else "WRONG"
        print(f"[{ex_id}] gold={example.gold_label:8s} ungrounded={ungrounded.decision:7s}({u_mark})  "
              f"grounded={grounded.decision:7s}({g_mark})" + ("  <-- CHANGED" if row["changed"] else ""))

    output_path = Path("results") / "grounding_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison (including rationale/grounding text) written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
