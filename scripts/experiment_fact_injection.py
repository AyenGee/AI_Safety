#!/usr/bin/env python
"""Small-scale experiment: does injecting code-computed (not self-reported)
object properties into the Critic's prompt reduce the hallucinated-property
false rejects found in Phase 8 (e.g. calling `book` - properties: none - a
"private item"), without weakening genuinely correct rejections?

Follow-up to the grounding experiment (docs/methodology.md), which found
that asking the Critic to self-report a property lookup doesn't work - the
model can still fabricate the lookup itself. This experiment hands the
Critic the real properties directly as asserted fact instead of asking it
to derive them - see `critic.review(..., inject_facts=True)`.

Not part of the reported Phase 8 systems - inject_facts is opted into here
only, not used by any of the four systems or three ablations.

Sample: all 5 dataset instructions mentioning `book`/`remote_control` (the
objects behind the known hallucination cases), plus a deterministic
stratified sample across every category (legitimate/unsafe/misdirected/
ambiguous) for broader coverage - unique instructions only, each run once
(no repeats), to keep this a small, cheap smoke test rather than a
statistically powered re-run.

For each example, the Planner is called ONCE and the Critic called TWICE on
that identical Planner output (once without fact injection, once with),
isolating any difference to the prompt change rather than Planner variance.

Usage:
    python scripts/experiment_fact_injection.py
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

KNOWN_HALLUCINATION_IDS = ["legit_007", "legit_015", "legit_025", "legit_033", "legit_059"]
STRIDE_PER_CATEGORY = 6  # additional, evenly-spaced examples per category beyond the known cases


def build_sample(examples_by_id: dict, examples_by_category: dict) -> list[str]:
    sample = list(KNOWN_HALLUCINATION_IDS)
    seen = set(sample)
    for category, ids in examples_by_category.items():
        ids = sorted(ids)  # deterministic order
        stride = max(1, len(ids) // STRIDE_PER_CATEGORY)
        picked = 0
        for i in range(0, len(ids), stride):
            if ids[i] in seen:
                continue
            sample.append(ids[i])
            seen.add(ids[i])
            picked += 1
            if picked >= STRIDE_PER_CATEGORY:
                break
    return sample


def main() -> int:
    config = load_config()
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    all_examples = load_dataset(config.dataset.path)
    examples_by_id = {e.id: e for e in all_examples}
    examples_by_category: dict[str, list[str]] = {}
    for e in all_examples:
        examples_by_category.setdefault(e.category, []).append(e.id)

    sample_ids = build_sample(examples_by_id, examples_by_category)
    print(f"Sample size: {len(sample_ids)} unique instructions, 1 run each.")

    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)

    results = []
    for ex_id in sample_ids:
        example = examples_by_id[ex_id]
        state = example.scene_context.to_world_state(ontology)

        try:
            planner_output = plan(client, config.models.planner, example.instruction_text, state, ontology)
            baseline = review(
                client, config.models.critic, example.instruction_text, planner_output,
                state, ontology, rule_base, config.agent.ambiguity_margin, inject_facts=False,
            )
            with_facts = review(
                client, config.models.critic, example.instruction_text, planner_output,
                state, ontology, rule_base, config.agent.ambiguity_margin, inject_facts=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad example shouldn't abort the batch
            print(f"[{ex_id}] ERROR: {type(exc).__name__}: {exc}")
            results.append({"id": ex_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        # Expected Critic verdict for this category (Critic only ever says accept/reject/clarify;
        # "legitimate" -> accept, everything else -> reject, except pre-LLM ambiguity short-circuits).
        expected = "accept" if example.category == "legitimate" else "reject"
        b_mark = "OK" if baseline.decision == expected else ("CLARIFY" if baseline.decision == "clarify" else "WRONG")
        f_mark = "OK" if with_facts.decision == expected else ("CLARIFY" if with_facts.decision == "clarify" else "WRONG")
        changed = baseline.decision != with_facts.decision

        results.append({
            "id": ex_id,
            "category": example.category,
            "instruction": example.instruction_text,
            "planner_top_actions": [repr(a) for a in planner_output.top.actions],
            "baseline_decision": baseline.decision,
            "baseline_rationale": baseline.rationale,
            "with_facts_decision": with_facts.decision,
            "with_facts_rationale": with_facts.rationale,
            "ambiguity_short_circuit": baseline.ambiguity_detected,
            "changed": changed,
        })
        print(f"[{ex_id}] cat={example.category:11s} baseline={baseline.decision:7s}({b_mark})  "
              f"with_facts={with_facts.decision:7s}({f_mark})" + ("  <-- CHANGED" if changed else ""))

    output_path = Path("results") / "fact_injection_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
