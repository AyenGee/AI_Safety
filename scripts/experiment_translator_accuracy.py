#!/usr/bin/env python
"""Translator formula accuracy: does the NL->LTL Translator's per-instruction
formula (logged on every LTL-augmented run but never used to gate a
decision - see intent_filter/decision.py's module docstring) actually draw
the same accept/reject line the reported system's fixed rule base draws?

No existing data answers this: RunRecord (intent_filter/evaluation/types.py)
never persists PipelineResult.stages, so the Translator's ltl_formula was
computed live during every Phase 8 run but never written to raw_results.jsonl
- there is nothing to retroactively compute this from.

Method: for each of the 8 reported rules' two canonical examples
(violating_example, safe_example - config/safety_rules.yaml), translate the
instruction (1 LLM call) and separately plan it (1 LLM call, to get a
concrete action sequence to build a trajectory from - the Translator's
formula and the rule's own formula are checked against the exact same
trajectory, so this is an apples-to-apples semantic comparison, not a
string comparison). "Correct" = the translated formula's SAT/UNSAT verdict
on that trajectory matches the rule's own formula's verdict on the same
trajectory (verify_state_trajectory, intent_filter/verifier/verifier.py) -
i.e. does the translated formula behave the same way the intended rule
does on the case it was designed for, not "is it textually identical".

16 instructions x 2 calls (translate + plan) = 32 calls.

Usage:
    python scripts/experiment_translator_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import AnthropicLLMClient  # noqa: E402
from intent_filter.agents.planner import plan  # noqa: E402
from intent_filter.agents.translator import translate  # noqa: E402
from intent_filter.config import load_config, load_secrets  # noqa: E402
from intent_filter.environment import (  # noqa: E402
    apply_sequence,
    initial_state,
    load_ontology,
    load_safety_rules,
)
from intent_filter.environment.actions import InvalidActionError  # noqa: E402
from intent_filter.verifier import VerificationResult, verify_state_trajectory  # noqa: E402

# (rule_id, example_text, expected_result, issuing_role) - the three
# role-gated rules' examples are written in safety_rules.yaml with a
# "(issued by X)" prefix that's metadata, not instruction text; stripped
# here and encoded as issuing_role instead, matching how the dataset
# actually represents it via SceneContext.issuing_role.
ROLE_OVERRIDE = {
    "no_medication_access_by_child": {"violating": "child", "safe": "owner"},
    "no_private_item_access_by_guest": {"violating": "guest", "safe": "guest"},
    "no_restricted_room_entry_by_guest": {"violating": "guest", "safe": "guest"},
}


def _cases(rule):
    role_map = ROLE_OVERRIDE.get(rule.id, {})
    yield (rule.id, "violating", rule.violating_example, "UNSAT", role_map.get("violating", "owner"))
    yield (rule.id, "safe", rule.safe_example, "SAT", role_map.get("safe", "owner"))


def main() -> int:
    config = load_config()
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)

    results = []
    n_correct = 0
    n_total = 0
    n_fallback = 0
    n_plan_invalid = 0
    n_ground_truth_mismatch = 0

    for rule in rule_base.rules:
        for rule_id, kind, instruction, expected, role in _cases(rule):
            state = initial_state(ontology, issuing_role=role)

            translation = translate(client, config.models.translator, instruction, ontology)
            planner_output = plan(client, config.models.planner, instruction, state, ontology)

            try:
                trajectory = apply_sequence(state, list(planner_output.top.actions), ontology)
            except InvalidActionError as exc:
                n_plan_invalid += 1
                results.append(
                    {
                        "rule_id": rule_id, "kind": kind, "instruction": instruction,
                        "error": f"planner produced a physically invalid sequence: {exc}",
                    }
                )
                print(f"[{rule_id}/{kind}] PLAN INVALID: {exc}")
                continue

            ground_truth = verify_state_trajectory(rule.ltl, trajectory, ontology)
            if ground_truth.result.value != expected:
                n_ground_truth_mismatch += 1
                print(
                    f"[{rule_id}/{kind}] WARNING: rule's own formula gave "
                    f"{ground_truth.result.value}, expected {expected} - Planner's plan for "
                    f"this canonical example doesn't exercise the rule as intended"
                )

            n_total += 1
            row = {
                "rule_id": rule_id,
                "kind": kind,
                "instruction": instruction,
                "expected": expected,
                "ground_truth_result": ground_truth.result.value,
                "translated_formula": translation.ltl_formula,
                "translation_success": translation.success,
                "used_fallback": translation.used_fallback,
            }

            if not translation.success or translation.ltl_formula is None:
                row["translated_result"] = None
                row["correct"] = False
            else:
                if translation.used_fallback:
                    n_fallback += 1
                translated_outcome = verify_state_trajectory(
                    translation.ltl_formula, trajectory, ontology
                )
                row["translated_result"] = translated_outcome.result.value
                row["correct"] = translated_outcome.result == VerificationResult(expected)

            if row["correct"]:
                n_correct += 1
            results.append(row)
            mark = "OK" if row["correct"] else "WRONG"
            print(f"[{rule_id}/{kind}] expected={expected} translated={row.get('translated_result')} "
                  f"[{mark}]  formula={translation.ltl_formula!r}")

    print(f"\n=== Translator formula accuracy: {n_correct}/{n_total} "
          f"({100 * n_correct / n_total:.1f}%) ===")
    print(f"    used_fallback (template, not LLM translation): {n_fallback}/{n_total}")
    print(f"    planner produced an invalid sequence (excluded above): {n_plan_invalid}")
    print(f"    rule's own formula didn't match expected on its own canonical example: "
          f"{n_ground_truth_mismatch}")

    output_path = Path("results") / "translator_accuracy_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_correct": n_correct,
                "n_total": n_total,
                "n_fallback": n_fallback,
                "n_plan_invalid": n_plan_invalid,
                "n_ground_truth_mismatch": n_ground_truth_mismatch,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nFull results written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
