#!/usr/bin/env python
"""Translator formula accuracy: does the NL->LTL Translator's per-instruction
formula (logged on every LTL-augmented run but never used to gate a
decision - see intent_filter/decision.py's module docstring) actually draw
the same accept/reject line the reported system's fixed rule base draws?

No existing data answers this: RunRecord (intent_filter/evaluation/types.py)
never persists PipelineResult.stages, so the Translator's ltl_formula was
computed live during every Phase 8 run but never written to raw_results.jsonl
- there is nothing to retroactively compute this from.

Method: translate the instruction (1 LLM call) and separately plan it (1
LLM call, to get a concrete action sequence to build a trajectory from -
the Translator's formula and the rule's own formula are checked against the
exact same trajectory, so this is an apples-to-apples semantic comparison,
not a string comparison). "Correct" = the translated formula's SAT/UNSAT
verdict on that trajectory matches the rule's own formula's verdict on the
same trajectory (verify_state_trajectory, intent_filter/verifier/verifier.py)
- i.e. does the translated formula behave the same way the intended rule
does on the case it was designed for, not "is it textually identical".

Scaled to ~100 cases (up from 16-20) in the open-weight-model scale-up
round: each rule's own canonical violating_example/safe_example
(config/safety_rules.yaml) is always included, extended with up to 4 more
Reject-linked and 4 more Accept-linked instructions per rule drawn directly
from the (now 450-row, mechanically audited - see
scripts/audit_golden_labels.py) main dataset via related_rule_ids. This
reuses already-verified content instead of hand-authoring 100 new sentences
from scratch, and is if anything a MORE realistic test than the canonical
examples alone - real dataset rows carry real scene_context (room, held
objects, role) rather than a bare initial_state with only the role field set.

~100 instructions x 2 calls (translate + plan) = ~200 calls.

Usage:
    python scripts/experiment_translator_accuracy.py
    python scripts/experiment_translator_accuracy.py --shard-index 0 --shard-count 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import OllamaLLMClient  # noqa: E402
from intent_filter.agents.planner import plan  # noqa: E402
from intent_filter.agents.translator import translate  # noqa: E402
from intent_filter.config import load_config  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.environment import (  # noqa: E402
    apply_sequence,
    initial_state,
    load_ontology,
    load_safety_rules,
)
from intent_filter.environment.actions import InvalidActionError  # noqa: E402
from intent_filter.sharding import shard_slice  # noqa: E402
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
N_DATASET_CASES_PER_SIDE = 4  # additional Reject- and Accept-linked dataset rows, per rule


class Case:
    """One translator-accuracy test case. `example_row` is None for the two
    canonical safety_rules.yaml examples (built from a bare initial_state +
    `role`); set to a real InstructionExample for dataset-sourced cases
    (built from that row's own scene_context instead - `role` unused then)."""

    def __init__(self, rule, kind, instruction, expected, role="owner", example_row=None):
        self.rule = rule
        self.kind = kind
        self.instruction = instruction
        self.expected = expected
        self.role = role
        self.example_row = example_row

    def build_state(self, ontology):
        if self.example_row is not None:
            return self.example_row.scene_context.to_world_state(ontology)
        return initial_state(ontology, issuing_role=self.role)


def _cases_for_rule(rule, dataset) -> list[Case]:
    role_map = ROLE_OVERRIDE.get(rule.id, {})
    cases = [
        Case(rule, "canonical_violating", rule.violating_example, "UNSAT", role=role_map.get("violating", "owner")),
        Case(rule, "canonical_safe", rule.safe_example, "SAT", role=role_map.get("safe", "owner")),
    ]
    linked = sorted((e for e in dataset if rule.id in e.related_rule_ids), key=lambda e: e.id)
    for e in [e for e in linked if e.gold_label == "Reject"][:N_DATASET_CASES_PER_SIDE]:
        cases.append(Case(rule, f"dataset_violating:{e.id}", e.instruction_text, "UNSAT", example_row=e))
    for e in [e for e in linked if e.gold_label == "Accept"][:N_DATASET_CASES_PER_SIDE]:
        cases.append(Case(rule, f"dataset_safe:{e.id}", e.instruction_text, "SAT", example_row=e))
    return cases


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
    dataset = load_dataset(config.dataset.path)
    client = OllamaLLMClient(base_url=config.ollama.base_url, timeout=config.ollama.timeout, max_retries=config.ollama.max_retries)

    all_cases: list[Case] = []
    for rule in rule_base.rules:
        all_cases.extend(_cases_for_rule(rule, dataset))

    shard_suffix = ""
    if args.shard_index is not None:
        all_cases = shard_slice(all_cases, args.shard_index, args.shard_count)
        shard_suffix = f"_shard{args.shard_index}of{args.shard_count}"
    print(f"Running {len(all_cases)} translator-accuracy cases.\n")

    results = []
    n_correct = 0
    n_total = 0
    n_fallback = 0
    n_plan_invalid = 0
    n_ground_truth_mismatch = 0

    for case in all_cases:
        rule, rule_id, kind, instruction, expected = case.rule, case.rule.id, case.kind, case.instruction, case.expected
        state = case.build_state(ontology)

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
                f"this example doesn't exercise the rule as intended"
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
    print(f"    rule's own formula didn't match expected on its own example: "
          f"{n_ground_truth_mismatch}")

    output_path = Path("results") / f"translator_accuracy_experiment{shard_suffix}.json"
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
