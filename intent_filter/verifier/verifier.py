"""The deterministic LTLf verifier.

Interface: given an LTL formula (in the human-readable atom vocabulary used
by config/safety_rules.yaml) and a finite trace of atomic-proposition
snapshots (one per state in a candidate trajectory), decide SAT / UNSAT /
UNKNOWN, and - on UNSAT - explain which atoms were true at which step,
for the Critic's reprompting-loop feedback (Phase 5).

This module is deliberately non-LLM: it is the deterministic ground-truth
check the whole research question hinges on, so it must be simple enough to
read top to bottom and trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from flloat.ltlf import LTLfAlways

from intent_filter.environment.actions import Action
from intent_filter.environment.actions import apply_sequence as apply_action_sequence
from intent_filter.environment.ontology import Ontology
from intent_filter.environment.rules import SafetyRule, SafetyRuleBase
from intent_filter.environment.state import WorldState, derived_propositions
from intent_filter.verifier.atoms import build_atom_map, desanitize_atom, sanitize_formula, sanitize_trace
from intent_filter.verifier.formula import LTLFormulaError, parse_formula


class VerificationResult(Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VerificationOutcome:
    result: VerificationResult
    formula: str
    rule_id: str | None = None
    violating_step: int | None = None
    violating_atoms: tuple[str, ...] = field(default_factory=tuple)
    explanation: str | None = None


def _explain_violation(
    formula, sanitized_trace: list[dict[str, bool]], atom_map: dict[str, str]
) -> tuple[int | None, tuple[str, ...]]:
    """Best-effort localization of *where* and *why* an UNSAT formula failed.

    All rules in config/safety_rules.yaml are of the shape `G(phi)` for a
    purely propositional `phi` (no nested temporal operators), so for that
    shape we can point at the exact step and atoms that violated `phi` by
    evaluating the inner sub-formula (`formula.f`) at each step directly.
    For any other formula shape (e.g. a translator-produced formula using F/U
    in Phase 4), we fall back to reporting only that the trace as a whole is
    UNSAT, since "which step is at fault" isn't well-defined in general for
    finite-trace LTL without a fuller diagnostic pass.
    """
    if not isinstance(formula, LTLfAlways) or not sanitized_trace:
        return None, ()

    inner = formula.f
    referenced_atoms = inner.find_labels()
    for step_index, step_aps in enumerate(sanitized_trace):
        if not inner.truth(sanitized_trace, step_index):
            true_atoms = tuple(
                sorted(
                    desanitize_atom(atom, atom_map)
                    for atom in referenced_atoms
                    if step_aps.get(atom, False)
                )
            )
            return step_index, true_atoms
    return None, ()  # pragma: no cover - unreachable if formula.truth() reported UNSAT


def verify_trace(
    ltl_text: str,
    ap_trace: list[dict[str, bool]],
    atom_map: dict[str, str] | None = None,
    rule_id: str | None = None,
) -> VerificationOutcome:
    """Check whether `ltl_text` holds over `ap_trace` (a list of AP snapshots, step 0..n).

    Returns UNKNOWN rather than raising if the formula fails to parse, so
    callers (e.g. the Phase 4 NL->LTL translator's output, or a malformed
    rule) can route to a safe default (reject) instead of crashing the
    pipeline on bad input.
    """
    atom_map = atom_map or {}
    sanitized_text = sanitize_formula(ltl_text, atom_map)

    try:
        formula = parse_formula(sanitized_text)
    except LTLFormulaError as exc:
        return VerificationOutcome(
            result=VerificationResult.UNKNOWN,
            formula=ltl_text,
            rule_id=rule_id,
            explanation=str(exc),
        )

    sanitized_trace = sanitize_trace(ap_trace, atom_map)
    satisfied = formula.truth(sanitized_trace, 0)

    if satisfied:
        return VerificationOutcome(
            result=VerificationResult.SAT, formula=ltl_text, rule_id=rule_id
        )

    violating_step, violating_atoms = _explain_violation(formula, sanitized_trace, atom_map)
    if violating_step is not None:
        explanation = (
            f"Violated at step {violating_step}: "
            f"{', '.join(f'{atom}=True' for atom in violating_atoms) or '(no atoms true)'}"
        )
    else:
        explanation = "Formula is not satisfied over the given trace."

    return VerificationOutcome(
        result=VerificationResult.UNSAT,
        formula=ltl_text,
        rule_id=rule_id,
        violating_step=violating_step,
        violating_atoms=violating_atoms,
        explanation=explanation,
    )


def verify_state_trajectory(
    ltl_text: str,
    trajectory: list[WorldState],
    ontology: Ontology,
    atom_map: dict[str, str] | None = None,
    rule_id: str | None = None,
) -> VerificationOutcome:
    """Convenience wrapper: derive the AP trace from a WorldState trajectory, then verify."""
    atom_map = atom_map if atom_map is not None else build_atom_map(ontology)
    ap_trace = [derived_propositions(state, ontology) for state in trajectory]
    return verify_trace(ltl_text, ap_trace, atom_map=atom_map, rule_id=rule_id)


def verify_action_sequence(
    ltl_text: str,
    initial: WorldState,
    actions: list[Action],
    ontology: Ontology,
    atom_map: dict[str, str] | None = None,
    rule_id: str | None = None,
) -> VerificationOutcome:
    """Convenience wrapper: execute `actions` from `initial`, then verify the resulting trajectory."""
    trajectory = apply_action_sequence(initial, actions, ontology)
    return verify_state_trajectory(
        ltl_text, trajectory, ontology, atom_map=atom_map, rule_id=rule_id
    )


def check_rule_base(
    rule_base: SafetyRuleBase,
    trajectory: list[WorldState],
    ontology: Ontology,
) -> dict[str, VerificationOutcome]:
    """Check every rule in `rule_base` against `trajectory`, keyed by rule id.

    This is what the Decision Layer (Phase 5) calls: a candidate plan is safe
    only if every rule in the base returns SAT.
    """
    atom_map = build_atom_map(ontology)
    return {
        rule.id: verify_state_trajectory(
            rule.ltl, trajectory, ontology, atom_map=atom_map, rule_id=rule.id
        )
        for rule in rule_base
    }


def overall_result(outcomes: dict[str, VerificationOutcome]) -> VerificationResult:
    """Combine per-rule outcomes: UNSAT if any rule is violated, else UNKNOWN if any
    rule couldn't be evaluated (and none were violated), else SAT.
    """
    results = [outcome.result for outcome in outcomes.values()]
    if VerificationResult.UNSAT in results:
        return VerificationResult.UNSAT
    if VerificationResult.UNKNOWN in results:
        return VerificationResult.UNKNOWN
    return VerificationResult.SAT


def violated_rules(outcomes: dict[str, VerificationOutcome]) -> tuple[SafetyRule | str, ...]:
    """Return the rule ids among `outcomes` whose result is UNSAT."""
    return tuple(
        rule_id
        for rule_id, outcome in outcomes.items()
        if outcome.result is VerificationResult.UNSAT
    )
