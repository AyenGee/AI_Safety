"""Deterministic LTLf verifier.

Checks a candidate plan's resulting state trajectory against the safety rule
base (config/safety_rules.yaml) using finite-trace LTL (LTLf, via `flloat`).
See docs/methodology.md for the LTL-vs-LTLf formalism discussion.
"""

from intent_filter.verifier.atoms import build_atom_map, desanitize_atom, sanitize_formula, sanitize_trace
from intent_filter.verifier.formula import LTLFormulaError, parse_formula
from intent_filter.verifier.verifier import (
    VerificationOutcome,
    VerificationResult,
    check_rule_base,
    overall_result,
    verify_action_sequence,
    verify_state_trajectory,
    verify_trace,
    violated_rules,
)

__all__ = [
    "build_atom_map",
    "desanitize_atom",
    "sanitize_formula",
    "sanitize_trace",
    "LTLFormulaError",
    "parse_formula",
    "VerificationOutcome",
    "VerificationResult",
    "check_rule_base",
    "overall_result",
    "verify_action_sequence",
    "verify_state_trajectory",
    "verify_trace",
    "violated_rules",
]
