"""Atom vocabulary and name sanitization for the LTLf verifier.

flloat's grammar treats parentheses purely as grouping syntax, so
function-style atom names like `agent_at(child_room)` (used in
config/safety_rules.yaml and produced by
intent_filter.environment.state.derived_propositions for readability) are
not valid flloat atom tokens - `agent_at(child_room)` parses as an atom
`agent_at` immediately followed by an unexpected `(`. This module builds the
fixed mapping between the ontology's human-readable, grounded atom names and
flat identifiers flloat can parse, and applies it consistently to both
formula strings and trace/AP dictionaries so a verification result never
depends on which form was used to write the rule.

Only grounded, parenthesized atoms (agent_at(<room>), has_object(<object>),
issued_by(<role>)) need an entry in the map; the generic derived atoms
(door_locked, holds_sharp_item, at_child_zone, ...) are already valid flat
identifiers and pass through unchanged.
"""

from __future__ import annotations

import re

from intent_filter.environment.ontology import Ontology

_NON_IDENTIFIER_CHARS = re.compile(r"\W+")


def _flatten(atom: str) -> str:
    """Turn e.g. 'agent_at(child_room)' into 'agent_at__child_room'."""
    return _NON_IDENTIFIER_CHARS.sub("__", atom).strip("_")


def build_atom_map(ontology: Ontology) -> dict[str, str]:
    """Map every grounded AP name in `ontology`'s vocabulary to a flat identifier."""
    atom_map: dict[str, str] = {}
    for room_name in ontology.rooms:
        atom = f"agent_at({room_name})"
        atom_map[atom] = _flatten(atom)
    for obj_name in ontology.objects:
        atom = f"has_object({obj_name})"
        atom_map[atom] = _flatten(atom)
    for role_name in ontology.roles:
        atom = f"issued_by({role_name})"
        atom_map[atom] = _flatten(atom)
    return atom_map


def sanitize_formula(ltl_text: str, atom_map: dict[str, str]) -> str:
    """Replace every grounded atom occurrence in `ltl_text` with its flat identifier.

    Substring replacement (not regex-driven) is deliberate: grouping
    parentheses in the formula must be left untouched, and only exact atom
    occurrences from `atom_map` are rewritten. Longest-key-first ordering
    avoids one atom's text being a prefix of another's (not currently
    possible given the vocabulary, but keeps this correct if it grows).
    """
    result = ltl_text
    for atom, flat in sorted(atom_map.items(), key=lambda kv: -len(kv[0])):
        result = result.replace(atom, flat)
    return result


def sanitize_trace(
    ap_trace: list[dict[str, bool]], atom_map: dict[str, str]
) -> list[dict[str, bool]]:
    """Rename AP dict keys in a trace using `atom_map`, for feeding to flloat."""
    return [{atom_map.get(key, key): value for key, value in step.items()} for step in ap_trace]


def desanitize_atom(flat_name: str, atom_map: dict[str, str]) -> str:
    """Reverse-lookup a flat identifier back to its human-readable atom name.

    Used when building violation explanations, so a Critic's reprompting
    feedback can say `agent_at(child_room)` rather than `agent_at__child_room`.
    Falls through to `flat_name` for atoms that were never grounded (already
    human-readable, e.g. `holds_sharp_item`).
    """
    for atom, flat in atom_map.items():
        if flat == flat_name:
            return atom
    return flat_name
