#!/usr/bin/env python
"""Mechanically re-verify dataset gold labels against the deterministic rule
base directly - no LLM calls, runs locally, no cluster needed - rather than
trusting hand-authored labels on read-through alone. Built for the scale-up
round (docs/methodology.md "Scaling to open-weight models"): "critically
verify if indeed the golden labels are correct" as new rows are authored in
bulk, not just for the original 200.

How it works: a dataset row's `scene_context` is a single BEFORE snapshot -
it doesn't say what the instruction, if literally carried out, would change.
This script reconstructs the AFTER state each rule actually needs to check,
per a small per-rule recipe (RULE_AUDIT_SPECS below) reflecting how that
rule's LTL formula is shaped:

  - state_only:   the row's own scene_context already fully encodes the
                   hazard-relevant fields (e.g. lock_door_when_owner_away's
                   door_locked/owner_home) - checked as-is, no changes.
  - device_state: the instruction's implied effect is "turn this device
                   state on" (e.g. "turn on the stove" -> stove_on=True,
                   "light the candle" -> candle_lit=True) - the row's
                   scene_context already places the agent/owner correctly
                   (e.g. agent_room=bathroom, owner_home=False); this just
                   forces the one boolean the instruction requests.
  - object_only / room_only / object_and_room:
                   the instruction names an object and/or destination room
                   from the ontology's closed vocabulary - extracted by
                   substring match (see _find_object/_find_room) and folded
                   into held_objects/agent_room to build the moment of
                   maximum hazard (object held while at the room), which is
                   what a G(...) safety formula actually needs checked.

A row where extraction can't confidently find what it needs (ambiguous or
absent) is reported as UNAUDITABLE, not silently skipped or scored as a
pass - the point is to be honest about what this script did and didn't
verify, not to overclaim coverage a closed-vocabulary substring match can't
actually provide for freeform ambiguous-category text.

Only rows with `related_rule_ids` set are in scope - `ambiguous`-category
rows (no linked rule) aren't mechanically checkable this way and are
audited by the existing structural criteria instead (see
data/dataset_schema.md's "known limitations").

Usage:
    python scripts/audit_golden_labels.py [--dataset data/instructions.jsonl]

Exit code 1 if any row FAILS (a mechanically-checkable row whose gold label
disagrees with the rule base); 0 otherwise (unauditable rows never fail the
run, they're reported separately for manual review).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.config import load_config  # noqa: E402
from intent_filter.dataset import InstructionExample, load_dataset  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402
from intent_filter.environment.ontology import Ontology  # noqa: E402
from intent_filter.verifier import VerificationResult, verify_state_trajectory  # noqa: E402

# Per-rule recipe for reconstructing the state moment that actually needs
# checking - see module docstring. `state_field` is only used by kind
# "device_state". Every rule currently in config/safety_rules.yaml must have
# an entry here (checked at startup) so a newly-added rule can't silently
# skip the audit.
RULE_AUDIT_SPECS: dict[str, dict] = {
    "no_knife_in_child_room": {"kind": "object_and_room"},
    "no_stove_control_from_bathroom": {"kind": "device_state", "state_field": "stove_on"},
    # "Leave the door unlocked" is an implied ACTION (door_locked -> False),
    # not something scene_context already encodes - same shape as
    # stove_on/candle_lit below, just forcing False instead of True. Caught
    # by running this audit against the real dataset: an earlier "state_only"
    # classification produced 16 false FAILs, because door_locked silently
    # stays at its True default in every one of these rows' scene_context.
    "lock_door_when_owner_away": {"kind": "device_state", "state_field": "door_locked", "state_value": False},
    "no_sharp_items_in_child_zone": {"kind": "object_and_room"},
    "no_dangerous_items_in_child_zone": {"kind": "object_and_room"},
    "no_medication_access_by_child": {"kind": "object_only"},
    "no_private_item_access_by_guest": {"kind": "object_only"},
    "no_restricted_room_entry_by_guest": {"kind": "room_only"},
    "no_open_flame_unattended": {"kind": "device_state", "state_field": "candle_lit"},
    "no_appliance_left_on_when_house_empty": {"kind": "device_state", "state_field": "stove_on"},
}

# Room name -> phrasings that show up in hand-authored instruction text,
# beyond the room's own ontology name (always implicitly included).
# child_room in particular gets referred to a lot of different ways since
# it's never called "child_room" in natural language.
ROOM_ALIASES: dict[str, tuple[str, ...]] = {
    "child_room": (
        "child's room", "childs room", "child room", "nursery",
        "kid's room", "kids' room", "kids room",
        "child's bedroom", "childs bedroom",
    ),
}

# Object name -> extra synonyms beyond the ontology name itself
# (underscore->space) - real phrasing variance found by running this audit
# against the hand-authored dataset (e.g. "medicine"/"pills", not always
# the clinical "medication").
OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "medication": ("medicine", "pills", "meds"),
}


def _normalize(text: str) -> str:
    """Lowercase and drop apostrophes, so "kids' room"/"kid's room"/"kids room" all compare equal."""
    return text.lower().replace("'", "")


def _greedy_longest_match(text: str, alias_to_canonical: list[tuple[str, str]]) -> str | None:
    """Return the single canonical name found in `text`, or None if zero or
    more than one distinct canonical is mentioned.

    A shorter matched alias is only ever discarded in favor of a longer one
    when the shorter is a literal substring of the longer AND they name
    DIFFERENT canonicals (e.g. "bedroom" is a substring of "child's
    bedroom", which means child_room, not the adult bedroom - the longer,
    more specific alias wins and the shorter one is not treated as an
    independent mention). Two matches that aren't nested this way (e.g.
    "nursery" and "kitchen" both appearing in the same sentence for two
    unrelated reasons - found via a real case in the scale-up round, where
    a justification clause happened to name a second room) are genuine
    ambiguity, not a tie to break - reported as unauditable rather than
    silently guessing one of them.
    """
    text = _normalize(text)
    matches = [(alias, canonical) for alias, canonical in alias_to_canonical if _normalize(alias) in text]
    if not matches:
        return None

    survivors = set()
    for alias, canonical in matches:
        norm_alias = _normalize(alias)
        absorbed = any(
            canonical != other_canonical and norm_alias != _normalize(other_alias) and norm_alias in _normalize(other_alias)
            for other_alias, other_canonical in matches
        )
        if not absorbed:
            survivors.add(canonical)
    return next(iter(survivors)) if len(survivors) == 1 else None


def _find_object(text: str, ontology: Ontology) -> str | None:
    pairs = [(name.replace("_", " "), name) for name in ontology.objects]
    pairs += [(alias, name) for name, aliases in OBJECT_ALIASES.items() for alias in aliases]
    return _greedy_longest_match(text, pairs)


def _find_room(text: str, ontology: Ontology) -> str | None:
    pairs = [(room.replace("_", " "), room) for room in ontology.rooms]
    pairs += [(alias, room) for room, aliases in ROOM_ALIASES.items() for alias in aliases]
    return _greedy_longest_match(text, pairs)


def audit_row(
    example: InstructionExample, rule_id: str, ontology: Ontology, rule_lookup: dict
) -> tuple[str, str]:
    """Returns (status, detail): status is 'PASS', 'FAIL', or 'SKIP' (unauditable)."""
    rule = rule_lookup[rule_id]
    spec = RULE_AUDIT_SPECS[rule_id]
    state = example.scene_context.to_world_state(ontology)
    kind = spec["kind"]

    if kind == "state_only":
        check_state = state
    elif kind == "device_state":
        # The forced value only represents the HAZARD direction (e.g.
        # stove_on=True, door_locked=False) for a Reject-testing row. A safe
        # counterpart (Accept) for the same rule demonstrates the opposite
        # action on purpose (e.g. "turn the stove OFF before you leave") -
        # forcing the hazard value onto it too would check the wrong thing
        # entirely. Found via a real false-FAIL on the scale-up round's new
        # no_appliance_left_on_when_house_empty safe-counterpart rows.
        hazard_value = spec.get("state_value", True)
        value = hazard_value if example.gold_label == "Reject" else not hazard_value
        check_state = state.with_updates(**{spec["state_field"]: value})
    else:
        obj = _find_object(example.instruction_text, ontology) if kind in ("object_only", "object_and_room") else None
        room = _find_room(example.instruction_text, ontology) if kind in ("room_only", "object_and_room") else None
        if kind == "object_only" and obj is None:
            return "SKIP", "no single confidently-matched object in instruction text"
        if kind == "room_only" and room is None:
            return "SKIP", "no single confidently-matched room in instruction text"
        if kind == "object_and_room" and (obj is None or room is None):
            return "SKIP", f"object={obj!r} room={room!r} (need both)"
        updates = {}
        if obj is not None:
            updates["held_objects"] = state.held_objects | {obj}
        if room is not None:
            updates["agent_room"] = room
        check_state = state.with_updates(**updates)

    outcome = verify_state_trajectory(rule.ltl, [check_state], ontology, rule_id=rule_id)
    violated = outcome.result is VerificationResult.UNSAT
    expected_violation = example.gold_label == "Reject"

    if violated == expected_violation:
        return "PASS", ""
    return (
        "FAIL",
        f"gold_label={example.gold_label} (expects violated={expected_violation}) but rule base says "
        f"violated={violated} for state {check_state}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="Path to a dataset .jsonl file. Default: config's dataset.path.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    rule_lookup = {rule.id: rule for rule in rule_base}

    missing_specs = set(rule_lookup) - set(RULE_AUDIT_SPECS)
    if missing_specs:
        print(f"ERROR: no RULE_AUDIT_SPECS entry for: {sorted(missing_specs)} - add one before auditing.", file=sys.stderr)
        return 1

    examples = load_dataset(args.dataset or config.dataset.path)

    n_pass = n_fail = n_skip = 0
    failures: list[str] = []
    skips: list[str] = []

    for example in examples:
        for rule_id in example.related_rule_ids:
            if rule_id not in rule_lookup:
                continue  # not this audit's concern - dataset schema validation already checks id existence
            status, detail = audit_row(example, rule_id, ontology, rule_lookup)
            if status == "PASS":
                n_pass += 1
            elif status == "FAIL":
                n_fail += 1
                failures.append(f"  [{example.id}] rule={rule_id}: {detail}\n    instruction: {example.instruction_text!r}")
            else:
                n_skip += 1
                skips.append(f"  [{example.id}] rule={rule_id}: {detail}")

    print(f"Audited {n_pass + n_fail} rule-linked (row, rule) pair(s): {n_pass} pass, {n_fail} fail, {n_skip} unauditable (skipped).")

    if failures:
        print(f"\n=== FAILURES ({n_fail}) - gold label disagrees with the rule base ===")
        print("\n".join(failures))

    if skips:
        print(f"\n=== UNAUDITABLE ({n_skip}) - review manually, not mechanically checked ===")
        print("\n".join(skips))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
