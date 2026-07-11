"""Dataset schema, loader, and validator for the labeled instruction dataset.

See data/dataset_schema.md for the human-readable schema description and
category definitions. This module is the single source of truth for what a
valid dataset row looks like - the seed dataset (data/instructions.jsonl)
and any future LLM-assisted generation (Phase 7) must both produce rows that
validate against `InstructionExample`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from intent_filter.environment.ontology import Ontology
from intent_filter.environment.state import WorldState, initial_state

Category = Literal["legitimate", "unsafe", "ambiguous", "misdirected"]
GoldLabel = Literal["Accept", "Reject", "Clarify"]

# Ground truth: a category determines exactly one correct adjudication.
# Enforced at load time so a mislabeled row fails fast rather than silently
# corrupting evaluation metrics later.
EXPECTED_LABEL_BY_CATEGORY: dict[Category, GoldLabel] = {
    "legitimate": "Accept",
    "unsafe": "Reject",
    "ambiguous": "Clarify",
    "misdirected": "Reject",
}


class SceneContext(BaseModel):
    """The room/object/role state needed to evaluate one instruction.

    Mirrors intent_filter.environment.state.WorldState (plus issuing_role,
    which WorldState also carries but which is most naturally authored here
    as "who issued this instruction"). Fields default to the environment's
    standard initial state, so a dataset row only needs to specify what's
    different about its scene.
    """

    issuing_role: str = "owner"
    agent_room: str | None = None
    held_objects: list[str] = Field(default_factory=list)
    object_locations: dict[str, str] = Field(default_factory=dict)
    door_locked: bool = True
    alarm_on: bool = False
    stove_on: bool = False
    owner_home: bool = True

    def to_world_state(self, ontology: Ontology) -> WorldState:
        """Build the concrete WorldState a pipeline should start from for this row."""
        base = initial_state(ontology, issuing_role=self.issuing_role)
        merged_locations = {**base.object_locations, **self.object_locations}
        return base.with_updates(
            agent_room=self.agent_room or base.agent_room,
            held_objects=frozenset(self.held_objects),
            object_locations=merged_locations,
            door_locked=self.door_locked,
            alarm_on=self.alarm_on,
            stove_on=self.stove_on,
            owner_home=self.owner_home,
        )


class InstructionExample(BaseModel):
    """One labeled row of the instruction dataset."""

    id: str
    instruction_text: str
    category: Category
    gold_label: GoldLabel
    scene_context: SceneContext = Field(default_factory=SceneContext)
    notes: str = ""
    # Safety rule ids (config/safety_rules.yaml) this example is designed to
    # exercise - not part of the brief's minimum schema, but needed to make
    # "every rule has a violating and a non-violating example" checkable.
    # Empty for ambiguous rows and for legitimate rows unrelated to any rule.
    related_rule_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_label_matches_category(self) -> "InstructionExample":
        expected = EXPECTED_LABEL_BY_CATEGORY[self.category]
        if self.gold_label != expected:
            raise ValueError(
                f"{self.id}: category {self.category!r} requires gold_label "
                f"{expected!r}, got {self.gold_label!r}"
            )
        return self


def load_dataset(path: str | Path) -> list[InstructionExample]:
    """Load and validate the JSONL instruction dataset.

    Raises ValueError on the first malformed row (bad JSON, schema
    violation, or duplicate id) so a broken dataset fails fast at load time
    rather than silently dropping or misinterpreting rows during evaluation.
    """
    examples: list[InstructionExample] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc

            example = InstructionExample.model_validate(raw)
            if example.id in seen_ids:
                raise ValueError(f"Duplicate instruction id {example.id!r} on line {line_no}")
            seen_ids.add(example.id)
            examples.append(example)
    return examples
