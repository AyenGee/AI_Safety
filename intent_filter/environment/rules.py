"""Safety rule base: loads config/safety_rules.yaml as structured, auditable data.

Phase 1 only loads and validates rule *schema* (every rule has the required
fields, categories are covered). LTL *semantics* (parsing the `ltl` field
into an evaluable formula and checking it against a trajectory) is Phase 2's
responsibility (intent_filter/verifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_FIELDS = ("id", "description", "category", "severity", "ltl")
VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


@dataclass(frozen=True)
class SafetyRule:
    id: str
    description: str
    category: str
    severity: str
    ltl: str
    violating_example: str | None = None
    safe_example: str | None = None


@dataclass(frozen=True)
class SafetyRuleBase:
    rules: tuple[SafetyRule, ...]

    def by_id(self, rule_id: str) -> SafetyRule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(f"Unknown safety rule id: {rule_id!r}")

    def by_category(self, category: str) -> tuple[SafetyRule, ...]:
        return tuple(r for r in self.rules if r.category == category)

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(r.category for r in self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)


def load_safety_rules(path: str | Path) -> SafetyRuleBase:
    """Load and validate the safety rule base from YAML.

    Raises ValueError on malformed entries (missing required fields, invalid
    severity, or duplicate rule ids) so config errors are caught at startup.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    entries = raw.get("rules") or []
    rules: list[SafetyRule] = []
    seen_ids: set[str] = set()

    for i, entry in enumerate(entries):
        missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            raise ValueError(f"Safety rule at index {i} is missing required field(s): {missing}")

        rule_id = entry["id"]
        if rule_id in seen_ids:
            raise ValueError(f"Duplicate safety rule id: {rule_id!r}")
        seen_ids.add(rule_id)

        severity = entry["severity"]
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Rule {rule_id!r} has invalid severity {severity!r}; "
                f"must be one of {sorted(VALID_SEVERITIES)}"
            )

        rules.append(
            SafetyRule(
                id=rule_id,
                description=entry["description"],
                category=entry["category"],
                severity=severity,
                ltl=entry["ltl"],
                violating_example=entry.get("violating_example"),
                safe_example=entry.get("safe_example"),
            )
        )

    return SafetyRuleBase(rules=tuple(rules))
