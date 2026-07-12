"""Shared pytest fixtures: the real environment ontology and safety rule base,
loaded once per session. Used by tests/test_decision.py and tests/test_systems.py
(other test files predate this conftest and declare their own local fixtures,
which is fine - a same-named local fixture simply shadows this one).
"""

from pathlib import Path

import pytest

from intent_filter.environment import load_ontology, load_safety_rules

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = REPO_ROOT / "config" / "environment_ontology.yaml"
SAFETY_RULES_PATH = REPO_ROOT / "config" / "safety_rules.yaml"


@pytest.fixture(scope="session")
def ontology():
    return load_ontology(ONTOLOGY_PATH)


@pytest.fixture(scope="session")
def rule_base():
    return load_safety_rules(SAFETY_RULES_PATH)
