"""Phase 1 smoke test for config loading (falls back to config.example.yaml
on a fresh checkout, since config/config.yaml is gitignored)."""

from intent_filter.config import load_config


def test_load_config_falls_back_to_example():
    config = load_config()
    assert config.environment.ontology_path.name == "environment_ontology.yaml"
    assert config.dataset.path.name == "instructions.jsonl"
    assert config.models.planner
    assert config.agent.max_refinement_attempts == 2
