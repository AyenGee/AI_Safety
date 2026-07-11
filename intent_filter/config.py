"""Central configuration loading.

Two sources are combined deliberately: `config/config.yaml` (or, if absent,
`config/config.example.yaml` as a fallback for a fresh checkout) holds
non-secret experiment settings and is validated with plain pydantic models;
`.env` / real environment variables hold secrets (API keys) via
pydantic-settings so they never need to touch a YAML file that could be
committed by accident.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


class EnvironmentPathsConfig(BaseModel):
    ontology_path: Path
    safety_rules_path: Path


class DatasetConfig(BaseModel):
    path: Path


class ModelsConfig(BaseModel):
    planner: str
    critic: str
    translator: str
    single_llm: str


class AgentConfig(BaseModel):
    ambiguity_margin: float = 0.15
    max_refinement_attempts: int = 2
    translation_max_retries: int = 3


class EvaluationConfig(BaseModel):
    repeats: int = 3
    confidence_level: float = 0.95
    ablations: list[str] = Field(default_factory=list)
    results_dir: Path = Path("results")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class AppConfig(BaseModel):
    """Validated contents of config/config.yaml."""

    environment: EnvironmentPathsConfig
    dataset: DatasetConfig
    models: ModelsConfig
    agent: AgentConfig = AgentConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()


class Secrets(BaseSettings):
    """Secrets sourced only from the environment / .env, never from YAML."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    user_config = REPO_ROOT / "config" / "config.yaml"
    if user_config.exists():
        return user_config
    example_config = REPO_ROOT / "config" / "config.example.yaml"
    logger.warning(
        "config/config.yaml not found, falling back to config/config.example.yaml. "
        "Copy it to config/config.yaml to customize settings."
    )
    return example_config


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the experiment config (non-secret settings)."""
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy config/config.example.yaml to config/config.yaml first."
        )
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def load_secrets() -> Secrets:
    """Load secrets from environment variables / .env."""
    return Secrets()
