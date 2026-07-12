"""LLM-backed agents: Planner, Critic, NL->LTL Translator, and the combined Single-LLM agent.

All depend only on the `LLMClient` protocol (client.py), never on the
`anthropic` package directly, so they are fully mockable in tests via
`ScriptedLLMClient`. See docs/architecture.md for how they compose into the
four intent-filtering systems (intent_filter/systems/).
"""

from intent_filter.agents.client import AnthropicLLMClient, LLMClient, ScriptedLLMClient
from intent_filter.agents.critic import (
    CriticError,
    CriticOutput,
    check_ambiguity,
    explain_violation,
    review,
)
from intent_filter.agents.parsing import parse_action, strip_code_fences
from intent_filter.agents.planner import (
    PlannerError,
    PlannerInterpretation,
    PlannerOutput,
    plan,
)
from intent_filter.agents.single_llm import SingleLLMError, SingleLLMOutput
from intent_filter.agents.single_llm import run as run_single_llm
from intent_filter.agents.translator import TranslationResult, template_translate, translate

__all__ = [
    "AnthropicLLMClient",
    "LLMClient",
    "ScriptedLLMClient",
    "CriticError",
    "CriticOutput",
    "check_ambiguity",
    "explain_violation",
    "review",
    "parse_action",
    "strip_code_fences",
    "PlannerError",
    "PlannerInterpretation",
    "PlannerOutput",
    "plan",
    "SingleLLMError",
    "SingleLLMOutput",
    "run_single_llm",
    "TranslationResult",
    "template_translate",
    "translate",
]
