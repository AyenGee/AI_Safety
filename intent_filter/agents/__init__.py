"""LLM-backed agents: Planner, Critic, and NL->LTL Translator.

All three depend only on the `LLMClient` protocol (client.py), never on the
`anthropic` package directly, so they are fully mockable in tests via
`ScriptedLLMClient`. See docs/architecture.md for how they compose into the
four intent-filtering systems (Phase 5).
"""

from intent_filter.agents.client import AnthropicLLMClient, LLMClient, ScriptedLLMClient
from intent_filter.agents.critic import (
    CriticError,
    CriticOutput,
    check_ambiguity,
    explain_violation,
    review,
)
from intent_filter.agents.parsing import strip_code_fences
from intent_filter.agents.planner import (
    PlannerError,
    PlannerInterpretation,
    PlannerOutput,
    plan,
)
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
    "strip_code_fences",
    "PlannerError",
    "PlannerInterpretation",
    "PlannerOutput",
    "plan",
    "TranslationResult",
    "template_translate",
    "translate",
]
