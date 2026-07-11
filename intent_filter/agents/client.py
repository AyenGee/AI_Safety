"""LLM client abstraction so agents are mockable without hitting the network.

Every agent (Planner, Critic, Translator) depends only on the `LLMClient`
protocol below, never on the `anthropic` package directly. Tests inject a
`ScriptedLLMClient` with pre-programmed responses instead of making real API
calls; only the manual debug CLI and the evaluation harness (Phase 5/6)
construct a real `AnthropicLLMClient`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        """Return the model's text response to a single-turn (system, user) prompt."""
        ...


class AnthropicLLMClient:
    """Real LLMClient backed by the Anthropic Messages API."""

    def __init__(self, api_key: str):
        import anthropic  # imported lazily so tests never need the package installed to run

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


@dataclass
class ScriptedLLMClient:
    """Test double: returns pre-programmed responses in call order, no network access.

    Records every call in `.calls` so tests can assert on the exact prompts
    an agent sent (e.g. that a retry attempt includes the previous error).
    """

    responses: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False, repr=False)

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls.append(
            {"model": model, "system": system, "user": user, "max_tokens": max_tokens}
        )
        if self._index >= len(self.responses):
            raise AssertionError(
                f"ScriptedLLMClient ran out of scripted responses after {self._index} call(s)"
            )
        response = self.responses[self._index]
        self._index += 1
        return response
