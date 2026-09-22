"""LLM client abstraction so agents are mockable without hitting the network.

Every agent (Planner, Critic, Translator) depends only on the `LLMClient`
protocol below, never on a specific provider SDK/HTTP API. Tests inject a
`ScriptedLLMClient` with pre-programmed responses instead of making real API
calls.

`OllamaLLMClient` is the one every script builds by default (see
`scripts/run_evaluation.py` and `scripts/experiment_*.py`) - the project
moved from the paid Anthropic API to free open-weight models (Qwen3.5,
Gemma4) served locally via Ollama on the university's Slurm cluster.
`AnthropicLLMClient` is kept, unused by default, only so the already-reported
Phase 8 results (produced against real Anthropic models) remain explainable
and reproducible later; nothing new references it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        """Return the model's text response to a single-turn (system, user) prompt."""
        ...


class AnthropicLLMClient:
    """Real LLMClient backed by the Anthropic Messages API.

    `timeout` bounds each request explicitly (the underlying SDK/httpx
    default was observed, during Phase 7 live testing, to let one call hang
    for over an hour with no error - unacceptable in a batch harness that
    makes thousands of sequential calls, where one stall silently balloons
    the whole run's wall-clock time). `max_retries` uses the SDK's own
    built-in retry-with-backoff for transient network/5xx errors, which is
    a different concern from this project's own agent-level retries on
    malformed (but successfully returned) responses.
    """

    def __init__(self, api_key: str, timeout: float = 120.0, max_retries: int = 2):
        import anthropic  # imported lazily so tests never need the package installed to run

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

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


class OllamaLLMClient:
    """Real LLMClient backed by a local Ollama server (`/api/chat`).

    Talks to whatever Ollama instance `base_url` points at - on the cluster
    this is a private, per-Slurm-job port (`~/llm/env.sh` sets `OLLAMA_HOST`
    to `127.0.0.1:<20000 + SLURM_JOB_ID % 10000>`, chosen precisely so
    multiple jobs on the same or different nodes never collide). If
    `base_url` isn't passed explicitly, it's built from that same
    `OLLAMA_HOST` env var so no extra cluster-side config is needed; falls
    back to Ollama's own default port for local/interactive testing off the
    cluster.

    `think=False` is sent on every request - `gemma4:e4b` otherwise emits a
    `Thinking...` preamble ahead of its actual answer (see
    mscluster_LLM_setup_guide.pdf), which would corrupt every downstream
    JSON-parsing agent expecting a clean response.

    Uses stdlib `urllib.request` only - no extra pip dependency, so a bare
    cluster Python install (no internet access to PyPI mirrors guaranteed
    beyond the login node) needs nothing beyond this project's own existing
    dependencies to run the harness against Ollama.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        self._base_url = (base_url or f"http://{os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')}").rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["message"]["content"]
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
        raise RuntimeError(
            f"Ollama request to {self._base_url}/api/chat (model={model}) failed after "
            f"{self._max_retries + 1} attempt(s): {last_error}"
        ) from last_error


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
