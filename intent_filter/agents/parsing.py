"""Shared response-parsing helpers for the Planner, Critic, and Translator agents.

`strip_code_fences` exists because of an observed real-world failure mode:
despite every agent's system prompt explicitly saying "respond with ONLY a
JSON object, no markdown fences", live testing against the real Anthropic
API showed Claude sometimes wraps the JSON response in a ```json ... ```
fence anyway. Rather than relying solely on prompt wording (which the
model doesn't always follow) or spending a retry on it, every agent strips
a fence if present before calling `json.loads`.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Strip a single leading/trailing markdown code fence, if present."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped
