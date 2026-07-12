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

from intent_filter.environment.actions import Action, ActionType

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

_VALID_ACTION_TYPES = {t.name for t in ActionType}


def strip_code_fences(text: str) -> str:
    """Strip a single leading/trailing markdown code fence, if present."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def parse_action(raw: dict) -> Action:
    """Parse one {"type": ..., "argument": ...} dict from an agent's JSON
    response into an Action. Shared by every agent that asks the LLM for an
    action sequence (Planner, single-LLM). Raises ValueError on an unknown
    action type, which callers treat as a retryable parse failure.
    """
    type_name = str(raw.get("type", "")).upper()
    if type_name not in _VALID_ACTION_TYPES:
        raise ValueError(f"Unknown action type: {raw.get('type')!r}")
    return Action(action_type=ActionType[type_name], argument=raw.get("argument"))
