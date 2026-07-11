"""NL -> LTL Translator agent: LLM proposes an LTLf formula over the fixed
atom vocabulary of the environment ontology, validated by the deterministic
verifier's own formula parser (so a formula this agent accepts is guaranteed
parseable by intent_filter.verifier), with a bounded retry loop and a small
template-based fallback for common instruction patterns as a last resort.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from intent_filter.agents.client import LLMClient
from intent_filter.agents.parsing import strip_code_fences
from intent_filter.agents.prompts import describe_atom_vocabulary
from intent_filter.environment.ontology import Ontology
from intent_filter.verifier.atoms import build_atom_map, sanitize_formula
from intent_filter.verifier.formula import LTLFormulaError, parse_formula

_RETRYABLE_ERRORS = (json.JSONDecodeError, KeyError, ValueError, TypeError, LTLFormulaError)

TRANSLATOR_SYSTEM_PROMPT_TEMPLATE = """You are the NL-to-LTL Translator component of a household \
robot's intent-filtering pipeline. Translate a natural-language command into a single LTLf \
(finite-trace linear temporal logic) formula that captures its safety-relevant behavior, using ONLY \
the atomic propositions listed below - never invent new atom names.

Atomic propositions:
{atom_vocabulary}

Operators: G (always), F (eventually), X (next), U (until), ! (not), & (and), | (or), -> (implies), \
<-> (iff). Parentheses are required for grouping.

Respond with ONLY a JSON object (no prose, no markdown fences) of the form:
{{"ltl": "<formula>"}}

If the command has no safety-relevant temporal property to check (it doesn't touch any room, object, \
or world variable that appears in the atom list above), respond with {{"ltl": "G(true)"}}.
"""

_VERB_RE = re.compile(r"\b(bring|take|carry|put|leave|move|hold)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TranslationResult:
    ltl_formula: str | None
    success: bool
    attempts: int
    used_fallback: bool
    raw_responses: tuple[str, ...] = field(default_factory=tuple)


def _extract_formula(text: str) -> str:
    data = json.loads(strip_code_fences(text))
    formula = data["ltl"]
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Translator returned an empty or non-string formula")
    return formula


def template_translate(instruction_text: str, ontology: Ontology) -> str | None:
    """Best-effort fallback for a handful of common instruction patterns.

    Not a general translator - it only covers the "<verb> <object> ... <room>"
    shape plus a couple of stove/door-specific phrasings, as a safe-ish last
    resort when the LLM fails to produce a valid formula after retrying.
    Anything it can't match returns None; callers should treat that as
    translation failure (route to Reject via the decision layer, Phase 5).
    """
    text = instruction_text.lower()

    def _mentioned(names: list[str]) -> list[str]:
        # Match on each underscore-separated word rather than the exact
        # phrase, so e.g. "child_room" matches "the child's room" (possessive
        # "'s" breaks an exact "child room" substring match).
        return [name for name in names if all(word in text for word in name.split("_"))]

    mentioned_objects = _mentioned(list(ontology.objects))
    mentioned_rooms = _mentioned(list(ontology.rooms))
    if _VERB_RE.search(text) and mentioned_objects and mentioned_rooms:
        return f"G(!(has_object({mentioned_objects[0]}) & agent_at({mentioned_rooms[0]})))"

    if "stove" in text and "bathroom" in text:
        return "G(!(stove_on & agent_at(bathroom)))"

    if "door" in text and "unlock" in text:
        return "G(door_locked | owner_home)"

    return None


def translate(
    client: LLMClient,
    model: str,
    instruction: str,
    ontology: Ontology,
    max_retries: int = 3,
) -> TranslationResult:
    """Translate `instruction` into a validated LTLf formula string.

    The returned `ltl_formula` (when `success`) uses the same human-readable,
    parenthesized atom names as config/safety_rules.yaml, so callers pass it
    through intent_filter.verifier exactly like a rule's `ltl` field.
    """
    system = TRANSLATOR_SYSTEM_PROMPT_TEMPLATE.format(
        atom_vocabulary=describe_atom_vocabulary(ontology)
    )
    atom_map = build_atom_map(ontology)
    user = f"Command: {instruction!r}"
    responses: list[str] = []

    attempts = 0
    for _ in range(max_retries + 1):
        attempts += 1
        response_text = client.complete(model=model, system=system, user=user, max_tokens=256)
        responses.append(response_text)
        try:
            formula = _extract_formula(response_text)
            parse_formula(sanitize_formula(formula, atom_map))  # validate; result unused here
            return TranslationResult(
                ltl_formula=formula,
                success=True,
                attempts=attempts,
                used_fallback=False,
                raw_responses=tuple(responses),
            )
        except _RETRYABLE_ERRORS as exc:
            user = (
                f"Command: {instruction!r}\n\nYour previous response was invalid ({exc}). "
                f"Respond again with ONLY the JSON object described in the system prompt, "
                f"using only the listed atoms."
            )

    fallback = template_translate(instruction, ontology)
    if fallback is not None:
        return TranslationResult(
            ltl_formula=fallback,
            success=True,
            attempts=attempts,
            used_fallback=True,
            raw_responses=tuple(responses),
        )

    return TranslationResult(
        ltl_formula=None,
        success=False,
        attempts=attempts,
        used_fallback=False,
        raw_responses=tuple(responses),
    )
