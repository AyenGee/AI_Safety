"""LTLf formula parsing, wrapping `flloat` with a single stable error type.

See docs/methodology.md for why this project uses LTLf (finite-trace LTL,
via flloat) rather than infinite-trace LTL / spot: commands here are finite
action sequences that complete or are rejected, which is exactly the
semantics LTLf is built for, and `spot` has no PyPI distribution (confirmed
by `pip index versions spot` returning no match) and only weak native
Windows support, whereas flloat is pure Python and pip-installable.
"""

from __future__ import annotations

from flloat.parser.ltlf import LTLfParser

_parser = LTLfParser()


class LTLFormulaError(ValueError):
    """Raised when an LTL formula string fails to parse.

    flloat/lark raise several distinct exception types for different kinds
    of malformed input (unexpected token, unexpected character, ...); all
    are normalized to this one type so callers - the verifier here, and the
    NL->LTL translator's bounded retry loop in Phase 4 - only need to catch
    a single exception type.
    """


def parse_formula(ltl_text: str):
    """Parse an LTLf formula string (already atom-sanitized) into a flloat formula object."""
    try:
        return _parser(ltl_text)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see class docstring
        raise LTLFormulaError(f"Failed to parse LTL formula {ltl_text!r}: {exc}") from exc
