"""Deterministic crisis screen applied to every input before any model sees it.

A keyword screen, not a risk assessment. In deployment a flagged message would
bypass the model and go to a human; here flagged items are excluded from
generation and counted, so the exclusion is reported rather than silent.
"""

from __future__ import annotations

import re

CRISIS_PATTERNS = (
    r"\bsuicid\w*",
    r"\bkill(ing)? (myself|me)\b",
    r"\bend(ing)? (my life|it all)\b",
    r"\b(want|wanted|wish) to die\b",
    r"\b(wish|wished) i (was|were) dead\b",
    r"\bbetter off (dead|without me)\b",
    r"\b(don'?t|do not) want to (live|be alive|exist|be here)\b",
    r"\bnot worth living\b",
    r"\bno reason to live\b",
    r"\bself[- ]?harm\w*",
    r"\b(hurt|harm|cut|cutting|burn|burning) (myself|my (arm|arms|wrist|wrists|leg|legs))\b",
    r"\boverdos\w*",
)

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]


def screen(text: str) -> list[str]:
    """Return the patterns that matched; an empty list means the input is eligible."""
    return [p.pattern for p in _COMPILED if p.search(text or "")]
