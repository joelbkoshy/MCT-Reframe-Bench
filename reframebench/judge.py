"""Model-based adjudication of content reframing.

The judge is a model that is not under test (Mistral 7B), so no arm is rated
by itself. The rubric defines reframing by the moves of cognitive
restructuring and lists what does not count, so that a process-focused or
merely empathic reply is not mistaken for reframing. The judge is calibrated
on a fixed set of known positives and known negatives before it rates any
generated reply (see ``calibration``).
"""

from __future__ import annotations

import re

from reframebench.backend import generate

JUDGE_MODEL = "mistral"
JUDGE_NUM_PREDICT = 80

JUDGE_SYSTEM = (
    "You rate short texts written in response to a person who described a "
    "situation and a negative thought. The text is either a reply to the "
    "person or a rewritten version of the thought. Decide whether the text "
    "REFRAMES the thought's content.\n\n"
    "The text REFRAMES if it does any of these:\n"
    "- questions whether the thought is true or accurate;\n"
    "- weighs evidence for or against the thought;\n"
    "- points out a thinking error or distortion;\n"
    "- offers an alternative, more balanced, or more positive version of the "
    "thought or of the situation;\n"
    "- argues against the thought's conclusion (for example, \"that does not "
    "mean you are a failure\").\n\n"
    "The text does NOT reframe if it only does these:\n"
    "- acknowledges or reflects the person's feelings;\n"
    "- names the thinking process, such as worry or rumination;\n"
    "- suggests a way of relating to the thought without disputing it, such "
    "as noticing it, letting it pass, postponing it, or shifting attention;\n"
    "- asks about the person's beliefs about their thinking;\n"
    "- gives practical suggestions without disputing the thought.\n\n"
    "Answer with one word on the first line, REFRAMES or NO, then one short "
    "reason on the second line."
)

_VERDICT = re.compile(r"\b(reframes|no)\b", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[str, str]:
    """Read the verdict from the first line only; returns ('reframes'|'no'|'unparsed', reason)."""
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    match = _VERDICT.search(first)
    if not match:
        return "unparsed", (text or "")[:200]
    return match.group(1).lower(), " ".join(lines[1:]).strip()[:300]


def adjudicate(situation: str, thought: str, text: str) -> tuple[str, str]:
    user = (
        f"SITUATION:\n{situation}\n\nTHOUGHT:\n{thought}\n\n"
        f"TEXT TO RATE:\n{text}\n\nDoes the text reframe the thought's content?"
    )
    try:
        out, _ = generate(JUDGE_SYSTEM, user, JUDGE_MODEL, num_predict=JUDGE_NUM_PREDICT)
    except Exception as exc:  # a failed call must never be scored as 'no'
        return "unparsed", f"judge error: {exc}"
    return parse_verdict(out)
