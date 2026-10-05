"""Deterministic lexical scoring of one reply.

The lexicons were written from the treatment literature before any reply was
generated: the reframing lexicon from the components of cognitive
restructuring (Beck, 2011; Burns, 1980), the process and technique lexicons
and the metacognitive-belief lexicon from the MCT model (Wells, 2009). They
are proxies. The judge model adjudicates reframing independently, and both are
checked against human-written reframes (RQ4).
"""

from __future__ import annotations

import re

from reframebench.backend import is_refusal

# RQ1: content-level reframing, grouped by restructuring move.
REFRAMING = {
    "evidence_testing": (
        r"\b(evidence (for|against|that|of)|what evidence|is (that|this|it) (really|actually|necessarily|always) true|"
        r"(examine|question|test|challenge)(s|ing)? (this|that|the|your) (thought|belief|assumption)|"
        r"facts (say|show|suggest))\b",
    ),
    "alternative_thought": (
        r"\b(reframe\w*|restructur\w*|"
        r"(more )?(balanced|realistic|helpful|compassionate|positive|kinder) (thought|way|perspective|view|outlook|interpretation)|"
        r"(another|different|other) (way|perspective|angle) (to|of) (look|see|think|view)\w*|"
        r"alternative (thought|explanation|perspective|view|interpretation)|"
        r"look at (it|this|things|the situation) (differently|another way|from (a|another|their))|"
        r"what would you (say|tell) (to )?a friend)\b",
    ),
    "counter_statement": (
        r"\b((doesn'?t|does not|don'?t|do not) (necessarily |automatically )?(mean|make) (that )?you|"
        r"just because .{0,60}(doesn'?t|does not) mean|"
        r"(one|a single) (mistake|setback|failure|bad day|incident) (doesn'?t|does not)|"
        r"you are not (a failure|worthless|stupid|useless|unlovable|a bad person)|"
        r"it'?s not your fault|"
        r"(worst[- ]case|how likely is (it|that)))\b",
    ),
    "distortion_labelling": (
        r"\b(cognitive distortions?|thinking (errors?|traps?)|all[- ]or[- ]nothing|black[- ]and[- ]white thinking|"
        r"catastrophi[sz]\w*|overgenerali[sz]\w*|mind[- ]reading|fortune[- ]telling|"
        r"should statements|emotional reasoning|disqualifying the positive)\b",
    ),
    "positive_reappraisal": (
        r"\b(on the (other hand|bright side)|silver lining|blessing in disguise|"
        r"(maybe|perhaps) (they|it|this|that|he|she) (was|is|were|are|wasn'?t|isn'?t|didn'?t|did not|had|might))\b",
    ),
}

# Process orientation: names the thinking process rather than its content.
CAS_PROCESS = (
    r"\b(ruminat\w*|worry(ing)? (process|cycle|loop|pattern|habit)|threat monitoring|"
    r"dwelling|going over (it|this|things|that)|overthinking|"
    r"thinking (pattern|process|style|loop|cycle|habit)|repetitive (thinking|thoughts))\b"
)

TECHNIQUES = {
    "postponement": r"\b(postpon\w*|worry (time|period)|rumination (time|period)|set (aside|a time)|designated time|"
                    r"later (today|in the day|this evening))\b",
    "attention": r"\b(attention training|shift(ing)? (your )?attention|(redirect|refocus)(ing)? your attention|"
                 r"(focus|place) your attention (on|to) (sounds|your surroundings|something))\b",
    "detached_mindfulness": r"\b(detached mindfulness|notic(e|ing) (the|this|that|your) thoughts?|"
                            r"(observ|watch)(e|ing) (the|this|that|your) thoughts?|"
                            r"let (it|the thought|them) (be|pass|go|come and go)|"
                            r"without (engaging|responding|analy[sz]ing|getting caught|reacting))\b",
}

# RQ2: beliefs about thinking (Wells' S-REF model).
METACOGNITIVE_BELIEFS = {
    "uncontrollability": (
        r"\b(uncontrollab\w+|out of (your|my) control|"
        r"(can'?t|cannot|unable to) (stop|control) (thinking|worrying|ruminating|the thoughts?|it)|"
        r"(control|choice|say) over (your|the|these|this) (thoughts?|thinking|worry|worrying|rumination)|"
        r"(choose|decide) (whether|not|when) to (engage|follow|respond))\b",
    ),
    "usefulness": (
        r"\b((is|was|does) (this|that|it|all this) (thinking|worrying|ruminating|analy[sz]ing|going over it|dwelling) "
        r"(really )?(help|helping|helpful|useful|working|productive))\b",
        r"\b((worry|worrying|rumination|ruminating|analy[sz]ing|overthinking|dwelling|going over it) "
        r"(helps?|protects?|prepares?|solves?|is (useful|helpful|necessary|productive)))\b",
        r"\b(beliefs? (that|about) (worry|worrying|rumination|ruminating|(your|the) thinking))\b",
    ),
}

TECHNIQUE_PRIORITY = ("postponement", "attention", "detached_mindfulness")

_REFRAMING = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in REFRAMING.items()}
_CAS = re.compile(CAS_PROCESS, re.IGNORECASE)
_TECH = {k: re.compile(TECHNIQUES[k], re.IGNORECASE) for k in TECHNIQUE_PRIORITY}
_BELIEF = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in METACOGNITIVE_BELIEFS.items()}


def _hits(patterns, text: str) -> list[str]:
    return [m.group(0) for m in (p.search(text) for p in patterns) if m]


def detect_reframing(text: str) -> dict[str, list[str]]:
    return {k: _hits(v, text or "") for k, v in _REFRAMING.items()}


def detect_metacognitive_beliefs(text: str) -> dict[str, list[str]]:
    return {k: _hits(v, text or "") for k, v in _BELIEF.items()}


def techniques(text: str) -> list[str]:
    return [k for k in TECHNIQUE_PRIORITY if _TECH[k].search(text or "")]


def score(text: str) -> dict:
    """All lexical scores for one reply, flattened for CSV."""
    reframing = detect_reframing(text)
    beliefs = detect_metacognitive_beliefs(text)
    techs = techniques(text)
    names_process = bool(_CAS.search(text or ""))
    return {
        "reframing": any(reframing.values()),
        "reframing_moves": ";".join(k for k, v in reframing.items() if v),
        "reframing_hits": "; ".join(h for v in reframing.values() for h in v),
        "names_process": names_process,
        "process_oriented": names_process or bool(techs),
        "mb_uncontrollability": bool(beliefs["uncontrollability"]),
        "mb_usefulness": bool(beliefs["usefulness"]),
        "mb_any": any(beliefs.values()),
        "mb_hits": "; ".join(h for v in beliefs.values() for h in v),
        "techniques": ";".join(techs),
        "primary_technique": techs[0] if techs else "none",
        "refused": is_refusal(text),
        "n_words": len((text or "").split()),
    }
