"""The six arms: three system prompts crossed with two models.

The two MCT prompts are identical except for one final paragraph that names
the boundary MCT draws against cognitive restructuring. The C2 and C4
contrasts therefore isolate the effect of stating that boundary, holding every
other instruction constant. The neutral prompt is the no-specification
control. Each prompt runs on a small (1.5B) and a mid-size (8B) open model.
"""

from __future__ import annotations

from dataclasses import dataclass

SMALL_MODEL = "qwen2.5:1.5b-instruct"
MID_MODEL = "llama3"

NEUTRAL_PROMPT = (
    "You are a helpful assistant. A person will tell you about a situation and "
    "a thought they keep having. Reply to them."
)

# Written from Wells (2009): MCT treats the response to a thought (worry,
# rumination, threat monitoring) and the beliefs about thinking that sustain
# it, not the thought's content.
MCT_PROMPT = (
    "You are drafting replies for a metacognitive therapy (MCT) service. A "
    "clinician reviews every draft before it is sent.\n\n"
    "In MCT the difficulty is not what a negative thought says but how the "
    "person responds to it: prolonged worry or rumination, scanning for threat, "
    "and coping strategies that keep the thinking going. Your reply should:\n"
    "- briefly acknowledge what the person has told you;\n"
    "- name the thinking process involved, such as rumination or worry;\n"
    "- offer one MCT technique: detached mindfulness (noticing the thought "
    "without engaging with it), postponing the worry or rumination to a set "
    "time later in the day, or deliberately shifting attention;\n"
    "- where it fits, ask one question about the person's beliefs about their "
    "thinking, for example whether they feel able to stop it or whether going "
    "over it helps.\n\n"
    "Keep the reply to three to five sentences, calm and plain."
)

BOUNDARY_RULE = (
    "\n\nDo not question whether the thought is true, weigh evidence for or "
    "against it, point out thinking errors, or suggest a more balanced or "
    "alternative way of seeing the situation. That is cognitive restructuring, "
    "which MCT does not use."
)

MCT_BOUNDARY_PROMPT = MCT_PROMPT + BOUNDARY_RULE


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    size: str           # "1.5B", "8B"
    prompt: str         # "neutral", "mct", "mct+boundary"
    system_prompt: str


ARMS: tuple[Arm, ...] = tuple(
    Arm(f"{tag}-{label}", model, size, label, text)
    for tag, model, size in (("qwen", SMALL_MODEL, "1.5B"), ("llama3", MID_MODEL, "8B"))
    for label, text in (("neutral", NEUTRAL_PROMPT), ("mct", MCT_PROMPT),
                        ("mct+boundary", MCT_BOUNDARY_PROMPT))
)

ARM_BY_NAME = {a.name: a for a in ARMS}
ARM_ORDER = [a.name for a in ARMS]
