"""Known negatives for judge calibration.

Forty researcher-written texts that do not reframe content, in three kinds:
process-level MCT replies, purely empathic replies, and practical
suggestions. None mentions a specific situation, so each can be paired with
any item. The known positives are the human reframes of the 40 calibration
items. Positives and negatives are balanced, and none of these texts is ever
used as a model reply.
"""

from __future__ import annotations

PROCESS = (
    "It sounds as though this thought keeps coming back and pulling you into going over it. Rather than working through it again, see if you can notice it as a thought and let it sit there without answering it. Does going over it actually help you?",
    "What you describe sounds like rumination, the mind circling the same ground. You could try postponing it: set fifteen minutes later today as your thinking time, and when it shows up before then, note it and leave it for later.",
    "This sounds like a worry loop that keeps restarting. When it comes up, try simply noticing 'there is that thought again' and let it pass in its own time. Do you feel you have any choice about whether you engage with it?",
    "It seems the hard part is how long you spend with this thought. For a moment, shift your attention to the sounds around you and notice whether the thought gets quieter when it is not being fed.",
    "That thought sounds very familiar to you, and it seems to bring a lot of mulling over with it. Can you watch it the way you might watch traffic passing, without stepping into it?",
    "It sounds like your mind keeps returning to this. One option is to give the rumination a set time at six this evening and treat any thoughts before then as things to come back to later.",
    "I hear how often this goes round in your head. What do you think happens if you leave the thought alone instead of analysing it?",
    "This sounds like a pattern of dwelling that wears you down. Try noticing the thought, labelling it 'rumination', and bringing your attention back to what you were doing.",
    "It sounds like you have been going over this a lot. Is there a part of you that believes thinking it through will settle it? It might be worth testing what happens if you postpone it instead.",
    "The thought itself may keep showing up, and that is fine. The question is whether you need to follow it. Can you let it be there in the background while you carry on?",
    "It sounds as though worry has taken up a lot of your day. A useful step is to notice when the worry starts and choose to set it aside until a fixed worry period.",
    "This seems to be a thought that hooks you into long stretches of thinking. Practising moving your attention between different sounds for a few minutes can help you see that you can step out of it.",
    "You seem to be spending a lot of energy on this one thought. Try observing it without responding, as if it were a radio playing in another room.",
    "It sounds like the thinking has become a habit that runs on its own. Do you feel it is out of your control, or have there been moments when you set it down?",
    "What strikes me is how much time this takes up for you. You could try treating it as an event in the mind, noticing it arrive and leaving it unanswered.",
    "It sounds like the mind keeps replaying this. Next time it starts, try postponing it to a designated time and see what happens to the urge to go over it.",
    "This sounds like repetitive thinking that keeps you stuck. Rather than engaging with it, can you notice it and gently bring your attention back to the room around you?",
    "It seems going over this has become automatic. Is the going over it helping, or does it leave you feeling the same afterwards?",
    "The thought sounds loud right now. You do not need to argue with it or agree with it; you can let it come and go while you focus on something in front of you.",
    "It sounds like you have been caught in a cycle of worrying about this. A small experiment: notice the worry, tell yourself you will think about it at seven tonight, and see whether it can wait.",
)

EMPATHIC = (
    "That sounds really painful, and it makes sense that it has stayed with you.",
    "I'm sorry you're going through this. It sounds like it has been weighing on you heavily.",
    "That sounds hard. Thank you for sharing it with me.",
    "It sounds like this has left you feeling hurt and unsettled.",
    "I can hear how upsetting this has been for you.",
    "That must be a difficult thing to carry around.",
    "It sounds like you have been feeling very low about this.",
    "That sounds frustrating and disappointing.",
    "I'm sorry this happened. It sounds like it has affected you a lot.",
    "It sounds like this has been on your mind and bringing you down.",
)

PRACTICAL = (
    "It might help to write down the next small step you could take and when you will take it.",
    "If it feels manageable, you could talk to someone you trust about what happened.",
    "Getting some rest and something to eat may help before you decide what to do next.",
    "You might find it useful to make a short list of tasks for tomorrow so they are not all in your head.",
    "If this is affecting your work, it may be worth speaking to your manager about support.",
    "Going for a short walk outside can be a helpful break when things feel heavy.",
    "If there is a practical problem to solve, you could break it into one or two concrete actions.",
    "It could help to check whether there is a support service available to you.",
    "Planning one enjoyable activity for this week might give you something to look forward to.",
    "If sleep has been difficult, keeping a regular bedtime may make the coming days easier.",
)

NEGATIVES: tuple[tuple[str, str], ...] = (
    tuple(("process", t) for t in PROCESS)
    + tuple(("empathic", t) for t in EMPATHIC)
    + tuple(("practical", t) for t in PRACTICAL)
)
