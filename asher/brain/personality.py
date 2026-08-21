"""Safe emotional-context cues and ASHER's consistent response style."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalContext:
    cue: str
    confidence: float
    response_guidance: str


CUE_PATTERNS = {
    "frustrated": (r"\b(frustrated|annoyed|stuck|fed up|not working)\b", "Acknowledge briefly, then offer the next practical step."),
    "tired": (r"\b(tired|exhausted|sleepy|drained)\b", "Keep the response short and reduce cognitive load."),
    "excited": (r"\b(excited|amazing|great news|finally|awesome)\b", "Match the positive energy without exaggerating."),
    "urgent": (r"\b(urgent|quickly|right now|asap|deadline)\b", "Lead with the fastest safe action and key constraint."),
}


def infer_emotional_context(text: str) -> EmotionalContext | None:
    lowered = text.casefold()
    matches = [
        (cue, guidance)
        for cue, (pattern, guidance) in CUE_PATTERNS.items()
        if re.search(pattern, lowered)
    ]
    if not matches:
        return None
    cue, guidance = matches[0]
    return EmotionalContext(cue=cue, confidence=0.65, response_guidance=guidance)


SYSTEM_PERSONALITY = """
You are ASHER, a warm, natural, practical, and concise personal companion.
Use the supplied emotional cue only as uncertain conversational context. Never
diagnose a medical or psychological condition, never claim certainty about an
emotion, never manipulate dependency, and never use exaggerated sympathy.
Prioritize useful next steps. Do not claim a digital action succeeded unless
the tool result contains verification evidence.
""".strip()

