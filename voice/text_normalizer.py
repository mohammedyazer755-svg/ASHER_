"""Text repair and contact matching for Asher's voice pipeline.

This module contains no microphone, Whisper, Windows UI, or LLM imports, so it
can be tested independently and safely reused by both the listener and the
WhatsApp action layer.
"""

from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
VOCABULARY_FILE = PROJECT_DIR / "data" / "voice_contacts.json"

# Personal contacts are loaded only from the local ignored vocabulary file.
# Keeping a non-empty default here would leak private names into source/Git.
DEFAULT_CONTACTS: tuple[str, ...] = ()
DEFAULT_ALIASES: dict[str, str] = {}

# Common ways Whisper writes spoken letters. Only used when every token in the
# contact fragment is a letter name, which avoids modifying normal sentences.
SPOKEN_LETTERS = {
    "a": "a", "ay": "a", "alpha": "a",
    "b": "b", "bee": "b", "bravo": "b",
    "c": "c", "cee": "c", "sea": "c", "charlie": "c",
    "d": "d", "dee": "d", "delta": "d",
    "e": "e", "ee": "e", "echo": "e",
    "f": "f", "ef": "f", "foxtrot": "f",
    "g": "g", "gee": "g", "golf": "g",
    "h": "h", "aitch": "h", "hotel": "h",
    "i": "i", "eye": "i", "india": "i",
    "j": "j", "jay": "j", "juliet": "j",
    "k": "k", "kay": "k", "kilo": "k",
    "l": "l", "el": "l", "lima": "l",
    "m": "m", "em": "m", "mike": "m",
    "n": "n", "en": "n", "november": "n",
    "o": "o", "oh": "o", "oscar": "o",
    "p": "p", "pee": "p", "papa": "p",
    "q": "q", "queue": "q", "quebec": "q",
    "r": "r", "ar": "r", "are": "r", "romeo": "r",
    "s": "s", "ess": "s", "sierra": "s",
    "t": "t", "tee": "t", "tango": "t",
    "u": "u", "you": "u", "uniform": "u",
    "v": "v", "vee": "v", "victor": "v",
    "w": "w", "doubleyou": "w", "whiskey": "w",
    "x": "x", "ex": "x", "xray": "x",
    "y": "y", "why": "y", "yankee": "y",
    "z": "z", "zee": "z", "zed": "z", "zulu": "z",
}

BASIC_CORRECTIONS = {
    "note pad": "notepad",
    "calc later": "calculator",
    "good bye": "goodbye",
    "go to slip": "go to sleep",
    "open crew": "open chrome",
    "open grown": "open chrome",
    "hey usher": "hey asher",
    "hey asha": "hey asher",
    "okay usher": "okay asher",
    "ok usher": "ok asher",
}


def _comparison_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@lru_cache(maxsize=1)
def load_voice_vocabulary() -> tuple[tuple[str, ...], dict[str, str]]:
    contacts = list(DEFAULT_CONTACTS)
    aliases = dict(DEFAULT_ALIASES)

    try:
        data = json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))

        for contact in data.get("contacts", []):
            contact = str(contact).strip()
            if contact and contact not in contacts:
                contacts.append(contact)

        for spoken, canonical in data.get("aliases", {}).items():
            spoken = str(spoken).strip().lower()
            canonical = str(canonical).strip()
            if spoken and canonical:
                aliases[spoken] = canonical

    except (OSError, json.JSONDecodeError, TypeError):
        # Defaults keep Asher working even if the optional JSON file is broken.
        pass

    return tuple(contacts), aliases


def get_contact_names() -> tuple[str, ...]:
    return load_voice_vocabulary()[0]


def get_initial_prompt() -> str:
    contacts = ", ".join(get_contact_names())
    return (
        "A Windows voice assistant named Asher. Commands include: Hey Asher, "
        "open Chrome, close Chrome, open WhatsApp, search WhatsApp, search for "
        "a contact, send a WhatsApp message, send it, don't send it, increase "
        "volume, decrease volume, take a screenshot, go to sleep, and goodbye. "
        f"Contact names include: {contacts}. Names may be spelled letter by letter."
    )


def collapse_spelled_name(text: str) -> str:
    """Join a contact name spoken or written one letter at a time."""

    text = text.strip().lower()
    if not text:
        return ""

    # Make forms such as t-h-a-r-i-k-a and t.h.a.r.i.k.a tokenizable.
    separated = re.sub(r"(?<=[a-z])[-.,](?=[a-z])", " ", text)
    tokens = re.findall(r"[a-z]+", separated)

    if len(tokens) < 2:
        return text

    letters: list[str] = []

    for token in tokens:
        compact = token.replace("-", "")
        letter = SPOKEN_LETTERS.get(compact)

        if letter is None:
            return text

        letters.append(letter)

    return "".join(letters).title()


def normalise_contact_name(contact: str, minimum_similarity: float = 0.68) -> str:
    contact = str(contact).strip(" \t\r\n,.-")
    if not contact:
        return ""

    contacts, aliases = load_voice_vocabulary()
    lowered = " ".join(contact.lower().split())

    if lowered in aliases:
        return aliases[lowered]

    collapsed = collapse_spelled_name(contact)
    candidate = collapsed if collapsed != lowered else contact
    candidate_key = _comparison_key(candidate)

    if not candidate_key:
        return contact

    # Exact match after removing spaces and punctuation.
    for known in contacts:
        if candidate_key == _comparison_key(known):
            return known

    # Alias matching after punctuation removal.
    for spoken, canonical in aliases.items():
        if candidate_key == _comparison_key(spoken):
            return canonical

    best_name = ""
    best_score = 0.0

    for known in contacts:
        score = difflib.SequenceMatcher(
            None,
            candidate_key,
            _comparison_key(known),
        ).ratio()

        if score > best_score:
            best_score = score
            best_name = known

    # Short fragments are easier to match accidentally, so require a slightly
    # stronger score for them.
    threshold = minimum_similarity + (0.08 if len(candidate_key) <= 4 else 0.0)
    return best_name if best_score >= threshold else candidate.strip().title()


def _clean_surface_text(command: str) -> str:
    command = command.replace("’", "'").replace("`", "'")
    command = command.lower().strip()
    command = re.sub(r"[^\w\s'.,-]", " ", command)
    command = " ".join(command.split())

    for wrong, correct in BASIC_CORRECTIONS.items():
        command = command.replace(wrong, correct)

    return " ".join(command.split())


def strip_terminal_punctuation(text: str) -> str:
    """Remove sentence punctuation that should not change command intent.

    Whisper frequently returns short controls as ``send it.`` or ``goodbye!``.
    The punctuation is useful in prose but must not prevent exact command
    matching. Internal punctuation is preserved so WhatsApp message content is
    not damaged.
    """

    return str(text).strip().rstrip(" \t\r\n.,!?;:")


def normalise_voice_command(command: str) -> str:
    """Repair a Whisper transcript without changing ordinary message content."""

    command = strip_terminal_punctuation(_clean_surface_text(str(command)))
    if not command:
        return ""

    # Whisper regularly hears "search" as "touch" or "such". These are only
    # corrected when they occur as a command prefix.
    prefix_match = re.match(r"^(search|touch|such|find)\s+(.+)$", command)
    if prefix_match:
        contact = normalise_contact_name(prefix_match.group(2))
        return f"search {contact}".strip()

    # Preserve the message exactly as much as possible; repair only the contact
    # after the final "to" in commands such as "send hi to T H A R I K A".
    if command.startswith("send ") and " to " in command:
        content, contact = command.rsplit(" to ", 1)
        contact = normalise_contact_name(contact)
        return f"{content} to {contact}".strip()

    # A wake phrase may contain a direct search command.
    wake_search = re.match(
        r"^(hey asher|hello asher|okay asher|ok asher|asher)\s+"
        r"(search|touch|such|find)\s+(.+)$",
        command,
    )
    if wake_search:
        contact = normalise_contact_name(wake_search.group(3))
        return f"{wake_search.group(1)} search {contact}".strip()

    return command


def transcript_mentions_contact_command(text: str) -> bool:
    text = text.lower()
    markers = ("search", "touch", "such", "find", "send", "whatsapp")
    return any(marker in text for marker in markers)
