"""Compatibility helpers backed by the ambiguity-safe vocabulary resolver."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .transcription import normalize_transcript_surface
from .vocabulary import CommandResolution, DynamicVocabulary, NameResolution
from .vocabulary import collapse_spelled_name as _collapse_spelled_name


@lru_cache(maxsize=1)
def default_vocabulary() -> DynamicVocabulary:
    root = Path(__file__).resolve().parents[2]
    return DynamicVocabulary(
        contacts_path=root / "data" / "voice_contacts.json",
        applications_path=root / "data" / "apps.json",
    )


def resolve_contact_name(
    contact: str,
    *,
    vocabulary: DynamicVocabulary | None = None,
) -> NameResolution:
    return (vocabulary or default_vocabulary()).resolve_contact(contact)


def collapse_spelled_name(value: str) -> str:
    """Legacy-shaped helper; callers needing the flag can use the resolver."""

    collapsed, was_spelled = _collapse_spelled_name(value)
    return collapsed.title() if was_spelled else collapsed


def get_contact_names(*, vocabulary: DynamicVocabulary | None = None) -> tuple[str, ...]:
    return (vocabulary or default_vocabulary()).contacts


def get_initial_prompt(*, vocabulary: DynamicVocabulary | None = None) -> str:
    terms = (vocabulary or default_vocabulary()).prompt_terms()
    suffix = ", ".join(terms)
    return (
        "A Windows voice assistant named Asher. Use English commands and preserve "
        f"names exactly. Local vocabulary: {suffix}."
    )


def transcript_mentions_contact_command(text: str) -> bool:
    return bool(
        __import__("re").search(
            r"\b(?:search|touch|such|find|send|whatsapp)\b",
            str(text),
            flags=__import__("re").IGNORECASE,
        )
    )


def resolve_voice_command(
    command: str,
    *,
    vocabulary: DynamicVocabulary | None = None,
    contact_expected: bool | None = None,
) -> CommandResolution:
    resolver = vocabulary or default_vocabulary()
    return resolver.repair_contact_command(
        command,
        assume_plain_search_is_contact=True if contact_expected is None else contact_expected,
    )


def normalise_contact_name(
    contact: str,
    *,
    vocabulary: DynamicVocabulary | None = None,
) -> str:
    result = resolve_contact_name(contact, vocabulary=vocabulary)
    return result.candidate if result.resolved and result.candidate else str(contact).strip()


def normalise_voice_command(
    command: str,
    *,
    vocabulary: DynamicVocabulary | None = None,
    contact_expected: bool | None = None,
) -> str:
    value = normalize_transcript_surface(command)
    result = resolve_voice_command(
        value,
        vocabulary=vocabulary,
        contact_expected=contact_expected,
    )
    return result.resolved_text if result.executable else value


__all__ = [
    "default_vocabulary",
    "collapse_spelled_name",
    "get_contact_names",
    "get_initial_prompt",
    "normalise_contact_name",
    "normalise_voice_command",
    "normalize_transcript_surface",
    "resolve_contact_name",
    "resolve_voice_command",
    "transcript_mentions_contact_command",
]
