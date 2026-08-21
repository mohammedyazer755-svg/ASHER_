"""Dynamic local vocabulary and ambiguity-safe name resolution."""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import threading
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ResolutionStatus(str, Enum):
    EXACT = "exact"
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NameResolution:
    query: str
    status: ResolutionStatus
    candidate: str | None = None
    alternatives: tuple[str, ...] = ()
    score: float = 0.0
    second_score: float = 0.0
    spelled: bool = False

    @property
    def resolved(self) -> bool:
        return self.status in {ResolutionStatus.EXACT, ResolutionStatus.MATCHED}

    @property
    def requires_clarification(self) -> bool:
        return self.status in {ResolutionStatus.AMBIGUOUS, ResolutionStatus.UNKNOWN}


@dataclass(frozen=True)
class CommandResolution:
    original_text: str
    resolved_text: str
    contact: NameResolution | None = None
    clarification: str | None = None

    @property
    def executable(self) -> bool:
        return self.clarification is None


SPOKEN_LETTERS: dict[str, str] = {
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
    "w": "w", "doubleyou": "w", "double-you": "w", "whiskey": "w",
    "x": "x", "ex": "x", "xray": "x", "x-ray": "x",
    "y": "y", "why": "y", "yankee": "y",
    "z": "z", "zee": "z", "zed": "z", "zulu": "z",
}


def comparison_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(character for character in normalized if character.isalnum())


def collapse_spelled_name(value: str) -> tuple[str, bool]:
    """Collapse a name only when every separated token is a spoken letter."""

    original = " ".join(str(value).strip().split())
    if not original:
        return "", False
    separated = re.sub(r"(?<=[A-Za-z])\s*[-.,]\s*(?=[A-Za-z])", " ", original)
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", separated.casefold())
    if len(tokens) < 2:
        return original, False
    combined: list[str] = []
    index = 0
    while index < len(tokens):
        pair = tuple(tokens[index : index + 2])
        if pair in {("double", "you"), ("x", "ray")}:
            combined.append("-".join(pair))
            index += 2
        else:
            combined.append(tokens[index])
            index += 1
    tokens = combined
    letters: list[str] = []
    for token in tokens:
        letter = SPOKEN_LETTERS.get(token)
        if letter is None:
            return original, False
        letters.append(letter)
    return "".join(letters), True


def _phonetic_key(value: str) -> str:
    """Small locale-neutral Soundex-like signal used only as a tie-breaker."""

    key = comparison_key(value)
    if not key:
        return ""
    groups = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    encoded: list[str] = []
    previous = groups.get(key[0], "")
    for character in key[1:]:
        current = groups.get(character, "")
        if current and current != previous:
            encoded.append(current)
        previous = current
    return (key[0] + "".join(encoded) + "000")[:4]


def _safe_json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def discover_windows_applications(*, timeout_seconds: float = 8.0) -> tuple[str, ...]:
    """Read installed Start-menu names through a fixed, read-only command.

    The command contains no user-provided text and is invoked only when a
    caller explicitly refreshes the dynamic vocabulary.  Non-Windows hosts and
    unavailable PowerShell simply return an empty tuple.
    """

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-StartApps | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(0.1, float(timeout_seconds)),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()
    if result.returncode != 0 or not result.stdout.strip():
        return ()
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    return ()


def _fingerprint(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


class DynamicVocabulary:
    """Reloadable contacts/applications kept entirely on the local device."""

    def __init__(
        self,
        *,
        contacts: Iterable[str] = (),
        applications: Iterable[str] = (),
        aliases: Mapping[str, str] | None = None,
        application_aliases: Mapping[str, str] | None = None,
        contacts_path: str | Path | None = None,
        applications_path: str | Path | None = None,
        application_provider: Callable[[], Iterable[str]] | None = None,
        minimum_score: float = 0.72,
        ambiguity_margin: float = 0.10,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        if not 0.0 <= ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be between 0 and 1")
        self.contacts_path = Path(contacts_path) if contacts_path else None
        self.applications_path = Path(applications_path) if applications_path else None
        self.application_provider = application_provider
        self.minimum_score = float(minimum_score)
        self.ambiguity_margin = float(ambiguity_margin)
        self._base_contacts = tuple(self._clean_names(contacts))
        self._base_applications = tuple(self._clean_names(applications))
        self._base_aliases = {
            " ".join(str(key).casefold().split()): str(value).strip()
            for key, value in (aliases or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._base_application_aliases = {
            " ".join(str(key).casefold().split()): str(value).strip()
            for key, value in (application_aliases or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._contacts: tuple[str, ...] = ()
        self._applications: tuple[str, ...] = ()
        self._aliases: dict[str, str] = {}
        self._application_aliases: dict[str, str] = {}
        self._fingerprints: tuple[Any, ...] | None = None
        self._lock = threading.RLock()
        self.refresh(force=True)

    @staticmethod
    def _clean_names(values: Iterable[Any]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(value).strip()
                for value in values
                if value is not None and str(value).strip()
            )
        )

    @staticmethod
    def _contacts_from_data(data: Any) -> tuple[tuple[str, ...], dict[str, str]]:
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return tuple(str(item).strip() for item in data if str(item).strip()), {}
        if not isinstance(data, Mapping):
            return (), {}
        contacts: list[str] = []
        aliases: dict[str, str] = {}
        raw_contacts = data.get("contacts", ())
        if isinstance(raw_contacts, Mapping):
            raw_contacts = tuple(raw_contacts.keys())
        if isinstance(raw_contacts, Sequence) and not isinstance(raw_contacts, (str, bytes)):
            for item in raw_contacts:
                if isinstance(item, Mapping):
                    name = str(item.get("name", "")).strip()
                    if name:
                        contacts.append(name)
                        raw_aliases = item.get("aliases", ())
                        if isinstance(raw_aliases, Sequence) and not isinstance(
                            raw_aliases, (str, bytes)
                        ):
                            for alias in raw_aliases:
                                if str(alias).strip():
                                    aliases[" ".join(str(alias).casefold().split())] = name
                elif str(item).strip():
                    contacts.append(str(item).strip())
        raw_aliases = data.get("aliases", {})
        if isinstance(raw_aliases, Mapping):
            for alias, target in raw_aliases.items():
                if str(alias).strip() and str(target).strip():
                    aliases[" ".join(str(alias).casefold().split())] = str(target).strip()
        return tuple(contacts), aliases

    @staticmethod
    def _applications_from_data(data: Any) -> tuple[str, ...]:
        if isinstance(data, Mapping):
            if "applications" in data:
                data = data.get("applications", ())
            else:
                return tuple(
                    str(name).strip()
                    for name in data
                    if str(name).strip() and str(name).casefold() != "aliases"
                )
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            values: list[str] = []
            for item in data:
                if isinstance(item, Mapping):
                    item = item.get("name", "")
                if str(item).strip():
                    values.append(str(item).strip())
            return tuple(values)
        return ()

    def refresh(self, *, force: bool = False) -> bool:
        fingerprints = (
            _fingerprint(self.contacts_path),
            _fingerprint(self.applications_path),
        )
        with self._lock:
            if not force and self.application_provider is None and fingerprints == self._fingerprints:
                return False
            file_contacts, file_aliases = self._contacts_from_data(
                _safe_json(self.contacts_path)
            )
            file_apps = self._applications_from_data(_safe_json(self.applications_path))
            provided_apps: tuple[str, ...] = ()
            if self.application_provider is not None:
                try:
                    provided_apps = self._clean_names(self.application_provider())
                except Exception:
                    provided_apps = ()
            self._contacts = self._clean_names((*self._base_contacts, *file_contacts))
            self._applications = self._clean_names(
                (*self._base_applications, *file_apps, *provided_apps)
            )
            self._aliases = {**self._base_aliases, **file_aliases}
            self._application_aliases = dict(self._base_application_aliases)
            self._fingerprints = fingerprints
            return True

    @property
    def contacts(self) -> tuple[str, ...]:
        self.refresh()
        with self._lock:
            return self._contacts

    @property
    def applications(self) -> tuple[str, ...]:
        self.refresh()
        with self._lock:
            return self._applications

    def contacts_and_aliases(self) -> tuple[tuple[str, ...], dict[str, str]]:
        """Return the current local contact vocabulary for another local component.

        The caller is responsible for keeping this data on-device; this method
        does not log or transmit it.
        """
        self.refresh()
        with self._lock:
            return self._contacts, dict(self._aliases)

    def prompt_terms(self, *, max_contacts: int = 80, max_applications: int = 80) -> tuple[str, ...]:
        # This data is intended only for a local decoder prompt.  Callers must
        # not log it or transmit it to a remote provider without consent.
        return self.contacts[:max_contacts] + self.applications[:max_applications]

    def replace_contacts(
        self,
        contacts: Iterable[str],
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        with self._lock:
            self._base_contacts = self._clean_names(contacts)
            self._base_aliases = {
                " ".join(str(key).casefold().split()): str(value).strip()
                for key, value in (aliases or {}).items()
                if str(key).strip() and str(value).strip()
            }
        self.refresh(force=True)

    def replace_applications(self, applications: Iterable[str]) -> None:
        with self._lock:
            self._base_applications = self._clean_names(applications)
        self.refresh(force=True)

    def replace_application_aliases(self, aliases: Mapping[str, str]) -> None:
        with self._lock:
            self._base_application_aliases = {
                " ".join(str(key).casefold().split()): str(value).strip()
                for key, value in aliases.items()
                if str(key).strip() and str(value).strip()
            }
        self.refresh(force=True)

    def refresh_installed_applications(self, *, timeout_seconds: float = 8.0) -> tuple[str, ...]:
        """Explicitly refresh the local Start-menu application vocabulary."""

        discovered = discover_windows_applications(timeout_seconds=timeout_seconds)
        with self._lock:
            self._base_applications = self._clean_names(
                (*self._base_applications, *discovered)
            )
        self.refresh(force=True)
        return discovered

    # Naming parallel to the legacy app manager makes UI integration clearer.
    refresh_app_catalog = refresh_installed_applications

    def _resolve(self, query: str, names: Sequence[str], *, aliases: Mapping[str, str]) -> NameResolution:
        original = " ".join(str(query).strip(" \t\r\n,.;:!?").split())
        collapsed, spelled = collapse_spelled_name(original)
        candidate_text = collapsed if spelled else original
        key = comparison_key(candidate_text)
        if not key:
            return NameResolution(query=original, status=ResolutionStatus.UNKNOWN, spelled=spelled)

        alias_target = aliases.get(" ".join(original.casefold().split()))
        if alias_target:
            return NameResolution(
                query=original,
                status=ResolutionStatus.EXACT,
                candidate=alias_target,
                alternatives=(alias_target,),
                score=1.0,
                spelled=spelled,
            )

        exact = tuple(name for name in names if comparison_key(name) == key)
        if len(exact) == 1:
            return NameResolution(
                query=original,
                status=ResolutionStatus.EXACT,
                candidate=exact[0],
                alternatives=exact,
                score=1.0,
                spelled=spelled,
            )
        if len(exact) > 1:
            return NameResolution(
                query=original,
                status=ResolutionStatus.AMBIGUOUS,
                alternatives=exact,
                score=1.0,
                second_score=1.0,
                spelled=spelled,
            )

        query_phonetic = _phonetic_key(candidate_text)
        scored: list[tuple[float, str]] = []
        for name in names:
            name_key = comparison_key(name)
            sequence_score = difflib.SequenceMatcher(None, key, name_key).ratio()
            # Spoken users often provide only the first name.  Treat an exact
            # token match as strong evidence, while still applying the normal
            # ambiguity-margin check when several people share that token.
            name_tokens = tuple(
                comparison_key(token)
                for token in re.findall(r"[\w-]+", name, flags=re.UNICODE)
            )
            if key in name_tokens:
                sequence_score = max(sequence_score, 0.93)
            elif any(
                token.startswith(key) or key.startswith(token)
                for token in name_tokens
                if token
            ):
                sequence_score = max(sequence_score, 0.78)
            if query_phonetic and query_phonetic == _phonetic_key(name):
                sequence_score = min(1.0, sequence_score + 0.04)
            scored.append((sequence_score, name))
        scored.sort(key=lambda item: (-item[0], item[1].casefold()))
        if not scored:
            return NameResolution(query=original, status=ResolutionStatus.UNKNOWN, spelled=spelled)
        top_score, top_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        alternatives = tuple(name for score, name in scored[:3] if score >= self.minimum_score)
        if top_score < self.minimum_score:
            return NameResolution(
                query=original,
                status=ResolutionStatus.UNKNOWN,
                alternatives=alternatives,
                score=top_score,
                second_score=second_score,
                spelled=spelled,
            )
        if len(scored) > 1 and second_score >= self.minimum_score and (
            top_score - second_score < self.ambiguity_margin
        ):
            return NameResolution(
                query=original,
                status=ResolutionStatus.AMBIGUOUS,
                alternatives=alternatives or (top_name, scored[1][1]),
                score=top_score,
                second_score=second_score,
                spelled=spelled,
            )
        return NameResolution(
            query=original,
            status=ResolutionStatus.MATCHED,
            candidate=top_name,
            alternatives=(top_name,),
            score=top_score,
            second_score=second_score,
            spelled=spelled,
        )

    def resolve_contact(self, query: str) -> NameResolution:
        self.refresh()
        with self._lock:
            return self._resolve(query, self._contacts, aliases=self._aliases)

    def resolve_application(self, query: str) -> NameResolution:
        self.refresh()
        with self._lock:
            return self._resolve(query, self._applications, aliases=self._application_aliases)

    @staticmethod
    def _clarification(result: NameResolution) -> str:
        if result.status == ResolutionStatus.AMBIGUOUS and result.alternatives:
            choices = " or ".join(result.alternatives[:3])
            return f"Which contact did you mean: {choices}?"
        return "I could not match that contact. Please say or spell the full name."

    def repair_contact_command(
        self,
        text: str,
        *,
        assume_plain_search_is_contact: bool = True,
    ) -> CommandResolution:
        original = " ".join(str(text).strip().split())
        search = re.match(
            r"^(?P<wake>(?:(?:hey|hello|okay|ok)\s+asher|asher)\s+)?"
            r"(?P<verb>search|touch|such|find)\s+(?P<target>.+)$",
            original,
            flags=re.IGNORECASE,
        )
        if search and assume_plain_search_is_contact:
            target = re.sub(r"^(?:whatsapp\s+)?(?:for\s+)?", "", search.group("target"), flags=re.I)
            target = re.sub(r"\s+(?:please|thanks|thank you)$", "", target, flags=re.I)
            result = self.resolve_contact(target)
            if not result.resolved:
                return CommandResolution(
                    original_text=original,
                    resolved_text=original,
                    contact=result,
                    clarification=self._clarification(result),
                )
            wake = search.group("wake") or ""
            resolved = f"{wake}search {result.candidate}".strip()
            return CommandResolution(original, resolved, result)

        sending = re.match(
            r"^(?P<head>send\s+.+\s+to\s+)(?P<target>.+)$",
            original,
            flags=re.IGNORECASE,
        )
        if sending:
            target = re.sub(
                r"\s+(?:please|thanks|thank you)$",
                "",
                sending.group("target"),
                flags=re.I,
            )
            result = self.resolve_contact(target)
            if not result.resolved:
                return CommandResolution(
                    original_text=original,
                    resolved_text=original,
                    contact=result,
                    clarification=self._clarification(result),
                )
            return CommandResolution(
                original,
                sending.group("head") + str(result.candidate),
                result,
            )
        return CommandResolution(original, original)
