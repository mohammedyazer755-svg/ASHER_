"""Secret detection and log/prompt redaction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


REDACTED = "[REDACTED]"
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "passcode",
    "pin",
    "secret",
    "token",
}

PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    # Common provider/token formats.  These are intentionally conservative;
    # ordinary words such as ``tokenize`` should not be treated as secrets.
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat|xox[baprs]-)[A-Za-z0-9_-]{10,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:password|passcode|pin|api[ _-]?key|secret|token)\s*(?:is|=|:)\s*[^\s,;]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
)


def redact_text(value: Any) -> str:
    text = str(value)
    for pattern in PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            output[str(key)] = REDACTED if normalized in SENSITIVE_KEYS else redact_mapping(item)
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_mapping(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def contains_prohibited_secret(text: str) -> bool:
    original = str(text)
    return redact_text(original) != original
