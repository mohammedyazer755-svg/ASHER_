"""Strong-authentication adapter; unavailable platforms fail closed."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StrongAuthResult:
    verified: bool
    method: str
    reason: str


class StrongAuthenticator(Protocol):
    def verify(self, prompt: str) -> StrongAuthResult: ...


class DenyStrongAuthenticator:
    def verify(self, prompt: str) -> StrongAuthResult:
        return StrongAuthResult(False, "unavailable", "Strong device authentication is unavailable; action denied")


class WindowsHelloAuthenticator:
    """Use Windows UserConsentVerifier when its WinRT binding is installed."""

    def verify(self, prompt: str) -> StrongAuthResult:
        try:
            from winrt.windows.security.credentials.ui import (  # type: ignore[import-not-found]
                UserConsentVerificationResult,
                UserConsentVerifier,
            )
        except Exception:
            return StrongAuthResult(False, "windows_hello", "Windows Hello binding is unavailable; action denied")

        async def request() -> object:
            availability = await UserConsentVerifier.check_availability_async()
            # Availability enum zero is Available in the WinRT API. Avoid
            # weakening the check if bindings expose an unexpected value.
            if str(availability).lower().split(".")[-1] not in {"available", "0"}:
                return None
            return await UserConsentVerifier.request_verification_async(prompt)

        try:
            result = asyncio.run(request())
        except Exception as error:
            return StrongAuthResult(False, "windows_hello", f"Windows Hello failed: {type(error).__name__}")

        verified = result == UserConsentVerificationResult.VERIFIED
        return StrongAuthResult(
            verified,
            "windows_hello",
            "Device authentication verified" if verified else "Device authentication was not verified",
        )

