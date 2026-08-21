"""In-process PC mock used by integration tests and faculty demos.

It intentionally has no socket listener or side effects. A real desktop
adapter can wrap the same invitation/request/response methods around its
chosen authenticated transport.
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import PcInvitation, complete_request, make_invitation


@dataclass
class PcPairingMock:
    invitation: PcInvitation
    channel_key: bytes | None = None

    @classmethod
    def create(cls, *, pairing_id: str = "demo-pairing", code: str = "ASHER42") -> "PcPairingMock":
        return cls(make_invitation(pairing_id=pairing_id, code=code))

    def accept_request(self, request: dict[str, object]) -> dict[str, object]:
        response, key = complete_request(self.invitation, request)
        self.channel_key = key
        return response
