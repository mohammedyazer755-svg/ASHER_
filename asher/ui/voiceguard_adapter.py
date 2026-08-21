"""Optional real VoiceGuard recording/training bridge for the desktop UI.

The adapter never opens a microphone at import or construction time.  A UI
caller must first call :meth:`begin_user` with explicit consent; each sample
is stored in a consented, session-separated manifest owned by the VoiceGuard
package.  Missing audio/ML dependencies produce a clear error instead of a
simulated enrollment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from asher.voiceguard import EnrollmentError, EnrollmentManager, SpeakerRole


class VoiceGuardDesktopAdapter:
    def __init__(self, root: str | Path, *, duration_seconds: float = 3.0, device: int | str | None = None) -> None:
        self.manager = EnrollmentManager(Path(root) / "voiceguard")
        self.duration_seconds = max(0.5, min(float(duration_seconds), 30.0))
        self.device = device
        self._consented: dict[str, str] = {}

    def begin_user(self, user_id: str, role: str, *, consent: bool) -> None:
        if not consent:
            raise PermissionError("Explicit consent is required before retaining voice recordings")
        self._consented[user_id] = role

    def capture_sample(self, user_id: str, *, contains_wake_phrase: bool = False) -> int:
        role = self._consented.get(user_id)
        if role is None:
            raise PermissionError("Confirm voice-recording consent before capturing a sample")
        selected_role = {
            "owner": SpeakerRole.OWNER,
            "trusted": SpeakerRole.TRUSTED,
            "guest": SpeakerRole.UNKNOWN,
        }.get(role, SpeakerRole.UNKNOWN)
        session = self.manager.begin_enrollment(
            user_id,
            role=selected_role,
            environment="desktop_ui",
            consent=True,
        )
        session.record_microphone(
            self.duration_seconds,
            contains_wake_phrase=contains_wake_phrase,
            expected_authorized=role in {"owner", "trusted"},
            device=self.device,
        )
        self.manager.finalize_enrollment(session, minimum_samples=1)
        return sum(
            len(item.session_ids)
            for item in self.manager.list_users(include_revoked=False)
            if item.user_id == user_id
        )

    def train(self, user_id: str) -> dict[str, Any]:
        # Training is deliberately delegated to the honest session-separated
        # trainer; it raises when there are too few sessions/classes or optional
        # ML dependencies are unavailable.
        result = self.manager.retrain()
        return {
            "model_version": result.model.metadata.get("model_version"),
            "measured_test": result.measured_test,
            "test_samples": len(result.split.test),
            "user_id": user_id,
        }

    def revoke(self, user_id: str) -> None:
        try:
            self.manager.revoke_user(user_id, delete_recordings=False)
        except EnrollmentError:
            # A user may be revoked from the account store before the first
            # consented sample exists; there is then no VoiceGuard registry
            # entry to revoke.
            pass
        self._consented.pop(user_id, None)


__all__ = ["VoiceGuardDesktopAdapter"]
