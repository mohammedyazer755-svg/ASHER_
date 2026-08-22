"""Optional real VoiceGuard recording/training bridge for the desktop UI.

The adapter never opens a microphone at import or construction time.  A UI
caller must first call :meth:`begin_user` with explicit consent; each sample
is added to a bounded multi-clip recording session owned by the VoiceGuard
package.  Partial sessions remain unregistered until the complete collection
is captured. Missing audio/ML dependencies produce a clear error instead of
a simulated enrollment.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from asher.voiceguard import (
    DatasetError,
    EnrollmentError,
    EnrollmentManager,
    RecordingSession,
    SpeakerRole,
    load_dataset,
)


DEFAULT_SESSION_CLIPS = 6
MIN_SESSION_CLIPS = 3
MAX_SESSION_CLIPS = 20
DEFAULT_SESSION_GAP_SECONDS = 1_800.0


class VoiceGuardDesktopAdapter:
    def __init__(
        self,
        root: str | Path,
        *,
        duration_seconds: float = 3.0,
        device: int | str | None = None,
        samples_per_session: int = DEFAULT_SESSION_CLIPS,
        environment: str = "desktop_ui",
        minimum_session_gap_seconds: float = DEFAULT_SESSION_GAP_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.manager = EnrollmentManager(Path(root) / "voiceguard")
        self.duration_seconds = max(0.5, min(float(duration_seconds), 30.0))
        self.device = device
        self.samples_per_session = max(
            MIN_SESSION_CLIPS,
            min(int(samples_per_session), MAX_SESSION_CLIPS),
        )
        if not environment.strip():
            raise ValueError("environment is required")
        if minimum_session_gap_seconds < 0:
            raise ValueError("minimum_session_gap_seconds cannot be negative")
        self.environment = environment.strip()
        self.minimum_session_gap_seconds = float(minimum_session_gap_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._consented: dict[str, tuple[str, str]] = {}
        self._pending_sessions: dict[str, RecordingSession] = {}

    def _registry_token(self) -> str:
        """Bind UI consent to the exact enrollment generation it observed."""

        try:
            payload = self.manager.registry_path.read_bytes()
        except FileNotFoundError:
            payload = b"<missing-registry>"
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _speaker_role(role: str) -> SpeakerRole:
        return {
            "owner": SpeakerRole.OWNER,
            "trusted": SpeakerRole.TRUSTED,
            "guest": SpeakerRole.UNKNOWN,
        }.get(role, SpeakerRole.UNKNOWN)

    @staticmethod
    def _parse_aware_timestamp(value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed

    def _unregistered_sessions(self, user_id: str) -> tuple[RecordingSession, ...]:
        if not self.manager.recordings_root.is_dir():
            return ()
        registered = {
            session_id
            for record in self.manager.list_users(include_revoked=True)
            for session_id in record.session_ids
        }
        dataset = load_dataset(
            self.manager.recordings_root,
            include_revoked=True,
            verify_checksums=True,
        )
        output: list[RecordingSession] = []
        for manifest in dataset.sessions:
            if (
                manifest.speaker_id == user_id
                and not manifest.revoked
                and manifest.session_id not in registered
            ):
                output.append(
                    RecordingSession.open(
                        self.manager.recordings_root / manifest.session_id
                    )
                )
        return tuple(output)

    def _recover_partial_session(
        self,
        user_id: str,
        selected_role: SpeakerRole,
    ) -> RecordingSession | None:
        candidates = self._unregistered_sessions(user_id)
        if not candidates:
            return None
        if len(candidates) != 1:
            raise DatasetError(
                "Multiple unfinished VoiceGuard collections require explicit recovery; "
                "no recording was opened or registered."
            )
        candidate = candidates[0]
        if (
            candidate.manifest.role != selected_role.value
            or candidate.manifest.environment != self.environment
        ):
            raise EnrollmentError(
                "An unfinished VoiceGuard collection has different role or environment metadata; "
                "use the original collector or revoke it explicitly."
            )
        return candidate

    def begin_user(self, user_id: str, role: str, *, consent: bool) -> int:
        if not consent:
            raise PermissionError("Explicit consent is required before retaining voice recordings")
        if not user_id.strip():
            raise ValueError("user_id is required")
        selected_role = self._speaker_role(role)
        with self._lock, self.manager._lifecycle_transaction():
            pending = self._recover_partial_session(user_id, selected_role)
            if pending is None:
                self._pending_sessions.pop(user_id, None)
            else:
                self._pending_sessions[user_id] = pending
            self._consented[user_id] = (role, self._registry_token())
            return self._reported_clip_count(user_id)

    def _registered_clip_count(self, user_id: str) -> int:
        dataset = self.manager.load_training_dataset()
        return sum(sample.speaker_id == user_id for sample in dataset.samples)

    def _reported_clip_count(self, user_id: str) -> int:
        pending = self._pending_sessions.get(user_id)
        pending_count = len(pending.manifest.samples) if pending is not None else 0
        return self._registered_clip_count(user_id) + pending_count

    def _require_independent_collection_window(self, user_id: str) -> None:
        dataset = self.manager.load_training_dataset()
        same_environment = tuple(
            manifest
            for manifest in dataset.sessions
            if manifest.speaker_id == user_id
            and " ".join(manifest.environment.casefold().split())
            == " ".join(self.environment.casefold().split())
        )
        if not same_environment:
            return
        completed_values: list[datetime] = []
        for manifest in same_environment:
            timestamps = (
                self._parse_aware_timestamp(manifest.created_at),
                *(
                    self._parse_aware_timestamp(sample.created_at)
                    for sample in manifest.samples
                ),
            )
            if any(value is None for value in timestamps):
                break
            completed_values.append(
                max(value for value in timestamps if value is not None)
            )
        if len(completed_values) != len(same_environment):
            raise DatasetError(
                "A prior VoiceGuard session lacks valid collection timing metadata; "
                "use explicit recovery before recording again."
            )
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise DatasetError("VoiceGuard collection requires a timezone-aware clock")
        elapsed = (now - max(completed_values)).total_seconds()
        if elapsed < self.minimum_session_gap_seconds:
            raise DatasetError(
                "Start the next VoiceGuard session later, after the collection pause, "
                "or use the guided collector in a genuinely new environment. "
                "Session metadata is only a conservative guard and does not prove physical independence."
            )

    def capture_sample(self, user_id: str, *, contains_wake_phrase: bool = False) -> int:
        # One UI click captures one clip. Calls for the same user remain in one
        # manifest until the bounded collection is complete; only then does it
        # become a trainable recording session.
        with self._lock, self.manager._lifecycle_transaction():
            consent_state = self._consented.get(user_id)
            if consent_state is None:
                raise PermissionError("Confirm voice-recording consent before capturing a sample")
            role, registry_token = consent_state
            if registry_token != self._registry_token():
                self._consented.pop(user_id, None)
                self._pending_sessions.pop(user_id, None)
                raise PermissionError(
                    "VoiceGuard enrollment changed; confirm recording consent again"
                )
            selected_role = self._speaker_role(role)
            cached = self._pending_sessions.get(user_id)
            if cached is not None:
                try:
                    refreshed_cached = RecordingSession.open(cached.directory)
                except Exception as exc:
                    self._consented.pop(user_id, None)
                    self._pending_sessions.pop(user_id, None)
                    raise PermissionError(
                        "The consented VoiceGuard collection changed; confirm consent again"
                    ) from exc
                if refreshed_cached.manifest.revoked:
                    self._consented.pop(user_id, None)
                    self._pending_sessions.pop(user_id, None)
                    raise PermissionError(
                        "The consented VoiceGuard collection was revoked; confirm consent again"
                    )
            session = self._recover_partial_session(user_id, selected_role)
            if session is None:
                self._pending_sessions.pop(user_id, None)
                self._require_independent_collection_window(user_id)
                session = self.manager.begin_enrollment(
                    user_id,
                    role=selected_role,
                    environment=self.environment,
                    consent=True,
                )
                self._pending_sessions[user_id] = session
            elif session.manifest.role != selected_role.value:
                raise EnrollmentError("an in-progress collection cannot change speaker role")
            else:
                self._pending_sessions[user_id] = session

            # If finalization previously failed after the last clip was safely
            # persisted, retry registration without recording an extra clip.
            if len(session.manifest.samples) < self.samples_per_session:
                session.record_microphone(
                    self.duration_seconds,
                    contains_wake_phrase=contains_wake_phrase,
                    expected_authorized=role in {"owner", "trusted"},
                    device=self.device,
                )
            if len(session.manifest.samples) >= self.samples_per_session:
                self.manager.finalize_enrollment(
                    session,
                    minimum_samples=self.samples_per_session,
                )
                self._pending_sessions.pop(user_id, None)
                self._consented[user_id] = (role, self._registry_token())
            return self._reported_clip_count(user_id)

    def train(self, user_id: str) -> dict[str, Any]:
        # Training uses finalized manifests only. A partially collected batch
        # is deliberately not promoted into a train/validation/test session.
        with self._lock, self.manager._lifecycle_transaction():
            cached = self._pending_sessions.get(user_id)
            unfinished = self._unregistered_sessions(user_id)
            pending = next(
                (
                    item
                    for item in unfinished
                    if cached is not None
                    and item.manifest.session_id == cached.manifest.session_id
                ),
                None,
            )
            if pending is not None:
                self._pending_sessions[user_id] = pending
                captured = len(pending.manifest.samples)
                raise DatasetError(
                    "VoiceGuard training is not ready. Complete the current guided "
                    f"{self.samples_per_session}-clip session first "
                    f"({captured}/{self.samples_per_session} clips captured); "
                    "partial sessions remain unregistered."
                )
            if cached is not None:
                # A peer finalized, revoked, removed, or otherwise changed the
                # exact partial this consent was bound to. Never retain that
                # stale consent as authority to create a replacement session.
                self._consented.pop(user_id, None)
            self._pending_sessions.pop(user_id, None)
            if unfinished:
                if len(unfinished) == 1:
                    captured = len(unfinished[0].manifest.samples)
                    raise DatasetError(
                        "VoiceGuard training is not ready. An unfinished consented "
                        f"collection has {captured}/{self.samples_per_session} clips; "
                        "confirm recording consent again to resume it."
                    )
                raise DatasetError(
                    "VoiceGuard training is not ready. Multiple unfinished consented "
                    "collections require explicit recovery before training."
                )
            readiness = self.manager.assess_training_readiness()
            if not readiness.ready:
                details = "; ".join(issue.message for issue in readiness.issues)
                raise DatasetError(
                    "VoiceGuard training is not ready. Complete guided multi-clip "
                    "sessions for every authorized and unauthorized speaker, then "
                    "run the readiness check again. "
                    f"Finalized clips={readiness.sample_count}, "
                    f"sessions={readiness.session_count}. {details}"
                )
        result = self.manager.retrain()
        return {
            "model_version": result.model.metadata.get("model_version"),
            "measured_test": result.measured_test,
            "test_samples": len(result.split.test),
            "finalized_clips": readiness.sample_count,
            "finalized_sessions": readiness.session_count,
        }

    def revoke(self, user_id: str) -> None:
        with self._lock, self.manager._lifecycle_transaction():
            pending = self._pending_sessions.pop(user_id, None)
            pending_error: Exception | None = None
            registered = any(
                record.user_id == user_id
                for record in self.manager.list_users(include_revoked=True)
            )
            try:
                discovered = self._unregistered_sessions(user_id)
            except Exception as error:
                pending_error = error
                discovered = ()
            try:
                if registered:
                    self.manager.revoke_user(user_id, delete_recordings=False)
            except Exception as error:  # registry is invalidated before manifest cleanup
                if pending_error is None:
                    pending_error = error
            partials = {
                item.manifest.session_id: item
                for item in ((*discovered, pending) if pending is not None else discovered)
            }
            for partial in partials.values():
                try:
                    self.manager.revoke_unregistered_session(partial)
                except Exception as error:  # pragma: no cover - rare storage failure
                    if pending_error is None:
                        pending_error = error
            self._consented.pop(user_id, None)
            if pending_error is not None:
                raise pending_error


__all__ = ["VoiceGuardDesktopAdapter"]
