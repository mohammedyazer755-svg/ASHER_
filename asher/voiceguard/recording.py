"""Consent-aware VoiceGuard recording sessions and manifests."""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping
from uuid import uuid4

from .audio import PcmAudio, read_wav, sha256_file, write_wav
from .exceptions import ManifestError, RecordingUnavailableError
from .schema import (
    SampleCondition,
    SampleOrigin,
    SampleRecord,
    SessionManifest,
    SpeakerRole,
    utc_timestamp,
)


MANIFEST_FILENAME = "manifest.json"
SESSION_LOCK_FILENAME = ".recording-session.lock"
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[Path, threading.RLock] = {}
_SESSION_TRANSACTION_STATE = threading.local()


def _shared_session_lock(directory: Path) -> threading.RLock:
    with _SESSION_LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(directory, threading.RLock())


@contextmanager
def _session_filesystem_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {11, 13} and getattr(exc, "winerror", None) not in {33, 36}:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def load_manifest(path: str | Path) -> SessionManifest:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not load recording manifest: {manifest_path.name}") from exc
    if not isinstance(value, dict):
        raise ManifestError("recording manifest root must be a JSON object")
    return SessionManifest.from_dict(value)


class RecordingSession:
    """A mutable façade over an atomically persisted immutable manifest."""

    def __init__(self, directory: str | Path, manifest: SessionManifest) -> None:
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / MANIFEST_FILENAME
        self._manifest = manifest
        self._lock = threading.RLock()
        self._shared_lock = _shared_session_lock(self.directory)

    @property
    def manifest(self) -> SessionManifest:
        return self._manifest

    @classmethod
    def create(
        cls,
        recordings_root: str | Path,
        *,
        speaker_id: str,
        role: str | SpeakerRole,
        environment: str,
        consent: bool,
        session_id: str | None = None,
    ) -> "RecordingSession":
        """Begin a session only after explicit consent to retain raw voice audio."""

        if not consent:
            raise ManifestError("explicit recording consent is required before creating a session")
        identifier = session_id or uuid4().hex
        manifest = SessionManifest(
            session_id=identifier,
            speaker_id=speaker_id,
            role=role.value if isinstance(role, SpeakerRole) else str(role),
            environment=environment,
            consent_given_at=utc_timestamp(),
        )
        root = Path(recordings_root).resolve()
        directory = (root / identifier).resolve()
        if directory.parent != root:
            raise ManifestError("session path must stay inside the recordings directory")
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ManifestError("recording session already exists") from exc
        session = cls(directory, manifest)
        session._persist()
        return session

    @classmethod
    def open(cls, directory: str | Path) -> "RecordingSession":
        location = Path(directory).resolve()
        manifest = load_manifest(location)
        if location.name != manifest.session_id:
            raise ManifestError("session directory and manifest identifier do not match")
        return cls(location, manifest)

    def _persist(self) -> None:
        _atomic_json(self.manifest_path, self._manifest.to_dict())

    def _refresh(self) -> SessionManifest:
        """Refresh mutable state so a stale facade cannot undo disk changes."""

        current = load_manifest(self.manifest_path)
        if current.session_id != self.directory.name:
            raise ManifestError("session directory and manifest identifier do not match")
        self._manifest = current
        return current

    @contextmanager
    def _mutation_transaction(self):
        """Serialize manifest mutations across facades, threads, and processes."""

        with self._shared_lock:
            depths = getattr(_SESSION_TRANSACTION_STATE, "depths", None)
            if depths is None:
                depths = {}
                _SESSION_TRANSACTION_STATE.depths = depths
            depth = int(depths.get(self.directory, 0))
            if depth:
                depths[self.directory] = depth + 1
                try:
                    yield
                finally:
                    depths[self.directory] -= 1
                return
            with _session_filesystem_lock(self.directory / SESSION_LOCK_FILENAME):
                depths[self.directory] = 1
                try:
                    yield
                finally:
                    depths.pop(self.directory, None)

    def add_audio(
        self,
        audio: PcmAudio,
        *,
        contains_wake_phrase: bool,
        condition: str | SampleCondition = SampleCondition.CLEAN,
        origin: str | SampleOrigin = SampleOrigin.RECORDED,
        expected_authorized: bool | None = None,
        source_sample_id: str | None = None,
        sample_id: str | None = None,
    ) -> SampleRecord:
        """Persist PCM audio and add its measured metadata to the session manifest."""

        with self._lock, self._mutation_transaction():
            current = self._refresh()
            if current.revoked:
                raise ManifestError("cannot add audio to a revoked recording session")
            if current.finalized:
                raise ManifestError("cannot add audio to a finalized recording session")
            identifier = sample_id or uuid4().hex
            relative_path = f"clips/{identifier}.wav"
            destination = self.directory / Path(relative_path)
            if destination.exists():
                raise ManifestError("sample already exists in this recording session")
            write_wav(destination, audio)
            try:
                sample = SampleRecord(
                    sample_id=identifier,
                    path=relative_path,
                    sha256=sha256_file(destination),
                    duration_seconds=audio.duration_seconds,
                    sample_rate=audio.sample_rate,
                    channels=audio.channels,
                    contains_wake_phrase=contains_wake_phrase,
                    condition=(condition.value if isinstance(condition, SampleCondition) else str(condition)),
                    origin=origin.value if isinstance(origin, SampleOrigin) else str(origin),
                    expected_authorized=expected_authorized,
                    source_sample_id=source_sample_id,
                )
                updated = self._manifest.add_sample(sample)
                previous = self._manifest
                self._manifest = updated
                try:
                    self._persist()
                except OSError:
                    self._manifest = previous
                    raise
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            return sample

    def add_pcm16(
        self,
        samples: Iterable[int],
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        contains_wake_phrase: bool,
        condition: str | SampleCondition = SampleCondition.CLEAN,
        origin: str | SampleOrigin = SampleOrigin.RECORDED,
        expected_authorized: bool | None = None,
        source_sample_id: str | None = None,
        sample_id: str | None = None,
    ) -> SampleRecord:
        audio = PcmAudio(tuple(int(value) for value in samples), sample_rate, channels)
        return self.add_audio(
            audio,
            contains_wake_phrase=contains_wake_phrase,
            condition=condition,
            origin=origin,
            expected_authorized=expected_authorized,
            source_sample_id=source_sample_id,
            sample_id=sample_id,
        )

    def import_wav(
        self,
        source: str | Path,
        *,
        contains_wake_phrase: bool,
        condition: str | SampleCondition = SampleCondition.CLEAN,
        expected_authorized: bool | None = None,
        sample_id: str | None = None,
    ) -> SampleRecord:
        """Validate and copy a WAV into this session instead of referencing external data."""

        return self.add_audio(
            read_wav(source),
            contains_wake_phrase=contains_wake_phrase,
            condition=condition,
            origin=SampleOrigin.IMPORTED,
            expected_authorized=expected_authorized,
            sample_id=sample_id,
        )

    def import_replay_wav(
        self,
        source: str | Path,
        *,
        contains_wake_phrase: bool = True,
        sample_id: str | None = None,
    ) -> SampleRecord:
        """Add a real replay-capture trial, explicitly expected to be rejected."""

        return self.import_wav(
            source,
            contains_wake_phrase=contains_wake_phrase,
            condition=SampleCondition.REPLAY,
            expected_authorized=False,
            sample_id=sample_id,
        )

    def record_microphone(
        self,
        duration_seconds: float,
        *,
        contains_wake_phrase: bool,
        sample_rate: int = 16_000,
        channels: int = 1,
        device: int | str | None = None,
        condition: str | SampleCondition = SampleCondition.CLEAN,
        expected_authorized: bool | None = None,
        sample_id: str | None = None,
    ) -> SampleRecord:
        """Capture microphone input through optional ``sounddevice`` at call time only."""

        if duration_seconds <= 0 or duration_seconds > 120:
            raise RecordingUnavailableError("recording duration must be between 0 and 120 seconds")
        try:
            sounddevice = importlib.import_module("sounddevice")
        except (ImportError, OSError) as exc:
            raise RecordingUnavailableError(
                "live recording requires the optional 'sounddevice' package and a working input device"
            ) from exc
        frame_count = max(1, int(round(duration_seconds * sample_rate)))
        try:
            capture = sounddevice.rec(
                frame_count,
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device,
                blocking=False,
            )
            sounddevice.wait()
            samples = tuple(int(value) for row in capture.tolist() for value in (row if isinstance(row, list) else [row]))
        except Exception as exc:
            raise RecordingUnavailableError("microphone capture failed; no sample was enrolled") from exc
        return self.add_pcm16(
            samples,
            sample_rate=sample_rate,
            channels=channels,
            contains_wake_phrase=contains_wake_phrase,
            condition=condition,
            expected_authorized=expected_authorized,
            sample_id=sample_id,
        )

    def remove_sample(self, sample_id: str) -> bool:
        """Remove one manifest-owned clip; returns False when it was not present."""

        with self._lock, self._mutation_transaction():
            self._refresh()
            if self._manifest.finalized:
                raise ManifestError("cannot remove audio from a finalized recording session")
            match = next((item for item in self._manifest.samples if item.sample_id == sample_id), None)
            if match is None:
                return False
            source = self._manifest.resolve_sample_path(self.directory, match)
            previous = self._manifest
            self._manifest = self._manifest.without_sample(sample_id)
            try:
                self._persist()
            except OSError:
                self._manifest = previous
                raise
            source.unlink(missing_ok=True)
            return True

    def _revoke(self, *, allow_finalized: bool) -> SessionManifest:
        with self._lock, self._mutation_transaction():
            self._refresh()
            if self._manifest.finalized and not allow_finalized:
                raise ManifestError(
                    "finalized recording sessions must be revoked through EnrollmentManager"
                )
            if not self._manifest.revoked:
                self._manifest = self._manifest.revoke()
                self._persist()
            return self._manifest

    def revoke(self) -> SessionManifest:
        """Reject lifecycle-bypassing revocation.

        Even an unregistered partial must be revoked through EnrollmentManager
        so the shared lifecycle generation invalidates consent held by peer
        desktop instances and processes.
        """

        raise ManifestError(
            "recording sessions must be revoked through EnrollmentManager"
        )

    def _revoke_finalized(self) -> SessionManifest:
        """Manager-only finalized revocation under the lifecycle lock."""

        return self._revoke(allow_finalized=True)

    def finalize(self) -> SessionManifest:
        """Persist an immutable completion seal; idempotent for safe retries."""

        with self._lock, self._mutation_transaction():
            self._refresh()
            self._manifest = self._manifest.finalize()
            self._persist()
            return self._manifest
