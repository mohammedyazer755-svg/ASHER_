"""Enrollment, revocation, and retraining lifecycle for VoiceGuard users."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .dataset import VoiceDataset, load_dataset
from .exceptions import DatasetError, EnrollmentError, ManifestError
from .model import model_content_fingerprint
from .readiness import ReadinessPolicy, TrainingReadiness, assess_dataset_readiness
from .recording import RecordingSession, load_manifest
from .schema import SessionManifest, SpeakerRole, TrainingTask
from .training import TrainingArtifacts, TrainingConfig, TrainingResult, VoiceGuardTrainer


REGISTRY_FILENAME = "enrollment_registry.json"
LIFECYCLE_LOCK_FILENAME = ".voiceguard-lifecycle.lock"
_PRESERVE_REGISTRY_VALUE = object()
_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[Path, threading.RLock] = {}
_TRANSACTION_STATE = threading.local()


def _root_lock(root: Path) -> threading.RLock:
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(root, threading.RLock())


@contextmanager
def _filesystem_lock(path: Path):
    """Serialize lifecycle mutations across manager instances and processes."""

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
                    # ``LK_LOCK`` gives up after a short fixed retry window,
                    # which is shorter than real model training. Keep consent
                    # revocation/finalization serialized for the full fit.
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


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_registry(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class EnrollmentRecord:
    user_id: str
    role: str
    enrolled_at: str
    session_ids: tuple[str, ...]
    revoked_at: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "enrolled_at": self.enrolled_at,
            "session_ids": list(self.session_ids),
            "revoked_at": self.revoked_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnrollmentRecord":
        try:
            role = SpeakerRole(str(value["role"]))
            sessions = tuple(str(item) for item in value.get("session_ids", []))
            return cls(
                user_id=str(value["user_id"]),
                role=role.value,
                enrolled_at=str(value["enrolled_at"]),
                session_ids=sessions,
                revoked_at=None if value.get("revoked_at") is None else str(value["revoked_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EnrollmentError("enrollment registry contains an invalid user record") from exc


@dataclass(frozen=True)
class RevocationResult:
    user_id: str
    session_ids: tuple[str, ...]
    recordings_deleted: bool
    revoked_at: str


class EnrollmentManager:
    """Own the lifecycle of consented sessions and active user records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.recordings_root = self.root / "recordings"
        self.models_root = self.root / "models"
        self.evaluations_root = self.root / "evaluations"
        self.registry_path = self.root / REGISTRY_FILENAME
        self._lock = _root_lock(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lifecycle_transaction(self):
        with self._lock:
            depths = getattr(_TRANSACTION_STATE, "depths", None)
            if depths is None:
                depths = {}
                _TRANSACTION_STATE.depths = depths
            depth = int(depths.get(self.root, 0))
            if depth:
                depths[self.root] = depth + 1
                try:
                    yield
                finally:
                    depths[self.root] -= 1
                return
            with _filesystem_lock(self.root / LIFECYCLE_LOCK_FILENAME):
                depths[self.root] = 1
                try:
                    yield
                finally:
                    depths.pop(self.root, None)

    def _read_registry(self) -> dict[str, EnrollmentRecord]:
        if not self.registry_path.exists():
            return {}
        try:
            with self.registry_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise EnrollmentError("could not read the local enrollment registry") from exc
        if not isinstance(value, Mapping):
            raise EnrollmentError("enrollment registry root must be an object")
        users = value.get("users", {})
        if not isinstance(users, Mapping):
            raise EnrollmentError("enrollment registry users must be an object")
        return {str(key): EnrollmentRecord.from_dict(item) for key, item in users.items()}

    def _registry_token(self) -> str:
        try:
            payload = self.registry_path.read_bytes()
        except FileNotFoundError:
            payload = b"<missing-registry>"
        return hashlib.sha256(payload).hexdigest()

    def _write_registry(
        self,
        users: Mapping[str, EnrollmentRecord],
        *,
        active_model: str | None | object = _PRESERVE_REGISTRY_VALUE,
        active_model_dataset_fingerprint: str | None | object = _PRESERVE_REGISTRY_VALUE,
        active_model_content_fingerprint: str | None | object = _PRESERVE_REGISTRY_VALUE,
        wake_word_model: str | None | object = _PRESERVE_REGISTRY_VALUE,
        wake_word_model_dataset_fingerprint: str | None | object = _PRESERVE_REGISTRY_VALUE,
        wake_word_model_content_fingerprint: str | None | object = _PRESERVE_REGISTRY_VALUE,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "updated_at": _timestamp(),
            "users": {key: value.to_dict() for key, value in users.items()},
        }
        previous: Mapping[str, Any] = {}
        if self.registry_path.exists():
            try:
                with self.registry_path.open("r", encoding="utf-8") as stream:
                    loaded = json.load(stream)
                if isinstance(loaded, Mapping):
                    previous = loaded
            except (OSError, json.JSONDecodeError):
                pass
        if active_model is None:
            if active_model_dataset_fingerprint is _PRESERVE_REGISTRY_VALUE:
                active_model_dataset_fingerprint = None
            if active_model_content_fingerprint is _PRESERVE_REGISTRY_VALUE:
                active_model_content_fingerprint = None
        if wake_word_model is None:
            if wake_word_model_dataset_fingerprint is _PRESERVE_REGISTRY_VALUE:
                wake_word_model_dataset_fingerprint = None
            if wake_word_model_content_fingerprint is _PRESERVE_REGISTRY_VALUE:
                wake_word_model_content_fingerprint = None
        for key, replacement in (
            ("active_model", active_model),
            ("active_model_dataset_fingerprint", active_model_dataset_fingerprint),
            ("active_model_content_fingerprint", active_model_content_fingerprint),
            ("wake_word_model", wake_word_model),
            ("wake_word_model_dataset_fingerprint", wake_word_model_dataset_fingerprint),
            ("wake_word_model_content_fingerprint", wake_word_model_content_fingerprint),
        ):
            value = (
                previous.get(key)
                if replacement is _PRESERVE_REGISTRY_VALUE
                else replacement
            )
            if isinstance(value, str) and value.strip():
                payload[key] = value
        previous_generation = previous.get("lifecycle_generation", 0)
        if (
            not isinstance(previous_generation, int)
            or isinstance(previous_generation, bool)
            or previous_generation < 0
        ):
            previous_generation = 0
        payload["lifecycle_generation"] = previous_generation + 1
        _write_registry(self.registry_path, payload)

    def advance_lifecycle_generation(self) -> None:
        """Invalidate stale collector consent without registering a partial session."""

        with self._lifecycle_transaction():
            self._write_registry(self._read_registry())

    def list_users(self, *, include_revoked: bool = False) -> tuple[EnrollmentRecord, ...]:
        users = self._read_registry()
        return tuple(
            sorted(
                (item for item in users.values() if include_revoked or item.active),
                key=lambda item: item.user_id,
            )
        )

    def begin_enrollment(
        self,
        user_id: str,
        *,
        role: str | SpeakerRole,
        environment: str,
        consent: bool,
    ) -> RecordingSession:
        """Start a new consented session; no raw audio is collected implicitly."""

        if not user_id.strip():
            raise EnrollmentError("user_id is required")
        try:
            selected_role = SpeakerRole(role)
        except ValueError as exc:
            raise EnrollmentError("role must be owner, trusted, or unknown") from exc
        with self._lifecycle_transaction():
            previous = self._read_registry().get(user_id)
            if previous and previous.active and previous.role != selected_role.value:
                raise EnrollmentError("an active identity cannot mix speaker roles across sessions")
            return RecordingSession.create(
                self.recordings_root,
                speaker_id=user_id,
                role=selected_role,
                environment=environment,
                consent=consent,
            )

    def finalize_enrollment(self, session: RecordingSession, *, minimum_samples: int = 1) -> EnrollmentRecord:
        """Register a completed session only after checking its persisted manifest."""

        if minimum_samples <= 0:
            raise EnrollmentError("minimum_samples must be positive")
        recordings_root = self.recordings_root.resolve()
        session_directory = session.directory.resolve()
        if session_directory.parent != recordings_root:
            raise EnrollmentError("enrollment session is outside this manager's recordings directory")
        with self._lifecycle_transaction():
            manifest = load_manifest(session_directory)
            if (
                manifest.session_id != session.manifest.session_id
                or manifest.session_id != session_directory.name
                or manifest.revoked
            ):
                raise EnrollmentError("session is missing, mismatched, or revoked")
            if len(manifest.samples) < minimum_samples:
                raise EnrollmentError("enrollment session does not contain enough recorded samples")
            try:
                manifest = RecordingSession.open(session_directory).finalize()
            except (ManifestError, OSError) as exc:
                raise EnrollmentError("could not seal the finalized recording session") from exc
            users = self._read_registry()
            for registered in users.values():
                if (
                    manifest.session_id in registered.session_ids
                    and registered.user_id != manifest.speaker_id
                ):
                    raise EnrollmentError("recording session is already registered to another identity")
            previous = users.get(manifest.speaker_id)
            if previous and previous.active and previous.role != manifest.role:
                raise EnrollmentError("an active identity cannot mix speaker roles across sessions")
            session_ids = list(previous.session_ids) if previous and previous.active else []
            if manifest.session_id not in session_ids:
                session_ids.append(manifest.session_id)
            record = EnrollmentRecord(
                user_id=manifest.speaker_id,
                role=manifest.role,
                enrolled_at=previous.enrolled_at if previous and previous.active else _timestamp(),
                session_ids=tuple(session_ids),
                revoked_at=None,
            )
            users[manifest.speaker_id] = record
            # Any finalized dataset mutation makes prior speaker and wake
            # artifacts stale until an explicit retrain completes.
            self._write_registry(users, active_model=None, wake_word_model=None)
            return record

    def revoke_user(self, user_id: str, *, delete_recordings: bool = False) -> RevocationResult:
        """Immediately disable a user; optional deletion is explicit and scoped."""

        with self._lifecycle_transaction():
            users = self._read_registry()
            record = users.get(user_id)
            if record is None:
                raise EnrollmentError("user is not enrolled")
            when = _timestamp()
            sessions = tuple(record.session_ids)
            users[user_id] = EnrollmentRecord(
                user_id=record.user_id,
                role=record.role,
                enrolled_at=record.enrolled_at,
                session_ids=sessions,
                revoked_at=when,
            )
            # Commit the fail-closed registry mutation first. Even if private
            # manifest cleanup later fails, no runtime may retain the user or a
            # model trained while that biometric enrollment was active.
            self._write_registry(users, active_model=None, wake_word_model=None)
            cleanup_error: Exception | None = None
            for session_id in sessions:
                directory = (self.recordings_root / session_id).resolve()
                if directory.parent != self.recordings_root.resolve():
                    cleanup_error = EnrollmentError(
                        "registry session path escaped recordings root"
                    )
                    continue
                if directory.is_dir():
                    try:
                        session = RecordingSession.open(directory)
                        session._revoke_finalized()
                    except (ManifestError, OSError) as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
            if delete_recordings and cleanup_error is None:
                for session_id in sessions:
                    directory = (self.recordings_root / session_id).resolve()
                    if directory.parent == self.recordings_root.resolve() and directory.is_dir():
                        shutil.rmtree(directory)
            if cleanup_error is not None:
                raise EnrollmentError(
                    "biometric access was revoked, but one private recording session needs cleanup"
                ) from cleanup_error
            return RevocationResult(
                user_id=user_id,
                session_ids=sessions,
                recordings_deleted=delete_recordings,
                revoked_at=when,
            )

    def revoke_unregistered_session(
        self,
        session: RecordingSession,
    ) -> SessionManifest:
        """Revoke a partial or sealed orphan only while it is registry-unbound."""

        recordings_root = self.recordings_root.resolve()
        session_directory = session.directory.resolve()
        if session_directory.parent != recordings_root:
            raise EnrollmentError("recording session is outside this manager's recordings directory")
        with self._lifecycle_transaction():
            manifest = load_manifest(session_directory)
            if (
                manifest.session_id != session_directory.name
                or manifest.session_id != session.manifest.session_id
            ):
                raise EnrollmentError("recording session is missing or mismatched")
            users = self._read_registry()
            if any(
                manifest.session_id in record.session_ids
                for record in users.values()
            ):
                raise EnrollmentError(
                    "registered recording sessions must be revoked through their enrollment"
                )
            # Commit consent invalidation before private-manifest cleanup. A
            # stale peer must never be able to replace this revoked partial if
            # the subsequent disk mutation fails.
            self._write_registry(users)
            try:
                return RecordingSession.open(session_directory)._revoke_finalized()
            except (ManifestError, OSError) as exc:
                raise EnrollmentError("could not revoke the unregistered recording session") from exc

    def _active_session_index(self) -> dict[str, EnrollmentRecord]:
        """Map finalized active sessions to their registry owner."""

        index: dict[str, EnrollmentRecord] = {}
        for record in self._read_registry().values():
            if not record.active:
                continue
            for session_id in record.session_ids:
                if (
                    not session_id
                    or session_id.strip() != session_id
                    or any(char in session_id for char in "/\\:\x00")
                    or Path(session_id).name != session_id
                ):
                    raise DatasetError("the enrollment registry contains an invalid session identifier")
                if session_id in index:
                    raise DatasetError("a finalized session appears more than once in the enrollment registry")
                index[session_id] = record
        return index

    def _load_training_dataset_locked(self) -> VoiceDataset:
        index = self._active_session_index()
        dataset = load_dataset(
            self.recordings_root,
            include_revoked=False,
            verify_checksums=True,
            session_ids=index,
        )
        if len(dataset.sessions) != len(index):
            raise DatasetError("one or more finalized recording sessions are missing or revoked")
        # Schema-v1 compatibility: registry membership was the original
        # completion marker. Persist the explicit immutable seal once, under
        # the lifecycle lock, before exposing such a session for training.
        legacy_unsealed = tuple(item for item in dataset.sessions if not item.finalized)
        if legacy_unsealed:
            try:
                for manifest in legacy_unsealed:
                    RecordingSession.open(
                        self.recordings_root / manifest.session_id
                    ).finalize()
            except (ManifestError, OSError) as exc:
                raise DatasetError("a finalized recording session could not be sealed") from exc
            dataset = load_dataset(
                self.recordings_root,
                include_revoked=False,
                verify_checksums=True,
                session_ids=index,
            )
        for manifest in dataset.sessions:
            registered = index.get(manifest.session_id)
            if (
                registered is None
                or registered.user_id != manifest.speaker_id
                or registered.role != manifest.role
                or not manifest.finalized
            ):
                raise DatasetError("a finalized recording manifest conflicts with its enrollment record")
        return dataset

    def load_training_dataset(self) -> VoiceDataset:
        """Load only sealed sessions owned by active enrollment records."""

        with self._lifecycle_transaction():
            return self._load_training_dataset_locked()

    def assess_training_readiness(
        self,
        *,
        config: TrainingConfig | None = None,
        policy: ReadinessPolicy | None = None,
    ) -> TrainingReadiness:
        """Return aggregate readiness without extracting audio features or loading ML."""

        selected = config or TrainingConfig()
        dataset = self.load_training_dataset()
        try:
            authorized_labels = selected.resolved_authorized_labels_for_dataset(dataset)
        except DatasetError:
            authorized_labels = ()
        return assess_dataset_readiness(
            dataset,
            task=selected.task,
            authorized_labels=authorized_labels,
            policy=policy,
            minimum_training_samples=selected.minimum_training_samples,
            seed=selected.seed,
            validation_fraction=selected.validation_fraction,
            test_fraction=selected.test_fraction,
        )

    def _private_model_path(self, model_path: str | Path | None, *, model_version: str) -> Path:
        """Resolve the immutable private model path used before registry activation.

        Overrides remain supported for API compatibility, but their basename
        must be exactly ``voiceguard-<model_version>.json``.  This prevents a
        failed retrain from overwriting the file referenced by an older active
        model before all reports have persisted.
        """

        models_root = self.models_root.resolve()
        expected_name = f"voiceguard-{model_version}.json"
        destination = (
            (models_root / expected_name)
            if model_path is None
            else Path(model_path).expanduser().resolve()
        )
        destination = destination.resolve()
        if destination == models_root or models_root not in destination.parents:
            raise EnrollmentError("active VoiceGuard models must remain inside the private models directory")
        if destination.name != expected_name:
            raise EnrollmentError(
                f"active VoiceGuard model filename must be exactly {expected_name}"
            )
        return destination

    def _persist_training_artifacts(
        self,
        result: TrainingResult,
        *,
        model_path: str | Path | None,
    ) -> TrainingArtifacts:
        version = result.model.model_version.strip()
        if (
            len(version) != 64
            or version == "unknown"
            or any(char not in "0123456789abcdef" for char in version.casefold())
        ):
            raise EnrollmentError("trained VoiceGuard model has no valid content version")
        if version != model_content_fingerprint(result.model):
            raise EnrollmentError("trained VoiceGuard model content version does not match its model")
        destination = self._private_model_path(model_path, model_version=version)
        evaluations_root = self.evaluations_root.resolve()
        validation_path = evaluations_root / f"voiceguard-{version}-validation.json"
        test_path = evaluations_root / f"voiceguard-{version}-test.json"
        targets = (destination, validation_path, test_path)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.evaluations_root.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in targets):
            raise EnrollmentError("a versioned VoiceGuard artifact already exists; refusing to overwrite it")
        created: list[Path] = []
        try:
            with tempfile.TemporaryDirectory(
                prefix=".voiceguard-artifact-stage-",
                dir=self.root,
            ) as temporary:
                staging = Path(temporary)
                staged_model = result.model.save(staging / "model.json")
                staged_validation = result.validation_report.save(
                    staging / "validation.json"
                )
                staged_test = result.test_report.save(staging / "test.json")
                for source, target in zip(
                    (staged_model, staged_validation, staged_test),
                    targets,
                ):
                    os.link(source, target)
                    created.append(target)
        except Exception as exc:
            for path in created:
                path.unlink(missing_ok=True)
            raise EnrollmentError("could not persist the private VoiceGuard training artifacts") from exc
        return TrainingArtifacts(
            model_path=destination,
            validation_report_path=validation_path,
            test_report_path=test_path,
        )

    def retrain(
        self,
        *,
        trainer: VoiceGuardTrainer | None = None,
        config: TrainingConfig | None = None,
        model_path: str | Path | None = None,
    ) -> TrainingResult:
        """Retrain exclusively from currently non-revoked session manifests."""

        # Snapshot and validate under the lifecycle lock, but never hold the
        # revocation boundary across feature extraction or an arbitrary
        # trainer. Activation later uses a compare-and-swap revalidation.
        with self._lifecycle_transaction():
            dataset = self.load_training_dataset()
            selected = config or TrainingConfig()
            try:
                authorized_labels = selected.resolved_authorized_labels_for_dataset(dataset)
            except DatasetError:
                authorized_labels = ()
            readiness = assess_dataset_readiness(
                dataset,
                task=selected.task,
                authorized_labels=authorized_labels,
                minimum_training_samples=selected.minimum_training_samples,
                seed=selected.seed,
                validation_fraction=selected.validation_fraction,
                test_fraction=selected.test_fraction,
            )
            readiness.require_ready()
            from .readiness import independent_dataset_view

            training_dataset = independent_dataset_view(
                dataset,
                task=selected.task,
                separation_seconds=ReadinessPolicy().minimum_session_separation_seconds,
            )
            snapshot_fingerprint = dataset.fingerprint
            snapshot_registry_token = self._registry_token()

        selected_trainer = trainer or VoiceGuardTrainer()
        result = selected_trainer.train_dataset(training_dataset, selected)
        try:
            trained_task = TrainingTask(result.model.task)
        except ValueError as exc:
            raise EnrollmentError("trainer returned a model with an unsupported task") from exc
        if trained_task is not TrainingTask(selected.task):
            raise EnrollmentError("trainer returned a model for a different task")
        if set(result.model.authorized_labels) != set(authorized_labels):
            raise EnrollmentError(
                "trainer returned a model with different authorized classes"
            )
        model_dataset_metadata = result.model.metadata.get("dataset", {})
        trained_dataset_fingerprint = (
            model_dataset_metadata.get("fingerprint")
            if isinstance(model_dataset_metadata, Mapping)
            else None
        )
        if trained_dataset_fingerprint != training_dataset.fingerprint:
            raise EnrollmentError(
                "trainer returned a model with missing or different dataset provenance"
            )

        with self._lifecycle_transaction():
            # Compare the complete finalized dataset, including sessions that
            # were conservatively excluded as correlated extras.
            current_dataset = self.load_training_dataset()
            if self._registry_token() != snapshot_registry_token:
                raise EnrollmentError("VoiceGuard enrollment registry changed during training; retry safely")
            if current_dataset.fingerprint != snapshot_fingerprint:
                raise EnrollmentError("VoiceGuard enrollment changed during training; retry safely")
            try:
                current_authorized = selected.resolved_authorized_labels_for_dataset(
                    current_dataset
                )
            except DatasetError:
                current_authorized = ()
            if set(current_authorized) != set(authorized_labels):
                raise EnrollmentError("VoiceGuard authorization changed during training; retry safely")

            artifacts = self._persist_training_artifacts(result, model_path=model_path)
            path = str(artifacts.model_path.resolve())
            try:
                activation_dataset = self.load_training_dataset()
                if activation_dataset.fingerprint != snapshot_fingerprint:
                    raise EnrollmentError(
                        "VoiceGuard enrollment changed before activation; retry safely"
                    )
                if trained_task is TrainingTask.SPEAKER_AUTH:
                    self._write_registry(
                        self._read_registry(),
                        active_model=path,
                        active_model_dataset_fingerprint=snapshot_fingerprint,
                        active_model_content_fingerprint=result.model.model_version,
                    )
                else:
                    # Activated separately: the runtime uses this only as the
                    # standby wake gate, never as a speaker-identity model.
                    self._write_registry(
                        self._read_registry(),
                        wake_word_model=path,
                        wake_word_model_dataset_fingerprint=snapshot_fingerprint,
                        wake_word_model_content_fingerprint=result.model.model_version,
                    )
            except Exception as exc:
                for artifact_path in (
                    artifacts.model_path,
                    artifacts.validation_report_path,
                    artifacts.test_report_path,
                ):
                    artifact_path.unlink(missing_ok=True)
                if isinstance(exc, EnrollmentError):
                    raise
                raise EnrollmentError(
                    "could not activate the private VoiceGuard training artifacts"
                ) from exc
            return replace(result, artifacts=artifacts)
