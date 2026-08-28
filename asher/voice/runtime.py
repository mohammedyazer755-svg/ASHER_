"""Explicit voice runtime orchestration; importing it never opens hardware."""

from __future__ import annotations

import io
import array
import json
import math
import os
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from asher.agent.controller import CompanionController, CompanionReply
from asher.config import AsherConfig
from asher.core.cancellation import CancellationToken, CancelledError
from asher.core.state import AssistantState, StateStore
from asher.voice.capture import AudioFrame, TurnCapture, VadConfig, VoiceActivityDetector
from asher.voice.pipeline import PipelineStatus, VoiceAccuracyPipeline
from asher.voice.transcription import FasterWhisperTranscriber, TranscriptionConfig
from asher.voice.types import TranscriptResult
from asher.voice.vocabulary import DynamicVocabulary
from asher.voice.wakeword import EnergyGate, TextWakeDetector


# VOICE-2C keeps wake and command capture as distinct acoustic turns.  The
# command endpoint is intentionally more tolerant than the standby/wake turn so
# ordinary sentence pauses do not split one request into multiple commands.
WAKE_TURN_VAD = VadConfig()

COMMAND_TURN_VAD = VadConfig(
    end_silence_ms=900,
    pre_roll_ms=240,
    max_turn_ms=15_000,
)


SLEEP_COMMANDS = frozenset(
    {
        "sleep",
        "go to sleep",
        "standby",
        "go to standby",
        "that's all",
        "thatâ€™s all",
        "nothing else",
    }
)


def normalized_pcm16_rms(pcm16: bytes | bytearray | memoryview) -> float:
    """Return bounded RMS for little-endian signed PCM16 without retaining it."""

    try:
        raw = memoryview(pcm16).cast("B")
        usable_bytes = raw.nbytes - (raw.nbytes % 2)
        if usable_bytes <= 0:
            return 0.0
        samples = array.array("h")
        samples.frombytes(raw[:usable_bytes])
        if sys.byteorder != "little":
            samples.byteswap()
    except (TypeError, ValueError, BufferError):
        return 0.0
    if not samples:
        return 0.0
    mean_square = sum(int(sample) * int(sample) for sample in samples) / len(samples)
    return max(0.0, min(1.0, math.sqrt(mean_square) / 32768.0))


class AudioBackend(Protocol):
    def frames(self, cancellation: CancellationToken | None = None): ...


class SoundDeviceBackend:
    """16 kHz mono PCM stream with no recording retained beyond the turn."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        block_samples: int = 320,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self.device = device

    def frames(self, cancellation: CancellationToken | None = None):
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("sounddevice is required for voice mode") from error
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_samples,
            channels=1,
            dtype="int16",
            device=self.device,
        ) as stream:
            self._active_stream = stream
            try:
                while cancellation is None or not cancellation.cancelled:
                    data, _overflowed = stream.read(self.block_samples)
                    yield AudioFrame(bytes(data), self.sample_rate)
            finally:
                self._active_stream = None

    def flush(self) -> None:
        stream = getattr(self, "_active_stream", None)
        if stream is not None:
            try:
                while stream.read_available > 0:
                    frames_to_read = min(stream.read_available, 1024)
                    if frames_to_read > 0:
                        stream.read(frames_to_read)
            except Exception:
                pass


class CloudTranscriber:
    """Optional low-confidence fallback using the official audio transcription endpoint."""

    def __init__(self, model: str = "gpt-4o-transcribe", client: Any | None = None) -> None:
        self.model = model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # type: ignore[import-not-found]

            if not os.getenv("OPENAI_API_KEY", "").strip():
                raise RuntimeError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        return self._client

    def transcribe(self, audio: Any, *, vocabulary=(), cancellation=None, **kwargs: Any) -> TranscriptResult:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if hasattr(audio, "pcm16"):
            pcm = bytes(audio.pcm16)
        elif isinstance(audio, (str, os.PathLike)):
            try:
                with wave.open(str(audio), "rb") as reader:
                    pcm = reader.readframes(reader.getnframes())
            except (OSError, wave.Error) as error:
                raise RuntimeError("The temporary voice turn could not be read") from error
        else:
            pcm = bytes(audio)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(pcm)
        buffer.seek(0)
        buffer.name = "asher-low-confidence.wav"  # SDK uses this for multipart MIME.
        prompt = ", ".join(str(item) for item in vocabulary if str(item).strip())
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=buffer,
            language="en",
            prompt=prompt or None,
        )
        text = str(getattr(response, "text", response)).strip()
        from asher.voice.transcription import normalize_transcript_surface

        return TranscriptResult(
            raw_text=text,
            normalized_text=normalize_transcript_surface(text),
            acoustic_confidence=0.76,
            no_speech_probability=0.0,
            provider="openai-transcription",
            model=self.model,
            device="remote",
            metadata={"remote_fallback": True},
        )


class VoiceGuardVerifier(Protocol):
    def authenticate(self, pcm16: bytes, sample_rate: int) -> tuple[str | None, float, str]: ...


class WakeWordVerifier(Protocol):
    def verify(self, pcm16: bytes, sample_rate: int) -> tuple[bool, float, str]: ...


class FileVoiceGuardVerifier:
    """Inference adapter that deletes temporary audio immediately."""

    def __init__(
        self,
        model_path: str | Path,
        label_to_user: dict[str, str],
        *,
        extractor: Any,
        temp_root: str | Path | None = None,
        registry_path: str | Path | None = None,
        required_enrollments: set[str] | None = None,
        expected_dataset_fingerprint: str | None = None,
        expected_model_content_fingerprint: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.label_to_user = dict(label_to_user)
        self.extractor = extractor
        self.temp_root = Path(temp_root) if temp_root else None
        self.registry_path = Path(registry_path).resolve() if registry_path else None
        self.required_enrollments = frozenset(required_enrollments or ())
        self.expected_dataset_fingerprint = expected_dataset_fingerprint
        self.expected_model_content_fingerprint = expected_model_content_fingerprint
        self._model: Any | None = None

    def _binding_is_current(self) -> bool:
        if self.registry_path is None:
            return True
        try:
            with self.registry_path.open("r", encoding="utf-8") as stream:
                registry = json.load(stream)
            if not isinstance(registry, Mapping):
                return False
            active_model = registry.get("active_model")
            if not isinstance(active_model, str) or Path(active_model).expanduser().resolve() != self.model_path:
                return False
            dataset_fingerprint = registry.get("active_model_dataset_fingerprint")
            content_fingerprint = registry.get("active_model_content_fingerprint")
            if (
                not self.expected_dataset_fingerprint
                or dataset_fingerprint != self.expected_dataset_fingerprint
                or not self.expected_model_content_fingerprint
                or content_fingerprint != self.expected_model_content_fingerprint
                or not self.model_path.is_file()
            ):
                return False
            users = registry.get("users")
            if not isinstance(users, Mapping):
                return False
            active_enrollments = {
                str(key)
                for key, value in users.items()
                if isinstance(value, Mapping)
                and value.get("user_id") == str(key)
                and value.get("role") in {"owner", "trusted"}
                and value.get("revoked_at") is None
            }
            if not self.required_enrollments.issubset(active_enrollments):
                return False

            from asher.voiceguard.enrollment import EnrollmentManager
            from asher.voiceguard.model import (
                CalibratedVoiceGuardModel,
                model_content_fingerprint,
            )

            current_model = CalibratedVoiceGuardModel.load(self.model_path)
            if (
                current_model.model_version != self.expected_model_content_fingerprint
                or model_content_fingerprint(current_model)
                != self.expected_model_content_fingerprint
            ):
                return False
            current_dataset = EnrollmentManager(
                self.registry_path.parent
            ).load_training_dataset()
            return current_dataset.fingerprint == self.expected_dataset_fingerprint
        except Exception:
            return False

    def _model_matches_expected_content(self, model: Any) -> bool:
        """Validate the exact in-memory model that will perform inference."""

        if self.expected_model_content_fingerprint is None:
            return True
        try:
            from asher.voiceguard.model import (
                CalibratedVoiceGuardModel,
                model_content_fingerprint,
            )

            return (
                isinstance(model, CalibratedVoiceGuardModel)
                and model.task == "speaker_auth"
                and model.model_version == self.expected_model_content_fingerprint
                and model_content_fingerprint(model)
                == self.expected_model_content_fingerprint
            )
        except Exception:
            return False

    def authenticate(self, pcm16: bytes, sample_rate: int) -> tuple[str | None, float, str]:
        from asher.voiceguard.model import CalibratedVoiceGuardModel

        if not self._binding_is_current():
            return None, 0.0, "VoiceGuard enrollment or active model changed; guest access only"
        candidate = self._model
        if candidate is None:
            try:
                candidate = CalibratedVoiceGuardModel.load(self.model_path)
            except Exception:
                return None, 0.0, "VoiceGuard enrollment or active model changed; guest access only"
        if not self._model_matches_expected_content(candidate):
            self._model = None
            return None, 0.0, "VoiceGuard enrollment or active model changed; guest access only"
        self._model = candidate
        self.temp_root and self.temp_root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_root)
        path = Path(handle.name)
        handle.close()
        try:
            _write_wav(path, pcm16, sample_rate)
            result = candidate.verify_wav(path, self.extractor)
            if (
                not self._model_matches_expected_content(candidate)
                or not self._binding_is_current()
            ):
                return None, 0.0, "VoiceGuard enrollment or active model changed; guest access only"
            user_id = self.label_to_user.get(result.predicted_label) if result.accepted else None
            return user_id, result.score, result.reason
        finally:
            path.unlink(missing_ok=True)


def load_active_voiceguard_verifier(controller: CompanionController) -> FileVoiceGuardVerifier | None:
    """Load only a current identity-labelled speaker model, fail-closed.

    Old role-labelled models are deliberately rejected: a role such as
    ``trusted`` is not enough to identify which local user should receive a
    session. The model's authorized labels must be active user IDs in the
    current SQLite user store.
    """

    root = (controller.config.runtime.root / "voiceguard").resolve()
    registry_path = root / "enrollment_registry.json"
    try:
        with registry_path.open("r", encoding="utf-8") as stream:
            registry = json.load(stream)
        model_value = registry.get("active_model") if isinstance(registry, dict) else None
        if not isinstance(model_value, str) or not model_value.strip():
            return None
        dataset_fingerprint = registry.get("active_model_dataset_fingerprint")
        content_fingerprint = registry.get("active_model_content_fingerprint")
        if (
            not isinstance(dataset_fingerprint, str)
            or len(dataset_fingerprint) != 64
            or not isinstance(content_fingerprint, str)
            or len(content_fingerprint) != 64
        ):
            return None
        model_path = Path(model_value).expanduser().resolve()
        models_root = (root / "models").resolve()
        if model_path != models_root and models_root not in model_path.parents:
            return None
        if not model_path.is_file():
            return None

        from asher.voiceguard.features import StatisticalFeatureExtractor
        from asher.voiceguard.enrollment import EnrollmentManager
        from asher.voiceguard.model import (
            CalibratedVoiceGuardModel,
            model_content_fingerprint,
        )

        model = CalibratedVoiceGuardModel.load(model_path)
        if (
            model.task != "speaker_auth"
            or model.model_version != content_fingerprint
            or model_content_fingerprint(model) != content_fingerprint
        ):
            return None
        dataset = EnrollmentManager(root).load_training_dataset()
        if dataset.fingerprint != dataset_fingerprint:
            return None
        registry_users = registry.get("users")
        if not isinstance(registry_users, Mapping):
            return None
        active_voice_enrollments: set[str] = set()
        for registry_key, value in registry_users.items():
            if not isinstance(value, Mapping):
                return None
            user_id = value.get("user_id")
            role = value.get("role")
            session_ids = value.get("session_ids")
            if (
                not isinstance(user_id, str)
                or user_id != str(registry_key)
                or role not in {"owner", "trusted", "unknown"}
                or not isinstance(session_ids, list)
                or any(not isinstance(item, str) or not item for item in session_ids)
            ):
                return None
            if value.get("revoked_at") is None and role in {"owner", "trusted"}:
                active_voice_enrollments.add(user_id)
        active = {
            actor.user_id: actor
            for actor in controller.users.list_active()
            if actor.role.value in {"owner", "trusted"}
        }
        authorized = set(model.authorized_labels)
        if (
            not authorized
            or not authorized.issubset(active)
            or not authorized.issubset(active_voice_enrollments)
        ):
            return None
        extractor = StatisticalFeatureExtractor()
        if extractor.metadata.extractor_id != model.extractor_metadata.extractor_id:
            return None
        return FileVoiceGuardVerifier(
            model_path,
            {label: label for label in model.classes if label in active},
            extractor=extractor,
            temp_root=controller.config.runtime.root / "voiceguard" / "tmp",
            registry_path=registry_path,
            required_enrollments=authorized,
            expected_dataset_fingerprint=dataset_fingerprint,
            expected_model_content_fingerprint=content_fingerprint,
        )
    except Exception:
        # A missing/corrupt/stale model must degrade to guest access, never to
        # an optimistic owner session.
        return None


_WAKE_WORD_REGISTRY_KEYS = frozenset(
    {
        "wake_word_model",
        "wake_word_model_dataset_fingerprint",
        "wake_word_model_content_fingerprint",
    }
)
_WAKE_WORD_CLASSES = frozenset({"wake_negative", "wake_positive"})
_WAKE_WORD_AUTHORIZED_LABELS = frozenset({"wake_positive"})
_WAKE_BINDING_REJECTED = (
    "The active wake-word model is unavailable, stale, or no longer bound to the current dataset"
)


def _registry_has_no_active_wake_artifact(registry_path: Path) -> bool:
    """Prove that transcript-only fallback is still allowed at decision time."""

    try:
        with registry_path.open("r", encoding="utf-8") as stream:
            registry = json.load(stream)
    except FileNotFoundError:
        return True
    except Exception:
        return False
    return isinstance(registry, Mapping) and not any(
        key in registry for key in _WAKE_WORD_REGISTRY_KEYS
    )


@dataclass(frozen=True)
class WakeWordModelBinding:
    """A registry-backed standby gate with an explicit no-model fallback.

    Transcript-only matching is allowed only when the registry proves that no
    wake artifact is active.  A partial, corrupt, stale, or otherwise invalid
    active binding is represented by ``active_artifact=True`` with no verifier
    and always rejects activation.
    """

    active_artifact: bool
    verifier: WakeWordVerifier | None = None
    registry_path: Path | None = None
    reason: str = _WAKE_BINDING_REJECTED

    def verify(self, pcm16: bytes, sample_rate: int) -> tuple[bool, float | None, str]:
        if self.active_artifact:
            if self.verifier is None:
                return False, 0.0, self.reason
            try:
                return self.verifier.verify(pcm16, sample_rate)
            except Exception:
                return False, 0.0, _WAKE_BINDING_REJECTED
        if self.registry_path is not None and not _registry_has_no_active_wake_artifact(
            self.registry_path
        ):
            return False, 0.0, _WAKE_BINDING_REJECTED
        return True, None, "No active trained wake-word artifact; transcript boundary matched"


class FileWakeWordVerifier:
    """Inference adapter for a content- and dataset-bound wake-word model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        extractor: Any,
        temp_root: str | Path | None = None,
        registry_path: str | Path | None = None,
        expected_dataset_fingerprint: str | None = None,
        expected_model_content_fingerprint: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.extractor = extractor
        self.temp_root = Path(temp_root) if temp_root else None
        self.registry_path = Path(registry_path).resolve() if registry_path else None
        self.expected_dataset_fingerprint = expected_dataset_fingerprint
        self.expected_model_content_fingerprint = expected_model_content_fingerprint
        self._model: Any | None = None

    def _model_matches_expected_content(self, model: Any) -> bool:
        try:
            from asher.voiceguard.model import (
                CalibratedVoiceGuardModel,
                model_content_fingerprint,
            )

            if not isinstance(model, CalibratedVoiceGuardModel):
                return False
            if (
                model.task != "wake_word"
                or set(model.classes) != _WAKE_WORD_CLASSES
                or set(model.authorized_labels) != _WAKE_WORD_AUTHORIZED_LABELS
                or model.extractor_metadata.extractor_id
                != self.extractor.metadata.extractor_id
            ):
                return False
            if (
                len(model.coefficients) == 1
                and model.binary_positive_class != "wake_positive"
            ):
                return False
            if self.expected_model_content_fingerprint is None:
                return True
            return (
                model.model_version == self.expected_model_content_fingerprint
                and model_content_fingerprint(model)
                == self.expected_model_content_fingerprint
            )
        except Exception:
            return False

    def _binding_is_current(self) -> bool:
        if self.registry_path is None:
            return True
        try:
            with self.registry_path.open("r", encoding="utf-8") as stream:
                registry = json.load(stream)
            if not isinstance(registry, Mapping):
                return False
            active_model = registry.get("wake_word_model")
            if (
                not isinstance(active_model, str)
                or Path(active_model).expanduser().resolve() != self.model_path
                or registry.get("wake_word_model_dataset_fingerprint")
                != self.expected_dataset_fingerprint
                or registry.get("wake_word_model_content_fingerprint")
                != self.expected_model_content_fingerprint
                or not self.expected_dataset_fingerprint
                or not self.expected_model_content_fingerprint
                or not self.model_path.is_file()
            ):
                return False

            from asher.voiceguard.enrollment import EnrollmentManager
            from asher.voiceguard.model import CalibratedVoiceGuardModel

            current_model = CalibratedVoiceGuardModel.load(self.model_path)
            if not self._model_matches_expected_content(current_model):
                return False
            current_dataset = EnrollmentManager(
                self.registry_path.parent
            ).load_training_dataset()
            return current_dataset.fingerprint == self.expected_dataset_fingerprint
        except Exception:
            return False

    def verify(self, pcm16: bytes, sample_rate: int) -> tuple[bool, float, str]:
        """Verify one transient turn, rejecting every lifecycle/model error."""

        from asher.voiceguard.model import CalibratedVoiceGuardModel

        if not self._binding_is_current():
            return False, 0.0, _WAKE_BINDING_REJECTED
        candidate = self._model
        if candidate is None:
            try:
                candidate = CalibratedVoiceGuardModel.load(self.model_path)
            except Exception:
                return False, 0.0, _WAKE_BINDING_REJECTED
        if not self._model_matches_expected_content(candidate):
            self._model = None
            return False, 0.0, _WAKE_BINDING_REJECTED
        self._model = candidate
        path: Path | None = None
        try:
            if self.temp_root is not None:
                self.temp_root.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir=self.temp_root,
            )
            path = Path(handle.name)
            handle.close()
            _write_wav(path, pcm16, sample_rate)
            result = candidate.verify_wav(path, self.extractor)
            if (
                not self._model_matches_expected_content(candidate)
                or not self._binding_is_current()
            ):
                return False, 0.0, _WAKE_BINDING_REJECTED
            return bool(result.accepted), float(result.score), str(result.reason)
        except Exception:
            self._model = None
            return False, 0.0, _WAKE_BINDING_REJECTED
        finally:
            if path is not None:
                path.unlink(missing_ok=True)


def load_active_wake_word_binding(controller: CompanionController) -> WakeWordModelBinding:
    """Resolve the active wake artifact independently from speaker identity.

    A well-formed registry with no wake keys permits the legacy text boundary.
    Once any wake binding key exists, every field, model byte, extractor, and
    finalized-dataset fingerprint must validate or standby activation is
    rejected.
    """

    root = (controller.config.runtime.root / "voiceguard").resolve()
    registry_path = root / "enrollment_registry.json"
    try:
        with registry_path.open("r", encoding="utf-8") as stream:
            registry = json.load(stream)
    except FileNotFoundError:
        return WakeWordModelBinding(False, registry_path=registry_path)
    except Exception:
        return WakeWordModelBinding(True, registry_path=registry_path)
    if not isinstance(registry, Mapping):
        return WakeWordModelBinding(True, registry_path=registry_path)
    if not any(key in registry for key in _WAKE_WORD_REGISTRY_KEYS):
        return WakeWordModelBinding(False, registry_path=registry_path)

    try:
        model_value = registry.get("wake_word_model")
        dataset_fingerprint = registry.get("wake_word_model_dataset_fingerprint")
        content_fingerprint = registry.get("wake_word_model_content_fingerprint")
        if (
            not isinstance(model_value, str)
            or not model_value.strip()
            or not isinstance(dataset_fingerprint, str)
            or len(dataset_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in dataset_fingerprint.casefold())
            or not isinstance(content_fingerprint, str)
            or len(content_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in content_fingerprint.casefold())
        ):
            return WakeWordModelBinding(True, registry_path=registry_path)
        model_path = Path(model_value).expanduser().resolve()
        models_root = (root / "models").resolve()
        if (
            (model_path != models_root and models_root not in model_path.parents)
            or not model_path.is_file()
        ):
            return WakeWordModelBinding(True, registry_path=registry_path)

        from asher.voiceguard.enrollment import EnrollmentManager
        from asher.voiceguard.features import StatisticalFeatureExtractor
        from asher.voiceguard.model import CalibratedVoiceGuardModel

        extractor = StatisticalFeatureExtractor()
        verifier = FileWakeWordVerifier(
            model_path,
            extractor=extractor,
            temp_root=root / "tmp",
            registry_path=registry_path,
            expected_dataset_fingerprint=dataset_fingerprint,
            expected_model_content_fingerprint=content_fingerprint,
        )
        model = CalibratedVoiceGuardModel.load(model_path)
        if not verifier._model_matches_expected_content(model):
            return WakeWordModelBinding(True, registry_path=registry_path)
        dataset = EnrollmentManager(root).load_training_dataset()
        if dataset.fingerprint != dataset_fingerprint:
            return WakeWordModelBinding(True, registry_path=registry_path)
        verifier._model = model
        return WakeWordModelBinding(
            True,
            verifier=verifier,
            registry_path=registry_path,
            reason="Active trained wake-word artifact",
        )
    except Exception:
        return WakeWordModelBinding(True, registry_path=registry_path)


@dataclass(frozen=True)
class VoiceRuntimeEvent:
    kind: str
    message: str
    transcript: TranscriptResult | None = None
    reply: CompanionReply | None = None
    confidence: float | None = None


class VoiceRuntime:
    """Standby â†’ wake â†’ authenticated session â†’ command loop."""

    def __init__(
        self,
        controller: CompanionController,
        *,
        config: AsherConfig | None = None,
        backend: AudioBackend | None = None,
        transcriber: Any | None = None,
        cloud_transcriber: Any | None = None,
        vocabulary: DynamicVocabulary | None = None,
        wake_detector: Any | None = None,
        wake_word_binding: WakeWordModelBinding | None = None,
        voiceguard: VoiceGuardVerifier | None = None,
        tts: Any | None = None,
        active_window_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[VoiceRuntimeEvent], None] | None = None,
    ) -> None:
        self.controller = controller
        self.config = config or controller.config
        project_root = Path(__file__).resolve().parents[2]
        self.vocabulary = vocabulary or DynamicVocabulary(
            contacts_path=project_root / "data" / "voice_contacts.json",
            applications_path=project_root / "data" / "apps.json",
        )
        self.backend = backend or SoundDeviceBackend()
        self.transcriber = transcriber or FasterWhisperTranscriber(
            TranscriptionConfig(
                model_size=self.config.whisper_model,
                device=self.config.whisper_device,
                cuda_compute_type=self.config.whisper_compute_type if self.config.whisper_compute_type != "auto" else "float16",
            )
        )
        self.cloud_transcriber = cloud_transcriber
        self.pipeline = VoiceAccuracyPipeline(
            self.transcriber,
            self.vocabulary,
            fallback_transcriber=cloud_transcriber,
        )
        self.wake_detector = wake_detector or TextWakeDetector()
        self._wake_word_binding_override = wake_word_binding
        self.energy_gate = EnergyGate()
        self.voiceguard = voiceguard
        if active_window_seconds <= 0:
            raise ValueError("active_window_seconds must be positive")
        self.active_window_seconds = float(active_window_seconds)
        self._clock = clock
        if tts is None:
            from asher.voice.tts import build_default_tts

            tts = build_default_tts(selected_profile=self.config.voice_profile)
        self.tts = tts
        self.on_event = on_event or (lambda event: None)
        loop = getattr(controller, "loop", None)
        self.states = getattr(loop, "states", None) or StateStore()
        self._stop = CancellationToken()
        self._microphone_level_lock = threading.Lock()
        self._microphone_level = 0.0
        self._wake_fallback_announced = False
        self._wake_turn_vad = WAKE_TURN_VAD
        self._command_turn_vad = COMMAND_TURN_VAD

    @property
    def microphone_level(self) -> float:
        """Latest presentation-only live-microphone RMS scalar.

        No PCM is retained or published. TTS does not drive this value, so
        synthetic speech has no fabricated amplitude animation.
        """

        with self._microphone_level_lock:
            return self._microphone_level

    def _observe_microphone_frame(self, pcm16: bytes, *, speech_detected: bool) -> None:
        level = normalized_pcm16_rms(pcm16)
        with self._microphone_level_lock:
            if speech_detected:
                self._microphone_level = level
                return
            # Frames below the standby gate are presentation silence. Decay
            # quickly to an exact zero instead of preserving a noise floor.
            decayed = self._microphone_level * 0.45
            self._microphone_level = 0.0 if decayed < 0.005 else decayed

    def _reset_microphone_level(self) -> None:
        with self._microphone_level_lock:
            self._microphone_level = 0.0

    def stop(self) -> None:
        """Stop microphone and speech work without latching emergency stop.

        The controller's emergency-stop path remains separate and is used only
        for an explicit emergency command/button. Pausing ordinary listening
        must not disable the rest of the application.
        """

        self._stop.cancel("Voice runtime stopped")
        self._reset_microphone_level()
        try:
            self.tts.stop()
        except Exception:
            pass

    def run_forever(self) -> None:
        """Run until stop or KeyboardInterrupt; all hardware is opened here."""
        try:
            frames = self.backend.frames(self._stop)
            active_session = None
            active_until = 0.0
            while not self._stop.cancelled:
                active = active_session is not None and self._clock() < active_until
                if self._tts_is_speaking():
                    self._transition(AssistantState.SPEAKING, "Speaking")
                elif not active:
                    active_session = None
                    active_until = 0.0
                    self._transition(AssistantState.STANDBY, "Standby: listening for Hey Asher")
                    self._emit("standby", "Standby: listening for Hey Asher")
                else:
                    self._transition(AssistantState.LISTENING, "Listening for your command")
                    self._emit("listening", "Listening for your command")

                turn = self._capture_trigger(
                    frames,
                    deadline=active_until if active else None,
                    vad_config=self._command_turn_vad if active else self._wake_turn_vad,
                )
                if turn is None:
                    if active and not self._stop.cancelled:
                        active_session = None
                        active_until = 0.0
                        self._transition(AssistantState.STANDBY, "Active listening timed out. Say Hey Asher when you need me.")
                        self._emit("standby", "Active listening timed out. Say Hey Asher when you need me.")
                    continue
                if not turn.pcm16:
                    continue

                result = None
                command = ""
                if active_session is None:
                    binding = (
                        self._wake_word_binding_override
                        if self._wake_word_binding_override is not None
                        else load_active_wake_word_binding(self.controller)
                    )
                    if binding.active_artifact:
                        wake_accepted, wake_score, _wake_reason = binding.verify(
                            turn.pcm16,
                            turn.sample_rate,
                        )
                        if not wake_accepted:
                            self._reject_wake(
                                "Wake phrase was not verified by the active wake-word model",
                                confidence=wake_score,
                            )
                            continue

                        # A trained acoustic wake turn is activation audio only.
                        # Never re-use or decode that buffer as the command: the
                        # next VAD turn is captured independently, which prevents
                        # a mis-decoded wake prefix from leaking into the request.
                        self._accept_wake(
                            "Hey Asher detected by the trained acoustic wake model",
                            confidence=wake_score,
                        )
                        active_session = self._authenticate_speaker(turn)
                        active_until = self._clock() + self.active_window_seconds
                        self._transition(AssistantState.LISTENING, "Listening for your command")
                        self._emit("listening", "Yes?")
                        self._speak("Yes?", return_state=AssistantState.LISTENING)
                        continue

                    # No personalized acoustic artifact exists yet. This is the
                    # explicitly degraded transcript-only wake fallback.
                    fallback_allowed, fallback_score, fallback_reason = binding.verify(
                        turn.pcm16,
                        turn.sample_rate,
                    )
                    if not fallback_allowed:
                        self._reject_wake(
                            fallback_reason,
                            confidence=fallback_score,
                        )
                        continue
                    self._announce_wake_fallback()
                    result = self._transcribe_turn(turn)

                    # Wake activation is a lower-risk boundary than command
                    # execution. A short phrase such as ``Hey Asher`` can be
                    # decoded correctly while still receiving a confidence
                    # score below the general command gate. Detect the wake
                    # phrase before applying that command-confidence gate.
                    # Speaker/session policy still runs after wake acceptance.
                    heard = result.transcript.normalized_text or result.transcript.raw_text
                    try:
                        wake = self.wake_detector.detect(heard, fuzzy=True)
                    except TypeError:
                        wake = self.wake_detector.detect(heard)
                    if not wake.detected:
                        self._transition(AssistantState.STANDBY, "Wake phrase not detected")
                        self._emit(
                            "wake_fallback_rejected"
                            if result.status is not PipelineStatus.ACCEPTED
                            else "wake_rejected",
                            result.clarification
                            or "Wake phrase not detected by transcript fallback",
                            result.transcript,
                            confidence=result.transcript.acoustic_confidence,
                        )
                        continue

                    self._accept_wake(
                        f"Hey Asher detected by transcript-only fallback ({wake.provider})",
                        transcript=result.transcript,
                        confidence=result.transcript.acoustic_confidence,
                    )

                    active_session = self._authenticate_speaker(
                        turn,
                        transcript=result.transcript,
                    )
                    active_until = self._clock() + self.active_window_seconds

                    # A low-confidence inline command may wake ASHER, but it is
                    # never executed. Capture a fresh command turn instead.
                    if result.status is not PipelineStatus.ACCEPTED:
                        self._transition(AssistantState.LISTENING, "Listening for your command")
                        self._emit("listening", "Yes?", result.transcript)
                        self._speak("Yes?", return_state=AssistantState.LISTENING)
                        continue

                    command = wake.command
                    if not command:
                        self._transition(AssistantState.LISTENING, "Listening for your command")
                        self._emit("listening", "Yes?", result.transcript)
                        self._speak("Yes?", return_state=AssistantState.LISTENING)
                        continue
                else:
                    result = self._transcribe_turn(turn)
                    if result.status is not PipelineStatus.ACCEPTED:
                        self._emit(
                            "clarification",
                            result.clarification or "Please repeat that.",
                            result.transcript,
                        )
                        self._speak(result.clarification or "Please repeat that.")
                        active_until = self._clock() + self.active_window_seconds
                        continue
                    heard = result.executable_command or ""
                    wake = self.wake_detector.detect(heard)
                    # Repeating the wake phrase inside an already-active session
                    # is harmless; strip it if Whisper decoded it correctly.
                    command = wake.command if wake.detected else heard

                assert result is not None
                command = command.strip()
                if not command:
                    self._transition(AssistantState.LISTENING, "Listening for your command")
                    continue

                # This is the only permanent final-user-transcript event.  Wake
                # candidates, rejected standby audio and low-confidence turns do
                # not use this event kind.  The event message is the canonical
                # command after wake-prefix removal/entity normalization.
                self._emit(
                    "transcript",
                    command,
                    result.transcript,
                    confidence=result.transcript.acoustic_confidence,
                )

                if command.casefold().strip().rstrip(".,!?;:") in SLEEP_COMMANDS:
                    active_session = None
                    active_until = 0.0
                    self._emit("standby", "Going back to standby.", result.transcript)
                    self._speak("Going back to standby.", return_state=AssistantState.STANDBY)
                    continue
                reply = self.controller.handle_text(command, active_session)
                self._emit("reply", reply.text, result.transcript, reply=reply)
                speech_return_state = (
                    AssistantState.AWAITING_CONFIRMATION
                    if reply.confirmation_id
                    else AssistantState.LISTENING
                )
                self._speak(reply.text, return_state=speech_return_state)
                active_until = self._clock() + self.active_window_seconds
                if reply.confirmation_id:
                    self._emit("confirmation", "Open the desktop confirmation panel to approve this action.", result.transcript, reply=reply)
        except (KeyboardInterrupt, CancelledError):
            self.stop()
        finally:
            self._reset_microphone_level()

    def _transcribe_turn(self, turn: Any):
        """Transcribe one captured turn without retaining its temporary WAV."""

        # Faster-Whisper accepts a path/array rather than arbitrary PCM bytes.
        # Keep the turn in a short-lived WAV and remove it immediately after
        # local/optional remote transcription; no recording is retained.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            turn_path = Path(temporary.name)
        try:
            self._transition(AssistantState.TRANSCRIBING, "Understanding speech")
            self._emit("transcribing", "Understanding speech")
            _write_wav(turn_path, turn.pcm16, turn.sample_rate)
            return self.pipeline.process(
                turn_path,
                allow_remote_fallback=self.cloud_transcriber is not None,
                cancellation=self._stop,
            )
        finally:
            turn_path.unlink(missing_ok=True)

    def _announce_wake_fallback(self) -> None:
        if self._wake_fallback_announced:
            return
        self._wake_fallback_announced = True
        self._emit(
            "wake_fallback",
            "Personalized wake model is not trained; using transcript-only wake fallback",
        )

    def _accept_wake(
        self,
        message: str,
        *,
        transcript: TranscriptResult | None = None,
        confidence: float | None = None,
    ) -> None:
        self._transition(AssistantState.WAKE_DETECTED, message)
        self._emit(
            "wake_detected",
            message,
            transcript,
            confidence=confidence,
        )

    def _reject_wake(
        self,
        message: str,
        *,
        transcript: TranscriptResult | None = None,
        confidence: float | None = None,
    ) -> None:
        self._transition(AssistantState.STANDBY, message)
        self._emit(
            "wake_rejected",
            message,
            transcript,
            confidence=confidence,
        )

    def _authenticate_speaker(
        self,
        turn: Any,
        *,
        transcript: TranscriptResult | None = None,
    ) -> Any:
        """Authenticate only after wake acceptance; wake is not authorization."""

        self._transition(AssistantState.AUTHENTICATING, "Verifying speaker")
        self._emit("authenticating", "Verifying speaker", transcript)
        owner_id = None
        score = None
        reason = "VoiceGuard not enrolled; guest session"
        if self.voiceguard is not None:
            owner_id, score, reason = self.voiceguard.authenticate(
                turn.pcm16,
                turn.sample_rate,
            )
        actor = self.controller.users.get(owner_id) if owner_id else None
        if actor is not None and actor.role.value in {"owner", "trusted"}:
            session = self.controller.create_voice_session(actor)
            self._transition(
                AssistantState.AUTHENTICATED,
                "Speaker authenticated",
                actor_id=getattr(actor, "user_id", None),
                confidence=score,
            )
            self._emit("authenticated", reason, transcript, confidence=score)
            return session

        session = self.controller.create_guest_session()
        self._transition(
            AssistantState.LOCKED,
            "Private actions locked; guest conversation only",
            confidence=score,
        )
        self._emit(
            "guest",
            "Wake phrase heard, but speaker authentication did not grant private access.",
            transcript,
            confidence=score,
        )
        return session

    def _capture_trigger(
        self,
        frames: Any,
        *,
        deadline: float | None = None,
        vad_config: VadConfig | None = None,
    ):
        """Capture one VAD-bounded turn while preserving pre-trigger audio.

        The deadline applies only while waiting for speech to start. Once a
        user has started a turn, let endpointing finish it instead of truncating
        a sentence because the active-listening window expired mid-utterance.
        """
        flush_fn = getattr(self.backend, "flush", None)
        if callable(flush_fn):
            try:
                flush_fn()
            except Exception:
                pass

        config = vad_config or WAKE_TURN_VAD
        pre_roll: deque[AudioFrame] = deque(maxlen=config.pre_roll_frames)
        tts_was_speaking = self._tts_is_speaking()

        while not self._stop.cancelled:
            if deadline is not None and self._clock() >= deadline:
                self._reset_microphone_level()
                return None
            try:
                frame = next(frames)
            except StopIteration:
                self.stop()
                return None

            is_speaking = self._tts_is_speaking()
            if tts_was_speaking and not is_speaking:
                if callable(flush_fn):
                    try:
                        flush_fn()
                    except Exception:
                        pass
                self._last_tts_speak_time = self._clock()
                pre_roll.clear()
                tts_was_speaking = False
                continue

            tts_was_speaking = is_speaking

            cooldown_active = (self._clock() - getattr(self, "_last_tts_speak_time", 0.0)) < 0.4
            if is_speaking or cooldown_active:
                pre_roll.clear()
                self._observe_microphone_frame(frame.pcm16, speech_detected=False)
                continue

            speech_detected = self.energy_gate.detect(frame.pcm16).detected
            self._observe_microphone_frame(
                frame.pcm16,
                speech_detected=speech_detected,
            )
            if not speech_detected:
                pre_roll.append(frame)
                continue


            # A detected user turn is a barge-in signal. Providers are
            # interruptible; stopping here keeps speech from blocking the next
            # command or an emergency request.
            try:
                self.tts.stop()
            except Exception:
                pass

            vad = VoiceActivityDetector(config)
            if pre_roll:
                # Calibrate only from frames that the cheap standby energy gate
                # already classified as non-speech. This gives TurnCapture a
                # local noise-floor estimate without retaining raw audio.
                vad.calibrate(tuple(pre_roll))
            capture = TurnCapture(vad)
            buffered = tuple(pre_roll)
            pre_roll.clear()

            def remaining_frames():
                yield from buffered
                yield frame
                for _ in range(capture.config.max_turn_frames - len(buffered) - 1):
                    if self._stop.cancelled:
                        return
                    try:
                        next_frame = next(frames)
                    except StopIteration:
                        return
                    next_frame_detected = self.energy_gate.detect(
                        next_frame.pcm16
                    ).detected
                    self._observe_microphone_frame(
                        next_frame.pcm16,
                        speech_detected=next_frame_detected,
                    )
                    yield next_frame

            try:
                return capture.capture(remaining_frames(), cancellation=self._stop)
            finally:
                self._reset_microphone_level()
        self._reset_microphone_level()
        raise CancelledError(self._stop.reason)

    def _speak(
        self,
        text: str,
        *,
        return_state: AssistantState = AssistantState.LISTENING,
    ) -> None:
        clean = str(text).strip()
        if not clean or self._stop.cancelled:
            return
        self._transition(
            AssistantState.SPEAKING,
            "Speaking",
            voice_profile=getattr(self.tts, "selected_profile_name", None),
        )
        self._emit("speaking", clean)
        try:
            handle = self.tts.speak_async(clean, interrupt=True)
        except Exception as error:
            # Speech failure must not discard the text reply or stop listening.
            self._transition(
                AssistantState.ERROR,
                "Speech output unavailable",
                error=type(error).__name__,
            )
            self._emit("speech_error", f"Speech output unavailable: {type(error).__name__}")
            return

        wait = getattr(handle, "wait", None)
        if callable(wait):
            watcher = threading.Thread(
                target=self._watch_speech,
                args=(handle, return_state),
                name="asher-tts-state",
                daemon=True,
            )
            watcher.start()
        else:
            # Lightweight test/fallback providers may not return a waitable
            # handle. The next real runtime event will advance the state.
            self._transition(return_state, "Ready")

    def _watch_speech(self, handle: Any, return_state: AssistantState) -> None:
        try:
            result = handle.wait()
        except Exception as error:
            self._transition(
                AssistantState.ERROR,
                "Speech output failed",
                error=type(error).__name__,
            )
            self._emit("speech_error", f"Speech output unavailable: {type(error).__name__}")
            return
        if self._stop.cancelled:
            return
        if result is not None and not getattr(result, "success", False):
            if getattr(result, "cancelled", False):
                return
            self._transition(
                AssistantState.ERROR,
                "Speech output failed",
                error=getattr(result, "error", None),
            )
            self._emit("speech_error", "Speech output failed")
            return
        self._transition(return_state, "Ready")
        self._emit("speech_finished", "Speech finished")

    def _tts_is_speaking(self) -> bool:
        try:
            return bool(getattr(self.tts, "is_speaking", False))
        except Exception:
            return False

    def _transition(self, state: AssistantState, message: str, **details: Any) -> None:
        self.states.transition(state, message, **details)

    def _emit(self, kind: str, message: str, transcript: TranscriptResult | None = None, *, reply: CompanionReply | None = None, confidence: float | None = None) -> None:
        self.on_event(VoiceRuntimeEvent(kind, message, transcript, reply, confidence))


def _write_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm16)

