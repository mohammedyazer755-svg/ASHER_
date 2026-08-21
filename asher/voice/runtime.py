"""Explicit voice runtime orchestration; importing it never opens hardware."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from asher.agent.controller import CompanionController, CompanionReply
from asher.config import AsherConfig
from asher.core.cancellation import CancellationToken, CancelledError
from asher.core.state import AssistantState, StateStore
from asher.voice.capture import AudioFrame, TurnCapture, VadConfig
from asher.voice.pipeline import PipelineStatus, VoiceAccuracyPipeline
from asher.voice.transcription import FasterWhisperTranscriber, TranscriptionConfig
from asher.voice.types import TranscriptResult
from asher.voice.vocabulary import DynamicVocabulary
from asher.voice.wakeword import EnergyGate, TextWakeDetector


SLEEP_COMMANDS = frozenset(
    {
        "sleep",
        "go to sleep",
        "standby",
        "go to standby",
        "that's all",
        "that’s all",
        "nothing else",
    }
)


class AudioBackend(Protocol):
    def frames(self, cancellation: CancellationToken | None = None): ...


class SoundDeviceBackend:
    """16 kHz mono PCM stream with no recording retained beyond the turn."""

    def __init__(self, *, sample_rate: int = 16_000, block_samples: int = 320, device: int | None = None) -> None:
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
            while cancellation is None or not cancellation.cancelled:
                data, _overflowed = stream.read(self.block_samples)
                yield AudioFrame(bytes(data), self.sample_rate)


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


class FileVoiceGuardVerifier:
    """Inference adapter that deletes temporary audio immediately."""

    def __init__(self, model_path: str | Path, label_to_user: dict[str, str], *, extractor: Any, temp_root: str | Path | None = None) -> None:
        self.model_path = Path(model_path)
        self.label_to_user = dict(label_to_user)
        self.extractor = extractor
        self.temp_root = Path(temp_root) if temp_root else None
        self._model: Any | None = None

    def authenticate(self, pcm16: bytes, sample_rate: int) -> tuple[str | None, float, str]:
        from asher.voiceguard.model import CalibratedVoiceGuardModel

        if self._model is None:
            self._model = CalibratedVoiceGuardModel.load(self.model_path)
        self.temp_root and self.temp_root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_root)
        path = Path(handle.name)
        handle.close()
        try:
            _write_wav(path, pcm16, sample_rate)
            result = self._model.verify_wav(path, self.extractor)
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
        model_path = Path(model_value).expanduser().resolve()
        models_root = (root / "models").resolve()
        if model_path != models_root and models_root not in model_path.parents:
            return None
        if not model_path.is_file():
            return None

        from asher.voiceguard.features import StatisticalFeatureExtractor
        from asher.voiceguard.model import CalibratedVoiceGuardModel

        model = CalibratedVoiceGuardModel.load(model_path)
        if model.task != "speaker_auth":
            return None
        active = {
            actor.user_id: actor
            for actor in controller.users.list_active()
            if actor.role.value in {"owner", "trusted"}
        }
        authorized = set(model.authorized_labels)
        if not authorized or not authorized.issubset(active):
            return None
        extractor = StatisticalFeatureExtractor()
        if extractor.metadata.extractor_id != model.extractor_metadata.extractor_id:
            return None
        return FileVoiceGuardVerifier(
            model_path,
            {label: label for label in model.classes if label in active},
            extractor=extractor,
            temp_root=controller.config.runtime.root / "voiceguard" / "tmp",
        )
    except Exception:
        # A missing/corrupt/stale model must degrade to guest access, never to
        # an optimistic owner session.
        return None


@dataclass(frozen=True)
class VoiceRuntimeEvent:
    kind: str
    message: str
    transcript: TranscriptResult | None = None
    reply: CompanionReply | None = None
    confidence: float | None = None


class VoiceRuntime:
    """Standby → wake → authenticated session → command loop."""

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

    def stop(self) -> None:
        """Stop microphone and speech work without latching emergency stop.

        The controller's emergency-stop path remains separate and is used only
        for an explicit emergency command/button. Pausing ordinary listening
        must not disable the rest of the application.
        """

        self._stop.cancel("Voice runtime stopped")
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
                # Faster-Whisper accepts a path/array rather than arbitrary
                # PCM bytes.  Keep the turn in a short-lived WAV and remove it
                # immediately after local/optional remote transcription; no
                # recording is retained by the runtime.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                    turn_path = Path(temporary.name)
                try:
                    self._transition(AssistantState.TRANSCRIBING, "Understanding speech")
                    self._emit("transcribing", "Understanding speech")
                    _write_wav(turn_path, turn.pcm16, turn.sample_rate)
                    result = self.pipeline.process(
                        turn_path,
                        allow_remote_fallback=self.cloud_transcriber is not None,
                        cancellation=self._stop,
                    )
                finally:
                    turn_path.unlink(missing_ok=True)
                self._emit("transcript", result.clarification or result.executable_command or "", result.transcript, confidence=result.transcript.acoustic_confidence)
                if result.status is not PipelineStatus.ACCEPTED:
                    self._emit("clarification", result.clarification or "Please repeat that.", result.transcript)
                    if active:
                        self._speak(result.clarification or "Please repeat that.")
                        active_until = self._clock() + self.active_window_seconds
                    continue

                heard = result.executable_command or ""
                wake = self.wake_detector.detect(heard)
                if active_session is None:
                    if not wake.detected:
                        self._transition(AssistantState.STANDBY, "Wake phrase not detected")
                        continue
                    self._transition(AssistantState.WAKE_DETECTED, "Hey Asher detected")
                    self._emit("wake_detected", "Hey Asher detected", result.transcript)
                    self._transition(AssistantState.AUTHENTICATING, "Verifying speaker")
                    self._emit("authenticating", "Verifying speaker", result.transcript)
                    owner_id = None
                    score = None
                    reason = "VoiceGuard not enrolled; guest session"
                    if self.voiceguard is not None:
                        owner_id, score, reason = self.voiceguard.authenticate(turn.pcm16, turn.sample_rate)
                    actor = self.controller.users.get(owner_id) if owner_id else None
                    if actor is not None and actor.role.value in {"owner", "trusted"}:
                        active_session = self.controller.create_voice_session(actor)
                        self._transition(
                            AssistantState.AUTHENTICATED,
                            "Speaker authenticated",
                            actor_id=getattr(actor, "user_id", None),
                            confidence=score,
                        )
                        self._emit("authenticated", reason, result.transcript, confidence=score)
                    else:
                        active_session = self.controller.create_guest_session()
                        self._transition(
                            AssistantState.LOCKED,
                            "Private actions locked; guest conversation only",
                            confidence=score,
                        )
                        self._emit("guest", "Wake phrase heard, but speaker authentication did not grant private access.", result.transcript, confidence=score)
                    active_until = self._clock() + self.active_window_seconds
                    command = wake.command
                else:
                    # During the short active window a follow-up command does
                    # not need to repeat the wake phrase. Repeating it remains
                    # harmless and strips it from the executable command.
                    command = wake.command if wake.detected else heard

                if not command:
                    self._transition(AssistantState.LISTENING, "Listening for your command")
                    self._emit("listening", "Yes?", result.transcript)
                    self._speak("Yes?", return_state=AssistantState.LISTENING)
                    continue
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

    def _capture_trigger(self, frames: Any, *, deadline: float | None = None):
        # Skip silence cheaply, then let VAD collect a complete turn.
        while not self._stop.cancelled:
            if deadline is not None and self._clock() >= deadline:
                return None
            try:
                frame = next(frames)
            except StopIteration:
                self.stop()
                return None
            if not self.energy_gate.detect(frame.pcm16).detected:
                continue
            # A detected user turn is a barge-in signal. Providers are
            # interruptible; stopping here keeps speech from blocking the next
            # command or an emergency request.
            try:
                self.tts.stop()
            except Exception:
                pass
            capture = TurnCapture()

            def remaining_frames():
                yield frame
                for _ in range(capture.config.max_turn_frames - 1):
                    if self._stop.cancelled:
                        return
                    if deadline is not None and self._clock() >= deadline:
                        return
                    try:
                        yield next(frames)
                    except StopIteration:
                        return

            return capture.capture(remaining_frames(), cancellation=self._stop)
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
