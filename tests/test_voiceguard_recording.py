"""Deterministic, hardware-free tests for VoiceGuard recording primitives."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asher.voiceguard import (
    AugmentationConfig,
    AudioFormatError,
    PcmAudio,
    RecordingSession,
    SampleCondition,
    SampleOrigin,
    augment_audio,
    augment_session,
    load_dataset,
    load_manifest,
    read_wav,
)


class VoiceGuardRecordingTests(unittest.TestCase):
    def test_import_is_safe_without_optional_audio_or_ml_initialization(self) -> None:
        code = (
            "import sys; import asher.voiceguard; "
            "print(any(name in sys.modules for name in ('sounddevice','speechbrain','torch','numpy','sklearn')))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False")

    def test_consent_is_required_and_manifest_tracks_real_wav(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "recordings"
            with self.assertRaises(Exception):
                RecordingSession.create(
                    root,
                    speaker_id="fixture-owner",
                    role="owner",
                    environment="test-room",
                    consent=False,
                )
            session = RecordingSession.create(
                root,
                speaker_id="fixture-owner",
                role="owner",
                environment="test-room",
                consent=True,
                session_id="session-clean",
            )
            sample = session.add_pcm16(
                [1000, -1000] * 800,
                contains_wake_phrase=True,
                sample_id="sample-clean",
            )
            self.assertEqual(sample.sample_rate, 16_000)
            self.assertEqual(read_wav(session.directory / sample.path).frame_count, 1600)
            loaded = load_manifest(session.directory)
            self.assertEqual(loaded.session_id, "session-clean")
            self.assertEqual(len(loaded.samples), 1)
            self.assertTrue((session.directory / sample.path).read_bytes().startswith(b"RIFF"))
            dataset = load_dataset(root)
            self.assertEqual(len(dataset.samples), 1)
            self.assertEqual(dataset.samples[0].wav_path.name, "sample-clean.wav")

    def test_manifest_rejects_path_traversal(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "recordings"
            session = RecordingSession.create(
                root,
                speaker_id="fixture-owner",
                role="owner",
                environment="test-room",
                consent=True,
                session_id="session-safe",
            )
            value = json.loads(session.manifest_path.read_text(encoding="utf-8"))
            value["samples"] = [
                {
                    "sample_id": "escape",
                    "path": "../escape.wav",
                    "sha256": "0" * 64,
                    "duration_seconds": 1,
                    "sample_rate": 16000,
                    "channels": 1,
                }
            ]
            session.manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(Exception):
                load_manifest(session.directory)

    def test_manifest_rejects_string_booleans(self) -> None:
        with TemporaryDirectory() as temporary:
            session = RecordingSession.create(
                Path(temporary) / "recordings",
                speaker_id="fixture-owner",
                role="owner",
                environment="test-room",
                consent=True,
                session_id="session-strict-booleans",
            )
            session.add_pcm16(
                [1000, -1000] * 32,
                contains_wake_phrase=False,
                expected_authorized=False,
                sample_id="strict-sample",
            )
            original = json.loads(session.manifest_path.read_text(encoding="utf-8"))
            for field in ("contains_wake_phrase", "expected_authorized"):
                with self.subTest(field=field):
                    value = json.loads(json.dumps(original))
                    value["samples"][0][field] = "false"
                    session.manifest_path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(Exception):
                        load_manifest(session.directory)

    def test_augmentation_must_preserve_source_wake_and_authorization_labels(self) -> None:
        with TemporaryDirectory() as temporary:
            session = RecordingSession.create(
                Path(temporary) / "recordings",
                speaker_id="fixture-owner",
                role="owner",
                environment="test-room",
                consent=True,
                session_id="session-label-binding",
            )
            source = session.add_pcm16(
                [1000, -1000] * 32,
                contains_wake_phrase=True,
                expected_authorized=True,
                sample_id="source-sample",
            )
            with self.assertRaises(Exception):
                session.add_pcm16(
                    [900, -900] * 32,
                    contains_wake_phrase=False,
                    expected_authorized=False,
                    origin=SampleOrigin.AUGMENTED,
                    source_sample_id=source.sample_id,
                    sample_id="mislabeled-augmentation",
                )

    def test_augmentation_is_deterministic_and_tagged_noisy(self) -> None:
        audio = PcmAudio(tuple([1000, -500] * 400))
        recipe = AugmentationConfig(gain_db=3, noise_snr_db=18, reverb_decay=0.2, seed=42)
        first = augment_audio(audio, recipe)
        second = augment_audio(audio, recipe)
        self.assertEqual(first, second)
        self.assertNotEqual(first.samples, audio.samples)
        with TemporaryDirectory() as temporary:
            session = RecordingSession.create(
                Path(temporary) / "recordings",
                speaker_id="fixture-owner",
                role="owner",
                environment="test-room",
                consent=True,
                session_id="session-augment",
            )
            original = session.add_audio(audio, contains_wake_phrase=True)
            derived = augment_session(session, [recipe])
            self.assertEqual(len(derived), 1)
            self.assertEqual(derived[0].condition, SampleCondition.NOISY.value)
            self.assertEqual(derived[0].source_sample_id, original.sample_id)


if __name__ == "__main__":
    unittest.main()
