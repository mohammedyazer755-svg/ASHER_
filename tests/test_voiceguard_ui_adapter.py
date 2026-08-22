"""Desktop VoiceGuard collection boundaries without microphone hardware."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from asher.ui.voiceguard_adapter import VoiceGuardDesktopAdapter
from asher.voiceguard import (
    DatasetError,
    EnrollmentError,
    RecordingSession,
    load_dataset,
    load_manifest,
)


class _MicrophoneRecorder:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self._lock = threading.Lock()

    def patch(self):
        recorder = self

        def record(
            session: RecordingSession,
            duration_seconds: float,
            *,
            contains_wake_phrase: bool,
            sample_rate: int = 16_000,
            channels: int = 1,
            device: int | str | None = None,
            condition: str = "clean",
            expected_authorized: bool | None = None,
            sample_id: str | None = None,
        ):
            del duration_seconds, device
            with recorder._lock:
                recorder.calls += 1
                call = recorder.calls
            if recorder.fail_on_call == call:
                raise RuntimeError("fixture microphone interruption")
            value = 500 + call
            return session.add_pcm16(
                [value, -value] * 200,
                sample_rate=sample_rate,
                channels=channels,
                contains_wake_phrase=contains_wake_phrase,
                condition=condition,
                expected_authorized=expected_authorized,
                sample_id=sample_id,
            )

        return patch.object(RecordingSession, "record_microphone", new=record)


class VoiceGuardDesktopAdapterTests(unittest.TestCase):
    def test_six_clicks_form_one_registered_six_clip_session(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(temporary)
            recorder = _MicrophoneRecorder()
            self.assertEqual(recorder.calls, 0)
            adapter.begin_user("fixture-owner", "owner", consent=True)

            with recorder.patch():
                counts = [
                    adapter.capture_sample(
                        "fixture-owner",
                        contains_wake_phrase=index == 0,
                    )
                    for index in range(6)
                ]

            self.assertEqual(counts, [1, 2, 3, 4, 5, 6])
            self.assertEqual(recorder.calls, 6)
            records = adapter.manager.list_users()
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].session_ids), 1)
            dataset = adapter.manager.load_training_dataset()
            self.assertEqual(len(dataset.sessions), 1)
            self.assertEqual(len(dataset.samples), 6)
            self.assertEqual({sample.role for sample in dataset.samples}, {"owner"})
            self.assertEqual(
                {sample.record.expected_authorized for sample in dataset.samples},
                {True},
            )
            self.assertEqual(
                sum(sample.record.contains_wake_phrase for sample in dataset.samples),
                1,
            )

    def test_partial_collection_is_unregistered_and_training_is_actionable(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(temporary)
            adapter.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()

            with recorder.patch():
                self.assertEqual(adapter.capture_sample("fixture-owner"), 1)
                self.assertEqual(adapter.capture_sample("fixture-owner"), 2)

            self.assertEqual(adapter.manager.list_users(), ())
            self.assertEqual(adapter.manager.load_training_dataset().sessions, ())
            on_disk = load_dataset(adapter.manager.recordings_root)
            self.assertEqual(len(on_disk.sessions), 1)
            self.assertEqual(len(on_disk.samples), 2)
            with self.assertRaisesRegex(
                DatasetError,
                r"guided 6-clip session.*2/6 clips captured.*unregistered",
            ):
                adapter.train("fixture-owner")

    def test_interrupted_collection_resumes_without_registering_partial_data(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(temporary)
            adapter.begin_user("fixture-owner", "owner", consent=True)
            interrupted = _MicrophoneRecorder(fail_on_call=3)

            with interrupted.patch():
                self.assertEqual(adapter.capture_sample("fixture-owner"), 1)
                self.assertEqual(adapter.capture_sample("fixture-owner"), 2)
                with self.assertRaisesRegex(RuntimeError, "microphone interruption"):
                    adapter.capture_sample("fixture-owner")

            self.assertEqual(adapter.manager.list_users(), ())
            self.assertEqual(len(load_dataset(adapter.manager.recordings_root).samples), 2)

            resumed = _MicrophoneRecorder()
            with resumed.patch():
                self.assertEqual(adapter.capture_sample("fixture-owner"), 3)
                self.assertEqual(adapter.capture_sample("fixture-owner"), 4)
                self.assertEqual(adapter.capture_sample("fixture-owner"), 5)
                self.assertEqual(adapter.capture_sample("fixture-owner"), 6)

            self.assertEqual(len(adapter.manager.load_training_dataset().sessions), 1)
            self.assertEqual(len(adapter.manager.load_training_dataset().samples), 6)

    def test_concurrent_clicks_are_serialized_into_one_session(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(temporary)
            adapter.begin_user("fixture-guest", "guest", consent=True)
            recorder = _MicrophoneRecorder()

            with recorder.patch(), ThreadPoolExecutor(max_workers=6) as executor:
                counts = tuple(
                    executor.map(
                        lambda _index: adapter.capture_sample("fixture-guest"),
                        range(6),
                    )
                )

            self.assertEqual(sorted(counts), [1, 2, 3, 4, 5, 6])
            dataset = adapter.manager.load_training_dataset()
            self.assertEqual(len(dataset.sessions), 1)
            self.assertEqual(len(dataset.samples), 6)
            self.assertEqual({sample.role for sample in dataset.samples}, {"unknown"})
            self.assertEqual(
                {sample.record.expected_authorized for sample in dataset.samples},
                {False},
            )

    def test_revoke_marks_registered_and_partial_sessions_unusable(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(
                temporary,
                minimum_session_gap_seconds=0,
            )
            adapter.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()

            with recorder.patch():
                for expected in range(1, 8):
                    self.assertEqual(
                        adapter.capture_sample("fixture-owner"),
                        expected,
                    )

            pending = adapter._pending_sessions["fixture-owner"]
            registered_session = adapter.manager.list_users()[0].session_ids[0]
            adapter.revoke("fixture-owner")

            self.assertEqual(adapter.manager.list_users(), ())
            self.assertEqual(adapter.manager.load_training_dataset().sessions, ())
            self.assertTrue(load_manifest(pending.directory).revoked)
            self.assertTrue(
                load_manifest(
                    Path(adapter.manager.recordings_root) / registered_session
                ).revoked
            )
            with self.assertRaises(PermissionError):
                adapter.capture_sample("fixture-owner")

    def test_restart_recovers_consented_partial_only_after_fresh_consent(self) -> None:
        with TemporaryDirectory() as temporary:
            first = VoiceGuardDesktopAdapter(temporary)
            first.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()
            with recorder.patch():
                self.assertEqual(first.capture_sample("fixture-owner"), 1)
                self.assertEqual(first.capture_sample("fixture-owner"), 2)
            original_session = first._pending_sessions["fixture-owner"].manifest.session_id

            restarted = VoiceGuardDesktopAdapter(temporary)
            with self.assertRaises(PermissionError):
                restarted.capture_sample("fixture-owner")
            self.assertEqual(
                restarted.begin_user("fixture-owner", "owner", consent=True),
                2,
            )
            self.assertEqual(
                restarted._pending_sessions["fixture-owner"].manifest.session_id,
                original_session,
            )
            with recorder.patch():
                self.assertEqual(restarted.capture_sample("fixture-owner"), 3)
                self.assertEqual(restarted.capture_sample("fixture-owner"), 4)
                self.assertEqual(restarted.capture_sample("fixture-owner"), 5)
                self.assertEqual(restarted.capture_sample("fixture-owner"), 6)

            self.assertEqual(len(tuple(restarted.manager.recordings_root.iterdir())), 1)
            self.assertEqual(len(restarted.manager.load_training_dataset().sessions), 1)

    def test_restart_finalizes_persisted_complete_session_without_seventh_clip(self) -> None:
        with TemporaryDirectory() as temporary:
            first = VoiceGuardDesktopAdapter(temporary)
            first.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()
            with recorder.patch(), patch.object(
                first.manager,
                "finalize_enrollment",
                side_effect=EnrollmentError("fixture finalization interruption"),
            ):
                for expected in range(1, 6):
                    self.assertEqual(first.capture_sample("fixture-owner"), expected)
                with self.assertRaisesRegex(EnrollmentError, "finalization interruption"):
                    first.capture_sample("fixture-owner")
            self.assertEqual(recorder.calls, 6)

            restarted = VoiceGuardDesktopAdapter(temporary)
            self.assertEqual(
                restarted.begin_user("fixture-owner", "owner", consent=True),
                6,
            )
            with recorder.patch():
                self.assertEqual(restarted.capture_sample("fixture-owner"), 6)
            self.assertEqual(recorder.calls, 6)
            self.assertEqual(len(restarted.manager.load_training_dataset().sessions), 1)

    def test_restart_surfaces_partial_to_training_and_revocation(self) -> None:
        with TemporaryDirectory() as temporary:
            first = VoiceGuardDesktopAdapter(temporary)
            first.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()
            with recorder.patch():
                first.capture_sample("fixture-owner")
                first.capture_sample("fixture-owner")
            partial_path = first._pending_sessions["fixture-owner"].directory

            restarted = VoiceGuardDesktopAdapter(temporary)
            with self.assertRaisesRegex(
                DatasetError,
                r"unfinished consented collection has 2/6 clips.*resume",
            ):
                restarted.train("fixture-owner")
            restarted.revoke("fixture-owner")
            self.assertTrue(load_manifest(partial_path).revoked)
            self.assertEqual(restarted.manager.list_users(include_revoked=True), ())

    def test_multiple_partial_sessions_fail_closed_without_registration(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(temporary)
            for token in (31, 32):
                session = adapter.manager.begin_enrollment(
                    "fixture-owner",
                    role="owner",
                    environment="desktop_ui",
                    consent=True,
                )
                session.add_pcm16([token, -token] * 200, contains_wake_phrase=False)

            with self.assertRaisesRegex(DatasetError, "Multiple unfinished"):
                adapter.begin_user("fixture-owner", "owner", consent=True)
            self.assertEqual(adapter.manager.list_users(), ())
            self.assertEqual(adapter._pending_sessions, {})

    def test_desktop_requires_later_same_environment_session(self) -> None:
        with TemporaryDirectory() as temporary:
            now = datetime.now(UTC)
            clock_value = [now]
            adapter = VoiceGuardDesktopAdapter(
                temporary,
                minimum_session_gap_seconds=1_800,
                clock=lambda: clock_value[0],
            )
            adapter.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()
            with recorder.patch():
                for _index in range(6):
                    adapter.capture_sample("fixture-owner")
                with self.assertRaisesRegex(
                    DatasetError,
                    r"later.*new environment.*does not prove physical independence",
                ):
                    adapter.capture_sample("fixture-owner")
                clock_value[0] = now + timedelta(minutes=31)
                self.assertEqual(adapter.capture_sample("fixture-owner"), 7)

            self.assertEqual(len(adapter.manager.list_users()[0].session_ids), 1)
            self.assertEqual(
                len(load_dataset(adapter.manager.recordings_root).sessions),
                2,
            )

    def test_completed_but_sparse_data_points_to_guided_readiness(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(temporary)
            adapter.begin_user("fixture-owner", "owner", consent=True)
            recorder = _MicrophoneRecorder()
            with recorder.patch():
                for _index in range(6):
                    adapter.capture_sample("fixture-owner")

            with self.assertRaisesRegex(
                DatasetError,
                r"Complete guided multi-clip sessions.*readiness check.*"
                r"Finalized clips=6, sessions=1",
            ):
                adapter.train("fixture-owner")


if __name__ == "__main__":
    unittest.main()
