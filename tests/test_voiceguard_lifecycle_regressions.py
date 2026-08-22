"""Regression tests for VoiceGuard lifecycle concurrency and provenance."""

from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from asher.ui.voiceguard_adapter import VoiceGuardDesktopAdapter
from asher.voice.runtime import load_active_voiceguard_verifier
from asher.voiceguard import (
    DatasetError,
    CalibratedVoiceGuardModel,
    EnrollmentError,
    EnrollmentManager,
    FeatureExample,
    ManifestError,
    RecordingSession,
    TrainingConfig,
    load_manifest,
)
from asher.voiceguard.dataset import session_separated_split
from asher.voiceguard.readiness import independent_dataset_view
from tests.test_voiceguard_readiness import (
    _StaticTrainer,
    _fixture_training_result,
    _pcm,
    _populate_ready_manager,
    _stamp_training_result,
)


def _record_without_hardware(
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
    token = 1_700 + len(session.manifest.samples)
    return session.add_pcm16(
        [token, -token] * 200,
        sample_rate=sample_rate,
        channels=channels,
        contains_wake_phrase=contains_wake_phrase,
        condition=condition,
        expected_authorized=expected_authorized,
        sample_id=sample_id,
    )


class VoiceGuardLifecycleRegressionTests(unittest.TestCase):
    def test_direct_partial_revoke_cannot_bypass_peer_consent_invalidation(self) -> None:
        with TemporaryDirectory() as temporary:
            first = VoiceGuardDesktopAdapter(temporary, samples_per_session=6)
            peer = VoiceGuardDesktopAdapter(temporary, samples_per_session=6)
            first.begin_user("fixture-owner", "owner", consent=True)
            peer.begin_user("fixture-owner", "owner", consent=True)

            with patch.object(
                RecordingSession,
                "record_microphone",
                new=_record_without_hardware,
            ):
                self.assertEqual(first.capture_sample("fixture-owner"), 1)
            partial = first._pending_sessions["fixture-owner"]
            with self.assertRaisesRegex(ManifestError, "EnrollmentManager"):
                partial.revoke()

            first.manager.revoke_unregistered_session(partial)
            self.assertTrue(load_manifest(partial.directory).revoked)
            with patch.object(
                RecordingSession,
                "record_microphone",
                new=_record_without_hardware,
            ):
                with self.assertRaisesRegex(PermissionError, "changed"):
                    peer.capture_sample("fixture-owner")

            self.assertEqual(
                tuple(
                    path
                    for path in first.manager.recordings_root.iterdir()
                    if path.is_dir()
                ),
                (partial.directory,),
            )

    def test_runtime_rejects_swapped_in_memory_model_during_disk_aba(self) -> None:
        with TemporaryDirectory() as temporary:
            runtime_root = Path(temporary)
            manager = EnrollmentManager(runtime_root / "voiceguard")
            _populate_ready_manager(manager)
            trained = manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))
            assert trained.artifacts is not None
            model_path = trained.artifacts.model_path
            original_bytes = model_path.read_bytes()
            original_model = CalibratedVoiceGuardModel.load(model_path)
            malicious_model = replace(
                original_model,
                coefficients=((0.0,),),
                intercepts=(60.0,),
            )
            self.assertEqual(
                malicious_model.model_version,
                original_model.model_version,
            )

            actors = (
                SimpleNamespace(
                    user_id="private-owner-97",
                    role=SimpleNamespace(value="owner"),
                ),
            )
            controller = SimpleNamespace(
                config=SimpleNamespace(runtime=SimpleNamespace(root=runtime_root)),
                users=SimpleNamespace(list_active=lambda: actors),
            )
            verifier = load_active_voiceguard_verifier(controller)
            self.assertIsNotNone(verifier)
            assert verifier is not None

            class RestoringExtractor:
                metadata = original_model.extractor_metadata

                def __init__(self) -> None:
                    self.calls = 0

                def extract_wav(self, _path):
                    self.calls += 1
                    model_path.write_bytes(original_bytes)
                    return (1.0,)

            extractor = RestoringExtractor()
            verifier.extractor = extractor
            actual_binding_check = verifier._binding_is_current
            binding_checks = 0

            def swap_after_precheck() -> bool:
                nonlocal binding_checks
                current = actual_binding_check()
                if binding_checks == 0 and current:
                    malicious_model.save(model_path)
                binding_checks += 1
                return current

            try:
                with patch.object(
                    verifier,
                    "_binding_is_current",
                    side_effect=swap_after_precheck,
                ):
                    user_id, score, reason = verifier.authenticate(
                        b"\0\0" * 40,
                        16_000,
                    )
            finally:
                model_path.write_bytes(original_bytes)

            self.assertIsNone(user_id)
            self.assertEqual(score, 0.0)
            self.assertIn("guest access only", reason)
            self.assertEqual(extractor.calls, 0)
            self.assertIsNone(verifier._model)

    def test_peer_train_after_partial_revoke_clears_stale_consent(self) -> None:
        with TemporaryDirectory() as temporary:
            seed = VoiceGuardDesktopAdapter(temporary)
            partial = seed.manager.begin_enrollment(
                "fixture-owner",
                role="owner",
                environment="desktop_ui",
                consent=True,
            )
            partial.add_pcm16(_pcm(77), contains_wake_phrase=False)

            revoker = VoiceGuardDesktopAdapter(temporary)
            peer = VoiceGuardDesktopAdapter(temporary)
            revoker.begin_user("fixture-owner", "owner", consent=True)
            peer.begin_user("fixture-owner", "owner", consent=True)
            revoker.revoke("fixture-owner")

            registry = json.loads(
                peer.manager.registry_path.read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(registry["lifecycle_generation"], 1)
            with self.assertRaises(DatasetError):
                peer.train("fixture-owner")
            self.assertNotIn("fixture-owner", peer._consented)
            with self.assertRaisesRegex(PermissionError, "consent"):
                peer.capture_sample("fixture-owner")

            self.assertTrue(load_manifest(partial.directory).revoked)
            self.assertEqual(peer.manager.list_users(), ())
            self.assertEqual(
                tuple(path for path in peer.manager.recordings_root.iterdir() if path.is_dir()),
                (partial.directory,),
            )

    def test_two_adapters_merge_partial_captures_and_stale_copy_cannot_undo_revoke(self) -> None:
        with TemporaryDirectory() as temporary:
            seed = VoiceGuardDesktopAdapter(temporary, samples_per_session=6)
            partial = seed.manager.begin_enrollment(
                "fixture-owner",
                role="owner",
                environment="desktop_ui",
                consent=True,
            )
            partial.add_pcm16(_pcm(1), contains_wake_phrase=False)

            first = VoiceGuardDesktopAdapter(temporary, samples_per_session=6)
            second = VoiceGuardDesktopAdapter(temporary, samples_per_session=6)
            first.begin_user("fixture-owner", "owner", consent=True)
            second.begin_user("fixture-owner", "owner", consent=True)

            with patch.object(
                RecordingSession,
                "record_microphone",
                new=_record_without_hardware,
            ), ThreadPoolExecutor(max_workers=2) as executor:
                counts = tuple(
                    executor.map(
                        lambda adapter: adapter.capture_sample("fixture-owner"),
                        (first, second),
                    )
                )

            self.assertEqual(sorted(counts), [2, 3])
            self.assertEqual(len(load_manifest(partial.directory).samples), 3)

            first.revoke("fixture-owner")
            self.assertTrue(load_manifest(partial.directory).revoked)
            with patch.object(
                RecordingSession,
                "record_microphone",
                new=_record_without_hardware,
            ):
                with self.assertRaisesRegex(PermissionError, "revoked|changed"):
                    second.capture_sample("fixture-owner")
            self.assertTrue(load_manifest(partial.directory).revoked)
            self.assertEqual(second.manager.list_users(), ())

    def test_finalized_session_is_immutable_and_runtime_detects_manual_manifest_change(self) -> None:
        with TemporaryDirectory() as temporary:
            runtime_root = Path(temporary)
            manager = EnrollmentManager(runtime_root / "voiceguard")
            _populate_ready_manager(manager)
            trained = manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))
            self.assertIsNotNone(trained.artifacts)

            record = manager.list_users()[0]
            session = RecordingSession.open(manager.recordings_root / record.session_ids[0])
            sample_id = session.manifest.samples[0].sample_id
            self.assertTrue(session.manifest.finalized)
            with self.assertRaisesRegex(ManifestError, "finalized"):
                session.add_pcm16(_pcm(9_001), contains_wake_phrase=False)
            with self.assertRaisesRegex(ManifestError, "finalized"):
                session.remove_sample(sample_id)

            actors = (
                SimpleNamespace(
                    user_id="private-owner-97",
                    role=SimpleNamespace(value="owner"),
                ),
            )
            controller = SimpleNamespace(
                config=SimpleNamespace(runtime=SimpleNamespace(root=runtime_root)),
                users=SimpleNamespace(list_active=lambda: actors),
            )
            verifier = load_active_voiceguard_verifier(controller)
            self.assertIsNotNone(verifier)
            assert verifier is not None
            self.assertTrue(verifier._binding_is_current())

            manifest_path = session.manifest_path
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["environment"] = "manually changed after activation"
            manifest_path.write_text(json.dumps(value), encoding="utf-8")

            self.assertFalse(verifier._binding_is_current())
            self.assertIsNone(load_active_voiceguard_verifier(controller))

    def test_manual_unseal_is_resealed_but_invalidates_active_runtime_binding(self) -> None:
        with TemporaryDirectory() as temporary:
            runtime_root = Path(temporary)
            manager = EnrollmentManager(runtime_root / "voiceguard")
            _populate_ready_manager(manager)
            manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))
            actors = (
                SimpleNamespace(
                    user_id="private-owner-97",
                    role=SimpleNamespace(value="owner"),
                ),
            )
            controller = SimpleNamespace(
                config=SimpleNamespace(runtime=SimpleNamespace(root=runtime_root)),
                users=SimpleNamespace(list_active=lambda: actors),
            )
            verifier = load_active_voiceguard_verifier(controller)
            self.assertIsNotNone(verifier)
            assert verifier is not None
            self.assertTrue(verifier._binding_is_current())

            session_id = manager.list_users()[0].session_ids[0]
            path = manager.recordings_root / session_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            original_finalized_at = value["finalized_at"]
            value["finalized_at"] = None
            path.write_text(json.dumps(value), encoding="utf-8")

            self.assertFalse(verifier._binding_is_current())
            resealed = load_manifest(path)
            self.assertTrue(resealed.finalized)
            self.assertNotEqual(resealed.finalized_at, original_finalized_at)
            self.assertIsNone(load_active_voiceguard_verifier(controller))

    def test_direct_revoke_cannot_enter_between_finalization_seal_and_registry_write(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            session = manager.begin_enrollment(
                "fixture-owner",
                role="owner",
                environment="fixture-room",
                consent=True,
            )
            for token in (901, 902, 903):
                session.add_pcm16(_pcm(token), contains_wake_phrase=False)

            entered_registry_write = threading.Event()
            release_registry_write = threading.Event()
            original_write = manager._write_registry

            def blocking_write(*args, **kwargs):
                entered_registry_write.set()
                if not release_registry_write.wait(5):
                    raise RuntimeError("fixture registry-write release timed out")
                return original_write(*args, **kwargs)

            with patch.object(manager, "_write_registry", side_effect=blocking_write):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    finalized = executor.submit(
                        manager.finalize_enrollment,
                        session,
                        minimum_samples=3,
                    )
                    self.assertTrue(entered_registry_write.wait(5))
                    try:
                        with self.assertRaisesRegex(ManifestError, "EnrollmentManager"):
                            RecordingSession.open(session.directory).revoke()
                    finally:
                        release_registry_write.set()
                    record = finalized.result(timeout=10)

            self.assertTrue(record.active)
            self.assertTrue(load_manifest(session.directory).finalized)
            self.assertFalse(load_manifest(session.directory).revoked)
            manager.revoke_user("fixture-owner")
            self.assertTrue(load_manifest(session.directory).revoked)

    def test_adapter_can_revoke_sealed_orphan_after_registry_write_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = VoiceGuardDesktopAdapter(
                temporary,
                samples_per_session=3,
            )
            session = adapter.manager.begin_enrollment(
                "fixture-owner",
                role="owner",
                environment="desktop_ui",
                consent=True,
            )
            for token in (1_301, 1_302, 1_303):
                session.add_pcm16(_pcm(token), contains_wake_phrase=False)

            with patch.object(
                adapter.manager,
                "_write_registry",
                side_effect=OSError("fixture registry persistence failure"),
            ):
                with self.assertRaises(OSError):
                    adapter.manager.finalize_enrollment(session, minimum_samples=3)

            self.assertTrue(load_manifest(session.directory).finalized)
            self.assertFalse(load_manifest(session.directory).revoked)
            self.assertEqual(adapter.manager.list_users(), ())

            adapter.revoke("fixture-owner")
            self.assertTrue(load_manifest(session.directory).revoked)
            self.assertEqual(adapter.manager.list_users(), ())

    def test_correlated_extras_are_excluded_from_readiness_and_actual_training_view(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            base = datetime(2026, 1, 1, tzinfo=UTC)
            rapid_by_identity: dict[str, set[str]] = {}
            identities = (
                ("private-owner-97", "owner"),
                ("private-outsider-42", "unknown"),
            )
            for identity_index, (user_id, role) in enumerate(identities):
                rapid_by_identity[user_id] = set()
                for session_index in range(5):
                    rapid = session_index < 3
                    session = manager.begin_enrollment(
                        user_id,
                        role=role,
                        environment=(
                            "shared-room"
                            if rapid
                            else f"independent-room-{session_index}"
                        ),
                        consent=True,
                    )
                    if rapid:
                        rapid_by_identity[user_id].add(session.manifest.session_id)
                    for clip_index in range(3):
                        token = identity_index * 1_000 + session_index * 10 + clip_index
                        session.add_pcm16(
                            _pcm(token),
                            contains_wake_phrase=False,
                        )
                    manager.finalize_enrollment(session, minimum_samples=3)

                    path = session.manifest_path
                    value = json.loads(path.read_text(encoding="utf-8"))
                    when = (
                        base + timedelta(minutes=session_index * 5 if rapid else 0)
                    ).isoformat()
                    value["consent_given_at"] = when
                    value["created_at"] = when
                    for sample in value["samples"]:
                        sample["created_at"] = when
                    path.write_text(json.dumps(value), encoding="utf-8")

            readiness = manager.assess_training_readiness(
                config=TrainingConfig(seed=0)
            )
            self.assertTrue(readiness.ready, readiness.to_dict())
            self.assertEqual(readiness.minimum_observed_sessions_per_class, 3)

            full = manager.load_training_dataset()
            independent = independent_dataset_view(full)
            selected_ids = set(independent.session_ids)
            for rapid_ids in rapid_by_identity.values():
                self.assertEqual(len(rapid_ids & selected_ids), 1)

            examples = tuple(
                FeatureExample(
                    sample.record.sample_id,
                    sample.session_id,
                    sample.speaker_id,
                    (0.0,),
                    sample.role == "owner",
                )
                for sample in independent.samples
            )
            split = session_separated_split(examples, seed=0)
            partitions = (
                split.train_sessions,
                split.validation_sessions,
                split.test_sessions,
            )
            for rapid_ids in rapid_by_identity.values():
                self.assertLessEqual(
                    sum(bool(rapid_ids & partition) for partition in partitions),
                    1,
                )

            class CapturingTrainer:
                def __init__(self) -> None:
                    self.session_ids: frozenset[str] = frozenset()

                def train_dataset(self, dataset, config=None):
                    del config
                    self.session_ids = frozenset(dataset.session_ids)
                    return _stamp_training_result(
                        _fixture_training_result(),
                        dataset,
                    )

            trainer = CapturingTrainer()
            manager.retrain(trainer=trainer, config=TrainingConfig(seed=0))
            self.assertEqual(trainer.session_ids, frozenset(selected_ids))

    def test_reentrant_revocation_during_fit_finishes_and_cas_rejects_result(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)

            class ReentrantTrainer:
                def train_dataset(self, dataset, config=None):
                    del config
                    result = _stamp_training_result(
                        _fixture_training_result(),
                        dataset,
                    )
                    manager.revoke_user("private-outsider-42")
                    return result

            with self.assertRaisesRegex(EnrollmentError, "changed during training"):
                manager.retrain(trainer=ReentrantTrainer())
            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)
            self.assertEqual(tuple(manager.models_root.glob("*.json")), ())

    def test_manager_rejects_model_without_exact_independent_dataset_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)

            with self.assertRaisesRegex(EnrollmentError, "dataset provenance"):
                manager.retrain(
                    trainer=_StaticTrainer(
                        _fixture_training_result(),
                        stamp_dataset=False,
                    )
                )
            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)
            self.assertEqual(tuple(manager.models_root.glob("*.json")), ())


if __name__ == "__main__":
    unittest.main()
