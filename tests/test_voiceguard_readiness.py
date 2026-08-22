"""Readiness and enrollment-boundary tests for the VoiceGuard training lifecycle."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from asher.voiceguard import (
    CalibratedVoiceGuardModel,
    DatasetError,
    DatasetSplit,
    EnrollmentError,
    EnrollmentManager,
    EvaluationObservation,
    ReadinessPolicy,
    RecordingSession,
    SampleCondition,
    SampleOrigin,
    StatisticalFeatureExtractor,
    TrainingConfig,
    TrainingResult,
    VoiceGuardTrainer,
    assess_dataset_readiness,
    evaluate_predictions,
    feature_examples_from_dataset,
    load_dataset,
    model_content_fingerprint,
    train_speaker_classifier,
)
from asher.voice.runtime import FileVoiceGuardVerifier, load_active_voiceguard_verifier


def _pcm(token: int) -> tuple[int, ...]:
    """Return a small deterministic signal whose WAV digest is token-specific."""

    value = 400 + token
    return tuple([value, -value] * 64 + [value + 1, -(value + 1)])


def _issue_codes(readiness: object) -> set[str]:
    return {issue.code for issue in readiness.issues}


def _populate_ready_manager(
    manager: EnrollmentManager,
    *,
    duplicate_within_one_session: bool = False,
    duplicate_across_sessions: bool = False,
    wake_word_classes: bool = False,
) -> None:
    identities = (
        ("private-owner-97", "owner"),
        ("private-outsider-42", "unknown"),
    )
    for identity_index, (user_id, role) in enumerate(identities):
        for session_index in range(3):
            session = manager.begin_enrollment(
                user_id,
                role=role,
                environment=f"fixture-room-{session_index}",
                consent=True,
            )
            for clip_index in range(3):
                token = identity_index * 100 + session_index * 10 + clip_index
                if duplicate_within_one_session and identity_index == 0 and session_index == 0:
                    token = 0
                if duplicate_across_sessions and identity_index == 1 and session_index == 0 and clip_index == 0:
                    token = 0
                session.add_pcm16(
                    _pcm(token),
                    contains_wake_phrase=wake_word_classes and identity_index == 0,
                    sample_id=f"clip-{identity_index}-{session_index}-{clip_index}",
                )
            manager.finalize_enrollment(session, minimum_samples=3)


class _FixtureExtractor:
    metadata = StatisticalFeatureExtractor.metadata

    def extract_wav(self, path: str | Path) -> tuple[float, ...]:
        # Training is mocked in the identity-label test. The extractor exists
        # only to exercise normal VoiceDataset preparation without importing ML.
        return (float(Path(path).stat().st_size),)


def _stamp_training_result(result: TrainingResult, dataset: object) -> TrainingResult:
    metadata = dict(result.model.metadata)
    dataset_metadata = dict(metadata.get("dataset", {}))
    dataset_metadata["fingerprint"] = dataset.fingerprint
    metadata["dataset"] = dataset_metadata
    metadata.pop("model_version", None)
    unversioned = replace(result.model, metadata=metadata)
    metadata["model_version"] = model_content_fingerprint(unversioned)
    return replace(
        result,
        model=replace(unversioned, metadata=metadata),
    )


class _StaticTrainer:
    def __init__(self, result: TrainingResult, *, stamp_dataset: bool = True) -> None:
        self.result = result
        self.calls = 0
        self.stamp_dataset = stamp_dataset

    def train_dataset(self, dataset: object, config: object = None) -> TrainingResult:
        del config
        self.calls += 1
        return (
            _stamp_training_result(self.result, dataset)
            if self.stamp_dataset
            else self.result
        )


class _FailingReport:
    def save(self, path: str | Path) -> Path:
        raise OSError("fixture report persistence failure")

    def to_dict(self) -> dict[str, object]:
        raise OSError("fixture report persistence failure")


def _fixture_training_result(*, failing_validation_report: bool = False) -> TrainingResult:
    unversioned = CalibratedVoiceGuardModel(
        task="speaker_auth",
        classes=("private-owner-97", "private-outsider-42"),
        coefficients=((1.0,),),
        intercepts=(0.0,),
        feature_mean=(0.0,),
        feature_scale=(1.0,),
        threshold=0.5,
        authorized_labels=("private-owner-97",),
        extractor_metadata=StatisticalFeatureExtractor.metadata,
        binary_positive_class="private-owner-97",
        metadata={},
    )
    model = replace(
        unversioned,
        metadata={"model_version": model_content_fingerprint(unversioned)},
    )
    report = evaluate_predictions(
        (
            EvaluationObservation("positive", "private-owner-97", 0.9, "private-owner-97", True),
            EvaluationObservation("negative", "private-outsider-42", 0.1, "unknown", False),
        ),
        threshold=0.5,
    )
    return TrainingResult(
        model=model,
        split=DatasetSplit(train=(), validation=(), test=()),
        validation_report=_FailingReport() if failing_validation_report else report,
        test_report=report,
    )


def _fixture_wake_training_result() -> TrainingResult:
    unversioned = CalibratedVoiceGuardModel(
        task="wake_word",
        classes=("wake_negative", "wake_positive"),
        coefficients=((1.0,),),
        intercepts=(0.0,),
        feature_mean=(0.0,),
        feature_scale=(1.0,),
        threshold=0.5,
        authorized_labels=("wake_positive",),
        extractor_metadata=StatisticalFeatureExtractor.metadata,
        binary_positive_class="wake_positive",
        metadata={},
    )
    model = replace(
        unversioned,
        metadata={"model_version": model_content_fingerprint(unversioned)},
    )
    report = evaluate_predictions(
        (
            EvaluationObservation("positive", "wake_positive", 0.9, "wake_positive", True),
            EvaluationObservation("negative", "wake_negative", 0.1, "unknown", False),
        ),
        threshold=0.5,
    )
    return TrainingResult(
        model=model,
        split=DatasetSplit(train=(), validation=(), test=()),
        validation_report=report,
        test_report=report,
    )


class VoiceGuardReadinessTests(unittest.TestCase):
    def test_readiness_import_keeps_optional_audio_and_ml_dependencies_lazy(self) -> None:
        code = (
            "import sys; "
            "from asher.voiceguard import ReadinessPolicy, assess_dataset_readiness; "
            "ReadinessPolicy(); "
            "print(any(name in sys.modules for name in "
            "('sounddevice','speechbrain','torch','numpy','sklearn')))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False")

    def test_manager_training_dataset_ignores_unfinalized_on_disk_session(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            finalized = manager.begin_enrollment(
                "registered-owner",
                role="owner",
                environment="fixture-room",
                consent=True,
            )
            finalized.add_pcm16(_pcm(1), contains_wake_phrase=False)
            manager.finalize_enrollment(finalized)

            unfinalized = manager.begin_enrollment(
                "unregistered-speaker",
                role="unknown",
                environment="fixture-room",
                consent=True,
            )
            unfinalized.add_pcm16(_pcm(2), contains_wake_phrase=False)

            self.assertEqual(len(load_dataset(manager.recordings_root).sessions), 2)
            training_dataset = manager.load_training_dataset()
            self.assertEqual(training_dataset.session_ids, (finalized.manifest.session_id,))
            self.assertEqual(
                {sample.speaker_id for sample in training_dataset.samples},
                {"registered-owner"},
            )

    def test_finalize_rejects_foreign_manager_session(self) -> None:
        with TemporaryDirectory() as temporary:
            first = EnrollmentManager(Path(temporary) / "first")
            second = EnrollmentManager(Path(temporary) / "second")
            session = first.begin_enrollment(
                "private-owner",
                role="owner",
                environment="fixture-room",
                consent=True,
            )
            session.add_pcm16(_pcm(3), contains_wake_phrase=False)

            with self.assertRaises(EnrollmentError):
                second.finalize_enrollment(session)
            self.assertEqual(second.list_users(), ())

    def test_active_identity_cannot_silently_change_role(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            original = manager.begin_enrollment(
                "stable-identity",
                role="owner",
                environment="fixture-room",
                consent=True,
            )
            original.add_pcm16(_pcm(4), contains_wake_phrase=False)
            manager.finalize_enrollment(original)

            with self.assertRaises(EnrollmentError):
                changed = manager.begin_enrollment(
                    "stable-identity",
                    role="trusted",
                    environment="fixture-room-2",
                    consent=True,
                )
                changed.add_pcm16(_pcm(5), contains_wake_phrase=False)
                manager.finalize_enrollment(changed)

            record = manager.list_users()[0]
            self.assertEqual(record.role, "owner")
            self.assertEqual(record.session_ids, (original.manifest.session_id,))

    def test_training_dataset_rejects_manifest_registry_role_conflict(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            session = manager.begin_enrollment(
                "stable-identity",
                role="owner",
                environment="fixture-room",
                consent=True,
            )
            session.add_pcm16(_pcm(5), contains_wake_phrase=False)
            manager.finalize_enrollment(session)

            manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
            manifest["role"] = "trusted"
            session.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(DatasetError):
                manager.load_training_dataset()

    def test_sparse_readiness_has_exact_codes_and_privacy_safe_dict(self) -> None:
        with TemporaryDirectory() as temporary:
            recordings = Path(temporary) / "private-recordings-root"
            session = RecordingSession.create(
                recordings,
                speaker_id="secret-speaker-identity",
                role="owner",
                environment="secret-room-name",
                consent=True,
                session_id="secret-session-identity",
            )
            sample = session.add_pcm16(
                _pcm(6),
                contains_wake_phrase=False,
                sample_id="secret-sample-identity",
            )
            readiness = assess_dataset_readiness(
                load_dataset(recordings),
                task="speaker_auth",
                authorized_labels=("secret-speaker-identity",),
                policy=ReadinessPolicy(
                    minimum_samples_per_session=3,
                    minimum_sessions_per_class=3,
                ),
                minimum_training_samples=4,
                seed=11,
            )

            self.assertFalse(readiness.ready)
            self.assertEqual(
                _issue_codes(readiness),
                {
                    "insufficient_samples",
                    "insufficient_classes",
                    "no_unauthorized_samples",
                    "no_unauthorized_identity_class",
                    "insufficient_samples_per_session",
                    "insufficient_independent_sessions_per_class",
                    "split_unavailable",
                },
            )

            public_value = readiness.to_dict()
            serialized = json.dumps(public_value, sort_keys=True)
            for secret in (
                "secret-speaker-identity",
                "secret-session-identity",
                "secret-sample-identity",
                "secret-room-name",
                recordings.name,
                sample.sha256,
            ):
                self.assertNotIn(secret, serialized)

            def all_keys(value: object) -> set[str]:
                if isinstance(value, dict):
                    return set(value) | set().union(*(all_keys(item) for item in value.values()))
                if isinstance(value, (list, tuple)):
                    return set().union(*(all_keys(item) for item in value))
                return set()

            self.assertTrue(
                {
                    "speaker_id",
                    "session_id",
                    "sample_id",
                    "path",
                    "sha256",
                    "digest",
                    "fingerprint",
                }.isdisjoint(all_keys(public_value))
            )

    def test_duplicate_audio_within_a_session_cannot_inflate_source_readiness(self) -> None:
        with TemporaryDirectory() as temporary:
            within = EnrollmentManager(Path(temporary) / "within")
            _populate_ready_manager(within, duplicate_within_one_session=True)
            within_readiness = within.assess_training_readiness()
            self.assertFalse(within_readiness.ready)
            self.assertIn("insufficient_samples_per_session", _issue_codes(within_readiness))
            self.assertNotIn("cross_session_duplicate_audio", _issue_codes(within_readiness))

            crossing = EnrollmentManager(Path(temporary) / "crossing")
            _populate_ready_manager(crossing, duplicate_across_sessions=True)
            crossing_readiness = crossing.assess_training_readiness()
            self.assertFalse(crossing_readiness.ready)
            self.assertIn("cross_session_duplicate_audio", _issue_codes(crossing_readiness))

            trainer = Mock()
            with self.assertRaises(DatasetError):
                crossing.retrain(trainer=trainer)
            trainer.train_dataset.assert_not_called()

    def test_replay_trials_cannot_replace_an_unauthorized_identity_class(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            for identity_index, (user_id, role) in enumerate(
                (("private-owner", "owner"), ("private-trusted", "trusted"))
            ):
                for session_index in range(3):
                    session = manager.begin_enrollment(
                        user_id,
                        role=role,
                        environment=f"fixture-room-{session_index}",
                        consent=True,
                    )
                    for clip_index in range(3):
                        replay = clip_index == 1
                        session.add_pcm16(
                            _pcm(800 + identity_index * 100 + session_index * 10 + clip_index),
                            contains_wake_phrase=False,
                            condition=(
                                SampleCondition.REPLAY if replay else SampleCondition.CLEAN
                            ),
                            expected_authorized=not replay,
                        )
                    manager.finalize_enrollment(session, minimum_samples=3)

            readiness = manager.assess_training_readiness()
            self.assertFalse(readiness.ready)
            self.assertEqual(readiness.unauthorized_class_count, 0)
            self.assertIn("no_unauthorized_identity_class", _issue_codes(readiness))
            trainer = Mock()
            with self.assertRaises(DatasetError):
                manager.retrain(trainer=trainer)
            trainer.train_dataset.assert_not_called()

    def test_augmented_clips_do_not_satisfy_source_readiness_or_noisy_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            for identity_index, (user_id, role) in enumerate(
                (("source-owner", "owner"), ("source-outsider", "unknown"))
            ):
                for session_index in range(3):
                    session = manager.begin_enrollment(
                        user_id,
                        role=role,
                        environment=f"source-room-{session_index}",
                        consent=True,
                    )
                    original = session.add_pcm16(
                        _pcm(500 + identity_index * 100 + session_index * 10),
                        contains_wake_phrase=False,
                    )
                    for derivative_index in range(2):
                        session.add_pcm16(
                            _pcm(
                                600
                                + identity_index * 100
                                + session_index * 10
                                + derivative_index
                            ),
                            contains_wake_phrase=False,
                            condition=SampleCondition.NOISY,
                            origin=SampleOrigin.AUGMENTED,
                            source_sample_id=original.sample_id,
                        )
                    manager.finalize_enrollment(session, minimum_samples=3)

            dataset = manager.load_training_dataset()
            prepared = feature_examples_from_dataset(
                dataset,
                _FixtureExtractor(),
                authorized_labels=("source-owner",),
            )
            self.assertEqual(sum(item.is_augmented for item in prepared), 12)

            readiness = manager.assess_training_readiness()
            self.assertFalse(readiness.ready)
            self.assertEqual(readiness.sample_count, 6)
            self.assertEqual(readiness.session_count, 6)
            self.assertIn("insufficient_samples_per_session", _issue_codes(readiness))
            self.assertIn(SampleCondition.NOISY.value, readiness.unavailable_conditions)

    def test_speaker_classifier_resolves_identity_labels_for_voice_dataset(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            dataset = manager.load_training_dataset()
            sentinel = object()

            with patch.object(
                VoiceGuardTrainer,
                "train_examples",
                autospec=True,
                return_value=sentinel,
            ) as train_examples:
                result = train_speaker_classifier(dataset, extractor=_FixtureExtractor())

            self.assertIs(result, sentinel)
            _, examples, config = train_examples.call_args.args
            self.assertEqual(
                {example.label for example in examples},
                {"private-owner-97", "private-outsider-42"},
            )
            self.assertEqual(config.authorized_labels, ("private-owner-97",))
            self.assertEqual(
                {
                    label: {item.expected_authorized for item in examples if item.label == label}
                    for label in {item.label for item in examples}
                },
                {
                    "private-owner-97": {True},
                    "private-outsider-42": {False},
                },
            )

    def test_retrain_persists_all_artifacts_before_registry_activation(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)

            failing = _StaticTrainer(_fixture_training_result(failing_validation_report=True))
            with self.assertRaises(EnrollmentError):
                manager.retrain(trainer=failing)
            failed_registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", failed_registry)

            trainer = _StaticTrainer(_fixture_training_result())
            result = manager.retrain(trainer=trainer)
            self.assertEqual(trainer.calls, 1)
            self.assertIsNotNone(result.artifacts)
            artifacts = result.artifacts
            assert artifacts is not None
            for artifact in (
                artifacts.model_path,
                artifacts.validation_report_path,
                artifacts.test_report_path,
            ):
                self.assertIsInstance(artifact, Path)
                self.assertTrue(artifact.is_file())
                self.assertIn(manager.root, artifact.resolve().parents)

            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            active_model = Path(registry["active_model"])
            if not active_model.is_absolute():
                active_model = manager.root / active_model
            self.assertEqual(active_model.resolve(), artifacts.model_path.resolve())

    def test_wake_training_never_replaces_active_speaker_model(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager, wake_word_classes=True)

            speaker = manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))
            assert speaker.artifacts is not None
            initial_registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            speaker_path = initial_registry["active_model"]

            wake = manager.retrain(
                trainer=_StaticTrainer(_fixture_wake_training_result()),
                config=TrainingConfig(task="wake_word"),
            )
            assert wake.artifacts is not None
            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["active_model"], speaker_path)
            self.assertEqual(
                Path(registry["wake_word_model"]).resolve(),
                wake.artifacts.model_path.resolve(),
            )
            self.assertNotEqual(registry["wake_word_model"], registry["active_model"])

    def test_wake_readiness_uses_phrase_labels_not_speaker_authorization_flags(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            for session_index in range(3):
                session = manager.begin_enrollment(
                    "private-owner",
                    role="owner",
                    environment=f"wake-room-{session_index}",
                    consent=True,
                )
                for clip_index in range(3):
                    session.add_pcm16(
                        _pcm(1200 + session_index * 10 + clip_index),
                        contains_wake_phrase=clip_index == 0,
                        expected_authorized=True,
                    )
                manager.finalize_enrollment(session, minimum_samples=3)

            readiness = manager.assess_training_readiness(
                config=TrainingConfig(task="wake_word")
            )
            self.assertTrue(readiness.ready, readiness.to_dict())
            self.assertNotIn("authorization_label_conflict", _issue_codes(readiness))

    def test_retrain_rejects_non_private_or_mutable_active_model_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(Path(temporary) / "private")
            _populate_ready_manager(manager)
            result = _fixture_training_result()

            outside = Path(temporary) / "shared" / "voiceguard.json"
            with self.assertRaises(EnrollmentError):
                manager.retrain(trainer=_StaticTrainer(result), model_path=outside)
            self.assertFalse(outside.exists())

            mutable = manager.models_root / "voiceguard.json"
            with self.assertRaisesRegex(
                EnrollmentError,
                r"voiceguard-[0-9a-f]{64}\.json",
            ):
                manager.retrain(trainer=_StaticTrainer(result), model_path=mutable)
            self.assertFalse(mutable.exists())
            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)

    def test_same_environment_sessions_require_time_separation(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            for identity_index, (user_id, role) in enumerate(
                (("private-owner", "owner"), ("private-outsider", "unknown"))
            ):
                for session_index in range(3):
                    session = manager.begin_enrollment(
                        user_id,
                        role=role,
                        environment="same fixture room",
                        consent=True,
                    )
                    for clip_index in range(3):
                        session.add_pcm16(
                            _pcm(1500 + identity_index * 100 + session_index * 10 + clip_index),
                            contains_wake_phrase=False,
                        )
                    manager.finalize_enrollment(session, minimum_samples=3)

            immediate = manager.assess_training_readiness()
            self.assertFalse(immediate.ready)
            self.assertIn(
                "insufficient_independent_sessions_per_class",
                _issue_codes(immediate),
            )
            self.assertEqual(immediate.minimum_observed_sessions_per_class, 1)

            base = datetime(2026, 1, 1, tzinfo=UTC)
            for record in manager.list_users():
                for index, session_id in enumerate(record.session_ids):
                    path = manager.recordings_root / session_id / "manifest.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    when = (base + timedelta(minutes=30 * index)).isoformat()
                    value["consent_given_at"] = when
                    value["created_at"] = when
                    for sample in value["samples"]:
                        sample["created_at"] = when
                    path.write_text(json.dumps(value), encoding="utf-8")

            separated = manager.assess_training_readiness()
            self.assertTrue(separated.ready, separated.to_dict())
            self.assertEqual(separated.minimum_observed_sessions_per_class, 3)

    def test_invalid_independence_timestamp_fails_with_aggregate_only_output(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            session_id = manager.list_users()[0].session_ids[0]
            path = manager.recordings_root / session_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            secret_timestamp = "2026-08-22T10:00:00"
            value["created_at"] = secret_timestamp
            path.write_text(json.dumps(value), encoding="utf-8")

            readiness = manager.assess_training_readiness()
            self.assertFalse(readiness.ready)
            self.assertIn(
                "invalid_session_independence_metadata",
                _issue_codes(readiness),
            )
            self.assertNotIn(secret_timestamp, json.dumps(readiness.to_dict()))

    def test_model_content_version_covers_security_relevant_fields(self) -> None:
        base = _fixture_training_result().model
        variants = (
            replace(base, feature_mean=(0.25,)),
            replace(base, feature_scale=(2.0,)),
            replace(base, threshold=0.75),
            replace(base, authorized_labels=("private-outsider-42",)),
            replace(base, binary_positive_class="private-outsider-42"),
            replace(
                base,
                metadata={
                    **base.metadata,
                    "dataset": {"fingerprint": "a" * 64},
                },
            ),
        )
        for variant in variants:
            self.assertNotEqual(
                model_content_fingerprint(base),
                model_content_fingerprint(variant),
            )

        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            stamped = _stamp_training_result(
                _fixture_training_result(),
                manager.load_training_dataset(),
            )
            stale_claim = replace(stamped.model, feature_scale=(2.0,))
            bad_result = replace(stamped, model=stale_claim)
            with self.assertRaisesRegex(EnrollmentError, "content version"):
                manager.retrain(
                    trainer=_StaticTrainer(bad_result, stamp_dataset=False)
                )
            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)

    def test_versioned_artifacts_are_immutable_on_retry(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            training_result = _fixture_training_result()
            completed = manager.retrain(trainer=_StaticTrainer(training_result))
            assert completed.artifacts is not None
            paths = (
                completed.artifacts.model_path,
                completed.artifacts.validation_report_path,
                completed.artifacts.test_report_path,
                manager.registry_path,
            )
            before = {path: path.read_bytes() for path in paths}

            with self.assertRaisesRegex(EnrollmentError, "refusing to overwrite"):
                manager.retrain(trainer=_StaticTrainer(training_result))

            self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_activation_failure_removes_new_artifact_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            result = _fixture_training_result()
            with patch.object(manager, "_write_registry", side_effect=OSError("fixture activation")):
                with self.assertRaisesRegex(EnrollmentError, "activate"):
                    manager.retrain(trainer=_StaticTrainer(result))
            version = result.model.model_version
            self.assertFalse((manager.models_root / f"voiceguard-{version}.json").exists())
            self.assertFalse(
                (manager.evaluations_root / f"voiceguard-{version}-validation.json").exists()
            )
            self.assertFalse(
                (manager.evaluations_root / f"voiceguard-{version}-test.json").exists()
            )

    def test_revocation_clears_bindings_and_preloaded_verifier_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(Path(temporary) / "voiceguard")
            _populate_ready_manager(manager, wake_word_classes=True)
            speaker = manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))
            manager.retrain(
                trainer=_StaticTrainer(_fixture_wake_training_result()),
                config=TrainingConfig(task="wake_word"),
            )
            assert speaker.artifacts is not None
            model_bytes = speaker.artifacts.model_path.read_bytes()
            verifier = FileVoiceGuardVerifier(
                speaker.artifacts.model_path,
                {"private-owner-97": "private-owner-97"},
                extractor=object(),
                registry_path=manager.registry_path,
                required_enrollments={"private-owner-97"},
            )

            manager.revoke_user("private-owner-97")

            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)
            self.assertNotIn("wake_word_model", registry)
            self.assertEqual(speaker.artifacts.model_path.read_bytes(), model_bytes)
            user_id, score, reason = verifier.authenticate(b"", 16_000)
            self.assertIsNone(user_id)
            self.assertEqual(score, 0.0)
            self.assertIn("guest access only", reason)

    def test_runtime_loader_requires_active_biometric_enrollment_for_authorized_identity(self) -> None:
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

            self.assertIsNotNone(load_active_voiceguard_verifier(controller))
            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            registry["users"]["private-owner-97"]["revoked_at"] = datetime.now(UTC).isoformat()
            manager.registry_path.write_text(json.dumps(registry), encoding="utf-8")
            self.assertIsNone(load_active_voiceguard_verifier(controller))

    def test_runtime_rechecks_model_binding_after_inference(self) -> None:
        with TemporaryDirectory() as temporary:
            verifier = FileVoiceGuardVerifier(
                Path(temporary) / "model.json",
                {"private-owner": "private-owner"},
                extractor=object(),
                temp_root=temporary,
            )
            verifier._model = SimpleNamespace(
                verify_wav=lambda _path, _extractor: SimpleNamespace(
                    accepted=True,
                    predicted_label="private-owner",
                    score=0.99,
                    reason="fixture accepted",
                )
            )
            with patch.object(
                verifier,
                "_binding_is_current",
                side_effect=(True, False),
            ):
                user_id, score, reason = verifier.authenticate(b"\0\0" * 40, 16_000)
            self.assertIsNone(user_id)
            self.assertEqual(score, 0.0)
            self.assertIn("guest access only", reason)

    def test_revocation_commits_fail_closed_before_manifest_cleanup(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))

            with patch.object(RecordingSession, "open", side_effect=OSError("fixture cleanup")):
                with self.assertRaisesRegex(EnrollmentError, "access was revoked"):
                    manager.revoke_user("private-owner-97")

            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)
            self.assertIsNotNone(registry["users"]["private-owner-97"]["revoked_at"])

    def test_retrain_and_revocation_are_serialized_without_model_resurrection(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer_manager = EnrollmentManager(root)
            revoker_manager = EnrollmentManager(root)
            _populate_ready_manager(trainer_manager)
            entered = threading.Event()
            release = threading.Event()
            result = _fixture_training_result()

            class BlockingTrainer:
                def train_dataset(self, dataset: object, config: object = None) -> TrainingResult:
                    del config
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("fixture release timed out")
                    return _stamp_training_result(result, dataset)

            with ThreadPoolExecutor(max_workers=2) as executor:
                training = executor.submit(trainer_manager.retrain, trainer=BlockingTrainer())
                self.assertTrue(entered.wait(5))
                revocation = executor.submit(revoker_manager.revoke_user, "private-owner-97")
                # Training runs outside the lifecycle lock, so revocation is
                # live even while the trainer remains blocked.
                revocation.result(timeout=10)
                release.set()
                with self.assertRaisesRegex(EnrollmentError, "changed during training"):
                    training.result(timeout=10)

            registry = json.loads(trainer_manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)
            self.assertIsNotNone(registry["users"]["private-owner-97"]["revoked_at"])

    def test_manifest_metadata_change_during_training_rejects_stale_activation(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            _populate_ready_manager(manager)
            entered = threading.Event()
            release = threading.Event()
            result = _fixture_training_result()

            class BlockingTrainer:
                def train_dataset(self, dataset: object, config: object = None) -> TrainingResult:
                    del config
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("fixture release timed out")
                    return _stamp_training_result(result, dataset)

            with ThreadPoolExecutor(max_workers=1) as executor:
                training = executor.submit(manager.retrain, trainer=BlockingTrainer())
                self.assertTrue(entered.wait(5))
                session_id = manager.list_users()[0].session_ids[0]
                manifest_path = manager.recordings_root / session_id / "manifest.json"
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                value["environment"] = "changed while training"
                manifest_path.write_text(json.dumps(value), encoding="utf-8")
                release.set()
                with self.assertRaisesRegex(EnrollmentError, "changed during training"):
                    training.result(timeout=10)

            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("active_model", registry)
            self.assertEqual(tuple(manager.models_root.glob("*.json")), ())


if __name__ == "__main__":
    unittest.main()
