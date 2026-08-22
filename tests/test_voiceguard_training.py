"""Session-split training, model persistence, provenance, and lifecycle tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from asher.voiceguard import (
    CalibratedVoiceGuardModel,
    EnrollmentManager,
    FeatureExample,
    PretrainedEmbeddingAdapter,
    SampleCondition,
    SampleOrigin,
    StatisticalFeatureExtractor,
    TrainingConfig,
    VoiceGuardTrainer,
    load_dataset,
    train_from_feature_examples,
)
from asher.voiceguard.dataset import session_separated_split
from asher.voiceguard.exceptions import DatasetError


def _fixture_examples() -> tuple[FeatureExample, ...]:
    rows: list[FeatureExample] = []
    for label, center, authorized in (("owner", 2.0, True), ("trusted", 1.0, True), ("unknown", -2.0, False)):
        for session_number in range(6):
            session = f"{label}-session-{session_number}"
            for clip in range(2):
                drift = session_number * 0.005
                rows.append(
                    FeatureExample(
                        sample_id=f"{session}-{clip}",
                        session_id=session,
                        label=label,
                        features=(center + drift, center - drift, center),
                        expected_authorized=authorized,
                    )
                )
    return tuple(rows)


def _augmented_condition_examples(*, include_real_noisy: bool) -> tuple[FeatureExample, ...]:
    rows: list[FeatureExample] = []
    for label, center, authorized in (
        ("owner-id", 2.0, True),
        ("unknown-id", -2.0, False),
    ):
        for session_number in range(3):
            session = f"{label}-source-session-{session_number}"
            drift = session_number * 0.01
            rows.append(
                FeatureExample(
                    sample_id=f"{session}-clean",
                    session_id=session,
                    label=label,
                    features=(center + drift, center - drift),
                    expected_authorized=authorized,
                    condition=SampleCondition.CLEAN.value,
                    origin=SampleOrigin.RECORDED.value,
                )
            )
            rows.append(
                FeatureExample(
                    sample_id=f"{session}-synthetic-noisy",
                    session_id=session,
                    label=label,
                    features=(center + drift + 0.02, center - drift - 0.02),
                    expected_authorized=authorized,
                    condition=SampleCondition.NOISY.value,
                    origin=SampleOrigin.AUGMENTED.value,
                    source_sample_id=f"{session}-clean",
                )
            )
            if include_real_noisy:
                rows.append(
                    FeatureExample(
                        sample_id=f"{session}-real-noisy",
                        session_id=session,
                        label=label,
                        features=(center + drift + 0.01, center - drift - 0.01),
                        expected_authorized=authorized,
                        condition=SampleCondition.NOISY.value,
                        origin=SampleOrigin.IMPORTED.value,
                    )
                )
    return tuple(rows)


class VoiceGuardTrainingTests(unittest.TestCase):
    def test_speaker_verification_authorizes_only_the_global_predicted_identity(self) -> None:
        model = CalibratedVoiceGuardModel(
            task="speaker_auth",
            classes=("owner-id", "trusted-id", "unknown-id"),
            coefficients=((0.0,), (0.0,), (0.0,)),
            intercepts=(0.0, 0.0, 0.0),
            feature_mean=(0.0,),
            feature_scale=(1.0,),
            threshold=0.25,
            authorized_labels=("owner-id", "trusted-id"),
            extractor_metadata=StatisticalFeatureExtractor.metadata,
        )
        with patch.object(
            CalibratedVoiceGuardModel,
            "predict_proba",
            return_value={"owner-id": 0.30, "trusted-id": 0.30, "unknown-id": 0.40},
        ):
            result = model.verify_features((0.0,))
        self.assertFalse(result.accepted)
        self.assertEqual(result.predicted_label, "unknown")
        self.assertEqual(result.score, 0.0)

    def test_direct_training_rejects_an_all_authorized_speaker_class_set(self) -> None:
        examples = tuple(
            item for item in _fixture_examples() if item.label in {"owner", "trusted"}
        )
        with self.assertRaisesRegex(DatasetError, "unauthorized identity class"):
            train_from_feature_examples(
                examples,
                config=TrainingConfig(authorized_labels=("owner", "trusted")),
            )

    def test_replay_trials_are_excluded_from_classifier_fit(self) -> None:
        examples = list(_fixture_examples())
        for item in tuple(examples):
            if item.sample_id.endswith("-0"):
                examples.append(
                    FeatureExample(
                        sample_id=f"{item.sample_id}-replay",
                        session_id=item.session_id,
                        label=item.label,
                        features=tuple(value * -20 for value in item.features),
                        expected_authorized=False,
                        condition=SampleCondition.REPLAY.value,
                    )
                )
        result = train_from_feature_examples(
            examples,
            config=TrainingConfig(authorized_labels=("owner", "trusted"), seed=9),
        )
        expected_fit_count = sum(
            item.condition != SampleCondition.REPLAY.value for item in result.split.train
        )
        self.assertEqual(
            result.model.metadata["split"]["fit_train_sample_count"],
            expected_fit_count,
        )
        self.assertGreater(
            result.model.metadata["split"]["excluded_replay_train_sample_count"],
            0,
        )

    def test_augmented_feature_must_preserve_its_explicit_source_labels(self) -> None:
        examples = list(_fixture_examples())
        source = examples[0]
        examples.append(
            FeatureExample(
                sample_id="mislabeled-derivative",
                session_id=source.session_id,
                label="trusted",
                features=source.features,
                expected_authorized=False,
                condition=SampleCondition.NOISY.value,
                origin=SampleOrigin.AUGMENTED.value,
                source_sample_id=source.sample_id,
            )
        )
        with self.assertRaisesRegex(DatasetError, "preserve its source"):
            train_from_feature_examples(examples)

    def test_ml_dependency_probe_works_in_a_clean_lazy_process(self) -> None:
        code = (
            "import sys; "
            "from asher.voiceguard import ml_dependencies_available; "
            "result = ml_dependencies_available(); "
            "print(isinstance(result, tuple) and len(result) == 2); "
            "print(any(name in sys.modules for name in ('numpy', 'sklearn')))"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.splitlines(), ["True", "False"])

    def test_session_split_is_deterministic_and_disjoint(self) -> None:
        examples = _fixture_examples()
        first = session_separated_split(examples, seed=17)
        second = session_separated_split(examples, seed=17)
        first.assert_session_separated()
        self.assertEqual(first, second)
        self.assertTrue(first.train_sessions)
        self.assertTrue(first.validation_sessions)
        self.assertTrue(first.test_sessions)

    def test_training_calibrates_and_round_trips_json_model(self) -> None:
        result = train_from_feature_examples(
            _fixture_examples(),
            config=TrainingConfig(seed=7, minimum_training_samples=4),
        )
        self.assertTrue(result.model.metadata["classifier"]["student_trained"])
        self.assertFalse(result.model.extractor_metadata.is_pretrained)
        self.assertTrue(result.test_report.measured)
        with TemporaryDirectory() as temporary:
            path = result.model.save(Path(temporary) / "voiceguard.json")
            loaded = result.model.load(path)
            self.assertEqual(loaded.model_version, result.model.model_version)
            self.assertEqual(
                loaded.verify_features((2.0, 2.0, 2.0)).accepted,
                result.model.verify_features((2.0, 2.0, 2.0)).accepted,
            )

    def test_augmented_noisy_examples_are_train_only_and_never_measured_as_real(self) -> None:
        config = TrainingConfig(
            authorized_labels=("owner-id",),
            seed=4,
            minimum_training_samples=4,
        )
        synthetic_only = train_from_feature_examples(
            _augmented_condition_examples(include_real_noisy=False),
            config=config,
        )
        synthetic_only.split.assert_session_separated()
        self.assertTrue(any(item.is_augmented for item in synthetic_only.split.train))
        self.assertTrue(
            all(not item.is_augmented for item in synthetic_only.split.validation)
        )
        self.assertTrue(all(not item.is_augmented for item in synthetic_only.split.test))
        self.assertIn(SampleCondition.NOISY.value, synthetic_only.validation_report.unavailable_conditions)
        self.assertIn(SampleCondition.NOISY.value, synthetic_only.test_report.unavailable_conditions)
        self.assertTrue(synthetic_only.model.metadata["split"]["held_out_source_only"])
        self.assertGreater(
            synthetic_only.model.metadata["split"]["augmented_train_sample_count"],
            0,
        )

        real_noisy = train_from_feature_examples(
            _augmented_condition_examples(include_real_noisy=True),
            config=config,
        )
        self.assertNotIn(SampleCondition.NOISY.value, real_noisy.test_report.unavailable_conditions)
        self.assertIn(SampleCondition.NOISY.value, real_noisy.test_report.condition_metrics)
        self.assertTrue(all(not item.is_augmented for item in real_noisy.split.test))

    def test_pretrained_adapter_is_explicitly_not_student_trained(self) -> None:
        adapter = PretrainedEmbeddingAdapter(
            lambda path: (0.1, 0.2, 0.3),
            extractor_id="fixture.pretrained",
            display_name="Fixture external embedding",
            model_version="fixture-1",
            source="fixture://external-model",
        )
        self.assertTrue(adapter.metadata.is_pretrained)
        self.assertFalse(adapter.metadata.is_student_trained)
        self.assertEqual(adapter.metadata.provenance, "pretrained_external")

    def test_enrollment_revoke_excludes_sessions_from_future_dataset(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(temporary)
            session = manager.begin_enrollment(
                "fixture-owner",
                role="owner",
                environment="test-room",
                consent=True,
            )
            session.add_pcm16([1000, -1000] * 800, contains_wake_phrase=True)
            manager.finalize_enrollment(session)
            self.assertEqual(len(load_dataset(manager.recordings_root).sessions), 1)
            result = manager.revoke_user("fixture-owner")
            self.assertFalse(result.recordings_deleted)
            self.assertEqual(len(load_dataset(manager.recordings_root).sessions), 0)
            self.assertEqual(len(manager.list_users()), 0)
            self.assertEqual(len(manager.list_users(include_revoked=True)), 1)

    def test_training_rejects_too_few_sessions_instead_of_fabricating_metrics(self) -> None:
        examples = tuple(
            FeatureExample(f"s{i}", f"only-session-{i}", "owner", (1.0, 1.0), True)
            for i in range(2)
        )
        with self.assertRaises(DatasetError):
            VoiceGuardTrainer().train_examples(examples)


if __name__ == "__main__":
    unittest.main()
