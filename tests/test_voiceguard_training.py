"""Session-split training, model persistence, provenance, and lifecycle tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asher.voiceguard import (
    EnrollmentManager,
    FeatureExample,
    PretrainedEmbeddingAdapter,
    SampleCondition,
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


class VoiceGuardTrainingTests(unittest.TestCase):
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
