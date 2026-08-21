"""Deterministic metric tests; no claims are made for absent conditions."""

from __future__ import annotations

import unittest

from asher.voiceguard import (
    EvaluationObservation,
    SampleCondition,
    evaluate_predictions,
)


class VoiceGuardMetricTests(unittest.TestCase):
    def test_confusion_f1_far_frr_and_replay_are_measured_from_inputs(self) -> None:
        observations = (
            EvaluationObservation("a", "owner", 0.95, "owner", True),
            EvaluationObservation("b", "unknown", 0.10, "unknown", False),
            EvaluationObservation("c", "owner", 0.20, "unknown", True),
            EvaluationObservation("d", "unknown", 0.85, "owner", False),
            EvaluationObservation("r", "owner", 0.60, "owner", False, SampleCondition.REPLAY.value),
        )
        report = evaluate_predictions(observations, threshold=0.5)
        self.assertEqual(report.binary_confusion_matrix, ((1, 2), (1, 1)))
        self.assertEqual(report.sample_count, 5)
        self.assertAlmostEqual(report.false_accept_rate, 2 / 3)
        self.assertAlmostEqual(report.false_reject_rate, 0.5)
        self.assertAlmostEqual(report.f1, 0.4)
        self.assertIn(SampleCondition.REPLAY.value, report.condition_metrics)
        self.assertAlmostEqual(report.replay_acceptance_rate, 1.0)

    def test_empty_and_missing_conditions_are_explicitly_unavailable(self) -> None:
        report = evaluate_predictions((), threshold=0.5)
        self.assertFalse(report.measured)
        self.assertIsNone(report.f1)
        self.assertEqual(set(report.unavailable_conditions), {"clean", "noisy", "replay"})
        self.assertTrue(any("No evaluation recordings" in note for note in report.notes))


if __name__ == "__main__":
    unittest.main()
