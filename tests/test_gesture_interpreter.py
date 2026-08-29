from __future__ import annotations

from dataclasses import fields
import math
import unittest

import asher.ui.gesture_interpreter as gesture_module
from asher.ui.gesture_interpreter import (
    GestureInterpreter,
    GestureOutput,
    HandGestureFrame,
)


_DT = 1.0 / 60.0


def _hand(
    label: str,
    *,
    center: tuple[float, float] = (0.5, 0.4),
    pinch_ratio: float = 0.20,
) -> HandGestureFrame:
    """Build a synthetic 21-landmark hand with a 0.20 hand-size."""

    center_x, center_y = center
    landmarks: list[tuple[float, float, float]] = [
        (center_x, center_y, 0.0) for _ in range(21)
    ]
    landmarks[0] = (center_x, center_y + 0.30, 0.0)
    landmarks[9] = (center_x, center_y + 0.10, 0.0)
    pinch_distance = 0.20 * pinch_ratio
    landmarks[4] = (center_x - (pinch_distance * 0.5), center_y, 0.0)
    landmarks[8] = (center_x + (pinch_distance * 0.5), center_y, 0.0)
    return HandGestureFrame(label=label, landmarks=landmarks)


def _repeat(
    interpreter: GestureInterpreter,
    hands: list[HandGestureFrame],
    count: int,
    *,
    dt: float = _DT,
) -> GestureOutput:
    output = interpreter.process(hands, dt)
    for _ in range(count - 1):
        output = interpreter.process(hands, dt)
    return output


class GestureInterpreterTests(unittest.TestCase):
    def test_pinch_ratio_uses_hand_size_and_has_hysteresis(self) -> None:
        interpreter = GestureInterpreter()

        output = interpreter.process([_hand("Left", pinch_ratio=0.31)], _DT)
        self.assertEqual(output.mode, "rotate")
        self.assertEqual(output.pinched_labels, ("Left",))

        # Inside the gap, an active pinch stays active.
        output = interpreter.process([_hand("Left", pinch_ratio=0.40)], _DT)
        self.assertEqual(output.pinched_labels, ("Left",))

        output = interpreter.process([_hand("Left", pinch_ratio=0.46)], _DT)
        self.assertEqual(output.mode, "idle")
        self.assertEqual(output.pinched_labels, ())

        # Inside the same gap, an inactive pinch stays inactive.
        output = interpreter.process([_hand("Left", pinch_ratio=0.40)], _DT)
        self.assertEqual(output.pinched_labels, ())
        output = interpreter.process([_hand("Left", pinch_ratio=0.31)], _DT)
        self.assertEqual(output.pinched_labels, ("Left",))

        # Scaling every relevant distance preserves the ratio decision.
        small = _hand("Small", center=(0.25, 0.25), pinch_ratio=0.31)
        points = list(small.landmarks)
        origin = (0.25, 0.25)
        scaled = [
            (
                origin[0] + ((point[0] - origin[0]) * 0.5),
                origin[1] + ((point[1] - origin[1]) * 0.5),
                point[2] * 0.5,
            )
            for point in points
        ]
        scaled_output = GestureInterpreter().process(
            [HandGestureFrame("Small", scaled)],
            _DT,
        )
        self.assertEqual(scaled_output.pinched_labels, ("Small",))

    def test_unpinched_hand_never_drives_motion(self) -> None:
        interpreter = GestureInterpreter()
        for center in ((0.15, 0.25), (0.80, 0.60), (0.30, 0.35)):
            output = interpreter.process(
                [_hand("Left", center=center, pinch_ratio=0.60)],
                _DT,
            )
            self.assertEqual(output.mode, "idle")
            self.assertEqual(output.rotation_delta, (0.0, 0.0))
            self.assertEqual(
                (output.rotation_velocity_x, output.rotation_velocity_y),
                (0.0, 0.0),
            )
            self.assertEqual(output.gesture_expansion, 0.0)

    def test_one_hand_rotation_mirrors_x_and_reanchors_on_entry(self) -> None:
        interpreter = GestureInterpreter()
        first = interpreter.process(
            [_hand("Left", center=(0.30, 0.35), pinch_ratio=0.20)],
            _DT,
        )
        self.assertEqual(first.rotation_delta, (0.0, 0.0))

        horizontal = interpreter.process(
            [_hand("Left", center=(0.58, 0.35), pinch_ratio=0.20)],
            _DT,
        )
        # Camera X increased, but the interaction midpoint is mirrored.
        self.assertLess(horizontal.rotation_delta_y, 0.0)
        self.assertAlmostEqual(horizontal.rotation_delta_x, 0.0)

        vertical = interpreter.process(
            [_hand("Left", center=(0.58, 0.62), pinch_ratio=0.20)],
            _DT,
        )
        self.assertGreater(vertical.rotation_delta_x, 0.0)

    def test_dead_zone_rejects_jitter_and_large_motion_is_bounded(self) -> None:
        interpreter = GestureInterpreter()
        interpreter.process([_hand("Left", center=(0.50, 0.40))], _DT)
        for offset in (0.002, -0.002, 0.0015, -0.0015, 0.0):
            output = interpreter.process(
                [_hand("Left", center=(0.50 + offset, 0.40 - offset))],
                _DT,
            )
            self.assertEqual(output.rotation_delta, (0.0, 0.0))

        for index in range(20):
            center = (0.10, 0.20) if index % 2 else (0.90, 0.65)
            output = interpreter.process([_hand("Left", center=center)], 0.8)
            self.assertLessEqual(
                abs(output.rotation_delta_x),
                interpreter.max_rotation_delta,
            )
            self.assertLessEqual(
                abs(output.rotation_delta_y),
                interpreter.max_rotation_delta,
            )
            self.assertLessEqual(
                abs(output.rotation_velocity_x),
                interpreter.max_rotation_velocity,
            )
            self.assertLessEqual(
                abs(output.rotation_velocity_y),
                interpreter.max_rotation_velocity,
            )

    def test_rotation_release_is_smooth_and_damped(self) -> None:
        interpreter = GestureInterpreter()
        interpreter.process([_hand("Left", center=(0.25, 0.40))], _DT)
        moving = interpreter.process([_hand("Left", center=(0.75, 0.40))], _DT)
        self.assertNotEqual(moving.rotation_delta_y, 0.0)

        released = interpreter.process(
            [_hand("Left", center=(0.75, 0.40), pinch_ratio=0.60)],
            _DT,
        )
        self.assertEqual(released.mode, "idle")
        self.assertNotEqual(released.rotation_delta_y, 0.0)
        self.assertLess(abs(released.rotation_delta_y), abs(moving.rotation_delta_y))

        previous = abs(released.rotation_delta_y)
        for _ in range(180):
            released = interpreter.process([], _DT)
            self.assertLessEqual(abs(released.rotation_delta_y), previous + 1.0e-12)
            previous = abs(released.rotation_delta_y)
        self.assertAlmostEqual(released.rotation_delta_y, 0.0, places=6)

    def test_two_hand_spread_and_close_is_smoothed_and_bounded(self) -> None:
        interpreter = GestureInterpreter()
        compact = [
            _hand("Left", center=(0.35, 0.40)),
            _hand("Right", center=(0.65, 0.40)),
        ]
        baseline = interpreter.process(compact, _DT)
        self.assertEqual(baseline.mode, "expand")
        self.assertEqual(baseline.gesture_expansion, 0.0)
        self.assertEqual(baseline.rotation_delta, (0.0, 0.0))

        spread = [
            _hand("Left", center=(0.08, 0.40)),
            _hand("Right", center=(0.92, 0.40)),
        ]
        expanded = _repeat(interpreter, spread, 45)
        self.assertGreater(expanded.gesture_expansion, 0.85)
        self.assertLessEqual(expanded.gesture_expansion, 1.0)
        self.assertEqual(expanded.rotation_delta, (0.0, 0.0))

        closing_first = interpreter.process(compact, _DT)
        self.assertLessEqual(closing_first.gesture_expansion, 1.0)
        reformed = _repeat(interpreter, compact, 90)
        self.assertLess(reformed.gesture_expansion, 0.01)
        self.assertGreaterEqual(reformed.gesture_expansion, 0.0)

    def test_release_returns_to_zero_and_repinch_forms_a_new_baseline(self) -> None:
        interpreter = GestureInterpreter()
        initial = [
            _hand("Left", center=(0.35, 0.40)),
            _hand("Right", center=(0.65, 0.40)),
        ]
        interpreter.process(initial, _DT)
        far = [
            _hand("Left", center=(0.20, 0.40)),
            _hand("Right", center=(0.80, 0.40)),
        ]
        expanded = _repeat(interpreter, far, 35)
        self.assertGreater(expanded.expansion, 0.60)

        one_hand = interpreter.process([far[0]], _DT)
        self.assertLess(one_hand.expansion, expanded.expansion)

        # Re-forming both pinches at their current distance establishes that
        # distance as zero, so stale expansion cannot kick back upward.
        reformed_pair = interpreter.process(far, _DT)
        self.assertLess(reformed_pair.expansion, one_hand.expansion)
        settling = _repeat(interpreter, far, 30)
        self.assertLess(settling.expansion, reformed_pair.expansion)

        farther = [
            _hand("Left", center=(0.05, 0.40)),
            _hand("Right", center=(0.95, 0.40)),
        ]
        newly_expanded = _repeat(interpreter, farther, 35)
        self.assertGreater(newly_expanded.expansion, settling.expansion)
        self.assertLessEqual(newly_expanded.expansion, 1.0)

        returned = _repeat(interpreter, [], 180)
        self.assertAlmostEqual(returned.expansion, 0.0, places=6)

    def test_stable_labels_make_record_reordering_deterministic(self) -> None:
        first = GestureInterpreter()
        second = GestureInterpreter()
        frames = [
            [
                _hand("Left", center=(0.35, 0.40)),
                _hand("Right", center=(0.65, 0.40)),
            ],
            [
                _hand("Left", center=(0.20, 0.42)),
                _hand("Right", center=(0.80, 0.38)),
            ],
            [
                _hand("Left", center=(0.10, 0.44)),
                _hand("Right", center=(0.90, 0.36)),
            ],
        ]
        for frame in frames:
            normal = first.process(frame, _DT)
            reversed_order = second.process(list(reversed(frame)), _DT)
            self.assertEqual(normal, reversed_order)
        self.assertEqual(normal.pinched_labels, ("Left", "Right"))

    def test_invalid_input_is_ignored_and_output_remains_finite(self) -> None:
        interpreter = GestureInterpreter()
        invalid_landmarks = [(0.5, 0.5, 0.0)] * 21
        invalid_landmarks[4] = (math.nan, 0.5, 0.0)
        zero_size = [(0.5, 0.5, 0.0)] * 21
        samples = (
            None,
            42,
            "not hands",
            [{"label": "Left", "landmarks": [(0.1, 0.2)]}],
            [HandGestureFrame("Left", invalid_landmarks)],
            [HandGestureFrame("Left", zero_size)],
            [{"label": "", "landmarks": [(0.5, 0.5)] * 21}],
        )
        for sample in samples:
            output = interpreter.process(sample, float("nan"))  # type: ignore[arg-type]
            self.assertEqual(output.mode, "idle")
            self.assertEqual(output.pinched_labels, ())
            for value in (
                output.rotation_delta_x,
                output.rotation_delta_y,
                output.rotation_velocity_x,
                output.rotation_velocity_y,
                output.gesture_expansion,
            ):
                self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(output.gesture_expansion, 0.0)
            self.assertLessEqual(output.gesture_expansion, 1.0)

        # Mapping landmarks and tuple records are accepted conveniences for a
        # thin camera/QWebChannel host.
        mapped = [
            {"x": point[0], "y": point[1], "z": point[2]}
            for point in _hand("Mapped").landmarks
        ]
        output = interpreter.process([("Mapped", mapped)], _DT)
        self.assertEqual(output.pinched_labels, ("Mapped",))
        self.assertEqual(output.to_dict()["gestureExpansion"], output.expansion)

        output = interpreter.process(("Mapped", mapped), dt_seconds=_DT)
        self.assertEqual(output.pinched_hands, ("Mapped",))

    def test_mode_changes_and_reset_do_not_emit_jumps(self) -> None:
        interpreter = GestureInterpreter()
        interpreter.process([_hand("Left", center=(0.20, 0.40))], _DT)
        interpreter.process([_hand("Left", center=(0.80, 0.40))], _DT)

        pair = [
            _hand("Left", center=(0.80, 0.40)),
            _hand("Right", center=(0.25, 0.40)),
        ]
        expanding = interpreter.process(pair, _DT)
        self.assertEqual(expanding.mode, "expand")
        self.assertEqual(expanding.gesture_expansion, 0.0)

        back_to_one = interpreter.process([pair[0]], _DT)
        self.assertEqual(back_to_one.mode, "rotate")
        self.assertEqual(back_to_one.rotation_delta, (0.0, 0.0))

        interpreter.reset()
        reset_output = interpreter.process([pair[0]], _DT)
        self.assertEqual(reset_output.rotation_delta, (0.0, 0.0))
        self.assertEqual(reset_output.gesture_expansion, 0.0)

    def test_public_surface_contains_only_presentation_math(self) -> None:
        self.assertEqual(
            set(gesture_module.__all__),
            {"GestureInterpreter", "GestureOutput", "HandGestureFrame"},
        )
        public_methods = {
            name
            for name, value in vars(GestureInterpreter).items()
            if not name.startswith("_") and callable(value)
        }
        self.assertEqual(public_methods, {"process", "reset"})
        self.assertEqual(
            {field.name for field in fields(GestureOutput)},
            {
                "mode",
                "rotation_x",
                "rotation_y",
                "expansion",
                "pinched_hands",
                "rotation_velocity_x",
                "rotation_velocity_y",
                "tracked_hands",
            },
        )


if __name__ == "__main__":
    unittest.main()
