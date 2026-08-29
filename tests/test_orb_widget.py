from __future__ import annotations

import math
import os
import unittest

from asher.core.state import AssistantState
from asher.ui import is_available
from asher.ui.orb_widget import (
    _CINEMATIC_SHELL_SEGMENTS,
    _CINEMATIC_SPARK_COUNT,
    _CINEMATIC_TRAJECTORY_COUNT,
    _CINEMATIC_WISP_COUNT,
    audio_activity_for_level,
    cinematic_scale_for_activity,
    visual_for_state,
)


class OrbVisualContractTests(unittest.TestCase):
    def test_every_canonical_and_legacy_state_has_a_visual_profile(self) -> None:
        for state in AssistantState:
            visual = visual_for_state(state)
            self.assertTrue(visual.primary.startswith("#"), state)
            self.assertTrue(visual.secondary.startswith("#"), state)
            self.assertGreaterEqual(visual.particles, 0, state)
            self.assertGreater(visual.speed, 0.0, state)

    def test_stopped_and_locked_do_not_emit_particles(self) -> None:
        self.assertEqual(visual_for_state(AssistantState.STOPPED).particles, 0)
        self.assertEqual(visual_for_state(AssistantState.LOCKED).particles, 0)

    def test_cinematic_complexity_budgets_are_fixed_and_small(self) -> None:
        self.assertEqual(_CINEMATIC_SHELL_SEGMENTS, 192)
        self.assertEqual(_CINEMATIC_SPARK_COUNT, 11)
        self.assertEqual(_CINEMATIC_TRAJECTORY_COUNT, 3)
        self.assertEqual(_CINEMATIC_WISP_COUNT, 7)

    def test_microphone_level_maps_to_bounded_perceptual_activity(self) -> None:
        self.assertEqual(audio_activity_for_level(-1.0), 0.0)
        self.assertEqual(audio_activity_for_level(0.008), 0.0)
        self.assertAlmostEqual(audio_activity_for_level(0.094), math.sqrt(0.5))
        self.assertEqual(audio_activity_for_level(0.18), 1.0)
        self.assertEqual(audio_activity_for_level(1.0), 1.0)
        for invalid in (None, "not-a-level", float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                self.assertEqual(audio_activity_for_level(invalid), 0.0)

    def test_cinematic_scale_is_bounded_and_eased(self) -> None:
        quiet = cinematic_scale_for_activity(0.0)
        midpoint = cinematic_scale_for_activity(0.5)
        voice = cinematic_scale_for_activity(1.0)

        self.assertAlmostEqual(quiet, 0.70)
        self.assertAlmostEqual(midpoint, 0.84)
        self.assertAlmostEqual(voice, 0.98)
        self.assertLess(quiet, midpoint)
        self.assertLess(midpoint, voice)
        self.assertEqual(cinematic_scale_for_activity(-1.0), quiet)
        self.assertEqual(cinematic_scale_for_activity(2.0), voice)


@unittest.skipUnless(is_available(), "PySide6 is optional; install it for the offscreen orb smoke test")
class OrbQtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-orb-test"])

    def test_orb_renders_real_states_without_changing_them(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        orb.resize(420, 420)
        orb.show()
        for state in (
            AssistantState.STANDBY,
            AssistantState.THINKING,
            AssistantState.AWAITING_CONFIRMATION,
            AssistantState.EXECUTING,
            AssistantState.SPEAKING,
            AssistantState.SUCCESS,
            AssistantState.ERROR,
            AssistantState.STOPPED,
        ):
            orb.set_state(state)
            self.app.processEvents()
            self.assertEqual(orb.state, state)
            self.assertFalse(orb.grab().isNull())
        orb.close()


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(is_available(), "PySide6 is optional; install it for orb smoke tests")
class CinematicOrbContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-cinematic-orb-test"])

    def test_cinematic_renderer_does_not_change_assistant_state(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        orb.set_state(AssistantState.THINKING)
        before = orb.state
        orb.set_cinematic_mode(True)
        orb.set_overlay_text("THINKING", "Planning a safe action")
        orb.resize(690, 690)
        orb.show()
        self.app.processEvents()
        self.assertIs(orb.state, before)
        self.assertTrue(orb._cinematic_mode)
        orb.close()

    def test_cinematic_renderer_uses_large_bounded_display(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        orb.set_cinematic_mode(True)
        orb.set_interactive_resize(True, initial_size=760)
        self.assertEqual(orb.width(), 760)
        self.assertEqual(orb.set_display_size(1200), 900)
        self.assertEqual(orb.set_display_size(100), 340)
        orb.close()

    def test_painted_voice_scale_is_about_forty_percent_larger_than_quiet(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        try:
            orb._display_activity = 0.0
            quiet = orb.current_visual_scale()
            orb._display_activity = 1.0
            voice = orb.current_visual_scale()

            self.assertAlmostEqual(voice / quiet, 1.4, places=2)
            self.assertAlmostEqual(quiet, cinematic_scale_for_activity(0.0))
            self.assertAlmostEqual(voice, cinematic_scale_for_activity(1.0))
        finally:
            orb.close()

    def test_activity_envelope_has_fast_attack_and_slow_release(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        try:
            orb.resize(420, 420)
            orb.show()
            self.app.processEvents()
            orb._timer.stop()
            orb.set_state(AssistantState.LISTENING)
            orb.set_audio_level(0.18)
            orb._timer.stop()
            orb._display_activity = 0.0

            attacked = orb._advance_activity(0.075)
            orb.set_audio_level(0.0)
            orb._timer.stop()
            released = orb._advance_activity(0.075)

            attack_change = attacked
            release_change = attacked - released
            self.assertAlmostEqual(attacked, 1.0 - math.exp(-1.0))
            self.assertAlmostEqual(released, attacked * math.exp(-0.25))
            self.assertGreater(attack_change, release_change * 4.0)
        finally:
            orb.close()

    def test_unrefreshed_microphone_sample_expires_and_releases(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        try:
            orb.resize(420, 420)
            orb.show()
            self.app.processEvents()
            orb._timer.stop()
            orb.set_state(AssistantState.LISTENING)
            orb.set_audio_level(0.18)
            orb._timer.stop()
            orb._display_activity = 1.0
            orb._audio_sample_deadline = 0.0

            samples = [orb._audio_level]
            for _index in range(5):
                samples.append(orb._decay_audio_sample(1.0, 0.16))

            self.assertEqual(samples[-1], 0.0)
            self.assertTrue(
                all(later <= earlier for earlier, later in zip(samples, samples[1:]))
            )
            orb._advance_activity(0.30)
            self.assertLess(orb._display_activity, 1.0)
        finally:
            orb.close()

    def test_speaking_expands_without_fabricated_microphone_amplitude(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        try:
            orb.resize(420, 420)
            orb.show()
            self.app.processEvents()
            orb._timer.stop()
            orb.set_audio_level(0.0)
            orb.set_state(AssistantState.SPEAKING)
            orb._timer.stop()
            orb._display_activity = 0.0
            quiet = orb.current_visual_scale()

            self.assertEqual(orb._audio_level, 0.0)
            self.assertGreater(orb._target_activity(), 0.8)
            orb._advance_activity(0.15)
            self.assertGreater(orb.current_visual_scale(), quiet)
        finally:
            orb.close()

    def test_accessibility_description_tracks_the_truthful_state(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        try:
            self.assertEqual(orb.accessibleName(), "Asher voice orb")
            orb.set_state(AssistantState.ERROR)
            description = orb.accessibleDescription().casefold()
            self.assertIn("error", description)
            self.assertIn(
                visual_for_state(AssistantState.ERROR).label.casefold(),
                description,
            )
        finally:
            orb.close()

    def _render_at_fixed_phase(self, state: AssistantState):
        from PySide6.QtGui import QImage

        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        orb.set_cinematic_mode(True)
        orb.set_reduced_motion(True)
        orb.setFixedSize(700, 700)
        orb.set_state(state)
        orb._phase = 1.234
        orb.show()
        self.app.processEvents()
        image = orb.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        return orb, image

    def test_cinematic_render_is_deterministic_bright_blue_glass_across_states(self) -> None:
        blue_orb, blue = self._render_at_fixed_phase(AssistantState.LISTENING)
        try:
            repeated = blue_orb.grab().toImage().convertToFormat(blue.format())
            self.assertEqual(blue.constBits().tobytes(), repeated.constBits().tobytes())
            self.assertIs(blue_orb.state, AssistantState.LISTENING)

            centre = blue.pixelColor(350, 350)
            self.assertGreater(sum(centre.getRgb()[:3]) / 3.0, 175.0)

            # The fallback's narrow plasma corona follows the actual frozen
            # cinematic geometry: a 0.345 canvas radius with the mesh at 82%.
            # Sample through a small turbulence band without weakening the
            # brightness contract now that WebGL is the primary renderer.
            shell_radius = (
                700.0
                * 0.345
                * 0.82
                * blue_orb.current_visual_scale()
            )
            shell_values = []
            for degree in range(0, 360, 3):
                angle = math.radians(degree)
                samples = []
                for offset in range(-15, 16, 3):
                    x = round(350 + math.cos(angle) * (shell_radius + offset))
                    y = round(350 + math.sin(angle) * (shell_radius + offset))
                    samples.append(sum(blue.pixelColor(x, y).getRgb()[:3]) / 3.0)
                shell_values.append(max(samples))
            self.assertGreater(sum(shell_values) / len(shell_values), 100.0)
        finally:
            blue_orb.close()

        executing_orb, executing = self._render_at_fixed_phase(AssistantState.EXECUTING)
        stopped_orb, stopped = self._render_at_fixed_phase(AssistantState.STOPPED)
        try:
            blue_channels = [0, 0, 0]
            executing_channels = [0, 0, 0]
            active_bright = 0
            stopped_bright = 0
            for y in range(0, 700, 8):
                for x in range(0, 700, 8):
                    blue_pixel = blue.pixelColor(x, y)
                    executing_pixel = executing.pixelColor(x, y)
                    stopped_pixel = stopped.pixelColor(x, y)
                    for index, value in enumerate(blue_pixel.getRgb()[:3]):
                        blue_channels[index] += value
                    for index, value in enumerate(executing_pixel.getRgb()[:3]):
                        executing_channels[index] += value
                    active_bright += max(blue_pixel.getRgb()[:3]) > 100
                    stopped_bright += max(stopped_pixel.getRgb()[:3]) > 100

            self.assertGreater(blue_channels[2], blue_channels[0] * 2.0)
            self.assertGreater(executing_channels[2], executing_channels[0] * 2.0)
            self.assertGreater(active_bright, stopped_bright * 10)
        finally:
            executing_orb.close()
            stopped_orb.close()

    def test_cinematic_text_font_has_real_latin_glyphs(self) -> None:
        from PySide6.QtGui import QFont, QRawFont

        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        try:
            raw_font = QRawFont.fromFont(QFont(orb._font_family))
            self.assertTrue(raw_font.isValid())
            self.assertTrue(raw_font.supportsCharacter("L"))
        finally:
            orb.close()
