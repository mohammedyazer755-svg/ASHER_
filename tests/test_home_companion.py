from pathlib import Path
import unittest
from unittest.mock import patch

from asher.core.state import AssistantState
from asher.ui.controller import DesktopController, DesktopSettings
from asher.ui.home_companion_widget import (
    FemaleCompanion,
    HOME_COMPANION_EXPERIMENTAL_3D_ENABLED,
    MaleCompanion,
    QT_AVAILABLE,
)
from asher.ui.web_orb_widget import CompanionOrbHost, WEB_ORB_INDEX
from asher.voice.tts import TTSManager, VoiceProfile, VoiceProfileRegistry

if QT_AVAILABLE:
    from asher.ui.home_companion_widget import (
        HomeCompanionFallbackWidget,
        HomeCompanionHost,
    )
    from PySide6.QtCore import QPoint, QThreadPool, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from asher.ui.window import AsherMainWindow

    _APP = QApplication.instance() or QApplication([])
else:
    _APP = None


def _controller() -> DesktopController:
    registry = VoiceProfileRegistry(
        (
            VoiceProfile("offline_male", "Male", "fake", gender_hint="male"),
            VoiceProfile("offline_female", "Female", "fake", gender_hint="female"),
        )
    )
    tts = TTSManager(registry, selected_profile="offline_male")
    tts.register_provider("fake", type("Provider", (), {"speak": lambda *_args: None, "stop": lambda *_args: None})())
    controller = DesktopController(tts=tts)
    controller.update_settings(voice_profile="offline_male", offline_only=False)
    return controller


@unittest.skipUnless(QT_AVAILABLE, "PySide6 required for Home Companion tests")
class HomeCompanionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _controller()

    def test_1_home_uses_home_companion_host(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            self.assertTrue(hasattr(window.home, "companion_host"))
            self.assertIsInstance(window.home.companion_host, HomeCompanionHost)
            # Backward compatibility alias
            self.assertIs(window.home.orb, window.home.companion_host)
        finally:
            window.close()

    def test_2_companion_mode_still_uses_companion_orb_host(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            self.assertTrue(hasattr(window.companion_mode, "orb"))
            self.assertIsInstance(window.companion_mode.orb, CompanionOrbHost)
            # Home companion and Companion mode orb must be completely distinct
            self.assertIsNot(window.home.companion_host, window.companion_mode.orb)
        finally:
            window.close()

    def test_3_companion_webgl_source_remains_untouched(self) -> None:
        self.assertTrue(WEB_ORB_INDEX.is_file())
        content = WEB_ORB_INDEX.read_text(encoding="utf-8")
        self.assertIn("ASHER Companion Orb", content)

    def test_4_male_appearance_loads_male_companion(self) -> None:
        settings = self.controller.settings()
        self.assertEqual(settings.companion_appearance, "male")
        window = AsherMainWindow(self.controller)
        try:
            self.assertEqual(window.home.companion_host.character, "male")
            self.assertEqual(MaleCompanion.id, "male")
            self.assertTrue(MaleCompanion.texture_path.is_file())
        finally:
            window.close()

    def test_5_female_appearance_loads_female_companion(self) -> None:
        self.controller.update_settings(companion_appearance="female")
        settings = self.controller.settings()
        self.assertEqual(settings.companion_appearance, "female")
        window = AsherMainWindow(self.controller)
        try:
            self.assertEqual(window.home.companion_host.character, "female")
            self.assertEqual(FemaleCompanion.id, "female")
            self.assertTrue(FemaleCompanion.texture_path.is_file())
        finally:
            window.close()

    def test_6_changing_appearance_switches_character(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            self.assertEqual(window.home.companion_host.character, "male")
            window.settings_page.companion_appearance.setCurrentIndex(
                window.settings_page.companion_appearance.findData("female")
            )
            window.apply_settings()
            QThreadPool.globalInstance().waitForDone(1500)
            if _APP:
                _APP.processEvents()
            self.assertEqual(self.controller.settings().companion_appearance, "female")
            self.assertEqual(window.home.companion_host.character, "female")

            # Switch back to male
            window.settings_page.companion_appearance.setCurrentIndex(
                window.settings_page.companion_appearance.findData("male")
            )
            window.apply_settings()
            QThreadPool.globalInstance().waitForDone(1500)
            if _APP:
                _APP.processEvents()
            self.assertEqual(self.controller.settings().companion_appearance, "male")
            self.assertEqual(window.home.companion_host.character, "male")
        finally:
            window.close()

    def test_7_character_rendering_pauses_when_home_is_hidden(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            host = window.home.companion_host
            # Selecting Home during construction must not start a hidden timer.
            self.assertFalse(host._active)
            self.assertFalse(host._timer.isActive())

            window.show()
            if _APP:
                _APP.processEvents()
            self.assertTrue(host._active)
            self.assertTrue(host._timer.isActive())

            # Navigating away from Home page pauses rendering
            window.select_page("Conversation")
            self.assertFalse(host._active)

            # Returning to Home resumes rendering
            window.select_page("Home")
            self.assertTrue(host._active)
            self.assertTrue(host._timer.isActive())

            window.close()
            if _APP:
                _APP.processEvents()
            self.assertFalse(host._active)
            self.assertFalse(host._timer.isActive())
        finally:
            window.close()

    def test_8_character_rendering_pauses_during_companion_mode(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            host = window.home.companion_host
            window.show()
            if _APP:
                _APP.processEvents()
            self.assertTrue(host._active)

            # Companion mode activates (e.g. user says Hey Asher)
            window._set_companion_mode(True)
            self.assertFalse(host._active)

            # Companion mode deactivates
            window._set_companion_mode(False)
            self.assertTrue(host._active)
        finally:
            window.close()

    def test_9_home_renderer_failure_does_not_break_assistant(self) -> None:
        """When WebEngine is disabled or unavailable, fallback widget takes over cleanly."""
        with patch("asher.ui.home_companion_widget.WEBENGINE_AVAILABLE", False):
            host = HomeCompanionHost()
            try:
                self.assertIsInstance(host._fallback, HomeCompanionFallbackWidget)
                self.assertFalse(host._web_supported)
                self.assertFalse(host.prewarm())
                self.assertIsNone(host._view)
                self.assertEqual(host._audio_level, 0.0)
                host.set_audio_level(0.45)
                self.assertAlmostEqual(host._audio_level, 0.45)
                host.set_state(AssistantState.LISTENING)
                self.assertEqual(host.state, AssistantState.LISTENING)
                self.assertEqual(host._fallback.state, AssistantState.LISTENING)
            finally:
                host.deleteLater()

    def test_10_presentation_state_changes_do_not_alter_controller_state(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            initial_state = self.controller.status().state
            self.assertEqual(initial_state, AssistantState.STANDBY)

            # Presentation changes to HomeCompanionHost
            window.home.companion_host.set_state(AssistantState.SPEAKING)
            window.home.companion_host.set_audio_level(0.85)

            # Controller state must remain authoritative and unchanged
            self.assertEqual(self.controller.status().state, AssistantState.STANDBY)
        finally:
            window.close()

    def test_11_hero_segmented_companion_selector_exists_and_switches_appearance(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            # Check segmented buttons exist
            self.assertTrue(hasattr(window.home, "companion_male_btn"))
            self.assertTrue(hasattr(window.home, "companion_female_btn"))

            # Default is male
            self.assertTrue(window.home.companion_male_btn.isChecked())
            self.assertFalse(window.home.companion_female_btn.isChecked())
            self.assertEqual(window.home.companion_host.character, "male")

            # Click Female button
            window.home.companion_female_btn.click()
            QThreadPool.globalInstance().waitForDone(1500)
            if _APP:
                _APP.processEvents()

            # Appearance switches to female without changing voice profile
            self.assertEqual(window.home.companion_host.character, "female")
            self.assertTrue(window.home.companion_female_btn.isChecked())
            self.assertFalse(window.home.companion_male_btn.isChecked())
            self.assertEqual(self.controller.settings().companion_appearance, "female")
            self.assertEqual(self.controller.settings().voice_profile, "offline_male")

            # Click Male button
            window.home.companion_male_btn.click()
            QThreadPool.globalInstance().waitForDone(1500)
            if _APP:
                _APP.processEvents()

            self.assertEqual(window.home.companion_host.character, "male")
            self.assertTrue(window.home.companion_male_btn.isChecked())
            self.assertFalse(window.home.companion_female_btn.isChecked())
            self.assertEqual(self.controller.settings().companion_appearance, "male")
            self.assertEqual(self.controller.settings().voice_profile, "offline_male")
        finally:
            window.close()

    def test_12_shelved_3d_scene_stays_local_transparent_and_dormant(self) -> None:
        self.assertFalse(HOME_COMPANION_EXPERIMENTAL_3D_ENABLED)
        scene_js = Path(__file__).resolve().parent.parent / "asher" / "ui" / "home_companion" / "companion_scene.js"
        self.assertTrue(scene_js.is_file())
        text = scene_js.read_text(encoding="utf-8")

        # 100% transparent canvas
        self.assertIn("alpha: true", text)
        self.assertIn("setClearColor(0x000000, 0)", text)

        # VRM Loader integration
        self.assertIn("VRMLoaderPlugin", text)
        self.assertIn("@pixiv/three-vrm", text)
        self.assertIn("fitCameraToVrm", text)
        self.assertIn("expressionManager", text)

        # Interactive 3D pointer tracking and drag rotation
        self.assertIn("setupInteractionListeners", text)
        self.assertIn("pointer", text)
        self.assertIn("targetDragAngle", text)
        self.assertIn("triggerClickReaction", text)

    def test_13_vrm_asset_targets_and_clean_fallbacks_configured(self) -> None:
        self.assertEqual(MaleCompanion.vrm_path.name, "male_asher.vrm")
        self.assertEqual(FemaleCompanion.vrm_path.name, "female_asher.vrm")
        self.assertTrue(MaleCompanion.texture_path.is_file())
        self.assertTrue(FemaleCompanion.texture_path.is_file())

    def test_14_production_diagnostics_keep_experimental_webengine_dormant(self) -> None:
        host = HomeCompanionHost()
        try:
            diag = host.diagnostics()
            self.assertEqual(diag["status"], "STATIC FALLBACK ACTIVE")
            self.assertEqual(diag["mode"], "STATIC FALLBACK MODE")
            self.assertFalse(diag["experimental_3d_enabled"])
            self.assertFalse(host._web_supported)
            self.assertFalse(host.prewarm())
            self.assertIsNone(host._view)
            self.assertIsNone(host._page)
            self.assertIn("vrm_assets", diag)
            self.assertIn("male", diag["vrm_assets"])
            self.assertIn("female", diag["vrm_assets"])
        finally:
            host.deleteLater()

    def test_15_reduced_motion_stops_animation_without_losing_state_updates(self) -> None:
        host = HomeCompanionHost()
        try:
            host.show()
            if _APP:
                _APP.processEvents()
            fallback = host._fallback
            self.assertTrue(fallback._timer.isActive())

            fallback.set_reduced_motion(True)
            phase = fallback._phase
            fallback._advance()
            self.assertFalse(fallback._timer.isActive())
            self.assertEqual(fallback._phase, phase)

            host.set_state(AssistantState.THINKING)
            self.assertEqual(fallback.state, AssistantState.THINKING)
            fallback.set_reduced_motion(False)
            self.assertTrue(fallback._timer.isActive())

            host.pause_rendering()
            fallback.set_reduced_motion(True)
            fallback.set_reduced_motion(False)
            self.assertFalse(fallback._timer.isActive())
            host.resume_rendering()
            self.assertTrue(fallback._timer.isActive())
        finally:
            host.close()
            host.deleteLater()

    def test_16_production_companion_images_have_transparent_edges(self) -> None:
        for path in (MaleCompanion.texture_path, FemaleCompanion.texture_path):
            image = QImage(str(path))
            self.assertFalse(image.isNull())
            self.assertTrue(image.hasAlphaChannel())
            corners = (
                (0, 0),
                (image.width() - 1, 0),
                (0, image.height() - 1),
                (image.width() - 1, image.height() - 1),
            )
            self.assertTrue(all(image.pixelColor(x, y).alpha() == 0 for x, y in corners))

        host = HomeCompanionHost()
        try:
            host.resize(400, 400)
            host.show()
            if _APP:
                _APP.processEvents()
            canvas = QImage(400, 400, QImage.Format.Format_ARGB32_Premultiplied)
            canvas.fill(Qt.GlobalColor.transparent)
            painter = QPainter(canvas)
            try:
                host.render(painter, QPoint())
            finally:
                painter.end()
            self.assertTrue(
                all(
                    canvas.pixelColor(x, y).alpha() == 0
                    for x, y in ((0, 0), (399, 0), (0, 399), (399, 399))
                )
            )
        finally:
            host.close()
            host.deleteLater()

    def test_17_leaving_companion_mode_does_not_resume_hidden_home(self) -> None:
        window = AsherMainWindow(self.controller)
        try:
            window.show()
            if _APP:
                _APP.processEvents()
            host = window.home.companion_host

            window.select_page("Conversation")
            self.assertFalse(host._active)
            self.assertFalse(host._timer.isActive())

            window._set_companion_mode(True)
            window._set_companion_mode(False)
            self.assertIs(window.stack.currentWidget(), window.conversation_page)
            self.assertFalse(host._active)
            self.assertFalse(host._timer.isActive())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
