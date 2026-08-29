from __future__ import annotations

import ast
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch


# These tests deliberately avoid constructing a real Chromium/WebGL context.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")

from asher.core.state import AssistantState
import asher.ui.web_orb_widget as web_orb_module
from asher.ui.web_orb_widget import (
    CompanionOrbHost,
    WEB_ORB_INDEX,
    WEB_ORB_REQUIRED_FILES,
    WEB_ORB_ROOT,
    WebOrbBridge,
    bounded_presentation_value,
    web_orb_missing_assets,
)


_FIRST_PARTY_WEB_FILES = (
    WEB_ORB_INDEX,
    WEB_ORB_ROOT / "orb.js",
    WEB_ORB_ROOT / "bridge.js",
    WEB_ORB_ROOT / "hand_tracker.js",
)


class _HtmlInventory(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.tags.append((tag.casefold(), dict(attrs)))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _relative_imports(source: str) -> tuple[str, ...]:
    return tuple(
        match.group(1)
        for match in re.finditer(
            r"(?:\bfrom\s*|\bimport\s*)[\"']([^\"']+)[\"']",
            source,
        )
    )


def _synthetic_hand(
    label: str,
    center: tuple[float, float],
    *,
    pinch_ratio: float = 0.20,
) -> dict[str, object]:
    """Return a JSON-safe, pinched MediaPipe-style 21-landmark hand."""

    center_x, center_y = center
    landmarks = [
        {"x": center_x, "y": center_y, "z": 0.0}
        for _index in range(21)
    ]
    landmarks[0] = {"x": center_x, "y": center_y + 0.30, "z": 0.0}
    landmarks[9] = {"x": center_x, "y": center_y + 0.10, "z": 0.0}
    pinch_distance = 0.20 * pinch_ratio
    landmarks[4] = {
        "x": center_x - (pinch_distance * 0.5),
        "y": center_y,
        "z": 0.0,
    }
    landmarks[8] = {
        "x": center_x + (pinch_distance * 0.5),
        "y": center_y,
        "z": 0.0,
    }
    return {"label": label, "landmarks": landmarks}


class WebOrbAssetContractTests(unittest.TestCase):
    def test_required_renderer_files_are_local_and_nonempty(self) -> None:
        root = WEB_ORB_ROOT.resolve()
        self.assertEqual(web_orb_missing_assets(), ())
        self.assertEqual(WEB_ORB_INDEX, WEB_ORB_ROOT / "index.html")
        for asset in WEB_ORB_REQUIRED_FILES:
            with self.subTest(asset=asset.relative_to(WEB_ORB_ROOT)):
                resolved = asset.resolve()
                self.assertTrue(resolved.is_relative_to(root))
                self.assertTrue(resolved.is_file())
                self.assertGreater(resolved.stat().st_size, 0)

        three_entries = tuple((root / "vendor" / "three").glob("three*.js"))
        wasm_entries = tuple((root / "vendor" / "mediapipe" / "wasm").glob("*.wasm"))
        model = root / "assets" / "hand_landmarker.task"
        self.assertTrue(three_entries, "A local Three.js runtime is required")
        self.assertTrue(wasm_entries, "At least one local MediaPipe WASM is required")
        self.assertTrue(model.is_file())
        for asset in (*three_entries, *wasm_entries, model):
            with self.subTest(local_binary=asset.name):
                self.assertGreater(asset.stat().st_size, 0)

    def test_html_is_a_parseable_local_entry_with_resolvable_resources(self) -> None:
        source = _text(WEB_ORB_INDEX)
        inventory = _HtmlInventory()
        inventory.feed(source)
        inventory.close()

        tags = [tag for tag, _attrs in inventory.tags]
        self.assertIn("html", tags)
        self.assertIn("head", tags)
        self.assertIn("body", tags)
        self.assertNotIn("base", tags)
        self.assertTrue(
            any(tag == "main" and attrs.get("id") == "orb-root" for tag, attrs in inventory.tags)
        )
        self.assertTrue(WEB_ORB_INDEX.resolve().as_uri().startswith("file:///"))

        resources = [
            value
            for _tag, attrs in inventory.tags
            for key in ("src", "href")
            if (value := attrs.get(key))
        ]
        self.assertIn("./bridge.js", resources)
        for reference in resources:
            with self.subTest(reference=reference):
                self.assertNotRegex(reference, r"(?i)^https?://|^//")
                if reference.startswith("qrc:///"):
                    continue
                target = (WEB_ORB_ROOT / reference).resolve()
                self.assertTrue(target.is_relative_to(WEB_ORB_ROOT.resolve()))
                self.assertTrue(target.is_file(), reference)

        # A local page should also explicitly prevent opportunistic remote loads.
        self.assertRegex(source, r"connect-src\s+'self'\s+file:\s+blob:")
        self.assertIn("object-src 'none'", source)
        self.assertIn("frame-src 'none'", source)

    def test_all_ecmascript_dependencies_resolve_inside_the_bundle(self) -> None:
        root = WEB_ORB_ROOT.resolve()
        scripts = tuple(root.rglob("*.js")) + tuple(root.rglob("*.mjs"))
        self.assertTrue(scripts)
        for script in scripts:
            for specifier in _relative_imports(_text(script)):
                with self.subTest(script=script.relative_to(root), import_=specifier):
                    self.assertNotRegex(specifier, r"(?i)^https?://|^//")
                    if specifier == "three" or specifier.startswith("three/addons/"):
                        # These are resolved by index.html's local import map.
                        continue
                    self.assertTrue(specifier.startswith("."), specifier)
                    target = (script.parent / specifier).resolve()
                    self.assertTrue(target.is_relative_to(root))
                    self.assertTrue(target.is_file(), f"Unresolved local import: {specifier}")

    def test_first_party_runtime_has_no_remote_url(self) -> None:
        network_string = re.compile(
            r"[\"'`]\s*(?:(?:https?:)?//)[^\"'`\s]+",
            re.IGNORECASE,
        )
        for path in _FIRST_PARTY_WEB_FILES:
            source = _text(path)
            with self.subTest(path=path.name):
                self.assertNotRegex(source, r"(?i)\bhttps?://")
                self.assertIsNone(network_string.search(source))

        if web_orb_module.WEBENGINE_AVAILABLE:
            allowed = web_orb_module._LocalOnlyRequestInterceptor._ALLOWED_SCHEMES
            self.assertTrue(allowed.issubset({"file", "qrc", "data", "blob", "about"}))
            self.assertTrue({"http", "https", "ws", "wss"}.isdisjoint(allowed))


@unittest.skipUnless(
    web_orb_module.QT_AVAILABLE,
    "PySide6 is optional; install it for the WebOrbBridge contract tests",
)
class WebOrbBridgeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-web-orb-test"])

    def test_every_canonical_assistant_state_is_transmitted_unchanged(self) -> None:
        bridge = WebOrbBridge()
        transmitted: list[str] = []
        bridge.stateChanged.connect(transmitted.append)

        canonical_states = tuple(AssistantState)
        for state in canonical_states:
            bridge.set_state(state)

        self.assertEqual(transmitted, [state.value for state in canonical_states])
        self.assertIs(bridge.state, canonical_states[-1])

    def test_audio_and_animation_values_are_finite_and_bounded(self) -> None:
        samples = (
            (-10, 0.0),
            (-0.01, 0.0),
            (0, 0.0),
            (0.37, 0.37),
            (1, 1.0),
            (9, 1.0),
            (None, 0.0),
            ("invalid", 0.0),
            (float("nan"), 0.0),
            (float("inf"), 0.0),
            (float("-inf"), 0.0),
        )
        for value, expected in samples:
            with self.subTest(value=value):
                self.assertEqual(bounded_presentation_value(value), expected)

        bridge = WebOrbBridge()
        audio: list[float] = []
        intensity: list[float] = []
        bridge.audioLevelChanged.connect(audio.append)
        bridge.animationIntensityChanged.connect(intensity.append)
        for value, _expected in samples:
            bridge.set_audio_level(value)
            bridge.set_animation_intensity(value)

        expected = [item[1] for item in samples]
        self.assertEqual(audio, expected)
        self.assertEqual(intensity, expected)
        self.assertTrue(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in audio))

    def test_reduced_motion_and_activity_are_truthful_signals(self) -> None:
        bridge = WebOrbBridge()
        reduced_motion: list[bool] = []
        activity: list[bool] = []
        gesture_enabled: list[bool] = []
        bridge.reducedMotionChanged.connect(reduced_motion.append)
        bridge.activeChanged.connect(activity.append)
        bridge.gestureEnabledChanged.connect(gesture_enabled.append)

        bridge.set_reduced_motion(True)
        bridge.set_reduced_motion(False)
        bridge.set_active(True)
        bridge.set_gesture_enabled(True)
        bridge.set_active(False)

        self.assertEqual(reduced_motion, [True, False])
        self.assertEqual(activity, [True, False])
        self.assertEqual(gesture_enabled, [True, False])
        self.assertFalse(bridge._active)
        self.assertFalse(bridge._gesture_enabled)

    def test_qwebchannel_surface_has_only_presentation_and_diagnostic_slots(self) -> None:
        from PySide6.QtCore import QMetaMethod

        bridge = WebOrbBridge()
        meta = bridge.metaObject()
        slots = {
            bytes(meta.method(index).name()).decode("ascii")
            for index in range(meta.methodOffset(), meta.methodCount())
            if meta.method(index).methodType() == QMetaMethod.MethodType.Slot
        }
        self.assertEqual(
            slots,
            {"rendererReady", "rendererError", "trackerStatus", "submitHandFrame"},
        )

        forbidden = (
            "command",
            "controller",
            "tool",
            "confirm",
            "approve",
            "authenticate",
            "authorize",
            "security",
            "permission",
            "execute",
            "unlock",
        )
        exposed = " ".join(slots).casefold()
        self.assertFalse(any(word in exposed for word in forbidden), exposed)

        tree = ast.parse(_text(Path(web_orb_module.__file__)))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(
                module.startswith(
                    (
                        "asher.agent",
                        "asher.security",
                        "asher.tools",
                        "asher.voice.voiceguard",
                    )
                )
                for module in imported_modules
            )
        )

    def test_synthetic_hand_frames_emit_only_bounded_visual_output(self) -> None:
        bridge = WebOrbBridge()
        visual_frames: list[tuple[float, float, float, str]] = []
        state_changes: list[str] = []
        bridge.gestureChanged.connect(
            lambda x, y, expansion, mode: visual_frames.append((x, y, expansion, mode))
        )
        bridge.stateChanged.connect(state_changes.append)
        bridge.set_state(AssistantState.LOCKED)
        state_changes.clear()
        bridge.set_active(True)
        bridge.set_gesture_enabled(True)
        visual_frames.clear()

        compact = {
            "hands": [
                _synthetic_hand("Left", (0.35, 0.40)),
                _synthetic_hand("Right", (0.65, 0.40)),
            ]
        }
        spread = {
            "hands": [
                _synthetic_hand("Left", (0.08, 0.40)),
                _synthetic_hand("Right", (0.92, 0.40)),
            ]
        }
        bridge.submitHandFrame(json.dumps(compact))
        for _index in range(60):
            bridge.submitHandFrame(json.dumps(spread))

        self.assertTrue(visual_frames)
        self.assertTrue(any(frame[2] > 0.0 for frame in visual_frames))
        self.assertTrue(all(-0.22 <= frame[0] <= 0.22 for frame in visual_frames))
        self.assertTrue(all(-0.22 <= frame[1] <= 0.22 for frame in visual_frames))
        self.assertTrue(all(0.0 <= frame[2] <= 1.0 for frame in visual_frames))
        self.assertTrue(all(frame[3] in {"idle", "spin", "unfold"} for frame in visual_frames))
        self.assertIs(bridge.state, AssistantState.LOCKED)
        self.assertEqual(state_changes, [])

        # Even a hostile interpreter result is clamped before crossing the channel.
        bridge._gestures = SimpleNamespace(
            process=lambda _hands, _elapsed: SimpleNamespace(
                mode="expand",
                rotation_x=999.0,
                rotation_y=-999.0,
                expansion=999.0,
                tracked_hands=("Left", "Right", "Extra"),
            ),
            reset=lambda: None,
        )
        bridge.submitHandFrame(json.dumps(compact))
        self.assertEqual(visual_frames[-1], (0.22, -0.22, 1.0, "unfold"))


@unittest.skipUnless(
    web_orb_module.QT_AVAILABLE,
    "PySide6 is optional; install it for the CompanionOrbHost smoke tests",
)
class CompanionOrbHostContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-web-orb-host-test"])

    def tearDown(self) -> None:
        self.app.processEvents()

    def test_qpainter_fallback_is_constructible_when_webgl_is_unavailable(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        missing = (WEB_ORB_ROOT / "missing-for-test.js",)
        with (
            patch.object(web_orb_module, "web_orb_available", return_value=False),
            patch.object(web_orb_module, "web_orb_missing_assets", return_value=missing),
        ):
            host = CompanionOrbHost()
        try:
            self.assertIsInstance(host._fallback, AsherOrbWidget)
            self.assertTrue(host.fallback_active)
            self.assertFalse(host.uses_webgl)
            self.assertIn("missing-for-test.js", host.renderer_error)
        finally:
            host.close()

    def test_web_view_is_selected_only_after_ready_and_failure_restores_fallback(self) -> None:
        from PySide6.QtWidgets import QWidget

        def create_fake_view(host: CompanionOrbHost) -> None:
            host._view = QWidget(host)
            host._layout.addWidget(host._view)

        with (
            patch.object(web_orb_module, "web_orb_available", return_value=True),
            patch.object(CompanionOrbHost, "_create_web_view", create_fake_view),
        ):
            host = CompanionOrbHost()
        try:
            host.show()
            self.app.processEvents()
            self.assertIsNotNone(host._view)
            self.assertTrue(host.fallback_active)
            self.assertFalse(host.uses_webgl)

            host.bridge.rendererReady()
            self.assertTrue(host.bridge.renderer_ready)
            self.assertTrue(host.uses_webgl)
            self.assertFalse(host.fallback_active)

            host.bridge.rendererError("synthetic WebGL failure")
            self.assertFalse(host.uses_webgl)
            self.assertTrue(host.fallback_active)
            self.assertEqual(host.renderer_error, "synthetic WebGL failure")
        finally:
            # The fake QWidget intentionally has no QWebEngine page/stop API.
            host._view = None
            host.close()

    def test_host_preserves_the_companion_orb_api_and_bounds(self) -> None:
        expected_surface = {
            "set_state",
            "set_audio_level",
            "set_overlay_text",
            "set_reduced_motion",
            "set_animation_intensity",
            "set_interactive_resize",
            "set_display_bounds",
            "set_display_size",
            "current_visual_scale",
            "set_gesture_enabled",
            "shutdown",
        }
        self.assertTrue(expected_surface.issubset(set(dir(CompanionOrbHost))))

        with patch.object(web_orb_module, "web_orb_available", return_value=False):
            host = CompanionOrbHost()
        try:
            host.set_state(AssistantState.SPEAKING)
            host.set_audio_level(8.0)
            host.set_overlay_text("SPEAKING", "Local speech")
            host.set_reduced_motion(True)
            host.set_animation_intensity(-4.0)
            self.assertIs(host.state, AssistantState.SPEAKING)
            self.assertEqual(host._audio_level, 1.0)
            self.assertEqual(host._animation_intensity, 0.0)
            self.assertEqual(host.set_display_bounds(340, 700), (340, 700))
            self.assertEqual(host.set_display_size(900), 700)
            self.assertEqual((host.width(), host.height()), (700, 700))
            self.assertTrue(math.isfinite(host.current_visual_scale()))
        finally:
            host.close()


class WebOrbRendererSourceContractTests(unittest.TestCase):
    def test_renderer_preserves_reference_geometry_and_only_changes_color_identity(self) -> None:
        adapter_source = _text(WEB_ORB_ROOT / "orb.js")
        renderer_source = _text(WEB_ORB_ROOT / "ultron_exact" / "orbScene.js")
        html = _text(WEB_ORB_INDEX)

        # Exact ULTRON renderer is the visual authority.
        self.assertIn(
            'import { createOrbScene } from "./ultron_exact/orbScene.js";',
            adapter_source,
        )

        # Reference geometry / particle budgets belong to orbScene.js.
        self.assertIn("const CROSS_LINES = 18;", renderer_source)
        self.assertIn("const EQ_LINES = 20;", renderer_source)
        self.assertRegex(renderer_source, r"for \(let i = -15; i <= 15; i\+\+\)")
        self.assertRegex(renderer_source, r"for \(let i = 0; i < 24; i\+\+\)")
        self.assertRegex(renderer_source, r"for \(let i = 0; i < 30; i\+\+\)")
        self.assertRegex(renderer_source, r"for \(let i = 0; i < 250; i\+\+\)")
        self.assertIn("const dustCount = 2000;", renderer_source)
        self.assertRegex(renderer_source, r"textOuter\s*=\s*scatterText\(\s*1200")
        self.assertRegex(renderer_source, r"textInner\s*=\s*scatterText\(\s*100")
        self.assertRegex(renderer_source, r"textAmbient\s*=\s*scatterText\(\s*400")

        # Original rendering characteristics remain unchanged.
        self.assertIn("renderer.toneMappingExposure = 0.8;", renderer_source)
        self.assertRegex(
            renderer_source,
            r"new UnrealBloomPass\(\s*new THREE\.Vector2\(width, height\),\s*1\.8,\s*(?://[^\n]*\n\s*)?0\.4,\s*(?://[^\n]*\n\s*)?0\.2",
        )
        self.assertIn("ShaderPass", renderer_source)
        self.assertIn("OrbitControls", renderer_source)

        # Only the original orange/amber color identity is recolored.
        for old_color in (
            "0xffaa30",
            "0xdd7700",
            "0x884400",
            "0x553300",
            "0xffcc66",
        ):
            self.assertNotIn(old_color, renderer_source.casefold())

        for icy_color in (
            "0x4f9dff",
            "0x2367d1",
            "0x123f8a",
            "0x081d3d",
            "0xaed5ff",
        ):
            self.assertIn(icy_color, renderer_source.casefold())

        # ASHER presentation integration lives in the adapter.
        self.assertIn("let expansion = 0;", adapter_source)
        self.assertIn("let lastExpansion = 0;", adapter_source)
        self.assertIn("scene.rotateBy", adapter_source)
        self.assertIn("scene.zoomBy", adapter_source)

        # SPEAKING breath is state-driven, not fake microphone audio.
        self.assertIn('state === "speaking"', adapter_source)
        self.assertIn("speakingZoom", adapter_source)
        self.assertIn(
            "requestAnimationFrame(animateSpeakingBreath)",
            adapter_source,
        )
        self.assertIn("scene.zoomBy(relativeFactor)", adapter_source)

        # No CSS-scale shortcut.
        css_and_dom = (
            html
            + "\n"
            + adapter_source
            + "\n"
            + renderer_source
            + "\n"
            + _text(WEB_ORB_ROOT / "bridge.js")
        )
        self.assertNotRegex(css_and_dom, r"(?i)transform\s*:\s*scale\s*\(")
        self.assertNotRegex(
            css_and_dom,
            r"(?i)\.style\.transform\s*=.*scale\s*\(",
        )

    def test_camera_is_hidden_local_only_and_released(self) -> None:
        html = _text(WEB_ORB_INDEX)
        tracker = _text(WEB_ORB_ROOT / "hand_tracker.js")
        runtime = "\n".join(_text(path) for path in _FIRST_PARTY_WEB_FILES)

        inventory = _HtmlInventory()
        inventory.feed(html)
        videos = [attrs for tag, attrs in inventory.tags if tag == "video"]
        self.assertEqual(len(videos), 1)
        camera = videos[0]
        self.assertEqual(camera.get("id"), "camera-source")
        self.assertEqual(camera.get("aria-hidden"), "true")
        self.assertIn("muted", camera)
        self.assertIn("playsinline", camera)
        self.assertNotIn("controls", camera)
        self.assertRegex(html, r"(?s)#camera-source\s*\{[^}]*opacity:\s*0")
        self.assertRegex(html, r"(?s)#camera-source\s*\{[^}]*(?:-10000px|display:\s*none)")

        self.assertIn("navigator.mediaDevices.getUserMedia", tracker)
        self.assertRegex(tracker, r"audio:\s*false")
        self.assertRegex(
            tracker,
            r"getTracks\(\)\.forEach\(\(track\)\s*=>\s*track\.stop\(\)\)",
        )
        self.assertIn("this.video.srcObject = null", tracker)
        self.assertNotRegex(runtime, r"\bMediaRecorder\b|\bVideoTexture\b")
        self.assertNotRegex(
            runtime,
            r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|RTCPeerConnection)\b",
        )

    def test_animation_visibility_and_disposal_have_single_owned_lifecycles(self) -> None:
        orb = _text(WEB_ORB_ROOT / "orb.js")
        bridge = _text(WEB_ORB_ROOT / "bridge.js")
        tracker = _text(WEB_ORB_ROOT / "hand_tracker.js")

        self.assertIn("requestAnimationFrame", orb)
        self.assertIn("cancelAnimationFrame", orb)
        self.assertRegex(orb, r"\bdispose\s*\(")
        self.assertIn("setDocumentVisible", orb)
        self.assertIn('"visibilitychange"', bridge)
        self.assertIn("document.hidden", bridge)
        self.assertIn("asherOrbShutdown", bridge)
        self.assertIn('"beforeunload"', bridge)
        self.assertIn("tracker?.dispose()", bridge)
        self.assertIn("renderer?.dispose()", bridge)
        self.assertIn("requestAnimationFrame", tracker)
        self.assertIn("cancelAnimationFrame", tracker)
        self.assertRegex(tracker, r"\bdispose\s*\(")

    def test_reduced_motion_is_bridged_and_used_by_the_renderer(self) -> None:
        orb = _text(WEB_ORB_ROOT / "orb.js")
        bridge = _text(WEB_ORB_ROOT / "bridge.js")

        self.assertIn("reducedMotionChanged.connect", bridge)
        self.assertIn("renderer.setReducedMotion", bridge)
        self.assertRegex(orb, r"\bsetReducedMotion\s*\(")
        # One occurrence can be a write-only setter; several occurrences prove
        # the value also participates in animation/render decisions.
        self.assertGreaterEqual(len(re.findall(r"\breducedMotion\b", orb)), 3)


if __name__ == "__main__":
    unittest.main()

