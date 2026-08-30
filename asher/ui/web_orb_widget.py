"""Local WebGL Companion renderer host and presentation-only QWebChannel bridge.

Python remains authoritative for assistant state, audio activity, safety, and
confirmation.  The web renderer receives bounded presentation values only;
the sole JavaScript-to-Python input is a short-lived hand-landmark frame used
by :mod:`asher.ui.gesture_interpreter` to derive visual rotation/expansion.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from asher.core.state import AssistantState
from asher.ui.gesture_interpreter import GestureInterpreter
from asher.ui.orb_widget import AsherOrbWidget


WEB_ORB_ROOT = Path(__file__).resolve().with_name("web_orb")
WEB_ORB_INDEX = WEB_ORB_ROOT / "index.html"
WEB_ORB_REQUIRED_FILES = (
    WEB_ORB_INDEX,
    WEB_ORB_ROOT / "orb.js",
    WEB_ORB_ROOT / "ultron_exact" / "orbScene.js",
    WEB_ORB_ROOT / "bridge.js",
    WEB_ORB_ROOT / "hand_tracker.js",
    WEB_ORB_ROOT / "vendor" / "three" / "three.module.min.js",
    WEB_ORB_ROOT / "vendor" / "three" / "three.core.min.js",
    WEB_ORB_ROOT / "vendor" / "mediapipe" / "vision_bundle.mjs",
    WEB_ORB_ROOT
    / "vendor"
    / "mediapipe"
    / "wasm"
    / "vision_wasm_internal.wasm",
    WEB_ORB_ROOT / "assets" / "hand_landmarker.task",
    WEB_ORB_ROOT / "THIRD_PARTY_NOTICES.md",
)


def bounded_presentation_value(value: Any) -> float:
    """Return a finite 0..1 scalar suitable for a renderer bridge."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def web_orb_missing_assets() -> tuple[Path, ...]:
    """Return required renderer assets that are not present on this machine."""

    return tuple(path for path in WEB_ORB_REQUIRED_FILES if not path.is_file())


try:
    from PySide6.QtCore import QObject, QTimer, QUrl, Qt, Signal, Slot
    from PySide6.QtWidgets import QSizePolicy, QStackedLayout, QWidget

    QT_AVAILABLE = True
except ImportError as _qt_error:  # pragma: no cover - optional dependency path
    _QT_IMPORT_ERROR = _qt_error
    QT_AVAILABLE = False


try:
    if not QT_AVAILABLE:  # pragma: no cover - keeps the import error precise
        raise ImportError("PySide6 is unavailable")
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import (
        QWebEnginePage,
        QWebEnginePermission,
        QWebEngineProfile,
        QWebEngineSettings,
        QWebEngineUrlRequestInfo,
        QWebEngineUrlRequestInterceptor,
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView

    WEBENGINE_AVAILABLE = True
except ImportError as _webengine_error:  # pragma: no cover - optional path
    _WEBENGINE_IMPORT_ERROR = _webengine_error
    WEBENGINE_AVAILABLE = False


def web_orb_available() -> bool:
    """Return whether the local WebEngine renderer can be constructed."""

    return bool(WEBENGINE_AVAILABLE and not web_orb_missing_assets())


if QT_AVAILABLE:

    class WebOrbBridge(QObject):
        """Narrow, presentation-only QWebChannel surface.

        No controller, tool, confirmation, authentication, permission, or
        security object is referenced here.  JavaScript can report renderer
        health and transient hand landmarks; it cannot request an ASHER action.
        """

        stateChanged = Signal(str)
        audioLevelChanged = Signal(float)
        reducedMotionChanged = Signal(bool)
        animationIntensityChanged = Signal(float)
        activeChanged = Signal(bool)
        gestureEnabledChanged = Signal(bool)
        overlayChanged = Signal(str, str)
        gestureChanged = Signal(float, float, float, str)

        rendererReadySignal = Signal()
        rendererFailedSignal = Signal(str)
        trackerStatusSignal = Signal(str, int, str)

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._state = AssistantState.STANDBY
            self._audio_level = 0.0
            self._reduced_motion = False
            self._animation_intensity = 1.0
            self._active = False
            self._gesture_enabled = False
            self._overlay_title = "STANDBY"
            self._overlay_message = ""
            self._renderer_ready = False
            self._last_landmark_time = time.monotonic()
            self._gestures = GestureInterpreter()

        @property
        def state(self) -> AssistantState:
            return self._state

        @property
        def audio_level(self) -> float:
            return self._audio_level

        @property
        def renderer_ready(self) -> bool:
            return self._renderer_ready

        def set_state(self, state: AssistantState) -> None:
            if not isinstance(state, AssistantState):
                state = AssistantState(str(state))
            self._state = state
            self.stateChanged.emit(state.value)

        def set_audio_level(self, level: Any) -> None:
            self._audio_level = bounded_presentation_value(level)
            self.audioLevelChanged.emit(self._audio_level)

        def set_reduced_motion(self, enabled: bool) -> None:
            self._reduced_motion = bool(enabled)
            self.reducedMotionChanged.emit(self._reduced_motion)

        def set_animation_intensity(self, value: Any) -> None:
            self._animation_intensity = bounded_presentation_value(value)
            self.animationIntensityChanged.emit(self._animation_intensity)

        def set_active(self, active: bool) -> None:
            self._active = bool(active)
            if not self._active:
                self.set_gesture_enabled(False)
                self._gestures.reset()
                self.gestureChanged.emit(0.0, 0.0, 0.0, "idle")
            self.activeChanged.emit(self._active)

        def set_gesture_enabled(self, enabled: bool) -> None:
            self._gesture_enabled = bool(enabled and self._active)
            if not self._gesture_enabled:
                self._gestures.reset()
                self.gestureChanged.emit(0.0, 0.0, 0.0, "idle")
            self.gestureEnabledChanged.emit(self._gesture_enabled)

        def set_overlay_text(self, title: str, message: str = "") -> None:
            self._overlay_title = str(title or "").strip()
            self._overlay_message = str(message or "").strip()
            self.overlayChanged.emit(self._overlay_title, self._overlay_message)

        def publish_snapshot(self) -> None:
            """Re-publish current values after JavaScript connects."""

            self.stateChanged.emit(self._state.value)
            self.audioLevelChanged.emit(self._audio_level)
            self.reducedMotionChanged.emit(self._reduced_motion)
            self.animationIntensityChanged.emit(self._animation_intensity)
            self.activeChanged.emit(self._active)
            self.gestureEnabledChanged.emit(self._gesture_enabled)
            self.overlayChanged.emit(self._overlay_title, self._overlay_message)

        @Slot()
        def rendererReady(self) -> None:  # noqa: N802 - QWebChannel API
            self._renderer_ready = True
            self.rendererReadySignal.emit()
            QTimer.singleShot(0, self.publish_snapshot)

        @Slot(str)
        def rendererError(self, message: str) -> None:  # noqa: N802
            safe = str(message or "WebGL renderer failed")[:500]
            self._renderer_ready = False
            self.rendererFailedSignal.emit(safe)

        @Slot(str)
        def trackerStatus(self, payload: str) -> None:  # noqa: N802
            """Accept diagnostics-only tracker state, never camera pixels."""

            mode = "idle"
            hands = 0
            error = ""
            try:
                data = json.loads(str(payload)[:2000])
                if isinstance(data, dict):
                    candidate = str(data.get("mode", "idle"))
                    if candidate in {"idle", "spin", "unfold"}:
                        mode = candidate
                    hands = max(0, min(2, int(data.get("hands", 0))))
                    error = str(data.get("error", ""))[:240]
            except (TypeError, ValueError, json.JSONDecodeError):
                error = "Invalid local tracker status"
            self.trackerStatusSignal.emit(mode, hands, error)

        @Slot(str)
        def submitHandFrame(self, payload: str) -> None:  # noqa: N802
            """Interpret one local landmark frame into bounded visuals.

            The JSON is capped and discarded after this call.  No image data is
            accepted, recorded, transmitted, or exposed to an ASHER controller.
            """

            if not (self._active and self._gesture_enabled):
                return
            text = str(payload)
            if len(text) > 24000:
                return
            try:
                data = json.loads(text)
                hands = data.get("hands", []) if isinstance(data, dict) else []
                if not isinstance(hands, list):
                    hands = []
            except (TypeError, ValueError, json.JSONDecodeError):
                hands = []

            now = time.monotonic()
            elapsed = max(1.0 / 120.0, min(0.10, now - self._last_landmark_time))
            self._last_landmark_time = now
            output = self._gestures.process(hands[:2], elapsed)
            renderer_mode = {
                "rotate": "spin",
                "expand": "unfold",
            }.get(str(output.mode), "idle")
            self.gestureChanged.emit(
                max(-0.22, min(0.22, float(output.rotation_x))),
                max(-0.22, min(0.22, float(output.rotation_y))),
                bounded_presentation_value(output.expansion),
                renderer_mode,
            )
            self.trackerStatusSignal.emit(
                renderer_mode,
                max(0, min(2, len(output.tracked_hands))),
                "",
            )


else:  # pragma: no cover - dependency-free import placeholder

    class WebOrbBridge:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for the WebGL orb bridge") from _QT_IMPORT_ERROR


if WEBENGINE_AVAILABLE:

    class _LocalOnlyRequestInterceptor(QWebEngineUrlRequestInterceptor):
        """Block every network scheme; the renderer is a sealed local surface."""

        _ALLOWED_SCHEMES = frozenset({"file", "qrc", "data", "blob", "about"})

        def interceptRequest(self, info: QWebEngineUrlRequestInfo) -> None:  # noqa: N802
            if info.requestUrl().scheme().casefold() not in self._ALLOWED_SCHEMES:
                info.block(True)


    class _LocalOrbPage(QWebEnginePage):
        consoleMessage = Signal(str)

        def javaScriptConsoleMessage(  # noqa: N802
            self,
            level: QWebEnginePage.JavaScriptConsoleMessageLevel,
            message: str,
            line_number: int,
            source_id: str,
        ) -> None:
            if level != QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel:
                source = Path(str(source_id)).name or "renderer"
                self.consoleMessage.emit(f"{source}:{line_number}: {message}"[:500])


if not QT_AVAILABLE:

    class CompanionOrbHost:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for Companion mode") from _QT_IMPORT_ERROR


else:

    class CompanionOrbHost(QWidget):
        """Companion-only WebGL host with a proven QPainter fallback."""

        rendererChanged = Signal(bool, str)
        gestureStateChanged = Signal(bool, str)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            # WebGL Companion mode is a viewport, not a fixed square.
            # Explicit set_display_* calls still preserve the legacy bounded API.
            self.setMinimumSize(1, 1)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setStyleSheet("background: #02050b; border: 0;")
            self.setAccessibleName("Asher interactive WebGL energy orb")

            self._state = AssistantState.STANDBY
            self._audio_level = 0.0
            self._reduced_motion = False
            self._animation_intensity = 1.0
            self._overlay_title = "STANDBY"
            self._overlay_message = ""
            self._active = False
            self._gesture_enabled = False
            self._cinematic_mode = True
            self._display_size = 700
            self._minimum_display_size = 340
            self._maximum_display_size = 900
            self._web_ready = False
            self._renderer_failed = False
            self._renderer_error = ""
            self._shutdown = False
            self._web_supported = web_orb_available()

            self._layout = QStackedLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)

            # Startup is intentionally visually neutral. The legacy QPainter
            # orb is reserved for a genuine WebGL failure, so it can never flash
            # before the exact local Three.js renderer becomes ready.
            self._startup_surface = QWidget(self)
            self._startup_surface.setAttribute(
                Qt.WidgetAttribute.WA_StyledBackground,
                True,
            )
            self._startup_surface.setStyleSheet(
                "background: #02050b; border: 0;"
            )
            self._layout.addWidget(self._startup_surface)

            self._fallback = AsherOrbWidget(self)
            self._fallback.set_cinematic_mode(True)
            self._fallback.set_interactive_resize(False)
            self._layout.addWidget(self._fallback)
            self._layout.setCurrentWidget(self._startup_surface)

            self.bridge = WebOrbBridge(self)
            self.bridge.rendererReadySignal.connect(self._activate_webgl)
            self.bridge.rendererFailedSignal.connect(self._activate_fallback)
            self.bridge.trackerStatusSignal.connect(self._on_tracker_status)

            self._view: Any | None = None
            self._page: Any | None = None
            self._profile: Any | None = None
            self._channel: Any | None = None
            self._interceptor: Any | None = None

            if not self._web_supported:
                missing = web_orb_missing_assets()
                detail = (
                    "QtWebEngine is unavailable"
                    if not WEBENGINE_AVAILABLE
                    else "Missing local renderer assets: "
                    + ", ".join(path.name for path in missing)
                )
                self._activate_fallback(detail)

        @property
        def state(self) -> AssistantState:
            return self._state

        @property
        def uses_webgl(self) -> bool:
            return bool(self._web_ready and self._layout.currentWidget() is self._view)

        @property
        def fallback_active(self) -> bool:
            """Preserve the legacy non-WebGL contract during safe startup/failure."""

            current = self._layout.currentWidget()
            return current is self._startup_surface or current is self._fallback

        @property
        def renderer_error(self) -> str:
            return self._renderer_error

        @property
        def presentation_ready(self) -> bool:
            """Return True once Companion can switch without exposing startup fallback."""

            return bool(
                self._web_ready
                or self._renderer_failed
                or not self._web_supported
            )

        @property
        def gestures_available(self) -> bool:
            return bool(self._web_ready and self._view is not None)

        def prewarm(self) -> bool:
            """Start the sealed local WebGL renderer while Companion is still hidden."""

            if self._shutdown or not self._web_supported:
                return self.presentation_ready
            if self._view is None:
                try:
                    self._create_web_view()
                except Exception as error:
                    self._activate_fallback(
                        f"Local WebGL startup failed: {type(error).__name__}"
                    )
            return self._web_ready

        def _create_web_view(self) -> None:
            self._profile = QWebEngineProfile(self)
            self._profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
            self._profile.setPersistentCookiesPolicy(
                QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies
            )
            self._interceptor = _LocalOnlyRequestInterceptor(self._profile)
            self._profile.setUrlRequestInterceptor(self._interceptor)

            self._view = QWebEngineView(self)
            self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self._view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._view.setStyleSheet("background: #02050b; border: 0;")
            self._page = _LocalOrbPage(self._profile, self._view)
            self._view.setPage(self._page)

            settings = self._page.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True
            )
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
            )
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
            )

            self._channel = QWebChannel(self._page)
            self._channel.registerObject("asherBridge", self.bridge)
            self._page.setWebChannel(self._channel)
            self._page.consoleMessage.connect(self._on_console_message)
            self._page.renderProcessTerminated.connect(
                lambda *_args: self._activate_fallback("WebGL render process stopped")
            )
            if hasattr(self._page, "permissionRequested"):
                self._page.permissionRequested.connect(self._on_permission_requested)
            self._view.loadFinished.connect(self._on_load_finished)
            self._layout.addWidget(self._view)
            self._view.load(QUrl.fromLocalFile(str(WEB_ORB_INDEX)))

        def _on_load_finished(self, loaded: bool) -> None:
            if not loaded:
                self._activate_fallback("Local WebGL page failed to load")

        def _on_console_message(self, message: str) -> None:
            # Informational WebEngine messages are intentionally ignored by the
            # page; warnings/errors keep the fallback reason diagnosable.
            self._renderer_error = str(message)[:500]

        def _activate_webgl(self) -> None:
            if self._shutdown or self._view is None:
                return
            self._web_ready = True
            self._renderer_failed = False
            self._renderer_error = ""
            self._layout.setCurrentWidget(self._view)
            self._fallback.set_audio_level(0.0)
            self.rendererChanged.emit(True, "Local Three.js renderer active")
            self.bridge.publish_snapshot()

        def _activate_fallback(self, reason: str = "") -> None:
            self._web_ready = False
            self._renderer_failed = True
            self._renderer_error = str(reason or "WebGL renderer unavailable")[:500]
            self._gesture_enabled = False
            self.bridge.set_gesture_enabled(False)
            self._layout.setCurrentWidget(self._fallback)
            self.rendererChanged.emit(False, self._renderer_error)
            self.gestureStateChanged.emit(False, self._renderer_error)

        def _on_permission_requested(self, permission: Any) -> None:
            current_local_file = ""
            if self._page is not None:
                current_local_file = self._page.url().toLocalFile()
            try:
                trusted_page = (
                    Path(current_local_file).resolve() == WEB_ORB_INDEX.resolve()
                    and permission.origin().scheme().casefold() == "file"
                )
            except (OSError, RuntimeError, ValueError):
                trusted_page = False
            allowed = (
                trusted_page
                and self._active
                and self._gesture_enabled
                and permission.permissionType()
                == QWebEnginePermission.PermissionType.MediaVideoCapture
            )
            if allowed:
                permission.grant()
            else:
                permission.deny()

        def _on_tracker_status(self, mode: str, hands: int, error: str) -> None:
            if error:
                self._gesture_enabled = False
                self.bridge.set_gesture_enabled(False)
                self.gestureStateChanged.emit(False, error)
                return
            labels = {
                "idle": "Gestures ready",
                "spin": "Pinch rotation",
                "unfold": "Energy unfolding",
            }
            detail = labels.get(mode, "Gestures ready")
            if hands:
                detail = f"{detail} · {hands} hand{'s' if hands != 1 else ''}"
            self.gestureStateChanged.emit(self._gesture_enabled, detail)

        def set_state(self, state: AssistantState) -> None:
            if not isinstance(state, AssistantState):
                state = AssistantState(str(state))
            self._state = state
            self._fallback.set_state(state)
            self.bridge.set_state(state)
            self.setAccessibleDescription(state.value.replace("_", " "))

        def set_audio_level(self, level: Any) -> None:
            self._audio_level = bounded_presentation_value(level)
            self._fallback.set_audio_level(self._audio_level)
            self.bridge.set_audio_level(self._audio_level)

        def set_overlay_text(self, title: str, message: str = "") -> None:
            self._overlay_title = str(title or "").strip()
            self._overlay_message = str(message or "").strip()
            self._fallback.set_overlay_text(self._overlay_title, self._overlay_message)
            self.bridge.set_overlay_text(self._overlay_title, self._overlay_message)
            description = ". ".join(
                part for part in (self._overlay_title, self._overlay_message) if part
            )
            if description:
                self.setAccessibleDescription(description)

        def set_reduced_motion(self, enabled: bool) -> None:
            self._reduced_motion = bool(enabled)
            self._fallback.set_reduced_motion(self._reduced_motion)
            self.bridge.set_reduced_motion(self._reduced_motion)

        def set_animation_intensity(self, intensity: Any) -> None:
            self._animation_intensity = bounded_presentation_value(intensity)
            self._fallback.set_animation_intensity(self._animation_intensity)
            self.bridge.set_animation_intensity(self._animation_intensity)

        def set_interactive_resize(
            self, enabled: bool, *, initial_size: int | None = None
        ) -> None:
            # Kept for compatibility with the QPainter milestone.  WebGL hand
            # gestures unfold true scene depth and never resize this QWidget.
            if initial_size is not None:
                self.set_display_size(initial_size)

        def set_display_bounds(self, minimum: int, maximum: int) -> tuple[int, int]:
            lower = max(340, min(900, int(minimum)))
            upper = max(lower, min(900, int(maximum)))
            self._minimum_display_size = lower
            self._maximum_display_size = upper
            self.set_display_size(self._display_size)
            return lower, upper

        def set_display_size(self, size: int) -> int:
            applied = max(
                self._minimum_display_size,
                min(self._maximum_display_size, int(size)),
            )
            self._display_size = applied
            self.setFixedSize(applied, applied)
            return applied

        def current_visual_scale(self) -> float:
            return self._fallback.current_visual_scale()

        def set_gesture_enabled(self, enabled: bool) -> bool:
            applied = bool(enabled and self.gestures_available and self._active)
            self._gesture_enabled = applied
            self.bridge.set_gesture_enabled(applied)
            self.gestureStateChanged.emit(
                applied,
                "Starting local hand tracking" if applied else "Gestures off",
            )
            return applied

        def showEvent(self, event: Any) -> None:  # noqa: N802
            super().showEvent(event)
            self._active = True
            self.bridge.set_active(True)
            self.prewarm()

        def hideEvent(self, event: Any) -> None:  # noqa: N802
            self._active = False
            self._gesture_enabled = False
            self.bridge.set_active(False)
            self.gestureStateChanged.emit(False, "Gestures off")
            super().hideEvent(event)

        def shutdown(self) -> None:
            if self._shutdown:
                return
            self._shutdown = True
            self._active = False
            self._gesture_enabled = False
            self.bridge.set_active(False)
            if self._view is not None:
                try:
                    self._view.page().runJavaScript(
                        "window.asherOrbShutdown && window.asherOrbShutdown();"
                    )
                    self._view.stop()
                except Exception:
                    pass
                if self._page is not None:
                    try:
                        self._page.setWebChannel(None)
                    except Exception:
                        pass
                page = self._page or self._view.page()
                try:
                    self._view.setPage(None)
                except Exception:
                    pass
                if page is not None:
                    page.deleteLater()
                self._view.close()
                self._view.deleteLater()
                self._view = None
                self._page = None
            if hasattr(self, "_profile") and self._profile is not None:
                self._profile.deleteLater()
                self._profile = None

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            self.shutdown()
            super().closeEvent(event)


__all__ = [
    "CompanionOrbHost",
    "WEBENGINE_AVAILABLE",
    "WEB_ORB_INDEX",
    "WEB_ORB_REQUIRED_FILES",
    "WebOrbBridge",
    "bounded_presentation_value",
    "web_orb_available",
    "web_orb_missing_assets",
]
