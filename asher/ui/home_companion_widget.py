"""Home Workspace companion presentation host.

The approved image-based Male/Female presentation is the production path.
The local Three.js experiment remains isolated and explicitly dormant.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

from asher.core.state import AssistantState

HOME_COMPANION_DIR = Path(__file__).resolve().parent / "home_companion"
HOME_COMPANION_HTML = HOME_COMPANION_DIR / "index.html"
HOME_COMPANION_ASSETS = HOME_COMPANION_DIR / "assets"
HOME_COMPANION_MODELS = HOME_COMPANION_DIR / "models"

# Product decision: the image-based companion is the approved production
# renderer.  Do not initialize the shelved WebEngine/VRM experiment implicitly.
HOME_COMPANION_EXPERIMENTAL_3D_ENABLED = False


class MaleCompanion:
    """Descriptor and presentation metadata for the Male ASHER Companion."""

    id: str = "male"
    label: str = "Male Companion"
    description: str = "Stylized young-adult companion with dark hoodie, joggers, and electric-blue ASHER accents."
    texture_path: Path = HOME_COMPANION_ASSETS / "male_companion.png"
    model_path: Path = HOME_COMPANION_ASSETS / "male_asher.glb"
    vrm_path: Path = HOME_COMPANION_ASSETS / "male_asher.vrm"


class FemaleCompanion:
    """Descriptor and presentation metadata for the Female ASHER Companion."""

    id: str = "female"
    label: str = "Female Companion"
    description: str = "Stylized young-adult companion with off-shoulder cream outfit, top bun, and violet ASHER accents."
    texture_path: Path = HOME_COMPANION_ASSETS / "female_companion.png"
    model_path: Path = HOME_COMPANION_ASSETS / "female_asher.glb"
    vrm_path: Path = HOME_COMPANION_ASSETS / "female_asher.vrm"


try:
    from PySide6.QtCore import QObject, QPointF, QRectF, QTimer, QUrl, Qt, Signal, Slot
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QImage,
        QLinearGradient,
        QPainter,
        QPen,
        QRadialGradient,
    )
    from PySide6.QtWidgets import QSizePolicy, QStackedLayout, QWidget

    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

try:
    if not QT_AVAILABLE:
        raise ImportError("PySide6 unavailable")
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import (
        QWebEnginePage,
        QWebEngineProfile,
        QWebEngineSettings,
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView

    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False


if QT_AVAILABLE:

    class HomeCompanionBridge(QObject):
        """Presentation-only QWebChannel bridge for the Home Companion."""

        stateChangedSignal = Signal(str)
        audioLevelSignal = Signal(float)
        characterSignal = Signal(str)
        activeSignal = Signal(bool)
        reducedMotionSignal = Signal(bool)
        rendererReadySignal = Signal()
        vrmMissingSignal = Signal(str)

        def __init__(
            self,
            parent: QObject | None = None,
            *,
            initial_state: str = "STANDBY",
            initial_character: str = "male",
            initial_audio_level: float = 0.0,
            initial_reduced_motion: bool = False,
            initial_active: bool = True,
        ) -> None:
            super().__init__(parent)
            self._state = initial_state
            self._character = initial_character
            self._audio_level = initial_audio_level
            self._reduced_motion = initial_reduced_motion
            self._active = initial_active

        @Slot()
        def rendererReady(self) -> None:  # noqa: N802
            self.rendererReadySignal.emit()

        @Slot(str)
        def vrmMissing(self, gender: str) -> None:  # noqa: N802
            self.vrmMissingSignal.emit(gender)

        @Slot(result=str)
        def initialState(self) -> str:  # noqa: N802
            return self._state

        @Slot(result=str)
        def initialCharacter(self) -> str:  # noqa: N802
            return self._character

        @Slot(result=float)
        def initialAudioLevel(self) -> float:  # noqa: N802
            return self._audio_level

        @Slot(result=bool)
        def initialReducedMotion(self) -> bool:  # noqa: N802
            return self._reduced_motion

        @Slot(result=bool)
        def initialActive(self) -> bool:  # noqa: N802
            return self._active


    class HomeCompanionFallbackWidget(QWidget):
        """Approved image-based Home Companion presentation.

        Renders the selected companion (Male or Female) with natural breathing,
        blinking, state lighting, and holographic platform rings using pure Qt.
        """

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setMinimumSize(250, 250)
            self.setMaximumSize(680, 680)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
            self.setAutoFillBackground(False)
            self.setStyleSheet("background: transparent; border: 0;")

            self._character = "male"
            self._state = AssistantState.STANDBY
            self._audio_level = 0.0
            self._phase = 0.0
            self._last_frame_time = time.monotonic()
            self._active = False
            self._reduced_motion = False
            self._animation_intensity = 1.0
            self._cinematic_mode = True
            self._transparent_canvas = True

            # Cached images
            self._images: dict[str, QImage] = {}
            self._load_character_images()

            self._timer = QTimer(self)
            self._timer.setInterval(16)
            self._timer.timeout.connect(self._advance)

        def _load_character_images(self) -> None:
            male_p = HOME_COMPANION_ASSETS / "male_companion.png"
            if male_p.exists():
                self._images["male"] = QImage(str(male_p))

            female_p = HOME_COMPANION_ASSETS / "female_companion.png"
            if female_p.exists():
                self._images["female"] = QImage(str(female_p))

        @property
        def character(self) -> str:
            return self._character

        def set_character(self, name: str) -> None:
            clean = "female" if str(name).lower() == "female" else "male"
            if clean != self._character:
                self._character = clean
                self.update()

        @property
        def state(self) -> AssistantState:
            return self._state

        def set_state(self, state: AssistantState | str) -> None:
            if not isinstance(state, AssistantState):
                try:
                    state = AssistantState(str(state))
                except ValueError:
                    return
            if state == self._state:
                return
            self._state = state
            self.update()

        def set_audio_level(self, level: float) -> None:
            try:
                clamped = max(0.0, min(1.0, float(level)))
            except (TypeError, ValueError, OverflowError):
                clamped = 0.0
            self._audio_level = clamped
            if not self._reduced_motion:
                self.update()

        def set_reduced_motion(self, enabled: bool) -> None:
            self._reduced_motion = bool(enabled)
            if self._reduced_motion:
                self._timer.stop()
            elif self._active and not self._timer.isActive():
                self._last_frame_time = time.monotonic()
                self._timer.start()
            self.update()

        def set_animation_intensity(self, intensity: float) -> None:
            self._animation_intensity = max(0.0, min(1.0, float(intensity)))
            self.update()

        def pause(self) -> None:
            self._active = False
            self._timer.stop()

        def resume(self) -> None:
            self._active = True
            self._last_frame_time = time.monotonic()
            if not self._reduced_motion and not self._timer.isActive():
                self._timer.start()

        def showEvent(self, event: Any) -> None:  # noqa: N802
            super().showEvent(event)
            self.resume()

        def hideEvent(self, event: Any) -> None:  # noqa: N802
            self.pause()
            super().hideEvent(event)

        def _advance(self) -> None:
            now = time.monotonic()
            dt = max(0.001, min(0.1, now - self._last_frame_time))
            self._last_frame_time = now

            if self._reduced_motion or self._animation_intensity <= 0.0:
                return

            speed = 1.0
            if self._state is AssistantState.LISTENING:
                speed = 1.2
            elif self._state in {AssistantState.THINKING, AssistantState.EXECUTING}:
                speed = 1.6
            elif self._state is AssistantState.SPEAKING:
                speed = 1.4 + self._audio_level * 1.5

            self._phase += dt * speed * self._animation_intensity
            self.update()

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            w = float(self.width())
            h = float(self.height())
            if w <= 10 or h <= 10:
                return

            cx = w / 2.0
            cy = h * 0.48

            # Holographic Platform Rings at base
            base_y = h * 0.88
            scale = min(w / 390.0, h / 390.0)
            self._draw_platform(painter, cx, base_y, scale)

            # Breathing oscillation
            breath_y = 0.0
            breath_scale = 1.0
            if not self._reduced_motion:
                breath = math.sin(self._phase * 1.74)
                breath_y = breath * 2.2 * self._animation_intensity
                breath_scale = 1.0 + breath * 0.008 * self._animation_intensity

            img = self._images.get(self._character)
            if img and not img.isNull():
                img_w = float(img.width())
                img_h = float(img.height())
                aspect = img_w / max(1.0, img_h)

                target_h = h * 0.90 * breath_scale
                target_w = target_h * aspect

                rect = QRectF(
                    cx - target_w / 2.0,
                    cy - target_h * 0.48 + breath_y,
                    target_w,
                    target_h,
                )
                painter.drawImage(rect, img)

            # State Lighting overlay
            self._draw_state_pendant_glow(painter, cx, cy + breath_y, scale)

        def _draw_platform(self, painter: QPainter, cx: float, base_y: float, scale: float) -> None:
            is_error = self._state in {AssistantState.ERROR, AssistantState.STOPPED}
            prim_color = QColor("#FF3D00") if is_error else QColor("#00E5FF")
            rim_color = QColor("#FF1744") if is_error else QColor("#7852FF")

            # Soft floor glow
            glow = QRadialGradient(QPointF(cx, base_y), 170.0 * scale)
            g_c = QColor(prim_color)
            g_c.setAlpha(40)
            glow.setColorAt(0.0, g_c)
            glow.setColorAt(0.50, QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 20))
            glow.setColorAt(1.0, QColor(0, 0, 0, 0))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(cx, base_y), 170.0 * scale, 34.0 * scale)

            # Concentric rings
            for i in range(3):
                rw = (140.0 - i * 26.0) * scale
                rh = (26.0 - i * 5.0) * scale
                offset_y = base_y + i * 3.5 * scale

                grad = QLinearGradient(cx - rw, offset_y, cx + rw, offset_y)
                grad.setColorAt(0.0, QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 0))
                grad.setColorAt(0.3, rim_color)
                grad.setColorAt(0.5, prim_color)
                grad.setColorAt(0.7, rim_color)
                grad.setColorAt(1.0, QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 0))

                pen_w = max(0.8, (1.5 - i * 0.3) * scale)
                painter.setPen(QPen(QBrush(grad), pen_w))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(cx, offset_y), rw, rh)

            # Rotating ticks
            rot = self._phase * 0.22 if not self._reduced_motion else 0.0
            ticks = 16
            for t in range(ticks):
                ang = rot + t * (2.0 * math.pi / ticks)
                tx = cx + math.cos(ang) * (140.0 * scale)
                ty = base_y + math.sin(ang) * (26.0 * scale)
                painter.setPen(QPen(prim_color, 1.1 * scale))
                painter.drawLine(QPointF(tx, ty - 2.0 * scale), QPointF(tx, ty + 2.0 * scale))

        def _draw_state_pendant_glow(self, painter: QPainter, cx: float, cy: float, scale: float) -> None:
            # Chest pendant position relative to center
            py = cy - 35.0 * scale
            pulse = 1.0
            if not self._reduced_motion:
                if self._state is AssistantState.SPEAKING:
                    pulse = 1.2 + self._audio_level * 0.8 + math.sin(self._phase * 8.0) * 0.2
                elif self._state is AssistantState.LISTENING:
                    pulse = 1.3 + math.sin(self._phase * 4.0) * 0.2
                elif self._state in {AssistantState.THINKING, AssistantState.EXECUTING}:
                    pulse = 1.2 + math.sin(self._phase * 6.0) * 0.2

            is_error = self._state in {AssistantState.ERROR, AssistantState.STOPPED}
            c = QColor("#FF3D00") if is_error else QColor("#00E5FF")

            aura = QRadialGradient(QPointF(cx, py), 22.0 * scale * pulse)
            c_alpha = QColor(c)
            c_alpha.setAlpha(min(220, int(110 * pulse)))
            aura.setColorAt(0.0, QColor("#FFFFFF"))
            aura.setColorAt(0.35, c_alpha)
            aura.setColorAt(1.0, QColor(0, 0, 0, 0))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(aura))
            painter.drawEllipse(QPointF(cx, py), 22.0 * scale * pulse, 22.0 * scale * pulse)


    class HomeCompanionHost(QWidget):
        """Dedicated Home Workspace Animated Companion Host.

        Integrates the local WebGL / Three.js companion scene via QWebEngineView
        and provides an instant, high-definition fallback widget.
        """

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setMinimumSize(250, 250)
            self.setMaximumSize(680, 680)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.setAutoFillBackground(False)
            self.setStyleSheet("background: transparent; border: 0;")

            self._character = "male"
            self._state = AssistantState.STANDBY
            self._audio_level = 0.0
            self._active = False
            self._web_ready = False
            self._reduced_motion = False
            self._animation_intensity = 1.0
            self._cinematic_mode = True
            self._transparent_canvas = True

            self._layout = QStackedLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)

            # High-definition fallback is always available immediately
            self._fallback = HomeCompanionFallbackWidget(self)
            self._layout.addWidget(self._fallback)

            self._web_supported = bool(
                HOME_COMPANION_EXPERIMENTAL_3D_ENABLED
                and WEBENGINE_AVAILABLE
                and HOME_COMPANION_HTML.exists()
            )
            self._view: QWebEngineView | None = None
            self._page: QWebEnginePage | None = None
            self._channel: QWebChannel | None = None
            self._bridge: HomeCompanionBridge | None = None

            self._layout.setCurrentWidget(self._fallback)

        @property
        def character(self) -> str:
            return self._character

        def set_character(self, name: str) -> None:
            clean = "female" if str(name).lower() == "female" else "male"
            if clean == self._character:
                return
            self._character = clean
            self._fallback.set_character(clean)
            if self._bridge:
                self._bridge.characterSignal.emit(clean)

        @property
        def state(self) -> AssistantState:
            return self._state

        def set_state(self, state: AssistantState | str) -> None:
            if not isinstance(state, AssistantState):
                try:
                    state = AssistantState(str(state))
                except ValueError:
                    return
            if state == self._state:
                return
            self._state = state
            self._fallback.set_state(state)
            if self._bridge:
                self._bridge.stateChangedSignal.emit(state.value)

        def set_audio_level(self, level: float) -> None:
            try:
                clamped = max(0.0, min(1.0, float(level)))
            except (TypeError, ValueError, OverflowError):
                clamped = 0.0
            self._audio_level = clamped
            self._fallback.set_audio_level(clamped)
            if self._bridge:
                self._bridge.audioLevelSignal.emit(clamped)

        def pause_rendering(self) -> None:
            """Halt rendering completely to preserve GPU resources for Companion mode or background tabs."""
            self._active = False
            self._fallback.pause()
            if self._bridge:
                self._bridge.activeSignal.emit(False)

        def resume_rendering(self) -> None:
            """Resume companion rendering when Home workspace becomes active."""
            if not self.isVisible():
                self._active = False
                self._fallback.pause()
                if self._bridge:
                    self._bridge.activeSignal.emit(False)
                return
            self._active = True
            self._fallback.resume()
            if self._bridge:
                self._bridge.activeSignal.emit(True)

        def set_reduced_motion(self, enabled: bool) -> None:
            self._reduced_motion = bool(enabled)
            self._fallback.set_reduced_motion(enabled)
            if self._bridge:
                self._bridge.reducedMotionSignal.emit(bool(enabled))

        def set_animation_intensity(self, intensity: float) -> None:
            self._animation_intensity = max(0.0, min(1.0, float(intensity)))
            self._fallback.set_animation_intensity(self._animation_intensity)

        def set_cinematic_mode(self, enabled: bool) -> None:
            self._cinematic_mode = bool(enabled)

        def set_transparent_canvas(self, enabled: bool) -> None:
            self._transparent_canvas = bool(enabled)

        @property
        def _timer(self) -> QTimer:
            """Expose timer for backward compatibility with existing UI test assertions."""
            return self._fallback._timer

        def prewarm(self) -> bool:
            """Start the local WebGL renderer if supported."""
            if not self._web_supported:
                return False
            if self._view is None:
                self._setup_web_view()
            return self._web_ready

        def showEvent(self, event: Any) -> None:  # noqa: N802
            super().showEvent(event)
            if self._view is None and self._web_supported:
                self.prewarm()
            self.resume_rendering()

        def hideEvent(self, event: Any) -> None:  # noqa: N802
            self.pause_rendering()
            super().hideEvent(event)

        def _setup_web_view(self) -> None:
            try:
                profile = QWebEngineProfile.defaultProfile()
                profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
                self._page = QWebEnginePage(profile, self)
                self._page.setBackgroundColor(Qt.GlobalColor.transparent)

                settings = self._page.settings()
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)

                self._bridge = HomeCompanionBridge(
                    self,
                    initial_state=self._state.value,
                    initial_character=self._character,
                    initial_audio_level=self._audio_level,
                    initial_reduced_motion=self._reduced_motion,
                    initial_active=self._active,
                )
                self._bridge.rendererReadySignal.connect(self._on_renderer_ready)
                self._bridge.vrmMissingSignal.connect(self._on_vrm_missing)

                self._channel = QWebChannel(self._page)
                self._channel.registerObject("bridge", self._bridge)
                self._page.setWebChannel(self._channel)

                self._view = QWebEngineView(self)
                self._view.setPage(self._page)
                self._view.setStyleSheet("background: transparent; border: 0;")
                self._view.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

                self._layout.addWidget(self._view)
                self._view.setUrl(QUrl.fromLocalFile(str(HOME_COMPANION_HTML)))
            except Exception as error:
                # Graceful fallback: maintain fallback without throwing error
                self._view = None
                self._page = None
                self._channel = None
                self._bridge = None

        def _on_renderer_ready(self) -> None:
            self._web_ready = True
            male_vrm_exists = MaleCompanion.vrm_path.is_file()
            female_vrm_exists = FemaleCompanion.vrm_path.is_file()
            if self._view is not None and (male_vrm_exists or female_vrm_exists):
                self._layout.setCurrentWidget(self._view)
            else:
                self._layout.setCurrentWidget(self._fallback)

        def _on_vrm_missing(self, gender: str) -> None:
            self._layout.setCurrentWidget(self._fallback)

        def mode(self) -> str:
            """Return 'REAL 3D MODE' if WebEngine 3D renderer is active, else 'STATIC FALLBACK MODE'."""
            male_vrm_exists = MaleCompanion.vrm_path.is_file()
            female_vrm_exists = FemaleCompanion.vrm_path.is_file()
            has_vrm = male_vrm_exists or female_vrm_exists
            if (
                HOME_COMPANION_EXPERIMENTAL_3D_ENABLED
                and has_vrm
                and self._view is not None
                and self._web_ready
                and self._layout.currentWidget() is self._view
            ):
                return "REAL 3D MODE"
            return "STATIC FALLBACK MODE"

        def diagnostics(self) -> dict[str, Any]:
            male_vrm_exists = MaleCompanion.vrm_path.is_file()
            female_vrm_exists = FemaleCompanion.vrm_path.is_file()
            status_msg = "STATIC FALLBACK ACTIVE"

            return {
                "mode": self.mode(),
                "status": status_msg,
                "experimental_3d_enabled": HOME_COMPANION_EXPERIMENTAL_3D_ENABLED,
                "webengine_available": WEBENGINE_AVAILABLE,
                "web_ready": self._web_ready,
                "active_character": self._character,
                "vrm_assets": {
                    "male": {
                        "path": str(MaleCompanion.vrm_path),
                        "exists": male_vrm_exists,
                    },
                    "female": {
                        "path": str(FemaleCompanion.vrm_path),
                        "exists": female_vrm_exists,
                    },
                },
            }
