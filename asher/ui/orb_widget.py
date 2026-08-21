"""Procedural, state-driven ASHER companion orb.

The widget is presentation-only: it never advances assistant state on its own.
It renders whatever canonical :class:`AssistantState` the controller reports,
so animation can never imply that a task progressed or succeeded before the
runtime publishes that truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from asher.core.state import AssistantState


@dataclass(frozen=True)
class OrbVisual:
    """Small immutable visual contract for one canonical ASHER state."""

    primary: str
    secondary: str
    accent: str
    speed: float
    pulse: float
    particles: int
    label: str


_DEFAULT = OrbVisual("#55D6FF", "#3D69FF", "#F4F7FB", 0.55, 0.12, 18, "Standing by")

_STATE_VISUALS: dict[AssistantState, OrbVisual] = {
    AssistantState.STANDBY: OrbVisual("#55D6FF", "#3D69FF", "#8D9AAC", 0.42, 0.08, 14, "Say “Hey Asher”"),
    AssistantState.WAKE_DETECTED: OrbVisual("#74E4FF", "#55D6FF", "#F4F7FB", 1.45, 0.22, 24, "Wake detected"),
    AssistantState.AUTHENTICATING: OrbVisual("#4F8CFF", "#A678FF", "#D6E4FF", 1.15, 0.12, 20, "Verifying speaker"),
    AssistantState.AUTHENTICATED: OrbVisual("#62E6A7", "#55D6FF", "#F4F7FB", 0.70, 0.16, 18, "Authenticated"),
    AssistantState.LISTENING: OrbVisual("#55D6FF", "#3D69FF", "#E8FBFF", 0.85, 0.20, 22, "Listening…"),
    AssistantState.TRANSCRIBING: OrbVisual("#55D6FF", "#6FA8FF", "#DDF7FF", 1.10, 0.10, 18, "Understanding speech…"),
    AssistantState.THINKING: OrbVisual("#A678FF", "#566CFF", "#F1E8FF", 1.05, 0.14, 24, "Thinking…"),
    AssistantState.AWAITING_CONFIRMATION: OrbVisual("#FFB44A", "#FFD166", "#FFF2D2", 0.52, 0.10, 12, "Awaiting confirmation"),
    AssistantState.EXECUTING: OrbVisual("#FFB44A", "#FF8B45", "#FFF0DC", 1.45, 0.15, 26, "Executing…"),
    AssistantState.OBSERVING: OrbVisual("#6CB9FF", "#FFB44A", "#E8F5FF", 1.18, 0.10, 22, "Checking result…"),
    AssistantState.SPEAKING: OrbVisual("#55D6FF", "#A678FF", "#F4F7FB", 0.95, 0.23, 24, "Speaking…"),
    AssistantState.SUCCESS: OrbVisual("#62E6A7", "#49CFA1", "#E8FFF5", 0.50, 0.12, 18, "Done"),
    AssistantState.ERROR: OrbVisual("#FF5C70", "#B64662", "#FFE4E8", 0.62, 0.09, 10, "Something needs attention"),
    AssistantState.OFFLINE: OrbVisual("#6D97A8", "#55D6FF", "#BFD4DC", 0.35, 0.06, 10, "Offline mode — Ollama"),
    AssistantState.STOPPED: OrbVisual("#717985", "#4D535D", "#B7BDC6", 0.12, 0.02, 0, "Stopped safely"),
    AssistantState.LOCKED: OrbVisual("#69717D", "#383E48", "#A9B0BA", 0.18, 0.03, 0, "Authentication required"),
    # Legacy/fallback controller states remain renderable while the real
    # CompanionController uses the precise cinematic vocabulary.
    AssistantState.UNDERSTANDING: OrbVisual("#A678FF", "#566CFF", "#F1E8FF", 0.95, 0.12, 20, "Understanding…"),
    AssistantState.ACTING: OrbVisual("#FFB44A", "#FF8B45", "#FFF0DC", 1.25, 0.13, 22, "Acting…"),
    AssistantState.COMPLETE: OrbVisual("#62E6A7", "#49CFA1", "#E8FFF5", 0.45, 0.10, 16, "Complete"),
}


def visual_for_state(state: AssistantState) -> OrbVisual:
    """Return the bounded visual profile for ``state`` without importing Qt."""

    return _STATE_VISUALS.get(state, _DEFAULT)


try:
    from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
    from PySide6.QtGui import QColor, QBrush, QPainter, QPen, QRadialGradient
    from PySide6.QtWidgets import QWidget

    QT_AVAILABLE = True
except ImportError as _qt_error:  # pragma: no cover - dependency-free path
    _QT_IMPORT_ERROR = _qt_error
    QT_AVAILABLE = False


if not QT_AVAILABLE:

    class AsherOrbWidget:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for the ASHER orb widget") from _QT_IMPORT_ERROR


else:

    class AsherOrbWidget(QWidget):
        """Lightweight QPainter orb that reflects real ASHER state.

        The timer only changes visual phase. It never changes assistant state.
        Rendering slows while idle and stops entirely while hidden/minimised.
        ``set_audio_level`` is intentionally a bounded presentation hook; the
        current milestone does not persist or inspect raw microphone audio.
        """

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setMinimumSize(340, 340)
            self.setMaximumSize(900, 900)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
            self._state = AssistantState.STANDBY
            self._visual = visual_for_state(self._state)
            self._phase = 0.0
            self._audio_level = 0.0
            self._reduced_motion = False
            self._animation_intensity = 1.0
            self._interactive_resize = False
            self._display_size = 560
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._advance)
            self._sync_timer()

        @property
        def state(self) -> AssistantState:
            return self._state

        def set_state(self, state: AssistantState) -> None:
            if not isinstance(state, AssistantState):
                state = AssistantState(str(state))
            if state == self._state:
                return
            self._state = state
            self._visual = visual_for_state(state)
            self._audio_level = 0.0 if state != AssistantState.SPEAKING else self._audio_level
            self._sync_timer()
            self.update()

        def set_audio_level(self, level: float) -> None:
            """Accept a short-lived normalized level without retaining audio."""

            self._audio_level = max(0.0, min(1.0, float(level)))
            if self.isVisible():
                self.update()

        def set_reduced_motion(self, enabled: bool) -> None:
            self._reduced_motion = bool(enabled)
            self._sync_timer()
            self.update()

        def set_animation_intensity(self, intensity: float) -> None:
            self._animation_intensity = max(0.0, min(1.0, float(intensity)))
            self._sync_timer()
            self.update()

        def set_interactive_resize(self, enabled: bool, *, initial_size: int | None = None) -> None:
            """Enable presentation-only resizing without changing assistant state.

            The same bounded ``set_display_size`` hook can later be driven by a
            webcam hand-gesture adapter. This milestone only wires desktop
            wheel input so the interaction can be tested without claiming hand
            tracking is already implemented.
            """

            self._interactive_resize = bool(enabled)
            if initial_size is not None:
                self.set_display_size(initial_size)
            elif not enabled:
                self.setMinimumSize(340, 340)
                self.setMaximumSize(900, 900)

        def set_display_size(self, size: int) -> int:
            """Set a bounded square display size and return the applied value."""

            applied = max(340, min(900, int(size)))
            self._display_size = applied
            self.setFixedSize(applied, applied)
            return applied

        def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback
            if not self._interactive_resize:
                super().wheelEvent(event)
                return
            delta = event.angleDelta().y()
            if delta:
                step = 28 if delta > 0 else -28
                self.set_display_size(self._display_size + step)
                event.accept()
                return
            super().wheelEvent(event)

        def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback
            super().showEvent(event)
            self._sync_timer()

        def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback
            self._timer.stop()
            super().hideEvent(event)

        def _sync_timer(self) -> None:
            if self._reduced_motion or self._animation_intensity <= 0.0:
                self._timer.stop()
                return
            if not self.isVisible() and self.parentWidget() is not None:
                self._timer.stop()
                return
            active = self._state not in {
                AssistantState.STANDBY,
                AssistantState.OFFLINE,
                AssistantState.STOPPED,
                AssistantState.LOCKED,
                AssistantState.SUCCESS,
                AssistantState.COMPLETE,
            }
            interval = 33 if active else 70
            if self._timer.interval() != interval:
                self._timer.setInterval(interval)
            if not self._timer.isActive():
                self._timer.start()

        def _advance(self) -> None:
            if not self.isVisible():
                self._timer.stop()
                return
            self._phase = (self._phase + 0.035 * self._visual.speed * self._animation_intensity) % (math.tau * 100)
            # Presentation-only audio influence decays rapidly and never changes state.
            self._audio_level *= 0.82
            self.update()

        def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt callback
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            width = float(self.width())
            height = float(self.height())
            center = QPointF(width / 2.0, height / 2.0)
            extent = min(width, height)
            base_radius = extent * 0.205
            activity = self._audio_level if self._state in {AssistantState.LISTENING, AssistantState.SPEAKING} else 0.0
            pulse = math.sin(self._phase * 2.1) * self._visual.pulse * self._animation_intensity
            radius = base_radius * (1.0 + pulse + activity * 0.10)

            painter.fillRect(self.rect(), QColor("#05070B"))
            self._draw_ambient_glow(painter, center, radius)
            self._draw_particles(painter, center, radius)
            self._draw_orbits(painter, center, radius)
            self._draw_core(painter, center, radius)

        def _draw_ambient_glow(self, painter: QPainter, center: QPointF, radius: float) -> None:
            primary = QColor(self._visual.primary)
            for multiplier, alpha in ((2.1, 12), (1.75, 18), (1.45, 26), (1.22, 34)):
                glow = QColor(primary)
                glow.setAlpha(int(alpha * max(0.25, self._animation_intensity)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(glow))
                r = radius * multiplier
                painter.drawEllipse(center, r, r)

        def _draw_particles(self, painter: QPainter, center: QPointF, radius: float) -> None:
            count = int(self._visual.particles * self._animation_intensity)
            if count <= 0:
                return
            primary = QColor(self._visual.primary)
            secondary = QColor(self._visual.secondary)
            for index in range(count):
                ratio = (index + 1) / (count + 1)
                angle = self._phase * (0.55 + (index % 4) * 0.08) + ratio * math.tau * 2.8
                orbit = radius * (1.22 + 0.72 * ((index * 37) % 11) / 10.0)
                wobble = 1.0 + 0.055 * math.sin(self._phase * 1.8 + index)
                point = QPointF(
                    center.x() + math.cos(angle) * orbit * wobble,
                    center.y() + math.sin(angle) * orbit * 0.72 * wobble,
                )
                color = QColor(primary if index % 2 == 0 else secondary)
                color.setAlpha(70 + int(100 * (1.0 - ratio * 0.65)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                size = 1.3 + (index % 3) * 0.65
                painter.drawEllipse(point, size, size)

        def _draw_orbits(self, painter: QPainter, center: QPointF, radius: float) -> None:
            primary = QColor(self._visual.primary)
            secondary = QColor(self._visual.secondary)
            rotations = (0.0, 48.0, -37.0)
            scales = (1.0, 1.17, 1.34)
            spans = (185.0, 126.0, 82.0)
            for index, (rotation, scale, span) in enumerate(zip(rotations, scales, spans)):
                painter.save()
                painter.translate(center)
                painter.rotate(rotation + math.degrees(self._phase) * (0.22 + index * 0.09))
                painter.translate(-center)
                color = QColor(primary if index != 1 else secondary)
                color.setAlpha(185 - index * 38)
                pen = QPen(color, max(1.0, radius * (0.012 - index * 0.002)))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                r = radius * scale
                rect = QRectF(center.x() - r, center.y() - r * 0.72, r * 2, r * 1.44)
                start = int((self._phase * (40 + index * 17) + index * 105) * 16)
                painter.drawArc(rect, start, int(span * 16))
                painter.restore()

        def _draw_core(self, painter: QPainter, center: QPointF, radius: float) -> None:
            gradient = QRadialGradient(center, radius)
            core = QColor(self._visual.accent)
            primary = QColor(self._visual.primary)
            secondary = QColor(self._visual.secondary)
            core.setAlpha(245)
            primary.setAlpha(220)
            secondary.setAlpha(135)
            gradient.setColorAt(0.0, core)
            gradient.setColorAt(0.18, primary)
            gradient.setColorAt(0.60, secondary)
            edge = QColor(self._visual.secondary)
            edge.setAlpha(18)
            gradient.setColorAt(1.0, edge)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(center, radius, radius)

            highlight = QColor("#FFFFFF")
            highlight.setAlpha(55)
            painter.setBrush(highlight)
            painter.drawEllipse(
                QPointF(center.x() - radius * 0.28, center.y() - radius * 0.31),
                radius * 0.18,
                radius * 0.11,
            )


__all__ = ["AsherOrbWidget", "OrbVisual", "QT_AVAILABLE", "visual_for_state"]
