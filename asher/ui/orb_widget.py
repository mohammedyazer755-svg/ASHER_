"""Procedural, state-driven ASHER companion orb.

The widget is presentation-only: it never advances assistant state on its own.
It renders whatever canonical :class:`AssistantState` the controller reports,
so animation can never imply that a task progressed or succeeded before the
runtime publishes that truth.
"""

from __future__ import annotations

import math
import os
import time
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

# Fixed budgets keep the cinematic renderer deterministic and inexpensive.
# These are intentionally module-level so regression tests can protect the
# CPU envelope without relying on frame-time thresholds from a shared runner.
_CINEMATIC_SHELL_SEGMENTS = 192
_CINEMATIC_SPARK_COUNT = 11
_CINEMATIC_TRAJECTORY_COUNT = 3
_CINEMATIC_WISP_COUNT = 7

# The cinematic orb is always rendered as ice-blue glass.  Canonical state
# colours are still available to the surrounding status/confirmation UI, but
# the living surface itself keeps one calm visual identity.
_GLASS_BACKGROUND = "#04101B"
_GLASS_PRIMARY = "#79DFFF"
_GLASS_SECONDARY = "#C8F3FF"
_GLASS_HIGHLIGHT = "#F7FDFF"
_GLASS_DEPTH = "#184D79"

# Real microphone RMS is deliberately quiet compared with a normalized peak.
# Lift values above the runtime's speech gate into a useful visual range while
# keeping presentation noise at (or very close to) zero.
_AUDIO_ACTIVITY_FLOOR = 0.008
_AUDIO_ACTIVITY_CEILING = 0.18
_CINEMATIC_IDLE_SCALE = 0.70
_CINEMATIC_VOICE_SCALE = 0.98


def _bounded_scalar(value: Any) -> float:
    """Return a finite 0..1 scalar for presentation-only inputs."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def audio_activity_for_level(level: Any) -> float:
    """Map real microphone RMS to a perceptually useful 0..1 activity value."""

    sample = _bounded_scalar(level)
    normalized = max(
        0.0,
        min(
            1.0,
            (sample - _AUDIO_ACTIVITY_FLOOR)
            / (_AUDIO_ACTIVITY_CEILING - _AUDIO_ACTIVITY_FLOOR),
        ),
    )
    # Square-root compression gives ordinary speech useful travel long before
    # PCM approaches a theoretical full-scale peak.  The scale function below
    # applies the final smoothstep, so the quiet boundary still stays calm.
    return math.sqrt(normalized)


def cinematic_scale_for_activity(activity: Any) -> float:
    """Return the bounded quiet-to-voice scale used by the glass sphere."""

    amount = _bounded_scalar(activity)
    eased = amount * amount * (3.0 - 2.0 * amount)
    return _CINEMATIC_IDLE_SCALE + (
        _CINEMATIC_VOICE_SCALE - _CINEMATIC_IDLE_SCALE
    ) * eased

_STATE_VISUALS: dict[AssistantState, OrbVisual] = {
    AssistantState.STANDBY: OrbVisual("#55D6FF", "#3D69FF", "#8D9AAC", 0.42, 0.08, 14, 'Say "Hey Asher"'),
    AssistantState.WAKE_DETECTED: OrbVisual("#74E4FF", "#55D6FF", "#F4F7FB", 1.45, 0.22, 24, "Wake detected"),
    AssistantState.AUTHENTICATING: OrbVisual("#4F8CFF", "#A678FF", "#D6E4FF", 1.15, 0.12, 20, "Verifying speaker"),
    AssistantState.AUTHENTICATED: OrbVisual("#62E6A7", "#55D6FF", "#F4F7FB", 0.70, 0.16, 18, "Authenticated"),
    AssistantState.LISTENING: OrbVisual("#55D6FF", "#3D69FF", "#E8FBFF", 0.85, 0.20, 22, "Listening..."),
    AssistantState.TRANSCRIBING: OrbVisual("#55D6FF", "#6FA8FF", "#DDF7FF", 1.10, 0.10, 18, "Understanding speech..."),
    AssistantState.THINKING: OrbVisual("#A678FF", "#566CFF", "#F1E8FF", 1.05, 0.14, 24, "Thinking..."),
    AssistantState.AWAITING_CONFIRMATION: OrbVisual("#FFB44A", "#FFD166", "#FFF2D2", 0.52, 0.10, 12, "Awaiting confirmation"),
    AssistantState.EXECUTING: OrbVisual("#FFB44A", "#FF8B45", "#FFF0DC", 1.45, 0.15, 26, "Executing..."),
    AssistantState.OBSERVING: OrbVisual("#6CB9FF", "#FFB44A", "#E8F5FF", 1.18, 0.10, 22, "Checking result..."),
    AssistantState.SPEAKING: OrbVisual("#55D6FF", "#A678FF", "#F4F7FB", 0.95, 0.23, 24, "Speaking..."),
    AssistantState.SUCCESS: OrbVisual("#62E6A7", "#49CFA1", "#E8FFF5", 0.50, 0.12, 18, "Done"),
    AssistantState.ERROR: OrbVisual("#FF5C70", "#B64662", "#FFE4E8", 0.62, 0.09, 10, "Something needs attention"),
    AssistantState.OFFLINE: OrbVisual("#6D97A8", "#55D6FF", "#BFD4DC", 0.35, 0.06, 10, "Offline mode - Ollama"),
    AssistantState.STOPPED: OrbVisual("#717985", "#4D535D", "#B7BDC6", 0.12, 0.02, 0, "Stopped safely"),
    AssistantState.LOCKED: OrbVisual("#69717D", "#383E48", "#A9B0BA", 0.18, 0.03, 0, "Authentication required"),
    # Legacy/fallback controller states remain renderable while the real
    # CompanionController uses the precise cinematic vocabulary.
    AssistantState.UNDERSTANDING: OrbVisual("#A678FF", "#566CFF", "#F1E8FF", 0.95, 0.12, 20, "Understanding..."),
    AssistantState.ACTING: OrbVisual("#FFB44A", "#FF8B45", "#FFF0DC", 1.25, 0.13, 22, "Acting..."),
    AssistantState.COMPLETE: OrbVisual("#62E6A7", "#49CFA1", "#E8FFF5", 0.45, 0.10, 16, "Complete"),
}

_SEMANTIC_ACCENT_STATES = frozenset(
    {
        AssistantState.AWAITING_CONFIRMATION,
        AssistantState.SUCCESS,
        AssistantState.COMPLETE,
        AssistantState.ERROR,
        AssistantState.STOPPED,
        AssistantState.LOCKED,
    }
)


def visual_for_state(state: AssistantState) -> OrbVisual:
    """Return the bounded visual profile for ``state`` without importing Qt."""

    return _STATE_VISUALS.get(state, _DEFAULT)


try:
    from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
    from PySide6.QtGui import (
        QColor,
        QBrush,
        QFont,
        QFontDatabase,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QRadialGradient,
    )
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
            self._audio_sample_deadline = time.monotonic()
            self._display_activity = 0.0
            self._last_frame_time = time.monotonic()
            self._reduced_motion = False
            self._animation_intensity = 1.0
            self._interactive_resize = False
            self._display_size = 560
            self._minimum_display_size = 340
            self._maximum_display_size = 900
            self._cinematic_mode = False
            self._transparent_canvas = False
            self._overlay_title = ""
            self._overlay_message = ""
            self._font_family = self._resolve_font_family()
            self.setAccessibleName("Asher voice orb")
            self.setAccessibleDescription(self._visual.label)
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._advance)
            self._sync_timer()

        @property
        def state(self) -> AssistantState:
            return self._state

        @staticmethod
        def _resolve_font_family() -> str:
            """Choose a legible UI font, including Qt's Windows offscreen mode."""

            families = set(QFontDatabase.families())
            for candidate in ("Segoe UI", "Inter", "Arial", "DejaVu Sans"):
                if candidate in families:
                    return candidate

            # The Windows offscreen platform can expose an empty font database
            # even though the normal desktop platform sees system fonts. Load
            # the same system UI face explicitly so visual tests and remote
            # sessions do not paint missing-glyph boxes.
            if os.name == "nt":
                windows_dir = os.environ.get("WINDIR", r"C:\Windows")
                font_path = os.path.join(windows_dir, "Fonts", "segoeui.ttf")
                if os.path.isfile(font_path):
                    font_id = QFontDatabase.addApplicationFont(font_path)
                    if font_id >= 0:
                        loaded = QFontDatabase.applicationFontFamilies(font_id)
                        if loaded:
                            return loaded[0]
            return next(iter(families), "Sans Serif")

        def set_state(self, state: AssistantState) -> None:
            if not isinstance(state, AssistantState):
                state = AssistantState(str(state))
            if state == self._state:
                return
            self._state = state
            self._visual = visual_for_state(state)
            if state not in {
                AssistantState.STANDBY,
                AssistantState.WAKE_DETECTED,
                AssistantState.LISTENING,
                AssistantState.SPEAKING,
            }:
                self._audio_level = 0.0
            self.setAccessibleDescription(
                f"{state.value.replace('_', ' ')}. {self._visual.label}"
            )
            if self._reduced_motion:
                self._display_activity = self._reduced_motion_activity()
            elif not self.isVisible():
                self._display_activity = self._target_activity()
            self._sync_timer()
            self.update()


        def set_cinematic_mode(self, enabled: bool) -> None:
            """Switch between compact workspace rendering and immersive HUD rendering."""

            self._cinematic_mode = bool(enabled)
            self.update()

        def set_transparent_canvas(self, enabled: bool) -> None:
            """Let a parent glass surface show through without changing orb geometry."""

            self._transparent_canvas = bool(enabled)
            self.setAutoFillBackground(False)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
            self.update()

        def set_overlay_text(self, title: str, message: str = "") -> None:
            """Keep truthful state/message copy available to accessibility tools."""

            self._overlay_title = str(title or "").strip()
            self._overlay_message = str(message or "").strip()
            accessible_parts = [
                part
                for part in (
                    self._overlay_title,
                    self._overlay_message or self._visual.label,
                )
                if part
            ]
            self.setAccessibleDescription(". ".join(accessible_parts))
            if self.isVisible():
                self.update()

        def set_audio_level(self, level: float) -> None:
            """Accept a short-lived normalized level without retaining audio."""

            self._audio_level = _bounded_scalar(level)
            self._audio_sample_deadline = time.monotonic() + (
                0.14 if self._audio_level > _AUDIO_ACTIVITY_FLOOR else 0.0
            )
            if not self.isVisible() and not self._reduced_motion:
                self._display_activity = self._target_activity()
            self._sync_timer()
            if (
                self.isVisible()
                and not self._reduced_motion
                and self._animation_intensity > 0.0
            ):
                self.update()

        def set_reduced_motion(self, enabled: bool) -> None:
            self._reduced_motion = bool(enabled)
            if self._reduced_motion:
                self._display_activity = self._reduced_motion_activity()
            self._last_frame_time = time.monotonic()
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

            applied = max(
                self._minimum_display_size,
                min(self._maximum_display_size, int(size)),
            )
            self._display_size = applied
            self.setFixedSize(applied, applied)
            return applied

        def set_display_bounds(self, minimum: int, maximum: int) -> tuple[int, int]:
            """Constrain wheel/gesture scaling to the current viewport."""

            lower = max(340, min(900, int(minimum)))
            upper = max(lower, min(900, int(maximum)))
            self._minimum_display_size = lower
            self._maximum_display_size = upper
            self.set_display_size(self._display_size)
            return lower, upper

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
            self._last_frame_time = time.monotonic()
            self._sync_timer()

        def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt callback
            self._timer.stop()
            super().hideEvent(event)

        def _reduced_motion_activity(self) -> float:
            """Keep state legible without continuously reacting to waveform data."""

            if self._state is AssistantState.SPEAKING:
                return 0.82
            if self._state in {
                AssistantState.WAKE_DETECTED,
                AssistantState.AUTHENTICATING,
                AssistantState.AUTHENTICATED,
            }:
                return 0.30
            return 0.0

        def _target_activity(self) -> float:
            """Combine truthful state and real microphone energy for animation."""

            microphone = audio_activity_for_level(self._audio_level)
            if self._state is AssistantState.SPEAKING:
                # No TTS waveform is exposed.  The canonical SPEAKING state is
                # truthful, so it may expand the orb without inventing audio.
                return 0.90
            if self._state is AssistantState.WAKE_DETECTED:
                return max(0.50, microphone)
            if self._state in {AssistantState.STANDBY, AssistantState.LISTENING}:
                return microphone
            if self._state in {
                AssistantState.AUTHENTICATING,
                AssistantState.AUTHENTICATED,
                AssistantState.TRANSCRIBING,
                AssistantState.UNDERSTANDING,
                AssistantState.THINKING,
                AssistantState.AWAITING_CONFIRMATION,
                AssistantState.EXECUTING,
                AssistantState.ACTING,
                AssistantState.OBSERVING,
            }:
                return 0.24
            return 0.0

        def _render_activity(self) -> float:
            if self._reduced_motion:
                return self._reduced_motion_activity()
            return _bounded_scalar(
                self._display_activity * self._animation_intensity
            )

        def current_visual_scale(self) -> float:
            """Expose the painted scale for accessibility and regression tests."""

            return cinematic_scale_for_activity(self._render_activity())

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
            } or self._audio_level > _AUDIO_ACTIVITY_FLOOR or self._display_activity > 0.02
            interval = 33 if active else 70
            if self._timer.interval() != interval:
                self._timer.setInterval(interval)
            if not self._timer.isActive():
                self._timer.start()

        def _advance(self) -> None:
            if not self.isVisible():
                self._timer.stop()
                return
            now = time.monotonic()
            elapsed = max(0.001, min(0.12, now - self._last_frame_time))
            self._last_frame_time = now
            self._phase = (
                self._phase
                + elapsed * 1.06 * self._visual.speed * self._animation_intensity
            ) % (math.tau * 100)
            self._decay_audio_sample(now, elapsed)
            self._advance_activity(elapsed)
            if self._target_activity() <= 0.0 and self._display_activity < 0.02:
                self._sync_timer()
            self.update()

        def _decay_audio_sample(self, now: float, elapsed: float) -> float:
            """Release a presentation sample if its producer stops refreshing it."""

            if (
                self._audio_level <= 0.0
                or now <= self._audio_sample_deadline
            ):
                return self._audio_level
            decay_time = min(max(0.0, float(elapsed)), now - self._audio_sample_deadline)
            self._audio_level *= math.exp(-decay_time / 0.16)
            if self._audio_level < _AUDIO_ACTIVITY_FLOOR:
                self._audio_level = 0.0
            return self._audio_level

        def _advance_activity(self, elapsed: float) -> float:
            """Advance the fast-attack/soft-release envelope by real time."""

            duration = max(0.0, min(0.25, float(elapsed)))
            target = self._target_activity()
            time_constant = 0.075 if target > self._display_activity else 0.30
            blend = 1.0 - math.exp(-duration / time_constant)
            self._display_activity += (target - self._display_activity) * blend
            if abs(target - self._display_activity) < 0.001:
                self._display_activity = target
            return self._display_activity

        def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt callback
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            width = float(self.width())
            height = float(self.height())
            center = QPointF(width / 2.0, height / 2.0)
            extent = min(width, height)
            activity = self._render_activity()
            breath = 0.0
            if not self._reduced_motion:
                breath = (
                    math.sin(self._phase * 1.42) * 0.012
                    + activity * math.sin(self._phase * 3.15 + 0.8) * 0.026
                    + activity * math.sin(self._phase * 5.7) * 0.010
                ) * self._animation_intensity

            if self._cinematic_mode:
                # The fixed canvas never reflows.  Only the painted glass body
                # grows, from a quiet 70% presence to a voice-active 98% one.
                base_radius = extent * 0.345
                radius = base_radius * self.current_visual_scale() * (1.0 + breath)
                self._paint_cinematic(painter, center, radius, width, height)
                return

            compact_scale = 0.72 + 0.28 * (
                activity * activity * (3.0 - 2.0 * activity)
            )
            base_radius = extent * 0.270
            radius = base_radius * compact_scale * (1.0 + breath)
            if not self._transparent_canvas:
                painter.fillRect(self.rect(), QColor(_GLASS_BACKGROUND))
            self._draw_ambient_glow(painter, center, radius)
            self._draw_particles(painter, center, radius)
            self._draw_orbits(painter, center, radius)
            self._draw_core(painter, center, radius)

        def _paint_cinematic(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
            width: float,
            height: float,
        ) -> None:
            """Render an open atomic-energy form inspired by the owner reference."""

            if not self._transparent_canvas:
                painter.fillRect(self.rect(), QColor("#02050B"))
            self._draw_atomic_ambient(painter, center, radius)
            self._draw_atomic_particles(painter, center, radius, foreground=False)
            self._draw_atomic_ribbons(painter, center, radius, foreground=False)
            self._draw_atomic_mesh(painter, center, radius)
            self._draw_atomic_spikes(painter, center, radius)
            self._draw_atomic_core(painter, center, radius)
            self._draw_atomic_ribbons(painter, center, radius, foreground=True)
            self._draw_atomic_particles(painter, center, radius, foreground=True)
            self._draw_voice_rings(painter, center, radius)
            self._draw_state_accent(painter, center, radius)
            self._draw_cinematic_text(painter, center, radius)

        def _draw_starfield(self, painter: QPainter, width: float, height: float) -> None:
            primary = QColor(_GLASS_PRIMARY)
            for index in range(34):
                x = ((index * 149 + 31) % 997) / 997.0 * width
                y = ((index * 263 + 73) % 991) / 991.0 * height
                drift = math.sin(self._phase * (0.11 + (index % 4) * 0.02) + index) * 3.0
                color = QColor("#EAFBFF" if index % 9 == 0 else primary)
                color.setAlpha(22 + (index % 5) * 16)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                size = 0.55 + (index % 3) * 0.55
                painter.drawEllipse(QPointF(x + drift, y), size, size)

        @staticmethod
        def _atomic_point(
            center: QPointF,
            radius: float,
            coordinates: tuple[float, float],
        ) -> QPointF:
            return QPointF(
                center.x() + coordinates[0] * radius,
                center.y() + coordinates[1] * radius,
            )

        def _draw_atomic_ambient(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Paint bloom around an open form without creating a solid shell."""

            energy = self._cinematic_energy()
            primary = QColor(_GLASS_PRIMARY)
            glow_radius = radius * 1.46
            glow = QRadialGradient(center, glow_radius)
            glow.setColorAt(0.0, QColor(247, 253, 255, round(18 * energy)))
            glow.setColorAt(
                0.12,
                QColor(primary.red(), primary.green(), primary.blue(), round(24 * energy)),
            )
            glow.setColorAt(
                0.34,
                QColor(primary.red(), primary.green(), primary.blue(), round(13 * energy)),
            )
            glow.setColorAt(
                0.68,
                QColor(primary.red(), primary.green(), primary.blue(), round(4 * energy)),
            )
            glow.setColorAt(1.0, QColor(primary.red(), primary.green(), primary.blue(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, glow_radius, glow_radius)

        def _draw_atomic_mesh(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Draw the dense curved/chorded energy lattice around the nucleus."""

            mesh_radius = radius * 0.82
            energy = self._cinematic_energy()
            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
            hot = QColor(_GLASS_HIGHLIGHT)
            clip = QPainterPath()
            clip.addEllipse(center, mesh_radius * 1.035, mesh_radius * 1.035)
            painter.save()
            painter.setClipPath(clip)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Rotated narrow ellipses establish the globe-like atomic volume.
            for index in range(16):
                painter.save()
                painter.translate(center)
                painter.rotate(
                    index * 22.5
                    + math.degrees(self._phase) * (0.010 if index % 2 else -0.008)
                )
                major = mesh_radius * (0.72 + (index % 5) * 0.055)
                minor = mesh_radius * (0.16 + (index % 6) * 0.075)
                color = QColor(secondary if index % 4 == 0 else primary)
                color.setAlpha(round((40 + (index % 4) * 12) * energy))
                pen = QPen(color, 0.55 + (index % 3) * 0.16)
                painter.setPen(pen)
                painter.drawEllipse(QRectF(-major, -minor, major * 2.0, minor * 2.0))
                painter.restore()

            # A deterministic web of curved chords supplies the reference's
            # tangled filament density without storing or loading an asset.
            for index in range(62):
                start_angle = index * 2.399963 + self._phase * (
                    0.010 if index % 2 else -0.008
                )
                angle_span = 0.72 + ((index * 37) % 97) / 97.0 * 2.75
                end_angle = start_angle + angle_span
                start_radius = mesh_radius * (0.79 + ((index * 17) % 19) / 95.0)
                end_radius = mesh_radius * (0.77 + ((index * 29) % 23) / 100.0)
                start = QPointF(
                    center.x() + math.cos(start_angle) * start_radius,
                    center.y() + math.sin(start_angle) * start_radius,
                )
                end = QPointF(
                    center.x() + math.cos(end_angle) * end_radius,
                    center.y() + math.sin(end_angle) * end_radius,
                )
                bend = ((index * 43) % 31 - 15) / 15.0
                control_one_radius = mesh_radius * (0.31 + 0.08 * bend)
                control_two_radius = mesh_radius * (0.33 - 0.07 * bend)
                control_one = QPointF(
                    center.x() + math.cos(start_angle + 0.56) * control_one_radius,
                    center.y() + math.sin(start_angle + 0.56) * control_one_radius,
                )
                control_two = QPointF(
                    center.x() + math.cos(end_angle - 0.61) * control_two_radius,
                    center.y() + math.sin(end_angle - 0.61) * control_two_radius,
                )
                filament = QPainterPath(start)
                filament.cubicTo(control_one, control_two, end)
                color = QColor(hot if index % 11 == 0 else primary)
                color.setAlpha(round((54 + (index % 6) * 13) * energy))
                pen = QPen(color, 0.48 + (index % 4) * 0.15)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(filament)
            painter.restore()

            # Broken boundary fragments imply a globe without enclosing it in
            # the rejected continuous glass ring.
            boundary = QRectF(
                center.x() - mesh_radius,
                center.y() - mesh_radius,
                mesh_radius * 2.0,
                mesh_radius * 2.0,
            )
            # A narrow complete plasma corona keeps the fallback volume
            # readable at every angle. Broken white-hot fragments are layered
            # over it below, so it still feels electrical rather than like a
            # smooth glass outline.
            corona = QColor(primary)
            corona.setAlpha(round(178 * energy))
            corona_pen = QPen(corona, 1.05)
            corona_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(corona_pen)
            painter.drawEllipse(boundary)
            edge = QColor(primary)
            edge.setAlpha(round(74 * energy))
            edge_pen = QPen(edge, 0.8)
            edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(edge_pen)
            for start, span in ((7, 44), (92, 31), (169, 48), (266, 35)):
                painter.drawArc(boundary, start * 16, span * 16)

        def _draw_atomic_spikes(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Emit uneven core-fed rays beyond the filament globe."""

            activity = self._render_activity()
            energy = self._cinematic_energy()
            count = 17 + round(activity * 8)
            primary = QColor(_GLASS_PRIMARY)
            hot = QColor(_GLASS_HIGHLIGHT)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for index in range(count):
                angle = index * 2.399963 + self._phase * (
                    0.020 if index % 2 else -0.014
                )
                inner_radius = radius * (0.06 if index % 4 == 0 else 0.18 + (index % 3) * 0.07)
                reach = (
                    0.73
                    + ((index * 41) % 37) / 100.0
                    + activity * (0.12 + (index % 4) * 0.025)
                )
                outer_radius = radius * reach
                start = QPointF(
                    center.x() + math.cos(angle) * inner_radius,
                    center.y() + math.sin(angle) * inner_radius,
                )
                end = QPointF(
                    center.x() + math.cos(angle) * outer_radius,
                    center.y() + math.sin(angle) * outer_radius,
                )
                glow = QColor(primary)
                glow.setAlpha(round((18 + activity * 22) * energy))
                painter.setPen(QPen(glow, 2.5 + (index % 3) * 0.7))
                painter.drawLine(start, end)
                ray = QLinearGradient(start, end)
                bright = QColor(hot)
                bright.setAlpha(min(255, round((142 + activity * 72) * energy)))
                blue = QColor(primary)
                blue.setAlpha(round((108 + activity * 62) * energy))
                clear = QColor(primary)
                clear.setAlpha(0)
                ray.setColorAt(0.0, bright)
                ray.setColorAt(0.62, blue)
                ray.setColorAt(1.0, clear)
                ray_pen = QPen(QBrush(ray), 0.75 + (index % 4) * 0.18)
                ray_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(ray_pen)
                painter.drawLine(start, end)

        def _draw_atomic_ribbons(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
            *,
            foreground: bool,
        ) -> None:
            """Draw the large broken orbital bands that define the reference."""

            configs = (
                (
                    False,
                    1.00,
                    ((-0.31, -1.32), (-0.78, -1.03), (-1.02, -0.47), (-0.82, 0.06)),
                ),
                (
                    False,
                    0.88,
                    ((0.68, -1.08), (1.08, -0.86), (1.25, -0.34), (1.04, 0.07)),
                ),
                (
                    True,
                    0.80,
                    ((1.09, 0.39), (0.98, 0.84), (0.58, 1.19), (0.13, 1.31)),
                ),
                (
                    True,
                    0.48,
                    ((-1.33, 0.10), (-1.06, -0.05), (-0.83, -0.05), (-0.61, 0.09)),
                ),
            )
            activity = self._render_activity()
            ribbon_activity = max(0.0, min(1.0, (activity - 0.12) / 0.72))
            ribbon_activity = ribbon_activity * ribbon_activity * (
                3.0 - 2.0 * ribbon_activity
            )
            if ribbon_activity <= 0.001:
                return
            energy = self._cinematic_energy()
            primary = QColor("#319CFF")
            cyan = QColor(_GLASS_PRIMARY)
            hot = QColor(_GLASS_HIGHLIGHT)
            for is_foreground, weight, coordinates in configs:
                if is_foreground is not foreground:
                    continue
                points = [
                    self._atomic_point(center, radius, coordinates_value)
                    for coordinates_value in coordinates
                ]
                path = QPainterPath(points[0])
                path.cubicTo(points[1], points[2], points[3])
                width = radius * (0.026 + ribbon_activity * 0.014) * weight
                bloom = QColor(primary)
                bloom.setAlpha(round((20 + ribbon_activity * 25) * energy * ribbon_activity))
                bloom_pen = QPen(bloom, width * 2.15)
                bloom_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                bloom_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(bloom_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                body = QColor(primary)
                body.setAlpha(round((92 + ribbon_activity * 72) * energy * ribbon_activity))
                body_pen = QPen(body, width)
                body_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(body_pen)
                painter.drawPath(path)
                edge = QColor(cyan)
                edge.setAlpha(
                    min(255, round((146 + ribbon_activity * 64) * energy * ribbon_activity))
                )
                edge_pen = QPen(edge, max(1.2, width * 0.22))
                edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(edge_pen)
                painter.drawPath(path)
                highlight = QColor(hot)
                highlight.setAlpha(
                    min(255, round((176 + ribbon_activity * 48) * energy * ribbon_activity))
                )
                painter.setPen(QPen(highlight, max(0.65, width * 0.065)))
                painter.drawPath(path)

        def _draw_atomic_core(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Paint the white-hot nucleus where the filament system converges."""

            energy = self._cinematic_energy()
            activity = self._render_activity()
            primary = QColor(_GLASS_PRIMARY)
            hot = QColor(_GLASS_HIGHLIGHT)
            glow_radius = radius * (0.27 + activity * 0.035)
            glow = QRadialGradient(center, glow_radius)
            glow.setColorAt(0.0, QColor(255, 255, 255, min(255, round(255 * energy))))
            glow.setColorAt(0.10, QColor(hot.red(), hot.green(), hot.blue(), min(255, round(248 * energy))))
            glow.setColorAt(0.28, QColor(primary.red(), primary.green(), primary.blue(), round(185 * energy)))
            glow.setColorAt(0.58, QColor(primary.red(), primary.green(), primary.blue(), round(66 * energy)))
            glow.setColorAt(1.0, QColor(primary.red(), primary.green(), primary.blue(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, glow_radius, glow_radius)
            nucleus = radius * (0.050 + activity * 0.008)
            painter.setBrush(QColor(250, 255, 255, min(255, round(248 * energy))))
            painter.drawEllipse(center, nucleus, nucleus)

        def _draw_atomic_particles(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
            *,
            foreground: bool,
        ) -> None:
            """Wrap the atom in several broken particle belts and loose sparks."""

            if self._animation_intensity <= 0.0 or self._visual.particles <= 0:
                return
            activity = self._render_activity()
            energy = self._cinematic_energy()
            primary = QColor(_GLASS_PRIMARY)
            hot = QColor(_GLASS_HIGHLIGHT)
            count = 158 + round(activity * 178)
            rotations = (-6.0, 51.0, -58.0)
            flattening = (0.20, 0.31, 0.37)
            painter.setPen(Qt.PenStyle.NoPen)
            for index in range(count):
                if (index % 2 == 0) is not foreground:
                    continue
                belt = index % 3
                angle = (
                    index * 2.399963
                    + self._phase * (0.22 + belt * 0.05) * (1 if belt != 1 else -1)
                )
                jitter = ((index * 47) % 53) / 52.0
                orbit = radius * (0.91 + jitter * (0.60 + activity * 0.16))
                local_x = math.cos(angle) * orbit
                local_y = math.sin(angle) * orbit * flattening[belt]
                rotation = math.radians(rotations[belt])
                point = QPointF(
                    center.x() + local_x * math.cos(rotation) - local_y * math.sin(rotation),
                    center.y() + local_x * math.sin(rotation) + local_y * math.cos(rotation),
                )
                flicker = 0.46 + 0.54 * math.sin(self._phase * 1.7 + index * 1.31) ** 2
                color = QColor(hot if index % 13 == 0 else primary)
                color.setAlpha(
                    min(255, round((45 + 124 * flicker + activity * 38) * energy))
                )
                painter.setBrush(color)
                size = 0.35 + (index % 5) * 0.25 + activity * 0.18
                painter.drawEllipse(point, size, size)

            # A sparse asymmetric spray breaks the perfect orbital symmetry.
            loose_count = 18 + round(activity * 20)
            for index in range(loose_count):
                if (index % 2 == 0) is not foreground:
                    continue
                angle = index * 2.399963 + self._phase * 0.06
                distance = radius * (
                    0.64 + ((index * 61) % 71) / 71.0 * (0.94 + activity * 0.18)
                )
                x = center.x() + math.cos(angle) * distance
                y = center.y() + math.sin(angle) * distance * 0.86
                if math.cos(angle) > 0.25:
                    x += radius * activity * 0.13
                point = QPointF(x, y)
                color = QColor(hot if index % 9 == 0 else primary)
                color.setAlpha(round((48 + (index % 5) * 24) * energy))
                painter.setBrush(color)
                size = 0.45 + (index % 4) * 0.37
                painter.drawEllipse(point, size, size)

        def _cinematic_energy(self) -> float:
            """Return a truthful brightness factor for the canonical state."""

            if self._state in {AssistantState.STOPPED, AssistantState.LOCKED}:
                return 0.22
            if self._state in {AssistantState.STANDBY, AssistantState.OFFLINE}:
                return 0.56
            if self._state in {AssistantState.SUCCESS, AssistantState.COMPLETE}:
                return 0.74
            return min(1.22, 0.88 + self._render_activity() * 0.34)

        @staticmethod
        def _hot_tint(color: QColor, amount: float = 0.82) -> QColor:
            """Blend a state color toward white without changing its hue family."""

            mix = max(0.0, min(1.0, amount))
            return QColor(
                round(color.red() + (255 - color.red()) * mix),
                round(color.green() + (255 - color.green()) * mix),
                round(color.blue() + (255 - color.blue()) * mix),
            )

        def _draw_scanline(self, painter: QPainter, center: QPointF, width: float, radius: float) -> None:
            """Draw a restrained energy plane, not a full-screen HUD scan."""

            primary = QColor(_GLASS_PRIMARY)
            y = center.y() + math.sin(self._phase * 0.42) * radius * 0.025
            left = max(0.0, center.x() - radius * 1.42)
            right = min(width, center.x() + radius * 1.42)
            fade = QLinearGradient(left, y, right, y)
            edge = QColor(primary)
            edge.setAlpha(0)
            mid = QColor(primary)
            mid.setAlpha(round(26 * self._cinematic_energy()))
            fade.setColorAt(0.0, edge)
            fade.setColorAt(0.16, edge)
            fade.setColorAt(0.47, mid)
            fade.setColorAt(0.53, mid)
            fade.setColorAt(0.84, edge)
            fade.setColorAt(1.0, edge)
            painter.setPen(QPen(QBrush(fade), 4.5))
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            core = self._hot_tint(primary)
            core.setAlpha(round(42 * self._cinematic_energy()))
            painter.setPen(QPen(core, 0.65))
            painter.drawLine(QPointF(left, y), QPointF(right, y))

        def _draw_cinematic_glow(self, painter: QPainter, center: QPointF, radius: float) -> None:
            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
            energy = self._cinematic_energy()
            glow_radius = radius * 1.43
            gradient = QRadialGradient(center, glow_radius)
            gradient.setColorAt(0.0, QColor(primary.red(), primary.green(), primary.blue(), round(9 * energy)))
            gradient.setColorAt(0.48, QColor(primary.red(), primary.green(), primary.blue(), round(7 * energy)))
            gradient.setColorAt(0.62, QColor(secondary.red(), secondary.green(), secondary.blue(), round(18 * energy)))
            gradient.setColorAt(0.69, QColor(primary.red(), primary.green(), primary.blue(), round(72 * energy)))
            gradient.setColorAt(0.735, QColor(primary.red(), primary.green(), primary.blue(), round(49 * energy)))
            gradient.setColorAt(0.84, QColor(primary.red(), primary.green(), primary.blue(), round(13 * energy)))
            gradient.setColorAt(1.0, QColor(primary.red(), primary.green(), primary.blue(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(center, glow_radius, glow_radius)

            # Broad translucent rings fake optical bloom without blur shaders.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for scale, alpha, thickness in (
                (1.095, 18, 27.0),
                (1.052, 29, 14.0),
                (1.020, 42, 6.0),
            ):
                halo = QColor(primary)
                halo.setAlpha(round(alpha * energy))
                painter.setPen(QPen(halo, thickness))
                painter.drawEllipse(center, radius * scale, radius * scale)

        def _draw_cinematic_body(self, painter: QPainter, center: QPointF, radius: float) -> None:
            """Build the translucent, off-centre depth of an ice-glass sphere."""

            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_DEPTH)
            energy = self._cinematic_energy()
            gradient = QRadialGradient(center, radius)
            gradient.setFocalPoint(
                QPointF(
                    center.x() - radius * 0.30,
                    center.y() - radius * 0.34,
                )
            )
            gradient.setColorAt(0.0, QColor(247, 253, 255, round(48 * energy)))
            gradient.setColorAt(0.16, QColor(primary.red(), primary.green(), primary.blue(), round(31 * energy)))
            gradient.setColorAt(0.43, QColor(9, 35, 58, round(44 * energy)))
            gradient.setColorAt(0.70, QColor(secondary.red(), secondary.green(), secondary.blue(), round(55 * energy)))
            gradient.setColorAt(0.90, QColor(primary.red(), primary.green(), primary.blue(), round(61 * energy)))
            gradient.setColorAt(0.982, QColor(200, 243, 255, round(94 * energy)))
            gradient.setColorAt(1.0, QColor(primary.red(), primary.green(), primary.blue(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(center, radius, radius)

        def _draw_hot_core(self, painter: QPainter, center: QPointF, radius: float) -> None:
            """Paint the compact hot intersection shared by the large filaments."""

            primary = QColor(_GLASS_PRIMARY)
            hot = self._hot_tint(primary, 0.88)
            energy = self._cinematic_energy()
            core_radius = radius * 0.235
            glow = QRadialGradient(center, core_radius)
            glow.setColorAt(0.0, QColor(hot.red(), hot.green(), hot.blue(), min(255, round(255 * energy))))
            glow.setColorAt(0.08, QColor(hot.red(), hot.green(), hot.blue(), min(255, round(230 * energy))))
            glow.setColorAt(0.24, QColor(primary.red(), primary.green(), primary.blue(), round(150 * energy)))
            glow.setColorAt(0.52, QColor(primary.red(), primary.green(), primary.blue(), round(62 * energy)))
            glow.setColorAt(0.78, QColor(primary.red(), primary.green(), primary.blue(), round(19 * energy)))
            glow.setColorAt(1.0, QColor(primary.red(), primary.green(), primary.blue(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, core_radius, core_radius)

            knot = QColor(hot)
            knot.setAlpha(min(255, round(238 * energy)))
            painter.setBrush(knot)
            painter.drawEllipse(center, radius * 0.022, radius * 0.022)

            clip = QPainterPath()
            clip.addEllipse(center, radius * 0.93, radius * 0.93)
            painter.save()
            painter.setClipPath(clip)
            for rotation, span, width in ((9.0, 0.47, 7.0), (68.0, 0.42, 4.5), (132.0, 0.37, 3.2)):
                angle = math.radians(rotation) + self._phase * 0.012
                dx = math.cos(angle) * radius * span
                dy = math.sin(angle) * radius * span
                ray = QLinearGradient(center.x() - dx, center.y() - dy, center.x() + dx, center.y() + dy)
                transparent = QColor(primary)
                transparent.setAlpha(0)
                bright = QColor(primary)
                bright.setAlpha(round(60 * energy))
                ray.setColorAt(0.0, transparent)
                ray.setColorAt(0.42, transparent)
                ray.setColorAt(0.5, bright)
                ray.setColorAt(0.58, transparent)
                ray.setColorAt(1.0, transparent)
                painter.setPen(QPen(QBrush(ray), width))
                painter.drawLine(QPointF(center.x() - dx, center.y() - dy), QPointF(center.x() + dx, center.y() + dy))
            painter.restore()

        def _draw_plasma_shell(self, painter: QPainter, center: QPointF, radius: float) -> None:
            """Layer bounded turbulence into a thick volumetric corona."""

            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
            hot = self._hot_tint(primary, 0.90)
            energy = self._cinematic_energy()
            layers = (
                (1.050, 18, 23.0, 0.018, secondary),
                (1.034, 32, 13.0, 0.014, primary),
                (1.020, 57, 7.0, 0.010, primary),
                (1.010, 104, 3.8, 0.008, primary),
                (1.002, 188, 1.75, 0.005, hot),
                (0.997, 224, 0.85, 0.003, hot),
            )
            for layer, (scale, alpha, thickness, jitter, color_value) in enumerate(layers):
                path = QPainterPath()
                for step in range(_CINEMATIC_SHELL_SEGMENTS + 1):
                    angle = math.tau * step / _CINEMATIC_SHELL_SEGMENTS
                    wave = (
                        0.58 * math.sin(angle * (7 + layer) + self._phase * (2.0 + layer * 0.07))
                        + 0.28 * math.sin(angle * (17 + layer * 2) - self._phase * 1.35 + layer)
                        + 0.14 * math.sin(angle * 31 + self._phase * 0.83)
                    )
                    r = radius * scale * (1.0 + jitter * wave)
                    point = QPointF(center.x() + math.cos(angle) * r, center.y() + math.sin(angle) * r)
                    if step == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
                path.closeSubpath()
                color = QColor(color_value)
                color.setAlpha(min(255, round(alpha * energy)))
                pen = QPen(color, thickness)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)

            # A few hot beads thicken the uneven edge without adding another
            # continuously sampled layer.
            painter.setPen(Qt.PenStyle.NoPen)
            for index in range(13):
                angle = index * 2.399963 + self._phase * (0.025 if index % 2 else -0.019)
                flicker = 0.55 + 0.45 * math.sin(self._phase * 1.3 + index * 1.71) ** 2
                knot_radius = radius * (1.004 + 0.011 * math.sin(index * 2.3 + self._phase))
                point = QPointF(
                    center.x() + math.cos(angle) * knot_radius,
                    center.y() + math.sin(angle) * knot_radius,
                )
                bloom = QColor(primary)
                bloom.setAlpha(round(31 * energy * flicker))
                painter.setBrush(bloom)
                painter.drawEllipse(point, 3.6 + flicker * 2.7, 3.6 + flicker * 2.7)
                core = QColor(hot)
                core.setAlpha(min(255, round(200 * energy * flicker)))
                painter.setBrush(core)
                painter.drawEllipse(point, 0.65 + flicker * 0.75, 0.65 + flicker * 0.75)

            # Sparse intermittent discharges establish the electric character
            # while keeping complexity fixed across states and frame sizes.
            for index in range(_CINEMATIC_SPARK_COUNT):
                phase = self._phase * (1.55 + (index % 3) * 0.11) + index * 2.43
                strength = max(0.0, math.sin(phase))
                if strength < 0.67 or energy < 0.35:
                    continue
                angle = (index / _CINEMATIC_SPARK_COUNT) * math.tau + self._phase * 0.055
                start_r = radius * 1.005
                length = radius * (0.024 + 0.070 * strength)
                p1 = QPointF(center.x() + math.cos(angle) * start_r, center.y() + math.sin(angle) * start_r)
                bend_angle = angle + math.sin(index * 1.7 + self._phase) * 0.12
                mid_r = start_r + length * 0.54
                midpoint = QPointF(
                    center.x() + math.cos(bend_angle - 0.045 * math.sin(index)) * mid_r,
                    center.y() + math.sin(bend_angle - 0.045 * math.sin(index)) * mid_r,
                )
                p2 = QPointF(
                    center.x() + math.cos(bend_angle) * (start_r + length),
                    center.y() + math.sin(bend_angle) * (start_r + length),
                )
                discharge = QPainterPath(p1)
                discharge.lineTo(midpoint)
                discharge.lineTo(p2)
                glow = QColor(primary if index % 2 else secondary)
                glow.setAlpha(round((25 + 52 * strength) * energy))
                painter.setPen(QPen(glow, 3.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(discharge)
                spark = QColor(hot)
                spark.setAlpha(min(255, round((128 + 102 * strength) * energy)))
                painter.setPen(QPen(spark, 0.75))
                painter.drawPath(discharge)

        def _draw_glass_reflections(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Layer restrained specular highlights over the energetic interior."""

            energy = self._cinematic_energy()
            highlight = QColor(_GLASS_HIGHLIGHT)
            primary = QColor(_GLASS_PRIMARY)
            clip = QPainterPath()
            clip.addEllipse(center, radius * 0.965, radius * 0.965)
            painter.save()
            painter.setClipPath(clip)

            # A broad upper-left reflection makes the volume read as glass,
            # with a narrow hot edge layered over the soft streak.
            reflection = QPainterPath(
                QPointF(center.x() - radius * 0.70, center.y() - radius * 0.23)
            )
            reflection.cubicTo(
                QPointF(center.x() - radius * 0.60, center.y() - radius * 0.68),
                QPointF(center.x() - radius * 0.16, center.y() - radius * 0.83),
                QPointF(center.x() + radius * 0.20, center.y() - radius * 0.70),
            )
            soft = QColor(primary)
            soft.setAlpha(round(20 * energy))
            soft_pen = QPen(soft, max(7.0, radius * 0.075))
            soft_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(soft_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(reflection)

            bright = QColor(highlight)
            bright.setAlpha(min(255, round(142 * energy)))
            bright_pen = QPen(bright, max(1.1, radius * 0.007))
            bright_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(bright_pen)
            painter.drawPath(reflection)

            # A low-alpha diagonal sheen gives the transparent body a second
            # surface plane without relying on an expensive blur effect.
            sheen = QLinearGradient(
                center.x() - radius,
                center.y() - radius,
                center.x() + radius,
                center.y() + radius,
            )
            clear = QColor(highlight)
            clear.setAlpha(0)
            milky = QColor(highlight)
            milky.setAlpha(round(28 * energy))
            sheen.setColorAt(0.0, clear)
            sheen.setColorAt(0.30, milky)
            sheen.setColorAt(0.48, clear)
            sheen.setColorAt(1.0, clear)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(sheen))
            painter.drawEllipse(center, radius * 0.94, radius * 0.94)
            painter.restore()

            # Broken rim highlights preserve a glass edge instead of a flat,
            # uniformly outlined disc.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rim = QRectF(
                center.x() - radius * 0.997,
                center.y() - radius * 0.997,
                radius * 1.994,
                radius * 1.994,
            )
            rim_color = QColor(highlight)
            rim_color.setAlpha(min(255, round(196 * energy)))
            rim_pen = QPen(rim_color, max(1.0, radius * 0.006))
            rim_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(rim_pen)
            painter.drawArc(rim, 28 * 16, 104 * 16)
            rim_color.setAlpha(min(255, round(112 * energy)))
            painter.setPen(QPen(rim_color, max(0.8, radius * 0.004)))
            painter.drawArc(rim, 206 * 16, 72 * 16)

        def _draw_state_accent(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Keep actionable/safety states legible on the blue glass base."""

            if self._state not in _SEMANTIC_ACCENT_STATES:
                return
            accent = QColor(self._visual.primary)
            energy = self._cinematic_energy()
            accent.setAlpha(min(255, round(176 * energy)))
            pen = QPen(accent, max(1.2, radius * 0.007))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rim = QRectF(
                center.x() - radius * 1.014,
                center.y() - radius * 1.014,
                radius * 2.028,
                radius * 2.028,
            )
            painter.drawArc(rim, 302 * 16, 34 * 16)
            painter.drawArc(rim, 122 * 16, 22 * 16)

        def _draw_voice_rings(
            self,
            painter: QPainter,
            center: QPointF,
            radius: float,
        ) -> None:
            """Open outward-facing glass arcs only when voice activity is present."""

            activity = self._render_activity()
            if activity <= 0.025 or self._reduced_motion:
                return
            primary = QColor(_GLASS_PRIMARY)
            highlight = QColor(_GLASS_HIGHLIGHT)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for index, rotation in enumerate((-18.0, 56.0, 132.0)):
                painter.save()
                painter.translate(center)
                painter.rotate(
                    rotation
                    + math.degrees(self._phase) * (0.018 if index != 1 else -0.014)
                )
                painter.translate(-center)
                ring_radius = radius * (1.08 + index * 0.075)
                rect = QRectF(
                    center.x() - ring_radius,
                    center.y() - ring_radius * (0.66 + index * 0.04),
                    ring_radius * 2.0,
                    ring_radius * (1.32 + index * 0.08),
                )
                glow = QColor(primary)
                glow.setAlpha(round((18 + index * 5) * activity))
                glow_pen = QPen(glow, 7.0 - index * 1.2)
                glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(glow_pen)
                start = int((38 + index * 91 + self._phase * 4.0) * 16)
                span = int((88 - index * 9) * 16)
                painter.drawArc(rect, start, span)
                thread = QColor(highlight)
                thread.setAlpha(round((92 - index * 13) * activity))
                thread_pen = QPen(thread, 0.85)
                thread_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(thread_pen)
                painter.drawArc(rect, start, span)
                painter.restore()

        def _draw_internal_wisps(self, painter: QPainter, center: QPointF, radius: float) -> None:
            """Add low-alpha turbulent depth without filling the volume."""

            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
            energy = self._cinematic_energy()
            clip = QPainterPath()
            clip.addEllipse(center, radius * 0.91, radius * 0.91)
            painter.save()
            painter.setClipPath(clip)
            for index in range(_CINEMATIC_WISP_COUNT):
                painter.save()
                painter.translate(center)
                painter.rotate(-31.0 + index * 10.5 + math.degrees(self._phase) * (0.002 if index % 2 else -0.002))
                painter.translate(-center)
                offset = (index - (_CINEMATIC_WISP_COUNT - 1) / 2.0) * 0.080
                wave = math.sin(self._phase * 0.32 + index * 1.17) * 0.025
                path = QPainterPath(QPointF(center.x() - radius * 0.88, center.y() + radius * offset))
                path.cubicTo(
                    QPointF(center.x() - radius * 0.52, center.y() + radius * (offset - 0.22 - wave)),
                    QPointF(center.x() - radius * 0.18, center.y() + radius * (offset + 0.18)),
                    QPointF(center.x(), center.y() + radius * (offset * 0.24 - wave)),
                )
                path.cubicTo(
                    QPointF(center.x() + radius * 0.20, center.y() + radius * (offset - 0.19)),
                    QPointF(center.x() + radius * 0.54, center.y() + radius * (offset + 0.23 + wave)),
                    QPointF(center.x() + radius * 0.88, center.y() + radius * offset),
                )
                haze = QColor(secondary if index % 3 == 0 else primary)
                haze.setAlpha(round((11 + index % 3 * 3) * energy))
                painter.setPen(QPen(haze, 8.5 + (index % 2) * 2.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                thread = QColor(primary)
                thread.setAlpha(round((27 + index % 2 * 8) * energy))
                painter.setPen(QPen(thread, 0.55))
                painter.drawPath(path)
                painter.restore()
            painter.restore()

        def _draw_inner_filaments(self, painter: QPainter, center: QPointF, radius: float) -> None:
            """Draw three short core-fed tendrils inside the larger trajectories."""

            clip = QPainterPath()
            clip.addEllipse(center, radius * 0.93, radius * 0.93)
            painter.save()
            painter.setClipPath(clip)
            primary = QColor(_GLASS_PRIMARY)
            hot = self._hot_tint(primary, 0.86)
            energy = self._cinematic_energy()
            for index, rotation in enumerate((17.0, 142.0, 258.0)):
                painter.save()
                painter.translate(center)
                painter.rotate(rotation + math.degrees(self._phase) * (0.018 if index % 2 else -0.014))
                painter.translate(-center)
                bend = math.sin(self._phase * 0.7 + index * 1.9) * 0.035
                path = QPainterPath(QPointF(center.x() + radius * 0.025, center.y() + radius * 0.018 * (index - 1)))
                path.cubicTo(
                    QPointF(center.x() + radius * 0.22, center.y() - radius * (0.07 + bend)),
                    QPointF(center.x() + radius * 0.43, center.y() + radius * (0.13 - bend)),
                    QPointF(center.x() + radius * (0.68 + index * 0.055), center.y() - radius * 0.035),
                )
                glow = QColor(primary)
                glow.setAlpha(round(54 * energy))
                painter.setPen(QPen(glow, 5.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                filament = QColor(hot)
                filament.setAlpha(min(255, round((168 - index * 18) * energy)))
                painter.setPen(QPen(filament, 0.85))
                painter.drawPath(path)
                painter.restore()
            painter.restore()

        def _draw_energy_orbits(self, painter: QPainter, center: QPointF, radius: float) -> None:
            """Draw a few large intersecting plasma trajectories, never a grid."""

            configs = (
                (
                    (-0.89, -0.43), (-0.73, 0.02), (-0.34, 0.25), (-0.025, 0.00),
                    (0.30, -0.25), (0.70, -0.12), (0.88, 0.43),
                ),
                (
                    (-0.61, 0.77), (-0.20, 0.61), (0.08, 0.29), (0.015, 0.015),
                    (-0.14, -0.31), (0.25, -0.66), (0.62, -0.77),
                ),
                (
                    (-0.15, -0.92), (0.27, -0.65), (0.27, -0.25), (0.00, 0.025),
                    (-0.28, 0.33), (-0.23, 0.67), (0.15, 0.92),
                ),
            )
            assert len(configs) == _CINEMATIC_TRAJECTORY_COUNT
            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
            hot = self._hot_tint(primary, 0.88)
            energy = self._cinematic_energy()
            clip = QPainterPath()
            clip.addEllipse(center, radius * 0.945, radius * 0.945)
            painter.save()
            painter.setClipPath(clip)
            for index, (start, c1, c2, midpoint, c3, c4, end) in enumerate(configs):
                painter.save()
                painter.translate(center)
                painter.rotate(math.degrees(self._phase) * 0.008 * (1 if index != 1 else -1))
                painter.translate(-center)
                points = [
                    QPointF(center.x() + value[0] * radius, center.y() + value[1] * radius)
                    for value in (start, c1, c2, midpoint, c3, c4, end)
                ]
                path = QPainterPath(points[0])
                path.cubicTo(points[1], points[2], points[3])
                path.cubicTo(points[4], points[5], points[6])
                outer = QColor(secondary if index == 1 else primary)
                outer.setAlpha(round(33 * energy))
                painter.setPen(QPen(outer, 19.0 - index * 1.8))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                middle = QColor(primary)
                middle.setAlpha(round((104 - index * 7) * energy))
                painter.setPen(QPen(middle, 6.2 - index * 0.55))
                painter.drawPath(path)

                # Two slightly offset threads plus a hot centre make each
                # trajectory read as braided plasma instead of a vector line.
                offset_angle = math.radians(36.0 + index * 57.0)
                offset_x = math.cos(offset_angle) * (2.1 + index * 0.3)
                offset_y = math.sin(offset_angle) * (2.1 + index * 0.3)
                for strand, strand_color in ((-1.0, secondary), (1.0, hot)):
                    painter.save()
                    painter.translate(offset_x * strand, offset_y * strand)
                    color = QColor(strand_color)
                    color.setAlpha(min(255, round((142 - index * 11) * energy)))
                    painter.setPen(QPen(color, 1.15 - index * 0.08))
                    painter.drawPath(path)
                    painter.restore()
                core = QColor(hot)
                core.setAlpha(min(255, round((224 - index * 14) * energy)))
                painter.setPen(QPen(core, 0.88))
                painter.drawPath(path)
                painter.restore()
            painter.restore()

        def _draw_cinematic_particles(self, painter: QPainter, center: QPointF, radius: float) -> None:
            if self._visual.particles <= 0 or self._animation_intensity <= 0.0:
                return
            count = min(18, max(1, round(self._visual.particles * 0.72 * self._animation_intensity)))
            primary = QColor(_GLASS_PRIMARY)
            energy = self._cinematic_energy()
            for index in range(count):
                angle = self._phase * (0.055 + (index % 4) * 0.012) + index * 2.399963
                orbit = radius * (1.10 + 0.62 * (((index * 29) % 53) / 52.0))
                x = center.x() + math.cos(angle) * orbit
                y = center.y() + math.sin(angle) * orbit * 0.82
                color = self._hot_tint(primary) if index % 7 == 0 else QColor(primary)
                color.setAlpha(round((34 + (index % 5) * 18) * energy))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                size = 0.55 + (index % 3) * 0.62
                painter.drawEllipse(QPointF(x, y), size, size)

        def _draw_cinematic_text(self, painter: QPainter, center: QPointF, radius: float) -> None:
            title = self._overlay_title or self._state.value.upper().replace("_", " ")
            message = self._overlay_message or self._visual.label
            title_color = QColor(
                self._visual.primary
                if self._state in _SEMANTIC_ACCENT_STATES
                else "#EDF9FC"
            )
            painter.setPen(title_color)
            font = QFont(self._font_family)
            font.setBold(True)
            font.setPointSize(max(10, int(radius * 0.043)))
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.2)
            painter.setFont(font)
            # Keep the compact energetic knot visible; state copy sits just
            # below it rather than masking the centre of the sphere.
            title_rect = QRectF(
                center.x() - radius * 0.78,
                center.y() + radius * 0.14,
                radius * 1.56,
                34.0,
            )
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, title)

            if message:
                muted = QColor("#93AAB7")
                muted.setAlpha(150)
                painter.setPen(muted)
                small = QFont(self._font_family)
                small.setPointSize(max(7, int(radius * 0.027)))
                painter.setFont(small)
                message_rect = QRectF(
                    center.x() - radius * 0.72,
                    center.y() + radius * 0.31,
                    radius * 1.44,
                    radius * 0.24,
                )
                painter.drawText(message_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, message[:96])

        def _draw_ambient_glow(self, painter: QPainter, center: QPointF, radius: float) -> None:
            primary = QColor(_GLASS_PRIMARY)
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
            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
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
            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_SECONDARY)
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
            core = QColor(_GLASS_HIGHLIGHT)
            primary = QColor(_GLASS_PRIMARY)
            secondary = QColor(_GLASS_DEPTH)
            core.setAlpha(238)
            primary.setAlpha(196)
            secondary.setAlpha(112)
            gradient.setColorAt(0.0, core)
            gradient.setColorAt(0.18, primary)
            gradient.setColorAt(0.60, secondary)
            edge = QColor(_GLASS_PRIMARY)
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


__all__ = [
    "AsherOrbWidget",
    "OrbVisual",
    "QT_AVAILABLE",
    "audio_activity_for_level",
    "cinematic_scale_for_activity",
    "visual_for_state",
]
