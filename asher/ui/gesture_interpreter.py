"""Deterministic hand-landmark interpretation for the companion renderer.

The module is intentionally independent of Qt and camera capture.  It accepts
already-derived landmarks and returns small, bounded presentation values that
are straightforward to serialize across QWebChannel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math


_DEFAULT_DT = 1.0 / 60.0
_MIN_DT = 1.0 / 240.0
_MAX_DT = 0.1
_MIN_HAND_SIZE = 1.0e-4


@dataclass(frozen=True, slots=True)
class HandGestureFrame:
    """One labelled MediaPipe-style hand observation.

    ``landmarks`` must contain at least the standard 21 entries.  Individual
    entries may be objects with ``x``/``y``/``z`` attributes, mappings, or
    numeric sequences.  Only the wrist, thumb tip, index tip, and middle MCP
    entries are consumed.
    """

    label: str
    landmarks: Sequence[object]


@dataclass(frozen=True, slots=True)
class GestureOutput:
    """Bounded presentation values produced for one video frame."""

    mode: str
    rotation_x: float
    rotation_y: float
    expansion: float
    pinched_hands: tuple[str, ...]
    rotation_velocity_x: float
    rotation_velocity_y: float
    tracked_hands: tuple[str, ...]

    @property
    def rotation_delta_x(self) -> float:
        return self.rotation_x

    @property
    def rotation_delta_y(self) -> float:
        return self.rotation_y

    @property
    def rotation_delta(self) -> tuple[float, float]:
        return (self.rotation_x, self.rotation_y)

    @property
    def gesture_expansion(self) -> float:
        return self.expansion

    @property
    def pinched_labels(self) -> tuple[str, ...]:
        return self.pinched_hands

    @property
    def tracked_labels(self) -> tuple[str, ...]:
        return self.tracked_hands

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible payload with renderer-friendly keys."""

        return {
            "mode": self.mode,
            "rotationX": self.rotation_x,
            "rotationY": self.rotation_y,
            "rotationDeltaX": self.rotation_x,
            "rotationDeltaY": self.rotation_y,
            "rotationVelocityX": self.rotation_velocity_x,
            "rotationVelocityY": self.rotation_velocity_y,
            "expansion": self.expansion,
            "gestureExpansion": self.expansion,
            "pinchedHands": list(self.pinched_hands),
            "pinchedLabels": list(self.pinched_hands),
            "trackedHands": list(self.tracked_hands),
            "trackedLabels": list(self.tracked_hands),
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class _Point:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class _Observation:
    key: str
    label: str
    center: tuple[float, float]
    pinch_ratio: float


@dataclass(slots=True)
class _HandState:
    label: str
    pinched: bool = False
    filtered_center: tuple[float, float] | None = None


class GestureInterpreter:
    """Convert at most two labelled hands into smooth visual gesture values.

    A pinch is normalized by wrist-to-middle-MCP hand size, so the default
    hysteresis thresholds remain useful as a hand approaches the camera.  The
    pinch midpoint's X coordinate is mirrored before filtering to match the
    usual mirrored-camera interaction.
    """

    def __init__(
        self,
        *,
        pinch_on_ratio: float = 0.32,
        pinch_off_ratio: float = 0.45,
        pointer_filter_time: float = 0.12,
        expansion_filter_time: float = 0.11,
        expansion_release_time: float = 0.20,
        rotation_response_time: float = 0.055,
        rotation_release_time: float = 0.16,
        rotation_dead_zone: float = 0.0035,
        rotation_gain: float = 5.0,
        max_rotation_delta: float = 0.065,
        max_rotation_velocity: float = 2.4,
        expansion_travel: float = 0.30,
    ) -> None:
        values = {
            "pinch_on_ratio": pinch_on_ratio,
            "pinch_off_ratio": pinch_off_ratio,
            "pointer_filter_time": pointer_filter_time,
            "expansion_filter_time": expansion_filter_time,
            "expansion_release_time": expansion_release_time,
            "rotation_response_time": rotation_response_time,
            "rotation_release_time": rotation_release_time,
            "rotation_dead_zone": rotation_dead_zone,
            "rotation_gain": rotation_gain,
            "max_rotation_delta": max_rotation_delta,
            "max_rotation_velocity": max_rotation_velocity,
            "expansion_travel": expansion_travel,
        }
        normalized: dict[str, float] = {}
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number")
            normalized[name] = number

        if not 0.0 < normalized["pinch_on_ratio"] < normalized["pinch_off_ratio"]:
            raise ValueError("pinch thresholds must be positive and ordered")
        for name in (
            "pointer_filter_time",
            "expansion_filter_time",
            "expansion_release_time",
            "rotation_response_time",
            "rotation_release_time",
            "rotation_gain",
            "max_rotation_delta",
            "max_rotation_velocity",
            "expansion_travel",
        ):
            if normalized[name] <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        if normalized["rotation_dead_zone"] < 0.0:
            raise ValueError("rotation_dead_zone cannot be negative")

        self.pinch_on_ratio = normalized["pinch_on_ratio"]
        self.pinch_off_ratio = normalized["pinch_off_ratio"]
        self.pointer_filter_time = normalized["pointer_filter_time"]
        self.expansion_filter_time = normalized["expansion_filter_time"]
        self.expansion_release_time = normalized["expansion_release_time"]
        self.rotation_response_time = normalized["rotation_response_time"]
        self.rotation_release_time = normalized["rotation_release_time"]
        self.rotation_dead_zone = normalized["rotation_dead_zone"]
        self.rotation_gain = normalized["rotation_gain"]
        self.max_rotation_delta = normalized["max_rotation_delta"]
        self.max_rotation_velocity = normalized["max_rotation_velocity"]
        self.expansion_travel = normalized["expansion_travel"]

        self._hands: dict[str, _HandState] = {}
        self._mode = "idle"
        self._rotation_label: str | None = None
        self._rotation_anchor: tuple[float, float] | None = None
        self._rotation_velocity = (0.0, 0.0)
        self._expansion_pair: tuple[str, str] | None = None
        self._expansion_baseline: float | None = None
        self._expansion = 0.0

    def reset(self) -> None:
        """Clear all temporal state before starting a new camera session."""

        self._hands.clear()
        self._mode = "idle"
        self._rotation_label = None
        self._rotation_anchor = None
        self._rotation_velocity = (0.0, 0.0)
        self._expansion_pair = None
        self._expansion_baseline = None
        self._expansion = 0.0

    def process(
        self,
        hands: Iterable[HandGestureFrame | Mapping[str, object] | object] | None,
        dt_seconds: float = _DEFAULT_DT,
    ) -> GestureOutput:
        """Interpret one frame without raising for malformed observations."""

        frame_dt = _safe_dt(dt_seconds)
        observations = _observations(hands)
        observation_keys = {item.key for item in observations}

        # A disappeared or malformed hand loses its hysteresis state.  If it
        # returns, it must establish a new pinch and motion anchor.
        for key in tuple(self._hands):
            if key not in observation_keys:
                del self._hands[key]

        filter_alpha = _smoothing_alpha(frame_dt, self.pointer_filter_time)
        for item in observations:
            state = self._hands.get(item.key)
            if state is None:
                state = _HandState(label=item.label)
                self._hands[item.key] = state
            else:
                state.label = item.label

            if state.pinched:
                state.pinched = item.pinch_ratio < self.pinch_off_ratio
            else:
                state.pinched = item.pinch_ratio <= self.pinch_on_ratio

            if state.filtered_center is None:
                state.filtered_center = item.center
            else:
                state.filtered_center = _lerp_point(
                    state.filtered_center,
                    item.center,
                    filter_alpha,
                )

        pinched = tuple(
            key
            for key in sorted(self._hands)
            if self._hands[key].pinched
            and self._hands[key].filtered_center is not None
        )

        if len(pinched) == 2:
            rotation_delta = self._process_expansion(pinched, frame_dt)
            mode = "expand"
        elif len(pinched) == 1:
            rotation_delta = self._process_rotation(pinched[0], frame_dt)
            self._release_expansion(frame_dt)
            mode = "rotate"
        else:
            rotation_delta = self._release_rotation(frame_dt)
            self._release_expansion(frame_dt)
            self._rotation_label = None
            self._rotation_anchor = None
            self._clear_expansion_baseline()
            mode = "idle"

        self._mode = mode
        pinched_labels = tuple(self._hands[key].label for key in pinched)
        tracked_labels = tuple(
            self._hands[key].label for key in sorted(self._hands)
        )
        return GestureOutput(
            mode=mode,
            rotation_x=rotation_delta[0],
            rotation_y=rotation_delta[1],
            expansion=_clamp(self._expansion, 0.0, 1.0),
            pinched_hands=pinched_labels,
            rotation_velocity_x=self._rotation_velocity[0],
            rotation_velocity_y=self._rotation_velocity[1],
            tracked_hands=tracked_labels,
        )

    def _process_rotation(
        self,
        key: str,
        dt: float,
    ) -> tuple[float, float]:
        center = self._hands[key].filtered_center
        if center is None:
            return self._release_rotation(dt)

        entering_rotation = self._mode != "rotate" or self._rotation_label != key
        if entering_rotation or self._rotation_anchor is None:
            self._rotation_label = key
            self._rotation_anchor = center
            self._rotation_velocity = (0.0, 0.0)
            self._clear_expansion_baseline()
            return (0.0, 0.0)

        movement = (
            center[0] - self._rotation_anchor[0],
            center[1] - self._rotation_anchor[1],
        )
        self._rotation_anchor = center
        movement = _radial_dead_zone(movement, self.rotation_dead_zone)

        # Vertical midpoint motion drives pitch.  Mirrored horizontal motion
        # drives yaw.  Both targets are bounded before velocity smoothing.
        target_delta = (
            _clamp(
                movement[1] * self.rotation_gain,
                -self.max_rotation_delta,
                self.max_rotation_delta,
            ),
            _clamp(
                movement[0] * self.rotation_gain,
                -self.max_rotation_delta,
                self.max_rotation_delta,
            ),
        )
        target_velocity = (
            _clamp(
                target_delta[0] / dt,
                -self.max_rotation_velocity,
                self.max_rotation_velocity,
            ),
            _clamp(
                target_delta[1] / dt,
                -self.max_rotation_velocity,
                self.max_rotation_velocity,
            ),
        )
        response = _smoothing_alpha(dt, self.rotation_response_time)
        self._rotation_velocity = _lerp_point(
            self._rotation_velocity,
            target_velocity,
            response,
        )
        self._rotation_velocity = (
            _clamp(
                self._rotation_velocity[0],
                -self.max_rotation_velocity,
                self.max_rotation_velocity,
            ),
            _clamp(
                self._rotation_velocity[1],
                -self.max_rotation_velocity,
                self.max_rotation_velocity,
            ),
        )
        return self._bounded_rotation_delta(dt)

    def _process_expansion(
        self,
        pinched: tuple[str, str],
        dt: float,
    ) -> tuple[float, float]:
        pair = tuple(sorted(pinched))
        first = self._hands[pair[0]].filtered_center
        second = self._hands[pair[1]].filtered_center
        if first is None or second is None:
            self._release_expansion(dt)
            return self._release_rotation(dt)

        distance = math.hypot(first[0] - second[0], first[1] - second[1])
        if self._mode != "expand" or self._expansion_pair != pair:
            self._expansion_pair = pair
            self._expansion_baseline = distance
            self._rotation_label = None
            self._rotation_anchor = None

        baseline = self._expansion_baseline
        if baseline is None:
            baseline = distance
            self._expansion_baseline = baseline
        target = _clamp((distance - baseline) / self.expansion_travel, 0.0, 1.0)
        alpha = _smoothing_alpha(dt, self.expansion_filter_time)
        self._expansion += (target - self._expansion) * alpha
        self._expansion = _clamp(self._expansion, 0.0, 1.0)
        return self._release_rotation(dt)

    def _release_rotation(self, dt: float) -> tuple[float, float]:
        decay = math.exp(-dt / self.rotation_release_time)
        self._rotation_velocity = (
            self._rotation_velocity[0] * decay,
            self._rotation_velocity[1] * decay,
        )
        if abs(self._rotation_velocity[0]) < 1.0e-6:
            self._rotation_velocity = (0.0, self._rotation_velocity[1])
        if abs(self._rotation_velocity[1]) < 1.0e-6:
            self._rotation_velocity = (self._rotation_velocity[0], 0.0)
        return self._bounded_rotation_delta(dt)

    def _bounded_rotation_delta(self, dt: float) -> tuple[float, float]:
        return (
            _clamp(
                self._rotation_velocity[0] * dt,
                -self.max_rotation_delta,
                self.max_rotation_delta,
            ),
            _clamp(
                self._rotation_velocity[1] * dt,
                -self.max_rotation_delta,
                self.max_rotation_delta,
            ),
        )

    def _release_expansion(self, dt: float) -> None:
        alpha = _smoothing_alpha(dt, self.expansion_release_time)
        self._expansion += (0.0 - self._expansion) * alpha
        if self._expansion < 1.0e-6:
            self._expansion = 0.0
        self._clear_expansion_baseline()

    def _clear_expansion_baseline(self) -> None:
        self._expansion_pair = None
        self._expansion_baseline = None


def _observations(
    hands: Iterable[HandGestureFrame | Mapping[str, object] | object] | None,
) -> tuple[_Observation, ...]:
    if hands is None:
        return ()
    if isinstance(hands, (HandGestureFrame, Mapping)):
        candidates: Iterable[object] = (hands,)
    elif (
        isinstance(hands, (tuple, list))
        and len(hands) == 2
        and isinstance(hands[0], str)
    ):
        candidates = (hands,)
    elif isinstance(hands, (str, bytes)):
        return ()
    else:
        try:
            candidates = iter(hands)
        except Exception:
            candidates = (hands,)

    by_key: dict[str, _Observation] = {}
    try:
        for candidate in candidates:
            try:
                item = _observation(candidate)
            except Exception:
                item = None
            if item is not None and item.key not in by_key:
                by_key[item.key] = item
    except Exception:
        # A failing live iterator is treated as an unusable frame.
        return ()
    return tuple(by_key[key] for key in sorted(by_key)[:2])


def _observation(record: object) -> _Observation | None:
    if isinstance(record, Mapping):
        label_value = record.get("label")
        landmarks_value = record.get("landmarks")
    elif isinstance(record, (tuple, list)) and len(record) == 2:
        label_value, landmarks_value = record
    else:
        label_value = getattr(record, "label", None)
        landmarks_value = getattr(record, "landmarks", None)

    if not isinstance(label_value, str):
        return None
    label = label_value.strip()
    if not label or isinstance(landmarks_value, (str, bytes, Mapping)):
        return None
    try:
        if len(landmarks_value) < 21:  # type: ignore[arg-type]
            return None
        wrist = _point(landmarks_value[0])  # type: ignore[index]
        thumb = _point(landmarks_value[4])  # type: ignore[index]
        index = _point(landmarks_value[8])  # type: ignore[index]
        middle_mcp = _point(landmarks_value[9])  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if wrist is None or thumb is None or index is None or middle_mcp is None:
        return None

    hand_size = _distance(wrist, middle_mcp)
    if hand_size <= _MIN_HAND_SIZE:
        return None
    pinch_ratio = _distance(thumb, index) / hand_size
    if not math.isfinite(pinch_ratio):
        return None
    center = (1.0 - ((thumb.x + index.x) * 0.5), (thumb.y + index.y) * 0.5)
    return _Observation(
        key=label.casefold(),
        label=label,
        center=center,
        pinch_ratio=pinch_ratio,
    )


def _point(value: object) -> _Point | None:
    if isinstance(value, Mapping):
        x_value = value.get("x")
        y_value = value.get("y")
        z_value = value.get("z", 0.0)
    elif hasattr(value, "x") and hasattr(value, "y"):
        x_value = getattr(value, "x")
        y_value = getattr(value, "y")
        z_value = getattr(value, "z", 0.0)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            return None
        x_value = value[0]
        y_value = value[1]
        z_value = value[2] if len(value) > 2 else 0.0
    else:
        return None
    try:
        x = float(x_value)  # type: ignore[arg-type]
        y = float(y_value)  # type: ignore[arg-type]
        z = float(z_value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(number) for number in (x, y, z)):
        return None
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    return _Point(x=x, y=y, z=z)


def _distance(first: _Point, second: _Point) -> float:
    return math.sqrt(
        ((first.x - second.x) ** 2)
        + ((first.y - second.y) ** 2)
        + ((first.z - second.z) ** 2)
    )


def _safe_dt(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return _DEFAULT_DT
    if not math.isfinite(number) or number <= 0.0:
        return _DEFAULT_DT
    return _clamp(number, _MIN_DT, _MAX_DT)


def _smoothing_alpha(dt: float, time_constant: float) -> float:
    return 1.0 - math.exp(-dt / time_constant)


def _lerp_point(
    current: tuple[float, float],
    target: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    return (
        current[0] + ((target[0] - current[0]) * alpha),
        current[1] + ((target[1] - current[1]) * alpha),
    )


def _radial_dead_zone(
    movement: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    magnitude = math.hypot(movement[0], movement[1])
    if magnitude <= radius or magnitude == 0.0:
        return (0.0, 0.0)
    retained = (magnitude - radius) / magnitude
    return (movement[0] * retained, movement[1] * retained)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


__all__ = ["GestureInterpreter", "GestureOutput", "HandGestureFrame"]
