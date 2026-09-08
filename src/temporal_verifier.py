from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Deque, Dict, Mapping, Optional, Tuple


TEMPORAL_OUTPUT_KEYS = {
    "temporally_verified",
    "confirmation_count",
    "history_size",
    "verification_mode",
    "blink_match",
}


@dataclass(frozen=True)
class TemporalConfig:
    """Settings for validating that detections are consistent over time."""

    enabled: bool = True
    mode: str = "persistence"
    history_window: int = 5
    min_confirmations: int = 3
    association_radius_px: float = 25.0
    blink_enabled: bool = False
    expected_period_frames: int = 6
    tolerance_frames: int = 2


@dataclass(frozen=True)
class TemporalObservation:
    """One frame of temporal evidence for a candidate."""

    frame_index: int
    present: bool
    x_px: Optional[float] = None
    y_px: Optional[float] = None
    candidate_id: Optional[int] = None


def validate_temporal_config(config: TemporalConfig) -> TemporalConfig:
    """Validate temporal-verification settings."""
    if config.mode not in {"persistence", "blink"}:
        raise ValueError("temporal mode must be 'persistence' or 'blink'")
    if config.history_window <= 0:
        raise ValueError("history_window must be positive")
    if config.min_confirmations <= 0:
        raise ValueError("min_confirmations must be positive")
    if config.min_confirmations > config.history_window and config.mode == "persistence":
        raise ValueError("min_confirmations cannot exceed history_window in persistence mode")
    if config.association_radius_px <= 0:
        raise ValueError("association_radius_px must be positive")
    if config.expected_period_frames <= 0:
        raise ValueError("expected_period_frames must be positive")
    if config.tolerance_frames < 0:
        raise ValueError("tolerance_frames must be zero or positive")
    return config


def temporal_config_from_mapping(config: TemporalConfig | Mapping[str, object]) -> TemporalConfig:
    """Build a TemporalConfig from a dataclass or YAML-style mapping."""
    if isinstance(config, TemporalConfig):
        return validate_temporal_config(config)

    history_window = int(config.get("history_window", config.get("max_history", 5)))
    min_confirmations = int(config.get("min_confirmations", config.get("min_history", 3)))
    mode = str(config.get("mode", "persistence"))
    blink_enabled = bool(config.get("blink_enabled", mode == "blink"))
    return validate_temporal_config(
        TemporalConfig(
            enabled=bool(config.get("enabled", True)),
            mode=mode,
            history_window=history_window,
            min_confirmations=min_confirmations,
            association_radius_px=float(config.get("association_radius_px", 25.0)),
            blink_enabled=blink_enabled,
            expected_period_frames=int(config.get("expected_period_frames", 6)),
            tolerance_frames=int(config.get("tolerance_frames", 2)),
        )
    )


def observation_from_mapping(
    frame_index: int,
    observation: Optional[Mapping[str, object]],
) -> TemporalObservation:
    """Normalize optional candidate data into a TemporalObservation."""
    if observation is None:
        return TemporalObservation(frame_index=frame_index, present=False)

    x_value = observation.get("x_px", observation.get("x"))
    y_value = observation.get("y_px", observation.get("y"))
    if x_value is None or y_value is None:
        return TemporalObservation(frame_index=frame_index, present=False)
    candidate_id = observation.get("candidate_id")
    return TemporalObservation(
        frame_index=frame_index,
        present=True,
        x_px=float(x_value),
        y_px=float(y_value),
        candidate_id=int(candidate_id) if candidate_id is not None else None,
    )


class TemporalVerifier:
    """Verify target persistence or blink timing across recent frames."""

    def __init__(self, config: TemporalConfig | Mapping[str, object] | None = None) -> None:
        self.config = temporal_config_from_mapping(config or TemporalConfig())
        self.history: Deque[TemporalObservation] = deque(maxlen=self.config.history_window)

    def reset(self) -> None:
        """Clear all temporal history."""
        self.history.clear()

    def update(
        self,
        frame_index: int,
        observation: Optional[Mapping[str, object]] = None,
        expected_position: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, object]:
        """Add one observation and return temporal verification status."""
        normalized = observation_from_mapping(frame_index, observation)
        self.history.append(normalized)

        if not self.config.enabled:
            result = self._result(True, self.present_count(), None)
            validate_temporal_result(result)
            return result

        if self.config.mode == "blink" or self.config.blink_enabled:
            blink_match = self._blink_match()
            confirmation_count = self.present_count()
            verified = bool(blink_match and confirmation_count >= min(2, self.config.min_confirmations))
        else:
            blink_match = None
            confirmation_count = self._persistence_count(expected_position)
            verified = confirmation_count >= self.config.min_confirmations

        result = self._result(verified, confirmation_count, blink_match)
        validate_temporal_result(result)
        return result

    def present_count(self) -> int:
        """Count present observations inside the history window."""
        return sum(1 for item in self.history if item.present)

    def _persistence_count(self, expected_position: Optional[Tuple[float, float]]) -> int:
        """Count consecutive recent observations following a smooth local path."""
        if not self.history or not self.history[-1].present:
            return 0

        current = self.history[-1]
        if expected_position is not None:
            distance = hypot(float(current.x_px) - expected_position[0], float(current.y_px) - expected_position[1])
            if distance > self.config.association_radius_px:
                return 0

        count = 0
        previous: Optional[TemporalObservation] = None
        for item in reversed(self.history):
            if not item.present:
                break
            if previous is not None:
                step = hypot(float(item.x_px) - float(previous.x_px), float(item.y_px) - float(previous.y_px))
                if step > self.config.association_radius_px:
                    break
            count += 1
            previous = item
        return count

    def _blink_match(self) -> bool:
        """Check whether recent present observations match the expected blink period."""
        present_frames = [item.frame_index for item in self.history if item.present]
        if len(present_frames) < 2:
            return False

        lower = self.config.expected_period_frames - self.config.tolerance_frames
        upper = self.config.expected_period_frames + self.config.tolerance_frames
        gaps = [later - earlier for earlier, later in zip(present_frames, present_frames[1:])]
        return any(lower <= gap <= upper for gap in gaps)

    def _result(self, verified: bool, confirmation_count: int, blink_match: Optional[bool]) -> Dict[str, object]:
        """Create a stable JSON-friendly temporal result."""
        return {
            "temporally_verified": bool(verified),
            "confirmation_count": int(confirmation_count),
            "history_size": len(self.history),
            "verification_mode": self.config.mode,
            "blink_match": blink_match,
        }


def validate_temporal_result(result: Mapping[str, object]) -> None:
    """Validate the stable temporal output schema."""
    missing = sorted(TEMPORAL_OUTPUT_KEYS - set(result.keys()))
    if missing:
        raise ValueError(f"temporal result missing keys: {missing}")
