from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - listed in requirements.txt
    yaml = None

try:
    from .pipeline import SingleFramePipeline, load_config, load_frame
    from .temporal_verifier import TemporalConfig, TemporalVerifier, temporal_config_from_mapping
except ImportError:  # pragma: no cover - allows direct script execution
    from pipeline import SingleFramePipeline, load_config, load_frame
    from temporal_verifier import TemporalConfig, TemporalVerifier, temporal_config_from_mapping


class LockState(str, Enum):
    """Tracker lock-state machine states."""

    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    COASTING = "COASTING"
    LOST = "LOST"


TRACKING_OUTPUT_KEYS = {
    "frame_index",
    "lock_state",
    "target_found",
    "measurement_available",
    "temporally_verified",
    "measured_x_px",
    "measured_y_px",
    "filtered_x_px",
    "filtered_y_px",
    "predicted_x_px",
    "predicted_y_px",
    "velocity_x_px_per_frame",
    "velocity_y_px_per_frame",
    "control_error_x",
    "control_error_y",
    "confidence",
    "candidate_count",
    "associated_candidate_id",
    "distance_to_prediction_px",
    "confirmation_count",
    "track_age_frames",
    "missed_frames",
    "using_prediction_only",
    "just_locked",
    "just_lost",
    "just_reacquired",
    "processing_time_ms",
}


@dataclass(frozen=True)
class KalmanFilterConfig:
    """Constant-velocity Kalman filter settings."""

    dt: float = 1.0
    process_noise: float = 0.03
    measurement_noise: float = 4.0
    initial_position_uncertainty: float = 10.0
    initial_velocity_uncertainty: float = 100.0


@dataclass(frozen=True)
class AssociationConfig:
    """Settings for associating current candidates with the predicted target."""

    gate_px: float = 60.0
    confidence_weight: float = 0.40
    position_weight: float = 0.60


@dataclass(frozen=True)
class TrackingConfig:
    """Settings for the complete temporal tracker."""

    frame_width: int = 640
    frame_height: int = 480
    target_id: str = "Terminal_B"
    confidence_threshold: float = 0.55
    max_missed_frames: int = 8
    kalman: KalmanFilterConfig = field(default_factory=KalmanFilterConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)


@dataclass(frozen=True)
class CandidateObservation:
    """One Phase 6 candidate normalized for tracking."""

    candidate_id: int
    x_px: float
    y_px: float
    correct_probability: float
    false_probability: float
    cv_baseline_score: float
    fused_score: float
    bbox: Optional[Dict[str, int]] = None

    @property
    def confidence(self) -> float:
        """Use the correct-beacon probability as the tracking confidence."""
        return self.correct_probability

    def to_temporal_mapping(self) -> Dict[str, object]:
        """Return the fields needed by the temporal verifier."""
        return {"candidate_id": self.candidate_id, "x_px": self.x_px, "y_px": self.y_px}


@dataclass(frozen=True)
class AssociationResult:
    """Selected candidate plus association diagnostics."""

    candidate: CandidateObservation
    distance_to_prediction_px: Optional[float]
    position_score: float
    association_score: float


def validate_kalman_config(config: KalmanFilterConfig) -> KalmanFilterConfig:
    """Validate Kalman filter settings."""
    if config.dt <= 0:
        raise ValueError("dt must be positive")
    if config.process_noise < 0:
        raise ValueError("process_noise must be zero or positive")
    if config.measurement_noise <= 0:
        raise ValueError("measurement_noise must be positive")
    if config.initial_position_uncertainty <= 0 or config.initial_velocity_uncertainty <= 0:
        raise ValueError("initial uncertainties must be positive")
    return config


def validate_association_config(config: AssociationConfig) -> AssociationConfig:
    """Validate association gate and score weights."""
    if config.gate_px <= 0:
        raise ValueError("association gate must be positive")
    if not 0.0 <= config.confidence_weight <= 1.0 or not 0.0 <= config.position_weight <= 1.0:
        raise ValueError("association weights must be between 0 and 1")
    total = config.confidence_weight + config.position_weight
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"association weights must sum to 1.0, got {total:.6f}")
    return config


def validate_tracking_config(config: TrackingConfig) -> TrackingConfig:
    """Validate all tracker settings."""
    if config.frame_width <= 0 or config.frame_height <= 0:
        raise ValueError("frame width and height must be positive")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if config.max_missed_frames < 0:
        raise ValueError("max_missed_frames must be zero or positive")
    validate_kalman_config(config.kalman)
    validate_association_config(config.association)
    temporal_config_from_mapping(config.temporal)
    return config


class ConstantVelocityKalmanFilter:
    """A visible, student-friendly constant-velocity Kalman filter.

    State vector: [x, y, velocity_x, velocity_y]
    Measurement vector: [x, y]
    """

    def __init__(self, config: KalmanFilterConfig | Mapping[str, object] | None = None) -> None:
        if isinstance(config, Mapping):
            config = KalmanFilterConfig(
                dt=float(config.get("dt", 1.0)),
                process_noise=float(config.get("process_noise", 0.03)),
                measurement_noise=float(config.get("measurement_noise", 4.0)),
            )
        self.config = validate_kalman_config(config or KalmanFilterConfig())
        self.state: Optional[np.ndarray] = None
        self.error_covariance: Optional[np.ndarray] = None

        dt = self.config.dt
        # Transition matrix: constant velocity motion model.
        self.transition_matrix = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Measurement matrix: camera measurement observes x and y only.
        self.measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        # Process-noise covariance: uncertainty from unknown acceleration.
        q = self.config.process_noise
        self.process_noise_covariance = q * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                [dt**3 / 2.0, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2.0, 0.0, dt**2],
            ],
            dtype=np.float64,
        )
        # Measurement-noise covariance: camera coordinate noise.
        self.measurement_noise_covariance = self.config.measurement_noise * np.eye(2, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        """Return whether the filter has a valid state."""
        return self.state is not None and self.error_covariance is not None

    def initialize(self, x_px: float, y_px: float) -> None:
        """Initialize state from the first accepted measurement."""
        self.state = np.array([x_px, y_px, 0.0, 0.0], dtype=np.float64)
        # Error covariance: high velocity uncertainty before observing motion.
        self.error_covariance = np.diag(
            [
                self.config.initial_position_uncertainty,
                self.config.initial_position_uncertainty,
                self.config.initial_velocity_uncertainty,
                self.config.initial_velocity_uncertainty,
            ]
        ).astype(np.float64)

    def reset(self) -> None:
        """Clear the filter state."""
        self.state = None
        self.error_covariance = None

    def predict(self) -> Dict[str, float]:
        """Prediction step: advance state using the transition matrix."""
        if not self.initialized:
            raise RuntimeError("Kalman filter must be initialized before prediction")
        assert self.state is not None and self.error_covariance is not None
        self.state = self.transition_matrix @ self.state
        self.error_covariance = (
            self.transition_matrix @ self.error_covariance @ self.transition_matrix.T
            + self.process_noise_covariance
        )
        return self.current_state()

    def correct(self, x_px: float, y_px: float) -> Dict[str, float]:
        """Correction step: update predicted state with a real measurement."""
        if not self.initialized:
            self.initialize(x_px, y_px)
            return self.current_state()

        assert self.state is not None and self.error_covariance is not None
        measurement = np.array([x_px, y_px], dtype=np.float64)
        innovation = measurement - (self.measurement_matrix @ self.state)
        innovation_covariance = (
            self.measurement_matrix @ self.error_covariance @ self.measurement_matrix.T
            + self.measurement_noise_covariance
        )
        kalman_gain = self.error_covariance @ self.measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.error_covariance = (identity - kalman_gain @ self.measurement_matrix) @ self.error_covariance
        return self.current_state()

    def current_state(self) -> Dict[str, float]:
        """Return current position and velocity."""
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized")
        assert self.state is not None
        return {
            "x_px": float(self.state[0]),
            "y_px": float(self.state[1]),
            "velocity_x_px_per_frame": float(self.state[2]),
            "velocity_y_px_per_frame": float(self.state[3]),
        }

    def next_prediction(self) -> Dict[str, float]:
        """Predict the next state without modifying the filter."""
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized")
        assert self.state is not None
        predicted = self.transition_matrix @ self.state
        return {
            "x_px": float(predicted[0]),
            "y_px": float(predicted[1]),
            "velocity_x_px_per_frame": float(predicted[2]),
            "velocity_y_px_per_frame": float(predicted[3]),
        }


def calculate_control_errors(x_px: float, y_px: float, width: int, height: int) -> Tuple[float, float]:
    """Return normalized pan/tilt errors using the Phase 6 convention."""
    center_x = width / 2.0
    center_y = height / 2.0
    control_x = (x_px - center_x) / center_x
    control_y = -((y_px - center_y) / center_y)
    return clamp(control_x), clamp(control_y)


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clamp a normalized control command."""
    return float(max(lower, min(upper, value)))


def candidate_from_mapping(candidate: Mapping[str, object]) -> CandidateObservation:
    """Normalize one Phase 6 candidate dictionary for tracking."""
    candidate_id = int(candidate.get("candidate_id", 0))
    x_value = candidate.get("x_px", candidate.get("x"))
    y_value = candidate.get("y_px", candidate.get("y"))
    if x_value is None or y_value is None:
        raise ValueError("candidate is missing x/y coordinates")
    correct_probability = float(candidate.get("correct_probability", candidate.get("cnn_probability", 0.0)))
    false_probability = float(candidate.get("false_probability", 1.0 - correct_probability))
    return CandidateObservation(
        candidate_id=candidate_id,
        x_px=float(x_value),
        y_px=float(y_value),
        correct_probability=correct_probability,
        false_probability=false_probability,
        cv_baseline_score=float(candidate.get("cv_baseline_score", 0.0)),
        fused_score=float(candidate.get("fused_score", correct_probability)),
        bbox=dict(candidate["bbox"]) if isinstance(candidate.get("bbox"), Mapping) else None,
    )


def candidates_from_pipeline_result(result: Mapping[str, object]) -> List[CandidateObservation]:
    """Extract all Phase 6 candidates while preserving single-candidate compatibility."""
    candidates = []
    raw_candidates = result.get("candidates", [])
    if isinstance(raw_candidates, Iterable) and not isinstance(raw_candidates, (str, bytes, Mapping)):
        for item in raw_candidates:
            if isinstance(item, Mapping):
                candidates.append(candidate_from_mapping(item))

    if not candidates and result.get("x_px") is not None and result.get("y_px") is not None:
        candidates.append(candidate_from_mapping(result))

    return sorted(candidates, key=lambda item: item.fused_score, reverse=True)


def best_acquisition_candidate(
    candidates: Sequence[CandidateObservation],
    confidence_threshold: float,
) -> Optional[AssociationResult]:
    """Choose the best candidate for initial acquisition."""
    valid = [candidate for candidate in candidates if candidate.correct_probability >= confidence_threshold]
    if not valid:
        return None
    best = max(valid, key=lambda item: item.fused_score)
    return AssociationResult(best, None, 1.0, best.fused_score)


def associate_candidate(
    candidates: Sequence[CandidateObservation],
    predicted_position: Tuple[float, float],
    config: AssociationConfig,
) -> Optional[AssociationResult]:
    """Associate the current detection closest to the predicted track."""
    validate_association_config(config)
    best: Optional[AssociationResult] = None
    for candidate in candidates:
        distance = hypot(candidate.x_px - predicted_position[0], candidate.y_px - predicted_position[1])
        if distance > config.gate_px:
            continue
        position_score = max(0.0, 1.0 - distance / config.gate_px)
        score = config.confidence_weight * candidate.fused_score + config.position_weight * position_score
        result = AssociationResult(candidate, float(distance), float(position_score), float(score))
        if best is None or result.association_score > best.association_score:
            best = result
    return best


class TargetTracker:
    """Stateful Phase 7 tracker over consecutive Phase 6 frame results."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = validate_tracking_config(config or TrackingConfig())
        self.kalman = ConstantVelocityKalmanFilter(self.config.kalman)
        self.temporal = TemporalVerifier(self.config.temporal)
        self.lock_state = LockState.SEARCHING
        self.frame_index = -1
        self.track_age_frames = 0
        self.missed_frames = 0
        self.reacquisition_count = 0
        self.last_confidence = 0.0
        self.last_confirmation_count = 0
        self.temporally_verified = False
        self.trajectory: List[Tuple[float, float]] = []

    def reset(self) -> None:
        """Reset lock state, Kalman state and temporal history."""
        self.kalman.reset()
        self.temporal.reset()
        self.lock_state = LockState.SEARCHING
        self.frame_index = -1
        self.track_age_frames = 0
        self.missed_frames = 0
        self.last_confidence = 0.0
        self.last_confirmation_count = 0
        self.temporally_verified = False
        self.trajectory.clear()

    def update(self, phase6_result: Mapping[str, object], frame_index: Optional[int] = None) -> Dict[str, object]:
        """Update the tracker with one Phase 6 result and return tracking output."""
        start = time.perf_counter()
        self.frame_index = self.frame_index + 1 if frame_index is None else int(frame_index)
        previous_state = self.lock_state
        width = int(phase6_result.get("frame_width", self.config.frame_width))
        height = int(phase6_result.get("frame_height", self.config.frame_height))
        candidates = candidates_from_pipeline_result(phase6_result)

        predicted_for_association = self._predict_for_active_track()
        association: Optional[AssociationResult]
        temporal_result = self._empty_temporal_result()
        measurement_available = False
        using_prediction_only = False
        just_locked = False
        just_lost = False
        just_reacquired = False

        if previous_state in {LockState.SEARCHING, LockState.LOST} or not self.kalman.initialized:
            association = best_acquisition_candidate(candidates, self.config.confidence_threshold)
            if association is None:
                if previous_state == LockState.LOST:
                    self.kalman.reset()
                    self.temporal.reset()
                    self.track_age_frames = 0
                    self.missed_frames = 0
                    self.temporally_verified = False
                self.lock_state = LockState.SEARCHING
            else:
                if previous_state == LockState.LOST:
                    just_reacquired = True
                    self.reacquisition_count += 1
                measurement_available = True
                self.kalman.reset()
                self.kalman.initialize(association.candidate.x_px, association.candidate.y_px)
                self.temporal.reset()
                temporal_result = self.temporal.update(
                    self.frame_index,
                    association.candidate.to_temporal_mapping(),
                )
                self._record_measurement(association, temporal_result)
                self.lock_state = LockState.LOCKED if self.temporally_verified else LockState.ACQUIRING
                just_locked = self.lock_state == LockState.LOCKED
        else:
            assert predicted_for_association is not None
            association = associate_candidate(candidates, predicted_for_association, self.config.association)
            if association is not None:
                measurement_available = True
                self.kalman.correct(association.candidate.x_px, association.candidate.y_px)
                temporal_result = self.temporal.update(
                    self.frame_index,
                    association.candidate.to_temporal_mapping(),
                    expected_position=predicted_for_association,
                )
                self._record_measurement(association, temporal_result)
                if previous_state == LockState.ACQUIRING:
                    self.lock_state = LockState.LOCKED if self.temporally_verified else LockState.ACQUIRING
                    just_locked = self.lock_state == LockState.LOCKED
                else:
                    self.lock_state = LockState.LOCKED
            else:
                temporal_result = self.temporal.update(self.frame_index, None)
                if previous_state == LockState.ACQUIRING:
                    self.kalman.reset()
                    self.temporal.reset()
                    self.lock_state = LockState.SEARCHING
                    self.track_age_frames = 0
                    self.missed_frames = 0
                    self.temporally_verified = False
                else:
                    self.missed_frames += 1
                    self.track_age_frames += 1
                    if self.missed_frames > self.config.max_missed_frames:
                        self.lock_state = LockState.LOST
                        self.temporally_verified = False
                        just_lost = True
                    else:
                        self.lock_state = LockState.COASTING
                        using_prediction_only = True

        output = self._build_output(
            width=width,
            height=height,
            candidates=candidates,
            association=association if measurement_available else None,
            measurement_available=measurement_available,
            using_prediction_only=using_prediction_only,
            just_locked=just_locked,
            just_lost=just_lost,
            just_reacquired=just_reacquired,
            processing_time_ms=round((time.perf_counter() - start) * 1000.0, 3),
        )
        validate_tracking_output(output)
        return output

    def _predict_for_active_track(self) -> Optional[Tuple[float, float]]:
        """Predict current-frame position before association."""
        if self.lock_state in {LockState.SEARCHING, LockState.LOST} or not self.kalman.initialized:
            return None
        predicted = self.kalman.predict()
        return predicted["x_px"], predicted["y_px"]

    def _record_measurement(self, association: AssociationResult, temporal_result: Mapping[str, object]) -> None:
        """Update counters and persistent confidence after a real measurement."""
        self.missed_frames = 0
        self.track_age_frames = self.track_age_frames + 1 if self.kalman.initialized else 1
        self.last_confidence = association.candidate.correct_probability
        self.last_confirmation_count = int(temporal_result["confirmation_count"])
        self.temporally_verified = bool(temporal_result["temporally_verified"]) or self.temporally_verified
        self.trajectory.append((association.candidate.x_px, association.candidate.y_px))

    def _empty_temporal_result(self) -> Dict[str, object]:
        """Return temporal defaults before any verifier update."""
        return {
            "temporally_verified": self.temporally_verified,
            "confirmation_count": self.last_confirmation_count,
            "history_size": len(self.temporal.history),
            "verification_mode": self.temporal.config.mode,
            "blink_match": None,
        }

    def _build_output(
        self,
        width: int,
        height: int,
        candidates: Sequence[CandidateObservation],
        association: Optional[AssociationResult],
        measurement_available: bool,
        using_prediction_only: bool,
        just_locked: bool,
        just_lost: bool,
        just_reacquired: bool,
        processing_time_ms: float,
    ) -> Dict[str, object]:
        """Create the stable Phase 7 tracking dictionary."""
        measured_x = association.candidate.x_px if association is not None else None
        measured_y = association.candidate.y_px if association is not None else None

        can_report_state = self.kalman.initialized and self.lock_state not in {LockState.SEARCHING, LockState.LOST}
        if can_report_state:
            current = self.kalman.current_state()
            next_prediction = self.kalman.next_prediction()
            filtered_x = current["x_px"]
            filtered_y = current["y_px"]
            predicted_x = next_prediction["x_px"]
            predicted_y = next_prediction["y_px"]
            velocity_x = current["velocity_x_px_per_frame"]
            velocity_y = current["velocity_y_px_per_frame"]
            control_x, control_y = calculate_control_errors(filtered_x, filtered_y, width, height)
        else:
            filtered_x = filtered_y = predicted_x = predicted_y = None
            velocity_x = velocity_y = None
            control_x = control_y = None

        target_found = self.lock_state in {LockState.LOCKED, LockState.COASTING}
        return {
            "frame_index": int(self.frame_index),
            "lock_state": self.lock_state.value,
            "target_found": bool(target_found),
            "measurement_available": bool(measurement_available),
            "temporally_verified": bool(self.temporally_verified),
            "measured_x_px": measured_x,
            "measured_y_px": measured_y,
            "filtered_x_px": filtered_x,
            "filtered_y_px": filtered_y,
            "predicted_x_px": predicted_x,
            "predicted_y_px": predicted_y,
            "velocity_x_px_per_frame": velocity_x,
            "velocity_y_px_per_frame": velocity_y,
            "control_error_x": control_x,
            "control_error_y": control_y,
            "confidence": float(self.last_confidence if target_found or measurement_available else 0.0),
            "candidate_count": int(len(candidates)),
            "associated_candidate_id": int(association.candidate.candidate_id) if association is not None else None,
            "distance_to_prediction_px": association.distance_to_prediction_px if association is not None else None,
            "confirmation_count": int(self.last_confirmation_count),
            "track_age_frames": int(self.track_age_frames if self.lock_state != LockState.SEARCHING else 0),
            "missed_frames": int(self.missed_frames),
            "using_prediction_only": bool(using_prediction_only),
            "just_locked": bool(just_locked),
            "just_lost": bool(just_lost),
            "just_reacquired": bool(just_reacquired),
            "processing_time_ms": float(processing_time_ms),
        }


def validate_tracking_output(result: Mapping[str, object]) -> None:
    """Validate stable tracking output keys."""
    missing = sorted(TRACKING_OUTPUT_KEYS - set(result.keys()))
    if missing:
        raise ValueError(f"tracking output missing keys: {missing}")


def tracking_config_from_mapping(raw: Mapping[str, object]) -> TrackingConfig:
    """Build TrackingConfig from project YAML data."""
    project = raw.get("project", {})
    image = raw.get("image", {})
    classifier = raw.get("classifier", {})
    tracker = raw.get("tracker", {})
    temporal = raw.get("temporal", {})
    if not isinstance(project, Mapping):
        project = {}
    if not isinstance(image, Mapping):
        image = {}
    if not isinstance(classifier, Mapping):
        classifier = {}
    if not isinstance(tracker, Mapping):
        tracker = {}
    if not isinstance(temporal, Mapping):
        temporal = {}

    kalman = KalmanFilterConfig(
        dt=float(tracker.get("dt", 1.0)),
        process_noise=float(tracker.get("process_noise", 0.03)),
        measurement_noise=float(tracker.get("measurement_noise", 4.0)),
    )
    association = AssociationConfig(
        gate_px=float(tracker.get("association_gate_px", tracker.get("gate_px", 60.0))),
        confidence_weight=float(tracker.get("association_confidence_weight", 0.40)),
        position_weight=float(tracker.get("association_position_weight", 0.60)),
    )
    config = TrackingConfig(
        frame_width=int(image.get("width", 640)),
        frame_height=int(image.get("height", 480)),
        target_id=str(project.get("target_id", "Terminal_B")),
        confidence_threshold=float(classifier.get("confidence_threshold", 0.55)),
        max_missed_frames=int(tracker.get("max_missed_frames", 8)),
        kalman=kalman,
        association=association,
        temporal=temporal_config_from_mapping(temporal),
    )
    return validate_tracking_config(config)


def load_tracking_config(config_path: str | Path) -> TrackingConfig:
    """Read tracker settings from the project YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required to load tracker configuration")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("configuration file must contain a mapping")
    return tracking_config_from_mapping(raw)


def write_tracking_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write one tracking result row per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(TRACKING_OUTPUT_KEYS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in sorted(TRACKING_OUTPUT_KEYS)})


def summarize_tracking(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Create a small tracking summary without claiming ground-truth accuracy."""
    measurement_distances = []
    association_distances = []
    processing_times = []
    max_consecutive_missed = 0
    current_missed = 0

    for row in rows:
        if row.get("measurement_available") and row.get("filtered_x_px") is not None:
            dx = float(row["measured_x_px"]) - float(row["filtered_x_px"])
            dy = float(row["measured_y_px"]) - float(row["filtered_y_px"])
            measurement_distances.append(hypot(dx, dy))
        if row.get("distance_to_prediction_px") is not None:
            association_distances.append(float(row["distance_to_prediction_px"]))
        processing_times.append(float(row.get("processing_time_ms", 0.0)))
        if int(row.get("missed_frames", 0)) > 0:
            current_missed += 1
        else:
            current_missed = 0
        max_consecutive_missed = max(max_consecutive_missed, current_missed)

    total = len(rows)
    return {
        "total_frames": total,
        "frames_with_measurements": sum(1 for row in rows if row.get("measurement_available")),
        "locked_frames": sum(1 for row in rows if row.get("lock_state") == LockState.LOCKED.value),
        "coasting_frames": sum(1 for row in rows if row.get("lock_state") == LockState.COASTING.value),
        "lost_frames": sum(1 for row in rows if row.get("lock_state") == LockState.LOST.value),
        "reacquisition_count": sum(1 for row in rows if row.get("just_reacquired")),
        "mean_measurement_to_filter_distance": mean_or_none(measurement_distances),
        "mean_association_distance": mean_or_none(association_distances),
        "average_processing_time_ms": mean_or_none(processing_times),
        "maximum_consecutive_missed_frames": max_consecutive_missed,
        "accuracy_note": "No real tracking accuracy is claimed without ground-truth trajectory data.",
    }


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    """Return a rounded mean or None for empty values."""
    if not values:
        return None
    return float(round(sum(values) / len(values), 6))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def image_paths_from_dir(image_dir: Path) -> List[Path]:
    """Return ordered image files for an image-sequence run."""
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in extensions)
    if not paths:
        raise ValueError(f"no image frames found in {image_dir}")
    return paths


def frame_source_from_args(args: argparse.Namespace) -> Tuple[Iterable[np.ndarray], float, Tuple[int, int]]:
    """Open a video or ordered image folder as a frame iterable."""
    if args.image_dir:
        paths = image_paths_from_dir(Path(args.image_dir))
        first = load_frame(paths[0])
        height, width = first.shape[:2]

        def frames() -> Iterable[np.ndarray]:
            for path in paths:
                yield load_frame(path)

        return frames(), float(args.fps), (width, height)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or args.fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames() -> Iterable[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    return frames(), fps, (width, height)


def draw_tracking_overlay(
    frame: np.ndarray,
    phase6_result: Mapping[str, object],
    tracking_result: Mapping[str, object],
    trajectory: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Draw candidates, measured point, filtered point, prediction and lock state."""
    overlay = frame.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for candidate in phase6_result.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        point = (int(round(float(candidate["x_px"]))), int(round(float(candidate["y_px"]))))
        cv2.circle(overlay, point, 5, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(overlay, f"C{candidate['candidate_id']}", (point[0] + 6, max(14, point[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    draw_point(overlay, tracking_result.get("measured_x_px"), tracking_result.get("measured_y_px"), (0, 255, 0), "M")
    draw_point(overlay, tracking_result.get("filtered_x_px"), tracking_result.get("filtered_y_px"), (255, 0, 0), "F")
    draw_point(overlay, tracking_result.get("predicted_x_px"), tracking_result.get("predicted_y_px"), (255, 0, 255), "P")

    if len(trajectory) >= 2:
        points = np.array([(int(round(x)), int(round(y))) for x, y in trajectory[-80:]], dtype=np.int32)
        cv2.polylines(overlay, [points], False, (255, 255, 0), 1, lineType=cv2.LINE_AA)

    lines = [
        f"State: {tracking_result['lock_state']}",
        f"Confidence: {float(tracking_result['confidence']):.2f}",
        f"Missed: {tracking_result['missed_frames']}",
        f"Prediction only: {tracking_result['using_prediction_only']}",
    ]
    for index, line in enumerate(lines):
        y = 26 + index * 24
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def draw_point(
    image: np.ndarray,
    x_value: object,
    y_value: object,
    color: Tuple[int, int, int],
    label: str,
) -> None:
    """Draw one optional tracking point."""
    if x_value is None or y_value is None:
        return
    point = (int(round(float(x_value))), int(round(float(y_value))))
    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(image, label, (point[0] + 8, max(14, point[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def run_sequence(
    frames: Iterable[np.ndarray],
    output_dir: Path,
    fps: float,
    frame_size: Tuple[int, int],
    pipeline: SingleFramePipeline,
    tracker: TargetTracker,
) -> Dict[str, object]:
    """Run Phase 6 plus Phase 7 on an ordered sequence and write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_dir / "annotated_tracking.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise OSError(f"could not create annotated video in {output_dir}")

    rows = []
    try:
        for frame_index, frame in enumerate(frames):
            phase6_result = pipeline.run(frame)
            tracking_result = tracker.update(phase6_result, frame_index=frame_index)
            rows.append(tracking_result)
            overlay = draw_tracking_overlay(frame, phase6_result, tracking_result, tracker.trajectory)
            writer.write(overlay)
    finally:
        writer.release()

    write_tracking_csv(output_dir / "tracking_results.csv", rows)
    summary = summarize_tracking(rows)
    write_json(output_dir / "tracking_summary.json", summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for sequence tracking."""
    parser = argparse.ArgumentParser(description="Run Phase 7 temporal verification and Kalman tracking.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Input video path.")
    source.add_argument("--image-dir", help="Ordered frame folder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/tracking-test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=15.0, help="FPS used for image-dir output videos.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point for sequence tracking."""
    args = parse_args(argv)
    pipeline_config = load_config(args.config, checkpoint_override=args.checkpoint, device_override=args.device)
    tracking_config = load_tracking_config(args.config)
    pipeline = SingleFramePipeline(pipeline_config)
    tracker = TargetTracker(tracking_config)
    frames, fps, frame_size = frame_source_from_args(args)
    summary = run_sequence(frames, Path(args.output), fps, frame_size, pipeline, tracker)

    print("Phase 7 sequence tracking")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Measurements: {summary['frames_with_measurements']}")
    print(f"  Locked frames: {summary['locked_frames']}")
    print(f"  Coasting frames: {summary['coasting_frames']}")
    print(f"  Lost frames: {summary['lost_frames']}")
    print(f"  Outputs: {args.output}")


if __name__ == "__main__":
    main()
