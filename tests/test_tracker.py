from __future__ import annotations

import math

import numpy as np
import pytest

from src.tracker import (
    TRACKING_OUTPUT_KEYS,
    AssociationConfig,
    CandidateObservation,
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
    LockState,
    TargetTracker,
    TrackingConfig,
    associate_candidate,
    candidates_from_pipeline_result,
    summarize_tracking,
)
from src.temporal_verifier import TemporalConfig


def candidate(
    candidate_id: int,
    x_px: float,
    y_px: float,
    correct_probability: float = 0.9,
    fused_score: float | None = None,
) -> dict[str, object]:
    score = correct_probability if fused_score is None else fused_score
    return {
        "candidate_id": candidate_id,
        "x_px": x_px,
        "y_px": y_px,
        "bbox": {"x": int(x_px) - 3, "y": int(y_px) - 3, "width": 6, "height": 6},
        "correct_probability": correct_probability,
        "false_probability": 1.0 - correct_probability,
        "cv_baseline_score": score,
        "fused_score": score,
    }


def phase6(candidates: list[dict[str, object]], width: int = 200, height: int = 120) -> dict[str, object]:
    selected = candidates[0] if candidates else None
    return {
        "target_found": selected is not None,
        "frame_width": width,
        "frame_height": height,
        "candidate_count": len(candidates),
        "x_px": selected["x_px"] if selected else None,
        "y_px": selected["y_px"] if selected else None,
        "confidence": selected["correct_probability"] if selected else 0.0,
        "candidates": candidates,
    }


def make_tracker(max_missed_frames: int = 3, min_confirmations: int = 3) -> TargetTracker:
    return TargetTracker(
        TrackingConfig(
            frame_width=200,
            frame_height=120,
            confidence_threshold=0.55,
            max_missed_frames=max_missed_frames,
            kalman=KalmanFilterConfig(dt=1.0, process_noise=0.03, measurement_noise=4.0),
            association=AssociationConfig(gate_px=35.0, confidence_weight=0.40, position_weight=0.60),
            temporal=TemporalConfig(history_window=5, min_confirmations=min_confirmations, association_radius_px=25.0),
        )
    )


def test_kalman_initialization() -> None:
    kalman = ConstantVelocityKalmanFilter()
    kalman.initialize(100.0, 50.0)
    state = kalman.current_state()

    assert state["x_px"] == 100.0
    assert state["y_px"] == 50.0
    assert state["velocity_x_px_per_frame"] == 0.0
    assert state["velocity_y_px_per_frame"] == 0.0


def test_kalman_prediction_and_correction() -> None:
    kalman = ConstantVelocityKalmanFilter(KalmanFilterConfig(dt=1.0, process_noise=0.01, measurement_noise=2.0))
    kalman.initialize(0.0, 0.0)
    kalman.predict()
    corrected = kalman.correct(10.0, 0.0)

    assert corrected["x_px"] > 0.0
    assert corrected["y_px"] == pytest.approx(0.0)
    assert corrected["velocity_x_px_per_frame"] > 0.0


def test_noisy_straight_line_movement_becomes_smoother() -> None:
    rng = np.random.default_rng(42)
    kalman = ConstantVelocityKalmanFilter(KalmanFilterConfig(process_noise=0.01, measurement_noise=9.0))
    true_points = [(50 + index * 4, 60.0) for index in range(30)]
    measurements = [(x + rng.normal(0, 3), y + rng.normal(0, 3)) for x, y in true_points]
    filtered = []
    kalman.initialize(*measurements[0])
    for x, y in measurements[1:]:
        kalman.predict()
        state = kalman.correct(x, y)
        filtered.append((state["x_px"], state["y_px"]))

    measurement_error = np.mean([math.hypot(mx - tx, my - ty) for (mx, my), (tx, ty) in zip(measurements[1:], true_points[1:])])
    filtered_error = np.mean([math.hypot(fx - tx, fy - ty) for (fx, fy), (tx, ty) in zip(filtered, true_points[1:])])
    assert filtered_error < measurement_error


def test_correct_candidate_association_near_prediction() -> None:
    candidates = [
        CandidateObservation(1, 102.0, 51.0, 0.65, 0.35, 0.5, 0.65),
        CandidateObservation(2, 160.0, 90.0, 0.99, 0.01, 0.9, 0.99),
    ]
    result = associate_candidate(candidates, (100.0, 50.0), AssociationConfig(gate_px=35.0))

    assert result is not None
    assert result.candidate.candidate_id == 1
    assert result.distance_to_prediction_px == pytest.approx(math.hypot(2.0, 1.0))


def test_distant_high_confidence_decoy_does_not_steal_track() -> None:
    tracker = make_tracker()
    tracker.update(phase6([candidate(1, 50, 50, 0.9)]), frame_index=0)
    tracker.update(phase6([candidate(1, 55, 50, 0.9)]), frame_index=1)
    tracker.update(phase6([candidate(1, 60, 50, 0.9)]), frame_index=2)
    result = tracker.update(
        phase6(
            [
                candidate(99, 170, 90, 0.99, 0.99),
                candidate(1, 65, 50, 0.60, 0.60),
            ]
        ),
        frame_index=3,
    )

    assert result["lock_state"] == LockState.LOCKED.value
    assert result["associated_candidate_id"] == 1


def test_transition_searching_to_acquiring() -> None:
    tracker = make_tracker()
    result = tracker.update(phase6([candidate(1, 50, 50, 0.9)]), frame_index=0)

    assert result["lock_state"] == LockState.ACQUIRING.value
    assert result["target_found"] is False
    assert result["measurement_available"] is True


def test_transition_acquiring_to_locked() -> None:
    tracker = make_tracker()
    tracker.update(phase6([candidate(1, 50, 50)]), frame_index=0)
    tracker.update(phase6([candidate(1, 55, 50)]), frame_index=1)
    result = tracker.update(phase6([candidate(1, 60, 50)]), frame_index=2)

    assert result["lock_state"] == LockState.LOCKED.value
    assert result["target_found"] is True
    assert result["just_locked"] is True
    assert result["confirmation_count"] == 3


def test_temporary_loss_produces_coasting() -> None:
    tracker = make_tracker()
    for frame_index, x_px in enumerate([50, 55, 60]):
        tracker.update(phase6([candidate(1, x_px, 50)]), frame_index=frame_index)

    result = tracker.update(phase6([]), frame_index=3)

    assert result["lock_state"] == LockState.COASTING.value
    assert result["measurement_available"] is False
    assert result["using_prediction_only"] is True
    assert result["target_found"] is True


def test_reappearance_near_prediction_restores_locked() -> None:
    tracker = make_tracker()
    for frame_index, x_px in enumerate([50, 55, 60]):
        tracker.update(phase6([candidate(1, x_px, 50)]), frame_index=frame_index)
    tracker.update(phase6([]), frame_index=3)
    result = tracker.update(phase6([candidate(1, 70, 50, 0.75)]), frame_index=4)

    assert result["lock_state"] == LockState.LOCKED.value
    assert result["measurement_available"] is True
    assert result["associated_candidate_id"] == 1


def test_exceeding_missed_frame_limit_produces_lost() -> None:
    tracker = make_tracker(max_missed_frames=1)
    for frame_index, x_px in enumerate([50, 55, 60]):
        tracker.update(phase6([candidate(1, x_px, 50)]), frame_index=frame_index)

    tracker.update(phase6([]), frame_index=3)
    result = tracker.update(phase6([]), frame_index=4)

    assert result["lock_state"] == LockState.LOST.value
    assert result["target_found"] is False
    assert result["just_lost"] is True


def test_reacquisition_after_lost() -> None:
    tracker = make_tracker(max_missed_frames=1)
    for frame_index, x_px in enumerate([50, 55, 60]):
        tracker.update(phase6([candidate(1, x_px, 50)]), frame_index=frame_index)
    tracker.update(phase6([]), frame_index=3)
    tracker.update(phase6([]), frame_index=4)
    result = tracker.update(phase6([candidate(2, 120, 70, 0.95)]), frame_index=5)

    assert result["lock_state"] == LockState.ACQUIRING.value
    assert result["associated_candidate_id"] == 2
    assert result["just_reacquired"] is True


def test_tracker_reset() -> None:
    tracker = make_tracker()
    tracker.update(phase6([candidate(1, 50, 50)]), frame_index=0)

    tracker.reset()

    assert tracker.lock_state == LockState.SEARCHING
    assert tracker.kalman.initialized is False
    assert tracker.track_age_frames == 0


def test_stable_output_schema() -> None:
    tracker = make_tracker()
    result = tracker.update(phase6([candidate(1, 50, 50)]), frame_index=0)

    assert set(result) == TRACKING_OUTPUT_KEYS


def test_candidates_from_phase6_result_preserves_all_candidates() -> None:
    items = candidates_from_pipeline_result(phase6([candidate(1, 10, 10, 0.6), candidate(2, 20, 20, 0.9)]))

    assert [item.candidate_id for item in items] == [2, 1]


def test_tracking_summary_fields() -> None:
    tracker = make_tracker()
    rows = [tracker.update(phase6([candidate(1, 50 + 5 * i, 50)]), frame_index=i) for i in range(3)]
    rows.append(tracker.update(phase6([]), frame_index=3))
    summary = summarize_tracking(rows)

    assert summary["total_frames"] == 4
    assert summary["frames_with_measurements"] == 3
    assert summary["coasting_frames"] == 1
    assert "accuracy_note" in summary


def test_invalid_tracker_configuration() -> None:
    with pytest.raises(ValueError, match="association weights"):
        TargetTracker(TrackingConfig(association=AssociationConfig(confidence_weight=0.7, position_weight=0.7)))

    with pytest.raises(ValueError, match="max_missed_frames"):
        TargetTracker(TrackingConfig(max_missed_frames=-1))
