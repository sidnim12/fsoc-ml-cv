from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluate_unity_sequence import EVALUATION_RESULT_KEYS, calculate_detection_metrics, evaluate_unity_sequence, evaluate_unity_sequences
from src.tracker import TRACKING_OUTPUT_KEYS


def write_image(path: Path, width: int = 64, height: int = 48, value: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.circle(image, (width // 2, height // 2), 2, (255, 255, 255), -1)
    assert cv2.imwrite(str(path), image)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sequence(tmp_path: Path, with_labels: bool = True) -> Path:
    sequence = tmp_path / "unity" / "smooth_horizontal01" / "sequence_001"
    for frame_index in range(3):
        write_image(sequence / f"frame_{frame_index:06d}.png", value=20 + frame_index)
    if with_labels:
        (sequence / "labels.csv").write_text(
            "sequence_id,frame_id,timestamp_s,image_path,scenario,target_present,target_id,target_x,target_y\n"
            "1,0,0.0000,frame_000000.png,smooth_horizontal,1,beacon_001,10,20\n"
            "1,1,0.0333,frame_000001.png,smooth_horizontal,1,beacon_001,20,20\n"
            "1,2,0.0667,frame_000002.png,smooth_horizontal,0,beacon_001,,\n",
            encoding="utf-8",
        )
    return sequence


class MockInferenceConfig:
    confidence_threshold = 0.55


class MockPipeline:
    def __init__(self) -> None:
        self.calls = 0
        self.inference_config = MockInferenceConfig()

    def run(self, frame: np.ndarray) -> dict[str, object]:
        del frame
        outputs = [
            self._found(0, 10.0, 20.0, 0.90),
            self._found(1, 25.0, 20.0, 0.70),
            self._missing(2),
        ]
        result = outputs[min(self.calls, len(outputs) - 1)]
        self.calls += 1
        return result

    @staticmethod
    def _found(candidate_id: int, x_px: float, y_px: float, probability: float) -> dict[str, object]:
        return {
            "target_found": True,
            "status": "target_detected",
            "frame_width": 64,
            "frame_height": 48,
            "selected_candidate_id": candidate_id,
            "x_px": x_px,
            "y_px": y_px,
            "confidence": probability,
            "cnn_probability": probability,
            "cv_baseline_score": 0.8,
            "fused_score": probability,
            "candidate_count": 1,
            "inference_time_ms": 1.0,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "x_px": x_px,
                    "y_px": y_px,
                    "correct_probability": probability,
                    "false_probability": 1.0 - probability,
                    "cv_baseline_score": 0.8,
                    "fused_score": probability,
                }
            ],
        }

    @staticmethod
    def _missing(candidate_id: int) -> dict[str, object]:
        return {
            "target_found": False,
            "status": "no_candidates",
            "frame_width": 64,
            "frame_height": 48,
            "selected_candidate_id": None,
            "x_px": None,
            "y_px": None,
            "confidence": 0.0,
            "cnn_probability": 0.0,
            "cv_baseline_score": 0.0,
            "fused_score": 0.0,
            "candidate_count": 0,
            "inference_time_ms": 1.0,
            "candidates": [],
        }


class MockTracker:
    def __init__(self) -> None:
        self.reset_count = 0
        self.calls = 0
        self.trajectory: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def update(self, phase6_result: dict[str, object], frame_index: int | None = None) -> dict[str, object]:
        frame = int(frame_index or 0)
        measured_x = phase6_result.get("x_px")
        measured_y = phase6_result.get("y_px")
        state = "ACQUIRING" if frame == 0 else "LOCKED" if measured_x is not None else "COASTING"
        if measured_x is not None and measured_y is not None:
            self.trajectory.append((float(measured_x), float(measured_y)))
        self.calls += 1
        result = {key: None for key in TRACKING_OUTPUT_KEYS}
        result.update(
            {
                "frame_index": frame,
                "lock_state": state,
                "target_found": state in {"LOCKED", "COASTING"},
                "measurement_available": measured_x is not None,
                "temporally_verified": frame >= 1,
                "measured_x_px": measured_x,
                "measured_y_px": measured_y,
                "filtered_x_px": measured_x,
                "filtered_y_px": measured_y,
                "predicted_x_px": measured_x,
                "predicted_y_px": measured_y,
                "velocity_x_px_per_frame": 0.0,
                "velocity_y_px_per_frame": 0.0,
                "control_error_x": 0.0,
                "control_error_y": 0.0,
                "confidence": phase6_result.get("confidence", 0.0),
                "candidate_count": phase6_result.get("candidate_count", 0),
                "associated_candidate_id": phase6_result.get("selected_candidate_id"),
                "distance_to_prediction_px": 0.0 if measured_x is not None else None,
                "confirmation_count": frame + 1,
                "track_age_frames": frame + 1,
                "missed_frames": 1 if measured_x is None else 0,
                "using_prediction_only": measured_x is None,
                "just_locked": frame == 1,
                "just_lost": False,
                "just_reacquired": False,
                "processing_time_ms": 0.2,
            }
        )
        return result


def test_evaluation_stable_schema_and_outputs(tmp_path: Path) -> None:
    make_sequence(tmp_path)
    pipeline = MockPipeline()
    tracker = MockTracker()
    output = tmp_path / "outputs"

    result = evaluate_unity_sequence(
        sequence_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_dir=output,
        expected_count=3,
        expected_width=64,
        expected_height=48,
        pipeline=pipeline,
        tracker=tracker,
    )

    assert set(result) == EVALUATION_RESULT_KEYS
    assert (output / "dataset_validation.json").exists()
    assert (output / "normalized_manifest.csv").exists()
    assert (output / "frame_predictions.csv").exists()
    assert (output / "annotated_tracking.mp4").exists()
    assert (output / "confidence_over_time.png").exists()
    assert (output / "lock_state_timeline.png").exists()
    assert (output / "trajectory_comparison.png").exists()
    assert (output / "coordinate_error_over_time.png").exists()
    assert (output / "failure_montage.png").exists()


def test_evaluation_autodetects_resolution_and_scales_video(tmp_path: Path) -> None:
    make_sequence(tmp_path)
    output = tmp_path / "outputs"

    result = evaluate_unity_sequence(
        sequence_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_dir=output,
        output_scale=2,
        pipeline=MockPipeline(),
        tracker=MockTracker(),
    )

    assert result["dataset_validation"]["resolution_autodetected"] is True
    assert result["evaluation_summary"]["resolution"] == "64x48"
    assert result["evaluation_summary"]["output_video_resolution"] == "128x96"

    video = cv2.VideoCapture(str(output / "annotated_tracking.mp4"))
    assert video.isOpened()
    assert int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == 128
    assert int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 96


def test_ground_truth_metrics_with_known_values(tmp_path: Path) -> None:
    make_sequence(tmp_path)
    result = evaluate_unity_sequence(
        sequence_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_dir=tmp_path / "outputs",
        expected_count=3,
        expected_width=64,
        expected_height=48,
        pipeline=MockPipeline(),
        tracker=MockTracker(),
        match_tolerance_px=12,
    )

    detection = result["detection_metrics"]
    tracking = result["tracking_metrics"]
    coordinate = tracking["coordinate_metrics"]

    assert detection["visible_target_frames"] == 2
    assert detection["target_absent_frames"] == 1
    assert detection["candidate_recall"] == 1.0
    assert detection["accepted_detection_recall"] == 1.0
    assert detection["missed_detections"] == 0
    assert detection["target_found_rate"] == pytest.approx(2 / 3)
    assert coordinate["measurement_mae_px"] == 2.5
    assert coordinate["filtered_rmse_px"] == pytest.approx(3.535534)
    assert tracking["time_to_first_lock_frames"] == 1
    assert tracking["locked_frame_percentage"] == pytest.approx(33.333333)


def test_null_metrics_and_label_template_when_labels_absent(tmp_path: Path) -> None:
    make_sequence(tmp_path, with_labels=False)
    output = tmp_path / "outputs"

    result = evaluate_unity_sequence(
        sequence_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_dir=output,
        expected_count=3,
        expected_width=64,
        expected_height=48,
        pipeline=MockPipeline(),
        tracker=MockTracker(),
    )

    assert result["ground_truth_available"] is False
    assert result["detection_metrics"]["available"] is False
    assert result["detection_metrics"]["candidate_recall"] is None
    assert result["tracking_metrics"]["coordinate_metrics"]["filtered_mae_px"] is None
    assert (output / "required_labels_template.csv").exists()


def test_model_initialized_once_and_tracker_reset_only_at_start(tmp_path: Path) -> None:
    make_sequence(tmp_path)
    created_pipelines: list[MockPipeline] = []
    created_trackers: list[MockTracker] = []

    def pipeline_factory() -> MockPipeline:
        created_pipelines.append(MockPipeline())
        return created_pipelines[-1]

    def tracker_factory() -> MockTracker:
        created_trackers.append(MockTracker())
        return created_trackers[-1]

    evaluate_unity_sequence(
        sequence_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_dir=tmp_path / "outputs",
        expected_count=3,
        expected_width=64,
        expected_height=48,
        pipeline_factory=pipeline_factory,
        tracker_factory=tracker_factory,
    )

    assert len(created_pipelines) == 1
    assert len(created_trackers) == 1
    assert created_pipelines[0].calls == 3
    assert created_trackers[0].reset_count == 1
    assert created_trackers[0].calls == 3


def test_original_images_remain_unchanged(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path)
    first = sequence / "frame_000000.png"
    before = file_hash(first)

    evaluate_unity_sequence(
        sequence_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_dir=tmp_path / "outputs",
        expected_count=3,
        expected_width=64,
        expected_height=48,
        pipeline=MockPipeline(),
        tracker=MockTracker(),
    )

    assert file_hash(first) == before


def test_missing_image_detection_in_labels(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path)
    (sequence / "labels.csv").write_text(
        "frame_id,filename,target_present,cx_px,cy_px\n0,missing.png,1,10,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing images"):
        evaluate_unity_sequence(
            sequence_dir=tmp_path / "unity",
            fps=30,
            coordinate_origin="top-left",
            output_dir=tmp_path / "outputs",
            expected_count=3,
            expected_width=64,
            expected_height=48,
            pipeline=MockPipeline(),
            tracker=MockTracker(),
        )


def test_out_of_bounds_labels_fail_validation(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path)
    (sequence / "labels.csv").write_text(
        "frame_id,filename,target_present,cx_px,cy_px\n"
        "0,frame_000000.png,1,999,20\n"
        "1,frame_000001.png,1,20,20\n"
        "2,frame_000002.png,0,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="labels failed validation"):
        evaluate_unity_sequence(
            sequence_dir=tmp_path / "unity",
            fps=30,
            coordinate_origin="top-left",
            output_dir=tmp_path / "outputs",
            pipeline=MockPipeline(),
            tracker=MockTracker(),
        )


def test_calculate_detection_metrics_does_not_use_predictions_as_ground_truth() -> None:
    rows = [
        {
            "frame_index": 0,
            "target_visible": True,
            "target_x": None,
            "target_y": None,
            "phase6_target_found": True,
            "phase6_x_px": 10,
            "phase6_y_px": 20,
            "_candidate_list": [{"x_px": 10, "y_px": 20}],
        }
    ]

    metrics = calculate_detection_metrics(rows, ground_truth_available=False, match_tolerance_px=12)

    assert metrics["available"] is False
    assert metrics["candidate_recall"] is None


def test_batch_evaluation_writes_combined_metrics(tmp_path: Path) -> None:
    make_sequence(tmp_path)
    second = tmp_path / "unity" / "disturbance01" / "sequence_001"
    for frame_index in range(3):
        write_image(second / f"frame_{frame_index:06d}.png", value=30 + frame_index)
    (second / "labels.csv").write_text(
        "sequence_id,frame_id,timestamp_s,image_path,scenario,target_present,target_id,target_x,target_y\n"
        "1,0,0.0000,frame_000000.png,disturbance,1,beacon_001,10,20\n"
        "1,1,0.0333,frame_000001.png,disturbance,1,beacon_001,20,20\n"
        "1,2,0.0667,frame_000002.png,disturbance,0,beacon_001,,\n",
        encoding="utf-8",
    )
    output = tmp_path / "batch_outputs"

    result = evaluate_unity_sequences(
        root_dir=tmp_path / "unity",
        fps=30,
        coordinate_origin="top-left",
        output_root=output,
        pipeline=MockPipeline(),
        tracker_factory=MockTracker,
    )

    assert result["summary"]["sequence_count"] == 2
    assert result["summary"]["successful_sequences"] == 2
    assert result["summary"]["failed_sequences"] == 0
    assert (output / "final_metrics.csv").exists()
    assert (output / "final_summary.json").exists()
