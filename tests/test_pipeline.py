from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from src.candidate_detector import DetectorConfig
from src.pipeline import (
    PIPELINE_OUTPUT_KEYS,
    InferenceConfig,
    PipelineConfig,
    SingleFramePipeline,
    calculate_coordinate_errors,
    calculate_fused_score,
    load_frame,
)
from src.preprocessing import PreprocessingConfig


class SequenceClassifier(nn.Module):
    """Small mock classifier that returns configured correct probabilities in order."""

    def __init__(self, correct_probabilities: list[float]) -> None:
        super().__init__()
        self.correct_probabilities = correct_probabilities

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rows = []
        for index in range(images.shape[0]):
            probability = self.correct_probabilities[min(index, len(self.correct_probabilities) - 1)]
            probability = min(max(probability, 1e-4), 1.0 - 1e-4)
            rows.append([1.0 - probability, probability])
        return torch.log(torch.tensor(rows, dtype=torch.float32, device=images.device))


def make_pipeline(
    probabilities: list[float],
    confidence_threshold: float = 0.55,
    cnn_weight: float = 0.80,
    cv_weight: float = 0.20,
) -> SingleFramePipeline:
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(threshold_method="fixed", threshold_value=180, blur_kernel=3, morph_kernel=3),
        detector=DetectorConfig(min_area=3.0, max_area=1000.0, min_radius=1.0, max_radius=25.0, min_circularity=0.10),
        inference=InferenceConfig(
            confidence_threshold=confidence_threshold,
            cnn_weight=cnn_weight,
            cv_weight=cv_weight,
            crop_size=40,
            patch_size=32,
            device="cpu",
        ),
    )
    return SingleFramePipeline(config=config, model=SequenceClassifier(probabilities))


def make_frame(points: list[tuple[int, int, int, int]], width: int = 120, height: int = 100) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y, radius, intensity in points:
        cv2.circle(frame, (x, y), radius, (intensity, intensity, intensity), -1, lineType=cv2.LINE_AA)
    return frame


def test_single_detectable_target_is_found() -> None:
    frame = make_frame([(70, 40, 5, 255)])
    result = make_pipeline([0.92]).run(frame)

    assert result["target_found"] is True
    assert result["status"] == "target_detected"
    assert result["candidate_count"] == 1
    assert result["x_px"] == pytest.approx(70, abs=1.0)
    assert result["y_px"] == pytest.approx(40, abs=1.0)


def test_multiple_candidates_are_all_kept() -> None:
    frame = make_frame([(25, 25, 5, 230), (85, 70, 5, 255)])
    result = make_pipeline([0.75, 0.90]).run(frame)

    assert result["target_found"] is True
    assert result["candidate_count"] >= 2
    assert len(result["candidates"]) == result["candidate_count"]


def test_target_absent_frame_with_false_candidate_is_rejected() -> None:
    frame = make_frame([(35, 35, 5, 255)])
    result = make_pipeline([0.20]).run(frame)

    assert result["target_found"] is False
    assert result["status"] == "below_confidence_threshold"
    assert result["candidate_count"] == 1


def test_completely_black_frame_returns_no_candidates() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    result = make_pipeline([0.90]).run(frame)

    assert result["target_found"] is False
    assert result["status"] == "no_candidates"
    assert result["candidate_count"] == 0


def test_candidate_below_confidence_threshold_is_not_found() -> None:
    frame = make_frame([(70, 40, 5, 255)])
    result = make_pipeline([0.70], confidence_threshold=0.80).run(frame)

    assert result["target_found"] is False
    assert result["status"] == "below_confidence_threshold"
    assert result["cnn_probability"] == pytest.approx(0.70, abs=1e-4)


def test_coordinate_error_calculation_matches_pid_convention() -> None:
    errors = calculate_coordinate_errors(target_x=486.0, target_y=192.0, width=640, height=480)

    assert errors["frame_center_x"] == 320.0
    assert errors["frame_center_y"] == 240.0
    assert errors["image_error_x_px"] == 166.0
    assert errors["image_error_y_px"] == -48.0
    assert errors["control_error_x"] == pytest.approx(0.51875)
    assert errors["control_error_y"] == pytest.approx(0.2)


def test_fusion_score_uses_configured_weights() -> None:
    config = InferenceConfig(cnn_weight=0.80, cv_weight=0.20)

    assert calculate_fused_score(0.91, 0.73, config) == pytest.approx(0.874)


def test_stable_output_keys_for_found_and_not_found_cases() -> None:
    found = make_pipeline([0.95]).run(make_frame([(70, 40, 5, 255)]))
    not_found = make_pipeline([0.95]).run(np.zeros((100, 120, 3), dtype=np.uint8))

    assert set(found) == PIPELINE_OUTPUT_KEYS
    assert set(not_found) == PIPELINE_OUTPUT_KEYS


def test_invalid_or_unreadable_input_handling(tmp_path: Path) -> None:
    pipeline = make_pipeline([0.95])
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        pipeline.run(missing_path)
    with pytest.raises(ValueError, match="missing or empty"):
        load_frame(np.array([], dtype=np.uint8))


def test_grayscale_and_bgra_inputs_are_converted_to_bgr() -> None:
    gray = np.zeros((20, 30), dtype=np.uint8)
    bgra = np.zeros((20, 30, 4), dtype=np.uint8)

    assert load_frame(gray).shape == (20, 30, 3)
    assert load_frame(bgra).shape == (20, 30, 3)
