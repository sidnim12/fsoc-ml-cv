from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from src.api import PREDICT_RESPONSE_KEYS, create_app
from src.candidate_detector import DetectorConfig
from src.pipeline import InferenceConfig, PipelineConfig, SingleFramePipeline
from src.preprocessing import PreprocessingConfig
from src.temporal_verifier import TemporalConfig
from src.tracker import AssociationConfig, KalmanFilterConfig, TrackingConfig


class SequenceClassifier(nn.Module):
    """Mock classifier that returns configured correct probabilities in order."""

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


def make_pipeline(probabilities: list[float] | None = None) -> SingleFramePipeline:
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(threshold_method="fixed", threshold_value=180, blur_kernel=3, morph_kernel=3),
        detector=DetectorConfig(min_area=3.0, max_area=1000.0, min_radius=1.0, max_radius=25.0, min_circularity=0.10),
        inference=InferenceConfig(
            confidence_threshold=0.55,
            cnn_weight=0.80,
            cv_weight=0.20,
            crop_size=40,
            patch_size=32,
            device="cpu",
        ),
    )
    return SingleFramePipeline(config=config, model=SequenceClassifier(probabilities or [0.92]))


def make_tracking_config() -> TrackingConfig:
    return TrackingConfig(
        frame_width=120,
        frame_height=100,
        confidence_threshold=0.55,
        max_missed_frames=3,
        kalman=KalmanFilterConfig(dt=1.0, process_noise=0.03, measurement_noise=4.0),
        association=AssociationConfig(gate_px=35.0, confidence_weight=0.40, position_weight=0.60),
        temporal=TemporalConfig(history_window=5, min_confirmations=3, association_radius_px=25.0),
    )


def make_frame(points: list[tuple[int, int, int, int]], width: int = 120, height: int = 100) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y, radius, intensity in points:
        cv2.circle(frame, (x, y), radius, (intensity, intensity, intensity), -1, lineType=cv2.LINE_AA)
    return frame


def encode_png(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    return buffer.tobytes()


def make_client() -> TestClient:
    app = create_app(
        pipeline=make_pipeline(),
        tracking_config=make_tracking_config(),
        config_path="configs/unity.yaml",
    )
    return TestClient(app)


def post_frame(client: TestClient, frame: np.ndarray, extra: dict[str, str] | None = None) -> dict:
    response = client.post(
        "/predict-frame",
        files={"file": ("frame.png", encode_png(frame), "image/png")},
        data=extra or {},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_loaded_runtime() -> None:
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["tracker_ready"] is True
    assert payload["device"] == "cpu"
    assert payload["config"] == "configs/unity.yaml"


def test_reset_clears_tracker_state() -> None:
    frame = make_frame([(70, 40, 5, 255)])
    with make_client() as client:
        first = post_frame(client, frame, {"frame_index": "0"})
        reset = client.post("/reset")
        second = post_frame(client, frame, {"frame_index": "0"})

    assert first["lock_state"] == "ACQUIRING"
    assert reset.status_code == 200
    assert reset.json() == {"status": "reset", "message": "Tracker state cleared"}
    assert second["lock_state"] == "ACQUIRING"
    assert second["frame_index"] == 0
    assert second["missed_frames"] == 0


def test_predict_frame_returns_unity_contract() -> None:
    frame = make_frame([(70, 40, 5, 255)])
    with make_client() as client:
        payload = post_frame(client, frame, {"frame_index": "42", "timestamp_s": "1.5"})

    assert set(PREDICT_RESPONSE_KEYS) <= set(payload)
    assert payload["frame_index"] == 42
    assert payload["candidate_count"] >= 1
    assert payload["measurement_available"] is True
    assert payload["using_prediction_only"] is False
    assert payload["confidence"] == pytest.approx(0.92, abs=1e-3)
    assert payload["processing_time_ms"] >= 0.0


def test_tracker_state_persists_between_frames() -> None:
    frames = [
        make_frame([(50, 50, 5, 255)]),
        make_frame([(55, 50, 5, 255)]),
        make_frame([(60, 50, 5, 255)]),
    ]
    with make_client() as client:
        results = [post_frame(client, frame, {"frame_index": str(index)}) for index, frame in enumerate(frames)]

    assert results[0]["lock_state"] == "ACQUIRING"
    assert results[1]["lock_state"] == "ACQUIRING"
    assert results[2]["lock_state"] == "LOCKED"
    assert results[2]["target_found"] is True
    assert results[2]["temporally_verified"] is True
    assert results[2]["filtered_x_px"] == pytest.approx(60.0, abs=6.0)
    assert results[2]["filtered_y_px"] == pytest.approx(50.0, abs=6.0)


def test_session_trackers_are_independent() -> None:
    frame = make_frame([(70, 40, 5, 255)])
    with make_client() as client:
        session_a = post_frame(client, frame, {"frame_index": "0", "session_id": "session_001"})
        session_b = post_frame(client, frame, {"frame_index": "0", "session_id": "session_002"})
        client.post("/reset", data={"session_id": "session_001"})
        after_reset_a = post_frame(client, frame, {"frame_index": "0", "session_id": "session_001"})
        after_reset_b = post_frame(client, frame, {"frame_index": "1", "session_id": "session_002"})

    assert session_a["lock_state"] == "ACQUIRING"
    assert session_b["lock_state"] == "ACQUIRING"
    assert after_reset_a["lock_state"] == "ACQUIRING"
    assert after_reset_a["frame_index"] == 0
    assert after_reset_b["frame_index"] == 1


def test_invalid_image_returns_400() -> None:
    with make_client() as client:
        response = client.post(
            "/predict-frame",
            files={"file": ("frame.png", b"not-an-image", "image/png")},
        )

    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def test_missing_checkpoint_message_mentions_gitignore() -> None:
    from src.api import missing_checkpoint_message

    message = missing_checkpoint_message(Path("models/checkpoints/best_classifier.pt"))
    assert "gitignored" in message
    assert "FSOC_CHECKPOINT" in message


def test_require_checkpoint_raises_when_weights_missing(monkeypatch) -> None:
    from src.api import create_runtime

    import src.api as api_module

    monkeypatch.setenv("FSOC_CHECKPOINT", "models/checkpoints/definitely_missing_test_checkpoint.pt")
    monkeypatch.setenv("FSOC_REQUIRE_CHECKPOINT", "1")
    monkeypatch.setattr(api_module, "FALLBACK_CHECKPOINTS", ())
    with pytest.raises(FileNotFoundError, match="gitignored"):
        create_runtime("configs/unity.yaml")





def test_config_endpoint_reports_unity_metadata() -> None:
    with make_client() as client:
        response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"] == "configs/unity.yaml"
    assert payload["frame_width"] == 120
    assert payload["frame_height"] == 100
    assert payload["target_id"] == "Terminal_B"
    assert payload["coordinate_origin"] == "top-left"
    assert payload["control_error_range"] == [-1.0, 1.0]
    assert set(PREDICT_RESPONSE_KEYS) <= set(payload["response_keys"])


def test_sessions_endpoint_lists_default_and_created_sessions() -> None:
    frame = make_frame([(70, 40, 5, 255)])
    with make_client() as client:
        before = client.get("/sessions").json()
        post_frame(client, frame, {"session_id": "unity_demo"})
        after = client.get("/sessions").json()

    assert before["active_sessions"] == ["default"]
    assert after["count"] == 2
    assert after["active_sessions"] == ["default", "unity_demo"]


def test_default_config_path_uses_detector_positive_config(monkeypatch) -> None:
    from src.api import default_config_path

    monkeypatch.delenv("FSOC_CONFIG", raising=False)
    assert default_config_path().as_posix() == "configs/unity_detector_positive.yaml"


def test_root_route_points_to_docs_and_health() -> None:
    with make_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "FSOC ML/CV Unity API"
    assert payload["status"] == "running"
    assert payload["docs"] == "/docs"
    assert payload["health"] == "/health"
    assert payload["predict_frame"] == "/predict-frame"

