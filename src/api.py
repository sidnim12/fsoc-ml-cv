from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .beacon_classifier import BeaconClassifier
    from .pipeline import SingleFramePipeline, load_config
    from .tracker import TargetTracker, TrackingConfig, load_tracking_config
except ImportError:  # pragma: no cover - allows direct script execution
    from beacon_classifier import BeaconClassifier
    from pipeline import SingleFramePipeline, load_config
    from tracker import TargetTracker, TrackingConfig, load_tracking_config


DEFAULT_CONFIG_PATH = Path("configs/unity_detector_positive.yaml")
DEFAULT_SESSION_ID = "default"
FALLBACK_CHECKPOINTS = (
    Path("models/checkpoints/official_1600x900_detector_positive/best_classifier.pt"),
    Path("models/checkpoints/best_classifier.pt"),
    Path("models/checkpoints/official_1600x900_v2/best_classifier.pt"),
    Path("models/checkpoints/unity_base_2400/best_classifier.pt"),
)

PREDICT_RESPONSE_KEYS = (
    "target_found",
    "lock_state",
    "measured_x_px",
    "measured_y_px",
    "filtered_x_px",
    "filtered_y_px",
    "predicted_x_px",
    "predicted_y_px",
    "control_error_x",
    "control_error_y",
    "confidence",
    "frame_index",
    "measurement_available",
    "candidate_count",
    "temporally_verified",
    "using_prediction_only",
    "missed_frames",
    "processing_time_ms",
)


def cors_origins() -> list[str]:
    """Return browser origins allowed to call the API dashboard."""
    configured = os.environ.get("FSOC_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


def default_config_path() -> Path:
    """Return the YAML config used to load the pipeline and tracker once."""
    return Path(os.environ.get("FSOC_CONFIG", DEFAULT_CONFIG_PATH))


def display_path(path: Path) -> str:
    """Return a stable relative path for API metadata."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_session_id(session_id: Optional[str]) -> str:
    """Use one default tracker when Unity omits session_id."""
    if session_id is None:
        return DEFAULT_SESSION_ID
    stripped = str(session_id).strip()
    return stripped if stripped else DEFAULT_SESSION_ID


def get_tracker(request: Request, session_id: Optional[str] = None) -> TargetTracker:
    """Return the per-session tracker, creating it on first use."""
    key = resolve_session_id(session_id)
    trackers: Dict[str, TargetTracker] = request.app.state.trackers
    if key not in trackers:
        trackers[key] = TargetTracker(request.app.state.tracking_config)
    return trackers[key]


def decode_frame(image_bytes: bytes) -> np.ndarray:
    """Decode a PNG/JPG upload into a BGR OpenCV frame."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image upload")
    array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="could not decode image frame")
    return frame


def predict_payload(tracking_result: Mapping[str, Any], processing_time_ms: float) -> Dict[str, Any]:
    """Map tracker output onto the Unity/PID JSON contract."""
    return {
        "target_found": bool(tracking_result["target_found"]),
        "lock_state": str(tracking_result["lock_state"]),
        "measured_x_px": tracking_result.get("measured_x_px"),
        "measured_y_px": tracking_result.get("measured_y_px"),
        "filtered_x_px": tracking_result.get("filtered_x_px"),
        "filtered_y_px": tracking_result.get("filtered_y_px"),
        "predicted_x_px": tracking_result.get("predicted_x_px"),
        "predicted_y_px": tracking_result.get("predicted_y_px"),
        "control_error_x": tracking_result.get("control_error_x"),
        "control_error_y": tracking_result.get("control_error_y"),
        "confidence": float(tracking_result.get("confidence", 0.0)),
        "frame_index": tracking_result.get("frame_index"),
        "measurement_available": bool(tracking_result.get("measurement_available")),
        "candidate_count": int(tracking_result.get("candidate_count", 0)),
        "temporally_verified": bool(tracking_result.get("temporally_verified")),
        "using_prediction_only": bool(tracking_result.get("using_prediction_only")),
        "missed_frames": int(tracking_result.get("missed_frames", 0)),
        "processing_time_ms": float(processing_time_ms),
    }


def require_checkpoint() -> bool:
    """Fail startup when no .pt file exists, instead of using an untrained model."""
    return os.environ.get("FSOC_REQUIRE_CHECKPOINT", "").strip().lower() in {"1", "true", "yes"}


def missing_checkpoint_message(expected: Path) -> str:
    """Explain that Git does not store classifier weights."""
    searched = ", ".join(str(path) for path in (expected, *FALLBACK_CHECKPOINTS))
    return (
        f"checkpoint does not exist: {expected}. "
        "Classifier weights are gitignored, so they are not in a fresh clone. "
        f"Looked for: {searched}. "
        "Copy a trained best_classifier.pt into models/checkpoints/, "
        "or set FSOC_CHECKPOINT=/path/to/best_classifier.pt"
    )


def find_checkpoint(preferred: Path) -> Optional[Path]:
    """Return the first existing checkpoint, preferring the configured path."""
    if preferred.exists():
        return preferred
    for candidate in FALLBACK_CHECKPOINTS:
        if candidate.exists():
            return candidate
    return None


def create_runtime(
    config_path: str | Path,
    pipeline: Optional[SingleFramePipeline] = None,
    tracking_config: Optional[TrackingConfig] = None,
) -> Dict[str, Any]:
    """Load the classifier once and create the default tracker."""
    path = Path(config_path)
    env_checkpoint = os.environ.get("FSOC_CHECKPOINT")
    pipeline_config = load_config(path, checkpoint_override=env_checkpoint)
    loaded_tracking = tracking_config if tracking_config is not None else load_tracking_config(path)
    model_loaded = True

    if pipeline is not None:
        loaded_pipeline = pipeline
    else:
        preferred = Path(pipeline_config.inference.checkpoint_path)
        found = find_checkpoint(preferred)
        if found is not None:
            if found != preferred:
                pipeline_config = load_config(path, checkpoint_override=found)
            loaded_pipeline = SingleFramePipeline(pipeline_config)
        elif require_checkpoint():
            raise FileNotFoundError(missing_checkpoint_message(preferred))
        else:
            print(f"WARNING: {missing_checkpoint_message(preferred)}")
            print("WARNING: API is starting with an untrained classifier. Predictions will not be meaningful.")
            loaded_pipeline = SingleFramePipeline(pipeline_config, model=BeaconClassifier())
            model_loaded = False

    return {
        "pipeline": loaded_pipeline,
        "pipeline_config": pipeline_config,
        "tracking_config": loaded_tracking,
        "trackers": {DEFAULT_SESSION_ID: TargetTracker(loaded_tracking)},
        "config_path": display_path(path),
        "checkpoint_path": display_path(Path(pipeline_config.inference.checkpoint_path)),
        "model_loaded": model_loaded,
    }


def create_app(
    pipeline: Optional[SingleFramePipeline] = None,
    tracking_config: Optional[TrackingConfig] = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Build the FastAPI app. Tests inject a pipeline so the checkpoint is not loaded."""
    resolved_config = Path(config_path) if config_path is not None else default_config_path()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = create_runtime(resolved_config, pipeline=pipeline, tracking_config=tracking_config)
        app.state.pipeline = runtime["pipeline"]
        app.state.pipeline_config = runtime["pipeline_config"]
        app.state.tracking_config = runtime["tracking_config"]
        app.state.trackers = runtime["trackers"]
        app.state.config_path = runtime["config_path"]
        app.state.checkpoint_path = runtime["checkpoint_path"]
        app.state.model_loaded = runtime["model_loaded"]
        yield
        app.state.trackers.clear()

    app = FastAPI(
        title="FSOC ML/CV Unity API",
        description="Single-frame pipeline plus stateful tracker for Unity camera frames.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "name": "FSOC ML/CV Unity API",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "config": "/config",
            "predict_frame": "/predict-frame",
        }

    @app.get("/health")
    def health_check(request: Request) -> Dict[str, Any]:
        pipeline_obj: SingleFramePipeline = request.app.state.pipeline
        trackers: Dict[str, TargetTracker] = request.app.state.trackers
        model_loaded = bool(request.app.state.model_loaded)
        return {
            "status": "healthy" if bool(trackers) else "unhealthy",
            "model_loaded": model_loaded,
            "tracker_ready": bool(trackers),
            "device": str(pipeline_obj.device),
            "config": request.app.state.config_path,
            "checkpoint": request.app.state.checkpoint_path,
            "active_sessions": len(trackers),
        }

    @app.get("/config")
    def config_info(request: Request) -> Dict[str, Any]:
        pipeline_config = request.app.state.pipeline_config
        tracking_config = request.app.state.tracking_config
        return {
            "config": request.app.state.config_path,
            "checkpoint": request.app.state.checkpoint_path,
            "frame_width": int(tracking_config.frame_width),
            "frame_height": int(tracking_config.frame_height),
            "target_id": str(pipeline_config.inference.target_id),
            "confidence_threshold": float(pipeline_config.inference.confidence_threshold),
            "response_keys": list(PREDICT_RESPONSE_KEYS),
            "coordinate_origin": "top-left",
            "control_error_range": [-1.0, 1.0],
        }

    @app.get("/sessions")
    def sessions(request: Request) -> Dict[str, Any]:
        trackers: Dict[str, TargetTracker] = request.app.state.trackers
        return {"active_sessions": sorted(trackers.keys()), "count": len(trackers)}

    @app.post("/reset")
    def reset_tracker(request: Request, session_id: Optional[str] = Form(None)) -> Dict[str, str]:
        tracker = get_tracker(request, session_id)
        tracker.reset()
        return {"status": "reset", "message": "Tracker state cleared"}

    @app.post("/predict-frame")
    async def predict_frame(
        request: Request,
        file: UploadFile = File(...),
        frame_index: Optional[int] = Form(None),
        timestamp_s: Optional[float] = Form(None),
        session_id: Optional[str] = Form(None),
    ) -> JSONResponse:
        del timestamp_s  # accepted for Unity metadata; not used by the ML/CV loop
        start = time.perf_counter()
        image_bytes = await file.read()
        frame = decode_frame(image_bytes)
        phase6_result = request.app.state.pipeline.run(frame)
        tracking_result = get_tracker(request, session_id).update(phase6_result, frame_index=frame_index)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        payload = predict_payload(tracking_result, elapsed_ms)
        missing = [key for key in PREDICT_RESPONSE_KEYS if key not in payload]
        if missing:
            raise HTTPException(status_code=500, detail=f"predict response missing keys: {missing}")
        return JSONResponse(payload)

    return app


app = create_app()



