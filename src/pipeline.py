from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - configuration dependency is listed in requirements.txt
    yaml = None

try:
    from .beacon_classifier import CLASS_NAMES, DEFAULT_IMAGE_SIZE, BeaconClassifier, load_classifier, patch_to_tensor
    from .candidate_detector import BeaconCandidate, DetectorConfig, detect_candidates
    from .preprocessing import PreprocessingConfig, load_image
    from .prepare_patches import crop_with_padding
except ImportError:  # pragma: no cover - allows direct script execution
    from beacon_classifier import CLASS_NAMES, DEFAULT_IMAGE_SIZE, BeaconClassifier, load_classifier, patch_to_tensor
    from candidate_detector import BeaconCandidate, DetectorConfig, detect_candidates
    from preprocessing import PreprocessingConfig, load_image
    from prepare_patches import crop_with_padding


PIPELINE_OUTPUT_KEYS = {
    "target_found",
    "target_id",
    "x_px",
    "y_px",
    "frame_width",
    "frame_height",
    "frame_center_x",
    "frame_center_y",
    "image_error_x_px",
    "image_error_y_px",
    "control_error_x",
    "control_error_y",
    "confidence",
    "cnn_probability",
    "cv_baseline_score",
    "fused_score",
    "candidate_count",
    "selected_candidate_id",
    "bbox",
    "inference_time_ms",
    "status",
    "candidates",
}


@dataclass(frozen=True)
class InferenceConfig:
    """Settings for single-frame inference and candidate ranking."""

    target_id: str = "Terminal_B"
    confidence_threshold: float = 0.55
    cnn_weight: float = 0.80
    cv_weight: float = 0.20
    crop_size: int = 40
    patch_size: int = DEFAULT_IMAGE_SIZE
    checkpoint_path: Path = Path("models/checkpoints/best_classifier.pt")
    device: str = "auto"


@dataclass(frozen=True)
class PipelineConfig:
    """Complete configuration for Phase 6 single-frame inference."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass(frozen=True)
class CandidatePrediction:
    """CNN and fused-ranking data for one detected bright candidate."""

    candidate: BeaconCandidate
    correct_probability: float
    false_probability: float
    predicted_class: str
    cnn_confidence: float
    cv_baseline_score: float
    fused_score: float

    def to_dict(self) -> Dict[str, object]:
        """Convert candidate prediction data into JSON-friendly output."""
        return {
            "candidate_id": int(self.candidate.candidate_id),
            "x_px": float(self.candidate.x),
            "y_px": float(self.candidate.y),
            "bbox": candidate_bbox(self.candidate),
            "area": float(self.candidate.area),
            "radius": float(self.candidate.radius),
            "mean_intensity": float(self.candidate.mean_intensity),
            "max_intensity": int(self.candidate.max_intensity),
            "circularity": float(self.candidate.circularity),
            "correct_probability": float(self.correct_probability),
            "false_probability": float(self.false_probability),
            "predicted_class": self.predicted_class,
            "cnn_confidence": float(self.cnn_confidence),
            "cv_baseline_score": float(self.cv_baseline_score),
            "fused_score": float(self.fused_score),
        }


def validate_inference_config(config: InferenceConfig) -> InferenceConfig:
    """Validate user-facing inference settings."""
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if config.crop_size <= 0 or config.patch_size <= 0:
        raise ValueError("crop_size and patch_size must be positive")
    if not 0.0 <= config.cnn_weight <= 1.0 or not 0.0 <= config.cv_weight <= 1.0:
        raise ValueError("fusion weights must be between 0 and 1")
    total_weight = config.cnn_weight + config.cv_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"fusion weights must sum to 1.0, got {total_weight:.6f}")
    return config


def select_device(requested: str) -> torch.device:
    """Select CUDA only when requested/available, otherwise use CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ensure_bgr_frame(frame: np.ndarray) -> np.ndarray:
    """Validate a NumPy frame and return a three-channel BGR uint8 copy."""
    if frame is None or frame.size == 0:
        raise ValueError("frame is missing or empty")

    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        bgr = frame.copy()
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"expected grayscale, BGR or BGRA frame, got shape {frame.shape}")

    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr.copy()


def load_frame(frame_or_path: str | Path | np.ndarray) -> np.ndarray:
    """Accept an OpenCV frame or image path and return a validated BGR frame."""
    if isinstance(frame_or_path, (str, Path)):
        return ensure_bgr_frame(load_image(frame_or_path))
    if isinstance(frame_or_path, np.ndarray):
        return ensure_bgr_frame(frame_or_path)
    raise TypeError("input must be an image path or a NumPy frame")


def calculate_fused_score(correct_probability: float, baseline_score: float, config: InferenceConfig) -> float:
    """Combine CNN probability and CV baseline score with validated weights."""
    validate_inference_config(config)
    return float(config.cnn_weight * correct_probability + config.cv_weight * baseline_score)


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clamp a numeric control value to a closed range."""
    return float(max(lower, min(upper, value)))


def calculate_coordinate_errors(
    target_x: float,
    target_y: float,
    width: int,
    height: int,
) -> Dict[str, float]:
    """Calculate image-space and normalized control-space pointing errors."""
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")

    frame_center_x = width / 2.0
    frame_center_y = height / 2.0
    image_error_x_px = float(target_x - frame_center_x)
    image_error_y_px = float(target_y - frame_center_y)
    return {
        "frame_center_x": frame_center_x,
        "frame_center_y": frame_center_y,
        "image_error_x_px": image_error_x_px,
        "image_error_y_px": image_error_y_px,
        "control_error_x": clamp(image_error_x_px / frame_center_x),
        "control_error_y": clamp(-image_error_y_px / frame_center_y),
    }


def candidate_bbox(candidate: BeaconCandidate) -> Dict[str, int]:
    """Return a nested bounding-box dictionary for one candidate."""
    return {
        "x": int(candidate.bbox_x),
        "y": int(candidate.bbox_y),
        "width": int(candidate.bbox_width),
        "height": int(candidate.bbox_height),
    }


def not_found_result(
    status: str,
    width: int,
    height: int,
    candidate_count: int,
    inference_time_ms: float,
    candidate_predictions: Optional[Sequence[CandidatePrediction]] = None,
    selected: Optional[CandidatePrediction] = None,
) -> Dict[str, object]:
    """Return a stable no-target result for normal detection failures."""
    center = calculate_coordinate_errors(width / 2.0, height / 2.0, width, height)
    selected_candidate_id = int(selected.candidate.candidate_id) if selected else None
    return {
        "target_found": False,
        "target_id": None,
        "x_px": None,
        "y_px": None,
        "frame_width": int(width),
        "frame_height": int(height),
        "frame_center_x": center["frame_center_x"],
        "frame_center_y": center["frame_center_y"],
        "image_error_x_px": None,
        "image_error_y_px": None,
        "control_error_x": None,
        "control_error_y": None,
        "confidence": 0.0,
        "cnn_probability": float(selected.correct_probability) if selected else 0.0,
        "cv_baseline_score": float(selected.cv_baseline_score) if selected else 0.0,
        "fused_score": float(selected.fused_score) if selected else 0.0,
        "candidate_count": int(candidate_count),
        "selected_candidate_id": selected_candidate_id,
        "bbox": candidate_bbox(selected.candidate) if selected else None,
        "inference_time_ms": float(inference_time_ms),
        "status": status,
        "candidates": [prediction.to_dict() for prediction in candidate_predictions or []],
    }


def found_result(
    selected: CandidatePrediction,
    width: int,
    height: int,
    target_id: str,
    candidate_predictions: Sequence[CandidatePrediction],
    inference_time_ms: float,
) -> Dict[str, object]:
    """Return a PID-ready target result for the accepted selected candidate."""
    errors = calculate_coordinate_errors(selected.candidate.x, selected.candidate.y, width, height)
    return {
        "target_found": True,
        "target_id": target_id,
        "x_px": float(selected.candidate.x),
        "y_px": float(selected.candidate.y),
        "frame_width": int(width),
        "frame_height": int(height),
        "frame_center_x": errors["frame_center_x"],
        "frame_center_y": errors["frame_center_y"],
        "image_error_x_px": errors["image_error_x_px"],
        "image_error_y_px": errors["image_error_y_px"],
        "control_error_x": errors["control_error_x"],
        "control_error_y": errors["control_error_y"],
        "confidence": float(selected.correct_probability),
        "cnn_probability": float(selected.correct_probability),
        "cv_baseline_score": float(selected.cv_baseline_score),
        "fused_score": float(selected.fused_score),
        "candidate_count": len(candidate_predictions),
        "selected_candidate_id": int(selected.candidate.candidate_id),
        "bbox": candidate_bbox(selected.candidate),
        "inference_time_ms": float(inference_time_ms),
        "status": "target_detected",
        "candidates": [prediction.to_dict() for prediction in candidate_predictions],
    }


def validate_output_schema(result: Mapping[str, object]) -> None:
    """Ensure found and not-found results expose the same top-level fields."""
    missing = sorted(PIPELINE_OUTPUT_KEYS - set(result.keys()))
    if missing:
        raise ValueError(f"pipeline output missing keys: {missing}")


class SingleFramePipeline:
    """Run preprocessing, detection, CNN inference and candidate selection for one frame."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model: Optional[BeaconClassifier] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.inference_config = validate_inference_config(self.config.inference)
        self.device = select_device(self.inference_config.device)

        if model is None:
            self.model = load_classifier(self.inference_config.checkpoint_path, self.device)
        else:
            self.model = model.to(self.device)
            self.model.eval()

    def run(self, frame_or_path: str | Path | np.ndarray) -> Dict[str, object]:
        """Run the complete Phase 6 pipeline for one frame or image path."""
        start = time.perf_counter()
        frame = load_frame(frame_or_path)
        height, width = frame.shape[:2]

        detection = detect_candidates(frame, self.config.preprocessing, self.config.detector)
        candidates = detection["candidates"]
        if not isinstance(candidates, list):
            raise ValueError("candidate detector returned invalid candidate list")

        if not candidates:
            elapsed = elapsed_ms(start)
            result = not_found_result("no_candidates", width, height, 0, elapsed)
            validate_output_schema(result)
            return result

        predictions = self.predict_candidates(frame, candidates)
        selected = max(predictions, key=lambda item: item.fused_score)
        elapsed = elapsed_ms(start)

        if selected.correct_probability < self.inference_config.confidence_threshold:
            result = not_found_result(
                "below_confidence_threshold",
                width,
                height,
                len(predictions),
                elapsed,
                candidate_predictions=predictions,
                selected=selected,
            )
        else:
            result = found_result(
                selected=selected,
                width=width,
                height=height,
                target_id=self.inference_config.target_id,
                candidate_predictions=predictions,
                inference_time_ms=elapsed,
            )
        validate_output_schema(result)
        return result

    def predict_candidates(self, frame_bgr: np.ndarray, candidates: Sequence[BeaconCandidate]) -> List[CandidatePrediction]:
        """Crop every candidate patch, run one CNN batch, and fuse the scores."""
        patches_rgb = [
            prepare_candidate_patch(frame_bgr, candidate, self.inference_config.crop_size, self.inference_config.patch_size)
            for candidate in candidates
        ]
        tensors = torch.stack([patch_to_tensor(patch, image_size=self.inference_config.patch_size) for patch in patches_rgb])
        tensors = tensors.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensors)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"classifier must return logits with shape [N, 2], got {tuple(logits.shape)}")
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

        if probabilities.shape[0] != len(candidates):
            raise ValueError("classifier output count does not match candidate count")

        predictions = []
        for candidate, probabilities_row in zip(candidates, probabilities):
            false_probability = float(probabilities_row[0])
            correct_probability = float(probabilities_row[1])
            label_id = int(np.argmax(probabilities_row))
            predicted_class = CLASS_NAMES[label_id] if label_id < len(CLASS_NAMES) else str(label_id)
            cnn_confidence = float(probabilities_row[label_id])
            baseline_score = float(candidate.baseline_score)
            fused_score = calculate_fused_score(correct_probability, baseline_score, self.inference_config)
            predictions.append(
                CandidatePrediction(
                    candidate=candidate,
                    correct_probability=correct_probability,
                    false_probability=false_probability,
                    predicted_class=predicted_class,
                    cnn_confidence=cnn_confidence,
                    cv_baseline_score=baseline_score,
                    fused_score=fused_score,
                )
            )
        return predictions


def prepare_candidate_patch(frame_bgr: np.ndarray, candidate: BeaconCandidate, crop_size: int, patch_size: int) -> np.ndarray:
    """Crop a BGR candidate using patch-prep logic and return an RGB patch for the CNN."""
    patch_bgr = crop_with_padding(frame_bgr, center=(candidate.x, candidate.y), crop_size=crop_size, patch_size=patch_size)
    return cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)


def elapsed_ms(start_time: float) -> float:
    """Return elapsed milliseconds from a perf_counter start."""
    return round((time.perf_counter() - start_time) * 1000.0, 3)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw readable text with a dark outline."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def candidate_label_origin(
    point: Tuple[int, int],
    text: str,
    width: int,
    height: int,
    scale: float = 0.45,
    thickness: int = 1,
) -> Tuple[int, int]:
    """Place candidate labels inside the image even near frame edges."""
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = point[0] + 8
    if x + text_width > width - 4:
        x = max(4, point[0] - text_width - 8)
    y = max(text_height + 4, point[1] - 8)
    if y > height - 4:
        y = height - 4
    return int(x), int(y)


def selected_candidate_from_result(result: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """Find the selected candidate dictionary inside a pipeline result."""
    selected_id = result.get("selected_candidate_id")
    if selected_id is None:
        return None
    for candidate in result.get("candidates", []):
        if isinstance(candidate, Mapping) and int(candidate["candidate_id"]) == int(selected_id):
            return candidate
    return None


def draw_pipeline_diagnostics(frame_or_path: str | Path | np.ndarray, result: Mapping[str, object]) -> np.ndarray:
    """Create an annotated frame showing candidates, scores and target decision."""
    overlay = load_frame(frame_or_path)
    height, width = overlay.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(overlay, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 1)

    selected_id = result.get("selected_candidate_id")
    target_found = bool(result.get("target_found"))
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = int(candidate["candidate_id"])
        point = (int(round(float(candidate["x_px"]))), int(round(float(candidate["y_px"]))))
        bbox = candidate.get("bbox", {})
        is_selected = selected_id is not None and candidate_id == int(selected_id)
        color = (0, 255, 0) if is_selected and target_found else (0, 128, 255) if is_selected else (0, 255, 255)
        if isinstance(bbox, Mapping):
            cv2.rectangle(
                overlay,
                (int(bbox["x"]), int(bbox["y"])),
                (int(bbox["x"]) + int(bbox["width"]), int(bbox["y"]) + int(bbox["height"])),
                color,
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.circle(overlay, point, 5, color, 1, lineType=cv2.LINE_AA)
        label = (
            f"#{candidate_id} CNN {float(candidate['correct_probability']):.2f} "
            f"CV {float(candidate['cv_baseline_score']):.2f} F {float(candidate['fused_score']):.2f}"
        )
        draw_text(overlay, label, candidate_label_origin(point, label, width, height), color, scale=0.45)

    if selected_id is not None:
        selected_candidate = selected_candidate_from_result(result)
        if selected_candidate is not None:
            point = (int(round(float(selected_candidate["x_px"]))), int(round(float(selected_candidate["y_px"]))))
            color = (0, 255, 0) if target_found else (0, 128, 255)
            cv2.circle(overlay, point, 14, color, 2, lineType=cv2.LINE_AA)
            cv2.line(overlay, center, point, color, 1, lineType=cv2.LINE_AA)

    status_text = "TARGET FOUND" if target_found else "TARGET NOT FOUND"
    status_color = (0, 255, 0) if target_found else (0, 128, 255)
    draw_text(overlay, status_text, (12, 28), status_color, scale=0.75, thickness=2)
    draw_text(overlay, f"Status: {result.get('status')}", (12, 56), (255, 255, 255), scale=0.55)
    draw_text(overlay, f"Candidates: {result.get('candidate_count')}", (12, 82), (255, 255, 255), scale=0.55)

    if result.get("image_error_x_px") is not None and result.get("image_error_y_px") is not None:
        error_text = f"Pixel error: dx={float(result['image_error_x_px']):.1f}px dy={float(result['image_error_y_px']):.1f}px"
    elif selected_id is not None:
        selected_candidate = selected_candidate_from_result(result)
        if selected_candidate is not None:
            dx = float(selected_candidate["x_px"]) - float(result["frame_center_x"])
            dy = float(selected_candidate["y_px"]) - float(result["frame_center_y"])
            error_text = f"Rejected error: dx={dx:.1f}px dy={dy:.1f}px"
        else:
            error_text = "Pixel error: n/a"
    else:
        error_text = "Pixel error: n/a"
    draw_text(overlay, error_text, (12, 108), (255, 255, 255), scale=0.55)
    return overlay


def load_config(config_path: str | Path, checkpoint_override: Optional[str | Path] = None, device_override: Optional[str] = None) -> PipelineConfig:
    """Load YAML configuration and normalize it into dataclasses."""
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration files")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("configuration file must contain a mapping")
    return pipeline_config_from_mapping(raw, checkpoint_override=checkpoint_override, device_override=device_override)


def section(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a nested mapping section or an empty mapping."""
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def pipeline_config_from_mapping(
    raw: Mapping[str, object],
    checkpoint_override: Optional[str | Path] = None,
    device_override: Optional[str] = None,
) -> PipelineConfig:
    """Build a PipelineConfig from the project YAML layout."""
    project = section(raw, "project")
    image = section(raw, "image")
    preprocessing = section(raw, "preprocessing")
    detector = section(raw, "detector")
    classifier = section(raw, "classifier")
    pipeline = section(raw, "pipeline")
    fusion = section(pipeline, "fusion") or section(classifier, "fusion")

    preprocessing_config = PreprocessingConfig(
        threshold_method=str(preprocessing.get("threshold_method", detector.get("threshold_method", "fixed"))),
        threshold_value=int(preprocessing.get("threshold_value", preprocessing.get("threshold", detector.get("threshold", 200)))),
        blur_kernel=int(preprocessing.get("blur_kernel", detector.get("blur_kernel", 5))),
        morph_kernel=int(preprocessing.get("morph_kernel", detector.get("morph_kernel", 3))),
        opening_iterations=int(preprocessing.get("opening_iterations", detector.get("opening_iterations", 1))),
        closing_iterations=int(preprocessing.get("closing_iterations", detector.get("closing_iterations", 1))),
        normalize_contrast=bool(preprocessing.get("normalize_contrast", False)),
    )
    detector_config = DetectorConfig(
        min_area=float(detector.get("min_area", 3.0)),
        max_area=float(detector.get("max_area", 1000.0)),
        min_radius=float(detector.get("min_radius", 1.0)),
        max_radius=float(detector.get("max_radius", 25.0)),
        min_circularity=float(detector.get("min_circularity", detector.get("circularity_min", 0.15))),
        match_tolerance=float(detector.get("match_tolerance", 12.0)),
    )
    checkpoint = checkpoint_override or classifier.get("model_path", "models/checkpoints/best_classifier.pt")
    inference_config = InferenceConfig(
        target_id=str(pipeline.get("target_id", project.get("target_id", "Terminal_B"))),
        confidence_threshold=float(classifier.get("confidence_threshold", pipeline.get("confidence_threshold", 0.55))),
        cnn_weight=float(fusion.get("cnn_weight", 0.80)),
        cv_weight=float(fusion.get("cv_weight", 0.20)),
        crop_size=int(pipeline.get("crop_size", image.get("crop_size", 40))),
        patch_size=int(pipeline.get("patch_size", image.get("patch_size", DEFAULT_IMAGE_SIZE))),
        checkpoint_path=Path(checkpoint),
        device=str(device_override or pipeline.get("device", "auto")),
    )
    validate_inference_config(inference_config)
    return PipelineConfig(preprocessing=preprocessing_config, detector=detector_config, inference=inference_config)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for single-frame inference."""
    parser = argparse.ArgumentParser(description="Run the Phase 6 FSOC single-frame inference pipeline.")
    parser.add_argument("--image", required=True, help="Input frame image path.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration path.")
    parser.add_argument("--checkpoint", default=None, help="Optional classifier checkpoint override.")
    parser.add_argument("--output", default="outputs/pipeline-test", help="Output folder for result JSON and annotated image.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu, cuda or auto.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, checkpoint_override=args.checkpoint, device_override=args.device)
    pipeline = SingleFramePipeline(config)
    result = pipeline.run(args.image)
    annotated = draw_pipeline_diagnostics(args.image, result)

    result_path = output_dir / "result.json"
    annotated_path = output_dir / "annotated_result.png"
    write_json(result_path, result)
    if not cv2.imwrite(str(annotated_path), annotated):
        raise OSError(f"failed to write annotated image: {annotated_path}")

    print("Phase 6 single-frame pipeline")
    print(f"  Image: {args.image}")
    print(f"  Target found: {result['target_found']}")
    print(f"  Status: {result['status']}")
    print(f"  Candidates: {result['candidate_count']}")
    print(f"  Confidence: {float(result['confidence']):.3f}")
    print(f"  Fused score: {float(result['fused_score']):.3f}")
    print(f"  Result JSON: {result_path}")
    print(f"  Annotated image: {annotated_path}")


if __name__ == "__main__":
    main()
