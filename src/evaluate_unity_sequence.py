from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .pipeline import SingleFramePipeline, load_config
    from .tracker import TRACKING_OUTPUT_KEYS, TargetTracker, load_tracking_config
    from .unity_dataset_adapter import (
        MANIFEST_COLUMNS,
        UnitySequenceInventory,
        bool_or_none,
        build_and_write_manifest,
        discover_unity_sequence,
        discover_unity_sequences,
        float_or_none,
        has_ground_truth,
        missing_label_image_references,
        normalize_labels,
        print_inventory,
        sequence_slug,
        validate_coordinate_origin,
        validate_unity_labels,
        validate_unity_sequence,
        write_required_labels_template,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from pipeline import SingleFramePipeline, load_config
    from tracker import TRACKING_OUTPUT_KEYS, TargetTracker, load_tracking_config
    from unity_dataset_adapter import (
        MANIFEST_COLUMNS,
        UnitySequenceInventory,
        bool_or_none,
        build_and_write_manifest,
        discover_unity_sequence,
        discover_unity_sequences,
        float_or_none,
        has_ground_truth,
        missing_label_image_references,
        normalize_labels,
        print_inventory,
        sequence_slug,
        validate_coordinate_origin,
        validate_unity_labels,
        validate_unity_sequence,
        write_required_labels_template,
    )


PHASE6_COLUMNS = [
    "phase6_target_found",
    "phase6_status",
    "phase6_selected_candidate_id",
    "phase6_x_px",
    "phase6_y_px",
    "phase6_cnn_probability",
    "phase6_cv_baseline_score",
    "phase6_fused_score",
    "phase6_inference_time_ms",
    "candidates_json",
]

EXTRA_PREDICTION_COLUMNS = [
    "total_processing_time_ms",
    "measurement_error_px",
    "filtered_error_px",
    "failure_reason",
]

TRACKING_COLUMNS = [key for key in sorted(TRACKING_OUTPUT_KEYS) if key not in MANIFEST_COLUMNS]
FRAME_PREDICTION_COLUMNS = MANIFEST_COLUMNS + PHASE6_COLUMNS + TRACKING_COLUMNS + EXTRA_PREDICTION_COLUMNS

EVALUATION_RESULT_KEYS = {
    "inventory",
    "dataset_validation",
    "ground_truth_available",
    "manifest_path",
    "output_manifest_path",
    "frame_predictions_path",
    "detection_metrics",
    "tracking_metrics",
    "performance_metrics",
    "domain_gap_report",
    "evaluation_summary",
    "output_dir",
}

BATCH_METRICS_COLUMNS = [
    "sequence_dir",
    "outputs_dir",
    "status",
    "image_count",
    "resolution",
    "labels_found",
    "ground_truth_available",
    "candidate_recall",
    "accepted_detection_recall",
    "filtered_mae_px",
    "locked_frame_percentage",
    "mean_processing_time_ms",
    "effective_processing_fps",
    "most_common_failure_reason",
    "error",
]


PipelineFactory = Callable[[], object]
TrackerFactory = Callable[[], object]


def json_ready(value: object) -> object:
    """Convert common NumPy, Path and nested values into JSON-safe data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write indented JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write CSV rows using an explicit stable field order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value: object) -> object:
    """Keep CSV cells readable."""
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    return value


def safe_folder_name(value: str) -> str:
    """Return a filesystem-friendly lowercase folder name."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "sequence"


def resolved_frame_size(inventory: UnitySequenceInventory, expected_width: Optional[int], expected_height: Optional[int]) -> Tuple[int, int]:
    """Resolve the frame size from CLI expectations or the first readable image."""
    width = expected_width if expected_width is not None else inventory.width
    height = expected_height if expected_height is not None else inventory.height
    if width is None or height is None:
        raise ValueError("could not infer frame resolution; pass --expected-width and --expected-height")
    return int(width), int(height)


def validate_output_scale(output_scale: int) -> int:
    """Validate output video scaling."""
    if output_scale < 1:
        raise ValueError("output_scale must be a positive integer")
    return int(output_scale)


def scaled_video_frame(overlay: np.ndarray, output_scale: int) -> np.ndarray:
    """Scale only the annotated video frame, never the source dataset image."""
    if output_scale == 1:
        return overlay
    height, width = overlay.shape[:2]
    return cv2.resize(overlay, (width * output_scale, height * output_scale), interpolation=cv2.INTER_CUBIC)


def batch_output_folder(output_root: Path, inventory: UnitySequenceInventory) -> Path:
    """Create a stable per-sequence output folder before labels are read."""
    return output_root / safe_folder_name(sequence_slug(inventory, []))


def make_default_pipeline(config_path: str | Path, checkpoint: Optional[str | Path], device: Optional[str]) -> SingleFramePipeline:
    """Create the Phase 6 pipeline exactly once for a sequence run."""
    pipeline_config = load_config(config_path, checkpoint_override=checkpoint, device_override=device)
    return SingleFramePipeline(pipeline_config)


def make_default_tracker(config_path: str | Path) -> TargetTracker:
    """Create the Phase 7 tracker exactly once for a sequence run."""
    return TargetTracker(load_tracking_config(config_path))


def ground_truth_point(row: Mapping[str, object]) -> Optional[Tuple[float, float]]:
    """Return OpenCV-space ground-truth target point when available."""
    if bool_or_none(row.get("target_visible")) is not True:
        return None
    x_value = float_or_none(row.get("target_x"))
    y_value = float_or_none(row.get("target_y"))
    if x_value is None or y_value is None:
        return None
    return x_value, y_value


def euclidean_error(x_value: object, y_value: object, gt: Optional[Tuple[float, float]]) -> Optional[float]:
    """Return Euclidean coordinate error in pixels."""
    if gt is None:
        return None
    x_number = float_or_none(x_value)
    y_number = float_or_none(y_value)
    if x_number is None or y_number is None:
        return None
    return float(math.hypot(x_number - gt[0], y_number - gt[1]))


def candidate_hits_ground_truth(candidates: Sequence[Mapping[str, object]], gt: Tuple[float, float], tolerance_px: float) -> bool:
    """Return whether any candidate is close enough to ground truth."""
    for candidate in candidates:
        x_value = float_or_none(candidate.get("x_px"))
        y_value = float_or_none(candidate.get("y_px"))
        if x_value is None or y_value is None:
            continue
        if math.hypot(x_value - gt[0], y_value - gt[1]) <= tolerance_px:
            return True
    return False


def selected_hits_ground_truth(row: Mapping[str, object], tolerance_px: float) -> bool:
    """Return whether the Phase 6 selected coordinate matches ground truth."""
    gt = ground_truth_point(row)
    if gt is None:
        return False
    error = euclidean_error(row.get("phase6_x_px"), row.get("phase6_y_px"), gt)
    return error is not None and error <= tolerance_px


def determine_failure_reason(row: Mapping[str, object], ground_truth_available: bool, match_tolerance_px: float) -> str:
    """Classify a compact failure reason for domain-gap analysis."""
    candidate_count = int(row.get("candidate_count") or 0)
    phase6_target_found = bool(row.get("phase6_target_found"))
    status = str(row.get("phase6_status") or "")
    visible = bool_or_none(row.get("target_visible"))

    if ground_truth_available:
        if visible is True:
            if candidate_count == 0:
                return "missed_no_candidates"
            if not phase6_target_found:
                return status or "below_confidence_threshold"
            if not selected_hits_ground_truth(row, match_tolerance_px):
                return "selected_candidate_far_from_ground_truth"
            if row.get("lock_state") in {"COASTING", "LOST"}:
                return f"tracking_{str(row['lock_state']).lower()}"
            return "ok"
        if visible is False and phase6_target_found:
            return "false_detection_target_absent"
        return "ok"

    if candidate_count == 0:
        return "no_candidates"
    if not phase6_target_found:
        return status or "below_confidence_threshold"
    if row.get("lock_state") in {"COASTING", "LOST"}:
        return f"tracking_{str(row['lock_state']).lower()}"
    return "unlabelled_detection"


def build_prediction_row(
    manifest_row: Mapping[str, object],
    phase6_result: Mapping[str, object],
    tracking_result: Mapping[str, object],
    total_processing_time_ms: float,
    ground_truth_available: bool,
    match_tolerance_px: float,
) -> Dict[str, object]:
    """Combine manifest, Phase 6 and Phase 7 results for one frame."""
    candidates = [dict(candidate) for candidate in phase6_result.get("candidates", []) if isinstance(candidate, Mapping)]
    gt = ground_truth_point(manifest_row)
    row: Dict[str, object] = {column: manifest_row.get(column, "") for column in MANIFEST_COLUMNS}
    row.update(
        {
            "phase6_target_found": bool(phase6_result.get("target_found")),
            "phase6_status": phase6_result.get("status"),
            "phase6_selected_candidate_id": phase6_result.get("selected_candidate_id"),
            "phase6_x_px": phase6_result.get("x_px"),
            "phase6_y_px": phase6_result.get("y_px"),
            "phase6_cnn_probability": phase6_result.get("cnn_probability"),
            "phase6_cv_baseline_score": phase6_result.get("cv_baseline_score"),
            "phase6_fused_score": phase6_result.get("fused_score"),
            "phase6_inference_time_ms": phase6_result.get("inference_time_ms"),
            "candidates_json": json.dumps(json_ready(candidates)),
        }
    )
    row.update({key: tracking_result.get(key) for key in sorted(TRACKING_OUTPUT_KEYS)})
    row["total_processing_time_ms"] = float(total_processing_time_ms)
    row["measurement_error_px"] = euclidean_error(row.get("measured_x_px"), row.get("measured_y_px"), gt)
    row["filtered_error_px"] = euclidean_error(row.get("filtered_x_px"), row.get("filtered_y_px"), gt)
    row["_candidate_list"] = candidates
    row["failure_reason"] = determine_failure_reason(row, ground_truth_available, match_tolerance_px)
    return row


def unavailable_metrics(metric_names: Sequence[str], reason: str) -> Dict[str, object]:
    """Return explicit null metrics with a reason."""
    result = {name: None for name in metric_names}
    result["available"] = False
    result["reason"] = reason
    return result


def calculate_detection_metrics(rows: Sequence[Mapping[str, object]], ground_truth_available: bool, match_tolerance_px: float) -> Dict[str, object]:
    """Calculate detection metrics against Unity ground truth."""
    metric_names = [
        "visible_target_frames",
        "target_absent_frames",
        "candidate_recall",
        "accepted_detection_recall",
        "missed_detections",
        "false_positive_frames",
        "false_negative_frames",
        "target_found_rate",
    ]
    if not ground_truth_available:
        return unavailable_metrics(metric_names, "ground-truth labels are unavailable")

    visible_rows = [row for row in rows if bool_or_none(row.get("target_visible")) is True]
    absent_rows = [row for row in rows if bool_or_none(row.get("target_visible")) is False]
    candidate_hits = 0
    accepted_hits = 0
    false_negatives = 0

    for row in visible_rows:
        gt = ground_truth_point(row)
        candidates = row.get("_candidate_list", [])
        if gt is not None and isinstance(candidates, list) and candidate_hits_ground_truth(candidates, gt, match_tolerance_px):
            candidate_hits += 1
        if selected_hits_ground_truth(row, match_tolerance_px):
            accepted_hits += 1
        else:
            false_negatives += 1

    total_frames = len(rows)
    target_found_count = sum(1 for row in rows if bool(row.get("phase6_target_found")))
    false_positive_frames = sum(1 for row in absent_rows if bool(row.get("phase6_target_found")))
    visible_count = len(visible_rows)
    return {
        "available": True,
        "visible_target_frames": visible_count,
        "target_absent_frames": len(absent_rows),
        "candidate_recall": rounded_ratio(candidate_hits, visible_count),
        "accepted_detection_recall": rounded_ratio(accepted_hits, visible_count),
        "missed_detections": visible_count - accepted_hits,
        "false_positive_frames": false_positive_frames,
        "false_negative_frames": false_negatives,
        "target_found_rate": rounded_ratio(target_found_count, total_frames),
        "match_tolerance_px": match_tolerance_px,
    }


def calculate_coordinate_metrics(rows: Sequence[Mapping[str, object]], ground_truth_available: bool) -> Dict[str, object]:
    """Calculate coordinate errors against Unity ground truth."""
    metric_names = [
        "measurement_mae_px",
        "filtered_mae_px",
        "measurement_rmse_px",
        "filtered_rmse_px",
        "measurement_max_error_px",
        "filtered_max_error_px",
        "measurement_median_error_px",
        "filtered_median_error_px",
        "measurement_p95_error_px",
        "filtered_p95_error_px",
    ]
    if not ground_truth_available:
        return unavailable_metrics(metric_names, "ground-truth coordinates are unavailable")

    measurement_errors = numeric_values(row.get("measurement_error_px") for row in rows)
    filtered_errors = numeric_values(row.get("filtered_error_px") for row in rows)
    return {
        "available": True,
        "measurement_mae_px": mean_or_none(measurement_errors),
        "filtered_mae_px": mean_or_none(filtered_errors),
        "measurement_rmse_px": rmse_or_none(measurement_errors),
        "filtered_rmse_px": rmse_or_none(filtered_errors),
        "measurement_max_error_px": max_or_none(measurement_errors),
        "filtered_max_error_px": max_or_none(filtered_errors),
        "measurement_median_error_px": percentile_or_none(measurement_errors, 50),
        "filtered_median_error_px": percentile_or_none(filtered_errors, 50),
        "measurement_p95_error_px": percentile_or_none(measurement_errors, 95),
        "filtered_p95_error_px": percentile_or_none(filtered_errors, 95),
        "measurement_count": len(measurement_errors),
        "filtered_count": len(filtered_errors),
    }


def calculate_tracking_metrics(rows: Sequence[Mapping[str, object]], fps: float) -> Dict[str, object]:
    """Calculate lock-state and temporal tracking metrics."""
    total = len(rows)
    states = [str(row.get("lock_state")) for row in rows]
    first_lock_index = next((int(row["frame_index"]) for row in rows if row.get("lock_state") == "LOCKED"), None)
    lock_events = 0
    previous_state = None
    for state in states:
        if state == "LOCKED" and previous_state != "LOCKED":
            lock_events += 1
        previous_state = state
    lost_lock_events = sum(1 for row in rows if bool(row.get("just_lost")))
    return {
        "available": True,
        "time_to_first_lock_frames": first_lock_index,
        "time_to_first_lock_seconds": round(first_lock_index / fps, 6) if first_lock_index is not None and fps > 0 else None,
        "locked_frame_percentage": state_percentage(states, "LOCKED"),
        "acquiring_frame_percentage": state_percentage(states, "ACQUIRING"),
        "coasting_frame_percentage": state_percentage(states, "COASTING"),
        "lost_frame_percentage": state_percentage(states, "LOST"),
        "searching_frame_percentage": state_percentage(states, "SEARCHING"),
        "number_of_lock_events": lock_events,
        "number_of_lost_lock_events": lost_lock_events,
        "number_of_reacquisitions": sum(1 for row in rows if bool(row.get("just_reacquired"))),
        "maximum_consecutive_missed_frames": max((int(row.get("missed_frames") or 0) for row in rows), default=0),
    }


def calculate_smoothness_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Compare frame-to-frame raw and filtered movement jitter."""
    measured_points = ordered_points(rows, "measured_x_px", "measured_y_px")
    filtered_points = ordered_points(rows, "filtered_x_px", "filtered_y_px")
    raw_jitter = movement_jitter(measured_points)
    filtered_jitter = movement_jitter(filtered_points)
    reduction = None
    if raw_jitter is not None and raw_jitter > 0 and filtered_jitter is not None:
        reduction = round(100.0 * (raw_jitter - filtered_jitter) / raw_jitter, 6)
    return {
        "available": raw_jitter is not None or filtered_jitter is not None,
        "raw_measurement_jitter": raw_jitter,
        "filtered_position_jitter": filtered_jitter,
        "jitter_reduction_percentage": reduction,
    }


def calculate_performance_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Calculate processing-speed metrics."""
    times = numeric_values(row.get("total_processing_time_ms") for row in rows)
    mean_time = mean_or_none(times)
    return {
        "available": bool(times),
        "mean_processing_time_ms": mean_time,
        "median_processing_time_ms": percentile_or_none(times, 50),
        "p95_processing_time_ms": percentile_or_none(times, 95),
        "max_processing_time_ms": max_or_none(times),
        "effective_processing_fps": round(1000.0 / mean_time, 6) if mean_time and mean_time > 0 else None,
        "frame_count": len(rows),
    }


def calculate_domain_gap_report(
    rows: Sequence[Mapping[str, object]],
    confidence_threshold: float,
    ground_truth_available: bool,
) -> Dict[str, object]:
    """Summarize baseline CNN behavior on Unity frames."""
    candidate_probs = []
    selected_probs = []
    for row in rows:
        candidates = row.get("_candidate_list", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                prob = float_or_none(candidate.get("correct_probability"))
                if prob is not None:
                    candidate_probs.append(prob)
        selected_prob = float_or_none(row.get("phase6_cnn_probability"))
        if selected_prob is not None:
            selected_probs.append(selected_prob)

    failure_counts = Counter(str(row.get("failure_reason")) for row in rows)
    common_failure = most_common_failure(failure_counts)
    candidate_counts = [int(row.get("candidate_count") or 0) for row in rows]
    return {
        "ground_truth_available": ground_truth_available,
        "average_correct_beacon_probability": mean_or_none(candidate_probs),
        "selected_average_correct_beacon_probability": mean_or_none(selected_probs),
        "minimum_probability": min_or_none(candidate_probs),
        "maximum_probability": max_or_none(candidate_probs),
        "percentage_candidates_above_confidence_threshold": rounded_ratio(
            sum(1 for value in candidate_probs if value >= confidence_threshold),
            len(candidate_probs),
        ),
        "average_number_of_candidates": mean_or_none(candidate_counts),
        "frames_with_zero_candidates": sum(1 for value in candidate_counts if value == 0),
        "frames_with_multiple_candidates": sum(1 for value in candidate_counts if value > 1),
        "most_common_failure_reason": common_failure,
        "failure_reason_counts": dict(sorted(failure_counts.items())),
    }


def most_common_failure(failure_counts: Counter[str]) -> Optional[str]:
    """Return the most common non-ok reason, or ok if no failure is present."""
    for reason, _ in failure_counts.most_common():
        if reason not in {"ok", "unlabelled_detection"}:
            return reason
    return failure_counts.most_common(1)[0][0] if failure_counts else None


def rounded_ratio(numerator: int, denominator: int) -> Optional[float]:
    """Return a rounded ratio or None."""
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def numeric_values(values: Iterable[object]) -> List[float]:
    """Return finite numeric values from an iterable."""
    numbers = []
    for value in values:
        number = float_or_none(value)
        if number is not None and math.isfinite(number):
            numbers.append(number)
    return numbers


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    """Return rounded mean or None."""
    return round(float(np.mean(values)), 6) if values else None


def rmse_or_none(values: Sequence[float]) -> Optional[float]:
    """Return rounded RMSE or None."""
    return round(float(np.sqrt(np.mean(np.square(values)))), 6) if values else None


def percentile_or_none(values: Sequence[float], percentile: float) -> Optional[float]:
    """Return rounded percentile or None."""
    return round(float(np.percentile(values, percentile)), 6) if values else None


def max_or_none(values: Sequence[float]) -> Optional[float]:
    """Return rounded maximum or None."""
    return round(float(max(values)), 6) if values else None


def min_or_none(values: Sequence[float]) -> Optional[float]:
    """Return rounded minimum or None."""
    return round(float(min(values)), 6) if values else None


def state_percentage(states: Sequence[str], state: str) -> Optional[float]:
    """Return state occupancy percentage."""
    if not states:
        return None
    return round(100.0 * sum(1 for item in states if item == state) / len(states), 6)


def ordered_points(rows: Sequence[Mapping[str, object]], x_key: str, y_key: str) -> List[Tuple[int, float, float]]:
    """Return frame-indexed points with valid coordinates."""
    points = []
    for row in rows:
        x_value = float_or_none(row.get(x_key))
        y_value = float_or_none(row.get(y_key))
        if x_value is not None and y_value is not None:
            points.append((int(row.get("frame_index") or len(points)), x_value, y_value))
    return points


def movement_jitter(points: Sequence[Tuple[int, float, float]]) -> Optional[float]:
    """Use standard deviation of consecutive step lengths as a simple jitter metric."""
    if len(points) < 3:
        return None
    steps = []
    previous = points[0]
    for current in points[1:]:
        if current[0] == previous[0] + 1:
            steps.append(math.hypot(current[1] - previous[1], current[2] - previous[2]))
        previous = current
    if len(steps) < 2:
        return None
    return round(float(np.std(steps)), 6)


def draw_text(image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.48) -> None:
    """Draw readable text with a dark outline."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_translucent_panel(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.68,
) -> None:
    """Draw a readable panel without fully hiding the frame."""
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def point_inside_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int], padding: int = 24) -> bool:
    """Return whether a point would be covered by a text panel."""
    x_min, y_min, x_max, y_max = rect
    return x_min - padding <= point[0] <= x_max + padding and y_min - padding <= point[1] <= y_max + padding


def rects_overlap(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int], padding: int = 8) -> bool:
    """Return whether two overlay panels overlap."""
    return not (
        first[2] + padding < second[0]
        or second[2] + padding < first[0]
        or first[3] + padding < second[1]
        or second[3] + padding < first[1]
    )


def choose_panel_rect(
    image_shape: Tuple[int, ...],
    avoid_points: Sequence[Tuple[float, float]],
    panel_size: Tuple[int, int] = (318, 150),
) -> Tuple[int, int, int, int]:
    """Choose a corner panel that avoids the target/track points when possible."""
    height, width = image_shape[:2]
    panel_width, panel_height = panel_size
    margin = 8
    candidates = [
        (margin, margin, margin + panel_width, margin + panel_height),
        (width - panel_width - margin, margin, width - margin, margin + panel_height),
        (margin, height - panel_height - margin, margin + panel_width, height - margin),
        (width - panel_width - margin, height - panel_height - margin, width - margin, height - margin),
    ]

    best_rect = candidates[0]
    best_overlap = None
    for rect in candidates:
        overlap_count = sum(1 for point in avoid_points if point_inside_rect(point, rect))
        if best_overlap is None or overlap_count < best_overlap:
            best_overlap = overlap_count
            best_rect = rect
        if overlap_count == 0:
            return rect
    return best_rect


def draw_optional_point(image: np.ndarray, x_value: object, y_value: object, color: Tuple[int, int, int], label: str) -> None:
    """Draw a labelled point when coordinates are available."""
    x_number = float_or_none(x_value)
    y_number = float_or_none(y_value)
    if x_number is None or y_number is None:
        return
    point = (int(round(x_number)), int(round(y_number)))
    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    draw_text(image, label, (point[0] + 8, max(16, point[1] - 8)), color, scale=0.45)


def draw_unity_evaluation_overlay(
    frame: np.ndarray,
    phase6_result: Mapping[str, object],
    tracking_result: Mapping[str, object],
    manifest_row: Mapping[str, object],
    total_processing_time_ms: float,
    selected_trajectory: Sequence[Tuple[float, float]],
    filtered_trajectory: Sequence[Tuple[float, float]],
    ground_truth_trajectory: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Draw candidate, ground-truth and tracking diagnostics for the output video."""
    overlay = frame.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for candidate in phase6_result.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        point = (int(round(float(candidate["x_px"]))), int(round(float(candidate["y_px"]))))
        color = (150, 150, 150)
        radius = 3
        thickness = 1
        if phase6_result.get("selected_candidate_id") is not None and int(candidate["candidate_id"]) == int(phase6_result["selected_candidate_id"]):
            color = (0, 255, 255)
            radius = 9
            thickness = 2
        cv2.circle(overlay, point, radius, color, thickness, lineType=cv2.LINE_AA)

    draw_polyline(overlay, selected_trajectory, (0, 255, 255), thickness=3)
    draw_polyline(overlay, filtered_trajectory, (255, 80, 0), thickness=2)
    draw_polyline(overlay, ground_truth_trajectory, (0, 0, 255), thickness=2)

    gt = ground_truth_point(manifest_row)
    if gt is not None:
        gt_point = (int(round(gt[0])), int(round(gt[1])))
        cv2.circle(overlay, gt_point, 10, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        draw_text(overlay, "GT", (gt_point[0] + 10, max(16, gt_point[1] - 10)), (0, 0, 255), scale=0.45)

    draw_optional_point(overlay, tracking_result.get("measured_x_px"), tracking_result.get("measured_y_px"), (0, 255, 0), "M")
    draw_optional_point(overlay, tracking_result.get("filtered_x_px"), tracking_result.get("filtered_y_px"), (255, 0, 0), "F")
    draw_optional_point(overlay, tracking_result.get("predicted_x_px"), tracking_result.get("predicted_y_px"), (255, 0, 255), "P")

    avoid_points = []
    for x_key, y_key in [
        ("target_x", "target_y"),
        ("measured_x_px", "measured_y_px"),
        ("filtered_x_px", "filtered_y_px"),
        ("predicted_x_px", "predicted_y_px"),
    ]:
        x_value = float_or_none(manifest_row.get(x_key, tracking_result.get(x_key)))
        y_value = float_or_none(manifest_row.get(y_key, tracking_result.get(y_key)))
        if x_value is not None and y_value is not None:
            avoid_points.append((x_value, y_value))

    panel = choose_panel_rect(overlay.shape, avoid_points)
    draw_translucent_panel(overlay, (panel[0], panel[1]), (panel[2], panel[3]))
    info_lines = [
        f"Frame: {manifest_row.get('frame_index')}",
        f"Candidates: {phase6_result.get('candidate_count')}",
        f"Selected: {phase6_result.get('selected_candidate_id')}",
        f"CNN conf: {float(phase6_result.get('confidence') or 0.0):.2f}",
        f"Lock: {tracking_result.get('lock_state')}",
        f"Missed: {tracking_result.get('missed_frames')}  Pred-only: {tracking_result.get('using_prediction_only')}",
        f"Time: {total_processing_time_ms:.1f} ms",
    ]
    for index, line in enumerate(info_lines):
        draw_text(overlay, line, (panel[0] + 8, panel[1] + 22 + index * 19), (255, 255, 255), scale=0.46)

    legend_points = avoid_points + list(ground_truth_trajectory[-20:]) + list(filtered_trajectory[-20:]) + list(selected_trajectory[-20:])
    legend_rect = choose_legend_rect(overlay.shape, legend_points, occupied_rects=[panel])
    draw_translucent_panel(overlay, (legend_rect[0], legend_rect[1]), (legend_rect[2], legend_rect[3]), alpha=0.68)
    draw_legend(overlay, legend_rect)
    return overlay


def draw_polyline(
    image: np.ndarray,
    points: Sequence[Tuple[float, float]],
    color: Tuple[int, int, int],
    thickness: int = 2,
    max_points: int = 90,
) -> None:
    """Draw a recent trajectory trail."""
    if len(points) < 2:
        return
    visible = points[-max_points:]
    array = np.array([(int(round(x)), int(round(y))) for x, y in visible], dtype=np.int32)
    cv2.polylines(image, [array], False, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.polylines(image, [array], False, color, thickness, lineType=cv2.LINE_AA)


def choose_legend_rect(
    image_shape: Tuple[int, ...],
    avoid_points: Sequence[Tuple[float, float]],
    occupied_rects: Sequence[Tuple[int, int, int, int]] = (),
) -> Tuple[int, int, int, int]:
    """Place the legend in the clearest corner."""
    height, width = image_shape[:2]
    legend_width = 220
    legend_height = 104
    margin = 8
    candidates = [
        (width - legend_width - margin, margin, width - margin, margin + legend_height),
        (margin, height - legend_height - margin, margin + legend_width, height - margin),
        (width - legend_width - margin, height - legend_height - margin, width - margin, height - margin),
        (margin, margin, margin + legend_width, margin + legend_height),
    ]

    best_rect = candidates[0]
    best_score = None
    for rect in candidates:
        point_overlap = sum(1 for point in avoid_points if point_inside_rect(point, rect, padding=12))
        panel_overlap = sum(1 for occupied in occupied_rects if rects_overlap(rect, occupied))
        score = point_overlap + panel_overlap * 10
        if best_score is None or score < best_score:
            best_score = score
            best_rect = rect
        if score == 0:
            return rect
    return best_rect


def draw_legend(image: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
    """Draw a compact legend for the annotated tracking video."""
    x_min, y_min, _, _ = rect
    rows = [
        ((0, 0, 255), "GT beacon/trail"),
        ((0, 255, 255), "selected candidate/trail"),
        ((255, 80, 0), "filtered track"),
        ((0, 255, 0), "M measured"),
        ((255, 0, 255), "P predicted"),
        ((150, 150, 150), "other bright spots"),
    ]
    for index, (color, label) in enumerate(rows):
        y = y_min + 18 + index * 15
        cv2.line(image, (x_min + 9, y - 4), (x_min + 28, y - 4), color, 2, lineType=cv2.LINE_AA)
        draw_text(image, label, (x_min + 34, y), (255, 255, 255), scale=0.34)


def save_confidence_plot(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Save confidence-over-time plot."""
    frames = [int(row.get("frame_index") or index) for index, row in enumerate(rows)]
    confidence = [float(row.get("confidence") or 0.0) for row in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(frames, confidence, color="#0b6bcb", linewidth=1.8)
    plt.xlabel("Frame")
    plt.ylabel("Tracker confidence")
    plt.title("Unity Sequence Confidence Over Time")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    save_current_plot(path)


def save_lock_state_timeline(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Save lock-state timeline plot."""
    state_order = ["SEARCHING", "ACQUIRING", "LOCKED", "COASTING", "LOST"]
    state_to_value = {state: index for index, state in enumerate(state_order)}
    frames = [int(row.get("frame_index") or index) for index, row in enumerate(rows)]
    values = [state_to_value.get(str(row.get("lock_state")), 0) for row in rows]
    plt.figure(figsize=(10, 3.8))
    plt.step(frames, values, where="post", color="#6f3cc3", linewidth=2)
    plt.yticks(list(state_to_value.values()), state_order)
    plt.xlabel("Frame")
    plt.title("Lock State Timeline")
    plt.grid(True, axis="x", alpha=0.25)
    save_current_plot(path)


def save_coordinate_error_plot(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Save coordinate error over time when ground truth exists."""
    frames = [int(row.get("frame_index") or index) for index, row in enumerate(rows)]
    measurement = [float_or_none(row.get("measurement_error_px")) for row in rows]
    filtered = [float_or_none(row.get("filtered_error_px")) for row in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(frames, [np.nan if value is None else value for value in measurement], label="Measured", color="#18864b")
    plt.plot(frames, [np.nan if value is None else value for value in filtered], label="Filtered", color="#0b6bcb")
    plt.xlabel("Frame")
    plt.ylabel("Error (px)")
    plt.title("Coordinate Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_plot(path)


def save_trajectory_plot(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Save ground-truth, measured and filtered trajectory comparison."""
    gt_points = []
    measured_points = []
    filtered_points = []
    for row in rows:
        gt = ground_truth_point(row)
        if gt is not None:
            gt_points.append(gt)
        measured_x = float_or_none(row.get("measured_x_px"))
        measured_y = float_or_none(row.get("measured_y_px"))
        if measured_x is not None and measured_y is not None:
            measured_points.append((measured_x, measured_y))
        filtered_x = float_or_none(row.get("filtered_x_px"))
        filtered_y = float_or_none(row.get("filtered_y_px"))
        if filtered_x is not None and filtered_y is not None:
            filtered_points.append((filtered_x, filtered_y))

    plt.figure(figsize=(7, 5))
    plot_points(gt_points, "Ground truth", "#d62728")
    plot_points(measured_points, "Measured", "#18864b")
    plot_points(filtered_points, "Filtered", "#0b6bcb")
    plt.gca().invert_yaxis()
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.title("Unity Trajectory Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_plot(path)


def plot_points(points: Sequence[Tuple[float, float]], label: str, color: str) -> None:
    """Plot xy points if present."""
    if not points:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.plot(xs, ys, label=label, color=color, linewidth=1.5)


def save_current_plot(path: Path) -> None:
    """Save the current matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def representative_indices(rows: Sequence[Mapping[str, object]], ground_truth_available: bool) -> List[int]:
    """Pick representative frames for a compact diagnostic montage."""
    chosen: List[int] = []

    def add(index: Optional[int]) -> None:
        if index is not None and index not in chosen and 0 <= index < len(rows):
            chosen.append(index)

    for index, row in enumerate(rows):
        if row.get("failure_reason") == "ok" and float(row.get("phase6_cnn_probability") or 0.0) >= 0.8:
            add(index)
            break
    for index, row in enumerate(rows):
        if row.get("failure_reason") == "ok" and float(row.get("phase6_cnn_probability") or 0.0) < 0.8:
            add(index)
            break
    for index, row in enumerate(rows):
        if "missed" in str(row.get("failure_reason")) or row.get("phase6_status") == "no_candidates":
            add(index)
            break
    for index, row in enumerate(rows):
        if "false_detection" in str(row.get("failure_reason")):
            add(index)
            break
    for index, row in enumerate(rows):
        if row.get("lock_state") in {"COASTING", "LOST"} or bool(row.get("just_lost")):
            add(index)
            break

    if ground_truth_available:
        ranked = sorted(
            ((index, float(row.get("filtered_error_px") or -1.0)) for index, row in enumerate(rows)),
            key=lambda item: item[1],
            reverse=True,
        )
        for index, error in ranked[:4]:
            if error >= 0:
                add(index)

    if not chosen:
        for index in np.linspace(0, max(0, len(rows) - 1), min(8, len(rows)), dtype=int):
            add(int(index))
    return chosen[:12]


def save_failure_montage(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Save a small montage of representative frames."""
    indices = representative_indices(rows, any(ground_truth_point(row) is not None for row in rows))
    if not indices:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_text(blank, "No frames available", (40, 120), (255, 255, 255), scale=0.6)
        cv2.imwrite(str(output_path), blank)
        return

    thumbnails = []
    for index in indices:
        row = rows[index]
        frame = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        phase6_result = {
            "candidates": row.get("_candidate_list", []),
            "selected_candidate_id": row.get("phase6_selected_candidate_id"),
            "candidate_count": row.get("candidate_count"),
            "confidence": row.get("phase6_cnn_probability"),
        }
        tracking_result = {key: row.get(key) for key in TRACKING_OUTPUT_KEYS}
        overlay = draw_unity_evaluation_overlay(
            frame,
            phase6_result,
            tracking_result,
            row,
            float(row.get("total_processing_time_ms") or 0.0),
            selected_trajectory=[],
            filtered_trajectory=[],
            ground_truth_trajectory=[],
        )
        label = f"F{row.get('frame_index')} {row.get('failure_reason')}"
        draw_text(overlay, label, (14, 180), (0, 255, 255), scale=0.46)
        thumbnails.append(cv2.resize(overlay, (320, 240), interpolation=cv2.INTER_AREA))

    if not thumbnails:
        return
    columns = 3
    rows_count = int(math.ceil(len(thumbnails) / columns))
    montage = np.zeros((rows_count * 240, columns * 320, 3), dtype=np.uint8)
    for idx, thumbnail in enumerate(thumbnails):
        y = (idx // columns) * 240
        x = (idx % columns) * 320
        montage[y : y + 240, x : x + 320] = thumbnail
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), montage):
        raise OSError(f"failed to write failure montage: {output_path}")


def evaluate_unity_sequence(
    sequence_dir: str | Path = "data/raw/unity",
    config_path: str | Path = "configs/default.yaml",
    checkpoint: Optional[str | Path] = None,
    fps: float = 30.0,
    coordinate_origin: str = "top-left",
    output_dir: str | Path = "outputs/unity-evaluation/smooth_horizontal_01",
    labels_path: Optional[str | Path] = None,
    expected_count: Optional[int] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    match_tolerance_px: float = 12.0,
    output_scale: int = 1,
    label_frame_offset: int = 0,
    device: Optional[str] = None,
    pipeline: Optional[object] = None,
    tracker: Optional[object] = None,
    pipeline_factory: Optional[PipelineFactory] = None,
    tracker_factory: Optional[TrackerFactory] = None,
) -> Dict[str, object]:
    """Run offline Unity-sequence evaluation with Phase 6 and Phase 7."""
    validate_coordinate_origin(coordinate_origin)
    output_scale = validate_output_scale(output_scale)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inventory = discover_unity_sequence(sequence_dir)
    print_inventory(inventory)
    frame_width, frame_height = resolved_frame_size(inventory, expected_width, expected_height)
    validation = validate_unity_sequence(inventory, expected_count, expected_width, expected_height)
    chosen_labels_path = Path(labels_path) if labels_path else inventory.labels_path
    labels = []
    if chosen_labels_path is not None and chosen_labels_path.exists():
        labels = normalize_labels(
            chosen_labels_path,
            inventory.sequence_dir,
            coordinate_origin=coordinate_origin,
            default_width=frame_width,
            default_height=frame_height,
            fps=fps,
            frame_index_offset=label_frame_offset,
        )
        if label_frame_offset:
            valid_frame_numbers = {frame.frame_index for frame in inventory.frames}
            labels = [row for row in labels if int(row["frame_index"]) in valid_frame_numbers]
    missing_references = missing_label_image_references(labels)
    label_validation = validate_unity_labels(labels, inventory, require_all_frames=label_frame_offset == 0)
    validation["label_image_references_valid"] = not missing_references
    validation["missing_label_image_references"] = missing_references
    validation["label_validation"] = label_validation
    write_json(output_path / "dataset_validation.json", {"inventory": inventory.to_dict(), "validation": validation})
    if not validation["usable"]:
        raise ValueError(f"Unity sequence failed validation; see {output_path / 'dataset_validation.json'}")
    if missing_references:
        raise ValueError(f"labels reference missing images; see {output_path / 'dataset_validation.json'}")
    if not label_validation["usable"]:
        raise ValueError(f"labels failed validation; see {output_path / 'dataset_validation.json'}")

    ground_truth_available = has_ground_truth(labels)
    if not ground_truth_available:
        write_required_labels_template(inventory, output_path / "required_labels_template.csv", fps=fps)
        print("Images are usable for inference, but quantitative tracking validation requires Unity ground-truth coordinates.")

    slug = sequence_slug(inventory, labels)
    processed_manifest_path = Path("data/processed/unity") / slug / "manifest.csv"
    manifest_rows = build_and_write_manifest(inventory, labels, processed_manifest_path, fps=fps)
    output_manifest_path = output_path / "normalized_manifest.csv"
    build_and_write_manifest(inventory, labels, output_manifest_path, fps=fps)

    if pipeline is None:
        pipeline = pipeline_factory() if pipeline_factory else make_default_pipeline(config_path, checkpoint, device)
    if tracker is None:
        tracker = tracker_factory() if tracker_factory else make_default_tracker(config_path)
    if hasattr(tracker, "reset"):
        tracker.reset()

    writer = cv2.VideoWriter(
        str(output_path / "annotated_tracking.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_width * output_scale, frame_height * output_scale),
    )
    if not writer.isOpened():
        raise OSError(f"could not create annotated video in {output_path}")

    prediction_rows: List[Dict[str, object]] = []
    selected_trajectory: List[Tuple[float, float]] = []
    filtered_trajectory: List[Tuple[float, float]] = []
    ground_truth_trajectory: List[Tuple[float, float]] = []
    try:
        for manifest_row in manifest_rows:
            frame = cv2.imread(str(manifest_row["image_path"]), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise ValueError(f"could not read frame listed in manifest: {manifest_row['image_path']}")
            frame_start = time.perf_counter()
            phase6_result = pipeline.run(frame)
            tracking_result = tracker.update(phase6_result, frame_index=int(manifest_row["frame_index"]))
            total_ms = round((time.perf_counter() - frame_start) * 1000.0, 3)
            prediction_row = build_prediction_row(
                manifest_row,
                phase6_result,
                tracking_result,
                total_processing_time_ms=total_ms,
                ground_truth_available=ground_truth_available,
                match_tolerance_px=match_tolerance_px,
            )
            prediction_rows.append(prediction_row)

            measured_x = float_or_none(tracking_result.get("measured_x_px"))
            measured_y = float_or_none(tracking_result.get("measured_y_px"))
            if measured_x is not None and measured_y is not None:
                selected_trajectory.append((measured_x, measured_y))
            filtered_x = float_or_none(tracking_result.get("filtered_x_px"))
            filtered_y = float_or_none(tracking_result.get("filtered_y_px"))
            if filtered_x is not None and filtered_y is not None:
                filtered_trajectory.append((filtered_x, filtered_y))
            gt = ground_truth_point(manifest_row)
            if gt is not None:
                ground_truth_trajectory.append(gt)
            overlay = draw_unity_evaluation_overlay(
                frame,
                phase6_result,
                tracking_result,
                manifest_row,
                total_ms,
                selected_trajectory=selected_trajectory,
                filtered_trajectory=filtered_trajectory,
                ground_truth_trajectory=ground_truth_trajectory,
            )
            writer.write(scaled_video_frame(overlay, output_scale))
    finally:
        writer.release()

    frame_predictions_path = output_path / "frame_predictions.csv"
    write_csv(frame_predictions_path, prediction_rows, FRAME_PREDICTION_COLUMNS)

    detection_metrics = calculate_detection_metrics(prediction_rows, ground_truth_available, match_tolerance_px)
    coordinate_metrics = calculate_coordinate_metrics(prediction_rows, ground_truth_available)
    tracking_metrics = calculate_tracking_metrics(prediction_rows, fps=fps)
    smoothness_metrics = calculate_smoothness_metrics(prediction_rows)
    performance_metrics = calculate_performance_metrics(prediction_rows)
    confidence_threshold = float(getattr(getattr(pipeline, "inference_config", object()), "confidence_threshold", 0.55))
    domain_gap_report = calculate_domain_gap_report(prediction_rows, confidence_threshold, ground_truth_available)
    tracking_metrics["smoothness"] = smoothness_metrics
    tracking_metrics["coordinate_metrics"] = coordinate_metrics

    write_json(output_path / "detection_metrics.json", detection_metrics)
    write_json(output_path / "tracking_metrics.json", tracking_metrics)
    write_json(output_path / "performance_metrics.json", performance_metrics)
    write_json(output_path / "domain_gap_report.json", domain_gap_report)

    save_confidence_plot(prediction_rows, output_path / "confidence_over_time.png")
    save_lock_state_timeline(prediction_rows, output_path / "lock_state_timeline.png")
    if ground_truth_available:
        save_trajectory_plot(prediction_rows, output_path / "trajectory_comparison.png")
        save_coordinate_error_plot(prediction_rows, output_path / "coordinate_error_over_time.png")
    save_failure_montage(prediction_rows, output_path / "failure_montage.png")

    evaluation_summary = {
        "sequence_dir": str(inventory.sequence_dir),
        "image_count": inventory.image_count,
        "image_extension": inventory.image_extension,
        "resolution": f"{frame_width}x{frame_height}",
        "output_video_resolution": f"{frame_width * output_scale}x{frame_height * output_scale}",
        "output_scale": output_scale,
        "label_frame_offset": label_frame_offset,
        "labels_found": str(chosen_labels_path) if chosen_labels_path else None,
        "ground_truth_available": ground_truth_available,
        "dataset_usable": bool(validation["usable"]),
        "candidate_recall": detection_metrics.get("candidate_recall"),
        "accepted_detection_recall": detection_metrics.get("accepted_detection_recall"),
        "filtered_mae_px": coordinate_metrics.get("filtered_mae_px"),
        "time_to_first_lock_frames": tracking_metrics.get("time_to_first_lock_frames"),
        "locked_frame_percentage": tracking_metrics.get("locked_frame_percentage"),
        "mean_processing_time_ms": performance_metrics.get("mean_processing_time_ms"),
        "effective_processing_fps": performance_metrics.get("effective_processing_fps"),
        "most_common_failure_reason": domain_gap_report.get("most_common_failure_reason"),
        "outputs_dir": str(output_path),
    }
    write_json(output_path / "evaluation_summary.json", evaluation_summary)

    result = {
        "inventory": inventory.to_dict(),
        "dataset_validation": validation,
        "ground_truth_available": ground_truth_available,
        "manifest_path": str(processed_manifest_path),
        "output_manifest_path": str(output_manifest_path),
        "frame_predictions_path": str(frame_predictions_path),
        "detection_metrics": detection_metrics,
        "tracking_metrics": tracking_metrics,
        "performance_metrics": performance_metrics,
        "domain_gap_report": domain_gap_report,
        "evaluation_summary": evaluation_summary,
        "output_dir": str(output_path),
    }
    if set(result) != EVALUATION_RESULT_KEYS:
        raise ValueError("evaluation result schema changed unexpectedly")
    return result


def summarize_batch_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Summarize a multi-sequence Unity evaluation run."""
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    return {
        "sequence_count": len(rows),
        "successful_sequences": len(successful),
        "failed_sequences": len(failed),
        "average_candidate_recall": mean_or_none(numeric_values(row.get("candidate_recall") for row in successful)),
        "average_accepted_detection_recall": mean_or_none(numeric_values(row.get("accepted_detection_recall") for row in successful)),
        "average_filtered_mae_px": mean_or_none(numeric_values(row.get("filtered_mae_px") for row in successful)),
        "average_locked_frame_percentage": mean_or_none(numeric_values(row.get("locked_frame_percentage") for row in successful)),
        "average_effective_processing_fps": mean_or_none(numeric_values(row.get("effective_processing_fps") for row in successful)),
        "failed_sequence_dirs": [row.get("sequence_dir") for row in failed],
    }


def evaluate_unity_sequences(
    root_dir: str | Path = "data/raw/unity",
    config_path: str | Path = "configs/unity.yaml",
    checkpoint: Optional[str | Path] = None,
    fps: float = 30.0,
    coordinate_origin: str = "top-left",
    output_root: str | Path = "outputs/unity-evaluation",
    expected_count: Optional[int] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    match_tolerance_px: float = 12.0,
    output_scale: int = 1,
    label_frame_offset: int = 0,
    device: Optional[str] = None,
    pipeline: Optional[object] = None,
    tracker_factory: Optional[TrackerFactory] = None,
) -> Dict[str, object]:
    """Evaluate every Unity image sequence under a root folder."""
    validate_coordinate_origin(coordinate_origin)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    inventories = discover_unity_sequences(root_dir)
    shared_pipeline = pipeline or make_default_pipeline(config_path, checkpoint, device)
    rows: List[Dict[str, object]] = []

    for inventory in inventories:
        sequence_output = batch_output_folder(output_path, inventory)
        try:
            result = evaluate_unity_sequence(
                sequence_dir=inventory.sequence_dir,
                config_path=config_path,
                checkpoint=checkpoint,
                fps=fps,
                coordinate_origin=coordinate_origin,
                output_dir=sequence_output,
                labels_path=None,
                expected_count=expected_count,
                expected_width=expected_width,
                expected_height=expected_height,
                match_tolerance_px=match_tolerance_px,
                output_scale=output_scale,
                label_frame_offset=label_frame_offset,
                device=device,
                pipeline=shared_pipeline,
                tracker_factory=tracker_factory,
            )
            summary = dict(result["evaluation_summary"])
            summary["status"] = "ok"
            summary["error"] = ""
        except Exception as exc:  # pragma: no cover - exercised through CLI/manual use
            summary = {
                "sequence_dir": str(inventory.sequence_dir),
                "outputs_dir": str(sequence_output),
                "status": "failed",
                "image_count": inventory.image_count,
                "resolution": f"{inventory.width}x{inventory.height}",
                "labels_found": str(inventory.labels_path) if inventory.labels_path else None,
                "ground_truth_available": None,
                "candidate_recall": None,
                "accepted_detection_recall": None,
                "filtered_mae_px": None,
                "locked_frame_percentage": None,
                "mean_processing_time_ms": None,
                "effective_processing_fps": None,
                "most_common_failure_reason": None,
                "error": str(exc),
            }
        rows.append({column: summary.get(column, "") for column in BATCH_METRICS_COLUMNS})

    batch_summary = summarize_batch_rows(rows)
    write_csv(output_path / "final_metrics.csv", rows, BATCH_METRICS_COLUMNS)
    write_json(
        output_path / "final_summary.json",
        {
            "root_dir": str(root_dir),
            "output_root": str(output_path),
            "fps": fps,
            "coordinate_origin": coordinate_origin,
            "expected_count": expected_count,
            "expected_width": expected_width,
            "expected_height": expected_height,
            "output_scale": output_scale,
            "label_frame_offset": label_frame_offset,
            "summary": batch_summary,
            "sequences": rows,
        },
    )
    return {
        "summary": batch_summary,
        "rows": rows,
        "final_metrics_path": str(output_path / "final_metrics.csv"),
        "final_summary_path": str(output_path / "final_summary.json"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for Unity sequence evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a Unity FSOC image sequence offline.")
    parser.add_argument("--sequence-dir", default="data/raw/unity", help="Unity root or sequence folder.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration path.")
    parser.add_argument("--checkpoint", default=None, help="Optional classifier checkpoint override.")
    parser.add_argument("--labels", default=None, help="Optional explicit labels.csv/.xlsx path.")
    parser.add_argument("--fps", type=float, required=True, help="Capture FPS. Use 30 for the current Unity export.")
    parser.add_argument("--coordinate-origin", choices=["top-left", "bottom-left"], required=True)
    parser.add_argument("--output", default="outputs/unity-evaluation/smooth_horizontal_01")
    parser.add_argument("--expected-count", type=int, default=None, help="Optional expected frame count.")
    parser.add_argument("--expected-width", type=int, default=None, help="Optional expected width; omitted means auto-detect.")
    parser.add_argument("--expected-height", type=int, default=None, help="Optional expected height; omitted means auto-detect.")
    parser.add_argument("--match-tolerance", type=float, default=12.0)
    parser.add_argument("--output-scale", type=int, default=1, help="Scale only the annotated MP4 for easier presentation viewing.")
    parser.add_argument("--label-frame-offset", type=int, default=0, help="Shift label frame IDs when Unity labels are offset from exported images.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument("--batch", action="store_true", help="Evaluate every sequence discovered under --sequence-dir.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    if args.batch:
        if args.labels is not None:
            raise ValueError("--labels is only supported for single-sequence evaluation")
        result = evaluate_unity_sequences(
            root_dir=args.sequence_dir,
            config_path=args.config,
            checkpoint=args.checkpoint,
            fps=args.fps,
            coordinate_origin=args.coordinate_origin,
            output_root=args.output,
            expected_count=args.expected_count,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            match_tolerance_px=args.match_tolerance,
            output_scale=args.output_scale,
            label_frame_offset=args.label_frame_offset,
            device=args.device,
        )
        summary = result["summary"]
        print("Phase 8 Unity batch evaluation")
        print(f"  Sequences: {summary['sequence_count']}")
        print(f"  Successful: {summary['successful_sequences']}")
        print(f"  Failed: {summary['failed_sequences']}")
        print(f"  Average candidate recall: {summary['average_candidate_recall']}")
        print(f"  Average accepted recall: {summary['average_accepted_detection_recall']}")
        print(f"  Average filtered MAE: {summary['average_filtered_mae_px']}")
        print(f"  Final metrics: {result['final_metrics_path']}")
        print(f"  Final summary: {result['final_summary_path']}")
        return

    result = evaluate_unity_sequence(
        sequence_dir=args.sequence_dir,
        config_path=args.config,
        checkpoint=args.checkpoint,
        fps=args.fps,
        coordinate_origin=args.coordinate_origin,
        output_dir=args.output,
        labels_path=args.labels,
        expected_count=args.expected_count,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        match_tolerance_px=args.match_tolerance,
        output_scale=args.output_scale,
        label_frame_offset=args.label_frame_offset,
        device=args.device,
    )

    summary = result["evaluation_summary"]
    print("Phase 8 Unity sequence evaluation")
    print(f"  Sequence: {summary['sequence_dir']}")
    print(f"  Images: {summary['image_count']} {result['inventory']['image_extension']}")
    print(f"  Labels: {summary['labels_found'] if summary['labels_found'] else 'not found'}")
    print(f"  Ground truth available: {summary['ground_truth_available']}")
    print(f"  Candidate recall: {summary['candidate_recall']}")
    print(f"  Accepted recall: {summary['accepted_detection_recall']}")
    print(f"  Filtered MAE: {summary['filtered_mae_px']}")
    print(f"  Locked frames: {summary['locked_frame_percentage']}%")
    print(f"  Mean processing: {summary['mean_processing_time_ms']} ms")
    print(f"  Effective FPS: {summary['effective_processing_fps']}")
    print(f"  Outputs: {summary['outputs_dir']}")


if __name__ == "__main__":
    main()
