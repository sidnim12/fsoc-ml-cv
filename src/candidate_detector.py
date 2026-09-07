from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .preprocessing import PreprocessingConfig, convert_to_grayscale, load_image, preprocess_frame
except ImportError:  # pragma: no cover - allows direct script execution
    from preprocessing import PreprocessingConfig, convert_to_grayscale, load_image, preprocess_frame


@dataclass(frozen=True)
class BeaconCandidate:
    """Measurable information for one possible bright beacon."""

    candidate_id: int
    x: float
    y: float
    area: float
    radius: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    mean_intensity: float
    max_intensity: int
    circularity: float
    distance_from_center: float
    baseline_score: float = 0.0

    def to_dict(self) -> Dict[str, float | int]:
        """Convert the candidate to a CSV/JSON friendly dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DetectorConfig:
    """Filtering and evaluation settings for candidate detection."""

    min_area: float = 3.0
    max_area: float = 1000.0
    min_radius: float = 1.0
    max_radius: float = 25.0
    min_circularity: float = 0.15
    match_tolerance: float = 12.0


def find_contours(binary_image: np.ndarray) -> List[np.ndarray]:
    """Find external contours in a binary image."""
    if binary_image.ndim != 2:
        raise ValueError("find_contours expects a one-channel binary image")
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def calculate_centroid(contour: np.ndarray) -> Optional[Tuple[float, float]]:
    """Calculate contour centre using image moments."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def calculate_circularity(contour: np.ndarray) -> float:
    """Return 4*pi*area/perimeter^2, close to 1 for circular objects."""
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def extract_candidate(
    contour: np.ndarray,
    original_gray: np.ndarray,
    frame_center: Tuple[float, float],
    candidate_id: int,
) -> Optional[BeaconCandidate]:
    """Extract all measurable properties for one contour."""
    centroid = calculate_centroid(contour)
    if centroid is None:
        return None

    x, y = centroid
    area = float(cv2.contourArea(contour))
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
    del circle_x, circle_y
    bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour)

    mask = np.zeros(original_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    pixels = original_gray[mask == 255]
    if pixels.size == 0:
        mean_intensity = 0.0
        max_intensity = 0
    else:
        mean_intensity = float(np.mean(pixels))
        max_intensity = int(np.max(pixels))

    distance = float(np.hypot(x - frame_center[0], y - frame_center[1]))
    candidate = BeaconCandidate(
        candidate_id=candidate_id,
        x=x,
        y=y,
        area=area,
        radius=float(radius),
        bbox_x=int(bbox_x),
        bbox_y=int(bbox_y),
        bbox_width=int(bbox_width),
        bbox_height=int(bbox_height),
        mean_intensity=mean_intensity,
        max_intensity=max_intensity,
        circularity=calculate_circularity(contour),
        distance_from_center=distance,
    )
    return candidate


def calculate_baseline_score(candidate: BeaconCandidate) -> float:
    """Score a candidate without ML or labels.

    This is only a baseline ranking for inspection. It is not final target
    identification, because stars and false beacons can score above the real
    target until the CNN stage is added.
    """
    area_score = min(candidate.area / 180.0, 1.0)
    mean_score = min(candidate.mean_intensity / 255.0, 1.0)
    max_score = min(candidate.max_intensity / 255.0, 1.0)
    circularity_score = min(max(candidate.circularity, 0.0), 1.0)
    score = 0.30 * area_score + 0.25 * mean_score + 0.25 * max_score + 0.20 * circularity_score
    return float(round(score, 6))


def filter_candidates(
    candidates: Iterable[BeaconCandidate],
    min_area: float,
    max_area: float,
    min_radius: float,
    max_radius: float,
    min_circularity: float,
) -> List[BeaconCandidate]:
    """Remove invalid bright regions while keeping dim or blurred beacons."""
    filtered = []
    for candidate in candidates:
        if not (min_area <= candidate.area <= max_area):
            continue
        if not (min_radius <= candidate.radius <= max_radius):
            continue
        if not math.isfinite(candidate.circularity) or candidate.circularity < min_circularity:
            continue
        filtered.append(candidate)
    return filtered


def validate_candidates(candidates: List[BeaconCandidate], width: int, height: int) -> None:
    """Validate candidate geometry and sort order."""
    seen_ids = set()
    previous_score = float("inf")
    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
        seen_ids.add(candidate.candidate_id)
        if not (0 <= candidate.x < width and 0 <= candidate.y < height):
            raise ValueError(f"candidate coordinates outside image: {candidate}")
        if candidate.area <= 0 or candidate.radius <= 0:
            raise ValueError(f"candidate has non-positive area/radius: {candidate}")
        if not math.isfinite(candidate.circularity):
            raise ValueError(f"candidate circularity is not finite: {candidate}")
        if candidate.baseline_score > previous_score:
            raise ValueError("candidate list is not sorted by baseline score")
        previous_score = candidate.baseline_score


def candidate_with_score(candidate: BeaconCandidate) -> BeaconCandidate:
    """Return a candidate copy with its baseline score filled in."""
    data = candidate.to_dict()
    data["baseline_score"] = calculate_baseline_score(candidate)
    return BeaconCandidate(**data)


def detect_candidates(
    frame: np.ndarray,
    preprocessing_config: PreprocessingConfig | Dict[str, object],
    detector_config: DetectorConfig | Dict[str, object],
) -> Dict[str, object]:
    """Preprocess a frame, detect contours, extract candidates and rank them."""
    prep_config = preprocessing_config_from_mapping(preprocessing_config)
    det_config = detector_config_from_mapping(detector_config)
    preprocessing_result = preprocess_frame(frame, prep_config)
    binary = preprocessing_result["binary"]
    gray = convert_to_grayscale(frame)
    if not isinstance(binary, np.ndarray):
        raise ValueError("preprocessing did not return a binary image")

    height, width = binary.shape[:2]
    frame_center = (width / 2.0, height / 2.0)
    contours = find_contours(binary)

    extracted = []
    for contour_index, contour in enumerate(contours, start=1):
        candidate = extract_candidate(contour, gray, frame_center, contour_index)
        if candidate is not None:
            extracted.append(candidate)

    filtered = filter_candidates(
        extracted,
        min_area=det_config.min_area,
        max_area=det_config.max_area,
        min_radius=det_config.min_radius,
        max_radius=det_config.max_radius,
        min_circularity=det_config.min_circularity,
    )
    ranked = sorted((candidate_with_score(candidate) for candidate in filtered), key=lambda item: item.baseline_score, reverse=True)
    validate_candidates(ranked, width, height)

    return {
        "preprocessing": preprocessing_result,
        "candidates": ranked,
        "best_candidate": ranked[0] if ranked else None,
    }


def draw_candidates(
    frame: np.ndarray,
    candidates: List[BeaconCandidate],
    best_candidate: Optional[BeaconCandidate],
    ground_truth: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw candidate overlays, camera centre, best baseline choice and labels."""
    overlay = frame.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    height, width = overlay.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(overlay, center, (255, 255, 255), cv2.MARKER_CROSS, 22, 1)

    for candidate in candidates:
        point = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.circle(overlay, point, int(round(candidate.radius)), (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(candidate.candidate_id),
            (point[0] + 6, max(12, point[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if best_candidate is not None:
        best_point = (int(round(best_candidate.x)), int(round(best_candidate.y)))
        cv2.circle(overlay, best_point, int(round(best_candidate.radius)) + 4, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(overlay, center, best_point, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        x_error = best_candidate.x - center[0]
        y_error = best_candidate.y - center[1]
        best_text = f"Best #{best_candidate.candidate_id}: ({best_candidate.x:.1f}, {best_candidate.y:.1f}) score={best_candidate.baseline_score:.3f}"
        error_text = f"Error: dx={x_error:.1f}px dy={y_error:.1f}px"
    else:
        best_text = "Best: none"
        error_text = "Error: n/a"

    if ground_truth is not None:
        gt_x, gt_y, gt_radius = ground_truth
        cv2.circle(overlay, (gt_x, gt_y), gt_radius + 6, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(overlay, (gt_x, gt_y), (255, 0, 0), cv2.MARKER_CROSS, 16, 2)

    info_lines = [
        f"Candidates: {len(candidates)}",
        best_text,
        error_text,
    ]
    for index, line in enumerate(info_lines):
        y = 24 + index * 24
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def process_single_image(
    input_path: str | Path,
    output_path: str | Path,
    configs: Dict[str, object],
    ground_truth: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, object]:
    """Detect candidates in one image, save an overlay and return data."""
    image_path = Path(input_path)
    output_dir = Path(output_path)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frame = load_image(image_path)
    preprocessing_config = configs.get("preprocessing", PreprocessingConfig())
    detector_config = configs.get("detector", DetectorConfig())
    detection = detect_candidates(frame, preprocessing_config, detector_config)
    candidates = detection["candidates"]
    best_candidate = detection["best_candidate"]
    if not isinstance(candidates, list):
        raise ValueError("candidate detection returned invalid candidate list")
    if best_candidate is not None and not isinstance(best_candidate, BeaconCandidate):
        raise ValueError("candidate detection returned invalid best candidate")

    overlay = draw_candidates(frame, candidates, best_candidate, ground_truth=ground_truth)
    overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"failed to write overlay image: {overlay_path}")

    return {
        "filename": image_path.name,
        "overlay_path": str(overlay_path),
        "candidates": candidates,
        "best_candidate": best_candidate,
        "candidate_count": len(candidates),
    }


def process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    configs: Dict[str, object],
    labels_path: str | Path | None = None,
    max_overlays: Optional[int] = None,
) -> Dict[str, object]:
    """Process PNG images, save candidate CSV, overlays and summary JSON."""
    source_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(source_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG images found in {source_dir}")
        summary = {"processed": 0, "failed": 0, "frames": [], "evaluation": None}
        write_summary(output_dir / "summary.json", summary)
        write_candidates_csv(output_dir / "candidates.csv", [])
        return summary

    labels_by_filename = read_labels(labels_path) if labels_path else {}
    overlay_filenames = choose_overlay_filenames(image_paths, max_overlays)
    candidate_rows: List[Dict[str, object]] = []
    frame_results: List[Dict[str, object]] = []
    failed_images: List[Dict[str, str]] = []
    overlays_saved = 0

    for image_path in image_paths:
        try:
            ground_truth = ground_truth_from_label(labels_by_filename.get(image_path.name))
            frame = load_image(image_path)
            detection = detect_candidates(
                frame,
                configs.get("preprocessing", PreprocessingConfig()),
                configs.get("detector", DetectorConfig()),
            )
            candidates = detection["candidates"]
            best_candidate = detection["best_candidate"]
            if not isinstance(candidates, list):
                raise ValueError("invalid candidate list")

            should_save_overlay = image_path.name in overlay_filenames
            overlay_path = None
            if should_save_overlay:
                overlay = draw_candidates(frame, candidates, best_candidate, ground_truth=ground_truth)
                overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise OSError(f"failed to write overlay image: {overlay_path}")
                overlays_saved += 1

            top_id = best_candidate.candidate_id if isinstance(best_candidate, BeaconCandidate) else None
            for candidate in candidates:
                row = {"filename": image_path.name, **candidate.to_dict()}
                row["is_top_baseline_candidate"] = int(candidate.candidate_id == top_id)
                candidate_rows.append(row)

            frame_results.append(
                {
                    "filename": image_path.name,
                    "candidate_count": len(candidates),
                    "best_candidate": best_candidate.to_dict() if isinstance(best_candidate, BeaconCandidate) else None,
                    "candidates": [candidate.to_dict() for candidate in candidates],
                    "overlay_path": str(overlay_path) if overlay_path else None,
                }
            )
        except Exception as exc:
            failed_images.append({"filename": image_path.name, "error": str(exc)})
            print(f"Failed {image_path.name}: {exc}")

    write_candidates_csv(output_dir / "candidates.csv", candidate_rows)
    evaluation = evaluate_candidate_recall(
        frame_results,
        labels_by_filename,
        match_tolerance=detector_config_from_mapping(configs.get("detector", DetectorConfig())).match_tolerance,
    ) if labels_by_filename else None
    summary = {
        "processed": len(frame_results),
        "failed": len(failed_images),
        "failed_images": failed_images,
        "frames_with_zero_candidates": sum(1 for frame in frame_results if int(frame["candidate_count"]) == 0),
        "overlays_saved": overlays_saved,
        "evaluation": evaluation,
    }
    write_summary(output_dir / "summary.json", summary)

    print(f"Processed images: {len(frame_results)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Candidate CSV: {output_dir / 'candidates.csv'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    if evaluation:
        print(f"Candidate recall: {evaluation['candidate_recall']:.4f}")
        print(f"Baseline top-1 accuracy: {evaluation['baseline_top1_accuracy']:.4f}")
    return summary


def choose_overlay_filenames(image_paths: List[Path], max_overlays: Optional[int]) -> set[str]:
    """Choose evenly spaced overlays so previews cover the dataset."""
    if max_overlays is None or max_overlays >= len(image_paths):
        return {path.name for path in image_paths}
    if max_overlays <= 0:
        return set()

    indices = np.linspace(0, len(image_paths) - 1, max_overlays, dtype=int)
    return {image_paths[int(index)].name for index in indices}


def evaluate_candidate_recall(
    results: List[Dict[str, object]],
    labels: Dict[str, Dict[str, str]],
    match_tolerance: float = 12.0,
) -> Dict[str, object]:
    """Evaluate whether visible targets appear somewhere in candidate lists."""
    visible_total = 0
    visible_covered = 0
    missed_visible_targets: List[str] = []
    top1_hits = 0
    absent_total = 0
    absent_with_candidates = 0
    total_candidates = 0
    zero_candidate_frames = 0

    for result in results:
        filename = str(result["filename"])
        label = labels.get(filename)
        candidates = result.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        total_candidates += len(candidates)
        if not candidates:
            zero_candidate_frames += 1
        if label is None:
            continue

        target_visible = int(label["target_visible"])
        if target_visible:
            visible_total += 1
            target_x = float(label["target_x"])
            target_y = float(label["target_y"])
            covered_ids = [
                int(candidate["candidate_id"])
                for candidate in candidates
                if distance(candidate, target_x, target_y) <= match_tolerance
            ]
            if covered_ids:
                visible_covered += 1
                best = result.get("best_candidate")
                if isinstance(best, dict) and int(best["candidate_id"]) in covered_ids:
                    top1_hits += 1
            else:
                missed_visible_targets.append(filename)
        else:
            absent_total += 1
            if candidates:
                absent_with_candidates += 1

    frame_count = len(results)
    candidate_recall = visible_covered / visible_total if visible_total else 0.0
    top1_accuracy = top1_hits / visible_total if visible_total else 0.0
    return {
        "candidate_recall": round(candidate_recall, 6),
        "missed_visible_targets": missed_visible_targets,
        "missed_visible_target_count": len(missed_visible_targets),
        "average_candidates_per_frame": round(total_candidates / frame_count, 4) if frame_count else 0.0,
        "frames_with_zero_candidates": zero_candidate_frames,
        "baseline_top1_accuracy": round(top1_accuracy, 6),
        "target_absent_frame_count": absent_total,
        "false_detections_in_target_absent_frames": absent_with_candidates,
        "match_tolerance": match_tolerance,
    }


def distance(candidate: Dict[str, object], target_x: float, target_y: float) -> float:
    """Calculate candidate distance from a ground-truth point."""
    return float(np.hypot(float(candidate["x"]) - target_x, float(candidate["y"]) - target_y))


def read_labels(labels_path: str | Path | None) -> Dict[str, Dict[str, str]]:
    """Read Phase 1 labels keyed by filename."""
    if labels_path is None:
        return {}
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def ground_truth_from_label(label: Optional[Dict[str, str]]) -> Optional[Tuple[int, int, int]]:
    """Return ground-truth tuple for visible targets."""
    if label is None or int(label["target_visible"]) == 0:
        return None
    return int(label["target_x"]), int(label["target_y"]), int(label["target_radius"])


def write_candidates_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write one row per detected candidate."""
    fieldnames = [
        "filename",
        "candidate_id",
        "x",
        "y",
        "area",
        "radius",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "mean_intensity",
        "max_intensity",
        "circularity",
        "distance_from_center",
        "baseline_score",
        "is_top_baseline_candidate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    """Write a readable JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def preprocessing_config_from_mapping(config: PreprocessingConfig | Dict[str, object]) -> PreprocessingConfig:
    """Normalize preprocessing config inputs."""
    if isinstance(config, PreprocessingConfig):
        return config
    return PreprocessingConfig(
        threshold_method=str(config.get("threshold_method", "fixed")),
        threshold_value=int(config.get("threshold_value", 200)),
        blur_kernel=int(config.get("blur_kernel", 5)),
        morph_kernel=int(config.get("morph_kernel", 3)),
        normalize_contrast=bool(config.get("normalize_contrast", False)),
    )


def detector_config_from_mapping(config: DetectorConfig | Dict[str, object]) -> DetectorConfig:
    """Normalize detector config inputs."""
    if isinstance(config, DetectorConfig):
        return config
    return DetectorConfig(
        min_area=float(config.get("min_area", 3.0)),
        max_area=float(config.get("max_area", 1000.0)),
        min_radius=float(config.get("min_radius", 1.0)),
        max_radius=float(config.get("max_radius", 25.0)),
        min_circularity=float(config.get("min_circularity", 0.15)),
        match_tolerance=float(config.get("match_tolerance", 12.0)),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect bright FSOC beacon candidates.")
    parser.add_argument("--input", required=True, help="Input PNG image or folder.")
    parser.add_argument("--output", required=True, help="Output folder for overlays, CSV and summary.")
    parser.add_argument("--labels", help="Optional Phase 1 labels CSV for evaluation.")
    parser.add_argument("--threshold-method", choices=["fixed", "otsu", "adaptive"], default="fixed")
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--min-area", type=float, default=3.0)
    parser.add_argument("--max-area", type=float, default=1000.0)
    parser.add_argument("--min-radius", type=float, default=1.0)
    parser.add_argument("--max-radius", type=float, default=25.0)
    parser.add_argument("--min-circularity", type=float, default=0.15)
    parser.add_argument("--match-tolerance", type=float, default=12.0)
    parser.add_argument("--preview", action="store_true", help="Save overlay for a single image.")
    parser.add_argument("--max-overlays", type=int, default=None, help="Maximum folder overlays to save.")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> Dict[str, object]:
    """Build preprocessing and detector configs from CLI arguments."""
    return {
        "preprocessing": PreprocessingConfig(
            threshold_method=args.threshold_method,
            threshold_value=args.threshold,
            blur_kernel=args.blur_kernel,
            morph_kernel=args.morph_kernel,
        ),
        "detector": DetectorConfig(
            min_area=args.min_area,
            max_area=args.max_area,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_circularity=args.min_circularity,
            match_tolerance=args.match_tolerance,
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    configs = build_configs(args)

    if input_path.is_file():
        labels = read_labels(args.labels) if args.labels else {}
        ground_truth = ground_truth_from_label(labels.get(input_path.name))
        result = process_single_image(input_path, output_path, configs, ground_truth=ground_truth if args.preview else None)
        print(f"Candidates found: {result['candidate_count']}")
        print(f"Overlay saved: {result['overlay_path']}")
    elif input_path.is_dir():
        process_folder(input_path, output_path, configs, labels_path=args.labels, max_overlays=args.max_overlays)
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
