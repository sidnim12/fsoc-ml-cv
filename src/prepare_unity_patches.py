from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from .candidate_detector import detect_candidates
    from .pipeline import load_config
    from .prepare_patches import LABEL_TO_ID, PATCH_LABEL_COLUMNS, crop_with_padding
except ImportError:  # pragma: no cover - allows direct script execution
    from candidate_detector import detect_candidates
    from pipeline import load_config
    from prepare_patches import LABEL_TO_ID, PATCH_LABEL_COLUMNS, crop_with_padding


@dataclass(frozen=True)
class UnityPatchConfig:
    """Settings for preparing Unity-domain classifier patches."""

    patch_size: int = 32
    crop_size: int = 64
    match_tolerance: float = 24.0
    label_frame_offset: int = 1
    snap_radius: int = 64
    snap_min_intensity: int = 180
    max_negatives_per_frame: int = 5
    max_negative_ratio: Optional[float] = 3.0
    seed: int = 42


def find_sequences(root: str | Path) -> List[Path]:
    """Return sequence folders that contain frames and labels."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Unity dataset root does not exist: {root_path}")
    sequences = sorted(
        path
        for path in root_path.glob("sequence_*")
        if path.is_dir() and (path / "labels.csv").exists() and list(path.glob("frame_*.png"))
    )
    if not sequences:
        raise ValueError(f"no labelled Unity sequence folders found under {root_path}")
    return sequences


def build_sequence_split_map(sequences: Sequence[Path]) -> Dict[Path, str]:
    """Assign sequences while keeping positives in validation and test."""
    positive_sequences: List[Path] = []
    for sequence_dir in sequences:
        labels = pd.read_csv(sequence_dir / "labels.csv")
        has_positive = "target_present" in labels and (labels["target_present"].fillna(0).astype(int) == 1).any()
        if has_positive:
            positive_sequences.append(sequence_dir)

    split_map = {sequence_dir: "train" for sequence_dir in sequences}
    if len(positive_sequences) >= 2:
        split_map[positive_sequences[-2]] = "validation"
        split_map[positive_sequences[-1]] = "test"
    elif len(positive_sequences) == 1:
        split_map[positive_sequences[-1]] = "test"
    return split_map


def snap_to_brightest(
    gray: np.ndarray,
    x_value: float,
    y_value: float,
    radius: int,
    min_intensity: int,
) -> Optional[Tuple[float, float, int]]:
    """Snap an approximate Unity label to the brightest nearby pixel."""
    height, width = gray.shape[:2]
    x = int(round(x_value))
    y = int(round(y_value))
    if not (0 <= x < width and 0 <= y < height):
        return None

    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    patch = gray[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return None

    _, max_value, _, max_location = cv2.minMaxLoc(patch)
    if int(max_value) < min_intensity:
        return None
    return float(x_min + max_location[0]), float(y_min + max_location[1]), int(max_value)


def candidate_distance(candidate: Mapping[str, object], x_value: float, y_value: float) -> float:
    """Return candidate distance from a point."""
    return float(math.hypot(float(candidate["x"]) - x_value, float(candidate["y"]) - y_value))


def make_patch_record(
    *,
    source_filename: str,
    source_path: Path,
    label_frame_id: int,
    sequence_id: int,
    scenario: str,
    split: str,
    candidate_id: int,
    label: str,
    candidate_x: float,
    candidate_y: float,
    target_x: float,
    target_y: float,
    distance_to_ground_truth: float,
    area: float,
    radius: float,
    mean_intensity: float,
    max_intensity: int,
    circularity: float,
    baseline_score: float,
) -> Dict[str, object]:
    """Create one patch metadata row."""
    patch_filename = f"{source_path.parent.name}_label_{label_frame_id:06d}_{source_path.stem}_candidate_{candidate_id:03d}_{label}.png"
    return {
        "patch_filename": patch_filename,
        "patch_path": str(Path(split) / label / patch_filename),
        "source_filename": source_filename,
        "scenario_id": sequence_id,
        "scenario_type": scenario,
        "split": split,
        "candidate_id": candidate_id,
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "candidate_x": candidate_x,
        "candidate_y": candidate_y,
        "target_x": target_x,
        "target_y": target_y,
        "distance_to_ground_truth": distance_to_ground_truth,
        "area": area,
        "radius": radius,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "circularity": circularity,
        "baseline_score": baseline_score,
    }


def build_unity_patch_table(
    root: str | Path,
    config_path: str | Path,
    config: UnityPatchConfig,
) -> pd.DataFrame:
    """Build Unity-domain positive and negative patch metadata."""
    pipeline_config = load_config(config_path)
    sequences = find_sequences(root)
    split_map = build_sequence_split_map(sequences)
    records: List[Dict[str, object]] = []

    for sequence_number, sequence_dir in enumerate(sequences):
        split = split_map[sequence_dir]
        labels = pd.read_csv(sequence_dir / "labels.csv")
        sequence_id = int(labels["sequence_id"].iloc[0]) if "sequence_id" in labels else sequence_number + 1
        scenario = str(labels["scenario"].iloc[0]) if "scenario" in labels else sequence_dir.name

        for row in labels.itertuples(index=False):
            label_frame_id = int(getattr(row, "frame_id"))
            if int(getattr(row, "target_present")) != 1:
                image_frame = label_frame_id
                target_x = -1.0
                target_y = -1.0
                snapped = None
            else:
                image_frame = label_frame_id + config.label_frame_offset
                target_x = float(getattr(row, "target_x"))
                target_y = float(getattr(row, "target_y"))

            image_path = sequence_dir / f"frame_{image_frame:06d}.png"
            if not image_path.exists():
                continue

            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            source_filename = str(image_path.relative_to(Path(root)))

            if int(getattr(row, "target_present")) == 1:
                snapped = snap_to_brightest(gray, target_x, target_y, config.snap_radius, config.snap_min_intensity)
                if snapped is not None:
                    positive_x, positive_y, max_intensity = snapped
                    records.append(
                        make_patch_record(
                            source_filename=source_filename,
                            source_path=image_path,
                            label_frame_id=label_frame_id,
                            sequence_id=sequence_id,
                            scenario=scenario,
                            split=split,
                            candidate_id=0,
                            label="correct",
                            candidate_x=positive_x,
                            candidate_y=positive_y,
                            target_x=positive_x,
                            target_y=positive_y,
                            distance_to_ground_truth=0.0,
                            area=1.0,
                            radius=1.0,
                            mean_intensity=float(max_intensity),
                            max_intensity=max_intensity,
                            circularity=1.0,
                            baseline_score=1.0,
                        )
                    )
                    target_x = positive_x
                    target_y = positive_y

            detection = detect_candidates(frame, pipeline_config.preprocessing, pipeline_config.detector)
            candidates = detection.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []

            negative_count = 0
            matched_positive_count = 0
            for candidate in candidates:
                candidate_data = candidate.to_dict()
                distance = math.inf if target_x < 0 or target_y < 0 else candidate_distance(candidate_data, target_x, target_y)
                if distance <= config.match_tolerance:
                    if int(getattr(row, "target_present")) == 1:
                        matched_positive_count += 1
                        records.append(
                            make_patch_record(
                                source_filename=source_filename,
                                source_path=image_path,
                                label_frame_id=label_frame_id,
                                sequence_id=sequence_id,
                                scenario=scenario,
                                split=split,
                                candidate_id=int(candidate_data["candidate_id"]),
                                label="correct",
                                candidate_x=float(candidate_data["x"]),
                                candidate_y=float(candidate_data["y"]),
                                target_x=target_x,
                                target_y=target_y,
                                distance_to_ground_truth=distance,
                                area=float(candidate_data["area"]),
                                radius=float(candidate_data["radius"]),
                                mean_intensity=float(candidate_data["mean_intensity"]),
                                max_intensity=int(candidate_data["max_intensity"]),
                                circularity=float(candidate_data["circularity"]),
                                baseline_score=float(candidate_data["baseline_score"]),
                            )
                        )
                    continue
                negative_count += 1
                records.append(
                    make_patch_record(
                        source_filename=source_filename,
                        source_path=image_path,
                        label_frame_id=label_frame_id,
                        sequence_id=sequence_id,
                        scenario=scenario,
                        split=split,
                        candidate_id=int(candidate_data["candidate_id"]),
                        label="false",
                        candidate_x=float(candidate_data["x"]),
                        candidate_y=float(candidate_data["y"]),
                        target_x=target_x,
                        target_y=target_y,
                        distance_to_ground_truth=distance,
                        area=float(candidate_data["area"]),
                        radius=float(candidate_data["radius"]),
                        mean_intensity=float(candidate_data["mean_intensity"]),
                        max_intensity=int(candidate_data["max_intensity"]),
                        circularity=float(candidate_data["circularity"]),
                        baseline_score=float(candidate_data["baseline_score"]),
                    )
                )
                if negative_count >= config.max_negatives_per_frame:
                    break

    if not records:
        raise ValueError("no Unity patches could be prepared")
    return pd.DataFrame(records, columns=PATCH_LABEL_COLUMNS)


def limit_negative_ratio_by_split(patches: pd.DataFrame, max_negative_ratio: Optional[float], seed: int) -> pd.DataFrame:
    """Cap false patches in each split so training is not dominated by negatives."""
    if max_negative_ratio is None:
        return patches
    if max_negative_ratio <= 0:
        raise ValueError("max_negative_ratio must be positive")

    rng = np.random.default_rng(seed)
    parts = []
    for split, group in patches.groupby("split", sort=False):
        correct = group[group["label"] == "correct"]
        false = group[group["label"] == "false"]
        max_false = int(math.floor(len(correct) * max_negative_ratio))
        if len(correct) == 0 or len(false) <= max_false:
            parts.append(group)
            continue
        chosen_false = false.iloc[rng.permutation(len(false))[:max_false]]
        parts.append(pd.concat([correct, chosen_false]).sort_index())
    return pd.concat(parts).reset_index(drop=True)


def save_unity_patches(root: str | Path, patches: pd.DataFrame, output: str | Path, config: UnityPatchConfig) -> pd.DataFrame:
    """Crop and save Unity patch images."""
    root_path = Path(root)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_rows = []
    image_cache: Dict[str, np.ndarray] = {}

    for row in patches.itertuples(index=False):
        image_path = root_path / str(row.source_filename)
        cache_key = str(image_path)
        if cache_key not in image_cache:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"source image unreadable: {image_path}")
            image_cache[cache_key] = image

        patch = crop_with_padding(
            image_cache[cache_key],
            center=(float(row.candidate_x), float(row.candidate_y)),
            crop_size=config.crop_size,
            patch_size=config.patch_size,
        )
        patch_path = output_path / str(row.patch_path)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(patch_path), patch):
            raise OSError(f"failed to save patch: {patch_path}")
        saved_rows.append(row._asdict())

    saved = pd.DataFrame(saved_rows, columns=PATCH_LABEL_COLUMNS)
    saved.to_csv(output_path / "patch_labels.csv", index=False)
    return saved


def summarize_patches(patches: pd.DataFrame, config: UnityPatchConfig) -> Dict[str, object]:
    """Return JSON-friendly patch summary."""
    return {
        "total_patches": int(len(patches)),
        "correct_patches": int((patches["label"] == "correct").sum()),
        "false_patches": int((patches["label"] == "false").sum()),
        "counts_per_split": nested_counts(patches, "split", "label"),
        "counts_per_scenario": nested_counts(patches, "scenario_type", "label"),
        "config": config.__dict__,
    }


def nested_counts(data: pd.DataFrame, outer: str, inner: str) -> Dict[str, Dict[str, int]]:
    """Build nested count dictionaries."""
    grouped = data.groupby([outer, inner]).size()
    result: Dict[str, Dict[str, int]] = {}
    for (outer_value, inner_value), count in grouped.items():
        result.setdefault(str(outer_value), {})[str(inner_value)] = int(count)
    return result


def write_summary(path: str | Path, summary: Mapping[str, object]) -> None:
    """Write a JSON summary."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_unity_patch_preparation(
    root: str | Path,
    output: str | Path,
    config_path: str | Path,
    config: UnityPatchConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Prepare Unity-domain classifier patches."""
    patches = build_unity_patch_table(root, config_path, config)
    patches = limit_negative_ratio_by_split(patches, config.max_negative_ratio, config.seed)
    saved = save_unity_patches(root, patches, output, config)
    summary = summarize_patches(saved, config)
    write_summary(Path(output) / "dataset_summary.json", summary)
    print("Unity patch dataset prepared")
    print(f"  Total patches: {summary['total_patches']}")
    print(f"  Correct patches: {summary['correct_patches']}")
    print(f"  False patches: {summary['false_patches']}")
    print(f"  Output: {output}")
    return saved, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Unity-domain CNN patches from labelled sequences.")
    parser.add_argument("--root", default="data/raw/unity/cont_dataset_2400/unity_base_2400")
    parser.add_argument("--output", default="data/processed/unity_base_2400_patches")
    parser.add_argument("--config", default="configs/unity.yaml")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--match-tolerance", type=float, default=24.0)
    parser.add_argument("--label-frame-offset", type=int, default=1)
    parser.add_argument("--snap-radius", type=int, default=64)
    parser.add_argument("--snap-min-intensity", type=int, default=180)
    parser.add_argument("--max-negatives-per-frame", type=int, default=5)
    parser.add_argument("--max-negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    config = UnityPatchConfig(
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        match_tolerance=args.match_tolerance,
        label_frame_offset=args.label_frame_offset,
        snap_radius=args.snap_radius,
        snap_min_intensity=args.snap_min_intensity,
        max_negatives_per_frame=args.max_negatives_per_frame,
        max_negative_ratio=args.max_negative_ratio,
        seed=args.seed,
    )
    try:
        run_unity_patch_preparation(args.root, args.output, args.config, config)
    except Exception as exc:
        raise SystemExit(f"Unity patch preparation failed: {exc}") from exc


if __name__ == "__main__":
    main()

