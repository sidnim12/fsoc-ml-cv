from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_TO_ID = {"false": 0, "correct": 1}
SPLITS = ("train", "validation", "test")
CLASSES = ("correct", "false")
PATCH_LABEL_COLUMNS = [
    "patch_filename",
    "patch_path",
    "source_filename",
    "scenario_id",
    "scenario_type",
    "split",
    "candidate_id",
    "label",
    "label_id",
    "candidate_x",
    "candidate_y",
    "target_x",
    "target_y",
    "distance_to_ground_truth",
    "area",
    "radius",
    "mean_intensity",
    "max_intensity",
    "circularity",
    "baseline_score",
]


@dataclass(frozen=True)
class PatchConfig:
    """Settings for building candidate patch datasets."""

    patch_size: int = 32
    crop_size: int = 40
    match_tolerance: float = 12.0
    seed: int = 42
    max_negative_ratio: Optional[float] = 3.0


def load_inputs(
    images_dir: str | Path,
    labels_path: str | Path,
    candidates_path: str | Path,
) -> Tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Load image directory, labels CSV and candidate CSV."""
    image_root = Path(images_dir)
    labels_file = Path(labels_path)
    candidates_file = Path(candidates_path)

    if not image_root.exists() or not image_root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_root}")
    if not labels_file.exists():
        raise FileNotFoundError(f"labels CSV does not exist: {labels_file}")
    if not candidates_file.exists():
        raise FileNotFoundError(f"candidates CSV does not exist: {candidates_file}")

    labels = pd.read_csv(labels_file)
    candidates = pd.read_csv(candidates_file)
    validate_input_tables(image_root, labels, candidates)
    return image_root, labels, candidates


def validate_input_tables(image_root: Path, labels: pd.DataFrame, candidates: pd.DataFrame) -> None:
    """Validate required columns and referenced files before patch creation."""
    label_columns = {
        "filename",
        "scenario_id",
        "scenario_type",
        "target_visible",
        "target_x",
        "target_y",
        "target_radius",
    }
    candidate_columns = {
        "filename",
        "candidate_id",
        "x",
        "y",
        "area",
        "radius",
        "mean_intensity",
        "max_intensity",
        "circularity",
        "baseline_score",
    }
    missing_labels = sorted(label_columns - set(labels.columns))
    missing_candidates = sorted(candidate_columns - set(candidates.columns))
    if missing_labels:
        raise ValueError(f"labels CSV missing required columns: {missing_labels}")
    if missing_candidates:
        raise ValueError(f"candidates CSV missing required columns: {missing_candidates}")
    if labels.empty:
        raise ValueError("labels CSV is empty")
    if candidates.empty:
        raise ValueError("candidates CSV is empty")

    label_filenames = set(labels["filename"].astype(str))
    candidate_filenames = set(candidates["filename"].astype(str))
    missing_label_rows = sorted(candidate_filenames - label_filenames)
    if missing_label_rows:
        raise ValueError(f"candidate rows missing ground-truth labels: {missing_label_rows[:5]}")

    missing_images = [name for name in sorted(candidate_filenames) if not (image_root / name).exists()]
    if missing_images:
        raise FileNotFoundError(f"candidate source images missing: {missing_images[:5]}")


def print_candidate_distribution_precheck(labels: pd.DataFrame, candidates: pd.DataFrame) -> Dict[str, float | int | bool]:
    """Report whether candidate output has enough negatives for CNN preparation."""
    frame_counts = candidates.groupby("filename").size().rename("candidate_count").reset_index()
    merged = labels.merge(frame_counts, on="filename", how="left").fillna({"candidate_count": 0})
    merged["candidate_count"] = merged["candidate_count"].astype(int)

    clean_avg = float(merged[merged["scenario_type"] == "clean_target"]["candidate_count"].mean())
    star_avg = float(merged[merged["scenario_type"] == "target_with_stars"]["candidate_count"].mean())
    multi_avg = float(merged[merged["scenario_type"] == "target_with_false_beacons"]["candidate_count"].mean())
    absent = merged[merged["scenario_type"] == "target_not_visible"]
    absent_candidate_count = int(absent["candidate_count"].sum())
    all_one_candidate = bool((merged["candidate_count"] == 1).all())

    print("Candidate-distribution precheck:")
    print(f"  Average candidates per clean frame: {clean_avg:.3f}")
    print(f"  Average candidates per star frame: {star_avg:.3f}")
    print(f"  Average candidates per multiple-beacon frame: {multi_avg:.3f}")
    print(f"  Candidates found in target-absent frames: {absent_candidate_count}")

    return {
        "clean_average_candidates": clean_avg,
        "star_average_candidates": star_avg,
        "multiple_beacon_average_candidates": multi_avg,
        "target_absent_candidate_count": absent_candidate_count,
        "all_frames_have_one_candidate": all_one_candidate,
    }


def match_candidate_to_target(candidate: pd.Series, label: pd.Series, match_tolerance: float) -> Tuple[str, float]:
    """Assign correct/false using only ground truth and candidate distance."""
    if int(label["target_visible"]) == 0:
        return "false", math.inf

    target_x = float(label["target_x"])
    target_y = float(label["target_y"])
    distance = float(np.hypot(float(candidate["x"]) - target_x, float(candidate["y"]) - target_y))
    patch_label = "correct" if distance <= match_tolerance else "false"
    return patch_label, distance


def crop_with_padding(image: np.ndarray, center: Tuple[float, float], crop_size: int, patch_size: int) -> np.ndarray:
    """Crop around a candidate, pad at edges, and resize to patch size."""
    if crop_size <= 0 or patch_size <= 0:
        raise ValueError("crop_size and patch_size must be positive")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB/BGR image with 3 channels, got shape {image.shape}")

    half = crop_size // 2
    x = int(round(center[0]))
    y = int(round(center[1]))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x += half
    y += half
    crop = padded[y - half : y - half + crop_size, x - half : x - half + crop_size]
    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    if crop_size != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return crop


def build_patch_table(
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PatchConfig,
) -> pd.DataFrame:
    """Merge labels with candidates and assign patch labels."""
    labels_by_filename = {str(row.filename): row for row in labels.itertuples(index=False)}
    records: List[Dict[str, object]] = []

    for candidate in candidates.itertuples(index=False):
        filename = str(candidate.filename)
        label_row = labels_by_filename[filename]
        label_series = pd.Series(label_row._asdict())
        candidate_series = pd.Series(candidate._asdict())
        patch_label, distance = match_candidate_to_target(candidate_series, label_series, config.match_tolerance)
        records.append(
            {
                "source_filename": filename,
                "scenario_id": int(label_series["scenario_id"]),
                "scenario_type": str(label_series["scenario_type"]),
                "candidate_id": int(candidate_series["candidate_id"]),
                "label": patch_label,
                "label_id": LABEL_TO_ID[patch_label],
                "candidate_x": float(candidate_series["x"]),
                "candidate_y": float(candidate_series["y"]),
                "target_x": int(label_series["target_x"]),
                "target_y": int(label_series["target_y"]),
                "distance_to_ground_truth": distance,
                "area": float(candidate_series["area"]),
                "radius": float(candidate_series["radius"]),
                "mean_intensity": float(candidate_series["mean_intensity"]),
                "max_intensity": int(candidate_series["max_intensity"]),
                "circularity": float(candidate_series["circularity"]),
                "baseline_score": float(candidate_series["baseline_score"]),
            }
        )

    patch_table = pd.DataFrame(records)
    if patch_table.empty or patch_table["label"].nunique() < 2:
        explain_insufficient_negatives()
        raise SystemExit("Insufficient negative candidates for CNN training.")
    return patch_table


def explain_insufficient_negatives() -> None:
    """Print likely causes when no useful negative patches exist."""
    print("Insufficient negative candidates for CNN training.")
    print("Likely causes to check:")
    print("  - False beacons are too dim")
    print("  - Candidate filtering is too strict")
    print("  - Thresholding is too strict")
    print("  - The montage did not include multiple-beacon frames")


def assign_data_splits(patches: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign train/validation/test splits while keeping each source frame together."""
    scenario_count = patches["scenario_id"].nunique()
    if scenario_count < 9:
        print(
            "Warning: not enough unique scenario groups for a reliable grouped scenario split; "
            "using deterministic source-frame stratified split."
        )

    frame_summary = (
        patches.groupby("source_filename")
        .agg(
            scenario_id=("scenario_id", "first"),
            scenario_type=("scenario_type", "first"),
            has_correct=("label", lambda values: int((values == "correct").any())),
            has_false=("label", lambda values: int((values == "false").any())),
        )
        .reset_index()
    )
    frame_summary["stratum"] = (
        frame_summary["scenario_id"].astype(str)
        + "_"
        + frame_summary["has_correct"].astype(str)
        + "_"
        + frame_summary["has_false"].astype(str)
    )

    assignments: Dict[str, str] = {}
    rng = np.random.default_rng(seed)
    for _, group in frame_summary.groupby("stratum", sort=True):
        names = group["source_filename"].astype(str).to_numpy()
        names = names[rng.permutation(len(names))]
        split_names = split_names_by_ratio(names)
        for split, filenames in split_names.items():
            for filename in filenames:
                assignments[str(filename)] = split

    patches = patches.copy()
    patches["split"] = patches["source_filename"].map(assignments)
    ensure_split_has_both_classes(patches)
    return patches


def split_names_by_ratio(names: np.ndarray) -> Dict[str, List[str]]:
    """Split filenames into 70/15/15 buckets with exact coverage."""
    count = len(names)
    train_count = int(round(count * 0.70))
    validation_count = int(round(count * 0.15))
    if count >= 3:
        train_count = min(max(train_count, 1), count - 2)
        validation_count = min(max(validation_count, 1), count - train_count - 1)
    test_count = count - train_count - validation_count
    return {
        "train": names[:train_count].tolist(),
        "validation": names[train_count : train_count + validation_count].tolist(),
        "test": names[train_count + validation_count : train_count + validation_count + test_count].tolist(),
    }


def ensure_split_has_both_classes(patches: pd.DataFrame) -> None:
    """Warn if a split lacks a class after deterministic splitting."""
    class_counts = patches.groupby(["split", "label"]).size().unstack(fill_value=0)
    for split in SPLITS:
        if split not in class_counts.index:
            print(f"Warning: split '{split}' has no patches.")
            continue
        missing = [label for label in CLASSES if class_counts.loc[split].get(label, 0) == 0]
        if missing:
            print(f"Warning: split '{split}' is missing class(es): {', '.join(missing)}")


def limit_negative_ratio(patches: pd.DataFrame, max_negative_ratio: Optional[float], seed: int) -> pd.DataFrame:
    """Cap false patches per correct patch without duplicating samples."""
    if max_negative_ratio is None:
        return patches
    if max_negative_ratio <= 0:
        raise ValueError("--max-negative-ratio must be positive")

    correct = patches[patches["label"] == "correct"]
    false = patches[patches["label"] == "false"]
    max_false = int(math.floor(len(correct) * max_negative_ratio))
    if len(false) <= max_false:
        return patches

    sampled_false_parts = []
    rng = np.random.default_rng(seed)
    for _, group in false.groupby("scenario_type", sort=True):
        group_quota = max(1, int(round(max_false * len(group) / len(false))))
        group_quota = min(group_quota, len(group))
        order = rng.permutation(len(group))
        sampled_false_parts.append(group.iloc[order[:group_quota]])

    sampled_false = pd.concat(sampled_false_parts).drop_duplicates()
    if len(sampled_false) > max_false:
        sampled_false = sampled_false.sample(n=max_false, random_state=seed)
    elif len(sampled_false) < max_false:
        remaining = false.drop(sampled_false.index)
        needed = max_false - len(sampled_false)
        sampled_false = pd.concat([sampled_false, remaining.sample(n=needed, random_state=seed)])

    limited = pd.concat([correct, sampled_false]).sort_index().reset_index(drop=True)
    print(f"Limited false patches from {len(false)} to {len(sampled_false)} using max ratio {max_negative_ratio}.")
    return limited


def safe_prepare_outputs(output_dir: Path, preview_dir: Path, overwrite: bool) -> None:
    """Create output directories and protect existing generated patch datasets."""
    generated_paths = [path for path in [output_dir / "patch_labels.csv", output_dir / "dataset_summary.json"] if path.exists()]
    generated_paths.extend(output_dir.glob("*/*/*.png"))
    generated_paths.extend(preview_dir.glob("*.png"))

    if generated_paths and not overwrite:
        raise FileExistsError(
            f"{output_dir} or {preview_dir} already contains generated patch outputs. "
            "Use --overwrite to replace only processed patch outputs."
        )

    if overwrite:
        for path in generated_paths:
            if path.is_file():
                path.unlink()

    for split in SPLITS:
        for label in CLASSES:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)


def patch_filename(row: pd.Series) -> str:
    """Create a clear deterministic patch filename."""
    source_stem = Path(str(row["source_filename"])).stem
    candidate_id = int(row["candidate_id"])
    label = str(row["label"])
    return f"{source_stem}_candidate_{candidate_id:03d}_{label}.png"


def save_patches(
    image_root: Path,
    patches: pd.DataFrame,
    output_dir: Path,
    patch_size: int,
    crop_size: int,
) -> pd.DataFrame:
    """Crop and save all candidate patches."""
    rows = []
    image_cache: Dict[str, np.ndarray] = {}

    for _, row in patches.iterrows():
        source_filename = str(row["source_filename"])
        if source_filename not in image_cache:
            image = cv2.imread(str(image_root / source_filename), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"source image unreadable: {image_root / source_filename}")
            image_cache[source_filename] = image

        patch = crop_with_padding(
            image_cache[source_filename],
            center=(float(row["candidate_x"]), float(row["candidate_y"])),
            crop_size=crop_size,
            patch_size=patch_size,
        )
        filename = patch_filename(row)
        relative_path = Path(str(row["split"])) / str(row["label"]) / filename
        absolute_path = output_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(absolute_path), patch):
            raise OSError(f"failed to save patch: {absolute_path}")

        out_row = row.to_dict()
        out_row["patch_filename"] = filename
        out_row["patch_path"] = str(relative_path)
        rows.append(out_row)

    return pd.DataFrame(rows, columns=PATCH_LABEL_COLUMNS)


def save_preview_grids(patch_labels: pd.DataFrame, output_dir: Path, preview_dir: Path, max_samples: int = 25) -> None:
    """Create separate preview montages for correct and false patches."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for label in CLASSES:
        subset = representative_patch_samples(patch_labels[patch_labels["label"] == label], max_samples)
        output_path = preview_dir / f"{label}_grid.png"
        save_one_preview_grid(subset, output_dir, output_path, label)


def representative_patch_samples(patches: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Pick preview patches across scenario types instead of only CSV order."""
    if patches.empty:
        return patches

    scenario_count = max(1, patches["scenario_type"].nunique())
    per_scenario = max(1, max_samples // scenario_count)
    parts = []
    for _, group in patches.groupby("scenario_type", sort=True):
        parts.append(group.head(per_scenario))

    selected = pd.concat(parts).head(max_samples)
    if len(selected) < max_samples:
        remaining = patches.drop(selected.index)
        selected = pd.concat([selected, remaining.head(max_samples - len(selected))])
    return selected.head(max_samples)


def save_one_preview_grid(subset: pd.DataFrame, output_dir: Path, output_path: Path, label: str) -> None:
    """Save one labelled preview grid."""
    cols = 5
    rows = max(1, int(math.ceil(max(len(subset), 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.2 * rows))
    axes_flat = np.array(axes).reshape(-1)

    for axis in axes_flat:
        axis.axis("off")

    for axis, (_, row) in zip(axes_flat, subset.iterrows()):
        patch_path = output_dir / str(row["patch_path"])
        patch = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
        if patch is None:
            raise ValueError(f"preview patch unreadable: {patch_path}")
        axis.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{Path(str(row['source_filename'])).stem}\nC{int(row['candidate_id']):03d}", fontsize=7)
        axis.axis("off")

    fig.suptitle(f"{label.capitalize()} candidate patches", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def validate_processed_dataset(
    patch_labels: pd.DataFrame,
    output_dir: Path,
    config: PatchConfig,
) -> None:
    """Validate saved patches, labels, splits and class assignments."""
    if patch_labels.empty:
        raise ValueError("processed patch dataset is empty")
    if set(patch_labels["label"].unique()) != set(CLASSES):
        raise ValueError("processed dataset must contain both correct and false classes")

    for _, row in patch_labels.iterrows():
        patch_path = output_dir / str(row["patch_path"])
        if not patch_path.exists():
            raise FileNotFoundError(f"patch label points to missing file: {patch_path}")
        patch = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
        if patch is None:
            raise ValueError(f"saved patch is unreadable: {patch_path}")
        if patch.shape != (config.patch_size, config.patch_size, 3):
            raise ValueError(f"patch has wrong shape {patch.shape}: {patch_path}")

        distance = float(row["distance_to_ground_truth"])
        if row["label"] == "correct" and distance > config.match_tolerance:
            raise ValueError(f"correct patch outside tolerance: {patch_path}")
        if row["label"] == "false":
            target_absent = int(row["target_x"]) == -1 and int(row["target_y"]) == -1
            if not target_absent and distance <= config.match_tolerance:
                raise ValueError(f"false patch is inside tolerance: {patch_path}")

    split_counts = patch_labels.groupby("source_filename")["split"].nunique()
    mixed_source_frames = split_counts[split_counts > 1]
    if not mixed_source_frames.empty:
        raise ValueError(f"source frames appear in multiple splits: {mixed_source_frames.index[:5].tolist()}")

    total_by_split = int(patch_labels.groupby("split").size().sum())
    if total_by_split != len(patch_labels):
        raise ValueError("split counts do not sum to total patch count")


def summarize_dataset(patch_labels: pd.DataFrame, precheck: Dict[str, object], config: PatchConfig) -> Dict[str, object]:
    """Build class, split and scenario summary data."""
    correct_count = int((patch_labels["label"] == "correct").sum())
    false_count = int((patch_labels["label"] == "false").sum())
    summary = {
        "total_patches": int(len(patch_labels)),
        "correct_patches": correct_count,
        "false_patches": false_count,
        "positive_to_negative_ratio": round(correct_count / false_count, 6) if false_count else None,
        "counts_per_split": nested_counts(patch_labels, ["split", "label"]),
        "counts_per_scenario_type": nested_counts(patch_labels, ["scenario_type", "label"]),
        "config": {
            "patch_size": config.patch_size,
            "crop_size": config.crop_size,
            "match_tolerance": config.match_tolerance,
            "seed": config.seed,
            "max_negative_ratio": config.max_negative_ratio,
        },
        "candidate_distribution_precheck": precheck,
    }
    return summary


def nested_counts(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, int]]:
    """Return nested count dictionaries for summary JSON."""
    counts = df.groupby(columns).size()
    nested: Dict[str, Dict[str, int]] = {}
    for keys, value in counts.items():
        outer, inner = keys
        nested.setdefault(str(outer), {})[str(inner)] = int(value)
    return nested


def write_summary(summary: Dict[str, object], output_path: Path) -> None:
    """Write dataset summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_patch_preparation(
    images: str | Path,
    labels_path: str | Path,
    candidates_path: str | Path,
    output: str | Path,
    config: PatchConfig,
    overwrite: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Prepare candidate patches and write all outputs."""
    image_root, labels, candidates = load_inputs(images, labels_path, candidates_path)
    precheck = print_candidate_distribution_precheck(labels, candidates)
    patch_table = build_patch_table(labels, candidates, config)
    false_count = int((patch_table["label"] == "false").sum())
    if bool(precheck["all_frames_have_one_candidate"]) or false_count == 0:
        explain_insufficient_negatives()
        raise SystemExit("Insufficient negative candidates for CNN training.")

    output_dir = Path(output)
    preview_dir = Path("outputs") / "patch-preview"
    safe_prepare_outputs(output_dir, preview_dir, overwrite)

    patch_table = limit_negative_ratio(patch_table, config.max_negative_ratio, config.seed)
    patch_table = assign_data_splits(patch_table, config.seed)
    patch_labels = save_patches(image_root, patch_table, output_dir, config.patch_size, config.crop_size)
    patch_labels.to_csv(output_dir / "patch_labels.csv", index=False)
    save_preview_grids(patch_labels, output_dir, preview_dir)
    validate_processed_dataset(patch_labels, output_dir, config)
    summary = summarize_dataset(patch_labels, precheck, config)
    write_summary(summary, output_dir / "dataset_summary.json")
    print_summary(summary)
    return patch_labels, summary


def print_summary(summary: Dict[str, object]) -> None:
    """Print a concise readable summary."""
    print("\nPatch dataset prepared")
    print(f"  Total patches: {summary['total_patches']}")
    print(f"  Correct patches: {summary['correct_patches']}")
    print(f"  False patches: {summary['false_patches']}")
    print(f"  Positive-to-negative ratio: {summary['positive_to_negative_ratio']}")
    print("  Counts per split:")
    for split, counts in summary["counts_per_split"].items():
        print(f"    {split}: {counts}")
    print("  Counts per scenario type:")
    for scenario, counts in summary["counts_per_scenario_type"].items():
        print(f"    {scenario}: {counts}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare 32x32 CNN candidate patches.")
    parser.add_argument("--images", default="data/raw/python-generated")
    parser.add_argument("--labels", default="data/labels/labels.csv")
    parser.add_argument("--candidates", default="outputs/candidate-detection/candidates.csv")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=40)
    parser.add_argument("--match-tolerance", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-negative-ratio", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    config = PatchConfig(
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        match_tolerance=args.match_tolerance,
        seed=args.seed,
        max_negative_ratio=args.max_negative_ratio,
    )
    try:
        run_patch_preparation(
            images=args.images,
            labels_path=args.labels,
            candidates_path=args.candidates,
            output=args.output,
            config=config,
            overwrite=args.overwrite,
        )
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Patch preparation failed: {exc}") from exc


if __name__ == "__main__":
    main()
