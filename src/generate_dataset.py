from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_ID = "Terminal_B"
DEFAULT_NUM_IMAGES = 500
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_SEED = 42
LABEL_COLUMNS = [
    "filename",
    "scenario_id",
    "scenario_type",
    "target_visible",
    "target_x",
    "target_y",
    "target_radius",
    "target_id",
    "beacon_intensity",
    "decoy_count",
    "star_count",
    "noise_level",
    "blur_kernel",
    "width",
    "height",
    "random_seed",
]


@dataclass(frozen=True)
class ScenarioSpec:
    """Configuration for one synthetic-data scenario."""

    scenario_id: int
    scenario_type: str
    base_count: int


SCENARIOS = [
    ScenarioSpec(1, "clean_target", 100),
    ScenarioSpec(2, "target_with_stars", 100),
    ScenarioSpec(3, "varied_target_size_brightness", 100),
    ScenarioSpec(4, "sensor_noise_and_blur", 75),
    ScenarioSpec(5, "target_with_false_beacons", 75),
    ScenarioSpec(6, "target_not_visible", 50),
]


@dataclass
class FrameLabel:
    """CSV label information for one generated image."""

    filename: str
    scenario_id: int
    scenario_type: str
    target_visible: int
    target_x: int
    target_y: int
    target_radius: int
    target_id: str
    beacon_intensity: int
    decoy_count: int
    star_count: int
    noise_level: float
    blur_kernel: int
    width: int
    height: int
    random_seed: int


def project_root() -> Path:
    """Return the repository root from this script location."""
    return Path(__file__).resolve().parents[1]


def scenario_counts(num_images: int) -> Dict[int, int]:
    """Scale the 500-image scenario distribution to any requested size."""
    if num_images <= 0:
        raise ValueError("--num-images must be greater than 0")

    total_base = sum(scenario.base_count for scenario in SCENARIOS)
    raw_counts = [
        (scenario.scenario_id, num_images * scenario.base_count / total_base)
        for scenario in SCENARIOS
    ]
    counts = {scenario_id: int(np.floor(raw)) for scenario_id, raw in raw_counts}
    remaining = num_images - sum(counts.values())

    # Give leftover images to the largest fractional parts for exact totals.
    remainders = sorted(
        ((raw - np.floor(raw), scenario_id) for scenario_id, raw in raw_counts),
        reverse=True,
    )
    for _, scenario_id in remainders[:remaining]:
        counts[scenario_id] += 1
    return counts


def create_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Create a dark RGB space-like background."""
    base = rng.normal(loc=8, scale=4, size=(height, width, 3))
    return np.clip(base, 0, 28).astype(np.uint8)


def draw_stars(
    image: np.ndarray,
    rng: np.random.Generator,
    count: int,
    max_radius: int = 2,
) -> None:
    """Draw small random stars that are not target labels."""
    height, width = image.shape[:2]
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, max_radius + 1))
        brightness = int(rng.integers(95, 220))
        cv2.circle(image, (x, y), radius, (brightness, brightness, brightness), -1, lineType=cv2.LINE_AA)


def draw_gaussian_beacon(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    intensity: int,
    glow_multiplier: float = 2.8,
) -> None:
    """Draw a bright circular beacon with a soft Gaussian glow."""
    height, width = image.shape[:2]
    x0, y0 = center
    glow_radius = max(radius + 2, int(round(radius * glow_multiplier)))
    x_min = max(0, x0 - glow_radius)
    x_max = min(width - 1, x0 + glow_radius)
    y_min = max(0, y0 - glow_radius)
    y_max = min(height - 1, y0 + glow_radius)

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    distance_sq = (xx - x0) ** 2 + (yy - y0) ** 2
    sigma = max(radius * 0.9, 1.0)
    glow = intensity * np.exp(-distance_sq / (2 * sigma**2))
    glow = np.clip(glow, 0, 255).astype(np.float32)

    patch = image[y_min : y_max + 1, x_min : x_max + 1].astype(np.float32)
    patch = np.maximum(patch, glow[..., None])
    image[y_min : y_max + 1, x_min : x_max + 1] = np.clip(patch, 0, 255).astype(np.uint8)
    cv2.circle(image, center, max(1, radius // 2), (intensity, intensity, intensity), -1, lineType=cv2.LINE_AA)


def overlaps_target(
    point: Tuple[int, int],
    radius: int,
    target: Optional[Tuple[int, int, int]],
    margin: int = 8,
) -> bool:
    """Return True when a decoy would overlap the labelled target."""
    if target is None:
        return False
    tx, ty, tr = target
    distance = float(np.hypot(point[0] - tx, point[1] - ty))
    return distance <= radius + tr + margin


def draw_false_beacons(
    image: np.ndarray,
    rng: np.random.Generator,
    count: int,
    target: Optional[Tuple[int, int, int]],
) -> int:
    """Draw target-like decoys and return how many were placed."""
    height, width = image.shape[:2]
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 40:
        attempts += 1
        radius = int(rng.integers(3, 8))
        x = int(rng.integers(radius + 2, width - radius - 2))
        y = int(rng.integers(radius + 2, height - radius - 2))
        if overlaps_target((x, y), radius, target):
            continue
        intensity = int(rng.integers(175, 240))
        draw_gaussian_beacon(image, (x, y), radius, intensity, glow_multiplier=2.2)
        placed += 1
    return placed


def apply_sensor_effects(
    image: np.ndarray,
    rng: np.random.Generator,
    noise_level: float,
    blur_kernel: int,
) -> np.ndarray:
    """Apply Gaussian sensor noise and optional blur."""
    output = image.astype(np.float32)
    if noise_level > 0:
        output += rng.normal(0.0, noise_level, size=output.shape)
    output = np.clip(output, 0, 255).astype(np.uint8)

    if blur_kernel > 1:
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
    return output


def random_target(
    width: int,
    height: int,
    rng: np.random.Generator,
    radius: Optional[int] = None,
    intensity: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Create target centre, radius and intensity values."""
    target_radius = radius if radius is not None else int(rng.integers(4, 11))
    margin = target_radius * 3 + 2
    x = int(rng.integers(margin, width - margin))
    y = int(rng.integers(margin, height - margin))
    beacon_intensity = intensity if intensity is not None else int(rng.integers(210, 256))
    return x, y, target_radius, beacon_intensity


def generate_frame(
    scenario: ScenarioSpec,
    index: int,
    width: int,
    height: int,
    seed: int,
) -> Tuple[np.ndarray, FrameLabel]:
    """Generate one synthetic image and its label row."""
    frame_seed = seed + index
    rng = np.random.default_rng(frame_seed)
    image = create_background(width, height, rng)

    target_visible = scenario.scenario_id != 6
    star_count = 0
    decoy_count = 0
    noise_level = 0.0
    blur_kernel = 0
    target_x = -1
    target_y = -1
    target_radius = 0
    beacon_intensity = 0
    target_tuple: Optional[Tuple[int, int, int]] = None

    if scenario.scenario_id == 2:
        star_count = int(rng.integers(45, 130))
    elif scenario.scenario_id == 4:
        star_count = int(rng.integers(20, 75))
        noise_level = float(rng.uniform(8.0, 22.0))
        blur_kernel = int(rng.choice([3, 5, 7]))
    elif scenario.scenario_id == 5:
        star_count = int(rng.integers(15, 60))
    elif scenario.scenario_id == 6:
        star_count = int(rng.integers(35, 120))

    if star_count:
        draw_stars(image, rng, star_count)

    if target_visible:
        if scenario.scenario_id == 3:
            radius = int(rng.integers(4, 11))
            intensity = int(rng.integers(210, 256))
        else:
            radius = int(rng.integers(5, 9))
            intensity = int(rng.integers(225, 256))
        target_x, target_y, target_radius, beacon_intensity = random_target(
            width, height, rng, radius=radius, intensity=intensity
        )
        target_tuple = (target_x, target_y, target_radius)
        draw_gaussian_beacon(image, (target_x, target_y), target_radius, beacon_intensity)

    if scenario.scenario_id == 5:
        decoy_count = draw_false_beacons(image, rng, int(rng.integers(2, 7)), target_tuple)
    elif scenario.scenario_id == 6:
        decoy_count = draw_false_beacons(image, rng, int(rng.integers(1, 5)), None)

    image = apply_sensor_effects(image, rng, noise_level=noise_level, blur_kernel=blur_kernel)
    filename = f"synthetic_{index:04d}_{scenario.scenario_type}.png"
    label = FrameLabel(
        filename=filename,
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        target_visible=int(target_visible),
        target_x=target_x,
        target_y=target_y,
        target_radius=target_radius,
        target_id=TARGET_ID,
        beacon_intensity=beacon_intensity,
        decoy_count=decoy_count,
        star_count=star_count,
        noise_level=round(noise_level, 3),
        blur_kernel=blur_kernel,
        width=width,
        height=height,
        random_seed=frame_seed,
    )
    return image, label


def safe_prepare_outputs(
    image_dir: Path,
    labels_path: Path,
    preview_dir: Path,
    overwrite: bool,
) -> None:
    """Create output folders and protect existing generated PNG files."""
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    existing_images = sorted(image_dir.glob("*.png"))
    preview_path = preview_dir / "sample_grid.png"
    if not overwrite and existing_images:
        raise FileExistsError(
            f"{image_dir} already contains {len(existing_images)} PNG files. "
            "Use --overwrite to replace the synthetic dataset outputs."
        )

    if overwrite:
        for png_path in existing_images:
            png_path.unlink()
        if labels_path.exists():
            labels_path.unlink()
        if preview_path.exists():
            preview_path.unlink()


def generate_dataset(
    num_images: int,
    width: int,
    height: int,
    seed: int,
    overwrite: bool,
) -> pd.DataFrame:
    """Generate images, labels CSV and preview image."""
    if width < 80 or height < 80:
        raise ValueError("width and height must be at least 80 pixels")

    root = project_root()
    image_dir = root / "data" / "raw" / "python-generated"
    labels_path = root / "data" / "labels" / "labels.csv"
    preview_dir = root / "outputs" / "dataset-preview"
    safe_prepare_outputs(image_dir, labels_path, preview_dir, overwrite)

    counts = scenario_counts(num_images)
    labels: List[FrameLabel] = []
    image_index = 0
    for scenario in SCENARIOS:
        for _ in range(counts[scenario.scenario_id]):
            image, label = generate_frame(scenario, image_index, width, height, seed)
            output_path = image_dir / label.filename
            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"failed to write image: {output_path}")
            labels.append(label)
            image_index += 1

    df = pd.DataFrame([label.__dict__ for label in labels], columns=LABEL_COLUMNS)
    df.to_csv(labels_path, index=False)
    validate_dataset(image_dir, labels_path, num_images, width, height)
    save_preview_grid(image_dir, df, preview_dir / "sample_grid.png")
    return df


def validate_dataset(
    image_dir: Path,
    labels_path: Path,
    expected_count: int,
    width: int,
    height: int,
) -> Dict[str, Dict[str, int]]:
    """Validate generated image files and labels."""
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels file: {labels_path}")

    image_paths = sorted(image_dir.glob("*.png"))
    if len(image_paths) != expected_count:
        raise ValueError(f"expected {expected_count} PNG images, found {len(image_paths)}")

    df = pd.read_csv(labels_path)
    missing_columns = [column for column in LABEL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"labels CSV missing columns: {missing_columns}")
    if len(df) != expected_count:
        raise ValueError(f"expected {expected_count} CSV rows, found {len(df)}")

    image_names = {path.name for path in image_paths}
    for _, row in df.iterrows():
        filename = str(row["filename"])
        if filename not in image_names:
            raise ValueError(f"CSV filename does not exist: {filename}")

        visible = int(row["target_visible"])
        x = int(row["target_x"])
        y = int(row["target_y"])
        radius = int(row["target_radius"])
        if visible:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"visible target has invalid coordinates in {filename}")
            if not (4 <= radius <= 10):
                raise ValueError(f"visible target has invalid radius in {filename}: {radius}")
        else:
            if (x, y) != (-1, -1) or radius != 0:
                raise ValueError(f"invisible target has invalid label values in {filename}")

    scenario_counts_found = df["scenario_type"].value_counts().sort_index().to_dict()
    scenario_total = int(sum(scenario_counts_found.values()))
    if scenario_total != expected_count:
        raise ValueError("scenario counts do not add up to requested image count")

    visible_counts = df["target_visible"].value_counts().sort_index().to_dict()
    return {
        "scenario_counts": {str(k): int(v) for k, v in scenario_counts_found.items()},
        "target_visible_counts": {str(k): int(v) for k, v in visible_counts.items()},
    }


def representative_samples(df: pd.DataFrame, sample_count: int = 12) -> pd.DataFrame:
    """Pick a small set of labels covering all scenarios where possible."""
    samples = []
    for _, group in df.groupby("scenario_id", sort=True):
        per_scenario = max(1, sample_count // max(1, df["scenario_id"].nunique()))
        for _, row in group.head(per_scenario).iterrows():
            samples.append(row)
    remaining = sample_count - len(samples)
    if remaining > 0:
        used = {sample["filename"] for sample in samples}
        extra = df[~df["filename"].isin(used)].head(remaining)
        samples.extend(row for _, row in extra.iterrows())
    return pd.DataFrame(samples[:sample_count])


def save_preview_grid(image_dir: Path, labels: pd.DataFrame, output_path: Path) -> None:
    """Create a labelled 4x3 preview grid with ground-truth overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = representative_samples(labels, sample_count=12)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes_flat = axes.flatten()

    for axis, (_, row) in zip(axes_flat, samples.iterrows()):
        image_path = image_dir / str(row["filename"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read preview sample: {image_path}")

        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if int(row["target_visible"]):
            center = (int(row["target_x"]), int(row["target_y"]))
            radius = int(row["target_radius"])
            cv2.circle(preview, center, radius + 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.drawMarker(preview, center, (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            title = f"S{row['scenario_id']}: ({center[0]}, {center[1]})"
        else:
            title = f"S{row['scenario_id']}: Target absent"

        axis.imshow(preview)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    for axis in axes_flat[len(samples) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    """Print a readable generation summary."""
    print("\nSynthetic FSOC dataset generated")
    print(f"Total images: {len(df)}")
    print("\nScenario counts:")
    for scenario in SCENARIOS:
        count = int((df["scenario_id"] == scenario.scenario_id).sum())
        print(f"  S{scenario.scenario_id} {scenario.scenario_type}: {count}")
    print("\nTarget-visible counts:")
    for visible, count in df["target_visible"].value_counts().sort_index().items():
        label = "visible" if int(visible) == 1 else "absent"
        print(f"  {label}: {count}")
    print("\nOutputs:")
    root = project_root()
    print(f"  Images: {root / 'data' / 'raw' / 'python-generated'}")
    print(f"  Labels: {root / 'data' / 'labels' / 'labels.csv'}")
    print(f"  Preview: {root / 'outputs' / 'dataset-preview' / 'sample_grid.png'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Generate Phase 1 synthetic FSOC beacon images.")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    try:
        df = generate_dataset(
            num_images=args.num_images,
            width=args.width,
            height=args.height,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print_summary(df)
    except Exception as exc:
        raise SystemExit(f"Dataset generation failed: {exc}") from exc


if __name__ == "__main__":
    main()
