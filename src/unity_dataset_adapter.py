from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LABEL_FILENAMES = {"labels.csv", "labels.xlsx", "metadata.csv", "metadata.xlsx"}
COORDINATE_ORIGINS = {"top-left", "bottom-left"}

CANONICAL_LABEL_COLUMNS = [
    "sequence_id",
    "frame_index",
    "timestamp_s",
    "image_path",
    "scenario",
    "width",
    "height",
    "target_visible",
    "target_id",
    "target_x",
    "target_y",
]

OPTIONAL_LABEL_COLUMNS = [
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "beacon_radius",
    "beacon_on",
    "faded",
    "occluded",
    "dropped_frame",
    "false_candidate_count",
    "jitter",
    "camera_pan",
    "camera_tilt",
    "target_world_x",
    "target_world_y",
    "target_world_z",
]

MANIFEST_COLUMNS = CANONICAL_LABEL_COLUMNS + OPTIONAL_LABEL_COLUMNS

LABEL_ALIASES = {
    "sequence_id": ["sequence_id", "seq_id", "sequence", "sequence_number"],
    "frame_index": ["frame_index", "frame_id", "frame", "index"],
    "timestamp_s": ["timestamp_s", "time_s", "timestamp", "time"],
    "image_path": ["image_path", "filename", "file_name", "file", "image", "frame_path"],
    "scenario": ["scenario", "scenario_name"],
    "width": ["width", "image_width", "frame_width"],
    "height": ["height", "image_height", "frame_height"],
    "target_visible": ["target_visible", "target_present", "visible", "target_found"],
    "target_id": ["target_id", "beacon_id", "id"],
    "target_x": ["target_x", "cx_px", "center_x", "x_px", "x"],
    "target_y": ["target_y", "cy_px", "center_y", "y_px", "y"],
    "bbox_xmin": ["bbox_xmin", "bbox_left", "xmin", "x_min"],
    "bbox_ymin": ["bbox_ymin", "bbox_bottom", "ymin", "y_min"],
    "bbox_xmax": ["bbox_xmax", "bbox_right", "xmax", "x_max"],
    "bbox_ymax": ["bbox_ymax", "bbox_top", "ymax", "y_max"],
    "beacon_radius": ["beacon_radius", "target_radius", "radius"],
    "beacon_on": ["beacon_on", "is_on"],
    "faded": ["faded", "is_faded"],
    "occluded": ["occluded", "is_occluded"],
    "dropped_frame": ["dropped_frame", "dropped"],
    "false_candidate_count": ["false_candidate_count", "false_beacons", "decoy_count"],
    "jitter": ["jitter", "jitter_level"],
    "camera_pan": ["camera_pan", "pan"],
    "camera_tilt": ["camera_tilt", "tilt"],
    "target_world_x": ["target_world_x", "world_x"],
    "target_world_y": ["target_world_y", "world_y"],
    "target_world_z": ["target_world_z", "world_z"],
}


@dataclass(frozen=True)
class UnityImageFrame:
    """One image in a Unity frame sequence."""

    path: Path
    filename: str
    frame_index: int
    extension: str
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    readable: bool = False


@dataclass(frozen=True)
class UnitySequenceInventory:
    """Discovered Unity sequence folder and basic frame inventory."""

    sequence_dir: Path
    images_dir: Path
    labels_path: Optional[Path]
    frames: List[UnityImageFrame]
    filename_pattern: str
    first_frame: Optional[int]
    last_frame: Optional[int]
    continuous_frame_numbers: bool
    duplicate_frame_numbers: List[int]
    missing_frame_numbers: List[int]
    unreadable_sample: bool

    @property
    def image_count(self) -> int:
        return len(self.frames)

    @property
    def image_extension(self) -> Optional[str]:
        if not self.frames:
            return None
        extensions = sorted({frame.extension for frame in self.frames})
        return extensions[0] if len(extensions) == 1 else ",".join(extensions)

    @property
    def width(self) -> Optional[int]:
        return self.frames[0].width if self.frames else None

    @property
    def height(self) -> Optional[int]:
        return self.frames[0].height if self.frames else None

    @property
    def channels(self) -> Optional[int]:
        return self.frames[0].channels if self.frames else None

    def to_dict(self) -> Dict[str, object]:
        """Return a compact JSON-friendly inventory."""
        return {
            "sequence_dir": str(self.sequence_dir),
            "images_dir": str(self.images_dir),
            "labels_path": str(self.labels_path) if self.labels_path else None,
            "image_count": self.image_count,
            "image_extension": self.image_extension,
            "filename_pattern": self.filename_pattern,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "continuous_frame_numbers": self.continuous_frame_numbers,
            "duplicate_frame_numbers": self.duplicate_frame_numbers,
            "missing_frame_numbers": self.missing_frame_numbers,
            "unreadable_sample": self.unreadable_sample,
        }


def normalize_column_name(name: object) -> str:
    """Normalize labels like 'Frame ID' and 'frame_id' to the same lookup key."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


NORMALIZED_ALIASES = {
    canonical: {normalize_column_name(alias) for alias in aliases}
    for canonical, aliases in LABEL_ALIASES.items()
}


def parse_frame_number(path: str | Path) -> Optional[int]:
    """Extract the last numeric group from a frame filename."""
    stem = Path(path).stem
    matches = re.findall(r"(\d+)", stem)
    if not matches:
        return None
    return int(matches[-1])


def detect_filename_pattern(path: Path) -> str:
    """Return a readable pattern such as frame_{frame:06d}.png."""
    stem = path.stem
    matches = list(re.finditer(r"(\d+)", stem))
    if not matches:
        return path.name
    match = matches[-1]
    width = len(match.group(1))
    return f"{stem[:match.start()]}{{frame:0{width}d}}{stem[match.end():]}{path.suffix.lower()}"


def sorted_image_paths(paths: Iterable[Path]) -> List[Path]:
    """Sort image paths by parsed frame number, then by filename."""
    return sorted(paths, key=lambda path: (parse_frame_number(path) is None, parse_frame_number(path) or 0, path.name))


def find_label_file(sequence_dir: Path) -> Optional[Path]:
    """Find a labels/metadata file near a sequence directory."""
    candidates = []
    for directory in [sequence_dir, sequence_dir.parent]:
        for filename in LABEL_FILENAMES:
            path = directory / filename
            if path.exists() and path.is_file():
                candidates.append(path)
    if candidates:
        return sorted(candidates)[0]

    for path in sequence_dir.rglob("*"):
        if path.is_file() and path.name.lower() in LABEL_FILENAMES:
            return path
    return None


def inspect_image(path: Path) -> UnityImageFrame:
    """Read one image enough to know size, channels and readability."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    frame_index = parse_frame_number(path)
    if image is None or image.size == 0:
        return UnityImageFrame(
            path=path,
            filename=path.name,
            frame_index=-1 if frame_index is None else frame_index,
            extension=path.suffix.lower(),
            readable=False,
        )
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return UnityImageFrame(
        path=path,
        filename=path.name,
        frame_index=-1 if frame_index is None else frame_index,
        extension=path.suffix.lower(),
        width=int(width),
        height=int(height),
        channels=channels,
        readable=True,
    )


def discover_unity_sequences(root: str | Path) -> List[UnitySequenceInventory]:
    """Find directories under root that contain Unity image frames."""
    search_root = Path(root)
    if not search_root.exists():
        raise FileNotFoundError(f"Unity data root does not exist: {search_root}")

    image_paths = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"no Unity image frames found under {search_root}")

    grouped: Dict[Path, List[Path]] = {}
    for path in image_paths:
        grouped.setdefault(path.parent, []).append(path)

    inventories = []
    for image_dir, paths in grouped.items():
        ordered_paths = sorted_image_paths(paths)
        frames = [inspect_image(path) for path in ordered_paths]
        numbers = [parse_frame_number(path) for path in ordered_paths]
        numeric_numbers = [int(number) for number in numbers if number is not None]
        duplicates = duplicate_numbers(numeric_numbers)
        missing = missing_numbers(numeric_numbers)
        continuous = bool(numeric_numbers) and not duplicates and not missing and len(numeric_numbers) == len(ordered_paths)
        first_frame = min(numeric_numbers) if numeric_numbers else None
        last_frame = max(numeric_numbers) if numeric_numbers else None
        inventories.append(
            UnitySequenceInventory(
                sequence_dir=image_dir,
                images_dir=image_dir,
                labels_path=find_label_file(image_dir),
                frames=frames,
                filename_pattern=detect_filename_pattern(ordered_paths[0]),
                first_frame=first_frame,
                last_frame=last_frame,
                continuous_frame_numbers=continuous,
                duplicate_frame_numbers=duplicates,
                missing_frame_numbers=missing,
                unreadable_sample=not bool(frames[0].readable),
            )
        )

    return sorted(inventories, key=lambda inventory: inventory.image_count, reverse=True)


def discover_unity_sequence(root: str | Path) -> UnitySequenceInventory:
    """Return the largest discovered Unity sequence under root."""
    return discover_unity_sequences(root)[0]


def duplicate_numbers(numbers: Sequence[int]) -> List[int]:
    """Return duplicated frame numbers."""
    seen = set()
    duplicates = set()
    for number in numbers:
        if number in seen:
            duplicates.add(number)
        seen.add(number)
    return sorted(duplicates)


def missing_numbers(numbers: Sequence[int]) -> List[int]:
    """Return missing integers between the first and last frame number."""
    if not numbers:
        return []
    expected = set(range(min(numbers), max(numbers) + 1))
    return sorted(expected - set(numbers))


def sha256_file(path: Path) -> str:
    """Hash a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unity_sequence(
    inventory: UnitySequenceInventory,
    expected_count: int = 300,
    expected_width: int = 640,
    expected_height: int = 480,
) -> Dict[str, object]:
    """Validate image readability, resolution, frame order and duplicate files."""
    unreadable = []
    resolution_mismatches = []
    channel_counts: Dict[int, int] = {}
    brightness_values = []
    file_hashes: Dict[str, List[str]] = {}
    frame_numbers = []

    for path in [frame.path for frame in inventory.frames]:
        number = parse_frame_number(path)
        if number is not None:
            frame_numbers.append(number)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            unreadable.append(path.name)
            continue

        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        channel_counts[channels] = channel_counts.get(channels, 0) + 1
        if width != expected_width or height != expected_height:
            resolution_mismatches.append({"filename": path.name, "width": int(width), "height": int(height)})
        brightness_values.append(float(np.mean(image)))
        file_hashes.setdefault(sha256_file(path), []).append(path.name)

    duplicates = duplicate_numbers(frame_numbers)
    missing = missing_numbers(frame_numbers)
    duplicate_hash_groups = [names for names in file_hashes.values() if len(names) > 1]
    exact_resolution = not resolution_mismatches
    count_matches = inventory.image_count == expected_count
    readable_all = not unreadable
    continuous = bool(frame_numbers) and not duplicates and not missing and len(frame_numbers) == inventory.image_count
    all_empty = not brightness_values or max(brightness_values) <= 1.0
    editor_ui_suspected = not exact_resolution

    return {
        "sequence_dir": str(inventory.sequence_dir),
        "labels_path": str(inventory.labels_path) if inventory.labels_path else None,
        "expected_image_count": expected_count,
        "actual_image_count": inventory.image_count,
        "count_matches_expected": count_matches,
        "first_frame": min(frame_numbers) if frame_numbers else None,
        "last_frame": max(frame_numbers) if frame_numbers else None,
        "frame_numbers_unique": not duplicates,
        "duplicate_frame_numbers": duplicates,
        "frame_numbers_continuous": continuous,
        "missing_frame_numbers": missing,
        "all_images_readable": readable_all,
        "unreadable_images": unreadable,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "all_images_expected_resolution": exact_resolution,
        "resolution_mismatches": resolution_mismatches,
        "channel_counts": channel_counts,
        "duplicate_image_hash_groups": duplicate_hash_groups,
        "duplicate_image_file_count": sum(len(group) for group in duplicate_hash_groups),
        "all_frames_empty": all_empty,
        "min_brightness": float(round(min(brightness_values), 6)) if brightness_values else None,
        "max_brightness": float(round(max(brightness_values), 6)) if brightness_values else None,
        "mean_brightness": float(round(sum(brightness_values) / len(brightness_values), 6)) if brightness_values else None,
        "virtual_camera_view_only": exact_resolution,
        "unity_editor_ui_suspected": editor_ui_suspected,
        "appears_shuffled": not continuous,
        "usable": bool(inventory.image_count > 0 and readable_all and exact_resolution and not all_empty and not duplicates and not missing),
        "count_note": "count differs from expected but sequence can still be usable" if not count_matches else "count matches expected",
    }


def print_inventory(inventory: UnitySequenceInventory) -> None:
    """Print a compact sequence inventory before evaluation."""
    print("Unity sequence inventory")
    print(f"  Sequence dir: {inventory.sequence_dir}")
    print(f"  Images dir: {inventory.images_dir}")
    print(f"  Labels: {inventory.labels_path if inventory.labels_path else 'not found'}")
    print(f"  Images: {inventory.image_count} ({inventory.image_extension})")
    print(f"  Pattern: {inventory.filename_pattern}")
    print(f"  Frames: {inventory.first_frame} -> {inventory.last_frame}")
    print(f"  Resolution: {inventory.width}x{inventory.height}, channels={inventory.channels}")
    print(f"  Continuous: {inventory.continuous_frame_numbers}")


def validate_coordinate_origin(coordinate_origin: str) -> str:
    """Require callers to explicitly choose the coordinate origin."""
    if coordinate_origin not in COORDINATE_ORIGINS:
        raise ValueError("coordinate_origin must be 'top-left' or 'bottom-left'")
    return coordinate_origin


def read_label_table(labels_path: str | Path) -> pd.DataFrame:
    """Read CSV or XLSX labels."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"unsupported labels file type: {path.suffix}")


def value_is_missing(value: object) -> bool:
    """Return True for pandas/CSV missing values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def clean_value(value: object) -> object:
    """Convert pandas missing values to None and NumPy scalars to Python scalars."""
    if value_is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def bool_or_none(value: object) -> Optional[bool]:
    """Parse labels that may use 0/1, true/false or yes/no."""
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "no", "n"}:
        return False
    return None


def float_or_none(value: object) -> Optional[float]:
    """Parse a numeric value or return None."""
    value = clean_value(value)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def int_or_none(value: object) -> Optional[int]:
    """Parse an integer value or return None."""
    number = float_or_none(value)
    return None if number is None else int(number)


def canonical_value(row: Mapping[str, object], column_lookup: Mapping[str, str], canonical: str) -> object:
    """Fetch a row value using accepted aliases."""
    for alias in NORMALIZED_ALIASES.get(canonical, {normalize_column_name(canonical)}):
        source = column_lookup.get(alias)
        if source is not None:
            return row.get(source)
    return None


def resolve_label_image_path(image_value: object, labels_path: Path, sequence_dir: Path) -> Optional[Path]:
    """Resolve label image references without changing the original files."""
    image_value = clean_value(image_value)
    if image_value is None or str(image_value).strip() == "":
        return None
    raw = Path(str(image_value))
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [
        labels_path.parent / raw,
        sequence_dir / raw,
        sequence_dir / raw.name,
        labels_path.parent / raw.name,
        labels_path.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_labels(
    labels_path: str | Path,
    sequence_dir: str | Path,
    coordinate_origin: str,
    default_width: Optional[int] = None,
    default_height: Optional[int] = None,
    fps: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Normalize CSV/XLSX Unity labels into OpenCV top-left coordinates."""
    validate_coordinate_origin(coordinate_origin)
    path = Path(labels_path)
    seq_dir = Path(sequence_dir)
    table = read_label_table(path)
    column_lookup = {normalize_column_name(column): str(column) for column in table.columns}
    rows: List[Dict[str, object]] = []

    for row_index, raw_row in enumerate(table.to_dict(orient="records")):
        image_path = resolve_label_image_path(canonical_value(raw_row, column_lookup, "image_path"), path, seq_dir)
        frame_index = int_or_none(canonical_value(raw_row, column_lookup, "frame_index"))
        if frame_index is None and image_path is not None:
            frame_index = parse_frame_number(image_path)
        if frame_index is None:
            frame_index = row_index

        width = int_or_none(canonical_value(raw_row, column_lookup, "width")) or default_width
        height = int_or_none(canonical_value(raw_row, column_lookup, "height")) or default_height
        target_x = float_or_none(canonical_value(raw_row, column_lookup, "target_x"))
        target_y = float_or_none(canonical_value(raw_row, column_lookup, "target_y"))
        target_visible = bool_or_none(canonical_value(raw_row, column_lookup, "target_visible"))
        if target_visible is None and target_x is not None and target_y is not None:
            target_visible = True

        normalized: Dict[str, object] = {
            "sequence_id": clean_value(canonical_value(raw_row, column_lookup, "sequence_id")),
            "frame_index": int(frame_index),
            "timestamp_s": float_or_none(canonical_value(raw_row, column_lookup, "timestamp_s")),
            "image_path": str(image_path) if image_path is not None else "",
            "scenario": clean_value(canonical_value(raw_row, column_lookup, "scenario")),
            "width": width,
            "height": height,
            "target_visible": target_visible,
            "target_id": clean_value(canonical_value(raw_row, column_lookup, "target_id")),
            "target_x": target_x,
            "target_y": target_y,
        }

        for column in OPTIONAL_LABEL_COLUMNS:
            normalized[column] = clean_value(canonical_value(raw_row, column_lookup, column))

        if normalized["timestamp_s"] is None and fps is not None:
            normalized["timestamp_s"] = float(frame_index) / float(fps)

        if coordinate_origin == "bottom-left" and height is not None:
            convert_bottom_left_row_to_top_left(normalized, int(height))

        rows.append(normalized)
    return rows


def convert_bottom_left_row_to_top_left(row: Dict[str, object], image_height: int) -> None:
    """Convert Unity bottom-left y values to OpenCV top-left y values in-place."""
    target_y = float_or_none(row.get("target_y"))
    if target_y is not None:
        row["target_y"] = float(image_height - target_y)

    bbox_ymin = float_or_none(row.get("bbox_ymin"))
    bbox_ymax = float_or_none(row.get("bbox_ymax"))
    if bbox_ymin is not None and bbox_ymax is not None:
        converted_min = image_height - bbox_ymax
        converted_max = image_height - bbox_ymin
        row["bbox_ymin"] = float(min(converted_min, converted_max))
        row["bbox_ymax"] = float(max(converted_min, converted_max))


def labels_by_frame(labels: Sequence[Mapping[str, object]]) -> Dict[int, Dict[str, object]]:
    """Index labels by frame_index."""
    indexed = {}
    for row in labels:
        indexed[int(row["frame_index"])] = dict(row)
    return indexed


def missing_label_image_references(labels: Sequence[Mapping[str, object]]) -> List[str]:
    """Return label image paths that do not exist on disk."""
    missing = []
    for row in labels:
        image_path = clean_value(row.get("image_path"))
        if image_path and not Path(str(image_path)).exists():
            missing.append(str(image_path))
    return missing


def scenario_from_inventory(inventory: UnitySequenceInventory, labels: Sequence[Mapping[str, object]]) -> str:
    """Infer a simple scenario name from labels or folder names."""
    for row in labels:
        scenario = clean_value(row.get("scenario"))
        if scenario:
            return str(scenario)
    parent_name = inventory.sequence_dir.parent.name
    return re.sub(r"[_-]?\d+$", "", parent_name) or parent_name


def sequence_id_from_inventory(inventory: UnitySequenceInventory, labels: Sequence[Mapping[str, object]]) -> int:
    """Infer numeric sequence id from labels or folder names."""
    for row in labels:
        sequence_id = int_or_none(row.get("sequence_id"))
        if sequence_id is not None:
            return sequence_id
    parsed = parse_frame_number(inventory.sequence_dir.name)
    return int(parsed) if parsed is not None else 1


def sequence_slug(inventory: UnitySequenceInventory, labels: Sequence[Mapping[str, object]]) -> str:
    """Return a stable folder name like smooth_horizontal_01."""
    scenario = scenario_from_inventory(inventory, labels)
    sequence_id = sequence_id_from_inventory(inventory, labels)
    scenario_slug = re.sub(r"[^a-z0-9]+", "_", scenario.strip().lower()).strip("_")
    return f"{scenario_slug}_{sequence_id:02d}"


def format_manifest_value(value: object) -> object:
    """Keep CSV cells readable and blank for missing values."""
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def build_manifest_rows(
    inventory: UnitySequenceInventory,
    labels: Sequence[Mapping[str, object]],
    fps: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Build normalized manifest rows that reference original image paths."""
    labels_lookup = labels_by_frame(labels)
    scenario = scenario_from_inventory(inventory, labels)
    sequence_id = sequence_id_from_inventory(inventory, labels)
    rows: List[Dict[str, object]] = []

    for frame in inventory.frames:
        label = labels_lookup.get(frame.frame_index, {})
        timestamp = clean_value(label.get("timestamp_s"))
        if timestamp is None and fps is not None:
            timestamp = float(frame.frame_index) / float(fps)
        row: Dict[str, object] = {
            "sequence_id": clean_value(label.get("sequence_id")) or sequence_id,
            "frame_index": frame.frame_index,
            "timestamp_s": timestamp,
            "image_path": str(frame.path),
            "scenario": clean_value(label.get("scenario")) or scenario,
            "width": frame.width or clean_value(label.get("width")) or "",
            "height": frame.height or clean_value(label.get("height")) or "",
            "target_visible": clean_value(label.get("target_visible")),
            "target_id": clean_value(label.get("target_id")) or "",
            "target_x": clean_value(label.get("target_x")),
            "target_y": clean_value(label.get("target_y")),
        }
        for column in OPTIONAL_LABEL_COLUMNS:
            row[column] = clean_value(label.get(column))
        rows.append({column: format_manifest_value(row.get(column)) for column in MANIFEST_COLUMNS})
    return rows


def write_manifest(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write normalized manifest CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def build_and_write_manifest(
    inventory: UnitySequenceInventory,
    labels: Sequence[Mapping[str, object]],
    output_path: str | Path,
    fps: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Create a normalized manifest and write it to disk."""
    rows = build_manifest_rows(inventory, labels, fps=fps)
    write_manifest(output_path, rows)
    return rows


def write_required_labels_template(
    inventory: UnitySequenceInventory,
    output_path: str | Path,
    fps: Optional[float] = None,
) -> None:
    """Write an empty ground-truth template for Unity developers."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = CANONICAL_LABEL_COLUMNS + ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame in inventory.frames:
            timestamp = float(frame.frame_index) / float(fps) if fps else ""
            writer.writerow(
                {
                    "sequence_id": sequence_id_from_inventory(inventory, []),
                    "frame_index": frame.frame_index,
                    "timestamp_s": timestamp,
                    "image_path": str(frame.path),
                    "scenario": scenario_from_inventory(inventory, []),
                    "width": frame.width or "",
                    "height": frame.height or "",
                    "target_visible": "",
                    "target_id": "",
                    "target_x": "",
                    "target_y": "",
                    "bbox_xmin": "",
                    "bbox_ymin": "",
                    "bbox_xmax": "",
                    "bbox_ymax": "",
                }
            )


def has_ground_truth(labels: Sequence[Mapping[str, object]]) -> bool:
    """Return whether labels contain usable visibility and target coordinates."""
    for row in labels:
        visible = bool_or_none(row.get("target_visible"))
        if visible is False:
            return True
        if visible is True and float_or_none(row.get("target_x")) is not None and float_or_none(row.get("target_y")) is not None:
            return True
    return False
