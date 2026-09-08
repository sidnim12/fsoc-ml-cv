from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.unity_dataset_adapter import (
    build_and_write_manifest,
    discover_unity_sequence,
    has_ground_truth,
    missing_label_image_references,
    normalize_labels,
    validate_unity_sequence,
    write_required_labels_template,
)


def write_image(path: Path, width: int = 64, height: int = 48, value: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (value * np.ones((height, width, 3))).astype("uint8")
    cv2.circle(image, (width // 2, height // 2), 3, (255, 255, 255), -1)
    assert cv2.imwrite(str(path), image)


def test_image_discovery_sorting_and_png_jpg_support(tmp_path: Path) -> None:
    sequence = tmp_path / "unity" / "smooth_horizontal01" / "sequence_001"
    write_image(sequence / "frame_000002.jpg")
    write_image(sequence / "frame_000000.png")
    write_image(sequence / "frame_000001.png")
    (sequence / "labels.csv").write_text("frame_id,image_path,target_present,cx_px,cy_px\n0,frame_000000.png,1,10,20\n", encoding="utf-8")

    inventory = discover_unity_sequence(tmp_path / "unity")

    assert inventory.sequence_dir == sequence
    assert [frame.frame_index for frame in inventory.frames] == [0, 1, 2]
    assert inventory.image_count == 3
    assert inventory.image_extension == ".jpg,.png"
    assert inventory.labels_path == sequence / "labels.csv"


def test_missing_frame_number_detection(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    write_image(sequence / "frame_000002.png")

    inventory = discover_unity_sequence(tmp_path)
    validation = validate_unity_sequence(inventory, expected_count=2, expected_width=64, expected_height=48)

    assert validation["frame_numbers_continuous"] is False
    assert validation["missing_frame_numbers"] == [1]
    assert validation["usable"] is False


def test_duplicate_frame_number_detection(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000001.png")
    write_image(sequence / "copy_000001.png")

    inventory = discover_unity_sequence(tmp_path)
    validation = validate_unity_sequence(inventory, expected_count=2, expected_width=64, expected_height=48)

    assert validation["frame_numbers_unique"] is False
    assert validation["duplicate_frame_numbers"] == [1]


def test_resolution_validation(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png", width=64, height=48)
    write_image(sequence / "frame_000001.png", width=32, height=48)

    inventory = discover_unity_sequence(tmp_path)
    validation = validate_unity_sequence(inventory, expected_count=2, expected_width=64, expected_height=48)

    assert validation["all_images_expected_resolution"] is False
    assert validation["resolution_mismatches"][0]["filename"] == "frame_000001.png"


def test_csv_label_loading_and_alternative_column_names(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "labels.csv"
    labels.write_text("frame_id,filename,target_present,cx_px,cy_px\n0,frame_000000.png,1,12.5,30.0\n", encoding="utf-8")

    rows = normalize_labels(labels, sequence, "top-left", default_width=64, default_height=48)

    assert rows[0]["frame_index"] == 0
    assert rows[0]["target_visible"] is True
    assert rows[0]["target_x"] == 12.5
    assert rows[0]["target_y"] == 30.0


def test_xlsx_label_loading(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "metadata.xlsx"
    pd.DataFrame([{"frame_id": 0, "filename": "frame_000000.png", "target_present": 1, "cx_px": 12, "cy_px": 30}]).to_excel(labels, index=False)

    rows = normalize_labels(labels, sequence, "top-left", default_width=64, default_height=48)

    assert rows[0]["frame_index"] == 0
    assert rows[0]["target_visible"] is True
    assert rows[0]["target_x"] == 12


def test_top_left_coordinate_handling(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "labels.csv"
    labels.write_text("frame_id,filename,target_present,cx_px,cy_px\n0,frame_000000.png,1,12,30\n", encoding="utf-8")

    rows = normalize_labels(labels, sequence, "top-left", default_width=64, default_height=48)

    assert rows[0]["target_y"] == 30


def test_bottom_left_coordinate_conversion_and_bbox(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "labels.csv"
    labels.write_text(
        "frame_id,filename,target_present,cx_px,cy_px,bbox_ymin,bbox_ymax\n"
        "0,frame_000000.png,1,12,10,10,20\n",
        encoding="utf-8",
    )

    rows = normalize_labels(labels, sequence, "bottom-left", default_width=64, default_height=48)

    assert rows[0]["target_y"] == 38
    assert rows[0]["bbox_ymin"] == 28
    assert rows[0]["bbox_ymax"] == 38


def test_missing_label_template(tmp_path: Path) -> None:
    sequence = tmp_path / "smooth_horizontal01" / "sequence_001"
    write_image(sequence / "frame_000000.png")
    inventory = discover_unity_sequence(tmp_path)
    template = tmp_path / "required_labels_template.csv"

    write_required_labels_template(inventory, template, fps=30)

    lines = template.read_text(encoding="utf-8").splitlines()
    assert "target_x" in lines[0]
    assert len(lines) == 2


def test_missing_image_reference_detection(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "labels.csv"
    labels.write_text("frame_id,filename,target_present,cx_px,cy_px\n0,missing.png,1,12,30\n", encoding="utf-8")

    rows = normalize_labels(labels, sequence, "top-left", default_width=64, default_height=48)

    assert missing_label_image_references(rows)


def test_manifest_generation_references_original_images(tmp_path: Path) -> None:
    sequence = tmp_path / "smooth_horizontal01" / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "labels.csv"
    labels.write_text("sequence_id,frame_id,image_path,scenario,target_present,target_id,target_x,target_y\n1,0,frame_000000.png,smooth_horizontal,1,beacon,20,30\n", encoding="utf-8")
    inventory = discover_unity_sequence(tmp_path)
    normalized = normalize_labels(labels, sequence, "top-left", default_width=64, default_height=48, fps=30)
    manifest = tmp_path / "manifest.csv"

    rows = build_and_write_manifest(inventory, normalized, manifest, fps=30)

    assert manifest.exists()
    assert rows[0]["scenario"] == "smooth_horizontal"
    assert rows[0]["image_path"].endswith("frame_000000.png")
    assert Path(str(rows[0]["image_path"])).exists()


def test_prediction_columns_are_not_treated_as_ground_truth(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence_001"
    write_image(sequence / "frame_000000.png")
    labels = sequence / "labels.csv"
    labels.write_text("frame_id,filename,target_present,predicted_x_px,predicted_y_px\n0,frame_000000.png,1,12,30\n", encoding="utf-8")

    rows = normalize_labels(labels, sequence, "top-left", default_width=64, default_height=48)

    assert rows[0]["target_x"] is None
    assert rows[0]["target_y"] is None
    assert has_ground_truth(rows) is False
