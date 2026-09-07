from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings for turning camera frames into clean binary images."""

    threshold_method: str = "fixed"
    threshold_value: int = 200
    blur_kernel: int = 5
    morph_kernel: int = 3
    opening_iterations: int = 1
    closing_iterations: int = 1
    normalize_contrast: bool = False


def validate_odd_kernel(kernel_size: int, name: str) -> int:
    """Validate that an OpenCV kernel size is positive and odd."""
    if kernel_size <= 0:
        raise ValueError(f"{name} must be positive, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"{name} must be odd, got {kernel_size}")
    return kernel_size


def load_image(path: str | Path) -> np.ndarray:
    """Load an image with OpenCV and fail clearly for bad paths."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"input is not a file: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"image is unreadable or unsupported: {image_path}")
    if image.size == 0:
        raise ValueError(f"image is empty: {image_path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR input to one-channel uint8 grayscale."""
    if image is None or image.size == 0:
        raise ValueError("image is missing or empty")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"expected grayscale or BGR image, got shape {image.shape}")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if gray.ndim != 2:
        raise ValueError("grayscale image must have exactly one channel")
    return gray


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Improve grayscale contrast using CLAHE."""
    if gray.ndim != 2:
        raise ValueError("normalize_contrast expects a one-channel grayscale image")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray.astype(np.uint8, copy=False))


def apply_gaussian_blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur with a valid odd kernel size."""
    validate_odd_kernel(kernel_size, "blur kernel")
    if gray.ndim != 2:
        raise ValueError("apply_gaussian_blur expects a one-channel image")
    if kernel_size == 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def apply_threshold(
    image: np.ndarray,
    method: str,
    threshold_value: int,
) -> Tuple[np.ndarray, Optional[float]]:
    """Threshold an image using fixed, Otsu or adaptive thresholding."""
    if image.ndim != 2:
        raise ValueError("apply_threshold expects a one-channel image")

    method = method.lower()
    if method == "fixed":
        used, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        threshold_used: Optional[float] = float(used)
    elif method == "otsu":
        used, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_used = float(used)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -2,
        )
        threshold_used = None
    else:
        raise ValueError("threshold method must be one of: fixed, otsu, adaptive")

    return ensure_binary_uint8(binary), threshold_used


def apply_morphology(
    binary: np.ndarray,
    kernel_size: int,
    opening_iterations: int,
    closing_iterations: int,
) -> np.ndarray:
    """Remove isolated noise and fill small gaps in bright regions."""
    validate_odd_kernel(kernel_size, "morphology kernel")
    if opening_iterations < 0 or closing_iterations < 0:
        raise ValueError("morphology iterations must be zero or positive")

    result = ensure_binary_uint8(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if opening_iterations:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    if closing_iterations:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
    return ensure_binary_uint8(result)


def ensure_binary_uint8(image: np.ndarray) -> np.ndarray:
    """Return a uint8 image containing only 0 and 255 values."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    binary = np.where(image > 0, 255, 0).astype(np.uint8)
    values = set(np.unique(binary).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"binary image contains invalid values: {sorted(values)}")
    return binary


def validate_preprocess_result(result: Dict[str, object]) -> None:
    """Validate preprocessing output shapes and binary format."""
    original = result["original"]
    if not isinstance(original, np.ndarray):
        raise ValueError("original image is missing from result")
    height, width = original.shape[:2]

    for key in ("grayscale", "normalized", "blurred", "binary"):
        value = result[key]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{key} image is missing from result")
        if value.shape[:2] != (height, width):
            raise ValueError(f"{key} changed image size from {(width, height)} to {value.shape[:2]}")

    grayscale = result["grayscale"]
    binary = result["binary"]
    if not isinstance(grayscale, np.ndarray) or grayscale.ndim != 2:
        raise ValueError("grayscale image must have one channel")
    if not isinstance(binary, np.ndarray) or binary.dtype != np.uint8:
        raise ValueError("binary image must be uint8")
    if not set(np.unique(binary).tolist()).issubset({0, 255}):
        raise ValueError("binary image must contain only 0 and 255")


def preprocess_frame(frame: np.ndarray, config: PreprocessingConfig | Dict[str, object]) -> Dict[str, object]:
    """Run the full preprocessing flow for one camera frame."""
    cfg = config_from_mapping(config)
    gray = convert_to_grayscale(frame)
    normalized = normalize_contrast(gray) if cfg.normalize_contrast else gray.copy()
    blurred = apply_gaussian_blur(normalized, cfg.blur_kernel)
    thresholded, threshold_used = apply_threshold(
        blurred,
        method=cfg.threshold_method,
        threshold_value=cfg.threshold_value,
    )
    binary = apply_morphology(
        thresholded,
        kernel_size=cfg.morph_kernel,
        opening_iterations=cfg.opening_iterations,
        closing_iterations=cfg.closing_iterations,
    )

    result: Dict[str, object] = {
        "original": frame.copy(),
        "grayscale": gray,
        "normalized": normalized,
        "blurred": blurred,
        "binary": binary,
        "threshold_used": threshold_used,
    }
    validate_preprocess_result(result)
    return result


def save_comparison(result: Dict[str, object], output_path: str | Path) -> None:
    """Save a four-panel preprocessing comparison image."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    original = result["original"]
    if not isinstance(original, np.ndarray):
        raise ValueError("comparison result is missing original image")
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original.ndim == 3 else original
    panels = [
        ("Original", original_rgb, None),
        ("Grayscale", result["grayscale"], "gray"),
        ("Blurred", result["blurred"], "gray"),
        ("Binary", result["binary"], "gray"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, (title, image, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=255 if cmap == "gray" else None)
        axis.set_title(title, fontsize=18, fontweight="bold")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def output_binary_path(input_path: Path, output_path: Path, input_is_dir: bool) -> Path:
    """Choose the binary output path for one input image."""
    if input_is_dir:
        return output_path / f"{input_path.stem}_binary.png"
    if output_path.suffix.lower() == ".png":
        return output_path
    return output_path / f"{input_path.stem}_binary.png"


def process_one_image(
    input_path: Path,
    output_path: Path,
    config: PreprocessingConfig,
    preview: bool,
    input_is_dir: bool = False,
) -> Path:
    """Process one image and save its binary output."""
    image = load_image(input_path)
    result = preprocess_frame(image, config)
    binary_path = output_binary_path(input_path, output_path, input_is_dir)
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(binary_path), result["binary"]):
        raise OSError(f"failed to write binary image: {binary_path}")

    if preview:
        if input_is_dir:
            preview_dir = output_path / "comparisons"
        else:
            preview_dir = output_path.parent if output_path.suffix.lower() == ".png" else output_path
        comparison_path = preview_dir / f"{input_path.stem}_comparison.png"
        save_comparison(result, comparison_path)

    return binary_path


def process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    config: PreprocessingConfig | Dict[str, object],
    preview: bool = False,
) -> Tuple[int, int]:
    """Process all PNG images in a folder and report failures."""
    cfg = config_from_mapping(config)
    source_dir = Path(input_folder)
    destination_dir = Path(output_folder)
    if not source_dir.exists():
        raise FileNotFoundError(f"input folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"input path is not a folder: {source_dir}")

    image_paths = sorted(source_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG images found in {source_dir}")
        return 0, 0

    processed = 0
    failed = 0
    for image_path in image_paths:
        try:
            process_one_image(image_path, destination_dir, cfg, preview=preview, input_is_dir=True)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"Failed {image_path.name}: {exc}")

    print(f"Processed images: {processed}")
    print(f"Failed images: {failed}")
    return processed, failed


def config_from_mapping(config: PreprocessingConfig | Dict[str, object]) -> PreprocessingConfig:
    """Build a config dataclass from either a dataclass or dictionary."""
    if isinstance(config, PreprocessingConfig):
        cfg = config
    else:
        cfg = PreprocessingConfig(
            threshold_method=str(config.get("threshold_method", config.get("method", "fixed"))),
            threshold_value=int(config.get("threshold_value", config.get("threshold", 200))),
            blur_kernel=int(config.get("blur_kernel", 5)),
            morph_kernel=int(config.get("morph_kernel", 3)),
            opening_iterations=int(config.get("opening_iterations", 1)),
            closing_iterations=int(config.get("closing_iterations", 1)),
            normalize_contrast=bool(config.get("normalize_contrast", config.get("normalize", False))),
        )

    validate_odd_kernel(cfg.blur_kernel, "blur kernel")
    validate_odd_kernel(cfg.morph_kernel, "morphology kernel")
    if cfg.threshold_method not in {"fixed", "otsu", "adaptive"}:
        raise ValueError("threshold method must be one of: fixed, otsu, adaptive")
    if not 0 <= cfg.threshold_value <= 255:
        raise ValueError("threshold must be between 0 and 255")
    return cfg


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess FSOC frames into binary beacon masks.")
    parser.add_argument("--input", required=True, help="Input PNG image or folder.")
    parser.add_argument("--output", required=True, help="Output PNG path or folder.")
    parser.add_argument("--method", choices=["fixed", "otsu", "adaptive"], default="fixed")
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    config = PreprocessingConfig(
        threshold_method=args.method,
        threshold_value=args.threshold,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
        normalize_contrast=args.normalize,
    )
    config = config_from_mapping(config)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.is_dir():
        process_folder(input_path, output_path, config, preview=args.preview)
    elif input_path.is_file():
        binary_path = process_one_image(input_path, output_path, config, preview=args.preview)
        print(f"Binary image saved: {binary_path}")
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
