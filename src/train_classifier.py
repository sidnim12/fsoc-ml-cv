from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .beacon_classifier import (
        CLASS_NAMES,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_MEAN,
        DEFAULT_STD,
        BeaconClassifier,
        count_parameters,
        load_classifier,
        predict_patch,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from beacon_classifier import (
        CLASS_NAMES,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_MEAN,
        DEFAULT_STD,
        BeaconClassifier,
        count_parameters,
        load_classifier,
        predict_patch,
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Command-line training settings."""

    metadata: Path
    data_root: Path
    output_dir: Path
    checkpoint_dir: Path
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 6
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


class PatchDataset(Dataset):
    """Dataset backed by data/processed/patch_labels.csv."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        data_root: Path,
        split: str,
        train: bool,
        image_size: int = DEFAULT_IMAGE_SIZE,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
    ) -> None:
        self.rows = metadata[metadata["split"] == split].reset_index(drop=True)
        self.data_root = data_root
        self.split = split
        self.train = train
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        row = self.rows.iloc[index]
        patch_path = self.data_root / str(row["patch_path"])
        image = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"patch image is unreadable: {patch_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if self.train:
            image = augment_image(image)

        tensor = normalize_to_tensor(image, self.mean, self.std)
        label = int(row["label_id"])
        metadata = {
            "patch_path": str(patch_path),
            "patch_filename": str(row["patch_filename"]),
            "source_filename": str(row["source_filename"]),
            "label": str(row["label"]),
            "label_id": label,
        }
        return tensor, torch.tensor(label, dtype=torch.long), metadata


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply mild augmentations that keep the beacon visible."""
    output = image.copy()
    if random.random() < 0.5:
        output = cv2.flip(output, 1)
    if random.random() < 0.5:
        output = cv2.flip(output, 0)
    if random.random() < 0.45:
        angle = random.uniform(-12.0, 12.0)
        center = (output.shape[1] / 2.0, output.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        output = cv2.warpAffine(output, matrix, (output.shape[1], output.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.45:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-8.0, 8.0)
        output = np.clip(output.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.15:
        output = cv2.GaussianBlur(output, (3, 3), 0)
    return np.ascontiguousarray(output)


def normalize_to_tensor(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 image to normalized CHW tensor."""
    array = image.astype(np.float32) / 255.0
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1).float()


def set_deterministic_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, PyTorch and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    """Choose CUDA when available, otherwise CPU."""
    if requested != "auto":
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    return device


def load_metadata(path: Path) -> pd.DataFrame:
    """Load and lightly normalize patch metadata."""
    if not path.exists():
        raise FileNotFoundError(f"metadata CSV does not exist: {path}")
    metadata = pd.read_csv(path)
    required = {
        "patch_filename",
        "patch_path",
        "source_filename",
        "split",
        "label",
        "label_id",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    metadata["patch_path"] = metadata["patch_path"].astype(str)
    metadata["split"] = metadata["split"].astype(str)
    metadata["label"] = metadata["label"].astype(str)
    metadata["label_id"] = metadata["label_id"].astype(int)
    return metadata


def validate_metadata(metadata: pd.DataFrame, data_root: Path) -> Dict[str, Dict[str, int]]:
    """Validate splits, classes, paths, folder labels and duplicates."""
    if metadata.empty:
        raise ValueError("patch metadata is empty")
    expected_splits = {"train", "validation", "test"}
    missing_splits = sorted(expected_splits - set(metadata["split"]))
    if missing_splits:
        raise ValueError(f"missing split(s): {missing_splits}")
    invalid_labels = sorted(set(metadata["label_id"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"invalid label IDs: {invalid_labels}")
    duplicates = metadata[metadata["patch_path"].duplicated()]["patch_path"].tolist()
    if duplicates:
        raise ValueError(f"duplicate patch paths found: {duplicates[:5]}")

    split_class_counts = metadata.groupby(["split", "label"]).size().unstack(fill_value=0)
    for split in expected_splits:
        for class_name in CLASS_NAMES:
            if split_class_counts.loc[split].get(class_name, 0) == 0:
                raise ValueError(f"split '{split}' is missing class '{class_name}'")

    mixed_source_frames = metadata.groupby("source_filename")["split"].nunique()
    mixed_source_frames = mixed_source_frames[mixed_source_frames > 1]
    if not mixed_source_frames.empty:
        raise ValueError(f"source frame appears in multiple splits: {mixed_source_frames.index[:5].tolist()}")

    for row in metadata.itertuples(index=False):
        patch_path = data_root / str(row.patch_path)
        if not patch_path.exists():
            raise FileNotFoundError(f"patch file missing: {patch_path}")
        folder_label = patch_path.parent.name
        if folder_label != str(row.label):
            raise ValueError(f"label/folder mismatch for {patch_path}: metadata={row.label}, folder={folder_label}")
        image = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"patch unreadable: {patch_path}")
        if image.shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3):
            raise ValueError(f"patch has shape {image.shape}, expected 32x32x3: {patch_path}")

    return {
        split: {label: int(count) for label, count in split_class_counts.loc[split].to_dict().items()}
        for split in sorted(expected_splits)
    }


def make_loaders(metadata: pd.DataFrame, config: TrainingConfig) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, validation and test."""
    datasets = {
        "train": PatchDataset(metadata, config.data_root, "train", train=True),
        "validation": PatchDataset(metadata, config.data_root, "validation", train=False),
        "test": PatchDataset(metadata, config.data_root, "test", train=False),
    }
    generator = torch.Generator().manual_seed(config.seed)
    return {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }


def validate_batch_shape(loaders: Dict[str, DataLoader]) -> None:
    """Ensure tensors enter the model as batch x 3 x 32 x 32."""
    images, labels, _ = next(iter(loaders["train"]))
    if tuple(images.shape[1:]) != (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        raise ValueError(f"input tensors have wrong shape: {tuple(images.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels should be a 1D tensor, got shape {tuple(labels.shape)}")


def class_weights(metadata: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Calculate CrossEntropy class weights from the training set only."""
    train = metadata[metadata["split"] == "train"]
    counts = train["label_id"].value_counts().to_dict()
    total = len(train)
    weights = []
    for label_id in (0, 1):
        count = counts.get(label_id, 0)
        if count == 0:
            raise ValueError(f"training split has no samples for label {label_id}")
        weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float | List[int] | List[float]]:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        total_loss += float(loss.item()) * images.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        y_prob.extend(probabilities[:, 1].detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


def classification_metrics(y_true: List[int], y_pred: List[int], y_prob: Optional[List[float]] = None) -> Dict[str, float]:
    """Calculate accuracy, precision, recall and F1 for class 1."""
    true = np.array(y_true, dtype=np.int64)
    pred = np.array(y_pred, dtype=np.int64)
    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    total = max(1, len(true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    metrics = {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    if y_prob is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score_manual(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def roc_auc_score_manual(y_true: List[int], y_prob: List[float]) -> float:
    """Compute binary ROC-AUC using average ranks."""
    labels = np.array(y_true, dtype=np.int64)
    scores = np.array(y_prob, dtype=np.float64)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average_rank
        index = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainingConfig,
) -> None:
    """Save model and training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_val_loss,
            "class_names": CLASS_NAMES,
            "image_size": DEFAULT_IMAGE_SIZE,
            "normalization_mean": DEFAULT_MEAN,
            "normalization_std": DEFAULT_STD,
            "training_arguments": serializable_config(config),
            "model_architecture": "BeaconClassifier",
        },
        path,
    )


def serializable_config(config: TrainingConfig) -> Dict[str, object]:
    """Convert paths in config to strings for JSON/checkpoints."""
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def train_model(config: TrainingConfig) -> Tuple[BeaconClassifier, List[Dict[str, float]], Dict[str, DataLoader]]:
    """Train the CNN with early stopping on validation loss."""
    set_deterministic_seed(config.seed)
    device = select_device(config.device)
    metadata = load_metadata(config.metadata)
    class_counts = validate_metadata(metadata, config.data_root)
    print(f"Class counts by split: {class_counts}")
    loaders = make_loaders(metadata, config)
    validate_batch_shape(loaders)

    model = BeaconClassifier().to(device)
    print(f"Trainable parameters: {count_parameters(model)}")
    weights = class_weights(metadata, device)
    print(f"Class weights: false={weights[0].item():.3f}, correct={weights[1].item():.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, criterion, optimizer)
        val_metrics = run_epoch(model, loaders["validation"], device, criterion)
        val_loss = float(val_metrics["loss"])
        scheduler.step(val_loss)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "validation_loss": val_loss,
            "validation_accuracy": float(val_metrics["accuracy"]),
            "validation_precision": float(val_metrics["precision"]),
            "validation_recall": float(val_metrics["recall"]),
            "validation_f1": float(val_metrics["f1"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"Train Loss: {row['train_loss']:.3f} | Train Acc: {row['train_accuracy'] * 100:.1f}% | "
            f"Val Loss: {row['validation_loss']:.3f} | Val Acc: {row['validation_accuracy'] * 100:.1f}% | "
            f"Val F1: {row['validation_f1']:.3f}"
        )

        save_checkpoint(config.checkpoint_dir / "last_classifier.pt", model, optimizer, epoch, best_val_loss, config)
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(config.checkpoint_dir / "best_classifier.pt", model, optimizer, epoch, best_val_loss, config)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.output_dir / "history.csv", index=False)
    save_training_curves(history, config.output_dir / "training_curves.png")
    warn_if_overfitting(history)
    best_model = load_classifier(config.checkpoint_dir / "best_classifier.pt", device)
    return best_model, history, loaders


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    """Evaluate a model once and return test metrics."""
    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(model, loader, device, criterion)
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "confusion_matrix": [
            [int(metrics["true_negative"]), int(metrics["false_positive"])],
            [int(metrics["false_negative"]), int(metrics["true_positive"])],
        ],
        "class_names": CLASS_NAMES,
        "y_true": metrics["y_true"],
        "y_pred": metrics["y_pred"],
        "y_prob": metrics["y_prob"],
    }


def save_training_curves(history: List[Dict[str, float]], output_path: Path) -> None:
    """Plot loss, accuracy, precision, recall and F1 curves."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train loss")
    axes[0].plot(df["epoch"], df["validation_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="Train accuracy")
    axes[1].plot(df["epoch"], df["validation_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["validation_precision"], label="Validation precision")
    axes[2].plot(df["epoch"], df["validation_recall"], label="Validation recall")
    axes[2].plot(df["epoch"], df["validation_f1"], label="Validation F1")
    axes[2].set_title("Validation Metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(matrix: List[List[int]], output_path: Path) -> None:
    """Save a clearly labelled confusion matrix image."""
    array = np.array(matrix, dtype=np.int64)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], ["Predicted False", "Predicted Correct"])
    axis.set_yticks([0, 1], ["True False Beacon", "True Correct Beacon"])
    axis.set_title("Test Confusion Matrix")
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(array[row, col]), ha="center", va="center", color="black", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def classification_report(metrics: Dict[str, object]) -> Dict[str, object]:
    """Create a small JSON classification report."""
    matrix = metrics["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    false_precision = tn / (tn + fn) if (tn + fn) else 0.0
    false_recall = tn / (tn + fp) if (tn + fp) else 0.0
    false_f1 = (2 * false_precision * false_recall / (false_precision + false_recall)) if (false_precision + false_recall) else 0.0
    return {
        "false": {"precision": false_precision, "recall": false_recall, "f1": false_f1},
        "correct": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
    }


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> List[Dict[str, object]]:
    """Collect predictions and metadata for sample montage."""
    model.eval()
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1).detach().cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            for index in range(len(labels)):
                label_id = int(labels[index].item())
                predicted_id = int(predictions[index])
                confidence = float(probabilities[index, predicted_id])
                rows.append(
                    {
                        "patch_path": metadata["patch_path"][index],
                        "true_label": CLASS_NAMES[label_id],
                        "predicted_label": CLASS_NAMES[predicted_id],
                        "confidence": confidence,
                        "correct": label_id == predicted_id,
                    }
                )
    return rows


def choose_sample_predictions(predictions: List[Dict[str, object]], max_samples: int = 25) -> List[Dict[str, object]]:
    """Choose correct, false, high-confidence, low-confidence and misclassified examples."""
    selected: List[Dict[str, object]] = []

    def add(items: List[Dict[str, object]]) -> None:
        seen = {item["patch_path"] for item in selected}
        for item in items:
            if item["patch_path"] not in seen and len(selected) < max_samples:
                selected.append(item)
                seen.add(item["patch_path"])

    add([item for item in predictions if item["correct"] and item["true_label"] == "correct"][:5])
    add([item for item in predictions if item["correct"] and item["true_label"] == "false"][:5])
    add(sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True)[:5])
    add(sorted(predictions, key=lambda item: float(item["confidence"]))[:5])
    add([item for item in predictions if not item["correct"]][:5])
    add(predictions[: max_samples - len(selected)])
    return selected[:max_samples]


def save_sample_predictions(predictions: List[Dict[str, object]], output_path: Path) -> None:
    """Save a montage of representative model predictions."""
    samples = choose_sample_predictions(predictions)
    cols = 5
    rows = max(1, int(np.ceil(len(samples) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows))
    axes_flat = np.array(axes).reshape(-1)
    for axis in axes_flat:
        axis.axis("off")
    for axis, sample in zip(axes_flat, samples):
        image = cv2.imread(str(sample["patch_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        title = f"T:{sample['true_label']}\nP:{sample['predicted_label']} {float(sample['confidence']):.2f}"
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def warn_if_overfitting(history: List[Dict[str, float]]) -> None:
    """Print simple overfitting warnings after training."""
    if not history:
        return
    last = history[-1]
    if last["train_accuracy"] > 0.98 and last["validation_accuracy"] < 0.90:
        print("Warning: training accuracy is very high but validation accuracy is much lower.")
    if len(history) >= 4:
        recent = history[-4:]
        train_falling = recent[-1]["train_loss"] < recent[0]["train_loss"]
        val_rising = recent[-1]["validation_loss"] > recent[0]["validation_loss"]
        if train_falling and val_rising:
            print("Warning: validation loss is rising while training loss is falling.")


def warn_test_gap(history: List[Dict[str, float]], test_metrics: Dict[str, object]) -> None:
    """Warn when test performance is far below validation performance."""
    if not history:
        return
    best_val_acc = max(row["validation_accuracy"] for row in history)
    if float(test_metrics["accuracy"]) + 0.10 < best_val_acc:
        print("Warning: test accuracy is substantially below validation accuracy.")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_inference_smoke_test(checkpoint_path: Path, metadata: pd.DataFrame, config: TrainingConfig, device: torch.device) -> List[Dict[str, object]]:
    """Load the saved classifier and predict at least five patches."""
    model = load_classifier(checkpoint_path, device)
    samples = metadata.head(5)
    outputs = []
    for row in samples.itertuples(index=False):
        output = predict_patch(model, config.data_root / str(row.patch_path), device=device)
        output["patch_filename"] = str(row.patch_filename)
        outputs.append(output)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    config = TrainingConfig(
        metadata=Path(args.metadata),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    try:
        model, history, loaders = train_model(config)
        device = select_device(config.device)
        model.to(device)
        test_metrics = evaluate_model(model, loaders["test"], device)
        metrics_for_json = {key: value for key, value in test_metrics.items() if key not in {"y_true", "y_pred", "y_prob"}}
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.output_dir / "test_metrics.json", metrics_for_json)
        write_json(config.output_dir / "classification_report.json", classification_report(test_metrics))
        save_confusion_matrix(test_metrics["confusion_matrix"], config.output_dir / "confusion_matrix.png")
        predictions = collect_predictions(model, loaders["test"], device)
        save_sample_predictions(predictions, config.output_dir / "sample_predictions.png")
        metadata = load_metadata(config.metadata)
        smoke_outputs = run_inference_smoke_test(config.checkpoint_dir / "best_classifier.pt", metadata, config, device)
        write_json(config.output_dir / "inference_smoke_test.json", {"predictions": smoke_outputs})
        warn_test_gap(history, test_metrics)

        print("\nFinal test metrics:")
        print(f"  Accuracy: {float(test_metrics['accuracy']):.4f}")
        print(f"  Precision: {float(test_metrics['precision']):.4f}")
        print(f"  Recall: {float(test_metrics['recall']):.4f}")
        print(f"  F1-score: {float(test_metrics['f1']):.4f}")
        print(f"  ROC-AUC: {float(test_metrics['roc_auc']):.4f}")
        print("\nSynthetic-data warning: current results are from synthetic patches only.")
        print("The classifier must later be tested on Unity-generated frames with varied lighting, movement and noise.")
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the lightweight FSOC beacon patch classifier.")
    parser.add_argument("--metadata", default="data/processed/patch_labels.csv")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
