from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from torch import nn


CLASS_NAMES = ["false", "correct"]
DEFAULT_IMAGE_SIZE = 32
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


class BeaconClassifier(nn.Module):
    """Small binary CNN for 32x32 RGB beacon candidate patches."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits. CrossEntropyLoss applies softmax internally."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_classifier(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> BeaconClassifier:
    """Load a classifier checkpoint on CPU or GPU using map_location."""
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_file}")

    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_file, map_location=target_device)
    model = BeaconClassifier()
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model


def patch_to_tensor(
    patch: str | Path | np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> torch.Tensor:
    """Convert a path or RGB/BGR NumPy patch to a normalized tensor."""
    if isinstance(patch, (str, Path)):
        image = cv2.imread(str(patch), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read patch image: {patch}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = patch.copy()
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected a 3-channel patch, got shape {image.shape}")

    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    array = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def predict_patch(
    model: BeaconClassifier,
    patch: str | Path | np.ndarray,
    device: str | torch.device = "cpu",
    transform: Optional[object] = None,
) -> Dict[str, float | int | str]:
    """Predict class probabilities for one candidate patch."""
    target_device = torch.device(device)
    if transform is not None:
        tensor = transform(patch)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("custom transform must return a torch.Tensor")
    else:
        tensor = patch_to_tensor(patch)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(target_device)
    model = model.to(target_device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    label_id = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[label_id],
        "label_id": label_id,
        "confidence": float(probabilities[label_id]),
        "false_probability": float(probabilities[0]),
        "correct_probability": float(probabilities[1]),
    }
