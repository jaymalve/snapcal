"""Model registry and image transforms."""

from __future__ import annotations

from typing import Callable, Tuple

from .constants import IMAGENET_MEAN, IMAGENET_STD


def build_image_transforms(image_size: int, train: bool) -> Callable:
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("torchvision is not installed. Install snapcal[train] to enable model training.") from exc

    common = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if not train:
        return transforms.Compose(common)
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            *common,
            transforms.RandomErasing(p=0.25),
        ]
    )


def build_model(model_name: str, num_classes: int):
    try:
        import torch.nn as nn
        from torchvision.models import (
            EfficientNet_B0_Weights,
            ResNet50_Weights,
            efficientnet_b0,
            resnet50,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("torch and torchvision are required for model creation.") from exc

    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "vit_b16":
        try:
            from transformers import ViTForImageClassification
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("transformers is required for ViT training.") from exc
        return ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def extract_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs
