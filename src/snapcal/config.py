"""Config loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .constants import SUPPORTED_DATASET_VARIANTS, SUPPORTED_MODELS


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_extends(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    extends = payload.pop("extends", None)
    if not extends:
        return payload
    extends_path = Path(extends)
    if extends_path.is_absolute():
        base_path = extends_path
    else:
        candidate = (path.parent / extends_path).resolve()
        base_path = candidate if candidate.exists() else (Path.cwd() / extends_path).resolve()
    merged = _resolve_extends(base_path)
    merged.update(payload)
    return merged


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str
    seed: int
    dataset_variant: str
    model_name: str
    num_classes: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    warmup_epochs: int
    num_workers: int
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path
    output_dir: Path
    report_dir: Path

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfig":
        resolved = _resolve_extends(path.resolve())
        instance = cls(
            experiment_name=str(resolved["experiment_name"]),
            seed=int(resolved["seed"]),
            dataset_variant=str(resolved["dataset_variant"]),
            model_name=str(resolved["model_name"]),
            num_classes=int(resolved["num_classes"]),
            image_size=int(resolved["image_size"]),
            batch_size=int(resolved["batch_size"]),
            epochs=int(resolved["epochs"]),
            learning_rate=float(resolved["learning_rate"]),
            weight_decay=float(resolved["weight_decay"]),
            label_smoothing=float(resolved["label_smoothing"]),
            warmup_epochs=int(resolved.get("warmup_epochs", 0)),
            num_workers=int(resolved.get("num_workers", 4)),
            train_manifest=Path(resolved["train_manifest"]),
            val_manifest=Path(resolved["val_manifest"]),
            test_manifest=Path(resolved["test_manifest"]),
            output_dir=Path(resolved["output_dir"]),
            report_dir=Path(resolved["report_dir"]),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        if self.dataset_variant not in SUPPORTED_DATASET_VARIANTS:
            raise ValueError(f"Unsupported dataset variant: {self.dataset_variant}")
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_name}")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")


@dataclass(frozen=True)
class SegmentationConfig:
    model_type: str
    checkpoint_path: Path
    points_per_side: int
    pred_iou_thresh: float
    stability_score_thresh: float
    crop_n_layers: int
    min_mask_region_area: int
    min_area_ratio: float
    secondary_mask_score_delta: float
    crop_margin_ratio: float
    output_image_size: int
    background_fill_rgb: Tuple[int, int, int]

    @classmethod
    def from_json(cls, path: Path) -> "SegmentationConfig":
        resolved = _resolve_extends(path.resolve())
        fill = tuple(int(value) for value in resolved.get("background_fill_rgb", (123, 116, 103)))
        if len(fill) != 3:
            raise ValueError("background_fill_rgb must contain three integers")
        return cls(
            model_type=str(resolved["model_type"]),
            checkpoint_path=Path(resolved["checkpoint_path"]),
            points_per_side=int(resolved["points_per_side"]),
            pred_iou_thresh=float(resolved["pred_iou_thresh"]),
            stability_score_thresh=float(resolved["stability_score_thresh"]),
            crop_n_layers=int(resolved["crop_n_layers"]),
            min_mask_region_area=int(resolved["min_mask_region_area"]),
            min_area_ratio=float(resolved["min_area_ratio"]),
            secondary_mask_score_delta=float(resolved["secondary_mask_score_delta"]),
            crop_margin_ratio=float(resolved["crop_margin_ratio"]),
            output_image_size=int(resolved["output_image_size"]),
            background_fill_rgb=fill,  # type: ignore[arg-type]
        )


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
