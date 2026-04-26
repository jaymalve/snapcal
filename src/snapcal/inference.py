"""Inference bundle loading and runtime prediction."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .config import SegmentationConfig
from .constants import FOOD101_CLASSES
from .models import build_image_transforms, build_model, extract_logits
from .nutrition import NutritionLookup
from .segmentation import MobileSAMSegmenter
from .torch_utils import resolve_torch_device


class InferenceNotReadyError(RuntimeError):
    """Raised when the exported model bundle is not ready for inference."""


@dataclass(frozen=True)
class BundleMetadata:
    model_name: str
    model_version: str
    checkpoint_path: Path
    nutrition_mapping_path: Path
    segmentation_config_path: Optional[Path]
    image_size: int

    @classmethod
    def from_bundle_dir(cls, bundle_dir: Path) -> "BundleMetadata":
        metadata_path = bundle_dir / "metadata.json"
        if not metadata_path.exists():
            raise InferenceNotReadyError(f"Missing bundle metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            checkpoint_path=bundle_dir / payload["checkpoint_path"],
            nutrition_mapping_path=bundle_dir / payload["nutrition_mapping_path"],
            segmentation_config_path=(bundle_dir / payload["segmentation_config_path"])
            if payload.get("segmentation_config_path")
            else None,
            image_size=int(payload["image_size"]),
        )


class LocalInferenceService:
    def __init__(self, bundle_dir: Path, enable_segmentation: bool = True):
        self.bundle_dir = bundle_dir
        self.enable_segmentation = enable_segmentation
        self.metadata = BundleMetadata.from_bundle_dir(bundle_dir)
        self.nutrition = NutritionLookup.from_csv(self.metadata.nutrition_mapping_path)
        self.transform = build_image_transforms(self.metadata.image_size, train=False)
        self._model = None
        self._device = None
        self._segmenter = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise InferenceNotReadyError("torch is required for inference.") from exc
        if not self.metadata.checkpoint_path.exists():
            raise InferenceNotReadyError(f"Model checkpoint not found: {self.metadata.checkpoint_path}")
        self._device = resolve_torch_device()
        model = build_model(self.metadata.model_name, len(FOOD101_CLASSES), pretrained=False)
        checkpoint = torch.load(self.metadata.checkpoint_path, map_location=self._device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self._device)
        model.eval()
        self._model = model
        return model

    def _load_segmenter(self):
        if not self.enable_segmentation or self.metadata.segmentation_config_path is None:
            return None
        if self._segmenter is None:
            config = SegmentationConfig.from_json(self.metadata.segmentation_config_path)
            self._segmenter = MobileSAMSegmenter(config)
        return self._segmenter

    def health(self) -> Dict[str, object]:
        ready = self.metadata.checkpoint_path.exists() and self.metadata.nutrition_mapping_path.exists()
        segmentation_available, segmentation_reason = self._segmentation_status()
        return {
            "ready": ready,
            "model_version": self.metadata.model_version,
            "model_name": self.metadata.model_name,
            "bundle_dir": str(self.bundle_dir),
            "segmentation_available": segmentation_available,
            "segmentation_reason": segmentation_reason,
        }

    def _segmentation_status(self) -> Tuple[bool, Optional[str]]:
        if not self.enable_segmentation:
            return (
                False,
                "Segmentation is disabled on this backend. Restart with SNAPCAL_ENABLE_SEGMENTATION=true to enable the comparison toggle.",
            )
        if self.metadata.segmentation_config_path is None:
            return False, "This bundle does not include a segmentation config."
        if not self.metadata.segmentation_config_path.exists():
            return False, f"Segmentation config not found: {self.metadata.segmentation_config_path}"
        try:
            config = SegmentationConfig.from_json(self.metadata.segmentation_config_path)
        except Exception as exc:
            return False, f"Segmentation config could not be loaded: {exc}"
        if not config.checkpoint_path.exists():
            return False, f"Segmentation checkpoint not found: {config.checkpoint_path}"
        if importlib.util.find_spec("mobile_sam") is None:
            return False, "MobileSAM is not installed on this backend."
        return True, None

    def predict(
        self,
        image_bytes: bytes,
        portion_unit: str = "serving",
        portion_value: Optional[int] = None,
        enable_segmentation: bool = False,
        model_id: Optional[str] = None,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise InferenceNotReadyError("torch is required for inference.") from exc

        model = self._load_model()
        warnings: List[str] = []
        segmentation_requested = enable_segmentation
        segmentation_applied = False
        preprocess_start = time.perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as input_image:
            image = input_image.convert("RGB")
            segmentation_preview_url = None
            if enable_segmentation:
                segmentation_available, segmentation_reason = self._segmentation_status()
                if not segmentation_available:
                    raise InferenceNotReadyError(segmentation_reason or "Segmentation is not available.")
                try:
                    segmenter = self._load_segmenter()
                except Exception as exc:
                    raise InferenceNotReadyError(f"Segmentation requested but could not be initialized: {exc}") from exc
                if segmenter is None:
                    raise InferenceNotReadyError("Segmentation was requested but could not be initialized.")
                try:
                    segmented_image, _mask_image, _meta_json = segmenter.segment_image(image)
                    image = segmented_image
                    segmentation_applied = True
                except Exception as exc:  # pragma: no cover - optional dependency path
                    raise InferenceNotReadyError(f"Segmentation requested but failed: {exc}") from exc
            tensor = self.transform(image).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        inference_start = time.perf_counter()
        tensor = tensor.to(self._device)
        with torch.no_grad():
            outputs = model(pixel_values=tensor) if self.metadata.model_name == "vit_b16" else model(tensor)
            logits = extract_logits(outputs)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_scores, top_indices = torch.topk(probabilities, k=3)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        ranked_predictions: List[Tuple[str, float]] = [
            (FOOD101_CLASSES[index], float(score))
            for score, index in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist())
        ]
        return self.nutrition.build_response(
            ranked_predictions=ranked_predictions,
            requested_portion_unit=portion_unit,
            requested_portion_value=portion_value,
            model_id=model_id or self.metadata.model_name,
            model_name=self.metadata.model_name,
            model_version=self.metadata.model_version,
            segmentation_requested=segmentation_requested,
            segmentation_applied=segmentation_applied,
            segmentation_preview_url=segmentation_preview_url,
            latency_ms={
                "preprocess": round(preprocess_ms, 3),
                "inference": round(inference_ms, 3),
                "total": round(preprocess_ms + inference_ms, 3),
            },
            warnings=warnings,
        )
