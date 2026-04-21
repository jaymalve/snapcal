"""Service layer for cached model inference."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from snapcal.inference import InferenceNotReadyError, LocalInferenceService

REPO_ROOT = Path(__file__).resolve().parents[3]


class PredictionRuntime:
    def __init__(self, bundle_dir: Path, enable_segmentation: bool = True):
        self.bundle_dir = bundle_dir
        self.enable_segmentation = enable_segmentation
        self._service = None

    def _load_service(self) -> LocalInferenceService:
        if self._service is None:
            if not self.bundle_dir.exists():
                raise InferenceNotReadyError(f"Bundle directory does not exist: {self.bundle_dir}")
            self._service = LocalInferenceService(
                bundle_dir=self.bundle_dir,
                enable_segmentation=self.enable_segmentation,
            )
        return self._service

    def health(self) -> dict:
        try:
            return self._load_service().health()
        except Exception as exc:
            return {
                "ready": False,
                "bundle_dir": str(self.bundle_dir),
                "error": str(exc),
            }

    def predict(self, image_bytes: bytes, portion_multiplier: float):
        return self._load_service().predict(image_bytes=image_bytes, portion_multiplier=portion_multiplier)


@lru_cache(maxsize=1)
def get_prediction_runtime() -> PredictionRuntime:
    bundle_setting = os.getenv("SNAPCAL_MODEL_BUNDLE", "artifacts/models/production_bundle")
    bundle_dir = Path(bundle_setting)
    if not bundle_dir.is_absolute():
        bundle_dir = (REPO_ROOT / bundle_dir).resolve()
    enable_segmentation = os.getenv("SNAPCAL_ENABLE_SEGMENTATION", "true").lower() == "true"
    return PredictionRuntime(bundle_dir=bundle_dir, enable_segmentation=enable_segmentation)
