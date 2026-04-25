"""Service layer for cached model inference."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Dict, Optional

from snapcal.inference import InferenceNotReadyError, LocalInferenceService

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = "artifacts/models/production_bundle"
DEFAULT_MODEL_ID = "default"
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b16": "ViT-B/16",
}


class UnknownModelError(ValueError):
    """Raised when a prediction request targets an unknown model id."""


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

    def predict(
        self,
        image_bytes: bytes,
        portion_unit: str,
        portion_value: Optional[int],
        enable_segmentation: bool,
        model_id: Optional[str] = None,
    ):
        return self._load_service().predict(
            image_bytes=image_bytes,
            portion_unit=portion_unit,
            portion_value=portion_value,
            enable_segmentation=enable_segmentation,
            model_id=model_id,
        )


def _resolve_bundle_dir(bundle_setting: str) -> Path:
    bundle_dir = Path(bundle_setting)
    if not bundle_dir.is_absolute():
        bundle_dir = (REPO_ROOT / bundle_dir).resolve()
    return bundle_dir


def _segmentation_enabled() -> bool:
    enable_segmentation = os.getenv("SNAPCAL_ENABLE_SEGMENTATION", "true").lower() == "true"
    return enable_segmentation


def _fallback_model_id(bundle_dir: Path) -> str:
    bundle_name = bundle_dir.name
    if bundle_name.startswith("production_bundle_"):
        suffix = bundle_name[len("production_bundle_") :]
        if suffix:
            return suffix
    return DEFAULT_MODEL_ID


def _infer_model_id(bundle_dir: Path) -> str:
    metadata_path = bundle_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = {}
        model_name = payload.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return _fallback_model_id(bundle_dir)


def _discover_local_bundles(primary_bundle_dir: Path) -> Dict[str, Path]:
    bundles: Dict[str, Path] = {}
    candidate_dirs = [primary_bundle_dir]
    parent_dir = primary_bundle_dir.parent
    if parent_dir.exists():
        sibling_dirs = sorted(
            path
            for path in parent_dir.iterdir()
            if path.is_dir() and path.name.startswith("production_bundle") and path != primary_bundle_dir
        )
        candidate_dirs.extend(sibling_dirs)

    for bundle_dir in candidate_dirs:
        model_id = _infer_model_id(bundle_dir)
        if model_id not in bundles:
            bundles[model_id] = bundle_dir
    return bundles


@lru_cache(maxsize=1)
def get_configured_bundles() -> Dict[str, Path]:
    bundle_mapping = os.getenv("SNAPCAL_MODEL_BUNDLES", "").strip()
    if bundle_mapping:
        try:
            payload = json.loads(bundle_mapping)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "SNAPCAL_MODEL_BUNDLES must be a JSON object mapping model ids to bundle directories."
            ) from exc
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(
                "SNAPCAL_MODEL_BUNDLES must be a non-empty JSON object mapping model ids to bundle directories."
            )
        bundles: Dict[str, Path] = {}
        for raw_model_id, raw_bundle_dir in payload.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                raise RuntimeError("SNAPCAL_MODEL_BUNDLES keys must be non-empty strings.")
            if not isinstance(raw_bundle_dir, str) or not raw_bundle_dir.strip():
                raise RuntimeError("SNAPCAL_MODEL_BUNDLES values must be non-empty bundle directory strings.")
            model_id = raw_model_id.strip()
            bundles[model_id] = _resolve_bundle_dir(raw_bundle_dir.strip())
        return bundles

    bundle_setting = os.getenv("SNAPCAL_MODEL_BUNDLE", DEFAULT_BUNDLE_DIR)
    return _discover_local_bundles(_resolve_bundle_dir(bundle_setting))


def get_default_model_id() -> Optional[str]:
    bundles = get_configured_bundles()
    configured_default = os.getenv("SNAPCAL_DEFAULT_MODEL_ID", "").strip()
    if configured_default and configured_default in bundles:
        return configured_default
    return next(iter(bundles), None)


def resolve_model_id(requested_model_id: Optional[str]) -> str:
    model_id = (requested_model_id or "").strip() or get_default_model_id()
    if not model_id:
        raise RuntimeError("No model bundles are configured for prediction.")
    if model_id not in get_configured_bundles():
        available_models = ", ".join(sorted(get_configured_bundles()))
        raise UnknownModelError(f"Unknown model_id '{model_id}'. Available models: {available_models}")
    return model_id


def _label_for_model(model_name: Optional[str], model_id: str) -> str:
    if model_name and model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    return model_id.replace("_", " ")


@lru_cache(maxsize=2)
def get_prediction_runtime(model_id: str) -> PredictionRuntime:
    bundle_dir = get_configured_bundles().get(model_id)
    if bundle_dir is None:
        available_models = ", ".join(sorted(get_configured_bundles()))
        raise UnknownModelError(f"Unknown model_id '{model_id}'. Available models: {available_models}")
    return PredictionRuntime(bundle_dir=bundle_dir, enable_segmentation=_segmentation_enabled())


def get_prediction_health() -> dict:
    try:
        bundles = get_configured_bundles()
    except RuntimeError as exc:
        return {
            "ready": False,
            "default_model_id": None,
            "models": [],
            "error": str(exc),
        }

    default_model_id = get_default_model_id()
    models = []
    any_ready = False
    default_entry = None

    for model_id, bundle_dir in bundles.items():
        entry = {
            "id": model_id,
            "label": _label_for_model(None, model_id),
            "bundle_dir": str(bundle_dir),
        }
        status = get_prediction_runtime(model_id).health()
        entry.update(status)
        entry["label"] = _label_for_model(entry.get("model_name"), model_id)
        models.append(entry)
        any_ready = any_ready or bool(entry.get("ready"))
        if model_id == default_model_id:
            default_entry = entry

    if default_entry is None and models:
        default_entry = models[0]
        default_model_id = str(default_entry["id"])

    payload = {
        "ready": any_ready,
        "default_model_id": default_model_id,
        "models": models,
    }
    if default_entry is not None:
        for key in (
            "model_version",
            "model_name",
            "bundle_dir",
            "segmentation_available",
            "segmentation_reason",
            "error",
        ):
            if key in default_entry:
                payload[key] = default_entry[key]
    return payload
