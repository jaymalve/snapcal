"""Service layer for cached model inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from snapcal.inference import InferenceNotReadyError, LocalInferenceService

from .remote import RemoteModelConfig, RemotePredictionRuntime

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = "artifacts/models/production_bundle"
DEFAULT_MAPPING_PATH = "data/reference/food101_usda_mapping.csv"
DEFAULT_MODEL_ID = "default"
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b16": "ViT-B/16",
}


@dataclass(frozen=True)
class LocalModelConfig:
    bundle_dir: Path
    label: Optional[str] = None


ModelConfig = Union[LocalModelConfig, RemoteModelConfig]


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


def _resolve_mapping_path() -> Path:
    mapping_setting = os.getenv("SNAPCAL_NUTRITION_MAPPING", DEFAULT_MAPPING_PATH)
    mapping_path = Path(mapping_setting)
    if not mapping_path.is_absolute():
        mapping_path = (REPO_ROOT / mapping_path).resolve()
    return mapping_path


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


def _discover_local_bundles(primary_bundle_dir: Path) -> Dict[str, LocalModelConfig]:
    bundles: Dict[str, LocalModelConfig] = {}
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
            bundles[model_id] = LocalModelConfig(bundle_dir=bundle_dir)
    return bundles


def _remote_auth_token(provider: str, payload: Dict[str, object]) -> Optional[str]:
    explicit_token = payload.get("auth_token")
    if isinstance(explicit_token, str) and explicit_token.strip():
        return explicit_token.strip()

    auth_token_env = payload.get("auth_token_env")
    if isinstance(auth_token_env, str) and auth_token_env.strip():
        token = os.getenv(auth_token_env.strip(), "").strip()
        return token or None

    if provider == "modal":
        token = os.getenv("MODAL_AUTH_TOKEN", "").strip()
        return token or None
    return None


def _parse_remote_model_config(model_id: str, payload: Dict[str, object]) -> RemoteModelConfig:
    provider = str(payload.get("provider", "modal")).strip() or "modal"
    predict_url = payload.get("predict_url")
    health_url = payload.get("health_url")
    if not isinstance(predict_url, str) or not predict_url.strip():
        raise RuntimeError(f"Remote model '{model_id}' must define a non-empty predict_url.")
    if not isinstance(health_url, str) or not health_url.strip():
        raise RuntimeError(f"Remote model '{model_id}' must define a non-empty health_url.")
    timeout_raw = payload.get("timeout_s", 30.0)
    try:
        timeout_s = float(timeout_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Remote model '{model_id}' has an invalid timeout_s value.") from exc
    label = payload.get("label")
    return RemoteModelConfig(
        provider=provider,
        predict_url=predict_url.strip(),
        health_url=health_url.strip(),
        label=label.strip() if isinstance(label, str) and label.strip() else None,
        timeout_s=timeout_s,
        auth_token=_remote_auth_token(provider, payload),
    )


def _parse_local_model_config(model_id: str, payload: Dict[str, object]) -> LocalModelConfig:
    bundle_setting = payload.get("bundle_dir") or payload.get("bundle_path")
    if not isinstance(bundle_setting, str) or not bundle_setting.strip():
        raise RuntimeError(f"Local model '{model_id}' must define a non-empty bundle_dir.")
    label = payload.get("label")
    return LocalModelConfig(
        bundle_dir=_resolve_bundle_dir(bundle_setting.strip()),
        label=label.strip() if isinstance(label, str) and label.strip() else None,
    )


@lru_cache(maxsize=1)
def get_configured_models() -> Dict[str, ModelConfig]:
    model_registry = os.getenv("SNAPCAL_MODEL_REGISTRY", "").strip()
    if model_registry:
        try:
            payload = json.loads(model_registry)
        except json.JSONDecodeError as exc:
            raise RuntimeError("SNAPCAL_MODEL_REGISTRY must be a JSON object mapping model ids to model configs.") from exc
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError("SNAPCAL_MODEL_REGISTRY must be a non-empty JSON object mapping model ids to configs.")
        models: Dict[str, ModelConfig] = {}
        for raw_model_id, raw_config in payload.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                raise RuntimeError("SNAPCAL_MODEL_REGISTRY keys must be non-empty strings.")
            if not isinstance(raw_config, dict):
                raise RuntimeError("SNAPCAL_MODEL_REGISTRY values must be JSON objects.")
            model_id = raw_model_id.strip()
            config_type = str(raw_config.get("type", "remote")).strip().lower() or "remote"
            if config_type == "local":
                models[model_id] = _parse_local_model_config(model_id, raw_config)
                continue
            if config_type == "remote":
                models[model_id] = _parse_remote_model_config(model_id, raw_config)
                continue
            raise RuntimeError(f"Model '{model_id}' has unsupported config type '{config_type}'.")
        return models

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
        models: Dict[str, ModelConfig] = {}
        for raw_model_id, raw_bundle_dir in payload.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                raise RuntimeError("SNAPCAL_MODEL_BUNDLES keys must be non-empty strings.")
            if not isinstance(raw_bundle_dir, str) or not raw_bundle_dir.strip():
                raise RuntimeError("SNAPCAL_MODEL_BUNDLES values must be non-empty bundle directory strings.")
            model_id = raw_model_id.strip()
            models[model_id] = LocalModelConfig(bundle_dir=_resolve_bundle_dir(raw_bundle_dir.strip()))
        return models

    bundle_setting = os.getenv("SNAPCAL_MODEL_BUNDLE", DEFAULT_BUNDLE_DIR)
    return _discover_local_bundles(_resolve_bundle_dir(bundle_setting))


def get_default_model_id() -> Optional[str]:
    models = get_configured_models()
    configured_default = os.getenv("SNAPCAL_DEFAULT_MODEL_ID", "").strip()
    if configured_default and configured_default in models:
        return configured_default
    return next(iter(models), None)


def resolve_model_id(requested_model_id: Optional[str]) -> str:
    model_id = (requested_model_id or "").strip() or get_default_model_id()
    if not model_id:
        raise RuntimeError("No model bundles are configured for prediction.")
    if model_id not in get_configured_models():
        available_models = ", ".join(sorted(get_configured_models()))
        raise UnknownModelError(f"Unknown model_id '{model_id}'. Available models: {available_models}")
    return model_id


def _label_for_model(config_label: Optional[str], model_name: Optional[str], model_id: str) -> str:
    if config_label:
        return config_label
    if model_name and model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    return model_id.replace("_", " ")


@lru_cache(maxsize=4)
def get_prediction_runtime(model_id: str):
    config = get_configured_models().get(model_id)
    if config is None:
        available_models = ", ".join(sorted(get_configured_models()))
        raise UnknownModelError(f"Unknown model_id '{model_id}'. Available models: {available_models}")
    if isinstance(config, LocalModelConfig):
        return PredictionRuntime(bundle_dir=config.bundle_dir, enable_segmentation=_segmentation_enabled())
    return RemotePredictionRuntime(
        model_id=model_id,
        config=config,
        mapping_path=_resolve_mapping_path(),
        enable_segmentation=_segmentation_enabled(),
    )


def get_prediction_health() -> dict:
    try:
        configured_models = get_configured_models()
    except RuntimeError as exc:
        return {
            "ready": False,
            "default_model_id": None,
            "models": [],
            "error": str(exc),
        }

    default_model_id = get_default_model_id()
    model_entries = []
    any_ready = False
    default_entry = None

    for model_id, config in configured_models.items():
        entry = {
            "id": model_id,
            "label": _label_for_model(getattr(config, "label", None), None, model_id),
        }
        if isinstance(config, LocalModelConfig):
            entry["bundle_dir"] = str(config.bundle_dir)
        status = get_prediction_runtime(model_id).health()
        entry.update(status)
        entry["label"] = _label_for_model(getattr(config, "label", None), entry.get("model_name"), model_id)
        model_entries.append(entry)
        any_ready = any_ready or bool(entry.get("ready"))
        if model_id == default_model_id:
            default_entry = entry

    if default_entry is None and model_entries:
        default_entry = model_entries[0]
        default_model_id = str(default_entry["id"])

    payload = {
        "ready": any_ready,
        "default_model_id": default_model_id,
        "models": model_entries,
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
