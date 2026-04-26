"""Modal deployment entrypoint for SnapCal remote classifier inference."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

NUM_CLASSES = 101
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

APP_NAME = os.getenv("SNAPCAL_MODAL_APP_NAME", "snapcal-inference")
VOLUME_NAME = os.getenv("SNAPCAL_MODAL_VOLUME_NAME", "snapcal-models")
SECRET_NAME = os.getenv("SNAPCAL_MODAL_SECRET_NAME", "snapcal-modal-auth")
AUTH_ENV_KEY = os.getenv("SNAPCAL_MODAL_AUTH_ENV_KEY", "AUTH_TOKEN")

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi>=0.115,<1.0",
    "pydantic>=2.9,<3.0",
    "numpy>=1.26,<3.0",
    "Pillow>=10.4,<11.0",
    "torch>=2.4,<3.0",
    "torchvision>=0.19,<1.0",
    "transformers>=4.45,<5.0",
)
volume = modal.Volume.from_name(VOLUME_NAME)
secret = modal.Secret.from_name(SECRET_NAME)

MODEL_ROOTS = {
    "resnet50": Path("/models/resnet50"),
    "efficientnet_b0": Path("/models/efficientnet_b0"),
    "vit_b16": Path("/models/vit_b16"),
}
VOLUME_ROOT = Path("/models")


class PredictRequest(BaseModel):
    image_base64: str
    top_k: int = Field(default=3, ge=1, le=5)


@dataclass(frozen=True)
class BundleRuntime:
    bundle_dir: Path
    model_name: str
    model_version: str
    image_size: int
    checkpoint_path: Path
    model: object
    transform: object


_RUNTIME_CACHE: Dict[str, BundleRuntime] = {}
_METADATA_CACHE: Dict[str, Dict[str, object]] = {}


def build_image_transforms(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(model_name: str, num_classes: int):
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, resnet50

    if model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "vit_b16":
        from transformers import ViTConfig, ViTForImageClassification

        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=224,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            encoder_stride=16,
            num_labels=num_classes,
        )
        return ViTForImageClassification(config)
    raise ValueError(f"Unsupported model: {model_name}")


def extract_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _authorize(authorization: Optional[str]) -> None:
    expected = os.getenv(AUTH_ENV_KEY, "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="Modal auth secret is not configured.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1].strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid authorization token.")


def _resolve_bundle_dir(model_id: str) -> Path:
    try:
        root = MODEL_ROOTS[model_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'.") from exc

    candidate_roots = [root, VOLUME_ROOT]
    seen_dirs = set()
    metadata_candidates = []
    for search_root in candidate_roots:
        root_key = str(search_root)
        if root_key in seen_dirs or not search_root.exists():
            continue
        seen_dirs.add(root_key)
        metadata_candidates.extend(search_root.rglob("metadata.json"))

    best_match: Optional[Path] = None
    for metadata_path in sorted(metadata_candidates, key=lambda path: (len(path.parts), str(path))):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        remote_model_name = payload.get("model_name")
        if remote_model_name == model_id:
            return metadata_path.parent
        if best_match is None:
            best_match = metadata_path.parent

    if best_match is not None:
        return best_match
    raise HTTPException(status_code=500, detail=f"No bundle metadata found for model '{model_id}' under {VOLUME_ROOT}.")


def _bundle_metadata(model_id: str) -> Dict[str, object]:
    cached = _METADATA_CACHE.get(model_id)
    if cached is not None:
        return cached
    bundle_dir = _resolve_bundle_dir(model_id)
    metadata_path = bundle_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["bundle_dir"] = str(bundle_dir)
    _METADATA_CACHE[model_id] = payload
    return payload


def _load_runtime(model_id: str) -> BundleRuntime:
    cached = _RUNTIME_CACHE.get(model_id)
    if cached is not None:
        return cached

    import torch

    metadata = _bundle_metadata(model_id)
    bundle_dir = Path(str(metadata["bundle_dir"]))
    model_name = str(metadata["model_name"])
    model_version = str(metadata.get("model_version", model_id))
    image_size = int(metadata["image_size"])
    checkpoint_path = bundle_dir / str(metadata["checkpoint_path"])
    if not checkpoint_path.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoint not found for model '{model_id}': {checkpoint_path}")

    model = build_model(model_name, NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = build_image_transforms(image_size)
    runtime = BundleRuntime(
        bundle_dir=bundle_dir,
        model_name=model_name,
        model_version=model_version,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        model=model,
        transform=transform,
    )
    _RUNTIME_CACHE[model_id] = runtime
    return runtime


def _run_prediction(model_id: str, payload: PredictRequest) -> Dict[str, object]:
    import torch
    from PIL import Image

    runtime = _load_runtime(model_id)
    try:
        image_bytes = base64.b64decode(payload.image_base64)
    except Exception as exc:  # pragma: no cover - FastAPI validates structure, this guards content
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64.") from exc

    with Image.open(io.BytesIO(image_bytes)) as input_image:
        image = input_image.convert("RGB")
        tensor = runtime.transform(image).unsqueeze(0)

    inference_start = time.perf_counter()
    with torch.no_grad():
        outputs = runtime.model(pixel_values=tensor) if runtime.model_name == "vit_b16" else runtime.model(tensor)
        logits = extract_logits(outputs)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_k = min(payload.top_k, probabilities.size(0))
        top_scores, top_indices = torch.topk(probabilities, k=top_k)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    return {
        "ready": True,
        "model_name": runtime.model_name,
        "model_version": runtime.model_version,
        "topk_indices": [int(index) for index in top_indices.cpu().tolist()],
        "topk_scores": [round(float(score), 6) for score in top_scores.cpu().tolist()],
        "latency_ms": {
            "inference": round(inference_ms, 3),
            "total": round(inference_ms, 3),
        },
    }


def _health_payload(model_id: str) -> Dict[str, object]:
    metadata = _bundle_metadata(model_id)
    bundle_dir = Path(str(metadata["bundle_dir"]))
    checkpoint_path = bundle_dir / str(metadata["checkpoint_path"])
    return {
        "ready": checkpoint_path.exists(),
        "model_name": str(metadata["model_name"]),
        "model_version": str(metadata.get("model_version", model_id)),
    }


@app.function(image=image, secrets=[secret], volumes={"/models": volume})
@modal.fastapi_endpoint(method="GET")
def health_resnet50(authorization: Optional[str] = Header(default=None)) -> Dict[str, object]:
    _authorize(authorization)
    return _health_payload("resnet50")


@app.function(image=image, secrets=[secret], volumes={"/models": volume})
@modal.fastapi_endpoint(method="POST")
def predict_resnet50(
    payload: PredictRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, object]:
    _authorize(authorization)
    return _run_prediction("resnet50", payload)


@app.function(image=image, secrets=[secret], volumes={"/models": volume})
@modal.fastapi_endpoint(method="GET")
def health_efficientnet_b0(authorization: Optional[str] = Header(default=None)) -> Dict[str, object]:
    _authorize(authorization)
    return _health_payload("efficientnet_b0")


@app.function(image=image, secrets=[secret], volumes={"/models": volume})
@modal.fastapi_endpoint(method="POST")
def predict_efficientnet_b0(
    payload: PredictRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, object]:
    _authorize(authorization)
    return _run_prediction("efficientnet_b0", payload)


@app.function(image=image, secrets=[secret], volumes={"/models": volume})
@modal.fastapi_endpoint(method="GET")
def health_vit_b16(authorization: Optional[str] = Header(default=None)) -> Dict[str, object]:
    _authorize(authorization)
    return _health_payload("vit_b16")


@app.function(image=image, secrets=[secret], volumes={"/models": volume})
@modal.fastapi_endpoint(method="POST")
def predict_vit_b16(
    payload: PredictRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, object]:
    _authorize(authorization)
    return _run_prediction("vit_b16", payload)
