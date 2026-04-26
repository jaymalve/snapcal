"""Modal deployment entrypoint for SnapCal remote classifier inference."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.constants import FOOD101_CLASSES
from snapcal.models import build_image_transforms, build_model, extract_logits

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
    if (root / "metadata.json").exists():
        return root
    candidates = sorted(root.rglob("metadata.json"), key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        raise HTTPException(status_code=500, detail=f"No bundle metadata found for model '{model_id}' in {root}.")
    return candidates[0].parent


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

    model = build_model(model_name, len(FOOD101_CLASSES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = build_image_transforms(image_size, train=False)
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
