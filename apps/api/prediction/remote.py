"""Remote inference runtimes for hosted model backends."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import importlib.util
import io
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from snapcal.constants import FOOD101_CLASSES
from snapcal.config import SegmentationConfig
from snapcal.inference import InferenceNotReadyError
from snapcal.nutrition import NutritionLookup
from snapcal.segmentation import MobileSAMSegmenter

REPO_ROOT = Path(__file__).resolve().parents[3]
SEGMENTATION_CONFIG_ENV = "SNAPCAL_SEGMENTATION_CONFIG"
REMOTE_SEGMENTATION_DISABLED_REASON = (
    "Segmentation is disabled on this backend. Restart with SNAPCAL_ENABLE_SEGMENTATION=true to enable the comparison toggle."
)
DEFAULT_REMOTE_TIMEOUT_S = 30.0
DEFAULT_TOP_K = 3

_SEGMENTATION_CONFIG_CACHE: Optional[SegmentationConfig] = None
_SEGMENTER_CACHE: Optional[MobileSAMSegmenter] = None


@dataclass(frozen=True)
class RemoteModelConfig:
    provider: str
    predict_url: str
    health_url: str
    label: Optional[str] = None
    timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S
    auth_token: Optional[str] = None


class RemotePredictionRuntime:
    def __init__(
        self,
        model_id: str,
        config: RemoteModelConfig,
        mapping_path: Path,
        enable_segmentation: bool = False,
    ):
        self.model_id = model_id
        self.config = config
        self.mapping_path = mapping_path
        self.enable_segmentation = enable_segmentation
        self.nutrition = NutritionLookup.from_csv(mapping_path)

    @staticmethod
    def _resolve_repo_path(path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    def _load_segmentation_config(self) -> SegmentationConfig:
        global _SEGMENTATION_CONFIG_CACHE

        if _SEGMENTATION_CONFIG_CACHE is not None:
            return _SEGMENTATION_CONFIG_CACHE

        config_setting = (os.getenv(SEGMENTATION_CONFIG_ENV, "") or "").strip()
        if not config_setting:
            raise InferenceNotReadyError(
                f"Segmentation config is not set. Define {SEGMENTATION_CONFIG_ENV} on this backend to enable the comparison toggle."
            )

        config_path = self._resolve_repo_path(config_setting)
        if not config_path.exists():
            raise InferenceNotReadyError(f"Segmentation config not found: {config_path}")

        try:
            config = SegmentationConfig.from_json(config_path)
        except Exception as exc:
            raise InferenceNotReadyError(f"Segmentation config could not be loaded: {exc}") from exc

        if not config.checkpoint_path.is_absolute():
            config = replace(config, checkpoint_path=(REPO_ROOT / config.checkpoint_path).resolve())

        _SEGMENTATION_CONFIG_CACHE = config
        return config

    def _segmentation_status(self) -> Tuple[bool, Optional[str]]:
        if not self.enable_segmentation:
            return False, REMOTE_SEGMENTATION_DISABLED_REASON

        try:
            config = self._load_segmentation_config()
        except InferenceNotReadyError as exc:
            return False, str(exc)

        if not config.checkpoint_path.exists():
            return False, f"Segmentation checkpoint not found: {config.checkpoint_path}"
        if importlib.util.find_spec("mobile_sam") is None:
            return False, "MobileSAM is not installed on this backend."
        return True, None

    def _load_segmenter(self) -> MobileSAMSegmenter:
        global _SEGMENTER_CACHE

        if _SEGMENTER_CACHE is not None:
            return _SEGMENTER_CACHE

        config = self._load_segmentation_config()
        _SEGMENTER_CACHE = MobileSAMSegmenter(config)
        return _SEGMENTER_CACHE

    def _segment_image_bytes(self, image_bytes: bytes) -> Tuple[bytes, float]:
        from PIL import Image

        segmenter = self._load_segmenter()
        preprocess_start = time.perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as input_image:
            segmented_image, _mask_image, _meta_json = segmenter.segment_image(input_image.convert("RGB"))
        buffer = io.BytesIO()
        segmented_image.save(buffer, format="PNG")
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        return buffer.getvalue(), preprocess_ms

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    def _request_json(self, url: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.config.timeout_s) as response:
                response_body = response.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} request failed for model '{self.model_id}' with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib_error.URLError as exc:
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} request failed for model '{self.model_id}': {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} request timed out for model '{self.model_id}'."
            ) from exc

        try:
            payload_obj = json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} response for model '{self.model_id}' was not valid JSON."
            ) from exc
        if not isinstance(payload_obj, dict):
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} response for model '{self.model_id}' must be a JSON object."
            )
        return payload_obj

    def health(self) -> Dict[str, object]:
        segmentation_available, segmentation_reason = self._segmentation_status()
        try:
            payload = self._request_json(self.config.health_url)
        except InferenceNotReadyError as exc:
            return {
                "ready": False,
                "model_name": self.model_id,
                "segmentation_available": segmentation_available,
                "segmentation_reason": segmentation_reason,
                "error": str(exc),
            }

        ready = bool(payload.get("ready", True))
        response = {
            "ready": ready,
            "model_name": str(payload.get("model_name", self.model_id)),
            "model_version": str(payload.get("model_version", "remote")),
            "segmentation_available": segmentation_available,
            "segmentation_reason": segmentation_reason,
        }
        if not ready and payload.get("detail"):
            response["error"] = str(payload["detail"])
        return response

    def _normalize_predictions(self, payload: Dict[str, object]) -> List[Tuple[str, float]]:
        raw_indices = payload.get("topk_indices")
        raw_scores = payload.get("topk_scores")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} response for model '{self.model_id}' is missing topk_indices."
            )
        if not isinstance(raw_scores, list) or len(raw_scores) != len(raw_indices):
            raise InferenceNotReadyError(
                f"Remote {self.config.provider} response for model '{self.model_id}' has mismatched top-k payloads."
            )

        ranked_predictions: List[Tuple[str, float]] = []
        for raw_index, raw_score in zip(raw_indices, raw_scores):
            try:
                index = int(raw_index)
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise InferenceNotReadyError(
                    f"Remote {self.config.provider} response for model '{self.model_id}' contains invalid top-k values."
                ) from exc
            if index < 0 or index >= len(FOOD101_CLASSES):
                raise InferenceNotReadyError(
                    f"Remote {self.config.provider} response for model '{self.model_id}' returned an out-of-range class index {index}."
                )
            ranked_predictions.append((FOOD101_CLASSES[index], score))
        return ranked_predictions

    @staticmethod
    def _normalize_latency(payload: Dict[str, object]) -> Dict[str, float]:
        raw_latency = payload.get("latency_ms")
        if not isinstance(raw_latency, dict):
            return {}
        latency: Dict[str, float] = {}
        for key, value in raw_latency.items():
            try:
                latency[str(key)] = round(float(value), 3)
            except (TypeError, ValueError):
                continue
        if "total" not in latency and "inference" in latency:
            latency["total"] = latency["inference"]
        return latency

    def predict(
        self,
        image_bytes: bytes,
        portion_unit: str,
        portion_value: Optional[int],
        enable_segmentation: bool,
        model_id: Optional[str] = None,
    ):
        segmentation_requested = enable_segmentation
        segmentation_applied = False
        preprocess_ms = 0.0

        if enable_segmentation:
            segmentation_available, segmentation_reason = self._segmentation_status()
            if not segmentation_available:
                raise InferenceNotReadyError(segmentation_reason or "Segmentation is not available.")
            image_bytes, preprocess_ms = self._segment_image_bytes(image_bytes)
            segmentation_applied = True

        payload = self._request_json(
            self.config.predict_url,
            payload={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "top_k": DEFAULT_TOP_K,
            },
        )
        ranked_predictions = self._normalize_predictions(payload)
        latency_ms = self._normalize_latency(payload)
        latency_ms["preprocess"] = round(preprocess_ms, 3)
        latency_ms["total"] = round(preprocess_ms + latency_ms.get("total", latency_ms.get("inference", 0.0)), 3)
        model_name = str(payload.get("model_name", self.model_id))
        model_version = str(payload.get("model_version", "remote"))
        return self.nutrition.build_response(
            ranked_predictions=ranked_predictions,
            requested_portion_unit=portion_unit,
            requested_portion_value=portion_value,
            model_id=model_id or self.model_id,
            model_name=model_name,
            model_version=model_version,
            segmentation_requested=segmentation_requested,
            segmentation_applied=segmentation_applied,
            segmentation_preview_url=None,
            latency_ms=latency_ms,
            warnings=[],
        )
