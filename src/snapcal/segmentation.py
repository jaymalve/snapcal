"""MobileSAM scoring and segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json
import math

import numpy as np
from PIL import Image

from .config import SegmentationConfig


@dataclass
class MaskCandidate:
    mask: np.ndarray
    area: int
    predicted_iou: float
    stability_score: float
    bbox: Tuple[int, int, int, int]
    center_distance: float
    area_ratio: float
    score: float


def _candidate_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _center_distance(bbox: Tuple[int, int, int, int], width: int, height: int) -> float:
    x, y, w, h = bbox
    candidate_center = (x + (w / 2.0), y + (h / 2.0))
    image_center = (width / 2.0, height / 2.0)
    max_distance = math.sqrt(image_center[0] ** 2 + image_center[1] ** 2)
    distance = math.sqrt((candidate_center[0] - image_center[0]) ** 2 + (candidate_center[1] - image_center[1]) ** 2)
    return 0.0 if max_distance == 0 else distance / max_distance


def rank_masks(
    raw_masks: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
    config: SegmentationConfig,
) -> List[MaskCandidate]:
    width, height = image_size
    image_area = max(1, width * height)
    candidates: List[MaskCandidate] = []
    for raw_mask in raw_masks:
        segmentation = np.asarray(raw_mask["segmentation"], dtype=np.uint8)
        area = int(raw_mask.get("area", int(segmentation.sum())))
        area_ratio = area / image_area
        if area < config.min_mask_region_area or area_ratio < config.min_area_ratio:
            continue
        bbox = tuple(raw_mask.get("bbox", _candidate_bbox(segmentation)))
        predicted_iou = float(raw_mask.get("predicted_iou", 0.0))
        stability_score = float(raw_mask.get("stability_score", 0.0))
        center_distance = _center_distance(bbox, width, height)
        score = (
            (0.45 * area_ratio)
            + (0.25 * predicted_iou)
            + (0.20 * stability_score)
            + (0.10 * (1.0 - center_distance))
        )
        candidates.append(
            MaskCandidate(
                mask=segmentation,
                area=area,
                predicted_iou=predicted_iou,
                stability_score=stability_score,
                bbox=bbox,  # type: ignore[arg-type]
                center_distance=center_distance,
                area_ratio=area_ratio,
                score=score,
            )
        )
    if not candidates and raw_masks:
        fallback = max(raw_masks, key=lambda mask: float(mask.get("area", 0.0)))
        segmentation = np.asarray(fallback["segmentation"], dtype=np.uint8)
        bbox = tuple(fallback.get("bbox", _candidate_bbox(segmentation)))
        area = int(fallback.get("area", int(segmentation.sum())))
        area_ratio = area / image_area
        candidates.append(
            MaskCandidate(
                mask=segmentation,
                area=area,
                predicted_iou=float(fallback.get("predicted_iou", 0.0)),
                stability_score=float(fallback.get("stability_score", 0.0)),
                bbox=bbox,  # type: ignore[arg-type]
                center_distance=_center_distance(bbox, width, height),
                area_ratio=area_ratio,
                score=area_ratio,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def select_masks(candidates: Sequence[MaskCandidate], config: SegmentationConfig) -> List[MaskCandidate]:
    if not candidates:
        return []
    selected = [candidates[0]]
    if len(candidates) > 1 and (candidates[0].score - candidates[1].score) <= config.secondary_mask_score_delta:
        selected.append(candidates[1])
    return selected


def combine_masks(candidates: Sequence[MaskCandidate], image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    combined = np.zeros((height, width), dtype=np.uint8)
    for candidate in candidates:
        combined = np.maximum(combined, candidate.mask.astype(np.uint8))
    return combined


def crop_with_margin(mask: np.ndarray, margin_ratio: float) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    margin_x = max(1, round((x1 - x0 + 1) * margin_ratio))
    margin_y = max(1, round((y1 - y0 + 1) * margin_ratio))
    return (
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(mask.shape[1], x1 + margin_x + 1),
        min(mask.shape[0], y1 + margin_y + 1),
    )


def apply_mask(
    image: Image.Image,
    mask: np.ndarray,
    config: SegmentationConfig,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    rgb_image = image.convert("RGB")
    rgb_array = np.asarray(rgb_image, dtype=np.uint8)
    if mask.max() == 0:
        mask = np.ones((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)
        status = "fallback_full_frame"
    else:
        status = "segmented"
    fill = np.array(config.background_fill_rgb, dtype=np.uint8)
    composed = np.where(mask[..., None] > 0, rgb_array, fill)
    x0, y0, x1, y1 = crop_with_margin(mask, config.crop_margin_ratio)
    cropped = composed[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1] * 255
    segmented = Image.fromarray(cropped.astype(np.uint8), mode="RGB").resize(
        (config.output_image_size, config.output_image_size),
        resample=Image.Resampling.BILINEAR,
    )
    mask_image = Image.fromarray(cropped_mask.astype(np.uint8), mode="L").resize(
        (config.output_image_size, config.output_image_size),
        resample=Image.Resampling.NEAREST,
    )
    meta = {
        "status": status,
        "crop_box": [x0, y0, x1, y1],
        "area_ratio": round(float(mask.sum()) / float(mask.shape[0] * mask.shape[1]), 6),
    }
    return segmented, mask_image, meta


def serialize_segmentation_meta(
    raw_masks: Sequence[Dict[str, Any]],
    selected: Sequence[MaskCandidate],
    extra_meta: Dict[str, Any],
) -> str:
    payload = {
        "candidate_count": len(raw_masks),
        "selected_masks": [
            {
                "area": candidate.area,
                "area_ratio": round(candidate.area_ratio, 6),
                "predicted_iou": round(candidate.predicted_iou, 6),
                "stability_score": round(candidate.stability_score, 6),
                "score": round(candidate.score, 6),
                "bbox": list(candidate.bbox),
            }
            for candidate in selected
        ],
    }
    payload.update(extra_meta)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class MobileSAMSegmenter:
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self._generator = None

    def _load_generator(self) -> Any:
        if self._generator is not None:
            return self._generator
        try:
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("mobile-sam is not installed. Install snapcal[train] to enable segmentation.") from exc

        sam = sam_model_registry[self.config.model_type](checkpoint=str(self.config.checkpoint_path))
        self._generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            crop_n_layers=self.config.crop_n_layers,
            min_mask_region_area=self.config.min_mask_region_area,
        )
        return self._generator

    def segment_image(self, image: Image.Image) -> Tuple[Image.Image, Image.Image, str]:
        generator = self._load_generator()
        rgb_array = np.asarray(image.convert("RGB"))
        raw_masks = generator.generate(rgb_array)
        ranked = rank_masks(raw_masks, image.size, self.config)
        selected = select_masks(ranked, self.config)
        combined = combine_masks(selected, image.size) if selected else np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        segmented_image, mask_image, extra_meta = apply_mask(image, combined, self.config)
        meta_json = serialize_segmentation_meta(raw_masks, selected, extra_meta)
        return segmented_image, mask_image, meta_json

    def segment_path(self, source_path: Path, segmented_path: Path, mask_path: Path) -> str:
        segmented_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            segmented_image, mask_image, meta_json = self.segment_image(image)
        segmented_image.save(segmented_path)
        mask_image.save(mask_path)
        return meta_json
