"""Manifest-backed dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from .manifests import ManifestRow, read_manifest_csv


class Food101ManifestDataset:
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        dataset_variant: str,
        transform: Optional[Callable] = None,
        fallback_to_raw: bool = True,
    ):
        rows = read_manifest_csv(manifest_path)
        self.rows: List[ManifestRow] = [row for row in rows if row.split == split]
        self.dataset_variant = dataset_variant
        self.transform = transform
        self.fallback_to_raw = fallback_to_raw

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, row: ManifestRow) -> Path:
        if self.dataset_variant == "segmented":
            segmented = Path(row.segmented_image_path)
            if segmented.exists():
                return segmented
            if not self.fallback_to_raw:
                raise FileNotFoundError(f"Segmented image not found for {row.image_id}: {segmented}")
        return Path(row.image_path)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image_path = self._resolve_path(row)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": row.class_index,
            "class_name": row.class_name,
            "image_id": row.image_id,
            "segmentation_meta_json": row.segmentation_meta_json,
        }
