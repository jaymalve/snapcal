"""Food-101 manifest generation helpers."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .constants import FOOD101_CLASSES, MANIFEST_COLUMNS


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    split: str
    class_index: int
    class_name: str
    image_path: str
    segmented_image_path: str
    mask_path: str
    segmentation_meta_json: str

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "image_id": self.image_id,
            "split": self.split,
            "class_index": str(self.class_index),
            "class_name": self.class_name,
            "image_path": self.image_path,
            "segmented_image_path": self.segmented_image_path,
            "mask_path": self.mask_path,
            "segmentation_meta_json": self.segmentation_meta_json,
        }


def _read_meta_split(meta_file: Path) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    with meta_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            image_stub = raw_line.strip()
            if not image_stub:
                continue
            class_name, _ = image_stub.split("/", 1)
            grouped.setdefault(class_name, []).append(image_stub)
    return grouped


def build_train_val_assignments(
    grouped_train_records: Dict[str, List[str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    assignments: Dict[str, str] = {}
    for class_name, image_stubs in grouped_train_records.items():
        shuffled = list(sorted(image_stubs))
        random.Random(f"{seed}:{class_name}").shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_set = set(shuffled[:val_count])
        for image_stub in shuffled:
            assignments[image_stub] = "val" if image_stub in val_set else "train"
    return assignments


def _default_segmentation_meta() -> str:
    return json.dumps(
        {
            "status": "pending",
            "selected_masks": [],
            "candidate_count": 0,
            "area_ratio": None,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def build_manifest_rows(
    dataset_root: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    processed_root: Path = Path("data/processed"),
) -> List[ManifestRow]:
    meta_root = dataset_root / "meta"
    images_root = dataset_root / "images"
    train_grouped = _read_meta_split(meta_root / "train.txt")
    test_grouped = _read_meta_split(meta_root / "test.txt")
    assignments = build_train_val_assignments(train_grouped, val_ratio=val_ratio, seed=seed)
    rows: List[ManifestRow] = []
    for grouped, fixed_split in ((train_grouped, None), (test_grouped, "test")):
        for class_name, image_stubs in grouped.items():
            if class_name not in FOOD101_CLASSES:
                raise ValueError(f"Unexpected Food-101 class '{class_name}' in dataset")
            class_index = FOOD101_CLASSES.index(class_name)
            for image_stub in sorted(image_stubs):
                split = fixed_split or assignments[image_stub]
                image_path = images_root / f"{image_stub}.jpg"
                image_id = image_stub.replace("/", "__")
                segmented_image_path = processed_root / "segmented" / split / class_name / f"{image_id}.png"
                mask_path = processed_root / "masks" / split / class_name / f"{image_id}.png"
                rows.append(
                    ManifestRow(
                        image_id=image_id,
                        split=split,
                        class_index=class_index,
                        class_name=class_name,
                        image_path=str(image_path),
                        segmented_image_path=str(segmented_image_path),
                        mask_path=str(mask_path),
                        segmentation_meta_json=_default_segmentation_meta(),
                    )
                )
    return rows


def write_manifest_csv(rows: Sequence[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def read_manifest_csv(path: Path) -> List[ManifestRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw_row in reader:
            rows.append(
                ManifestRow(
                    image_id=raw_row["image_id"],
                    split=raw_row["split"],
                    class_index=int(raw_row["class_index"]),
                    class_name=raw_row["class_name"],
                    image_path=raw_row["image_path"],
                    segmented_image_path=raw_row["segmented_image_path"],
                    mask_path=raw_row["mask_path"],
                    segmentation_meta_json=raw_row["segmentation_meta_json"],
                )
            )
    return rows
