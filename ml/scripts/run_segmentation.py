#!/usr/bin/env python3
"""Precompute MobileSAM segmentation outputs and refresh manifest metadata."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.config import SegmentationConfig
from snapcal.manifests import ManifestRow, read_manifest_csv, write_manifest_csv
from snapcal.segmentation import MobileSAMSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SegmentationConfig.from_json(args.config)
    segmenter = MobileSAMSegmenter(config)
    rows = read_manifest_csv(args.manifest)
    eligible_total = sum(1 for row in rows if not args.split or row.split == args.split)
    target_total = min(eligible_total, args.limit) if args.limit is not None else eligible_total
    print(f"Preparing to segment {target_total} images")
    updated_rows = []
    processed = 0
    for row in rows:
        if args.split and row.split != args.split:
            updated_rows.append(row)
            continue
        if args.limit is not None and processed >= args.limit:
            updated_rows.append(row)
            continue
        meta_json = segmenter.segment_path(
            source_path=Path(row.image_path),
            segmented_path=Path(row.segmented_image_path),
            mask_path=Path(row.mask_path),
        )
        updated_rows.append(replace(row, segmentation_meta_json=meta_json))
        processed += 1
        if processed == 1 or processed % 10 == 0 or processed == target_total:
            print(f"Segmented {processed}/{target_total}: {row.image_id}")
    output_manifest = args.output_manifest or args.manifest
    write_manifest_csv(updated_rows, output_manifest)
    print(f"Segmented {processed} images and wrote updated manifest to {output_manifest}")


if __name__ == "__main__":
    main()
