#!/usr/bin/env python3
"""Package a trained checkpoint for API inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.config import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("data/reference/food101_usda_mapping.csv"))
    parser.add_argument("--segmentation-config", type=Path, default=Path("configs/segmentation/mobilesam_default.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "model.pt"
    mapping_name = "nutrition_mapping.csv"
    segmentation_name = "segmentation_config.json"
    shutil.copy2(args.checkpoint, args.output_dir / checkpoint_name)
    shutil.copy2(args.mapping, args.output_dir / mapping_name)
    if args.segmentation_config.exists():
        shutil.copy2(args.segmentation_config, args.output_dir / segmentation_name)
        segmentation_ref = segmentation_name
    else:
        segmentation_ref = None
    metadata = {
        "model_name": config.model_name,
        "model_version": args.model_version,
        "checkpoint_path": checkpoint_name,
        "nutrition_mapping_path": mapping_name,
        "segmentation_config_path": segmentation_ref,
        "image_size": config.image_size,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Exported inference bundle to {args.output_dir}")


if __name__ == "__main__":
    main()
