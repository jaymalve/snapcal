#!/usr/bin/env python3
"""Run a tiny forward-pass smoke test for a training config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.config import TrainingConfig
from snapcal.datasets import Food101ManifestDataset
from snapcal.models import build_image_transforms, build_model, extract_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for smoke testing.") from exc
    args = parse_args()
    config = TrainingConfig.from_json(args.config)
    dataset = Food101ManifestDataset(
        manifest_path=config.train_manifest,
        split=args.split,
        dataset_variant=config.dataset_variant,
        transform=build_image_transforms(config.image_size, train=False),
        fallback_to_raw=config.dataset_variant != "segmented",
    )
    model = build_model(config.model_name, config.num_classes)
    batch = [dataset[index]["image"] for index in range(min(args.count, len(dataset)))]
    tensor = torch.stack(batch)
    outputs = model(pixel_values=tensor) if config.model_name == "vit_b16" else model(tensor)
    logits = extract_logits(outputs)
    print(f"Smoke test logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
