#!/usr/bin/env python3
"""Benchmark end-to-end inference latency for a single image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.inference import LocalInferenceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--portion-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LocalInferenceService(args.bundle_dir)
    image_bytes = args.image.read_bytes()
    start = time.perf_counter()
    response = service.predict(image_bytes=image_bytes, portion_multiplier=args.portion_multiplier)
    total_ms = (time.perf_counter() - start) * 1000.0
    print(f"Selected class: {response.selected_class}")
    print(f"Reported latency: {response.latency_ms}")
    print(f"Measured wall time: {total_ms:.3f} ms")


if __name__ == "__main__":
    main()
