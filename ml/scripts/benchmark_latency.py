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
from snapcal.constants import (
    LIQUID_PORTION_FLUID_OUNCE_VALUES,
    PORTION_UNIT_FL_OZ,
    PORTION_UNIT_OZ,
    PORTION_UNIT_SERVING,
    SOLID_PORTION_OUNCE_VALUES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--portion-unit",
        choices=(PORTION_UNIT_SERVING, PORTION_UNIT_OZ, PORTION_UNIT_FL_OZ),
        default=PORTION_UNIT_SERVING,
    )
    parser.add_argument("--portion-value", type=int)
    parser.add_argument("--enable-segmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.portion_unit == PORTION_UNIT_SERVING:
        args.portion_value = None
    else:
        allowed_values = (
            SOLID_PORTION_OUNCE_VALUES
            if args.portion_unit == PORTION_UNIT_OZ
            else LIQUID_PORTION_FLUID_OUNCE_VALUES
        )
        if args.portion_value not in allowed_values:
            raise SystemExit(
                f"--portion-value must be one of {allowed_values} for unit '{args.portion_unit}'."
            )
    service = LocalInferenceService(args.bundle_dir)
    image_bytes = args.image.read_bytes()
    start = time.perf_counter()
    response = service.predict(
        image_bytes=image_bytes,
        portion_unit=args.portion_unit,
        portion_value=args.portion_value,
        enable_segmentation=args.enable_segmentation,
    )
    total_ms = (time.perf_counter() - start) * 1000.0
    print(f"Selected class: {response.selected_class}")
    print(f"Requested portion: {response.requested_portion.label}")
    print(
        "Processing mode: "
        + ("segmented" if response.segmentation_applied else "raw")
    )
    print(f"Reported latency: {response.latency_ms}")
    print(f"Measured wall time: {total_ms:.3f} ms")


if __name__ == "__main__":
    main()
