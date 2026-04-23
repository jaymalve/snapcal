#!/usr/bin/env python3
"""Download the MobileSAM checkpoint used by offline segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen


DEFAULT_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "cache" / "mobile_sam.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists() and not args.force:
        print(f"Checkpoint already exists at {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MobileSAM checkpoint to {output_path}")
    with urlopen(args.url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"Saved MobileSAM checkpoint to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
