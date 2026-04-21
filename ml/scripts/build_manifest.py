#!/usr/bin/env python3
"""Generate a canonical Food-101 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.manifests import build_manifest_rows, write_manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest_rows(
        dataset_root=args.dataset_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        processed_root=args.processed_root,
    )
    write_manifest_csv(rows, args.output)
    print(f"Wrote {len(rows)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
