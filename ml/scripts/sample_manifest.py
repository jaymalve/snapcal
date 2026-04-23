#!/usr/bin/env python3
"""Create a smaller manifest for quick local experiments."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import DefaultDict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.manifests import ManifestRow, read_manifest_csv, write_manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--class-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _limit_for_split(args: argparse.Namespace, split: str) -> int:
    return {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }[split]


def main() -> None:
    args = parse_args()
    rows = read_manifest_csv(args.manifest)
    grouped: DefaultDict[Tuple[str, str], List[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.split, row.class_name)].append(row)

    class_names = sorted({row.class_name for row in rows})
    if args.class_limit is not None:
        random.Random(args.seed).shuffle(class_names)
        class_names = sorted(class_names[: args.class_limit])
    selected_classes = set(class_names)

    sampled_rows: List[ManifestRow] = []
    for split in ("train", "val", "test"):
        limit = _limit_for_split(args, split)
        if limit <= 0:
            continue
        for class_name in class_names:
            class_rows = list(grouped[(split, class_name)])
            random.Random(f"{args.seed}:{split}:{class_name}").shuffle(class_rows)
            sampled_rows.extend(class_rows[:limit])

    sampled_rows.sort(key=lambda row: (row.split, row.class_name, row.image_id))
    write_manifest_csv(sampled_rows, args.output)

    total = len(sampled_rows)
    print(f"Wrote {total} rows across {len(selected_classes)} classes to {args.output}")
    for split in ("train", "val", "test"):
        split_count = sum(1 for row in sampled_rows if row.split == split)
        print(f"{split}: {split_count}")


if __name__ == "__main__":
    main()
