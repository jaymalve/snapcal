#!/usr/bin/env python3
"""Rebuild an evaluation report from saved top-k predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.evaluation import save_report, summarize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.predictions.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = [item["label_index"] for item in payload]
    ranked_predictions = [item["topk_indices"] for item in payload]
    report = summarize_predictions(labels, ranked_predictions)
    save_report(report, args.output)
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
