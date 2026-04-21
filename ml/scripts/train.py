#!/usr/bin/env python3
"""Train a Food-101 classifier from a JSON config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from snapcal.config import TrainingConfig
from snapcal.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_json(args.config)
    artifacts = Trainer(config).fit()
    print(f"Best checkpoint: {artifacts.checkpoint_path}")
    print(f"Report: {artifacts.report_path}")
    print(f"Predictions: {artifacts.predictions_path}")


if __name__ == "__main__":
    main()
