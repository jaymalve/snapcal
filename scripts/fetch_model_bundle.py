#!/usr/bin/env python3
"""Fetch and normalize a production bundle archive for deployment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile


def _default_bundle_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_setting = os.getenv("SNAPCAL_MODEL_BUNDLE", "artifacts/models/production_bundle")
    bundle_dir = Path(bundle_setting)
    if bundle_dir.is_absolute():
        return bundle_dir
    return (repo_root / bundle_dir).resolve()


def _bundle_ready(bundle_dir: Path) -> bool:
    return (bundle_dir / "metadata.json").exists()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_archive(archive_path: Path, extract_root: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_root)
        return
    raise ValueError(f"Unsupported bundle archive format: {archive_path.name}")


def _normalize_extracted_bundle(extract_root: Path, target_dir: Path) -> None:
    metadata_candidates = sorted(extract_root.rglob("metadata.json"), key=lambda path: len(path.parts))
    if not metadata_candidates:
        raise FileNotFoundError("Downloaded archive does not contain metadata.json")
    source_dir = metadata_candidates[0].parent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)


def main() -> None:
    bundle_dir = _default_bundle_dir()
    if _bundle_ready(bundle_dir):
        print(f"Bundle already present at {bundle_dir}")
        return

    bundle_url = os.getenv("SNAPCAL_MODEL_BUNDLE_URL", "").strip()
    if not bundle_url:
        print(
            "SNAPCAL_MODEL_BUNDLE_URL is not set and no local production bundle was found. "
            "Continuing without a fetched bundle."
        )
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "bundle_archive"
        extract_root = temp_path / "extracted"
        print(f"Downloading model bundle from {bundle_url}")
        _download(bundle_url, archive_path)
        _extract_archive(archive_path, extract_root)
        _normalize_extracted_bundle(extract_root, bundle_dir)
    print(f"Prepared bundle at {bundle_dir}")


if __name__ == "__main__":
    main()
