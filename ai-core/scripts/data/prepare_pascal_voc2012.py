#!/usr/bin/env python3
"""
Download and prepare Pascal VOC 2012 segmentation data as a semantic source.

The output layout is compatible with build_cutout_dataset.py --semantic:

    ai-core/data/sources/voc2012/
        images/
        masks/
        label_map.json
        VOCdevkit/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


LOGGER = logging.getLogger("prepare_pascal_voc2012")

VOC_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    "VOCtrainval_11-May-2012.tar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Pascal VOC 2012 segmentation data for cutout training."
    )
    parser.add_argument(
        "--source-dir",
        default="ai-core/data/sources/voc2012",
        help="Directory where VOC files and prepared semantic source are stored.",
    )
    parser.add_argument("--url", default=VOC_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--clean-prepared",
        action="store_true",
        help="Remove prepared images/masks before recreating symlinks.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Using existing file: %s", destination)
        return

    LOGGER.info("Downloading %s -> %s", url, destination)
    if shutil.which("wget"):
        subprocess.run(
            [
                "wget",
                "--continue",
                "--timeout",
                str(timeout),
                "--output-document",
                str(destination),
                url,
            ],
            check=True,
        )
        return

    with urllib.request.urlopen(url, timeout=timeout) as response:
        with tempfile.NamedTemporaryFile(
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            delete=False,
        ) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(destination)


def extract_archive(archive_path: Path, source_dir: Path) -> None:
    voc_root = source_dir / "VOCdevkit" / "VOC2012"
    if (voc_root / "ImageSets" / "Segmentation" / "trainval.txt").is_file():
        LOGGER.info("Using existing VOC extraction: %s", voc_root)
        return

    LOGGER.info("Extracting %s -> %s", archive_path, source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    resolved_source_dir = source_dir.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            destination = (source_dir / member.name).resolve()
            if not destination.is_relative_to(resolved_source_dir):
                raise ValueError(f"Unsafe archive member path: {member.name}")
        archive.extractall(source_dir)


def reset_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def prepare_semantic_source(source_dir: Path, clean_prepared: bool) -> None:
    voc_root = source_dir / "VOCdevkit" / "VOC2012"
    split_path = voc_root / "ImageSets" / "Segmentation" / "trainval.txt"
    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    if not split_path.is_file():
        raise FileNotFoundError(f"VOC split file not found: {split_path}")

    prepared_images = source_dir / "images"
    prepared_masks = source_dir / "masks"
    reset_dir(prepared_images, clean=clean_prepared)
    reset_dir(prepared_masks, clean=clean_prepared)

    sample_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    linked = 0
    for sample_id in sample_ids:
        image_path = jpeg_dir / f"{sample_id}.jpg"
        mask_path = mask_dir / f"{sample_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            continue
        link_or_copy(image_path, prepared_images / image_path.name)
        link_or_copy(mask_path, prepared_masks / mask_path.name)
        linked += 1

    label_map = source_dir / "label_map.json"
    label_map.write_text(
        '{\n  "person": [15],\n  "car": [7]\n}\n',
        encoding="utf-8",
    )
    LOGGER.info(
        "Prepared VOC semantic source: images=%d | image_dir=%s | mask_dir=%s",
        linked,
        prepared_images,
        prepared_masks,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    source_dir = Path(args.source_dir)
    archive_path = source_dir / "downloads" / "VOCtrainval_11-May-2012.tar"

    download_file(args.url, archive_path, timeout=args.timeout)
    extract_archive(archive_path, source_dir)
    prepare_semantic_source(source_dir, clean_prepared=args.clean_prepared)


if __name__ == "__main__":
    main()
