"""
Offline COCO data inspection and conversion utilities.

Usage:
    python -m src.tools.convert_coco --annotation_file data/hvm_coco_512/annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def inspect_annotations(annotation_file: str) -> dict:
    """Print summary statistics for a COCO annotation file."""
    with open(annotation_file, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    info: dict = {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "categories": {c["id"]: c["name"] for c in categories},
        "images_by_category": Counter(),
        "annotations_per_image": len(annotations) / max(len(images), 1),
        "image_sizes": set(),
    }

    ann_by_image = Counter()
    cat_counter = Counter()
    sizes = set()
    for ann in annotations:
        ann_by_image[ann["image_id"]] += 1
        cat_counter[categories[ann["category_id"] - 1]["name"]] += 1
    for img in images:
        sizes.add((img["width"], img["height"]))

    info["annotations_per_image_distribution"] = dict(
        Counter(ann_by_image.values())
    )
    info["category_distribution"] = dict(cat_counter)
    info["image_sizes"] = list(sizes)

    logger.info("=== COCO Dataset Summary ===")
    for k, v in info.items():
        logger.info("  %s: %s", k, v)

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or convert COCO-style annotations."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to COCO annotations.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    inspect_annotations(args.annotation_file)


if __name__ == "__main__":
    main()
