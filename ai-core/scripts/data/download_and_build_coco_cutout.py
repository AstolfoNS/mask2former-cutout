#!/usr/bin/env python3
"""
Download a compact COCO person/car subset and build the project dataset format.

This script avoids downloading the full COCO train2017 image archive. It only:

1. Downloads the official COCO 2017 annotation zip.
2. Selects a small person/car subset from instances_train2017.json.
3. Downloads only the selected images.
4. Writes a subset COCO annotation JSON.
5. Calls build_cutout_dataset.py to create:

       output_dir/
           images/
           annotations.json
           quality_report.json

The final category ids are produced by build_cutout_dataset.py:
    1 = person
    2 = car
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


LOGGER = logging.getLogger("download_and_build_coco_cutout")

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"

COCO_PERSON_ID = 1
COCO_CAR_ID = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small COCO person/car subset and build cutout data."
    )
    parser.add_argument(
        "--source-dir",
        default="ai-core/data/sources/coco",
        help="Directory used for downloaded COCO source files.",
    )
    parser.add_argument(
        "--output-dir",
        default="ai-core/data/cutout_mix_512",
        help="Final processed dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
    )

    parser.add_argument("--final-total", type=int, default=1000)
    parser.add_argument("--final-both", type=int, default=550)
    parser.add_argument("--final-person-only", type=int, default=175)
    parser.add_argument("--final-car-only", type=int, default=175)
    parser.add_argument("--final-negative", type=int, default=0)
    parser.add_argument("--min-selected", type=int, default=800)
    parser.add_argument("--min-both-ratio", type=float, default=0.45)
    parser.add_argument(
        "--download-extra-ratio",
        type=float,
        default=2.5,
        help="Download extra source images so quality filters can reject bad samples.",
    )

    parser.add_argument("--min-image-side", type=int, default=384)
    parser.add_argument("--min-instance-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-instance-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-bbox-side", type=float, default=10.0)
    parser.add_argument("--max-foreground-ratio", type=float, default=0.75)
    parser.add_argument("--blur-threshold", type=float, default=30.0)

    parser.add_argument("--annotations-url", default=ANNOTATIONS_URL)
    parser.add_argument("--image-base-url", default=IMAGE_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--reuse-annotations",
        action="store_true",
        help="Skip annotation zip download if instances_train2017.json already exists.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the final processed output directory before writing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def download_file(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Using existing file: %s", destination)
        return

    LOGGER.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with tempfile.NamedTemporaryFile(
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            delete=False,
        ) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(destination)


def ensure_annotations(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_dir)
    annotations_dir = source_dir / "annotations"
    annotations_json = annotations_dir / "instances_train2017.json"
    if args.reuse_annotations and annotations_json.exists():
        LOGGER.info("Reusing annotation JSON: %s", annotations_json)
        return annotations_json

    archive_path = source_dir / "downloads" / "annotations_trainval2017.zip"
    download_file(args.annotations_url, archive_path, timeout=args.timeout)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        member = "annotations/instances_train2017.json"
        LOGGER.info("Extracting %s -> %s", member, annotations_json)
        with archive.open(member) as src, annotations_json.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    return annotations_json


def has_polygon_segmentation(annotation: Dict[str, Any]) -> bool:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return False
    return any(isinstance(poly, list) and len(poly) >= 6 for poly in segmentation)


def build_image_buckets(
    coco: Dict[str, Any],
) -> Tuple[Dict[str, List[int]], Dict[int, List[Dict[str, Any]]]]:
    valid_annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    class_ids_by_image: Dict[int, set[int]] = defaultdict(set)

    for annotation in coco.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))
        if category_id not in {COCO_PERSON_ID, COCO_CAR_ID}:
            continue
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        if not has_polygon_segmentation(annotation):
            continue

        image_id = int(annotation["image_id"])
        valid_annotations_by_image[image_id].append(annotation)
        class_ids_by_image[image_id].add(category_id)

    buckets = {
        "both": [],
        "person_only": [],
        "car_only": [],
    }
    for image_id, class_ids in class_ids_by_image.items():
        if COCO_PERSON_ID in class_ids and COCO_CAR_ID in class_ids:
            buckets["both"].append(image_id)
        elif COCO_PERSON_ID in class_ids:
            buckets["person_only"].append(image_id)
        elif COCO_CAR_ID in class_ids:
            buckets["car_only"].append(image_id)

    return buckets, valid_annotations_by_image


def quota(final_quota: int, extra_ratio: float) -> int:
    return max(final_quota, int(round(final_quota * extra_ratio)))


def select_image_ids(
    buckets: Dict[str, List[int]],
    args: argparse.Namespace,
) -> Dict[str, List[int]]:
    rng = random.Random(args.seed)
    requested = {
        "both": quota(args.final_both, args.download_extra_ratio),
        "person_only": quota(args.final_person_only, args.download_extra_ratio),
        "car_only": quota(args.final_car_only, args.download_extra_ratio),
    }

    selected: Dict[str, List[int]] = {}
    for bucket, image_ids in buckets.items():
        shuffled = list(image_ids)
        rng.shuffle(shuffled)
        selected[bucket] = shuffled[: requested[bucket]]
        LOGGER.info(
            "Selected %s: %d/%d available",
            bucket,
            len(selected[bucket]),
            len(image_ids),
        )
    return selected


def write_subset_annotations(
    coco: Dict[str, Any],
    selected: Dict[str, List[int]],
    annotations_by_image: Dict[int, List[Dict[str, Any]]],
    source_dir: Path,
) -> Tuple[Path, List[Dict[str, Any]]]:
    selected_ids = {
        image_id for image_ids in selected.values() for image_id in image_ids
    }
    selected_images = [
        image for image in coco.get("images", []) if int(image["id"]) in selected_ids
    ]
    selected_annotations: List[Dict[str, Any]] = []
    for image in selected_images:
        selected_annotations.extend(annotations_by_image[int(image["id"])])

    categories = [
        category
        for category in coco.get("categories", [])
        if int(category["id"]) in {COCO_PERSON_ID, COCO_CAR_ID}
    ]
    subset = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories,
    }

    subset_path = source_dir / "annotations" / "instances_train2017_person_car_small.json"
    save_json(subset_path, subset)

    manifest = {
        "selected_by_bucket": {key: len(value) for key, value in selected.items()},
        "selected_images": len(selected_images),
        "selected_annotations": len(selected_annotations),
        "subset_annotations": str(subset_path),
    }
    save_json(source_dir / "selected_manifest.json", manifest)
    LOGGER.info(
        "Subset annotations written: %s | images=%d | annotations=%d",
        subset_path,
        len(selected_images),
        len(selected_annotations),
    )
    return subset_path, selected_images


def image_url(image: Dict[str, Any], image_base_url: str) -> str:
    coco_url = image.get("coco_url")
    if isinstance(coco_url, str) and coco_url:
        return coco_url
    return f"{image_base_url.rstrip('/')}/{image['file_name']}"


def download_one_image(
    image: Dict[str, Any],
    image_dir: Path,
    image_base_url: str,
    timeout: int,
) -> Tuple[str, str]:
    destination = image_dir / image["file_name"]
    if destination.exists() and destination.stat().st_size > 0:
        return image["file_name"], "exists"

    url = image_url(image, image_base_url)
    try:
        download_file(url, destination, timeout=timeout)
        return image["file_name"], "downloaded"
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        return image["file_name"], f"failed: {exc}"


def download_selected_images(
    images: List[Dict[str, Any]],
    source_dir: Path,
    args: argparse.Namespace,
) -> None:
    image_dir = source_dir / "train2017"
    image_dir.mkdir(parents=True, exist_ok=True)

    failures: List[Tuple[str, str]] = []
    status_counts: Dict[str, int] = defaultdict(int)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one_image,
                image,
                image_dir,
                args.image_base_url,
                args.timeout,
            )
            for image in images
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            file_name, status = future.result()
            status_counts[status] += 1
            if status.startswith("failed"):
                failures.append((file_name, status))
            if index % 50 == 0 or index == len(futures):
                LOGGER.info("Image download progress: %d/%d", index, len(futures))

    LOGGER.info("Image download status: %s", dict(status_counts))
    if failures:
        preview = "\n".join(f"  - {name}: {status}" for name, status in failures[:20])
        raise RuntimeError(
            f"{len(failures)} image downloads failed. First failures:\n{preview}"
        )


def run_builder(
    subset_annotations: Path,
    source_dir: Path,
    args: argparse.Namespace,
) -> None:
    build_script = Path(__file__).resolve().with_name("build_cutout_dataset.py")
    command = [
        sys.executable,
        str(build_script),
        "--output-dir",
        args.output_dir,
        "--coco",
        "coco2017_small",
        str(subset_annotations),
        str(source_dir / "train2017"),
        "--image-size",
        str(args.image_size),
        "--resize-mode",
        args.resize_mode,
        "--max-total",
        str(args.final_total),
        "--max-both",
        str(args.final_both),
        "--max-person-only",
        str(args.final_person_only),
        "--max-car-only",
        str(args.final_car_only),
        "--max-negative",
        str(args.final_negative),
        "--min-selected",
        str(args.min_selected),
        "--min-both-ratio",
        str(args.min_both_ratio),
        "--min-image-side",
        str(args.min_image_side),
        "--min-instance-area-ratio",
        str(args.min_instance_area_ratio),
        "--max-instance-area-ratio",
        str(args.max_instance_area_ratio),
        "--min-bbox-side",
        str(args.min_bbox_side),
        "--max-foreground-ratio",
        str(args.max_foreground_ratio),
        "--blur-threshold",
        str(args.blur_threshold),
    ]
    if args.clean_output:
        command.append("--clean-output")

    LOGGER.info("Running builder: %s", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    source_dir = Path(args.source_dir)

    annotations_json = ensure_annotations(args)
    coco = load_json(annotations_json)
    buckets, annotations_by_image = build_image_buckets(coco)
    LOGGER.info(
        "Available COCO buckets | both=%d | person_only=%d | car_only=%d",
        len(buckets["both"]),
        len(buckets["person_only"]),
        len(buckets["car_only"]),
    )

    selected = select_image_ids(buckets, args)
    subset_annotations, selected_images = write_subset_annotations(
        coco,
        selected,
        annotations_by_image,
        source_dir,
    )
    download_selected_images(selected_images, source_dir, args)
    run_builder(subset_annotations, source_dir, args)


if __name__ == "__main__":
    main()
