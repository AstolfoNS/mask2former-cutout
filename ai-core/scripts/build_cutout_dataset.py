#!/usr/bin/env python3
"""
Build a compact high-quality person/car cutout dataset.

The output matches the existing training layout:

    output_dir/
        images/
            sample_000001.jpg
            ...
        annotations.json
        quality_report.json

Supported inputs:
    - COCO-style instance segmentation JSON.
    - Cityscapes gtFine polygon annotations.
    - Generic semantic mask directories, converted to connected components.
    - Negative image directories with no person/car annotations.

The script keeps all geometry aligned after resize, remaps categories to the
project ids (1=person, 2=car), filters low-quality samples, and prioritizes
images where person and car appear together.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


LOGGER = logging.getLogger("build_cutout_dataset")

PROJECT_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "human"},
    {"id": 2, "name": "car", "supercategory": "vehicle"},
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class RawInstance:
    category_id: int
    polygons: List[List[float]]
    iscrowd: int = 0


@dataclass
class RawCandidate:
    source: str
    image_path: Path
    width: int
    height: int
    instances: List[RawInstance] = field(default_factory=list)
    negative: bool = False

    @property
    def class_ids(self) -> set[int]:
        return {instance.category_id for instance in self.instances}

    @property
    def bucket(self) -> str:
        if self.negative:
            return "negative"
        class_ids = self.class_ids
        if 1 in class_ids and 2 in class_ids:
            return "both"
        if 1 in class_ids:
            return "person_only"
        if 2 in class_ids:
            return "car_only"
        return "negative"


@dataclass
class BuiltAnnotation:
    category_id: int
    segmentation: List[List[float]]
    bbox: List[float]
    area: float


@dataclass
class BuiltSample:
    source: str
    bucket: str
    image: Image.Image
    annotations: List[BuiltAnnotation]
    quality_score: float
    blur_score: float
    foreground_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact COCO-format person/car cutout dataset."
    )
    parser.add_argument(
        "--config",
        help="Optional JSON recipe. CLI arguments override recipe values.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square output image size.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
        help="letterbox preserves aspect ratio; stretch matches the old dataset style.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-selected",
        type=int,
        default=1,
        help="Require at least this many selected images after quality filtering.",
    )
    parser.add_argument("--max-total", type=int, default=1000)
    parser.add_argument("--max-both", type=int, default=500)
    parser.add_argument("--max-person-only", type=int, default=200)
    parser.add_argument("--max-car-only", type=int, default=200)
    parser.add_argument("--max-negative", type=int, default=100)
    parser.add_argument(
        "--scan-multiplier",
        type=int,
        default=8,
        help="Process at most quota * this many raw candidates per bucket.",
    )

    parser.add_argument(
        "--coco",
        nargs=3,
        action="append",
        metavar=("NAME", "ANNOTATIONS_JSON", "IMAGE_DIR"),
        default=[],
        help="Add a COCO-style instance segmentation source.",
    )
    parser.add_argument(
        "--cityscapes",
        nargs=3,
        action="append",
        metavar=("NAME", "LEFT_IMG_ROOT", "GTFINE_ROOT"),
        default=[],
        help="Add a Cityscapes gtFine polygon source.",
    )
    parser.add_argument(
        "--semantic",
        nargs=4,
        action="append",
        metavar=("NAME", "IMAGE_DIR", "MASK_DIR", "LABEL_MAP_JSON"),
        default=[],
        help=(
            "Add semantic masks. LABEL_MAP_JSON example: "
            '{"person": [24], "car": [26]}.'
        ),
    )
    parser.add_argument(
        "--negative",
        nargs=2,
        action="append",
        metavar=("NAME", "IMAGE_DIR"),
        default=[],
        help="Add image-only negative/background samples.",
    )

    parser.add_argument(
        "--person-labels",
        default="person,rider",
        help="Comma-separated source label names mapped to project person.",
    )
    parser.add_argument(
        "--car-labels",
        default="car",
        help="Comma-separated source label names mapped to project car.",
    )
    parser.add_argument(
        "--min-image-side",
        type=int,
        default=256,
        help="Reject source images whose shortest side is below this value.",
    )
    parser.add_argument(
        "--min-instance-area-ratio",
        type=float,
        default=0.002,
        help="Reject instances smaller than this ratio after resize.",
    )
    parser.add_argument(
        "--max-instance-area-ratio",
        type=float,
        default=0.85,
        help="Reject instances larger than this ratio after resize.",
    )
    parser.add_argument(
        "--min-bbox-side",
        type=float,
        default=8.0,
        help="Reject instances whose resized bbox side is too small.",
    )
    parser.add_argument(
        "--max-foreground-ratio",
        type=float,
        default=0.80,
        help="Reject samples where all masks cover too much of the image.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=25.0,
        help="Reject samples with lower Laplacian variance; use 0 to disable.",
    )
    parser.add_argument(
        "--min-both-ratio",
        type=float,
        default=0.35,
        help=(
            "Require at least this fraction of selected images to contain both "
            "person and car. Use 0 for smoke tests."
        ),
    )
    parser.add_argument(
        "--keep-crowd",
        action="store_true",
        help="Keep COCO crowd annotations. Disabled by default.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove output_dir before writing.",
    )
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    config_path = Path(args.config)
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset recipe must be a JSON object: {config_path}")

    parser_defaults = parse_args_defaults()

    scalar_keys = [
        "output_dir",
        "image_size",
        "resize_mode",
        "seed",
        "min_selected",
        "max_total",
        "max_both",
        "max_person_only",
        "max_car_only",
        "max_negative",
        "scan_multiplier",
        "person_labels",
        "car_labels",
        "min_image_side",
        "min_instance_area_ratio",
        "max_instance_area_ratio",
        "min_bbox_side",
        "max_foreground_ratio",
        "blur_threshold",
        "min_both_ratio",
    ]
    for key in scalar_keys:
        if key not in config:
            continue
        current_value = getattr(args, key)
        default_value = getattr(parser_defaults, key)
        if current_value == default_value or current_value is None:
            setattr(args, key, config[key])

    for key in ("keep_crowd", "clean_output"):
        if key in config and getattr(args, key) is False:
            setattr(args, key, bool(config[key]))

    sources = config.get("sources", {})
    if sources and not isinstance(sources, dict):
        raise ValueError("Recipe field 'sources' must be an object.")

    if not args.coco:
        args.coco = [
            [item["name"], item["annotations_json"], item["image_dir"]]
            for item in sources.get("coco", [])
        ]
    if not args.cityscapes:
        args.cityscapes = [
            [item["name"], item["left_img_root"], item["gtfine_root"]]
            for item in sources.get("cityscapes", [])
        ]
    if not args.semantic:
        args.semantic = [
            [item["name"], item["image_dir"], item["mask_dir"], item["label_map_json"]]
            for item in sources.get("semantic", [])
        ]
    if not args.negative:
        args.negative = [
            [item["name"], item["image_dir"]]
            for item in sources.get("negative", [])
        ]

    return args


def parse_args_defaults() -> argparse.Namespace:
    config_arg = None
    namespace = argparse.Namespace()
    namespace.config = config_arg
    namespace.output_dir = None
    namespace.image_size = 512
    namespace.resize_mode = "letterbox"
    namespace.seed = 42
    namespace.min_selected = 1
    namespace.max_total = 1000
    namespace.max_both = 500
    namespace.max_person_only = 200
    namespace.max_car_only = 200
    namespace.max_negative = 100
    namespace.scan_multiplier = 8
    namespace.coco = []
    namespace.cityscapes = []
    namespace.semantic = []
    namespace.negative = []
    namespace.person_labels = "person,rider"
    namespace.car_labels = "car"
    namespace.min_image_side = 256
    namespace.min_instance_area_ratio = 0.002
    namespace.max_instance_area_ratio = 0.85
    namespace.min_bbox_side = 8.0
    namespace.max_foreground_ratio = 0.80
    namespace.blur_threshold = 25.0
    namespace.min_both_ratio = 0.35
    namespace.keep_crowd = False
    namespace.clean_output = False
    return namespace


def split_labels(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def polygon_area(points: Sequence[float]) -> float:
    if len(points) < 6:
        return 0.0
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return float(abs(cv2.contourArea(arr)))


def normalize_coco_polygons(segmentation: Any) -> List[List[float]]:
    if not isinstance(segmentation, list):
        return []
    polygons: List[List[float]] = []
    for polygon in segmentation:
        if not isinstance(polygon, list):
            continue
        if len(polygon) >= 6 and polygon_area(polygon) > 0:
            polygons.append([float(value) for value in polygon])
    return polygons


def transform_polygon(
    polygon: List[float],
    original_size: Tuple[int, int],
    output_size: int,
    resize_mode: str,
) -> List[float]:
    width, height = original_size
    arr = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

    if resize_mode == "stretch":
        arr[:, 0] *= output_size / max(width, 1)
        arr[:, 1] *= output_size / max(height, 1)
    else:
        scale = min(output_size / max(width, 1), output_size / max(height, 1))
        new_width = width * scale
        new_height = height * scale
        pad_x = (output_size - new_width) / 2.0
        pad_y = (output_size - new_height) / 2.0
        arr[:, 0] = arr[:, 0] * scale + pad_x
        arr[:, 1] = arr[:, 1] * scale + pad_y

    arr[:, 0] = np.clip(arr[:, 0], 0, output_size - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, output_size - 1)
    return arr.reshape(-1).astype(float).tolist()


def resize_image(image: Image.Image, output_size: int, resize_mode: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if resize_mode == "stretch":
        return image.resize((output_size, output_size), Image.Resampling.BILINEAR)

    width, height = image.size
    scale = min(output_size / max(width, 1), output_size / max(height, 1))
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (output_size, output_size), (114, 114, 114))
    offset = ((output_size - new_size[0]) // 2, (output_size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def mask_from_polygons(polygons: List[List[float]], output_size: int) -> np.ndarray:
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask


def annotation_from_polygons(
    category_id: int,
    polygons: List[List[float]],
    output_size: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_bbox_side: float,
) -> Optional[BuiltAnnotation]:
    mask = mask_from_polygons(polygons, output_size)
    area = float(mask.sum())
    total_pixels = float(output_size * output_size)
    area_ratio = area / total_pixels
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min = float(xs.min())
    y_min = float(ys.min())
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    if width < min_bbox_side or height < min_bbox_side:
        return None

    valid_polygons = [
        [round(float(value), 2) for value in polygon]
        for polygon in polygons
        if len(polygon) >= 6 and polygon_area(polygon) > 0
    ]
    if not valid_polygons:
        return None

    return BuiltAnnotation(
        category_id=category_id,
        segmentation=valid_polygons,
        bbox=[round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)],
        area=round(area, 2),
    )


def blur_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_coco_source(
    name: str,
    annotation_path: Path,
    image_dir: Path,
    person_labels: set[str],
    car_labels: set[str],
    keep_crowd: bool,
) -> List[RawCandidate]:
    data = load_json(annotation_path)
    images = {image["id"]: image for image in data.get("images", [])}
    categories = {
        category["id"]: category.get("name", "").lower()
        for category in data.get("categories", [])
    }

    category_map: Dict[int, int] = {}
    for category_id, label in categories.items():
        if label in person_labels:
            category_map[category_id] = 1
        elif label in car_labels:
            category_map[category_id] = 2

    instances_by_image: Dict[int, List[RawInstance]] = {}
    for annotation in data.get("annotations", []):
        source_category = annotation.get("category_id")
        if source_category not in category_map:
            continue
        if annotation.get("iscrowd", 0) and not keep_crowd:
            continue
        polygons = normalize_coco_polygons(annotation.get("segmentation"))
        if not polygons:
            continue
        instances_by_image.setdefault(annotation["image_id"], []).append(
            RawInstance(
                category_id=category_map[source_category],
                polygons=polygons,
                iscrowd=int(annotation.get("iscrowd", 0)),
            )
        )

    candidates: List[RawCandidate] = []
    for image_id, instances in instances_by_image.items():
        image_meta = images.get(image_id)
        if not image_meta:
            continue
        image_path = image_dir / image_meta["file_name"]
        candidates.append(
            RawCandidate(
                source=name,
                image_path=image_path,
                width=int(image_meta.get("width", 0)),
                height=int(image_meta.get("height", 0)),
                instances=instances,
            )
        )

    LOGGER.info("Loaded COCO source %s: %d candidates", name, len(candidates))
    return candidates


def build_cityscapes_image_index(left_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in iter_image_files(left_root):
        key = path.stem.replace("_leftImg8bit", "")
        index[key] = path
    return index


def load_cityscapes_source(
    name: str,
    left_root: Path,
    gt_root: Path,
    person_labels: set[str],
    car_labels: set[str],
) -> List[RawCandidate]:
    image_index = build_cityscapes_image_index(left_root)
    candidates: List[RawCandidate] = []

    for annotation_path in gt_root.rglob("*_gtFine_polygons.json"):
        key = annotation_path.stem.replace("_gtFine_polygons", "")
        image_path = image_index.get(key)
        if not image_path:
            continue

        data = load_json(annotation_path)
        width = int(data.get("imgWidth", 0))
        height = int(data.get("imgHeight", 0))
        instances: List[RawInstance] = []

        for obj in data.get("objects", []):
            label = str(obj.get("label", "")).lower()
            if label.endswith("group"):
                continue
            if label in person_labels:
                category_id = 1
            elif label in car_labels:
                category_id = 2
            else:
                continue
            polygon_points = obj.get("polygon", [])
            if len(polygon_points) < 3:
                continue
            polygon: List[float] = []
            for x_value, y_value in polygon_points:
                polygon.extend([float(x_value), float(y_value)])
            if polygon_area(polygon) <= 0:
                continue
            instances.append(RawInstance(category_id=category_id, polygons=[polygon]))

        if instances:
            candidates.append(
                RawCandidate(
                    source=name,
                    image_path=image_path,
                    width=width,
                    height=height,
                    instances=instances,
                )
            )

    LOGGER.info("Loaded Cityscapes source %s: %d candidates", name, len(candidates))
    return candidates


def label_ids_from_map(label_map_path: Path) -> Dict[int, List[int]]:
    label_map = load_json(label_map_path)
    result: Dict[int, List[int]] = {1: [], 2: []}
    for key, project_id in (("person", 1), ("car", 2)):
        values = label_map.get(key, [])
        if isinstance(values, int):
            values = [values]
        result[project_id] = [int(value) for value in values]
    return result


def find_matching_mask(mask_dir: Path, image_path: Path) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def contours_to_polygons(mask: np.ndarray, min_points: int = 3) -> List[List[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if contour.shape[0] < min_points:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] < min_points:
            continue
        polygon = approx.reshape(-1, 2).astype(float).reshape(-1).tolist()
        if polygon_area(polygon) > 0:
            polygons.append(polygon)
    return polygons


def load_semantic_source(
    name: str,
    image_dir: Path,
    mask_dir: Path,
    label_map_path: Path,
) -> List[RawCandidate]:
    label_ids = label_ids_from_map(label_map_path)
    candidates: List[RawCandidate] = []

    for image_path in iter_image_files(image_dir):
        mask_path = find_matching_mask(mask_dir, image_path)
        if not mask_path:
            continue
        image = Image.open(image_path)
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        instances: List[RawInstance] = []
        for category_id, ids in label_ids.items():
            if not ids:
                continue
            binary = np.isin(mask, ids).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            for component_id in range(1, num_labels):
                component = (labels == component_id).astype(np.uint8)
                polygons = contours_to_polygons(component)
                if polygons:
                    instances.append(
                        RawInstance(category_id=category_id, polygons=polygons)
                    )

        if instances:
            candidates.append(
                RawCandidate(
                    source=name,
                    image_path=image_path,
                    width=image.width,
                    height=image.height,
                    instances=instances,
                )
            )

    LOGGER.info("Loaded semantic source %s: %d candidates", name, len(candidates))
    return candidates


def load_negative_source(name: str, image_dir: Path) -> List[RawCandidate]:
    candidates: List[RawCandidate] = []
    for image_path in iter_image_files(image_dir):
        with Image.open(image_path) as image:
            width, height = image.size
        candidates.append(
            RawCandidate(
                source=name,
                image_path=image_path,
                width=width,
                height=height,
                instances=[],
                negative=True,
            )
        )
    LOGGER.info("Loaded negative source %s: %d candidates", name, len(candidates))
    return candidates


def build_sample(
    raw: RawCandidate,
    args: argparse.Namespace,
    rejected: Dict[str, int],
) -> Optional[BuiltSample]:
    if not raw.image_path.exists():
        rejected["missing_image"] = rejected.get("missing_image", 0) + 1
        return None
    if min(raw.width, raw.height) < args.min_image_side:
        rejected["small_source_image"] = rejected.get("small_source_image", 0) + 1
        return None

    try:
        source_image = Image.open(raw.image_path)
        output_image = resize_image(source_image, args.image_size, args.resize_mode)
    except Exception:
        rejected["decode_image"] = rejected.get("decode_image", 0) + 1
        return None

    score_blur = blur_score(output_image)
    if args.blur_threshold > 0 and score_blur < args.blur_threshold:
        rejected["blurry"] = rejected.get("blurry", 0) + 1
        return None

    if raw.negative:
        return BuiltSample(
            source=raw.source,
            bucket="negative",
            image=output_image,
            annotations=[],
            quality_score=score_blur,
            blur_score=score_blur,
            foreground_ratio=0.0,
        )

    annotations: List[BuiltAnnotation] = []
    masks: List[np.ndarray] = []

    for instance in raw.instances:
        polygons = [
            transform_polygon(
                polygon,
                original_size=(raw.width, raw.height),
                output_size=args.image_size,
                resize_mode=args.resize_mode,
            )
            for polygon in instance.polygons
        ]
        annotation = annotation_from_polygons(
            category_id=instance.category_id,
            polygons=polygons,
            output_size=args.image_size,
            min_area_ratio=args.min_instance_area_ratio,
            max_area_ratio=args.max_instance_area_ratio,
            min_bbox_side=args.min_bbox_side,
        )
        if annotation:
            annotations.append(annotation)
            masks.append(mask_from_polygons(annotation.segmentation, args.image_size))

    if not annotations:
        rejected["no_valid_instances"] = rejected.get("no_valid_instances", 0) + 1
        return None

    union_mask = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
    for mask in masks:
        union_mask = np.maximum(union_mask, mask)
    foreground_ratio = float(union_mask.mean())
    if foreground_ratio > args.max_foreground_ratio:
        rejected["foreground_too_large"] = rejected.get("foreground_too_large", 0) + 1
        return None

    class_ids = {annotation.category_id for annotation in annotations}
    bucket = "both" if class_ids == {1, 2} else "person_only" if 1 in class_ids else "car_only"

    instance_bonus = min(len(annotations), 5) * 10.0
    both_bonus = 100.0 if bucket == "both" else 0.0
    area_bonus = min(foreground_ratio, 0.35) * 100.0
    quality_score = score_blur + instance_bonus + both_bonus + area_bonus

    return BuiltSample(
        source=raw.source,
        bucket=bucket,
        image=output_image,
        annotations=annotations,
        quality_score=quality_score,
        blur_score=score_blur,
        foreground_ratio=foreground_ratio,
    )


def collect_sources(args: argparse.Namespace) -> List[RawCandidate]:
    person_labels = split_labels(args.person_labels)
    car_labels = split_labels(args.car_labels)
    candidates: List[RawCandidate] = []

    for name, annotations_json, image_dir in args.coco:
        candidates.extend(
            load_coco_source(
                name=name,
                annotation_path=Path(annotations_json),
                image_dir=Path(image_dir),
                person_labels=person_labels,
                car_labels=car_labels,
                keep_crowd=args.keep_crowd,
            )
        )

    for name, left_root, gt_root in args.cityscapes:
        candidates.extend(
            load_cityscapes_source(
                name=name,
                left_root=Path(left_root),
                gt_root=Path(gt_root),
                person_labels=person_labels,
                car_labels=car_labels,
            )
        )

    for name, image_dir, mask_dir, label_map_json in args.semantic:
        candidates.extend(
            load_semantic_source(
                name=name,
                image_dir=Path(image_dir),
                mask_dir=Path(mask_dir),
                label_map_path=Path(label_map_json),
            )
        )

    for name, image_dir in args.negative:
        candidates.extend(load_negative_source(name=name, image_dir=Path(image_dir)))

    return candidates


def validate_source_paths(args: argparse.Namespace) -> None:
    missing: List[str] = []

    for name, annotations_json, image_dir in args.coco:
        if not Path(annotations_json).is_file():
            missing.append(f"COCO {name}: annotations not found: {annotations_json}")
        if not Path(image_dir).is_dir():
            missing.append(f"COCO {name}: image dir not found: {image_dir}")

    for name, left_root, gt_root in args.cityscapes:
        if not Path(left_root).is_dir():
            missing.append(f"Cityscapes {name}: left image root not found: {left_root}")
        if not Path(gt_root).is_dir():
            missing.append(f"Cityscapes {name}: gtFine root not found: {gt_root}")

    for name, image_dir, mask_dir, label_map_json in args.semantic:
        if not Path(image_dir).is_dir():
            missing.append(f"Semantic {name}: image dir not found: {image_dir}")
        if not Path(mask_dir).is_dir():
            missing.append(f"Semantic {name}: mask dir not found: {mask_dir}")
        if not Path(label_map_json).is_file():
            missing.append(f"Semantic {name}: label map not found: {label_map_json}")

    for name, image_dir in args.negative:
        if not Path(image_dir).is_dir():
            missing.append(f"Negative {name}: image dir not found: {image_dir}")

    if missing:
        details = "\n  - ".join(missing)
        raise SystemExit(
            "Dataset source validation failed. Add the missing source files or "
            f"edit the recipe.\n  - {details}"
        )


def group_candidates(candidates: List[RawCandidate]) -> Dict[str, List[RawCandidate]]:
    groups = {
        "both": [],
        "person_only": [],
        "car_only": [],
        "negative": [],
    }
    for candidate in candidates:
        groups[candidate.bucket].append(candidate)
    return groups


def select_samples(
    groups: Dict[str, List[RawCandidate]],
    args: argparse.Namespace,
) -> Tuple[List[BuiltSample], Dict[str, Any]]:
    rng = random.Random(args.seed)
    quotas = {
        "both": args.max_both,
        "person_only": args.max_person_only,
        "car_only": args.max_car_only,
        "negative": args.max_negative,
    }

    selected: List[BuiltSample] = []
    rejected: Dict[str, int] = {}
    processed_by_bucket: Dict[str, int] = {}

    for bucket in ("both", "person_only", "car_only", "negative"):
        quota = max(0, quotas[bucket])
        if quota == 0 or len(selected) >= args.max_total:
            continue

        raw_candidates = list(groups[bucket])
        rng.shuffle(raw_candidates)
        scan_limit = min(len(raw_candidates), max(quota, 1) * args.scan_multiplier)

        bucket_samples: List[BuiltSample] = []
        for raw in raw_candidates[:scan_limit]:
            sample = build_sample(raw, args, rejected)
            processed_by_bucket[bucket] = processed_by_bucket.get(bucket, 0) + 1
            if sample and sample.bucket == bucket:
                bucket_samples.append(sample)
            elif sample:
                rejected[f"bucket_changed_{bucket}_to_{sample.bucket}"] = (
                    rejected.get(f"bucket_changed_{bucket}_to_{sample.bucket}", 0) + 1
                )

        bucket_samples.sort(key=lambda item: item.quality_score, reverse=True)
        remaining_total = args.max_total - len(selected)
        selected.extend(bucket_samples[: min(quota, remaining_total)])

    return selected, {
        "rejected": rejected,
        "processed_by_bucket": processed_by_bucket,
        "available_by_bucket": {key: len(value) for key, value in groups.items()},
    }


def write_dataset(
    selected: List[BuiltSample],
    output_dir: Path,
    args: argparse.Namespace,
    report: Dict[str, Any],
) -> None:
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    bucket_counts: Dict[str, int] = {}
    annotation_id = 1

    for image_id, sample in enumerate(selected, start=1):
        safe_source = "".join(
            char.lower() if char.isalnum() else "_" for char in sample.source
        ).strip("_")
        file_name = f"{safe_source}_{image_id:06d}.jpg"
        sample.image.save(image_dir / file_name, quality=95, subsampling=1)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": args.image_size,
                "height": args.image_size,
            }
        )
        source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
        bucket_counts[sample.bucket] = bucket_counts.get(sample.bucket, 0) + 1

        for annotation in sample.annotations:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation.category_id,
                    "segmentation": annotation.segmentation,
                    "bbox": annotation.bbox,
                    "area": annotation.area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": PROJECT_CATEGORIES,
    }
    with (output_dir / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)

    report.update(
        {
            "selected_images": len(images),
            "selected_annotations": len(annotations),
            "selected_by_source": source_counts,
            "selected_by_bucket": bucket_counts,
            "output_dir": str(output_dir),
            "categories": PROJECT_CATEGORIES,
            "settings": {
                "image_size": args.image_size,
                "resize_mode": args.resize_mode,
                "max_total": args.max_total,
                "max_both": args.max_both,
                "max_person_only": args.max_person_only,
                "max_car_only": args.max_car_only,
                "max_negative": args.max_negative,
                "min_instance_area_ratio": args.min_instance_area_ratio,
                "max_instance_area_ratio": args.max_instance_area_ratio,
                "blur_threshold": args.blur_threshold,
            },
        }
    )
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def validate_quality(selected: List[BuiltSample], args: argparse.Namespace) -> None:
    if not selected:
        raise ValueError("No selected samples to validate.")
    if len(selected) < args.min_selected:
        raise ValueError(
            "Dataset quality gate failed: selected image count "
            f"{len(selected)} is below required {args.min_selected}. "
            "Add more valid sources or relax filters only after reviewing "
            "quality_report.json."
        )

    both_count = sum(1 for sample in selected if sample.bucket == "both")
    both_ratio = both_count / len(selected)
    if both_ratio < args.min_both_ratio:
        raise ValueError(
            "Dataset quality gate failed: both-person-car ratio "
            f"{both_ratio:.3f} is below required {args.min_both_ratio:.3f}. "
            "Add real scene data such as COCO/Cityscapes samples that contain "
            "person and car in the same image, or lower --min-both-ratio only "
            "for smoke tests."
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = apply_config(parse_args())

    if not args.output_dir:
        raise SystemExit("--output-dir is required unless provided by --config.")

    if not any([args.coco, args.cityscapes, args.semantic, args.negative]):
        raise SystemExit("At least one data source is required.")
    validate_source_paths(args)

    candidates = collect_sources(args)
    if not candidates:
        raise SystemExit("No usable candidates found from the provided sources.")

    groups = group_candidates(candidates)
    LOGGER.info(
        "Raw candidates | both=%d | person_only=%d | car_only=%d | negative=%d",
        len(groups["both"]),
        len(groups["person_only"]),
        len(groups["car_only"]),
        len(groups["negative"]),
    )

    selected, report = select_samples(groups, args)
    if not selected:
        raise SystemExit("All candidates were rejected by quality filters.")
    validate_quality(selected, args)

    write_dataset(selected, Path(args.output_dir), args, report)
    LOGGER.info(
        "Dataset written: %s | images=%d | annotations=%d",
        args.output_dir,
        report["selected_images"],
        report["selected_annotations"],
    )


if __name__ == "__main__":
    main()
