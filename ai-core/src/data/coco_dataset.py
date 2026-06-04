"""
COCO-format dataset for hvm_coco_512 (person & car cutout).

Reads polygon annotations from a COCO JSON, converts each polygon to a binary
mask at load time, and returns (image, mask, label) triples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import Compose, get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class HvmCocoDataset(Dataset):
    """COCO-style dataset for human/vehicle matting.

    Each image has exactly one annotation (one polygon). The polygon is
    rasterized into a binary mask of shape (H, W). The category_id is the
    integer label: 1 = person, 2 = car.
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transforms: Optional[Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.image_size = image_size

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

        # Build lookup: image_id -> annotation
        self._ann_by_image: Dict[int, dict] = {}
        for ann in self.annotations:
            self._ann_by_image[ann["image_id"]] = ann

        # Build file_name -> image_id index
        self._img_by_filename: Dict[str, dict] = {}
        for img in self.images:
            self._img_by_filename[img["file_name"]] = img

        logger.info(
            "Loaded %d images with %d annotations across %d categories.",
            len(self.images),
            len(self.annotations),
            len(self.categories),
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_meta = self.images[idx]
        file_name = img_meta["file_name"]
        image_id = img_meta["id"]

        # Load image
        image_path = self.image_dir / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        ann = self._ann_by_image.get(image_id)
        if ann is None:
            # No annotation for this image — use empty mask
            mask = np.zeros((img_meta["height"], img_meta["width"]), dtype=np.float32)
            category_id = 0
        else:
            mask = self._polygon_to_mask(
                ann["segmentation"],
                height=img_meta["height"],
                width=img_meta["width"],
            )
            category_id = ann["category_id"]

        label = category_id  # 1 = person, 2 = car, 0 = background (rare)

        sample = {
            "image": image,           # H x W x 3, uint8
            "mask": mask,             # H x W, float32
            "label": label,           # int
            "file_name": file_name,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def _polygon_to_mask(
        segmentation: List[List[float]],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Convert COCO polygon(s) to a binary mask."""
        mask = np.zeros((height, width), dtype=np.float32)
        for polygon in segmentation:
            pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1.0)
        return mask


def build_dataloaders(
    annotation_file: str,
    image_dir: str,
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    val_subset_size: int = 200,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders with a deterministic split.

    The first ``val_subset_size`` samples are used as the validation set;
    the rest form the training set.  This mirrors the deterministic behaviour
    of standard PyTorch training pipelines.
    """

    full_dataset = HvmCocoDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        transforms=None,  # transforms are attached per split
        image_size=image_size,
    )

    n_total = len(full_dataset)
    indices = list(range(n_total))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_indices = indices[val_subset_size:]
    val_indices = indices[:val_subset_size]

    train_dataset = HvmCocoDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        transforms=get_train_transforms(image_size),
        image_size=image_size,
    )
    train_dataset.images = [full_dataset.images[i] for i in train_indices]
    train_dataset._ann_by_image = {
        full_dataset.images[i]["id"]: full_dataset._ann_by_image.get(
            full_dataset.images[i]["id"]
        )
        for i in train_indices
    }

    val_dataset = HvmCocoDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        transforms=get_val_transforms(image_size),
        image_size=image_size,
    )
    val_dataset.images = [full_dataset.images[i] for i in val_indices]
    val_dataset._ann_by_image = {
        full_dataset.images[i]["id"]: full_dataset._ann_by_image.get(
            full_dataset.images[i]["id"]
        )
        for i in val_indices
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
    )

    logger.info(
        "Train samples: %d | Val samples: %d | Batch size: %d",
        len(train_dataset),
        len(val_dataset),
        batch_size,
    )
    return train_loader, val_loader
