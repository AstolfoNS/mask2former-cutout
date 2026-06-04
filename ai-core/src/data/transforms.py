"""
Image and mask transforms for the Mask2Former cutout pipeline.

All transforms operate on a dict with keys ``image`` (H, W, 3 uint8) and
``mask`` (H, W float32) so that spatial augmentations stay exactly aligned.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class Compose:
    """Compose a list of transforms."""

    def __init__(self, transforms: List[object]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    """Resize image and mask to a fixed size."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size  # (width, height) for cv2

    def __call__(self, sample: dict) -> dict:
        sample["image"] = cv2.resize(
            sample["image"], self.size, interpolation=cv2.INTER_LINEAR
        )
        sample["mask"] = cv2.resize(
            sample["mask"], self.size, interpolation=cv2.INTER_NEAREST
        )
        return sample


class RandomHorizontalFlip:
    """Flip image and mask horizontally with given probability."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            sample["image"] = np.fliplr(sample["image"]).copy()
            sample["mask"] = np.fliplr(sample["mask"]).copy()
        return sample


class RandomRotate90:
    """Randomly rotate image and mask by 0, 90, 180, or 270 degrees."""

    def __init__(self, prob: float = 0.3) -> None:
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            k = random.randint(0, 3)
            sample["image"] = np.rot90(sample["image"], k).copy()
            sample["mask"] = np.rot90(sample["mask"], k).copy()
        return sample


class RandomBrightnessContrast:
    """Adjust brightness and contrast within a small range."""

    def __init__(
        self,
        brightness_limit: float = 0.1,
        contrast_limit: float = 0.1,
    ) -> None:
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, sample: dict) -> dict:
        alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        beta = random.uniform(-self.brightness_limit, self.brightness_limit) * 255
        image = sample["image"].astype(np.float32)
        image = alpha * image + beta
        image = np.clip(image, 0, 255).astype(np.uint8)
        sample["image"] = image
        return sample


class Normalize:
    """Normalize image with mean and std (ImageNet defaults)."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, sample: dict) -> dict:
        img = sample["image"].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        sample["image"] = img
        return sample


class ToTensor:
    """Convert numpy arrays to torch tensors with correct layout."""

    def __call__(self, sample: dict) -> dict:
        sample["image"] = torch.from_numpy(
            sample["image"].transpose(2, 0, 1).copy()
        ).float()  # C, H, W
        sample["mask"] = torch.from_numpy(sample["mask"].copy()).float()  # H, W
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return sample


def get_train_transforms(image_size: Tuple[int, int] = (512, 512)) -> Compose:
    """Return the training augmentation pipeline."""
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(prob=0.5),
        RandomRotate90(prob=0.3),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        Normalize(),
        ToTensor(),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (512, 512)) -> Compose:
    """Return the validation transform pipeline (no augmentations)."""
    return Compose([
        Resize(image_size),
        Normalize(),
        ToTensor(),
    ])
