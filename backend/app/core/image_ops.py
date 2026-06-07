"""
Image pre-processing and post-processing utilities for the cutout pipeline.

Handles image loading, resizing, mask refinement, and transparent PNG generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (matching Swin backbone training)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry metadata for reversing letterbox preprocessing."""

    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int
    original_size: Tuple[int, int]
    target_size: Tuple[int, int]


def load_image_rgb(source: str | bytes | Image.Image) -> np.ndarray:
    """Load an image from file path, bytes, or PIL Image as a numpy RGB array.

    Returns:
        RGB image as ``np.ndarray`` of shape ``(H, W, 3)``, dtype uint8.
    """
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
        return np.array(img, dtype=np.uint8)

    if isinstance(source, bytes):
        nparr = np.frombuffer(source, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot decode image from bytes.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # File path
    img_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preprocess_for_model(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, LetterboxMeta]:
    """Preprocess an RGB image for Mask2Former inference.

    Steps:
        1. Resize with aspect ratio preserved and pad to target_size.
        2. Normalise from [0, 255] to [0, 1].
        3. Apply ImageNet normalisation.

    Args:
        image_rgb: RGB image of shape ``(H, W, 3)``, dtype uint8.
        target_size: ``(height, width)`` to resize to.

    Returns:
        Tuple of ``(normalised_tensor, resized_image)``, where:
            - ``normalised_tensor``: shape ``(1, 3, H, W)``, float32, normalised.
            - ``resized_image``: letterboxed image, shape ``(H, W, 3)``, uint8.
            - ``meta``: geometry needed to map masks back to original size.
    """
    target_h, target_w = target_size
    orig_h, orig_w = image_rgb.shape[:2]
    scale = min(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2

    resized_content = cv2.resize(
        image_rgb,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    resized[
        pad_top:pad_top + resized_h,
        pad_left:pad_left + resized_w,
    ] = resized_content

    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = tensor.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    tensor = np.expand_dims(tensor, axis=0)  # (1, C, H, W)

    meta = LetterboxMeta(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_w,
        resized_height=resized_h,
        original_size=(orig_h, orig_w),
        target_size=(target_h, target_w),
    )
    return tensor, resized, meta


def refine_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.001,
    morph_kernel_size: int = 3,
    blur_kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological operations and edge smoothing to a binary mask.

    Args:
        mask: Binary mask of shape ``(H, W)``, values in {0, 1}.
        min_area_ratio: Remove connected regions smaller than this fraction of
            total pixels.
        morph_kernel_size: Kernel size for morphological close.
        blur_kernel_size: Kernel size for Gaussian blur (edge feathering).

    Returns:
        Refined binary mask of shape ``(H, W)``, dtype float32.
    """
    binary = (mask >= 0.5).astype(np.uint8)

    if binary.sum() == 0:
        return mask.astype(np.float32)

    # Morphological close to fill small holes
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Remove small connected components
    if min_area_ratio > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        total_pixels = binary.size
        min_area = int(total_pixels * min_area_ratio)

        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                binary[labels == label_id] = 0

    # Gaussian blur for edge feathering
    if blur_kernel_size > 0:
        binary_float = binary.astype(np.float32)
        binary_float = cv2.GaussianBlur(
            binary_float,
            (blur_kernel_size, blur_kernel_size),
            0,
        )
        binary_float = np.clip(binary_float, 0.0, 1.0)
        return binary_float

    return binary.astype(np.float32)


def resize_mask_to_original(
    mask: np.ndarray,
    original_size: Tuple[int, int],
    letterbox_meta: Optional[LetterboxMeta] = None,
) -> np.ndarray:
    """Resize a binary mask back to the original image dimensions.

    Args:
        mask: Mask of shape ``(H', W')``.
        original_size: ``(original_height, original_width)``.

    Returns:
        Mask of shape ``(original_height, original_width)``.
    """
    orig_h, orig_w = original_size
    if letterbox_meta is not None:
        y0 = letterbox_meta.pad_top
        x0 = letterbox_meta.pad_left
        y1 = y0 + letterbox_meta.resized_height
        x1 = x0 + letterbox_meta.resized_width
        mask = mask[y0:y1, x0:x1]

    resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return resized


def combine_masks(
    masks: List[np.ndarray],
    class_indices: List[int],
) -> np.ndarray:
    """Combine per-class masks into a single labeled mask.

    Args:
        masks: List of binary masks, each shape ``(H, W)``.
        class_indices: Class indices corresponding to each mask (0=person, 1=car).

    Returns:
        Combined mask of shape ``(H, W)``, where pixel value indicates class
        (0=person, 1=car). Background pixels are 0 but masked by the binary
        union; use with caution.
    """
    if not masks:
        return np.zeros(masks[0].shape if masks else (512, 512), dtype=np.uint8)

    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask, cls_idx in zip(masks, class_indices):
        binary = (mask >= 0.5).astype(np.uint8)
        combined = np.maximum(combined, binary * (cls_idx + 1))
    return combined


def generate_overlay(
    image_rgb: np.ndarray,
    combined_mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay color-coded masks onto the original image.

    Colors: person = green, car = blue.

    Args:
        image_rgb: Original RGB image of shape ``(H, W, 3)``.
        combined_mask: Labeled mask of shape ``(H, W)`` (1=person, 2=car).
        alpha: Blend weight for overlay (0=original, 1=mask color only).

    Returns:
        Overlay image of shape ``(H, W, 3)``, dtype uint8.
    """
    colors = {
        1: np.array([0, 255, 0], dtype=np.float32),   # person: green
        2: np.array([0, 0, 255], dtype=np.float32),   # car: blue
    }

    overlay = np.zeros_like(image_rgb, dtype=np.float32)
    for cls_id, color in colors.items():
        overlay[combined_mask == cls_id] = color

    composite = (image_rgb.astype(np.float32) * (1.0 - alpha) + overlay * alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def generate_cutout(
    image_rgb: np.ndarray,
    combined_mask: np.ndarray,
) -> np.ndarray:
    """Generate a transparent cutout (RGBA) using the combined mask as alpha.

    Args:
        image_rgb: Original RGB image of shape ``(H, W, 3)``.
        combined_mask: Binary or labeled mask of shape ``(H, W)``.

    Returns:
        RGBA image of shape ``(H, W, 4)``, dtype uint8.
    """
    alpha = (combined_mask > 0).astype(np.uint8) * 255
    rgba = np.dstack([image_rgb, alpha])
    return rgba


def save_image(
    image: np.ndarray,
    path: str,
) -> None:
    """Save a numpy image to file.

    Supports RGB, RGBA, and single-channel (grayscale) images.
    """
    if image.ndim == 2:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="L")
    elif image.shape[2] == 4:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="RGBA")
    else:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")

    pil_img.save(path)
    logger.debug("Image saved: %s", path)
