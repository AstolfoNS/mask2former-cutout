"""
Main training script for Mask2Former person + car cutout on RTX 4090.

This script fine-tunes ``facebook/mask2former-swin-small-coco-instance`` on
the hvm_coco_512 dataset (2 classes: person, car) using the Hugging Face
Trainer API with full BF16 mixed-precision, TF32 acceleration, early stopping,
and mIoU-based model selection.

Usage::

    python src/train.py [--output_dir ./weights] [--epochs 40] [--batch_size 8]

Environment variables:
    WANDB_API_KEY   Weights & Biases API key (optional).
    HF_HUB_CACHE    Hugging Face cache directory.

Author: Mask2Former Cutout AI Core Team
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ensure_positive_int_env(name: str, default: str = "4") -> None:
    value = os.environ.get(name, "")
    if not value.isdigit() or int(value) <= 0:
        os.environ[name] = default


_ensure_positive_int_env("OMP_NUM_THREADS")
_ensure_positive_int_env("MKL_NUM_THREADS")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# HF / Transformers
# ---------------------------------------------------------------------------
from transformers import (
    EarlyStoppingCallback,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ============================================================================
# Performance tuning — Ampere (RTX 4090) specific
# ============================================================================

# Enable TF32 for matmul and cuDNN convolutions.  TF32 runs at full tensor-core
# throughput on Ampere while retaining more precision than FP16.  Combined with
# BF16 mixed-precision, this yields the best throughput/stability trade-off.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set float32 matmul to "high" precision so that TF32 is actually used.
torch.set_float32_matmul_precision("high")

# Pin the number of threads to avoid thread contention between DataLoader
# workers and the GPU compute stream.
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mask2former-cutout")


# ============================================================================
# Constants
# ============================================================================

# Hugging Face internal class indices (0-based) -> COCO category_id mapping.
#   HF class 0 = person  (COCO category_id 1)
#   HF class 1 = car     (COCO category_id 2)
ID2LABEL: Dict[int, str] = {0: "person", 1: "car"}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}

# COCO category_id -> HF internal class index.
COCO_CAT_TO_HF_CLASS: Dict[int, int] = {1: 0, 2: 1}

# ImageNet normalisation constants (used by the Swin backbone).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Data loading
# ============================================================================

def polygon_to_binary_mask(
    segmentation: List[List[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert COCO polygon(s) to a single binary mask.

    Args:
        segmentation: List of polygons; each polygon is a flat list of [x1, y1, x2, y2, ...].
        height: Image height.
        width: Image width.

    Returns:
        Binary mask of shape ``(height, width)`` with values 0.0 / 1.0.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for polygon in segmentation:
        pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
    return mask


class CocoCutoutDataset(Dataset):
    """COCO-format dataset for person + car instance segmentation.

    Each item returns a dict ready for Mask2Former:
        - ``pixel_values``: (3, H, W) normalised tensor.
        - ``mask_labels``: list of (num_instances, H, W) binary masks.
        - ``class_labels``: list of (num_instances,) class index tensors.
        - ``original_size``: (H, W) original image size for post-processing.
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        processor: Mask2FormerImageProcessor,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.split = split
        self.image_size = image_size

        # Load COCO JSON
        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self._images: List[dict] = coco["images"]
        self._categories: Dict[int, str] = {
            c["id"]: c["name"] for c in coco.get("categories", [])
        }

        # Build annotation lookup: image_id -> annotations. The current dataset
        # has one instance per image, but keeping the native COCO list structure
        # prevents silent overwrites if multi-instance samples are added later.
        self._ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self._ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Sanity checks
        logger.info(
            "Dataset [%s]: %d images, %d annotations, %d categories.",
            split,
            len(self._images),
            len(self._ann_by_image),
            len(self._categories),
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_meta = self._images[idx]
        image_id = img_meta["id"]
        file_name = img_meta["file_name"]

        # ------------------------------------------------------------------
        # 1. Load image (BGR -> RGB)
        # ------------------------------------------------------------------
        image_path = self.image_dir / file_name
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # 2. Load instance masks from COCO polygon annotations
        # ------------------------------------------------------------------
        anns = self._ann_by_image.get(image_id, [])
        instance_masks: List[torch.Tensor] = []
        instance_classes: List[int] = []

        for ann in anns:
            coco_cat = ann["category_id"]
            hf_class_id = COCO_CAT_TO_HF_CLASS.get(coco_cat, -1)
            if hf_class_id < 0:
                logger.warning(
                    "Unknown category_id=%d for image %s — skipping annotation.",
                    coco_cat,
                    file_name,
                )
                continue

            mask = polygon_to_binary_mask(
                ann["segmentation"],
                height=img_meta["height"],
                width=img_meta["width"],
            )
            instance_masks.append(torch.from_numpy(mask).float())
            instance_classes.append(hf_class_id)

        # ------------------------------------------------------------------
        # 3. Preprocess image with Mask2FormerImageProcessor
        #
        # We call the processor on the raw RGB image, which handles
        # normalisation and optional resizing.  ``do_resize=False`` is set
        # because our dataset is already 512x512 — this avoids unnecessary
        # interpolation and keeps masks pixel-aligned with the image.
        # ------------------------------------------------------------------
        pil_image = Image.fromarray(image_rgb)
        encoding = self.processor(
            images=pil_image,
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].squeeze(0)  # (3, H, W)

        # ------------------------------------------------------------------
        # 4. Build instance-level labels for Mask2Former
        #
        # Mask2Former expects:
        #   - mask_labels:  list of (num_instances, H, W) float32 tensors
        #   - class_labels: list of (num_instances,) int64 tensors
        #
        # Each image may have zero or more instances.
        # ------------------------------------------------------------------
        if instance_masks:
            mask_labels = [torch.stack(instance_masks, dim=0)]
            class_labels = [torch.tensor(instance_classes, dtype=torch.long)]
        else:
            height, width = img_meta["height"], img_meta["width"]
            mask_labels = [torch.zeros((0, height, width), dtype=torch.float32)]
            class_labels = [torch.zeros((0,), dtype=torch.long)]

        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "original_size": (img_meta["height"], img_meta["width"]),
            "image_id": image_id,
        }


# ============================================================================
# Batch collation
# ============================================================================

@dataclass
class DataCollatorForMask2Former:
    """Collate raw dataset items into a batch consumable by Mask2Former.

    Key behaviours:
        - Stacks ``pixel_values`` into a (B, 3, H, W) tensor.
        - Collects ``mask_labels`` and ``class_labels`` as lists (no stacking
          because images may have different instance counts).

    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

        # Flatten out the list-of-lists: each item already wraps its mask in
        # a single-element list, so we unwrap to preserve the shape.
        mask_labels: List[torch.Tensor] = []
        class_labels: List[torch.Tensor] = []

        for item in batch:
            mask_labels.append(item["mask_labels"][0])   # (1, H, W) or (0, H, W)
            class_labels.append(item["class_labels"][0])  # (1,) or (0,)

        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        }


# ============================================================================
# Metrics (mIoU)
# ============================================================================

def _semantic_predictions_from_queries(
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    target_size: Tuple[int, int],
    score_threshold: float = 0.5,
) -> torch.Tensor:
    """Convert Mask2Former query logits to compact semantic predictions.

    Background pixels are encoded as -1 because class index 0 is ``person``.
    This keeps mIoU computation correct and avoids caching raw query logits for
    the whole evaluation set.
    """
    class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]  # (B, Q, C)
    mask_probs = masks_queries_logits.sigmoid()                   # (B, Q, h, w)
    class_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

    class_scores = F.interpolate(
        class_scores,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    scores, predictions = class_scores.max(dim=1)  # (B, H, W)
    background = torch.full_like(predictions, fill_value=-1)
    return torch.where(scores >= score_threshold, predictions, background)


def _semantic_targets_from_instances(
    mask_labels: List[torch.Tensor],
    class_labels: List[torch.Tensor],
) -> torch.Tensor:
    """Convert instance masks/classes to semantic targets with -1 background."""
    targets: List[torch.Tensor] = []

    for masks, classes in zip(mask_labels, class_labels):
        height, width = masks.shape[-2:]
        target = torch.full(
            (height, width), -1, dtype=torch.long, device=masks.device
        )

        for mask, class_id in zip(masks, classes):
            target[mask > 0.5] = class_id.long()

        targets.append(target)

    return torch.stack(targets, dim=0)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute segmentation metrics from compact semantic masks.

    ``prediction_step`` returns predictions and labels as ``(N, H, W)`` arrays.
    Valid classes are 0=person and 1=car; background is encoded as -1.
    """
    pred_semantic = np.asarray(eval_pred.predictions)
    gt_semantic = np.asarray(eval_pred.label_ids)

    ious_per_class: Dict[int, List[float]] = {0: [], 1: []}
    dice_per_class: Dict[int, List[float]] = {0: [], 1: []}
    precision_per_class: Dict[int, float] = {}
    recall_per_class: Dict[int, float] = {}
    valid_images = int(np.any(gt_semantic >= 0, axis=(1, 2)).sum())

    for cls_id in range(2):
        pred_pixels = pred_semantic == cls_id
        gt_pixels = gt_semantic == cls_id

        intersection = np.logical_and(pred_pixels, gt_pixels).sum(axis=(1, 2))
        union = np.logical_or(pred_pixels, gt_pixels).sum(axis=(1, 2))
        pred_count = pred_pixels.sum(axis=(1, 2))
        gt_count = gt_pixels.sum(axis=(1, 2))

        valid_iou = union > 0
        if np.any(valid_iou):
            class_ious = intersection[valid_iou].astype(np.float64) / union[
                valid_iou
            ].astype(np.float64)
            ious_per_class[cls_id].extend(class_ious.tolist())

        dice_denominator = pred_count + gt_count
        valid_dice = dice_denominator > 0
        if np.any(valid_dice):
            class_dice = (2.0 * intersection[valid_dice].astype(np.float64)) / (
                dice_denominator[valid_dice].astype(np.float64)
            )
            dice_per_class[cls_id].extend(class_dice.tolist())

        total_intersection = float(intersection.sum())
        precision_per_class[cls_id] = _safe_divide(
            total_intersection, float(pred_count.sum())
        )
        recall_per_class[cls_id] = _safe_divide(
            total_intersection, float(gt_count.sum())
        )

    mean_iou_per_class = {
        cls_id: float(np.mean(ious)) if ious else 0.0
        for cls_id, ious in ious_per_class.items()
    }
    mean_dice_per_class = {
        cls_id: float(np.mean(dice)) if dice else 0.0
        for cls_id, dice in dice_per_class.items()
    }

    all_ious = [iou for ious in ious_per_class.values() for iou in ious]
    all_dice = [dice for values in dice_per_class.values() for dice in values]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0

    foreground_mask = gt_semantic >= 0
    background_mask = gt_semantic == -1
    pixel_accuracy = float(np.mean(pred_semantic == gt_semantic))
    foreground_accuracy = float(
        np.mean(pred_semantic[foreground_mask] == gt_semantic[foreground_mask])
    ) if np.any(foreground_mask) else 0.0
    background_accuracy = float(
        np.mean(pred_semantic[background_mask] == gt_semantic[background_mask])
    ) if np.any(background_mask) else 0.0
    foreground_ratio = float(np.mean(foreground_mask))

    metrics = {
        "eval_mean_iou": mean_iou,
        "eval_iou_person": mean_iou_per_class[0],
        "eval_iou_car": mean_iou_per_class[1],
        "eval_mean_dice": mean_dice,
        "eval_dice_person": mean_dice_per_class[0],
        "eval_dice_car": mean_dice_per_class[1],
        "eval_pixel_accuracy": pixel_accuracy,
        "eval_foreground_accuracy": foreground_accuracy,
        "eval_background_accuracy": background_accuracy,
        "eval_precision_person": precision_per_class[0],
        "eval_precision_car": precision_per_class[1],
        "eval_recall_person": recall_per_class[0],
        "eval_recall_car": recall_per_class[1],
        "eval_foreground_ratio": foreground_ratio,
    }

    logger.info(
        "Eval summary | mIoU=%.4f | Dice=%.4f | PixelAcc=%.4f | "
        "FgAcc=%.4f | BgAcc=%.4f | images=%d",
        mean_iou,
        mean_dice,
        pixel_accuracy,
        foreground_accuracy,
        background_accuracy,
        valid_images,
    )
    return metrics


class MetricsLoggingCallback(TrainerCallback):
    """Persist train/eval metrics to a CSV file for offline plotting.

    The file is opened in append mode for every write so interrupted runs keep
    their history.  A header is created only when the file does not exist or is
    empty.
    """

    FIELDNAMES = [
        "event",
        "step",
        "epoch",
        "loss",
        "learning_rate",
        "eval_loss",
        "mIoU",
        "Dice",
        "PixelAcc",
        "iou_person",
        "iou_car",
        "dice_person",
        "dice_car",
    ]

    def __init__(self, metrics_path: str | Path) -> None:
        self.metrics_path = Path(metrics_path)

    @staticmethod
    def _value(metrics: Dict[str, Any], key: str) -> Any:
        value = metrics.get(key)
        if isinstance(value, (float, np.floating)):
            return float(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    @staticmethod
    def _is_local_process_zero(state: Any) -> bool:
        return bool(getattr(state, "is_local_process_zero", True))

    def _append_row(self, row: Dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = (
            not self.metrics_path.exists()
            or self.metrics_path.stat().st_size == 0
        )

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        if not self._is_local_process_zero(state):
            return

        row = {
            "event": "train",
            "step": state.global_step,
            "epoch": self._value(logs, "epoch"),
            "loss": self._value(logs, "loss"),
            "learning_rate": self._value(logs, "learning_rate"),
            "eval_loss": None,
            "mIoU": None,
            "Dice": None,
            "PixelAcc": None,
            "iou_person": None,
            "iou_car": None,
            "dice_person": None,
            "dice_car": None,
        }
        self._append_row(row)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if not self._is_local_process_zero(state):
            return

        row = {
            "event": "eval",
            "step": state.global_step,
            "epoch": self._value(metrics, "epoch"),
            "loss": None,
            "learning_rate": None,
            "eval_loss": self._value(metrics, "eval_loss"),
            "mIoU": self._value(metrics, "eval_mean_iou"),
            "Dice": self._value(metrics, "eval_mean_dice"),
            "PixelAcc": self._value(metrics, "eval_pixel_accuracy"),
            "iou_person": self._value(metrics, "eval_iou_person"),
            "iou_car": self._value(metrics, "eval_iou_car"),
            "dice_person": self._value(metrics, "eval_dice_person"),
            "dice_car": self._value(metrics, "eval_dice_car"),
        }
        self._append_row(row)


class PrettyTrainingCallback(TrainerCallback):
    """Emit compact, readable progress logs from Hugging Face Trainer events."""

    def __init__(
        self,
        train_samples: int,
        val_samples: int,
        effective_batch_size: int,
    ) -> None:
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.effective_batch_size = effective_batch_size
        self.start_time = time.perf_counter()
        self.best_mean_iou = 0.0

    @staticmethod
    def _fmt(value: Any, digits: int = 4) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    @staticmethod
    def _elapsed(seconds: float) -> str:
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}h{minutes:02d}m{secs:02d}s"
        return f"{minutes:02d}m{secs:02d}s"

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("+%s+", "-" * 70)
        logger.info(
            "| Training started | train=%d | val=%d | effective_batch=%d",
            self.train_samples,
            self.val_samples,
            self.effective_batch_size,
        )
        logger.info(
            "| epochs=%.0f | max_steps=%s | log_every=%d | eval=%s | save=%s",
            args.num_train_epochs,
            state.max_steps,
            args.logging_steps,
            args.eval_strategy,
            args.save_strategy,
        )
        logger.info("+%s+", "-" * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs:
            return

        elapsed = self._elapsed(time.perf_counter() - self.start_time)
        logger.info(
            "Train | step=%d/%d | epoch=%s | loss=%s | lr=%s | grad_norm=%s | elapsed=%s",
            state.global_step,
            state.max_steps,
            self._fmt(logs.get("epoch", 0.0), digits=2),
            self._fmt(logs.get("loss", 0.0)),
            self._fmt(logs.get("learning_rate", 0.0), digits=8),
            self._fmt(logs.get("grad_norm", "n/a")),
            elapsed,
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        mean_iou = float(metrics.get("eval_mean_iou", 0.0))
        if mean_iou > self.best_mean_iou:
            self.best_mean_iou = mean_iou

        logger.info("+%s+", "-" * 70)
        logger.info(
            "| Eval | epoch=%s | mIoU=%s | best=%s | loss=%s",
            self._fmt(metrics.get("epoch", state.epoch or 0.0), digits=2),
            self._fmt(mean_iou),
            self._fmt(self.best_mean_iou),
            self._fmt(metrics.get("eval_loss", 0.0)),
        )
        logger.info(
            "| IoU  | person=%s | car=%s",
            self._fmt(metrics.get("eval_iou_person", 0.0)),
            self._fmt(metrics.get("eval_iou_car", 0.0)),
        )
        logger.info(
            "| Dice | mean=%s | person=%s | car=%s",
            self._fmt(metrics.get("eval_mean_dice", 0.0)),
            self._fmt(metrics.get("eval_dice_person", 0.0)),
            self._fmt(metrics.get("eval_dice_car", 0.0)),
        )
        logger.info(
            "| Acc  | pixel=%s | foreground=%s | background=%s | fg_ratio=%s",
            self._fmt(metrics.get("eval_pixel_accuracy", 0.0)),
            self._fmt(metrics.get("eval_foreground_accuracy", 0.0)),
            self._fmt(metrics.get("eval_background_accuracy", 0.0)),
            self._fmt(metrics.get("eval_foreground_ratio", 0.0)),
        )
        logger.info(
            "| P/R  | person=%s/%s | car=%s/%s",
            self._fmt(metrics.get("eval_precision_person", 0.0)),
            self._fmt(metrics.get("eval_recall_person", 0.0)),
            self._fmt(metrics.get("eval_precision_car", 0.0)),
            self._fmt(metrics.get("eval_recall_car", 0.0)),
        )
        logger.info("+%s+", "-" * 70)


# ============================================================================
# Custom Trainer for Mask2Former
# ============================================================================

class Mask2FormerTrainer(Trainer):
    """Hugging Face Trainer subclass that correctly handles Mask2Former's
    non-standard output format (query-level logits instead of per-pixel logits).

    Two key overrides:

    1. ``compute_loss`` — passes all input fields (including mask_labels and
       class_labels) directly to the model, which computes loss internally via
       Hungarian matching.

    2. ``prediction_step`` — converts query logits into compact semantic masks
       immediately, keeping evaluation memory bounded and metrics correct.
    """

    def compute_loss(
        self,
        model: Mask2FormerForUniversalSegmentation,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        """Forward pass with loss computation built into the model."""
        # The model's forward handles Hungarian matching and loss internally
        # when mask_labels and class_labels are present.
        outputs = model(**inputs)
        loss = outputs.loss

        if loss is None:
            raise RuntimeError(
                "Model returned None loss. Ensure mask_labels and class_labels "
                "are present in the batch."
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: Mask2FormerForUniversalSegmentation,
        inputs: Dict[str, Any],
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[Any], Optional[Any]]:
        """Run one evaluation step and return compact semantic masks."""
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None

        if loss is not None:
            loss = loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        target_size = tuple(inputs["pixel_values"].shape[-2:])
        predictions = _semantic_predictions_from_queries(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            target_size=target_size,
        )
        labels = _semantic_targets_from_instances(
            inputs["mask_labels"],
            inputs["class_labels"],
        )

        return (loss, predictions.detach().cpu(), labels.detach().cpu())


# ============================================================================
# Entry point
# ============================================================================

def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Train Mask2Former for person + car cutout on RTX 4090.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/hvm_coco_512/annotations.json",
        help="Path to COCO-format annotation JSON.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/hvm_coco_512/images",
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./weights/mask2former-cutout",
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/mask2former-swin-small-coco-instance",
        help="Hugging Face model identifier for Mask2Former.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-GPU training batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Per-GPU evaluation batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear warmup.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of eval epochs with no improvement before stopping.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--val_subset_size",
        type=int,
        default=500,
        help="Number of images held out for validation.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Experiment tracking backend.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mask2former-cutout",
        help="W&B project name (only used when --report_to wandb).",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name (only used when --report_to wandb).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve paths relative to this script's parent directory (ai-core/)
    # ------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent.parent  # ai-core/
    annotation_path = Path(args.annotation_file)
    image_path = Path(args.image_dir)
    output_path = Path(args.output_dir)

    annotation_file = str(
        annotation_path
        if annotation_path.is_absolute()
        else script_dir / annotation_path
    )
    image_dir = str(
        image_path if image_path.is_absolute() else script_dir / image_path
    )
    output_dir = str(
        output_path if output_path.is_absolute() else script_dir / output_path
    )

    # Verify data exists
    if not Path(annotation_file).exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not Path(image_dir).is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    # ------------------------------------------------------------------
    # Device / precision
    # ------------------------------------------------------------------
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Using device: %s", device_name)
    logger.info("TF32 enabled: %s", torch.backends.cuda.matmul.allow_tf32)
    logger.info("BF16 supported: %s", torch.cuda.is_bf16_supported())

    # ------------------------------------------------------------------
    # Image processor
    #
    # The image processor handles normalisation (ImageNet stats) and
    # optional resizing.  Since our dataset is pre-cropped to 512x512,
    # we disable resizing to avoid interpolation aliasing on masks.
    # ------------------------------------------------------------------
    logger.info("Loading image processor from %s ...", args.model_id)
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(
            args.model_id,
            do_resize=False,
            do_normalize=True,
            do_reduce_labels=False,  # keep original class indices
            size={"height": 512, "width": 512},
        )
    except Exception:
        # Fallback: if network is unavailable, construct the processor manually
        # using known defaults for the Swin backbone.
        logger.warning(
            "Could not download processor config — building from defaults."
        )
        processor = Mask2FormerImageProcessor(
            do_resize=False,
            do_normalize=True,
            do_reduce_labels=False,
            size={"height": 512, "width": 512},
            image_mean=list(IMAGENET_MEAN),
            image_std=list(IMAGENET_STD),
            ignore_index=255,
        )

    # ------------------------------------------------------------------
    # Model
    #
    # Load the pretrained checkpoint and replace the 80-class COCO head
    # with a 2-class head.  ``ignore_mismatched_sizes=True`` is critical:
    # it allows the classification head weights to be safely discarded and
    # reinitialised while preserving the backbone, pixel decoder, and mask
    # decoder weights.
    # ------------------------------------------------------------------
    logger.info("Loading model from %s (num_labels=2) ...", args.model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: total=%d, trainable=%d (%.1f%% trainable)",
        total_params,
        trainable_params,
        100.0 * trainable_params / total_params,
    )

    # ------------------------------------------------------------------
    # Dataset split
    #
    # Use a deterministic split: first ``val_subset_size`` images after
    # shuffling go to validation, the rest to training.
    # ------------------------------------------------------------------
    logger.info("Building datasets (val_subset_size=%d) ...", args.val_subset_size)

    # Read all image IDs for a deterministic split
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    all_indices = list(range(len(coco["images"])))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_indices)

    val_indices = set(all_indices[: args.val_subset_size])
    train_indices = set(all_indices[args.val_subset_size :])

    # Build filtered image lists
    train_images = [
        img for i, img in enumerate(coco["images"]) if i in train_indices
    ]
    val_images = [
        img for i, img in enumerate(coco["images"]) if i in val_indices
    ]

    # Create datasets by setting the image list directly
    train_dataset = CocoCutoutDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        processor=processor,
        split="train",
        image_size=(512, 512),
    )
    train_dataset._images = train_images
    # Rebuild annotation lookup scoped to training images
    train_image_ids = {img["id"] for img in train_images}
    train_dataset._ann_by_image = {
        img_id: anns
        for img_id, anns in train_dataset._ann_by_image.items()
        if img_id in train_image_ids
    }

    val_dataset = CocoCutoutDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        processor=processor,
        split="val",
        image_size=(512, 512),
    )
    val_dataset._images = val_images
    val_image_ids = {img["id"] for img in val_images}
    val_dataset._ann_by_image = {
        img_id: anns
        for img_id, anns in val_dataset._ann_by_image.items()
        if img_id in val_image_ids
    }

    logger.info(
        "Train samples: %d | Val samples: %d",
        len(train_dataset),
        len(val_dataset),
    )

    # ------------------------------------------------------------------
    # Data collator
    # ------------------------------------------------------------------
    data_collator = DataCollatorForMask2Former()

    # ------------------------------------------------------------------
    # Training arguments — tuned for RTX 4090 24 GB
    #
    # Key design decisions:
    #   - BF16 over FP16: 4090 has BF16 tensor cores; no gradient overflow.
    #   - Gradient checkpointing OFF: 24 GB is plenty for 512x512 inputs.
    #   - tf32=True: accelerates residual FP32 matmuls in attention.
    #   - remove_unused_columns=False: CRITICAL for segmentation — prevents
    #     the Trainer from dropping mask_labels / class_labels.
    #   - load_best_model_at_end + metric_for_best_model: selects the
    #     checkpoint with the highest eval mIoU across all epochs.
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Epoch schedule
        num_train_epochs=args.epochs,
        # Batch & accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        # Precision
        bf16=True,
        fp16=False,
        tf32=True,
        # Memory
        gradient_checkpointing=False,
        # DataLoader
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        eval_accumulation_steps=1,
        # Critical: prevent Trainer from dropping segmentation-specific columns
        remove_unused_columns=False,
        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        disable_tqdm=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_iou",
        greater_is_better=True,
        save_total_limit=3,
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=False,  # let cuDNN use fastest algo
        # Reporting
        report_to=[args.report_to] if args.report_to != "none" else [],
        # Push to Hub (disabled by default for local training)
        push_to_hub=False,
    )

    # ------------------------------------------------------------------
    # WandB run name (optional)
    # ------------------------------------------------------------------
    if args.report_to == "wandb" and args.wandb_name:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = args.wandb_name

    # ------------------------------------------------------------------
    # Early stopping callback
    #
    # Monitors ``eval_mean_iou`` — stops training when the metric does not
    # improve for ``patience`` consecutive evaluation epochs.
    # ------------------------------------------------------------------
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.001,  # minimum absolute improvement
    )

    # ------------------------------------------------------------------
    # Instantiate Trainer
    # ------------------------------------------------------------------
    trainer = Mask2FormerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            early_stopping,
            MetricsLoggingCallback(
                metrics_path=Path(output_dir) / "training_metrics.csv",
            ),
            PrettyTrainingCallback(
                train_samples=len(train_dataset),
                val_samples=len(val_dataset),
                effective_batch_size=(
                    args.batch_size * args.gradient_accumulation_steps
                ),
            ),
        ],
    )

    # Keep console output readable: use PrettyTrainingCallback instead of the
    # default dict-style PrinterCallback.
    trainer.remove_callback(PrinterCallback)

    # ------------------------------------------------------------------
    # Launch training
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info(
        "Starting training | effective_batch=%d | train=%d | val=%d",
        args.batch_size * args.gradient_accumulation_steps,
        len(train_dataset),
        len(val_dataset),
    )
    logger.info(
        "Epochs=%d | early_stopping_patience=%d | metric=eval_mean_iou",
        args.epochs,
        args.early_stopping_patience,
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 72)

    trainer.train()

    # ------------------------------------------------------------------
    # Final evaluation & save
    # ------------------------------------------------------------------
    logger.info("Training complete. Running final evaluation ...")
    final_metrics = trainer.evaluate()
    logger.info("Final metrics: %s", final_metrics)

    # The best model is automatically loaded via load_best_model_at_end=True,
    # but we save it explicitly here for clarity.
    best_model_path = Path(output_dir) / "best_model"
    trainer.save_model(str(best_model_path))
    processor.save_pretrained(str(best_model_path))
    logger.info("Best model saved to %s", best_model_path)


if __name__ == "__main__":
    main()
