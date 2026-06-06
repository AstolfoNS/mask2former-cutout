"""
Inference engine for Mask2Former person & car cutout.

Loads the fine-tuned model from a local HuggingFace-format directory (e.g.,
``best_model/``), runs single-image inference, and produces mask, overlay, and
transparent cutout outputs saved to disk.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation

from . import config
from . import image_ops
from . import storage

# Ensure ai-core src is importable (for Mask2FormerCutout wrapper if needed)
AI_CORE_SRC = Path(__file__).resolve().parents[3] / "ai-core" / "src"
if str(AI_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(AI_CORE_SRC))

logger = logging.getLogger(__name__)


class CutoutEngine:
    """Inference engine wrapping Mask2Former for person & car cutout.

    Loads the model once at startup and exposes a ``predict`` method that
    accepts a PIL image and returns file paths to generated outputs.
    """

    LABEL_NAMES: List[str] = ["person", "car"]

    def __init__(
        self,
        model_dir: Optional[str] = None,
        hf_model_id: str = "facebook/mask2former-swin-small-coco-instance",
        device: Optional[str] = None,
        num_labels: int = 2,
        input_size: Optional[Tuple[int, int]] = None,
        inference_dtype: Optional[str] = None,
    ) -> None:
        self.input_size = input_size or (config.INPUT_SIZE, config.INPUT_SIZE)
        self.model_dir = str(model_dir or config.MODEL_DIR)
        self.device = torch.device(
            device or config.DEVICE
        )

        # Determine dtype for autocast
        dtype_str = inference_dtype or config.INFERENCE_DTYPE
        if dtype_str == "float16" and self.device.type == "cuda":
            self.autocast_dtype = torch.float16
        elif dtype_str == "bfloat16" and self.device.type == "cuda":
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float32

        # Resolve model source
        resolved_model = self.model_dir
        model_path = Path(resolved_model)

        if model_path.exists() and model_path.is_dir() and (model_path / "config.json").exists():
            logger.info("Loading fine-tuned model from local path: %s", model_path)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                str(model_path),
            )
        else:
            logger.info(
                "Local model not found at %s — loading from HuggingFace: %s",
                model_path,
                hf_model_id,
            )
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                hf_model_id,
                num_labels=num_labels,
                id2label={0: "person", 1: "car"},
                label2id={"person": 0, "car": 1},
                ignore_mismatched_sizes=True,
            )

        self.model.to(self.device)
        self.model.eval()

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "CutoutEngine ready on %s (%s) | params: %d",
            self.device,
            self.gpu_name,
            total_params,
        )

    @property
    def gpu_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return "CPU"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        target_classes: Optional[List[str]] = None,
        score_threshold: float = 0.5,
        return_mask: bool = True,
        return_overlay: bool = True,
        return_cutout: bool = True,
    ) -> Dict:
        """Run cutout inference and save results to disk.

        Args:
            image: Input PIL image (RGB).
            target_classes: Subset of ``["person", "car"]`` to use, or None for all.
            score_threshold: Sigmoid threshold for mask binarization.
            return_mask: Generate mask output.
            return_overlay: Generate overlay preview.
            return_cutout: Generate transparent cutout PNG.

        Returns:
            Dict with keys:
                - ``job_id``: Unique job identifier.
                - ``status``: "success" or "error".
                - ``classes``: Detected class names.
                - ``files``: Dict mapping output type to static URL.
                - ``timing``: Dict with timing breakdown in ms.
        """
        t_start = time.perf_counter()

        # Validate input
        if target_classes is None:
            target_classes = list(self.LABEL_NAMES)

        class_indices = [
            i for i, name in enumerate(self.LABEL_NAMES) if name in target_classes
        ]

        # --- Preprocessing ---
        t_pre = time.perf_counter()
        image_rgb = image_ops.load_image_rgb(image)
        original_size = image_rgb.shape[:2]  # (H, W)

        tensor, resized_image = image_ops.preprocess_for_model(
            image_rgb,
            target_size=self.input_size,
        )
        tensor = torch.from_numpy(tensor).to(self.device)
        t_pre_ms = (time.perf_counter() - t_pre) * 1000.0

        # --- Inference ---
        t_inf = time.perf_counter()
        with torch.autocast(
            self.device.type,
            dtype=self.autocast_dtype,
            enabled=(self.device.type == "cuda"),
        ):
            outputs = self.model(pixel_values=tensor)

        # Convert query-level logits to per-pixel mask predictions
        masks = self._semantic_masks_from_queries(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            target_size=self.input_size,
            score_threshold=score_threshold,
        )  # (C, H, W)
        t_inf_ms = (time.perf_counter() - t_inf) * 1000.0

        # --- Post-processing ---
        t_post = time.perf_counter()

        # Resize each class mask to original image size
        masks_np = masks.cpu().numpy()
        resized_masks: Dict[str, np.ndarray] = {}
        detected_classes: List[str] = []

        for idx, name in enumerate(self.LABEL_NAMES):
            if idx in class_indices:
                mask = image_ops.resize_mask_to_original(
                    masks_np[idx], original_size
                )
                # Refine mask
                mask = image_ops.refine_mask(
                    mask,
                    min_area_ratio=config.MIN_AREA_RATIO,
                    morph_kernel_size=config.MORPH_KERNEL_SIZE,
                    blur_kernel_size=config.BLUR_KERNEL_SIZE,
                )
                resized_masks[name] = mask

                if mask.max() >= score_threshold:
                    detected_classes.append(name)

        if resized_masks:
            ordered_names = [
                name for name in self.LABEL_NAMES if name in resized_masks
            ]
            stacked_masks = np.stack(
                [resized_masks[name] for name in ordered_names],
                axis=0,
            )
            best_scores = stacked_masks.max(axis=0)
            best_classes = stacked_masks.argmax(axis=0)

            resized_masks = {
                name: (
                    (best_classes == local_idx) & (best_scores >= 0.5)
                ).astype(np.float32)
                for local_idx, name in enumerate(ordered_names)
            }
            detected_classes = [
                name for name in ordered_names if resized_masks[name].max() >= 0.5
            ]

        # Build combined mask
        combined_labels = np.zeros(original_size, dtype=np.uint8)
        for idx, name in enumerate(self.LABEL_NAMES):
            if name in resized_masks:
                binary = (resized_masks[name] >= 0.5).astype(np.uint8)
                combined_labels = np.maximum(combined_labels, binary * (idx + 1))

        combined_binary = (combined_labels > 0).astype(np.uint8)

        # --- Save outputs ---
        job_id = storage.generate_job_id()
        job_dir = storage.create_job_dir(job_id)

        # Save original image
        original_path = job_dir / "original.png"
        Image.fromarray(image_rgb).save(str(original_path))

        files: Dict[str, str] = {
            "original_url": storage.get_static_url(original_path),
        }

        if return_mask:
            mask_path = job_dir / "mask_combined.png"
            image_ops.save_image(combined_binary * 255, str(mask_path))
            files["mask_url"] = storage.get_static_url(mask_path)

            # Per-class masks
            for name, mask in resized_masks.items():
                cls_mask_path = job_dir / f"mask_{name}.png"
                image_ops.save_image((mask >= 0.5).astype(np.uint8) * 255, str(cls_mask_path))
                files[f"mask_{name}_url"] = storage.get_static_url(cls_mask_path)

        if return_overlay:
            overlay = image_ops.generate_overlay(image_rgb, combined_labels)
            overlay_path = job_dir / "overlay.png"
            image_ops.save_image(overlay, str(overlay_path))
            files["overlay_url"] = storage.get_static_url(overlay_path)

        if return_cutout:
            cutout_rgba = image_ops.generate_cutout(image_rgb, combined_binary)
            cutout_path = job_dir / "cutout.png"
            image_ops.save_image(cutout_rgba, str(cutout_path))
            files["cutout_url"] = storage.get_static_url(cutout_path)

        t_post_ms = (time.perf_counter() - t_post) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        logger.info(
            "Inference completed | job=%s | classes=%s | "
            "pre=%.1fms inf=%.1fms post=%.1fms total=%.1fms",
            job_id,
            detected_classes,
            t_pre_ms,
            t_inf_ms,
            t_post_ms,
            t_total_ms,
        )

        response = {
            "job_id": job_id,
            "status": "success",
            "classes": detected_classes,
            "model_id": getattr(self, "model_id", "default"),
            "model_label": getattr(
                self,
                "model_label",
                getattr(self, "model_id", "default"),
            ),
            "files": files,
            "timing": {
                "preprocess_ms": round(t_pre_ms, 2),
                "inference_ms": round(t_inf_ms, 2),
                "postprocess_ms": round(t_post_ms, 2),
                "total_ms": round(t_total_ms, 2),
            },
        }

        result_path = job_dir / "result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _semantic_masks_from_queries(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        target_size: Tuple[int, int],
        score_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Convert query-level logits to per-class semantic masks.

        Args:
            masks_queries_logits: ``(B, Q, h, w)`` mask logits.
            class_queries_logits: ``(B, Q, C+1)`` class logits (last is no-object).
            target_size: ``(H, W)`` to upsample masks to.
            score_threshold: Threshold for binary mask extraction.

        Returns:
            ``(C, H, W)`` tensor of mutually exclusive binary masks, one per
            class.
        """
        class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]  # (B, Q, C)
        mask_probs = masks_queries_logits.sigmoid()                     # (B, Q, h, w)

        # Upsample query masks to target resolution
        h_target, w_target = target_size
        mask_probs = F.interpolate(
            mask_probs,
            size=(h_target, w_target),
            mode="bilinear",
            align_corners=False,
        )  # (B, Q, H, W)

        # Weighted sum: for each class, sum query masks weighted by class prob.
        # Then assign each foreground pixel to exactly one class. Thresholding
        # every class independently allows person/car masks to overlap because
        # the same query masks can contribute probability mass to both classes.
        class_scores = torch.einsum(
            "bqc,bqhw->bchw", class_probs, mask_probs
        )  # (B, C, H, W)

        scores, predictions = class_scores.max(dim=1)  # (B, H, W)
        foreground = scores >= score_threshold

        binary_masks = torch.zeros_like(class_scores)
        binary_masks.scatter_(1, predictions.unsqueeze(1), 1.0)
        binary_masks = binary_masks * foreground.unsqueeze(1).float()
        return binary_masks.squeeze(0)  # (C, H, W)
