"""
Mask2Former Cutout model wrapper.

Wraps the HuggingFace ``Mask2FormerForUniversalSegmentation`` for the
person + car binary cutout task. The pretrained classification head is
replaced with a 2-label head and the mask decoder is kept frozen for the
initial fine-tuning phase.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

logger = logging.getLogger(__name__)


class Mask2FormerCutout(nn.Module):
    """Mask2Former binary segmentation model for person & car cutout.

    Loads a pretrained checkpoint from HuggingFace Hub, replaces the
    classification head with a 2-label head, and exposes a simplified
    ``forward`` that returns upsampled mask logits.
    """

    def __init__(
        self,
        hf_model_id: str = "facebook/mask2former-swin-small-coco-instance",
        num_labels: int = 2,
        ignore_mismatched_sizes: bool = True,
    ) -> None:
        super().__init__()

        logger.info("Loading Mask2Former from %s ...", hf_model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            hf_model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        self.num_labels = num_labels
        self.config: Mask2FormerConfig = self.model.config

        # The HuggingFace model already replaces the class head when
        # num_labels differs from the pretrained checkpoint.  Log the
        # resulting configuration for debugging.
        logger.info(
            "Mask2Former loaded. num_labels=%d, hidden_dim=%d, num_queries=%d",
            self.config.num_labels,
            self.config.hidden_dim,
            self.config.decoder_config.get("num_queries", "N/A"),
        )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: (B, 3, H, W) normalized images.

        Returns:
            Dict with keys:
                - ``mask_logits``: (B, num_labels, H, W) per-pixel logits for each class.
                - ``class_logits``: (B, num_queries, num_labels + 1) class predictions.
        """
        outputs = self.model(pixel_values=pixel_values)

        # masks_queries_logits: (B, num_queries, H/4, W/4)
        # class_queries_logits: (B, num_queries, num_labels + 1)
        mask_queries = outputs.masks_queries_logits  # (B, Q, h, w)
        class_queries = outputs.class_queries_logits  # (B, Q, C+1)

        # Upsample masks to input resolution
        B, Q, h, w = mask_queries.shape
        _, _, H, W = pixel_values.shape
        masks_upsampled = F.interpolate(
            mask_queries,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, Q, H, W)

        # For each class (excluding "no-object"), re-weight the upsampled masks
        # by the predicted class probability to obtain per-class masks.
        class_probs = F.softmax(class_queries, dim=-1)[:, :, :-1]  # (B, Q, C)
        class_probs = class_probs.permute(0, 2, 1)  # (B, C, Q)
        mask_logits = torch.einsum("bqhw,bcq->bchw", masks_upsampled, class_probs)

        return {
            "mask_logits": mask_logits,
            "class_logits": class_queries,
        }
