"""
Loss functions for binary mask segmentation.

Provides Binary Cross-Entropy (BCE) and Dice loss, combined with
configurable weights for the Mask2Former cutout task.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BinaryCrossEntropyLoss(nn.Module):
    """Binary cross-entropy loss for multi-label mask segmentation.

    Each class (person, car) is treated as an independent binary
    classification problem.  Numerically stable with log-sum-exp trick.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: (B, C, H, W) raw logits per class.
            targets: (B, H, W) integer labels (0=background, 1=person, 2=car).

        Returns:
            Scalar loss.
        """
        # Convert integer targets to one-hot: (B, C, H, W)
        C = logits.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=C + 1).float()
        targets_one_hot = targets_one_hot[:, :, :, 1:].permute(0, 3, 1, 2)  # drop bg channel

        bce = F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction="mean")
        return self.weight * bce


class DiceLoss(nn.Module):
    """Soft Dice loss for mask segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|).  Smoothing factor prevents division
    by zero on empty masks.
    """

    def __init__(self, weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: (B, C, H, W) raw logits per class.
            targets: (B, H, W) integer labels.

        Returns:
            Scalar loss (1 - dice, averaged over batch and classes).
        """
        C = logits.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=C + 1).float()
        targets_one_hot = targets_one_hot[:, :, :, 1:].permute(0, 3, 1, 2)  # (B, C, H, W)

        probs = torch.sigmoid(logits)

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.weight * dice_loss


class CombinedLoss(nn.Module):
    """Combine BCE and Dice losses with configurable weights."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = BinaryCrossEntropyLoss(weight=bce_weight)
        self.dice = DiceLoss(weight=dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return {"bce_loss": bce, "dice_loss": dice, "total_loss": bce + dice}


def build_loss(loss_cfg: Any) -> CombinedLoss:
    """Build the combined loss from Hydra config."""
    bce_w = loss_cfg.mask_loss.weight if hasattr(loss_cfg.mask_loss, "weight") else 1.0
    dice_w = loss_cfg.dice_loss.weight if hasattr(loss_cfg.dice_loss, "weight") else 1.0
    return CombinedLoss(bce_weight=bce_w, dice_weight=dice_w)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    loss_module: CombinedLoss,
) -> torch.Tensor:
    """Convenience wrapper: compute total loss from model outputs and targets."""
    losses = loss_module(outputs["mask_logits"], targets)
    return losses["total_loss"]
