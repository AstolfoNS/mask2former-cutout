"""
Inference engine that wraps the Mask2Former model for runtime use.

Loads a fine-tuned checkpoint (or the base HuggingFace model as fallback),
runs inference on a single image, and returns binary masks per class.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure ai-core src is importable
AI_CORE_SRC = Path(__file__).resolve().parents[3] / "ai-core" / "src"
if str(AI_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(AI_CORE_SRC))

logger = logging.getLogger(__name__)


class CutoutEngine:
    """Thin wrapper around Mask2FormerCutout for synchronous inference."""

    LABEL_NAMES = ["person", "car"]

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hf_model_id: str = "facebook/mask2former-swin-small-coco-instance",
        device: Optional[str] = None,
        num_labels: int = 2,
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.image_size = image_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        from src.models import Mask2FormerCutout

        logger.info("Loading model (hf=%s, ckpt=%s) ...", hf_model_id, checkpoint_path)
        self.model = Mask2FormerCutout(
            hf_model_id=hf_model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model_state_dict"], strict=False)
            logger.info("Fine-tuned checkpoint loaded: %s", checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization (must match training)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        logger.info("CutoutEngine ready on %s.", self.device)

    @property
    def gpu_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return "CPU"

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        return_format: str = "png",
    ) -> Dict:
        """Run cutout inference on a single PIL image.

        Args:
            image: Input PIL image (RGB).
            target_classes: Subset of ["person", "car"] to return.
            confidence_threshold: Sigmoid threshold for mask binarization.
            return_format: "png" (base64-encoded mask), "alpha" (RGBA with
                           mask as alpha), or "composite" (original overlaid
                           with mask).

        Returns:
            Dict with keys: mask_base64, detected_classes, processing_time_ms.
        """
        t_start = time.perf_counter()

        # --- Preprocessing ---
        img_np = np.array(image.convert("RGB").resize(self.image_size))
        tensor = (
            torch.from_numpy(img_np.transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)  # (1, 3, H, W)
        )
        tensor = tensor / 255.0
        tensor = (tensor - self.norm_mean) / self.norm_std
        tensor = tensor.to(self.device)

        # --- Inference ---
        outputs = self.model(tensor)
        mask_logits = outputs["mask_logits"]  # (1, C, H, W)

        probs = torch.sigmoid(mask_logits).squeeze(0).cpu().numpy()  # (C, H, W)

        # --- Post-processing ---
        if target_classes is None:
            target_classes = list(self.LABEL_NAMES)

        class_indices = [
            i for i, name in enumerate(self.LABEL_NAMES) if name in target_classes
        ]
        detected = [
            name
            for i, name in enumerate(self.LABEL_NAMES)
            if i in class_indices and probs[i].max() >= confidence_threshold
        ]

        # Build combined mask from requested classes
        combined = np.zeros(self.image_size, dtype=np.uint8)
        for ci in class_indices:
            binary = (probs[ci] >= confidence_threshold).astype(np.uint8)
            combined = np.maximum(combined, binary * (ci + 1))  # 1=person, 2=car

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        # --- Encode ---
        mask_base64 = ""
        if return_format == "png":
            mask_base64 = self._encode_png(combined)
        elif return_format == "alpha":
            mask_base64 = self._encode_alpha(img_np, combined)
        elif return_format == "composite":
            mask_base64 = self._encode_composite(img_np, combined)

        return {
            "mask_base64": mask_base64,
            "detected_classes": detected,
            "processing_time_ms": round(t_elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_png(mask: np.ndarray) -> str:
        """Encode a single-channel mask as base64 PNG."""
        import base64

        img = Image.fromarray(mask * 127, mode="L")  # 0, 127, 254
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _encode_alpha(image_rgb: np.ndarray, mask: np.ndarray) -> str:
        """Overlay mask as alpha channel on the original image."""
        import base64

        alpha = (mask > 0).astype(np.uint8) * 255
        rgba = np.dstack([image_rgb, alpha])
        img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _encode_composite(image_rgb: np.ndarray, mask: np.ndarray) -> str:
        """Composite: original image with semi-transparent mask overlay."""
        import base64

        colors = {1: [0, 255, 0], 2: [0, 0, 255]}  # person=green, car=blue
        overlay = np.zeros_like(image_rgb, dtype=np.float32)
        for cls_id, color in colors.items():
            overlay[mask == cls_id] = color
        composite = (image_rgb * 0.6 + overlay * 0.4).astype(np.uint8)
        img = Image.fromarray(composite)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
