"""
Application configuration loaded from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Resolve the project root (mask2former-cutout/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_or(name: str, default: str) -> str:
    value = os.getenv(name, "")
    return value if value else default


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_DIR: Path = Path(
    _env_or(
        "MODEL_DIR",
        str(PROJECT_ROOT / "ai-core" / "weights" / "mask2former-cutout" / "best_model"),
    )
).resolve()

MODEL_ROOT: Path = Path(
    _env_or("MODEL_ROOT", str(PROJECT_ROOT / "ai-core" / "weights"))
).resolve()

HF_MODEL_ID: str = _env_or(
    "HF_MODEL_ID",
    "facebook/mask2former-swin-small-coco-instance",
)

DEVICE: str = _env_or("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")

INFERENCE_DTYPE: str = _env_or("INFERENCE_DTYPE", "float16")

INPUT_SIZE: int = int(_env_or("INPUT_SIZE", "512"))

SCORE_THRESHOLD: float = float(_env_or("SCORE_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Upload & output
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = int(_env_or("MAX_UPLOAD_MB", "20"))

UPLOAD_DIR: Path = Path(
    _env_or("UPLOAD_DIR", str(PROJECT_ROOT / "backend" / "app" / "static" / "uploads"))
)

OUTPUT_DIR: Path = Path(
    _env_or("OUTPUT_DIR", str(PROJECT_ROOT / "backend" / "app" / "static" / "outputs"))
)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
MIN_AREA_RATIO: float = float(_env_or("MIN_AREA_RATIO", "0.001"))
MORPH_KERNEL_SIZE: int = int(_env_or("MORPH_KERNEL", "3"))
BLUR_KERNEL_SIZE: int = int(_env_or("BLUR_KERNEL", "3"))
