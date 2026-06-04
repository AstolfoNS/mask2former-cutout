"""
FastAPI route definitions for the cutout service.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..core.engine import CutoutEngine
from ..schemas.response import CutoutResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["cutout"])

# Engine instance is injected by main.py at startup
_engine: Optional[CutoutEngine] = None


def set_engine(engine: CutoutEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> CutoutEngine:
    if _engine is None:
        raise RuntimeError("CutoutEngine has not been initialized.")
    return _engine


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health-check endpoint."""
    try:
        engine = get_engine()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            gpu_available=engine.device.type == "cuda",
            gpu_name=engine.gpu_name,
            version="0.1.0",
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            gpu_available=False,
            gpu_name="",
            version="0.1.0",
        )


@router.post("/cutout", response_model=CutoutResponse)
async def cutout(
    file: UploadFile = File(..., description="Image file (JPG/PNG/WEBP)"),
    target_classes: Optional[str] = Form(
        default=None,
        description='Comma-separated classes, e.g. "person,car"',
    ),
    confidence_threshold: float = Form(default=0.5, ge=0.0, le=1.0),
    return_format: str = Form(default="png"),
) -> CutoutResponse:
    """Extract person and/or car masks from an uploaded image.

    Returns a base64-encoded mask image.
    """
    from PIL import Image

    # Validate input
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(__import__("io").BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, detail="Cannot decode the uploaded image.")

    # Parse target classes
    classes_list: Optional[List[str]] = None
    if target_classes:
        classes_list = [c.strip() for c in target_classes.split(",") if c.strip()]

    # Run inference
    engine = get_engine()
    result = engine.predict(
        image=image,
        target_classes=classes_list,
        confidence_threshold=confidence_threshold,
        return_format=return_format,
    )

    return CutoutResponse(
        status="success",
        message="Cutout completed successfully.",
        mask_base64=result["mask_base64"],
        detected_classes=result["detected_classes"],
        processing_time_ms=result["processing_time_ms"],
    )
