"""
FastAPI route definitions for the Mask2Former cutout service.

Endpoints:
    GET  /api/health          — Health check with model and GPU status.
    POST /api/segment         — Upload image and run cutout inference.
    GET  /api/results/{job_id} — Query a previous inference result.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..core import config
from ..core.engine import CutoutEngine
from ..core.storage import get_static_url
from ..schemas.response import (
    CutoutResponse,
    ErrorResponse,
    HealthResponse,
    ResultQueryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["cutout"])

# Engine instance is injected by main.py at startup
_engine: Optional[CutoutEngine] = None


def set_engine(engine: CutoutEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> CutoutEngine:
    if _engine is None:
        raise RuntimeError("CutoutEngine has not been initialized.")
    return _engine


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health-check endpoint.

    Returns model loading status and GPU availability.
    """
    try:
        engine = get_engine()
        return HealthResponse(
            status="ok",
            device=engine.device.type,
            model_loaded=True,
            gpu_name=engine.gpu_name,
            version="0.1.0",
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            device="cpu",
            model_loaded=False,
            gpu_name="",
            version="0.1.0",
        )


# ---------------------------------------------------------------------------
# Image segmentation
# ---------------------------------------------------------------------------

@router.post("/segment", response_model=CutoutResponse)
async def segment(
    file: UploadFile = File(..., description="Image file (JPG/PNG/WEBP)"),
    target_classes: Optional[str] = Form(
        default=None,
        description='Comma-separated class names, e.g. "person,car". '
                    'Omit for all classes.',
    ),
    score_threshold: float = Form(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for mask binarization.",
    ),
    return_overlay: bool = Form(
        default=True,
        description="Generate an overlay preview image.",
    ),
    return_mask: bool = Form(
        default=True,
        description="Generate mask output images.",
    ),
    return_cutout: bool = Form(
        default=True,
        description="Generate a transparent cutout PNG.",
    ),
) -> CutoutResponse:
    """Extract person and/or car masks from an uploaded image.

    Returns a JSON response with job ID, detected classes, file URLs, and
    timing breakdown. Generated files are saved under the static outputs
    directory and served via ``/static/outputs/{job_id}/...``.

    Supported image formats: JPG, PNG, WEBP.
    Maximum file size is controlled by ``MAX_UPLOAD_MB`` configuration.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, or WEBP).",
        )

    # Validate file size
    max_size_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size is {config.MAX_UPLOAD_MB} MB.",
        )

    # Decode image
    from PIL import Image
    import io

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode the uploaded image. Ensure it is a valid JPG, PNG, or WEBP file.",
        )

    # Parse target classes
    classes_list: Optional[List[str]] = None
    if target_classes:
        classes_list = [c.strip() for c in target_classes.split(",") if c.strip()]

    # Run inference
    try:
        engine = get_engine()
        result = engine.predict(
            image=image,
            target_classes=classes_list,
            score_threshold=score_threshold,
            return_mask=return_mask,
            return_overlay=return_overlay,
            return_cutout=return_cutout,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    return CutoutResponse(
        job_id=result["job_id"],
        status=result["status"],
        classes=result["classes"],
        files=result["files"],
        timing=result["timing"],
    )


# ---------------------------------------------------------------------------
# Result query
# ---------------------------------------------------------------------------

@router.get(
    "/results/{job_id}",
    response_model=ResultQueryResponse,
)
async def get_result(job_id: str) -> ResultQueryResponse:
    """Query a previously completed inference result by job ID.

    Returns 404 if the result directory does not exist.
    """
    job_dir = config.OUTPUT_DIR / job_id

    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Result not found for job_id: {job_id}",
        )

    # Discover available output files
    files: dict[str, str] = {}
    for f in job_dir.iterdir():
        if f.is_file():
            files[f.stem] = get_static_url(f)

    return ResultQueryResponse(
        job_id=job_id,
        status="success",
        files=files,
    )
