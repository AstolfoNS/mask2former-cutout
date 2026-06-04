"""
Pydantic response schemas for the cutout API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CutoutResponse(BaseModel):
    """Response returned after a successful cutout inference."""

    status: str = Field(default="success", description="Processing status.")
    message: str = Field(default="Cutout completed successfully.")
    mask_url: str = Field(
        default="",
        description="Base64-encoded PNG or relative path to the generated mask.",
    )
    mask_base64: str = Field(
        default="",
        description="Base64-encoded mask image (when return_type=inline).",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Inference wall-clock time in milliseconds.",
    )
    detected_classes: List[str] = Field(
        default_factory=list,
        description="List of classes detected in the image.",
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field(default="ok")
    model_loaded: bool = Field(default=False)
    gpu_available: bool = Field(default=False)
    gpu_name: str = Field(default="")
    version: str = Field(default="0.1.0")
