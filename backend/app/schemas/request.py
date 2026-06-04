"""
Pydantic request schemas for the cutout API.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field


class CutoutRequest(BaseModel):
    """Request body for the cutout inference endpoint.

    The image is uploaded as multipart/form-data.
    Target classes can be optionally specified to filter results.
    """

    target_classes: Optional[List[str]] = Field(
        default=None,
        description="Classes to include in the mask. None means all (person, car).",
        examples=[["person"], ["person", "car"]],
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for mask binarization.",
    )
    return_format: str = Field(
        default="png",
        description="Output mask format: 'png', 'alpha', or 'composite'.",
        examples=["png", "alpha", "composite"],
    )
