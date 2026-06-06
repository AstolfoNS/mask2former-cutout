"""
Pydantic response schemas for the cutout API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field(default="ok", description="Service status: 'ok' or 'degraded'.")
    device: str = Field(default="cpu", description="Device used for inference.")
    model_loaded: bool = Field(default=False, description="Whether the model is loaded.")
    gpu_name: str = Field(default="", description="GPU name, if available.")
    version: str = Field(default="0.1.0", description="Service version.")
    active_model_id: str = Field(default="default", description="Current model id.")


class ModelInfoResponse(BaseModel):
    """Available local model directory."""

    id: str = Field(..., description="Model identifier used by API requests.")
    label: str = Field(..., description="Human-readable model label.")
    path: str = Field(..., description="Resolved local model path.")
    active: bool = Field(default=False, description="Whether this model is active.")


class ModelListResponse(BaseModel):
    """List of available local models."""

    models: List[ModelInfoResponse] = Field(default_factory=list)


class TimingInfo(BaseModel):
    """Timing breakdown for an inference request."""

    preprocess_ms: float = Field(default=0.0, description="Preprocessing time in ms.")
    inference_ms: float = Field(default=0.0, description="Model inference time in ms.")
    postprocess_ms: float = Field(default=0.0, description="Post-processing time in ms.")
    total_ms: float = Field(default=0.0, description="Total wall-clock time in ms.")


class CutoutResponse(BaseModel):
    """Response returned after a successful cutout inference."""

    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(default="success", description="Processing status.")
    classes: List[str] = Field(
        default_factory=list,
        description="List of class names detected in the image.",
    )
    model_id: str = Field(default="default", description="Model id used for inference.")
    model_label: str = Field(default="default", description="Model label used for inference.")
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of output type to static file URL.",
    )
    timing: TimingInfo = Field(
        default_factory=TimingInfo,
        description="Timing breakdown in milliseconds.",
    )


class ResultQueryResponse(BaseModel):
    """Response for querying a previous result by job ID."""

    job_id: str = Field(..., description="Job identifier.")
    status: str = Field(default="success", description="Query status.")
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of output type to static file URL.",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    status: str = Field(default="error", description="Always 'error'.")
    code: str = Field(default="UNKNOWN", description="Error code for machine consumption.")
    message: str = Field(default="", description="Human-readable error description.")
