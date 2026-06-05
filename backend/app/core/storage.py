"""
File storage management for uploads and inference outputs.

Creates per-job directories under the configured output path and provides
cleanup utilities for disk space management.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from . import config

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Generate a unique job identifier.

    Format: ``YYYYMMDD_HHMMSS_<short-uuid>``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{ts}_{short_id}"


def create_job_dir(job_id: str) -> Path:
    """Create and return the output directory for a job.

    Creates the directory if it does not exist.
    """
    job_dir = config.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def save_upload(upload_dir: Path, file_bytes: bytes, original_filename: str) -> Path:
    """Save an uploaded file to disk.

    Args:
        upload_dir: Directory to save into (typically a job-specific dir).
        file_bytes: Raw file bytes.
        original_filename: Original filename for extension detection.

    Returns:
        Path to the saved file.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix or ".png"
    dest = upload_dir / f"original{suffix}"
    dest.write_bytes(file_bytes)
    logger.info("Upload saved: %s", dest)
    return dest


def get_static_url(absolute_path: Path) -> str:
    """Convert an absolute filesystem path to a static URL relative path.

    Example: ``/app/static/outputs/job123/cutout.png`` -> ``/static/outputs/job123/cutout.png``
    """
    try:
        rel = absolute_path.relative_to(config.PROJECT_ROOT / "backend" / "app")
        return "/" + str(rel).replace("\\", "/")
    except ValueError:
        # Fallback: assume the path is under the project root
        return str(absolute_path)


def cleanup_old_results(max_age_hours: int = 24) -> int:
    """Remove result directories older than ``max_age_hours``.

    Returns:
        Number of directories removed.
    """
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    removed = 0

    if not config.OUTPUT_DIR.exists():
        return 0

    for job_dir in config.OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        try:
            mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
            if mtime < cutoff:
                shutil.rmtree(job_dir)
                removed += 1
                logger.info("Cleaned up old result: %s", job_dir.name)
        except OSError:
            logger.warning("Failed to clean up: %s", job_dir)

    return removed
