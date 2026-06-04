"""
Mask2Former-Cutout Backend Service Entry Point.

Starts the FastAPI inference server with Uvicorn.

Usage:
    # Development (with auto-reload)
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # Production (with Gunicorn)
    gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_engine
from app.core.engine import CutoutEngine

# Load env vars from ai-core/.env if present
ENV_FILE = Path(__file__).resolve().parent.parent / "ai-core" / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI application."""
    logger.info("=== Mask2Former-Cutout Backend Starting ===")

    checkpoint = os.getenv("MASK2FORMER_CHECKPOINT", None)
    app.state.engine = CutoutEngine(
        checkpoint_path=checkpoint,
        hf_model_id=os.getenv(
            "HF_MODEL_ID", "facebook/mask2former-swin-small-coco-instance"
        ),
    )
    set_engine(app.state.engine)

    logger.info("Engine loaded. Server ready.")
    yield

    logger.info("=== Mask2Former-Cutout Backend Shutting Down ===")


app = FastAPI(
    title="Mask2Former-Cutout API",
    description="High-performance person & car image cutout microservice.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "Mask2Former-Cutout", "version": "0.1.0"}
