"""Model discovery and lazy loading for inference comparisons."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from . import config
from .engine import CutoutEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    path: str
    active: bool = False


def _is_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
    )


def discover_models() -> List[ModelInfo]:
    """Discover local Hugging Face model directories under MODEL_ROOT."""
    candidates: Dict[str, Path] = {}

    default_path = config.MODEL_DIR
    if _is_model_dir(default_path):
        candidates["default"] = default_path

    model_root = config.MODEL_ROOT
    if model_root.is_dir():
        for path in sorted(model_root.glob("*")):
            if not path.is_dir():
                continue
            best_model = path / "best_model"
            if _is_model_dir(best_model):
                candidates[f"{path.name}/best_model"] = best_model
            for checkpoint in sorted(path.glob("checkpoint-*")):
                if _is_model_dir(checkpoint):
                    candidates[f"{path.name}/{checkpoint.name}"] = checkpoint

    default_resolved = default_path.resolve()
    models = [
        ModelInfo(
            id=model_id,
            label=model_id,
            path=str(path.resolve()),
            active=path.resolve() == default_resolved,
        )
        for model_id, path in candidates.items()
    ]
    return sorted(models, key=lambda item: (not item.active, item.id))


class ModelManager:
    """Keep one model loaded at a time and switch lazily by model id."""

    def __init__(self, default_engine: CutoutEngine) -> None:
        self._default_engine = default_engine
        self._active_engine = default_engine
        self._active_model_id = "default"
        self._default_engine.model_id = "default"
        self._default_engine.model_label = "default"

    @property
    def active_model_id(self) -> str:
        return self._active_model_id

    def list_models(self) -> List[ModelInfo]:
        default_info = ModelInfo(
            id="default",
            label="default",
            path=str(Path(self._default_engine.model_dir).resolve()),
            active=self._active_model_id == "default",
        )
        discovered_models = [
            ModelInfo(
                id=model.id,
                label=model.label,
                path=model.path,
                active=model.id == self._active_model_id,
            )
            for model in discover_models()
            if model.id != "default" and model.path != default_info.path
        ]
        return [default_info, *discovered_models]

    def get_engine(self, model_id: Optional[str]) -> CutoutEngine:
        requested_id = model_id or self._active_model_id or "default"
        if requested_id == self._active_model_id:
            return self._active_engine

        old_engine = self._active_engine
        if requested_id == "default":
            self._active_engine = self._default_engine
            self._active_model_id = "default"
            self._default_engine.model_id = "default"
            self._default_engine.model_label = "default"
            self._release_engine(old_engine)
            return self._active_engine

        model = next(
            (item for item in discover_models() if item.id == requested_id),
            None,
        )
        if model is None:
            raise ValueError(f"Unknown model_id: {requested_id}")
        if model.path == str(Path(self._default_engine.model_dir).resolve()):
            self._active_engine = self._default_engine
            self._active_model_id = "default"
            self._default_engine.model_id = "default"
            self._default_engine.model_label = "default"
            self._release_engine(old_engine)
            return self._active_engine

        logger.info(
            "Switching inference model: %s -> %s",
            self._active_model_id,
            requested_id,
        )
        self._active_engine = CutoutEngine(model_dir=model.path)
        self._active_engine.model_id = model.id
        self._active_engine.model_label = model.label
        self._active_model_id = requested_id
        self._release_engine(old_engine)
        return self._active_engine

    def _release_engine(self, old_engine: CutoutEngine) -> None:
        if old_engine is not self._default_engine:
            del old_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
