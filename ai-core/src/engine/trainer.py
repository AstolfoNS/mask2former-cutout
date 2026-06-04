"""
Training engine with BF16 mixed-precision, TF32 acceleration, and wandb logging.

Optimized for a single RTX 4090 (24 GB VRAM).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import build_dataloaders
from ..models import Mask2FormerCutout, build_loss
from ..models.losses import compute_total_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the full training and validation loop for Mask2Former cutout."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

        # Enable TF32 for Ampere (RTX 4090)
        if cfg.train.use_tf32 and self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            logger.info("TF32 matmul precision set to 'high'.")

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Build loss module
        self.loss_module = build_loss(cfg.model.loss)

        # Optimizer & scheduler
        self.optimizer, self.scheduler = self._build_optimizer()

        # Data
        self.train_loader, self.val_loader = build_dataloaders(
            annotation_file=cfg.data.annotation_file,
            image_dir=cfg.data.image_dir,
            image_size=tuple(cfg.data.image_size),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers,
            val_subset_size=cfg.train.val.subset_size,
            seed=cfg.train.seed,
        )

        # Bookkeeping
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        self._init_wandb()

        # Gradient scaler (not needed for BF16, but kept for potential fp32 fallback)
        self.scaler = torch.GradScaler("cuda", enabled=not cfg.train.use_bf16)

    def _build_model(self) -> Mask2FormerCutout:
        """Instantiate the Mask2Former model."""
        model = Mask2FormerCutout(
            hf_model_id=self.cfg.model.hf_model_id,
            num_labels=self.cfg.model.num_labels,
            ignore_mismatched_sizes=self.cfg.model.ignore_mismatched_sizes,
        )
        return model

    def _build_optimizer(self):
        """Build AdamW optimizer with cosine-warmup LR schedule."""
        opt_cfg = self.cfg.train.optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
        )

        sch_cfg = self.cfg.train.scheduler
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=sch_cfg.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.total_steps - sch_cfg.warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[sch_cfg.warmup_steps],
        )
        return optimizer, scheduler

    def _init_wandb(self) -> None:
        """Initialize wandb if enabled."""
        if self.cfg.wandb.enabled:
            import wandb

            wandb.init(
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.name,
                tags=self.cfg.wandb.tags,
                config=dict(self.cfg),
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the main training loop."""
        logger.info("Starting training on device: %s", self.device)
        logger.info("Batch size: %d | Gradient accumulation: %d | BF16: %s",
                     self.cfg.train.batch_size,
                     self.cfg.train.gradient_accumulation_steps,
                     self.cfg.train.use_bf16)

        self.model.train()

        while self.global_step < self.cfg.train.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.cfg.train.max_steps:
                    break

                loss = self._training_step(batch)

                if self.global_step % self.cfg.train.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Step %d/%d | Loss: %.4f | LR: %.2e",
                        self.global_step,
                        self.cfg.train.max_steps,
                        loss,
                        lr,
                    )

                # Evaluation
                if self.global_step > 0 and self.global_step % self.cfg.train.eval_every == 0:
                    val_loss = self.validate()
                    self.model.train()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                        logger.info("New best model saved (val_loss=%.4f)", val_loss)

                # Checkpoint
                if self.global_step > 0 and self.global_step % self.cfg.train.save_every == 0:
                    self._save_checkpoint(f"step-{self.global_step}")

        # Final save
        self._save_checkpoint("final")
        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one training step with optional gradient accumulation."""
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)

        accumulation_steps = self.cfg.train.gradient_accumulation_steps

        with autocast("cuda", enabled=self.cfg.train.use_bf16, dtype=torch.bfloat16):
            outputs = self.model(images)
            loss = compute_total_loss(
                outputs,
                masks,
                loss_module=self.loss_module,
            )
            loss = loss / accumulation_steps

        self.scaler.scale(loss).backward()

        if (self.global_step + 1) % accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        if self.cfg.wandb.enabled:
            import wandb

            wandb.log({"train/loss": loss.item() * accumulation_steps, "step": self.global_step})

        self.global_step += 1
        return loss.item() * accumulation_steps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation on the held-out set."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.cfg.train.use_bf16, dtype=torch.bfloat16):
                outputs = self.model(images)
                loss = compute_total_loss(
                    outputs,
                    masks,
                    loss_module=self.loss_module,
                )

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples

        if self.cfg.wandb.enabled:
            import wandb

            wandb.log({"val/loss": avg_loss, "step": self.global_step})

        logger.info("Validation loss: %.4f", avg_loss)
        return avg_loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        """Save a checkpoint with model weights, optimizer state, and metadata."""
        path = self.output_dir / f"checkpoint-{tag}.pt"
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)
