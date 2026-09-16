"""Main training loop with checkpointing, early stopping, and logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from framework.contracts.task import Task
from framework.contracts.types import Batch, MetricResults
from framework.core.checkpoint import CheckpointManager
from framework.core.device import build_grad_scaler, move_batch_to_device, resolve_device
from framework.core.distributed import DistributedContext, average_metric_dict
from framework.core.evaluator import Evaluator
from framework.core.loss_components import (
    add_averaged_loss_components,
    ordered_loss_component_keys,
    update_loss_component_sums,
)
from framework.core.seed import set_global_seed
from framework.logging.structured_logger import ExperimentLogger, NullExperimentLogger
from framework.metrics.collection import MetricCollection


@dataclass(frozen=True)
class TrainerSettings:
    """Training hyperparameters."""

    epochs: int = 100
    patience: int = 10
    device: str = "auto"
    amp: bool = False
    grad_clip_norm: float | None = None
    seed: int | None = None
    lr_scheduler: bool = False
    lr_scheduler_type: str = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    lr_scheduler_power: float = 0.9


class Trainer:
    """Main training runner."""

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        train_metrics: MetricCollection,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        settings: TrainerSettings,
        checkpoint_manager: CheckpointManager,
        logger: ExperimentLogger | None = None,
        run_dir: Path | None = None,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.settings = settings
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or NullExperimentLogger()
        self.run_dir = run_dir or Path(".")

        self.distributed_context = distributed_context
        self.device = distributed_context.device if distributed_context is not None else resolve_device(settings.device)
        self.is_main_process = distributed_context.is_main_process if distributed_context is not None else True
        if not self.is_main_process:
            self.logger = NullExperimentLogger()

        self.model.to(self.device)
        if distributed_context is not None and distributed_context.enabled:
            if self.device.type == "cuda":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )
            else:
                self.model = DistributedDataParallel(self.model)
        self.scaler = build_grad_scaler(settings.amp)

        self._scheduler = None
        if settings.lr_scheduler:
            if settings.lr_scheduler_type == "poly":
                initial_lr = optimizer.param_groups[0]["lr"]
                max_epoch = settings.epochs
                power = settings.lr_scheduler_power

                def poly_lr_lambda(epoch: int) -> float:
                    return (1 - (epoch - 1) / max_epoch) ** power

                self._scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=poly_lr_lambda
                )
            else:
                monitor_key = checkpoint_manager.monitor
                mode = checkpoint_manager.mode
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=mode,
                    factor=settings.lr_scheduler_factor,
                    patience=settings.lr_scheduler_patience,
                )

        self._best_val_metrics: MetricResults = {}
        self._start_epoch = 1

    def _print(self, message: str) -> None:
        if self.is_main_process:
            print(message)

    def _model_state_dict(self) -> dict[str, object]:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_model_state_dict(self, state_dict: dict[str, object]) -> None:
        target = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        target.load_state_dict(state_dict)

    def _run_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.train_metrics.reset()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        total_batches = len(self.train_loader)
        component_totals: dict[str, torch.Tensor] = {}
        component_counts: dict[str, int] = {}
        log_every = max(1, total_batches // 10)

        for batch in self.train_loader:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            device_type = str(self.device).split(":")[0]
            with autocast(device_type=device_type, enabled=self.settings.amp):
                outputs = self.model(batch)
                loss = self.task.compute_loss(outputs, batch)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.settings.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.settings.grad_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.settings.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.settings.grad_clip_norm,
                    )
                self.optimizer.step()

            update_loss_component_sums(self.task, component_totals, component_counts)
            preds = self.task.postprocess_outputs(outputs)
            targets = self.task.extract_targets(batch)
            self.train_metrics.update(preds, targets)

            total_loss += loss.detach()
            num_batches += 1

            if num_batches % log_every == 0:
                self._print(
                    f"  Epoch {epoch:3d} [{num_batches:4d}/{total_batches:4d}] "
                    f"loss={loss.detach().item():.4f}"
                )

        avg_loss = (total_loss / max(num_batches, 1)).item()
        metrics = self.train_metrics.compute()
        metrics["loss"] = avg_loss
        add_averaged_loss_components(metrics, component_totals, component_counts)
        if self.distributed_context is not None:
            metrics = average_metric_dict(metrics, self.distributed_context)
        return metrics

    def train(self) -> MetricResults:
        """Run full training with validation and early stopping."""
        if self.settings.seed is not None:
            set_global_seed(self.settings.seed)

        evaluator = Evaluator(
            self.model,
            self.task,
            self.val_metrics,
            self.device,
            self.settings.amp,
        )

        # ── CUDA warmup: force kernel compilation before the first epoch ──
        self._print("Warming up CUDA kernels (first forward+backward pass)...")
        device_type = str(self.device).split(":")[0]
        warmup_batch = next(iter(self.train_loader))
        warmup_batch = move_batch_to_device(warmup_batch, self.device)
        self.optimizer.zero_grad()
        with autocast(device_type=device_type, enabled=self.settings.amp):
            warmup_out = self.model(warmup_batch)
            warmup_loss = self.task.compute_loss(warmup_out, warmup_batch)
        if self.scaler is not None:
            self.scaler.scale(warmup_loss).backward()
        else:
            warmup_loss.backward()
        self.optimizer.zero_grad()
        if device_type == "cuda":
            torch.cuda.synchronize(self.device)
        self._print("CUDA warmup complete, starting training...")

        epochs_no_improve = 0
        best_monitor_value = None

        metrics_log_path = self.run_dir / "metrics.jsonl"

        for epoch in range(self._start_epoch, self.settings.epochs + 1):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics = self._run_epoch(epoch)
            val_loss, val_metrics = evaluator.evaluate(self.val_loader)
            if self.distributed_context is not None:
                val_metrics = average_metric_dict(val_metrics, self.distributed_context)

            # Log metrics
            self.logger.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, epoch)
            self.logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, epoch)

            # Console output
            console_parts = [
                f"Epoch {epoch}/{self.settings.epochs} | "
                f"train_loss={train_metrics['loss']:.4f}",
                f"val_loss={val_loss:.4f}",
            ]
            loss_component_keys = ordered_loss_component_keys({**train_metrics, **val_metrics})
            for key in loss_component_keys:
                if key in train_metrics:
                    console_parts.append(f"train_{key}={train_metrics[key]:.4f}")
                if key in val_metrics:
                    console_parts.append(f"val_{key}={val_metrics[key]:.4f}")
            self._print(" | ".join(console_parts))

            # JSONL log
            if self.is_main_process:
                with open(metrics_log_path, "a") as f:
                    json.dump({"epoch": epoch, "train": train_metrics, "val": val_metrics}, f)
                    f.write("\n")

            # Checkpointing
            monitor_key = self.checkpoint_manager.monitor
            monitor_value = val_metrics.get(monitor_key)
            if monitor_value is None and monitor_key.startswith("val_"):
                monitor_value = val_metrics.get(monitor_key[4:])
            if monitor_value is not None:
                is_best = self.checkpoint_manager.is_improved(monitor_value)
                if is_best:
                    epochs_no_improve = 0
                    best_monitor_value = monitor_value
                    self._best_val_metrics = val_metrics
                else:
                    epochs_no_improve += 1
            else:
                is_best = False
                epochs_no_improve += 1

            if self.is_main_process:
                self.checkpoint_manager.save(
                    self._model_state_dict(),
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best,
                    monitor_value=monitor_value,
                )

            if self._scheduler is not None:
                if isinstance(self._scheduler, torch.optim.lr_scheduler.LambdaLR):
                    self._scheduler.step()
                elif monitor_value is not None:
                    self._scheduler.step(monitor_value)

            # Early stopping
            if epochs_no_improve >= self.settings.patience:
                self._print(f"Early stopping at epoch {epoch} (no improvement for {self.settings.patience} epochs)")
                break

        self._print(f"Training complete. Best val metric: {best_monitor_value}")
        return self._best_val_metrics

    def validate(self, checkpoint_path: str | None = None) -> MetricResults:
        """Run validation."""
        if checkpoint_path is not None:
            payload = self.checkpoint_manager.load(checkpoint_path)
            self._load_model_state_dict(payload["state_dict"])
        evaluator = Evaluator(
            self.model,
            self.task,
            self.val_metrics,
            self.device,
            self.settings.amp,
        )
        _, metrics = evaluator.evaluate(self.val_loader)
        if self.distributed_context is not None:
            metrics = average_metric_dict(metrics, self.distributed_context)
        self._print(f"Validation metrics: {metrics}")
        return metrics

    def test(self, checkpoint_path: str | None = None) -> MetricResults:
        """Run test evaluation."""
        if checkpoint_path is not None:
            payload = self.checkpoint_manager.load(checkpoint_path)
            self._load_model_state_dict(payload["state_dict"])
        evaluator = Evaluator(
            self.model,
            self.task,
            self.test_metrics,
            self.device,
            self.settings.amp,
        )
        _, metrics = evaluator.evaluate(self.test_loader)
        if self.distributed_context is not None:
            metrics = average_metric_dict(metrics, self.distributed_context)
        self._print(f"Test metrics: {metrics}")
        return metrics

    def fit(self) -> MetricResults:
        """Train then test with the best checkpoint."""
        self.train()
        return self.test(checkpoint_path=self.checkpoint_manager.best_path)
