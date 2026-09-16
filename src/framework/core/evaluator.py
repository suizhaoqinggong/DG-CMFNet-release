"""Evaluation loop for computing loss and metrics."""

from __future__ import annotations

import torch
from torch import nn

from framework.contracts.task import Task
from framework.contracts.types import Batch, MetricResults
from framework.core.device import move_batch_to_device
from framework.core.loss_components import add_averaged_loss_components, update_loss_component_sums
from framework.metrics.collection import MetricCollection


class Evaluator:
    """Evaluates a model on a dataset."""

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        metrics: MetricCollection,
        device: torch.device,
        amp_enabled: bool = False,
    ) -> None:
        self.model = model
        self.task = task
        self.metrics = metrics
        self.device = device
        self.amp_enabled = amp_enabled

    @torch.no_grad()
    def evaluate(self, data_loader: Any) -> tuple[float, MetricResults]:
        """Evaluate on a data loader. Returns (avg_loss, metrics)."""
        self.model.eval()
        self.metrics.reset()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        component_totals: dict[str, torch.Tensor] = {}
        component_counts: dict[str, int] = {}

        for batch in data_loader:
            batch = move_batch_to_device(batch, self.device)
            with torch.autocast(
                device_type=str(self.device).split(":")[0],
                enabled=self.amp_enabled,
            ):
                outputs = self.model(batch)
                loss = self.task.compute_loss(outputs, batch)

            update_loss_component_sums(self.task, component_totals, component_counts)
            preds = self.task.postprocess_outputs(outputs)
            targets = self.task.extract_targets(batch)
            self.metrics.update(preds, targets)

            total_loss += loss.detach()
            num_batches += 1

        avg_loss = (total_loss / max(num_batches, 1)).item()
        metrics = self.metrics.compute()
        metrics["loss"] = avg_loss
        add_averaged_loss_components(metrics, component_totals, component_counts)
        return avg_loss, metrics
