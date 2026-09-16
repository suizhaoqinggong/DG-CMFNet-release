"""Checkpoint save/load management."""

from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    """Manages model checkpointing with metric-based best selection."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "loss",
        mode: str = "min",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self._best_value: float | None = None
        self._best_path: Path | None = None

    def is_improved(self, value: float) -> bool:
        """Check if the given metric value is better than the current best."""
        if self._best_value is None:
            return True
        if self.mode == "max":
            return value > self._best_value
        return value < self._best_value

    def save(
        self,
        state_dict: dict[str, object],
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        monitor_value: float | None = None,
    ) -> None:
        """Save a checkpoint."""
        payload = {
            "epoch": epoch,
            "state_dict": state_dict,
            "metrics": metrics,
        }
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(payload, last_path)

        if is_best:
            best_value = monitor_value if monitor_value is not None else metrics.get(self.monitor)
            if best_value is not None:
                self._best_value = best_value
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(payload, best_path)
            self._best_path = best_path

    def load(self, path: str | Path | None = None) -> dict[str, object]:
        """Load a checkpoint. Returns the payload dict."""
        if path is None:
            path = self.checkpoint_dir / "best.pt"
        else:
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    @property
    def best_path(self) -> Path | None:
        return self._best_path
