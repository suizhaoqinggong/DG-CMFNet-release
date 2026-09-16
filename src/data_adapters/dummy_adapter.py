"""Dummy data adapter for testing and demonstration."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from framework.contracts.data import DataAdapter
from framework.contracts.types import Batch, Sample


class DummyDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 5,
        input_dim: int = 128,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_dim = input_dim
        rng = torch.Generator().manual_seed(seed)
        self.signals = torch.randn(num_samples, input_dim, generator=rng)
        self.labels = torch.randint(0, 2, (num_samples, num_classes), generator=rng).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            signal=self.signals[idx],
            label=self.labels[idx],
            id=f"sample_{idx}",
            meta={"index": idx},
        )


class DummyDataAdapter(DataAdapter):
    """Data adapter that serves synthetic data."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 5,
        input_dim: int = 128,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self._datasets: tuple[DummyDataset, DummyDataset, DummyDataset] | None = None

    def prepare(self) -> None:
        """Prepare synthetic datasets."""
        rng = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(self.num_samples, generator=rng).tolist()

        n_test = int(self.num_samples * self.test_ratio)
        n_val = int(self.num_samples * self.val_ratio)
        n_train = self.num_samples - n_val - n_test

        train_idx = set(indices[:n_train])
        val_idx = set(indices[n_train : n_train + n_val])
        test_idx = set(indices[n_train + n_val :])

        # Use deterministic seeds for each split so data is stable
        self._datasets = (
            DummyDataset(n_train, self.num_classes, self.input_dim, seed=self.seed + 1),
            DummyDataset(n_val, self.num_classes, self.input_dim, seed=self.seed + 2),
            DummyDataset(n_test, self.num_classes, self.input_dim, seed=self.seed + 3),
        )

    def get_splits(self) -> tuple[DummyDataset, DummyDataset, DummyDataset]:
        if self._datasets is None:
            raise RuntimeError("Data adapter not prepared. Call prepare() first.")
        return self._datasets

    def collate_fn(self, batch: list[Sample]) -> Batch:
        return Batch(
            signal=torch.stack([s["signal"] for s in batch]),
            label=torch.stack([s["label"] for s in batch]),
            id=[s["id"] for s in batch],
            meta=[s["meta"] for s in batch],
        )
