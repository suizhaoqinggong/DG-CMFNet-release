"""DataAdapter protocol definition."""

from typing import Protocol

from torch.utils.data import Dataset

from .types import Batch, Sample

DatasetLike = Dataset


class DataAdapter(Protocol):
    """Protocol for data adapters that prepare and serve datasets."""

    def prepare(self) -> None:
        """Prepare data: download, preprocess, split, etc."""
        ...

    def get_splits(self) -> tuple[DatasetLike, DatasetLike, DatasetLike]:
        """Return (train_dataset, val_dataset, test_dataset)."""
        ...

    def collate_fn(self, batch: list[Sample]) -> Batch:
        """Collate a list of samples into a batch."""
        ...
