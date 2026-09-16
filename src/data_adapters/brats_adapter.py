"""BraTS data adapter for multi-modal brain tumor segmentation.

Supports three data sources:
1. Raw .nii.gz directories (online preprocessing)
2. Preprocessed .npz files from scripts/preprocess_brats.py (fast loading)
3. Uncompressed .npy directories (fast loading, no decompression overhead)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from framework.contracts.data import DataAdapter
from framework.contracts.types import Batch, Sample

from .brats_augmentation import Compose, default_training_augmentation, default_validation_crop

BRATS_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
BRATS_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
BRATS_SUBJECT_SUFFIX_RE = re.compile(r"-\d{3}$")


def brats_subject_id(case_id: str) -> str:
    """Return the subject-level id shared by BraTS -000/-001 case variants."""
    return BRATS_SUBJECT_SUFFIX_RE.sub("", case_id)


def split_case_ids_by_subject(
    case_ids: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split case ids by subject group so related variants cannot cross splits."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")

    subject_groups: dict[str, list[str]] = {}
    for case_id in sorted(case_ids):
        subject_groups.setdefault(brats_subject_id(case_id), []).append(case_id)

    subject_ids = sorted(subject_groups)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(subject_ids))
    rng.shuffle(indices)
    shuffled_subjects = [subject_ids[i] for i in indices]

    n_test = int(len(shuffled_subjects) * test_ratio)
    n_val = int(len(shuffled_subjects) * val_ratio)
    n_train = len(shuffled_subjects) - n_val - n_test

    def expand(subjects: list[str]) -> list[str]:
        return [case_id for subject_id in subjects for case_id in subject_groups[subject_id]]

    return (
        expand(shuffled_subjects[:n_train]),
        expand(shuffled_subjects[n_train : n_train + n_val]),
        expand(shuffled_subjects[n_train + n_val :]),
    )


def _read_case_ids_file(path: str | None) -> list[str] | None:
    if path is None:
        return None
    ids_path = Path(path).expanduser()
    case_ids: list[str] = []
    for line in ids_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            case_ids.append(stripped)
    return case_ids


def _find_duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)


def split_case_ids_with_fixed_lists(
    case_ids: list[str],
    *,
    train_ids_file: str | None = None,
    val_ids_file: str | None = None,
    test_ids_file: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split case ids using explicit split files; unspecified train uses the remainder."""
    split_files = {
        "train": train_ids_file,
        "val": val_ids_file,
        "test": test_ids_file,
    }
    available = set(case_ids)
    explicit: dict[str, list[str]] = {}

    for split_name, path in split_files.items():
        ids = _read_case_ids_file(path)
        if ids is None:
            continue
        duplicates = _find_duplicates(ids)
        if duplicates:
            raise ValueError(f"{split_name}_ids_file contains duplicate ids: {duplicates}")
        missing = sorted(set(ids) - available)
        if missing:
            raise ValueError(f"{split_name}_ids_file contains ids not found in root_dir: {missing}")
        explicit[split_name] = ids

    split_names = sorted(explicit)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sorted(set(explicit[left]) & set(explicit[right]))
            if overlap:
                raise ValueError(f"{left} and {right} split files overlap: {overlap}")

    reserved = set(explicit.get("val", [])) | set(explicit.get("test", []))
    train_cases = explicit.get("train", [case_id for case_id in case_ids if case_id not in reserved])
    return train_cases, explicit.get("val", []), explicit.get("test", [])


def load_nifti(path: str | Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def z_score_normalize(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std == 0:
        return volume
    normalized = (volume - mean) / std
    out = np.zeros_like(volume)
    out[mask] = normalized[mask]
    return out


def crop_to_nonzero(
    volume: np.ndarray,
    label: np.ndarray | None = None,
    margin: int = 8,
):
    coords = np.argwhere(volume.sum(axis=0) > 0)
    if len(coords) == 0:
        return (volume, label) if label is not None else volume

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    slices = []
    for i in range(3):
        start = max(0, min_coords[i] - margin)
        end = min(volume.shape[i + 1], max_coords[i] + margin + 1)
        slices.append(slice(start, end))

    cropped_volume = volume[:, slices[0], slices[1], slices[2]]
    if label is not None:
        cropped_label = label[slices[0], slices[1], slices[2]]
        return cropped_volume, cropped_label
    return cropped_volume


class BraTSDatasetRaw(Dataset):
    """Dataset loading raw .nii.gz files with online preprocessing."""

    def __init__(
        self,
        root_dir: str,
        case_ids: list[str],
        modalities: list[str] | None = None,
        num_classes: int = 4,
        crop_to_brain: bool = True,
        normalize: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.case_ids = case_ids
        self.modalities = modalities or BRATS_MODALITIES
        self.num_classes = num_classes
        self.crop_to_brain = crop_to_brain
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.case_ids)

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        case_dir = self.root_dir / case_id
        mod_arrays = []
        for mod in self.modalities:
            path = case_dir / f"{case_id}-{mod}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(f"Missing modality file: {path}")
            mod_arrays.append(load_nifti(path))

        volume = np.stack(mod_arrays, axis=0)

        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
        seg = load_nifti(seg_path).astype(np.int64)

        remapped_seg = np.zeros_like(seg)
        for old, new in BRATS_LABEL_REMAP.items():
            remapped_seg[seg == old] = new

        return volume, remapped_seg

    def _preprocess(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.normalize:
            normalized = np.zeros_like(volume)
            for m in range(volume.shape[0]):
                normalized[m] = z_score_normalize(volume[m])
            volume = normalized

        if self.crop_to_brain:
            mask = volume[0] > 0
            volume, seg = crop_to_nonzero(volume, seg)

        return volume, seg

    def __getitem__(self, idx: int) -> Sample:
        case_id = self.case_ids[idx]
        volume, seg = self._load_case(case_id)
        volume, seg = self._preprocess(volume, seg)

        volume_tensor = torch.from_numpy(volume.copy()).permute(0, 3, 1, 2).contiguous()
        seg_tensor = torch.from_numpy(seg.copy()).permute(2, 0, 1).contiguous().long()

        return Sample(
            signal=volume_tensor,
            label=seg_tensor,
            id=case_id,
            meta={"shape_original": volume.shape, "modalities": self.modalities},
        )


def _load_npz_sample(case_id: str, root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single npz sample. Top-level for multiprocessing pickling."""
    path = Path(root_dir) / f"{case_id}.npz"
    data = np.load(path)
    volume = np.stack(
        [data["t1"], data["t1ce"], data["t2"], data["flair"]], axis=0
    ).astype(np.float32)
    seg = data["seg"].astype(np.int64)
    return volume, seg


def _load_npy_sample(case_id: str, root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single uncompressed npy sample from a directory. Top-level for multiprocessing pickling."""
    case_dir = Path(root_dir) / case_id
    volume = np.stack(
        [
            np.load(case_dir / "t1.npy"),
            np.load(case_dir / "t1ce.npy"),
            np.load(case_dir / "t2.npy"),
            np.load(case_dir / "flair.npy"),
        ],
        axis=0,
    ).astype(np.float32)
    seg = np.load(case_dir / "seg.npy").astype(np.int64)
    return volume, seg


class BraTSDatasetPreprocessed(Dataset):
    """Dataset loading preprocessed .npz files with optional augmentation."""

    def __init__(
        self,
        root_dir: str,
        case_ids: list[str],
        num_classes: int = 4,
        augment: Compose | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.case_ids = case_ids
        self.num_classes = num_classes
        self.augment = augment

        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cache_in_memory:
            from multiprocessing import Pool
            from functools import partial

            print(f"Caching {len(case_ids)} samples in RAM (parallel)...")
            loader = partial(_load_npz_sample, root_dir=str(self.root_dir))
            with Pool(processes=os.cpu_count() // 2) as pool:
                results = pool.map(loader, case_ids)
            self._cache = results
            print(f"Cache complete. {len(case_ids)} samples loaded.")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Sample:
        case_id = self.case_ids[idx]

        if self._cache is not None:
            volume, seg = self._cache[idx]
        else:
            path = self.root_dir / f"{case_id}.npz"
            data = np.load(path)
            volume = np.stack(
                [data["t1"], data["t1ce"], data["t2"], data["flair"]], axis=0
            ).astype(np.float32)
            seg = data["seg"].astype(np.int64)

        if self.augment is not None:
            volume, seg = self.augment(volume, seg)

        volume_tensor = torch.from_numpy(volume.copy()).permute(0, 3, 1, 2).contiguous()
        seg_tensor = torch.from_numpy(seg.copy()).permute(2, 0, 1).contiguous().long()

        return Sample(
            signal=volume_tensor,
            label=seg_tensor,
            id=case_id,
            meta={"shape": volume.shape, "preprocessed": True},
        )


class BraTSDatasetUncompressed(Dataset):
    """Dataset loading uncompressed .npy files from directories with optional augmentation."""

    def __init__(
        self,
        root_dir: str,
        case_ids: list[str],
        num_classes: int = 4,
        augment: Compose | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.case_ids = case_ids
        self.num_classes = num_classes
        self.augment = augment

        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cache_in_memory:
            from multiprocessing import Pool
            from functools import partial

            print(f"Caching {len(case_ids)} uncompressed samples in RAM (parallel)...")
            loader = partial(_load_npy_sample, root_dir=str(self.root_dir))
            with Pool(processes=os.cpu_count() // 2) as pool:
                results = pool.map(loader, case_ids)
            self._cache = results
            print(f"Cache complete. {len(case_ids)} samples loaded.")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Sample:
        case_id = self.case_ids[idx]

        if self._cache is not None:
            volume, seg = self._cache[idx]
        else:
            volume, seg = _load_npy_sample(case_id, str(self.root_dir))

        if self.augment is not None:
            volume, seg = self.augment(volume, seg)

        volume_tensor = torch.from_numpy(volume.copy()).permute(0, 3, 1, 2).contiguous()
        seg_tensor = torch.from_numpy(seg.copy()).permute(2, 0, 1).contiguous().long()

        return Sample(
            signal=volume_tensor,
            label=seg_tensor,
            id=case_id,
            meta={"shape": volume.shape, "preprocessed": True},
        )


class BraTSSliceDataset(Dataset):
    """Select several axial 2D slices from one patient volume for training.

    The wrapped dataset keeps patient-level split assignment and applies its
    3D augmentation before slices are selected. A volume is loaded once per
    item, avoiding a complete disk read per individual slice.
    """

    def __init__(
        self,
        volume_dataset: Dataset,
        slices_per_case: int = 16,
        foreground_slice_ratio: float = 0.5,
    ) -> None:
        if slices_per_case < 1:
            raise ValueError("slices_per_case must be positive")
        if not 0.0 <= foreground_slice_ratio <= 1.0:
            raise ValueError("foreground_slice_ratio must be in [0, 1]")
        self.volume_dataset = volume_dataset
        self.slices_per_case = int(slices_per_case)
        self.foreground_slice_ratio = float(foreground_slice_ratio)

    def __len__(self) -> int:
        return len(self.volume_dataset)

    def __getitem__(self, idx: int) -> Sample:
        volume_sample = self.volume_dataset[idx]
        volume = volume_sample["signal"]
        label = volume_sample["label"]
        if volume.ndim != 4 or label.ndim != 3:
            raise ValueError("Expected [C, D, H, W] signals and [D, H, W] labels")

        depth = label.shape[0]
        foreground_depths = torch.nonzero((label != 0).any(dim=(1, 2)), as_tuple=False).flatten()
        num_foreground = int(round(self.slices_per_case * self.foreground_slice_ratio))
        selected: list[torch.Tensor] = []
        if num_foreground and foreground_depths.numel():
            selected.append(foreground_depths[torch.randint(foreground_depths.numel(), (num_foreground,))])
        num_uniform = self.slices_per_case - sum(item.numel() for item in selected)
        if num_uniform:
            selected.append(torch.randint(depth, (num_uniform,)))
        slice_indices = torch.cat(selected).long()

        return Sample(
            signal=volume[:, slice_indices].permute(1, 0, 2, 3).contiguous(),
            label=label[slice_indices].contiguous(),
            id=volume_sample["id"],
            meta={
                **volume_sample["meta"],
                "slice_wise_2d": True,
                "slice_indices": slice_indices.tolist(),
            },
        )


def _pad_to_max(signals, labels):
    max_d = max(s.shape[1] for s in signals)
    max_h = max(s.shape[2] for s in signals)
    max_w = max(s.shape[3] for s in signals)

    padded_signals = []
    padded_labels = []
    for sig, lbl in zip(signals, labels):
        _, d, h, w = sig.shape
        pad_d, pad_h, pad_w = max_d - d, max_h - h, max_w - w
        padded_sig = torch.nn.functional.pad(sig, (0, pad_w, 0, pad_h, 0, pad_d), value=0)
        padded_lbl = torch.nn.functional.pad(lbl, (0, pad_w, 0, pad_h, 0, pad_d), value=0)
        padded_signals.append(padded_sig)
        padded_labels.append(padded_lbl)

    return padded_signals, padded_labels


class BraTSDataAdapter(DataAdapter):
    """Data adapter for BraTS with raw, preprocessed, and uncompressed data sources."""

    def __init__(
        self,
        root_dir: str = "data/brats2023",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        num_classes: int = 4,
        crop_to_brain: bool = False,
        normalize: bool = True,
        use_preprocessed: bool = False,
        preprocessed_dir: str | None = None,
        augment_train: bool = False,
        crop_size: tuple[int, int, int] = (128, 128, 128),
        cache_in_memory: bool = False,
        data_format: str = "npz",  # "npz" | "npy" | "raw"
        train_ids_file: str | None = None,
        val_ids_file: str | None = None,
        test_ids_file: str | None = None,
        slice_wise_2d: bool = False,
        slices_per_case: int = 16,
        foreground_slice_ratio: float = 0.5,
    ) -> None:
        self.root_dir = root_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.num_classes = num_classes
        self.crop_to_brain = crop_to_brain
        self.normalize = normalize
        self.use_preprocessed = use_preprocessed
        self.preprocessed_dir = preprocessed_dir
        self.augment_train = augment_train
        self.crop_size = crop_size
        self.cache_in_memory = cache_in_memory
        self.data_format = data_format
        self.train_ids_file = train_ids_file
        self.val_ids_file = val_ids_file
        self.test_ids_file = test_ids_file
        self.slice_wise_2d = slice_wise_2d
        self.slices_per_case = slices_per_case
        self.foreground_slice_ratio = foreground_slice_ratio
        self._datasets: tuple[Dataset, Dataset, Dataset] | None = None

    def prepare(self) -> None:
        root = Path(self.root_dir)
        if not root.exists():
            raise FileNotFoundError(f"BraTS data directory not found: {root}")

        case_ids = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if not case_ids:
            raise ValueError(f"No case directories found in {root}")

        if self.train_ids_file or self.val_ids_file or self.test_ids_file:
            train_cases, val_cases, test_cases = split_case_ids_with_fixed_lists(
                case_ids=case_ids,
                train_ids_file=self.train_ids_file,
                val_ids_file=self.val_ids_file,
                test_ids_file=self.test_ids_file,
            )
        else:
            train_cases, val_cases, test_cases = split_case_ids_by_subject(
                case_ids=case_ids,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )

        if self.data_format == "raw" or (not self.use_preprocessed and self.data_format not in ("npz", "npy")):
            common_kwargs = {
                "root_dir": str(root),
                "num_classes": self.num_classes,
                "crop_to_brain": self.crop_to_brain,
                "normalize": self.normalize,
            }
            self._datasets = (
                BraTSDatasetRaw(case_ids=train_cases, **common_kwargs),
                BraTSDatasetRaw(case_ids=val_cases, **common_kwargs),
                BraTSDatasetRaw(case_ids=test_cases, **common_kwargs),
            )
        else:
            pp_dir = self.preprocessed_dir or self.root_dir
            train_aug = default_training_augmentation(self.crop_size) if self.augment_train else None
            val_aug = default_validation_crop(self.crop_size)

            dataset_cls = BraTSDatasetUncompressed if self.data_format == "npy" else BraTSDatasetPreprocessed

            self._datasets = (
                dataset_cls(
                    root_dir=pp_dir,
                    case_ids=train_cases,
                    num_classes=self.num_classes,
                    augment=train_aug,
                    cache_in_memory=self.cache_in_memory,
                ),
                dataset_cls(
                    root_dir=pp_dir,
                    case_ids=val_cases,
                    num_classes=self.num_classes,
                    augment=val_aug,
                    cache_in_memory=self.cache_in_memory,
                ),
                dataset_cls(
                    root_dir=pp_dir,
                    case_ids=test_cases,
                    num_classes=self.num_classes,
                    augment=val_aug,
                    cache_in_memory=self.cache_in_memory,
                ),
            )

        if self.slice_wise_2d:
            train_dataset, val_dataset, test_dataset = self._datasets
            self._datasets = (
                BraTSSliceDataset(
                    train_dataset,
                    slices_per_case=self.slices_per_case,
                    foreground_slice_ratio=self.foreground_slice_ratio,
                ),
                val_dataset,
                test_dataset,
            )

    def get_splits(self) -> tuple[Dataset, Dataset, Dataset]:
        if self._datasets is None:
            raise RuntimeError("Data adapter not prepared. Call prepare() first.")
        return self._datasets

    def collate_fn(self, batch: list[Sample]) -> Batch:
        if batch[0]["meta"].get("slice_wise_2d", False):
            signals = torch.cat([sample["signal"] for sample in batch], dim=0)
            labels = torch.cat([sample["label"] for sample in batch], dim=0)
            ids = [sample["id"] for sample in batch for _ in range(sample["signal"].shape[0])]
            return Batch(signal=signals, label=labels, id=ids, meta=[sample["meta"] for sample in batch])

        signals = [s["signal"] for s in batch]
        labels = [s["label"] for s in batch]
        signals, labels = _pad_to_max(signals, labels)

        return Batch(
            signal=torch.stack(signals),
            label=torch.stack(labels),
            id=[s["id"] for s in batch],
            meta=[s["meta"] for s in batch],
        )


class BraTS2DDataAdapter(BraTSDataAdapter):
    """BraTS adapter with 2D training slices and whole-volume evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["slice_wise_2d"] = True
        super().__init__(*args, **kwargs)
