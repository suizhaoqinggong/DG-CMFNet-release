#!/usr/bin/env python3
"""Full-volume validation for BraTS segmentation runs.

This script evaluates an existing run checkpoint on complete validation volumes
using sliding-window inference, then reports WT/TC/ET Dice and HD95.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from framework.core.device import resolve_device
from framework.registry.defaults import create_default_registries
from framework.registry.factories import build_component_bundle, load_toml


MODALITY_KEYS = ("t1", "t1ce", "t2", "flair")
BRATS_REGIONS: dict[str, tuple[int, ...]] = {
    "wt": (1, 2, 3),
    "tc": (1, 3),
    "et": (3,),
}
BRATS_EMPTY_HD95_PENALTY = 373.13
DEFAULT_ET_EMPTY_PRED_THRESHOLD = 50
MC_SAMPLE_METRICS_CHOICES = ("none", "dice", "full")
_DROPOUT_TYPES = tuple(
    cls
    for cls in (
        getattr(torch.nn, "Dropout", None),
        getattr(torch.nn, "Dropout1d", None),
        getattr(torch.nn, "Dropout2d", None),
        getattr(torch.nn, "Dropout3d", None),
        getattr(torch.nn, "AlphaDropout", None),
        getattr(torch.nn, "FeatureAlphaDropout", None),
    )
    if cls is not None
)


@dataclass
class MCDropoutPrediction:
    """Aggregated MC-dropout probabilities and paper-aligned uncertainty maps."""

    pred: torch.Tensor
    mean_probs: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    sample_count: int

    @property
    def probability_variance(self) -> torch.Tensor:
        """Backward-compatible alias for epistemic uncertainty."""
        return self.epistemic_uncertainty

    @property
    def predictive_entropy(self) -> torch.Tensor:
        """Backward-compatible alias for aleatoric uncertainty."""
        return self.aleatoric_uncertainty


def sliding_window_starts(length: int, window: int, overlap: float) -> list[int]:
    """Return start indices that cover an axis exactly once or more."""
    if length <= 0:
        raise ValueError("length must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= window:
        return [0]

    step = max(1, int(window * (1.0 - overlap)))
    last_start = length - window
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _pad_to_window(volume: torch.Tensor, window_size: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
    _, depth, height, width = volume.shape
    pad_d = max(0, window_size[0] - depth)
    pad_h = max(0, window_size[1] - height)
    pad_w = max(0, window_size[2] - width)
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return volume, (depth, height, width)
    padded = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), value=0.0)
    return padded, (depth, height, width)


def enable_mc_dropout(model: torch.nn.Module) -> int:
    """Put model in eval mode, then enable only dropout-bearing stochastic modules."""
    model.eval()
    enabled = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
            enabled += 1
        elif isinstance(module, torch.nn.MultiheadAttention) and float(module.dropout) > 0.0:
            module.train()
            enabled += 1
    return enabled


def set_dropout_probability(model: torch.nn.Module, dropout_p: float) -> int:
    """Override dropout probability for MC sensitivity checks or MC-trained models."""
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError("dropout_p must be in [0, 1)")

    changed = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.p = float(dropout_p)
            changed += 1
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = float(dropout_p)
            changed += 1
    return changed


def count_active_mc_dropout_modules(model: torch.nn.Module) -> int:
    """Count modules that can create stochastic MC-dropout predictions."""
    count = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES) and float(module.p) > 0.0:
            count += 1
        elif isinstance(module, torch.nn.MultiheadAttention) and float(module.dropout) > 0.0:
            count += 1
    return count


@torch.no_grad()
def sliding_window_predict_proba(
    model: torch.nn.Module,
    volume: torch.Tensor,
    *,
    device: torch.device,
    num_classes: int,
    window_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    amp: bool = False,
    mc_dropout: bool = False,
    window_batch_size: int = 1,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predict complete class probabilities from a [M, D, H, W] input tensor."""
    if volume.ndim != 4:
        raise ValueError(f"volume must have shape [M, D, H, W], got {tuple(volume.shape)}")
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")

    was_training = model.training
    model.eval()
    if mc_dropout:
        enable_mc_dropout(model)

    padded, original_shape = _pad_to_window(volume.float(), window_size)
    padded = padded.to(device, non_blocking=True)
    _, depth, height, width = padded.shape
    wd, wh, ww = window_size

    prob_accum = torch.zeros(
        (num_classes, depth, height, width),
        dtype=torch.float32,
        device=device,
    )
    count_accum = torch.zeros(
        (1, depth, height, width),
        dtype=torch.float32,
        device=device,
    )

    d_starts = sliding_window_starts(depth, wd, overlap)
    h_starts = sliding_window_starts(height, wh, overlap)
    w_starts = sliding_window_starts(width, ww, overlap)
    window_coords = [
        (d0, h0, w0)
        for d0 in d_starts
        for h0 in h_starts
        for w0 in w_starts
    ]
    device_type = device.type

    for batch_start in range(0, len(window_coords), window_batch_size):
        batch_coords = window_coords[batch_start : batch_start + window_batch_size]
        patches = [
            padded[:, d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww]
            for d0, h0, w0 in batch_coords
        ]
        patch_batch = torch.stack(patches, dim=0)
        batch = {
            "signal": patch_batch,
            "label": torch.empty(0, device=device),
            "id": ["full-volume-window"] * len(batch_coords),
            "meta": [],
        }
        with torch.amp.autocast(device_type=device_type, enabled=amp):
            logits = model(batch)
            if postprocess_fn is None:
                batch_probs = torch.softmax(logits, dim=1)
            else:
                batch_probs = postprocess_fn(logits)
            batch_probs = batch_probs.detach().float()

        for (d0, h0, w0), probs in zip(batch_coords, batch_probs):
            prob_accum[:, d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww] += probs
            count_accum[:, d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww] += 1.0

    probs = prob_accum / count_accum.clamp_min(1.0)
    od, oh, ow = original_shape
    probs = probs[:, :od, :oh, :ow].contiguous().cpu()

    if was_training:
        model.train()
    else:
        model.eval()
    return probs


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: torch.Tensor,
    *,
    device: torch.device,
    num_classes: int,
    window_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    amp: bool = False,
    window_batch_size: int = 1,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predict a complete label volume from a [M, D, H, W] input tensor."""
    probs = sliding_window_predict_proba(
        model,
        volume,
        device=device,
        num_classes=num_classes,
        window_size=window_size,
        overlap=overlap,
        amp=amp,
        mc_dropout=False,
        window_batch_size=window_batch_size,
        postprocess_fn=postprocess_fn,
    )
    return torch.argmax(probs, dim=0).long()


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    volume: torch.Tensor,
    *,
    device: torch.device,
    num_classes: int,
    window_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    amp: bool = False,
    mc_samples: int = 20,
    mc_seed: int | None = None,
    sample_callback: Callable[[int, torch.Tensor], None] | None = None,
    window_batch_size: int = 1,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> MCDropoutPrediction:
    """Run repeated MC-dropout sliding-window predictions and aggregate uncertainty."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")

    prob_sum: torch.Tensor | None = None
    prob_sq_sum: torch.Tensor | None = None
    entropy_sum: torch.Tensor | None = None
    for sample_index in range(mc_samples):
        if mc_seed is not None:
            seed = int(mc_seed) + sample_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        probs = sliding_window_predict_proba(
            model,
            volume,
            device=device,
            num_classes=num_classes,
            window_size=window_size,
            overlap=overlap,
            amp=amp,
            mc_dropout=True,
            window_batch_size=window_batch_size,
            postprocess_fn=postprocess_fn,
        )
        if sample_callback is not None:
            sample_callback(sample_index, torch.argmax(probs, dim=0).long())

        if prob_sum is None:
            prob_sum = torch.zeros_like(probs)
            prob_sq_sum = torch.zeros_like(probs)
            entropy_sum = torch.zeros_like(probs[0])
        prob_sum += probs
        prob_sq_sum += probs.square()
        safe_sample_probs = probs.clamp_min(1e-8)
        assert entropy_sum is not None
        entropy_sum += (-(safe_sample_probs * safe_sample_probs.log()).sum(dim=0)).contiguous()

    assert prob_sum is not None and prob_sq_sum is not None and entropy_sum is not None
    mean_probs = (prob_sum / float(mc_samples)).contiguous()
    class_variance = (prob_sq_sum / float(mc_samples) - mean_probs.square()).clamp_min(0.0)
    epistemic_uncertainty = class_variance.mean(dim=0).contiguous()
    aleatoric_uncertainty = (entropy_sum / float(mc_samples)).contiguous()
    pred = torch.argmax(mean_probs, dim=0).long()

    return MCDropoutPrediction(
        pred=pred,
        mean_probs=mean_probs,
        epistemic_uncertainty=epistemic_uncertainty,
        aleatoric_uncertainty=aleatoric_uncertainty,
        sample_count=mc_samples,
    )


def _region_mask(labels: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    return np.isin(labels, classes)


def _dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_sum = int(pred_mask.sum())
    target_sum = int(target_mask.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    intersection = int(np.logical_and(pred_mask, target_mask).sum())
    return float(2.0 * intersection / (pred_sum + target_sum))


def _apply_empty_et_threshold(
    region_name: str,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    et_empty_pred_threshold: int | None,
) -> np.ndarray:
    if region_name != "et" or et_empty_pred_threshold is None:
        return pred_mask
    if bool(target_mask.any()):
        return pred_mask
    pred_sum = int(pred_mask.sum())
    if 0 < pred_sum <= et_empty_pred_threshold:
        return np.zeros_like(pred_mask, dtype=bool)
    return pred_mask


def compute_region_dice_metrics(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    et_empty_pred_threshold: int | None = DEFAULT_ET_EMPTY_PRED_THRESHOLD,
) -> dict[str, float]:
    """Compute only BraTS WT/TC/ET Dice for one complete volume."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = pred.astype(np.int64, copy=False)
    target = target.astype(np.int64, copy=False)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")

    metrics: dict[str, float] = {}
    dice_values = []
    for name, classes in BRATS_REGIONS.items():
        pred_mask = _region_mask(pred, classes)
        target_mask = _region_mask(target, classes)
        pred_mask = _apply_empty_et_threshold(name, pred_mask, target_mask, et_empty_pred_threshold)
        dice = _dice(pred_mask, target_mask)
        metrics[f"{name}_dice"] = dice
        dice_values.append(dice)
    metrics["mean_dice"] = float(np.mean(dice_values))
    return metrics


def _surface(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return mask.astype(bool)
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask.astype(bool) & ~eroded


def _surface_distances(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: tuple[float, float, float],
    empty_hd95_penalty: float = BRATS_EMPTY_HD95_PENALTY,
) -> np.ndarray:
    pred_any = bool(pred_mask.any())
    target_any = bool(target_mask.any())
    if not pred_any and not target_any:
        return np.array([0.0], dtype=np.float64)
    if pred_any != target_any:
        return np.array([float(empty_hd95_penalty)], dtype=np.float64)

    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)
    pred_to_target = distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
    target_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
    return np.concatenate([pred_to_target, target_to_pred]).astype(np.float64)


def _hd95(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: tuple[float, float, float],
    empty_hd95_penalty: float = BRATS_EMPTY_HD95_PENALTY,
) -> float:
    distances = _surface_distances(pred_mask, target_mask, spacing, empty_hd95_penalty=empty_hd95_penalty)
    return float(np.percentile(distances, 95))


def compute_region_metrics(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    empty_hd95_penalty: float = BRATS_EMPTY_HD95_PENALTY,
    et_empty_pred_threshold: int | None = DEFAULT_ET_EMPTY_PRED_THRESHOLD,
) -> dict[str, float]:
    """Compute BraTS WT/TC/ET Dice and HD95 for one complete volume."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = pred.astype(np.int64, copy=False)
    target = target.astype(np.int64, copy=False)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")

    metrics: dict[str, float] = {}
    dice_values = []
    hd95_values = []
    for name, classes in BRATS_REGIONS.items():
        pred_mask = _region_mask(pred, classes)
        target_mask = _region_mask(target, classes)
        pred_mask = _apply_empty_et_threshold(name, pred_mask, target_mask, et_empty_pred_threshold)
        dice = _dice(pred_mask, target_mask)
        hd95 = _hd95(pred_mask, target_mask, spacing, empty_hd95_penalty=empty_hd95_penalty)
        metrics[f"{name}_dice"] = dice
        metrics[f"{name}_hd95"] = hd95
        dice_values.append(dice)
        hd95_values.append(hd95)

    metrics["mean_dice"] = float(np.mean(dice_values))
    metrics["mean_hd95"] = float(np.mean(hd95_values))
    return metrics


def collect_sample_metrics(
    sample_metrics: list[dict[str, float]],
    mode: str,
    target: np.ndarray | torch.Tensor,
    spacing: tuple[float, float, float],
    _: int,
    sample_pred: torch.Tensor,
    *,
    empty_hd95_penalty: float = BRATS_EMPTY_HD95_PENALTY,
    et_empty_pred_threshold: int | None = DEFAULT_ET_EMPTY_PRED_THRESHOLD,
) -> None:
    """Append optional per-MC-sample metrics without forcing expensive HD95."""
    if mode == "dice":
        sample_metrics.append(
            compute_region_dice_metrics(
                sample_pred,
                target,
                et_empty_pred_threshold=et_empty_pred_threshold,
            )
        )
        return
    if mode == "full":
        sample_metrics.append(
            compute_region_metrics(
                sample_pred,
                target,
                spacing=spacing,
                empty_hd95_penalty=empty_hd95_penalty,
                et_empty_pred_threshold=et_empty_pred_threshold,
            )
        )
        return
    raise ValueError(f"Unsupported MC sample metrics mode: {mode}")


def load_preprocessed_case(
    preprocessed_dir: Path,
    case_id: str,
    *,
    data_format: str = "npz",
) -> tuple[torch.Tensor, np.ndarray, tuple[float, float, float]]:
    """Load one BraTS preprocessed case in the same axis order used by training."""
    if data_format == "npy":
        case_dir = preprocessed_dir / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Missing preprocessed case directory: {case_dir}")
        volume = np.stack([np.load(case_dir / f"{key}.npy") for key in MODALITY_KEYS], axis=0).astype(np.float32)
        target = np.load(case_dir / "seg.npy").astype(np.int64)
        spacing_path = case_dir / "target_spacing.npy"
        spacing_raw = np.load(spacing_path).astype(np.float32) if spacing_path.exists() else np.ones(3, dtype=np.float32)
    else:
        path = preprocessed_dir / f"{case_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed case: {path}")

        with np.load(path) as data:
            volume = np.stack([data[key] for key in MODALITY_KEYS], axis=0).astype(np.float32)
            target = data["seg"].astype(np.int64)
            spacing_raw = data["target_spacing"].astype(np.float32) if "target_spacing" in data else np.ones(3, dtype=np.float32)

    # BraTSDatasetPreprocessed trains in [M, W, D, H] / [W, D, H] order.
    volume_tensor = torch.from_numpy(volume.copy()).permute(0, 3, 1, 2).contiguous()
    target_model_order = np.transpose(target, (2, 0, 1)).copy()
    spacing = (float(spacing_raw[2]), float(spacing_raw[0]), float(spacing_raw[1]))
    return volume_tensor, target_model_order, spacing


def _split_ids_from_run(run_dir: Path, split: str) -> list[str] | None:
    path = run_dir / "splits" / f"{split}_ids.txt"
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_split_ids(bundle: Any, run_dir: Path, split: str) -> list[str]:
    """Resolve split ids from run snapshots, falling back to the current adapter split."""
    split_ids = _split_ids_from_run(run_dir, split)
    if split_ids is not None:
        return split_ids

    datasets = dict(zip(("train", "val", "test"), bundle.data_adapter.get_splits()))
    dataset = datasets[split]
    case_ids = getattr(dataset, "case_ids", None)
    if case_ids is None:
        raise ValueError(f"Dataset for split '{split}' does not expose case_ids")
    return list(case_ids)


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_keys = [f"{name}_{metric}" for name in BRATS_REGIONS for metric in ("dice", "hd95")]
    metric_keys.extend(["mean_dice", "mean_hd95"])
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "case_id" and isinstance(value, (int, float, np.floating))
        }
    )
    metric_keys = list(dict.fromkeys([*metric_keys, *numeric_keys]))
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = np.array([float(row[key]) for row in rows if key in row], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"{key}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
    return summary


def summarize_mc_sample_metrics(sample_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Summarize per-sample MC metrics for stability reporting."""
    if not sample_metrics:
        return {}
    keys = sorted({key for row in sample_metrics for key in row})
    summary: dict[str, float] = {}
    for key in keys:
        values = np.array([float(row[key]) for row in sample_metrics if key in row], dtype=np.float64)
        summary[f"sample_{key}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"sample_{key}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
    return summary


def summarize_uncertainty(result: MCDropoutPrediction) -> dict[str, float]:
    """Return compact scalar summaries of MC uncertainty maps."""
    aleatoric = result.aleatoric_uncertainty.detach().cpu().numpy().astype(np.float64)
    epistemic = result.epistemic_uncertainty.detach().cpu().numpy().astype(np.float64)
    return {
        "aleatoric_uncertainty_mean": float(aleatoric.mean()),
        "aleatoric_uncertainty_p95": float(np.percentile(aleatoric, 95)),
        "aleatoric_uncertainty_max": float(aleatoric.max(initial=0.0)),
        "epistemic_uncertainty_mean": float(epistemic.mean()),
        "epistemic_uncertainty_p95": float(np.percentile(epistemic, 95)),
        "epistemic_uncertainty_max": float(epistemic.max(initial=0.0)),
    }


def save_uncertainty_maps(output_dir: Path, case_id: str, result: MCDropoutPrediction) -> None:
    """Save per-voxel uncertainty maps for optional inspection."""
    map_dir = output_dir / "uncertainty_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        map_dir / f"{case_id}.npz",
        aleatoric_uncertainty=result.aleatoric_uncertainty.detach().cpu().numpy().astype(np.float32),
        epistemic_uncertainty=result.epistemic_uncertainty.detach().cpu().numpy().astype(np.float32),
        mean_prediction=result.pred.detach().cpu().numpy().astype(np.int16),
    )


def write_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "summary": summary,
        "cases": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    base_fieldnames = ["case_id", "wt_dice", "tc_dice", "et_dice", "mean_dice", "wt_hd95", "tc_hd95", "et_hd95", "mean_hd95"]
    extra_fieldnames = sorted({key for row in rows for key in row if key not in base_fieldnames})
    fieldnames = [*base_fieldnames, *extra_fieldnames]
    with open(output_dir / "per_case.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-volume BraTS validation with sliding-window inference.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing experiment.snapshot.toml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <run-dir>/checkpoints/best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--window-size", type=int, nargs=3, default=[128, 128, 128], help="Sliding window size")
    parser.add_argument(
        "--model-crop-size",
        type=int,
        nargs=3,
        default=None,
        help="Override [model].crop_size before constructing models that bake input resolution into the network.",
    )
    parser.add_argument("--window-batch-size", type=int, default=1, help="Number of sliding-window patches per forward pass")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap in [0, 1)")
    parser.add_argument("--device", default=None, help="Override device. Defaults to [trainer].device from snapshot")
    parser.add_argument("--amp", action="store_true", help="Use autocast during inference")
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only the first N cases")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <run-dir>/full_volume_validation/<split>")
    parser.add_argument(
        "--empty-hd95-penalty",
        type=float,
        default=BRATS_EMPTY_HD95_PENALTY,
        help="HD95 value used when exactly one mask is empty. Defaults to the BraTS 240x240x155 diagonal.",
    )
    parser.add_argument(
        "--et-empty-pred-threshold",
        type=int,
        default=DEFAULT_ET_EMPTY_PRED_THRESHOLD,
        help=(
            "When GT ET is empty, ignore ET predictions with at most this many voxels for ET Dice/HD95. "
            "Set to -1 to disable."
        ),
    )
    parser.add_argument("--mc-samples", type=int, default=1, help="Number of MC-dropout samples. Values >1 enable MC mode")
    parser.add_argument("--mc-dropout-p", type=float, default=None, help="Override dropout probability before MC validation")
    parser.add_argument(
        "--mc-sample-metrics",
        choices=MC_SAMPLE_METRICS_CHOICES,
        default="dice",
        help="Per-MC-sample metrics to compute: none, dice, or full Dice+HD95. Final MC mean metrics always include Dice+HD95.",
    )
    parser.add_argument("--mc-seed", type=int, default=None, help="Base seed for reproducible MC dropout masks")
    parser.add_argument("--save-uncertainty", action="store_true", help="Save per-case entropy and variance maps as compressed NPZ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "experiment.snapshot.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")

    config = load_toml(config_path)
    model_config = config.get("model", {})
    if model_config.get("name") == "unet" and "channel_multipliers" in model_config:
        # Older comparison runs used the configurable UNet3D under the
        # registry name "unet". That name now points to PaperUNet3D.
        model_config["name"] = "unet3d"
        print("Using legacy UNet3D registry mapping for this checkpoint")
    if args.model_crop_size is not None:
        config.setdefault("model", {})["crop_size"] = list(args.model_crop_size)
    bundle = build_component_bundle(config, create_default_registries())

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle.model.load_state_dict(payload["state_dict"])
    if args.mc_dropout_p is not None:
        changed = set_dropout_probability(bundle.model, float(args.mc_dropout_p))
        print(f"Overrode dropout probability on {changed} stochastic modules to p={float(args.mc_dropout_p):.4f}")

    trainer_cfg = config.get("trainer", {})
    device = resolve_device(args.device or trainer_cfg.get("device", "auto"))
    bundle.model.to(device)
    active_dropout_modules = count_active_mc_dropout_modules(bundle.model)
    if int(args.mc_samples) > 1:
        print(f"MC-dropout samples: {int(args.mc_samples)}; active stochastic modules: {active_dropout_modules}")
        if active_dropout_modules == 0:
            print(
                "WARNING: no active dropout modules found. MC samples will be deterministic unless --mc-dropout-p is set "
                "or the checkpoint was trained with nonzero dropout.",
                file=sys.stderr,
            )

    case_ids = resolve_split_ids(bundle, run_dir, args.split)
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise ValueError(f"Split '{args.split}' has no cases to evaluate")

    data_cfg = config.get("data", {})
    preprocessed_dir = Path(data_cfg.get("preprocessed_dir") or data_cfg["root_dir"])
    data_format = str(data_cfg.get("data_format", "npz")).lower()
    num_classes = int(config.get("model", {}).get("num_classes", len(config["experiment"]["class_names"])))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "full_volume_validation" / args.split
    empty_hd95_penalty = float(args.empty_hd95_penalty)
    et_empty_pred_threshold = (
        int(args.et_empty_pred_threshold)
        if args.et_empty_pred_threshold is not None and int(args.et_empty_pred_threshold) >= 0
        else None
    )

    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        print(f"[{index:4d}/{len(case_ids):4d}] {case_id}", flush=True)
        volume, target, spacing = load_preprocessed_case(preprocessed_dir, case_id, data_format=data_format)
        sample_metrics: list[dict[str, float]] = []
        sample_callback: Callable[[int, torch.Tensor], None] | None = None
        if int(args.mc_samples) > 1 and args.mc_sample_metrics != "none":
            sample_callback = partial(
                collect_sample_metrics,
                sample_metrics,
                args.mc_sample_metrics,
                target,
                spacing,
                empty_hd95_penalty=empty_hd95_penalty,
                et_empty_pred_threshold=et_empty_pred_threshold,
            )

        if int(args.mc_samples) > 1:
            mc_result = mc_dropout_predict(
                bundle.model,
                volume,
                device=device,
                num_classes=num_classes,
                window_size=tuple(args.window_size),
                overlap=float(args.overlap),
                amp=bool(args.amp),
                window_batch_size=int(args.window_batch_size),
                mc_samples=int(args.mc_samples),
                mc_seed=args.mc_seed,
                sample_callback=sample_callback,
                postprocess_fn=bundle.task.postprocess_outputs,
            )
            pred = mc_result.pred
        else:
            mc_result = None
            pred = sliding_window_predict(
                bundle.model,
                volume,
                device=device,
                num_classes=num_classes,
                window_size=tuple(args.window_size),
                overlap=float(args.overlap),
                amp=bool(args.amp),
                window_batch_size=int(args.window_batch_size),
                postprocess_fn=bundle.task.postprocess_outputs,
            )
        metrics = compute_region_metrics(
            pred,
            target,
            spacing=spacing,
            empty_hd95_penalty=empty_hd95_penalty,
            et_empty_pred_threshold=et_empty_pred_threshold,
        )
        if mc_result is not None:
            metrics.update(summarize_mc_sample_metrics(sample_metrics))
            metrics.update(summarize_uncertainty(mc_result))
            if args.save_uncertainty:
                save_uncertainty_maps(output_dir, case_id, mc_result)
        row = {"case_id": case_id, **metrics}
        rows.append(row)
        stability = f" sample_mean_std={row['sample_mean_dice_std']:.4f}" if "sample_mean_dice_std" in row else ""
        uncertainty = (
            f" u_ale={row['aleatoric_uncertainty_mean']:.4f} u_epi={row['epistemic_uncertainty_mean']:.6f}"
            if "aleatoric_uncertainty_mean" in row
            else ""
        )
        print(
            "  "
            f"mean_dice={row['mean_dice']:.4f} "
            f"WT={row['wt_dice']:.4f} TC={row['tc_dice']:.4f} ET={row['et_dice']:.4f} "
            f"mean_hd95={row['mean_hd95']:.2f}{stability}{uncertainty}",
            flush=True,
        )

    summary = _summary(rows)
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "split": args.split,
        "num_cases": len(case_ids),
        "window_size": list(args.window_size),
        "window_batch_size": int(args.window_batch_size),
        "overlap": float(args.overlap),
        "device": str(device),
        "preprocessed_dir": str(preprocessed_dir),
        "empty_hd95_penalty": empty_hd95_penalty,
        "et_empty_pred_threshold": et_empty_pred_threshold,
        "mc_samples": int(args.mc_samples),
        "mc_dropout_p_override": args.mc_dropout_p,
        "mc_sample_metrics": args.mc_sample_metrics if int(args.mc_samples) > 1 else "none",
        "mc_seed": args.mc_seed,
        "active_dropout_modules": active_dropout_modules,
        "save_uncertainty": bool(args.save_uncertainty),
    }
    write_outputs(output_dir, rows=rows, summary=summary, metadata=metadata)

    print("\nSummary")
    print(f"  mean Dice: {summary['mean_dice_mean']:.4f} +/- {summary['mean_dice_std']:.4f}")
    print(f"  WT Dice:   {summary['wt_dice_mean']:.4f} +/- {summary['wt_dice_std']:.4f}")
    print(f"  TC Dice:   {summary['tc_dice_mean']:.4f} +/- {summary['tc_dice_std']:.4f}")
    print(f"  ET Dice:   {summary['et_dice_mean']:.4f} +/- {summary['et_dice_std']:.4f}")
    print(f"  mean HD95: {summary['mean_hd95_mean']:.2f} +/- {summary['mean_hd95_std']:.2f}")
    if "sample_mean_dice_std_mean" in summary:
        print(f"  MC sample mean Dice std: {summary['sample_mean_dice_std_mean']:.4f}")
    if "aleatoric_uncertainty_mean_mean" in summary:
        print(f"  aleatoric uncertainty:   {summary['aleatoric_uncertainty_mean_mean']:.4f}")
        print(f"  epistemic uncertainty:   {summary['epistemic_uncertainty_mean_mean']:.6f}")
    print(f"Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
