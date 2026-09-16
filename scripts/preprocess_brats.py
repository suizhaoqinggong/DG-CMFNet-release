#!/usr/bin/env python3
"""Offline BraTS preprocessing: resample → normalise → save compact .npz.

Mirrors the DiffUNet preprocessing philosophy using scipy+nibabel.

Usage:
    python scripts/preprocess_brats.py \
        --input /path/to/BraTS-MEN-Train \
        --output /path/to/preprocessed \
        --spacing 1.0 1.0 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

MODALITY_KEYS = ["t1", "t1ce", "t2", "flair"]

# Possible file suffixes per modality — tried in order (supports BraTS MEN + GLI naming)
_MODALITY_SUFFIXES: dict[str, list[str]] = {
    "t1": ["t1.nii.gz", "t1n.nii.gz"],
    "t1ce": ["t1ce.nii.gz", "t1c.nii.gz"],
    "t2": ["t2.nii.gz", "t2w.nii.gz"],
    "flair": ["flair.nii.gz", "t2f.nii.gz"],
}

_SEG_SUFFIXES = ["truth.nii.gz", "seg.nii.gz"]
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}


def _extract_spacing(nii: nib.Nifti1Image) -> np.ndarray:
    pixdim = nii.header.get_zooms()
    return np.array(pixdim[:3], dtype=np.float64)


def _resample_volume(
    volume: np.ndarray,
    orig_spacing: np.ndarray,
    target_spacing: np.ndarray,
    *,
    is_label: bool = False,
) -> np.ndarray:
    factors = orig_spacing / target_spacing
    if np.allclose(factors, 1.0, atol=1e-3):
        return volume.copy()

    order = 0 if is_label else 3

    if volume.ndim == 4:
        return np.stack([zoom(v, factors, order=order) for v in volume], axis=0)

    return zoom(volume, factors, order=order).astype(np.int64 if is_label else np.float32)


def _zscore_normalize(volume: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(volume, dtype=np.float32)
    for m in range(volume.shape[0]):
        v = volume[m]
        mask = v > 1e-6
        if mask.sum() == 0:
            normalized[m] = v
        else:
            mu, sigma = v[mask].mean(), v[mask].std()
            sigma = max(sigma, 1e-8)
            out = (v - mu) / sigma
            out[~mask] = 0.0
            normalized[m] = out
    return normalized


def _remap_labels(seg: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(seg, dtype=np.int64)
    for old, new in LABEL_REMAP.items():
        remapped[seg == old] = new
    return remapped


def process_case(
    case_dir: Path,
    output_dir: Path,
    target_spacing: np.ndarray,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    case_id = case_dir.name
    out_path = output_dir / f"{case_id}.npz"

    if out_path.exists() and not overwrite:
        return {"case_id": case_id, "status": "skipped"}

    def _find_any_file(candidates: list[str]) -> Path:
        for suffix in candidates:
            exact = case_dir / suffix
            if exact.exists():
                return exact
            matches = list(case_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"{case_id}: missing any of {candidates}")

    mod_arrays: list[np.ndarray] = []
    spacings: list[np.ndarray] = []

    for mod_key in MODALITY_KEYS:
        fpath = _find_any_file(_MODALITY_SUFFIXES[mod_key])
        nii = nib.load(str(fpath))
        data = nii.get_fdata().astype(np.float32)
        mod_arrays.append(np.transpose(data, (2, 0, 1)))
        spacings.append(_extract_spacing(nii))

    volume = np.stack(mod_arrays, axis=0).astype(np.float32)

    seg_path = _find_any_file(_SEG_SUFFIXES)
    seg_nii = nib.load(str(seg_path))
    seg = seg_nii.get_fdata().astype(np.int64)
    seg = np.transpose(seg, (2, 0, 1))

    spacing_stack = np.stack(spacings, axis=0)
    if not np.allclose(spacing_stack, spacing_stack[0], atol=1e-3):
        sys.stderr.write(
            f"WARNING {case_id}: modalities have inconsistent spacings: {spacing_stack}\n"
        )
    orig_spacing = spacing_stack[0]

    volume_rs = _resample_volume(volume, orig_spacing, target_spacing, is_label=False)
    seg_rs = _resample_volume(seg, orig_spacing, target_spacing, is_label=True)
    seg_rs = _remap_labels(seg_rs)
    volume_rs = _zscore_normalize(volume_rs)

    save_dict: dict[str, np.ndarray] = {}
    for i, key in enumerate(MODALITY_KEYS):
        save_dict[key] = volume_rs[i].astype(np.float32)
    save_dict["seg"] = seg_rs.astype(np.int8)
    save_dict["orig_spacing"] = orig_spacing.astype(np.float32)
    save_dict["target_spacing"] = target_spacing.astype(np.float32)
    save_dict["orig_shape"] = np.array(volume.shape[1:], dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)

    return {
        "case_id": case_id,
        "status": "ok",
        "orig_shape": list(volume.shape[1:]),
        "new_shape": list(volume_rs.shape[1:]),
        "orig_spacing": orig_spacing.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BraTS preprocessing")
    parser.add_argument("--input", required=True, help="Path to raw BraTS directory")
    parser.add_argument("--output", required=True, help="Output directory for .npz files")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target isotropic spacing (d h w) in mm. Default: 1.0 1.0 1.0",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files")
    args = parser.parse_args()

    target_spacing = np.array(args.spacing, dtype=np.float64)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}")
        sys.exit(1)

    case_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not case_dirs:
        print(f"ERROR: no case directories found in {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(case_dirs)} cases")
    print(f"Target spacing:  {target_spacing.tolist()} mm")
    print(f"Output:           {output_dir}")
    print()

    summary: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, case_dir in enumerate(case_dirs, 1):
        try:
            info = process_case(case_dir, output_dir, target_spacing, overwrite=args.overwrite)
            summary.append(info)
            if info["status"] == "ok":
                print(
                    f"  [{i:3d}/{len(case_dirs)}] {info['case_id']:30s}  "
                    f"{str(info['orig_shape']):20s} → {str(info['new_shape']):20s}"
                )
            else:
                print(f"  [{i:3d}/{len(case_dirs)}] {info['case_id']:30s}  SKIPPED")
        except Exception as exc:
            msg = f"  [{i:3d}/{len(case_dirs)}] {case_dir.name:30s}  ERROR: {exc}"
            print(msg)
            errors.append(msg)

    summary_path = output_dir / "preprocess_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "target_spacing": target_spacing.tolist(),
                "num_cases": len(case_dirs),
                "num_processed": sum(1 for s in summary if s["status"] == "ok"),
                "num_skipped": sum(1 for s in summary if s["status"] == "skipped"),
                "num_errors": len(errors),
                "cases": summary,
                "errors": errors,
            },
            f,
            indent=2,
        )

    ok = sum(1 for s in summary if s["status"] == "ok")
    print(f"\nDone. {ok} processed, {len(errors)} errors. Summary → {summary_path}")


if __name__ == "__main__":
    main()
