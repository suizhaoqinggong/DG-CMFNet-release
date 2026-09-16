"""Evaluate checkpoints on training-style center-crop validation patches.

This mirrors the training validation input path: each validation case is loaded
through the run snapshot's val DataLoader, which applies CenterCrop3D(crop_size).
It then reports BraTS WT/TC/ET Dice and HD95 for those cropped predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from framework.core.device import move_batch_to_device, resolve_device
from framework.registry.defaults import create_default_registries
from framework.registry.factories import build_component_bundle, load_toml
from scripts.validate_full_volume import (
    BRATS_EMPTY_HD95_PENALTY,
    DEFAULT_ET_EMPTY_PRED_THRESHOLD,
    _summary,
    compute_region_metrics,
    write_outputs,
)


def _reset_import_side_effects() -> None:
    """Drop model-local package overrides left behind by previous evaluations.

    SegMamba (v1) injects its bundled ``mamba_ssm`` BiMamba fork onto ``sys.path``
    and into ``sys.modules``. SegMamba-V2 expects the stock package from
    site-packages. Without this reset, a later V2 evaluation in the same process
    fails with ``assert bimamba_type == "v3"``.
    """
    polluted_path_tokens = (
        "/SegMamba/mamba",
        "/SegMamba/causal-conv1d",
        "\\SegMamba\\mamba",
        "\\SegMamba\\causal-conv1d",
    )
    cleaned_paths: list[str] = []
    for entry in sys.path:
        normalized = entry.replace("\\", "/")
        if any(token.replace("\\", "/") in normalized for token in polluted_path_tokens):
            # Keep SegMamba-V2 source roots; only drop the v1 mamba fork paths.
            if normalized.rstrip("/").endswith("/SegMamba-V2") or "/SegMamba-V2/" in normalized:
                cleaned_paths.append(entry)
                continue
            if normalized.endswith("/mamba") or normalized.endswith("/causal-conv1d"):
                continue
        cleaned_paths.append(entry)
    sys.path[:] = cleaned_paths

    drop_prefixes = (
        "mamba_ssm",
        "causal_conv1d",
        "selective_scan_cuda",
        "causal_conv1d_cuda",
        "_dgcmfnet_upstream_segmamba",
        "_dgcmfnet_upstream_segmamba_v2",
        "_dgcmfnet_upstream_nnmamba",
    )
    for module_name in list(sys.modules):
        if module_name in drop_prefixes or any(
            module_name == prefix or module_name.startswith(prefix + ".") for prefix in drop_prefixes
        ):
            del sys.modules[module_name]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Run spec has an empty name: {spec}")
        if not path:
            raise ValueError(f"Run spec has an empty path: {spec}")
        return name, Path(path)

    path = Path(spec)
    return path.name, path


def _load_spacing(preprocessed_dir: Path, case_id: str, data_format: str) -> tuple[float, float, float]:
    if data_format == "npy":
        spacing_path = preprocessed_dir / case_id / "target_spacing.npy"
        spacing_raw = (
            np.load(spacing_path).astype(np.float32)
            if spacing_path.exists()
            else np.ones(3, dtype=np.float32)
        )
    else:
        path = preprocessed_dir / f"{case_id}.npz"
        with np.load(path) as data:
            spacing_raw = data["target_spacing"].astype(np.float32) if "target_spacing" in data else np.ones(3, dtype=np.float32)

    return (float(spacing_raw[2]), float(spacing_raw[0]), float(spacing_raw[1]))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def evaluate_run(
    name: str,
    run_dir: Path,
    *,
    output_root: Path,
    device: torch.device,
    amp_override: bool | None,
    empty_hd95_penalty: float,
    et_empty_pred_threshold: int | None,
) -> dict[str, Any]:
    config_path = run_dir / "experiment.snapshot.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")

    config = load_toml(config_path)
    bundle = build_component_bundle(config, create_default_registries())

    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle.model.load_state_dict(payload["state_dict"])
    bundle.model.to(device)
    bundle.model.eval()

    trainer_cfg = config.get("trainer", {})
    amp_enabled = bool(trainer_cfg.get("amp", False)) if amp_override is None else bool(amp_override)
    data_cfg = config.get("data", {})
    preprocessed_dir = Path(data_cfg.get("preprocessed_dir") or data_cfg["root_dir"])
    data_format = str(data_cfg.get("data_format", "npz")).lower()
    crop_size = list(data_cfg.get("crop_size", [128, 128, 128]))

    rows: list[dict[str, Any]] = []
    total = len(bundle.val_loader.dataset)
    seen = 0
    device_type = device.type
    print(f"\n{name}: {run_dir}")
    print(
        f"  checkpoint_epoch={payload.get('epoch')} cases={total} crop_size={crop_size} "
        f"amp={amp_enabled} device={device}",
        flush=True,
    )

    with torch.no_grad():
        for batch in bundle.val_loader:
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                outputs = bundle.model(batch)
                probs = bundle.task.postprocess_outputs(outputs)

            pred_labels = torch.argmax(probs, dim=1).detach().cpu()
            target_labels = batch["label"].detach().cpu()
            case_ids = list(batch["id"])
            for case_id, pred_case, target_case in zip(case_ids, pred_labels, target_labels):
                spacing = _load_spacing(preprocessed_dir, str(case_id), data_format)
                metrics = compute_region_metrics(
                    pred_case,
                    target_case,
                    spacing=spacing,
                    empty_hd95_penalty=empty_hd95_penalty,
                    et_empty_pred_threshold=et_empty_pred_threshold,
                )
                rows.append({"case_id": str(case_id), **metrics})
                seen += 1
                print(
                    f"  [{seen:3d}/{total:3d}] {case_id} "
                    f"mean_dice={metrics['mean_dice']:.4f} mean_hd95={metrics['mean_hd95']:.2f}",
                    flush=True,
                )

    summary = _summary(rows)
    out_dir = output_root / name
    metadata = {
        "variant": name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "split": "val",
        "num_cases": total,
        "evaluation": "training_validation_center_crop",
        "crop_size": crop_size,
        "device": str(device),
        "amp": amp_enabled,
        "preprocessed_dir": str(preprocessed_dir),
        "data_format": data_format,
        "empty_hd95_penalty": empty_hd95_penalty,
        "et_empty_pred_threshold": et_empty_pred_threshold,
    }
    write_outputs(out_dir, rows=rows, summary=summary, metadata=metadata)

    print(
        "  Summary: "
        f"mean_dice={summary['mean_dice_mean']:.4f} "
        f"WT={summary['wt_dice_mean']:.4f} TC={summary['tc_dice_mean']:.4f} ET={summary['et_dice_mean']:.4f} "
        f"mean_hd95={summary['mean_hd95_mean']:.2f}",
        flush=True,
    )

    bundle.model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"variant": name, **metadata, **summary}


def write_comparison(output_root: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "checkpoint_epoch",
        "num_cases",
        "mean_dice_mean",
        "wt_dice_mean",
        "tc_dice_mean",
        "et_dice_mean",
        "mean_hd95_mean",
        "wt_hd95_mean",
        "tc_hd95_mean",
        "et_hd95_mean",
        "mean_dice_std",
        "mean_hd95_std",
        "run_dir",
    ]
    with open(output_root / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _format_float(row[field]) if isinstance(row.get(field), float) else row.get(field, "")
                    for field in fields
                }
            )

    payload = {"runs": rows}
    (output_root / "comparison.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate best checkpoints on center-crop val patches.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as name=run_dir or just run_dir. May be repeated.",
    )
    parser.add_argument("--output-root", required=True, help="Directory for center-crop validation outputs")
    parser.add_argument("--device", default="auto", help="Device spec, e.g. cuda:0")
    parser.add_argument("--amp", action="store_true", help="Force AMP on, overriding the run snapshot")
    parser.add_argument("--no-amp", action="store_true", help="Force AMP off, overriding the run snapshot")
    parser.add_argument(
        "--empty-hd95-penalty",
        type=float,
        default=BRATS_EMPTY_HD95_PENALTY,
        help="HD95 value used when exactly one mask is empty.",
    )
    parser.add_argument(
        "--et-empty-pred-threshold",
        type=int,
        default=DEFAULT_ET_EMPTY_PRED_THRESHOLD,
        help="When GT ET is empty, ignore ET predictions with at most this many voxels. Set to -1 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.amp and args.no_amp:
        raise ValueError("--amp and --no-amp are mutually exclusive")

    amp_override = True if args.amp else False if args.no_amp else None
    et_empty_pred_threshold = (
        int(args.et_empty_pred_threshold)
        if args.et_empty_pred_threshold is not None and int(args.et_empty_pred_threshold) >= 0
        else None
    )
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for index, spec in enumerate(args.run):
        name, run_dir = _parse_run_spec(spec)
        # Isolate mamba/fork side effects between sequential model evaluations.
        if index > 0:
            _reset_import_side_effects()
        comparison_rows.append(
            evaluate_run(
                name,
                run_dir,
                output_root=output_root,
                device=device,
                amp_override=amp_override,
                empty_hd95_penalty=float(args.empty_hd95_penalty),
                et_empty_pred_threshold=et_empty_pred_threshold,
            )
        )
        # Also reset after each run so the next process-local import is clean.
        _reset_import_side_effects()

    write_comparison(output_root, comparison_rows)
    print(f"\nComparison saved to: {output_root / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
