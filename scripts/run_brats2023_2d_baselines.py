"""Run the slice-wise 2D U-Net and Res-U-Net BraTS2023 baselines.

The protocol mirrors the completed BraTS2020 2D runs:

- Paper U-Net: 4 axial slices per case, batch size 1.
- Res-U-Net: 16 axial slices per case, batch size 2.
- Both: 50% foreground-slice sampling, Dice + cross-entropy, and whole-volume
  validation reconstructed from slice-wise predictions.
"""

from __future__ import annotations

import argparse
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_original_loss_models import framework_train_command, load_toml, run_training, write_toml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    experiment_name: str
    model_config: Path
    batch_size: int
    slices_per_case: int


RUNS = (
    RunSpec(
        name="unet_2d",
        label="01_unet_2d_dc_ce",
        experiment_name="brats2023-slicewise-2d-paper-unet-dc-ce",
        model_config=Path("configs/model.unet_2d.toml"),
        batch_size=1,
        slices_per_case=4,
    ),
    RunSpec(
        name="resunet_2d",
        label="02_resunet_2d_dc_ce",
        experiment_name="brats2023-slicewise-2d-resunet-dc-ce",
        model_config=Path("configs/model.resunet_2d.toml"),
        batch_size=2,
        slices_per_case=16,
    ),
)

RUN_NAMES = tuple(spec.name for spec in RUNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/experiment.brats.toml",
        help="BraTS2023 experiment TOML used as the data and trainer template.",
    )
    parser.add_argument("--gpu", type=int, default=1, help="CUDA device index (default: 1).")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count for both runs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience for both runs.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=RUN_NAMES,
        default=None,
        help="Run only the selected models while preserving the canonical order.",
    )
    parser.add_argument(
        "--start-from",
        choices=RUN_NAMES,
        default=None,
        help="Resume the sequence from this model (inclusive).",
    )
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for frozen sequence configs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without training.")
    return parser.parse_args()


def selected_runs(only: list[str] | None, start_from: str | None) -> tuple[RunSpec, ...]:
    selected = RUNS
    if start_from is not None:
        selected = selected[RUN_NAMES.index(start_from) :]
    if only is not None:
        only_names = set(only)
        selected = tuple(spec for spec in selected if spec.name in only_names)
    return selected


def build_experiment_config(
    base_config: dict[str, Any],
    spec: RunSpec,
    *,
    gpu: int,
    epochs: int | None,
    patience: int | None,
) -> dict[str, Any]:
    config = deepcopy(base_config)

    experiment = dict(config.get("experiment", {}))
    experiment["name"] = spec.experiment_name
    config["experiment"] = experiment

    data = dict(config.get("data", {}))
    data["adapter"] = "brats_2d"
    data["batch_size"] = spec.batch_size
    data["augment_train"] = True
    data["slices_per_case"] = spec.slices_per_case
    data["foreground_slice_ratio"] = 0.5
    for split_file_key in ("train_ids_file", "val_ids_file", "test_ids_file"):
        data.pop(split_file_key, None)
    config["data"] = data

    task = dict(config.get("task", {}))
    task["name"] = "segmentation"
    task["loss"] = "dc_and_ce"
    task["loss_lambda_alpha"] = 0.0
    task["loss_lambda_beta"] = 0.0
    config["task"] = task

    trainer = dict(config.get("trainer", {}))
    trainer["device"] = f"cuda:{gpu}"
    trainer["amp"] = True
    if epochs is not None:
        trainer["epochs"] = epochs
    if patience is not None:
        trainer["patience"] = patience
    config["trainer"] = trainer
    return config


def validate_inputs(base_config: Path, runs: tuple[RunSpec, ...], gpu: int) -> None:
    if gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if not runs:
        raise ValueError("No models remain after applying --only and --start-from")

    paths = (base_config, *(spec.model_config for spec in runs))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    runs = selected_runs(args.only, args.start_from)
    base_config = Path(args.base_config)
    validate_inputs(base_config, runs, args.gpu)
    base = load_toml(base_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_root = Path(args.snapshot_root) / f"brats2023_2d_baseline_sequence_gpu{args.gpu}_{timestamp}"
    snapshot_dir = sequence_root / "configs"
    planned = [
        (
            spec,
            snapshot_dir / f"{spec.label}.experiment.toml",
            snapshot_dir / f"{spec.label}.model.toml",
        )
        for spec in runs
    ]

    print(f"BraTS2023 slice-wise 2D baseline sequence on cuda:{args.gpu}")
    print("Order:", " -> ".join(spec.name for spec in runs))
    print(f"Sequence artifacts: {sequence_root}")

    if args.dry_run:
        for spec, frozen_experiment, frozen_model in planned:
            command = framework_train_command(frozen_experiment, frozen_model)
            print(
                f"{spec.label}: loss=dc_and_ce batch_size={spec.batch_size} "
                f"slices_per_case={spec.slices_per_case} command={' '.join(command)}"
            )
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for spec, frozen_experiment, frozen_model in planned:
        experiment_config = build_experiment_config(
            base,
            spec,
            gpu=args.gpu,
            epochs=args.epochs,
            patience=args.patience,
        )
        write_toml(experiment_config, frozen_experiment)
        shutil.copy2(spec.model_config, frozen_model)

        print(
            f"\n{'=' * 80}\n{spec.label}\n"
            f"loss=dc_and_ce batch_size={spec.batch_size} slices_per_case={spec.slices_per_case}\n"
            f"model={spec.model_config}\n{'=' * 80}"
        )
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print(f"All BraTS2023 2D baseline runs completed. Sequence artifacts: {sequence_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
