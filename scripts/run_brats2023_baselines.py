"""Run selected BraTS2023 baselines sequentially on one GPU.

The default sequence is:

1. TransBTS
2. SegFormer3D
3. Slim UNETR
4. Attention U-Net
5. VT-UNet

Each run uses the model's original supervised loss and receives a frozen copy
of both the experiment and model configuration for reproducibility.
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
    model_config: Path
    loss: str
    loss_gamma: float = 2.0


RUNS = (
    RunSpec(
        name="transbts",
        label="01_transbts_original_softmax_dice",
        model_config=Path("configs/model.transbts.toml"),
        loss="transbts_softmax_dice",
    ),
    RunSpec(
        name="segformer3d",
        label="02_segformer3d_original_region_dice",
        model_config=Path("configs/model.segformer3d.toml"),
        loss="brats_region_dice",
    ),
    RunSpec(
        name="slim_unetr",
        label="03_slim_unetr_original_focal_dice",
        model_config=Path("configs/model.slim_unetr.toml"),
        loss="slim_unetr_focal_dice",
    ),
    RunSpec(
        name="attention_unet",
        label="04_attention_unet_original_bce_focal",
        model_config=Path("configs/model.attention_unet.toml"),
        loss="bce_focal",
        loss_gamma=0.0,
    ),
    RunSpec(
        name="vtunet",
        label="05_vtunet_original_dc_ce",
        model_config=Path("configs/model.vtunet.toml"),
        loss="dc_and_ce",
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
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count for every run.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience for every run.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=RUN_NAMES,
        default=None,
        help="Run only these models while preserving the canonical sequence order.",
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
        start_index = RUN_NAMES.index(start_from)
        selected = selected[start_index:]
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
    experiment["name"] = f"brats2023-{spec.label.split('_', 1)[1]}"
    config["experiment"] = experiment

    task = dict(config.get("task", {}))
    task["loss"] = spec.loss
    task["loss_gamma"] = spec.loss_gamma
    task["loss_balance_param"] = 1.0
    task["loss_lambda_alpha"] = 0.0
    task["loss_lambda_beta"] = 0.0
    config["task"] = task

    trainer = dict(config.get("trainer", {}))
    trainer["device"] = f"cuda:{gpu}"
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
    sequence_root = Path(args.snapshot_root) / f"brats2023_baseline_sequence_gpu{args.gpu}_{timestamp}"
    snapshot_dir = sequence_root / "configs"

    planned: list[tuple[RunSpec, Path, Path]] = []
    for spec in runs:
        frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
        frozen_model = snapshot_dir / f"{spec.label}.model.toml"
        planned.append((spec, frozen_experiment, frozen_model))

    print(f"BraTS2023 baseline sequence on cuda:{args.gpu}")
    print("Order:", " -> ".join(spec.name for spec in runs))
    print(f"Sequence artifacts: {sequence_root}")

    if args.dry_run:
        for spec, frozen_experiment, frozen_model in planned:
            command = framework_train_command(frozen_experiment, frozen_model)
            print(f"{spec.label}: loss={spec.loss} command={' '.join(command)}")
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

        print(f"\n{'=' * 80}\n{spec.label}\nloss={spec.loss}\nmodel={spec.model_config}\n{'=' * 80}")
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print(f"All BraTS2023 baseline runs completed. Sequence artifacts: {sequence_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
