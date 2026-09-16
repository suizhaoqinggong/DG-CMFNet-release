"""Run comparison models with their original loss choices inside DG-CMFNet.

The adapters keep DG-CMFNet data loading, metrics, checkpointing, and 4-class
BraTS outputs. Losses are mapped to the closest original supervised setup:

- U-Net, ResU-Net, Dense U-Net: Dice + CE
- Attention U-Net: BCE/Focal with gamma=0, matching its original default
- nnU-Net, VT-UNet, nnFormer, SwinUNETR: Dice + CE
- TransBTS: softmax Dice
- NestedFormer: sigmoid region Dice
- SegFormer3D: BraTS WT/TC/ET region Dice
- Slim UNETR: focal + Dice

The local Swin-UNETR source tree is self-supervised pretraining code; that SSL
loss is not compatible with this supervised segmentation adapter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli
import tomli_w


@dataclass(frozen=True)
class LossSpec:
    label: str
    loss: str
    gamma: float = 2.0
    balance_param: float = 1.0


@dataclass(frozen=True)
class RunSpec:
    label: str
    model_config: Path
    loss_spec: LossSpec


RUNS = (
    RunSpec(
        "01_unet_original_dc_ce",
        Path("configs/model.unet.toml"),
        LossSpec("unet_dc_ce", "dc_and_ce"),
    ),
    RunSpec(
        "02_resunet_original_dc_ce",
        Path("configs/model.resunet.toml"),
        LossSpec("resunet_dc_ce", "dc_and_ce"),
    ),
    RunSpec(
        "03_dense_unet_original_dc_ce",
        Path("configs/model.dense_unet.toml"),
        LossSpec("dense_unet_dc_ce", "dc_and_ce"),
    ),
    RunSpec(
        "04_attention_unet_original_bce_focal",
        Path("configs/model.attention_unet.toml"),
        LossSpec("attention_unet_bce_focal", "bce_focal", gamma=0.0),
    ),
    RunSpec(
        "05_nnunet_original_dc_ce",
        Path("configs/model.nnunet.toml"),
        LossSpec("nnunet_dc_ce", "dc_and_ce"),
    ),
    RunSpec(
        "06_vtunet_original_dc_ce",
        Path("configs/model.vtunet.toml"),
        LossSpec("vtunet_dc_ce", "dc_and_ce"),
    ),
    RunSpec(
        "07_transbts_original_softmax_dice",
        Path("configs/model.transbts.toml"),
        LossSpec("transbts_softmax_dice", "transbts_softmax_dice"),
    ),
    RunSpec(
        "08_nestedformer_original_sigmoid_region_dice",
        Path("configs/model.nestedformer.toml"),
        LossSpec("nestedformer_sigmoid_region_dice", "brats_sigmoid_region_dice"),
    ),
    RunSpec(
        "09_segformer3d_original_region_dice",
        Path("configs/model.segformer3d.toml"),
        LossSpec("segformer3d_region_dice", "brats_region_dice"),
    ),
    RunSpec(
        "10_slim_unetr_original_focal_dice",
        Path("configs/model.slim_unetr.toml"),
        LossSpec("slim_unetr_focal_dice", "slim_unetr_focal_dice"),
    ),
    RunSpec(
        "11_nnformer_original_dc_ce",
        Path("configs/model.nnformer.toml"),
        LossSpec("nnformer_dc_ce", "dc_and_ce"),
    ),
    RunSpec(
        "12_swinbts_original_dice_ce",
        Path("configs/model.swinbts.toml"),
        LossSpec("swinbts_dice_ce", "swinbts_dice_ce"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/experiment.brats2020.dice_only.toml",
        help="Experiment TOML used as the data/trainer template.",
    )
    parser.add_argument("--device", default=None, help='Override [trainer].device, for example "cuda:3".')
    parser.add_argument("--epochs", type=int, default=None, help="Override [trainer].epochs for every run.")
    parser.add_argument("--patience", type=int, default=None, help="Override [trainer].patience for every run.")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only specs whose labels contain one of these case-insensitive substrings.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without launching training.")
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for frozen configs.")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)


def write_toml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def framework_train_command(config_path: Path, model_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "framework.cli.main",
        "train",
        "--configs",
        str(config_path),
        "--model",
        str(model_path),
    ]


def selected_runs(patterns: list[str] | None) -> tuple[RunSpec, ...]:
    if not patterns:
        return RUNS
    lowered = [pattern.lower() for pattern in patterns]
    return tuple(spec for spec in RUNS if any(pattern in spec.label.lower() for pattern in lowered))


def build_experiment_config(
    base_config: dict[str, Any],
    spec: RunSpec,
    *,
    device: str | None,
    epochs: int | None,
    patience: int | None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    slug = spec.label.split("_", 1)[1]

    experiment = dict(config.get("experiment", {}))
    experiment["name"] = f"brats2020-{slug}"
    config["experiment"] = experiment

    task = dict(config.get("task", {}))
    task["loss"] = spec.loss_spec.loss
    task["loss_gamma"] = spec.loss_spec.gamma
    task["loss_balance_param"] = spec.loss_spec.balance_param
    task["loss_lambda_alpha"] = 0.0
    task["loss_lambda_beta"] = 0.0
    config["task"] = task

    trainer = dict(config.get("trainer", {}))
    if device is not None:
        trainer["device"] = device
    if epochs is not None:
        trainer["epochs"] = epochs
    if patience is not None:
        trainer["patience"] = patience
    config["trainer"] = trainer
    return config


def require_inputs(base_config: Path, runs: tuple[RunSpec, ...]) -> None:
    missing = [str(path) for path in [base_config, *(spec.model_config for spec in runs)] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(missing))


def run_training(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training command failed with exit code {return_code}: {' '.join(command)}")


def main() -> int:
    args = parse_args()
    runs = selected_runs(args.only)
    if not runs:
        raise ValueError("No run specs matched --only filters")

    base_config = Path(args.base_config)
    require_inputs(base_config, runs)
    base = load_toml(base_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(args.snapshot_root) / f"original_loss_model_sequence_{timestamp}" / "configs"

    planned: list[tuple[RunSpec, Path, Path]] = []
    for spec in runs:
        frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
        frozen_model = snapshot_dir / f"{spec.label}.model.toml"
        planned.append((spec, frozen_experiment, frozen_model))

    if args.dry_run:
        print(f"Frozen configs would be written to: {snapshot_dir}")
        for spec, frozen_experiment, frozen_model in planned:
            print(
                f"{spec.label}: loss={spec.loss_spec.loss} "
                f"model={spec.model_config} command={' '.join(framework_train_command(frozen_experiment, frozen_model))}"
            )
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    print(f"Frozen configs written to: {snapshot_dir}")

    for spec, frozen_experiment, frozen_model in planned:
        experiment_config = build_experiment_config(
            base,
            spec,
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
        )
        write_toml(experiment_config, frozen_experiment)
        shutil.copy2(spec.model_config, frozen_model)

        print(
            f"\n{'=' * 80}\n{spec.label}\n"
            f"loss={spec.loss_spec.loss}\nmodel={spec.model_config}\n{'=' * 80}"
        )
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print("All requested original-loss comparison runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
