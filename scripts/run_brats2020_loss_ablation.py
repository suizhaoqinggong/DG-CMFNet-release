"""Run DG-CMFNet loss ablations on BraTS2020 sequentially.

Order: Dice, cross-entropy, focal, and full U-AHL. Dice-only and focal-only
reuse the corresponding terms from U-AHL so the ablation definitions remain
directly comparable.
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
from typing import Any, Optional

import tomli
import tomli_w


@dataclass(frozen=True)
class LossSpec:
    label: str
    slug: str
    loss: str
    lambda_alpha: float
    lambda_beta: float
    gamma: float = 2.0


LOSSES = (
    LossSpec("L_Dice", "dice", "uahl", lambda_alpha=0.0, lambda_beta=0.0),
    LossSpec("L_CE", "ce", "ce", lambda_alpha=0.0, lambda_beta=0.0),
    LossSpec("L_Focal", "focal", "focal", lambda_alpha=0.0, lambda_beta=0.0),
    LossSpec("L_UAHL", "uahl", "uahl", lambda_alpha=1.0, lambda_beta=0.05),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/experiment.brats2020.toml",
        help="BraTS2020 experiment config used as the common training template.",
    )
    parser.add_argument(
        "--model",
        default="configs/model.dgcmfnet_V2.toml",
        help="Model config used unchanged for all four loss runs.",
    )
    parser.add_argument("--device", default=None, help='Override [trainer].device, for example "cuda:0".')
    parser.add_argument("--epochs", type=int, default=None, help="Override [trainer].epochs for all runs.")
    parser.add_argument("--patience", type=int, default=None, help="Override [trainer].patience for all runs.")
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for frozen ablation configs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the four planned runs without training.")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)


def write_toml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def build_config(
    base_config: dict[str, Any],
    loss_spec: LossSpec,
    *,
    device: Optional[str],
    epochs: Optional[int],
    patience: Optional[int],
) -> dict[str, Any]:
    config = deepcopy(base_config)

    experiment = dict(config.get("experiment", {}))
    experiment["name"] = f"brats2020-loss-ablation-{loss_spec.slug}"
    config["experiment"] = experiment

    task = dict(config.get("task", {}))
    task.update(
        {
            "loss": loss_spec.loss,
            "loss_lambda_alpha": loss_spec.lambda_alpha,
            "loss_lambda_beta": loss_spec.lambda_beta,
            "loss_gamma": loss_spec.gamma,
        }
    )
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


def framework_train_command(experiment_config: Path, model_config: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "framework.cli.main",
        "train",
        "--configs",
        str(experiment_config),
        "--model",
        str(model_config),
    ]


def run_training(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, check=True, env=env)


def main() -> int:
    args = parse_args()
    base_config_path = Path(args.base_config)
    model_config_path = Path(args.model)
    missing = [str(path) for path in (base_config_path, model_config_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(missing))

    base_config = load_toml(base_config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_dir = Path(args.snapshot_root) / f"brats2020_loss_ablation_{timestamp}"
    config_dir = sequence_dir / "configs"
    frozen_model = config_dir / model_config_path.name

    planned: list[tuple[LossSpec, dict[str, Any], Path]] = []
    for index, loss_spec in enumerate(LOSSES, start=1):
        frozen_experiment = config_dir / f"{index:02d}_{loss_spec.slug}.experiment.toml"
        runtime_config = build_config(
            base_config,
            loss_spec,
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
        )
        planned.append((loss_spec, runtime_config, frozen_experiment))

    if args.dry_run:
        print(f"Sequence directory would be: {sequence_dir}")
        for loss_spec, runtime_config, frozen_experiment in planned:
            task = runtime_config["task"]
            trainer = runtime_config["trainer"]
            print(
                f"{loss_spec.label}: loss={task['loss']} alpha={task['loss_lambda_alpha']} "
                f"beta={task['loss_lambda_beta']} gamma={task['loss_gamma']} device={trainer.get('device')}"
            )
            print("  " + " ".join(framework_train_command(frozen_experiment, frozen_model)))
        return 0

    config_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model_config_path, frozen_model)
    print(f"Frozen configs: {config_dir}")

    for loss_spec, runtime_config, frozen_experiment in planned:
        write_toml(runtime_config, frozen_experiment)
        print(f"\nStarting {loss_spec.label} with {frozen_experiment}")
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print(f"\nAll BraTS2020 loss ablations completed. Frozen configs: {config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
