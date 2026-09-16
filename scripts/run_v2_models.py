"""Run DG-CMFNet V2 and baseline V2 model configs sequentially."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MODELS = [
    Path("configs/model.dgcmfnet_V2.toml"),
    Path("configs/model.baseline_V2.toml"),
]


def parse_args(default_models: list[Path], description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment.brats2020.toml",
        help="Experiment config TOML passed to each training run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(path) for path in default_models],
        help="Model config TOMLs to run in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned frozen config paths and commands without launching training.",
    )
    parser.add_argument(
        "--snapshot-root",
        default="runs",
        help="Root directory where frozen config copies are written.",
    )
    return parser.parse_args()


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


def frozen_config_paths(snapshot_dir: Path, config_path: Path, model_paths: list[Path]) -> tuple[Path, list[Path]]:
    frozen_experiment = snapshot_dir / config_path.name
    frozen_models = [snapshot_dir / f"{index:02d}_{model_path.name}" for index, model_path in enumerate(model_paths, start=1)]
    return frozen_experiment, frozen_models


def freeze_configs(config_path: Path, model_paths: list[Path], snapshot_dir: Path) -> tuple[Path, list[Path]]:
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    frozen_experiment, frozen_models = frozen_config_paths(snapshot_dir, config_path, model_paths)

    shutil.copy2(config_path, frozen_experiment)
    for source_path, frozen_path in zip(model_paths, frozen_models):
        shutil.copy2(source_path, frozen_path)

    return frozen_experiment, frozen_models


def run_sequence(
    default_models: list[Path],
    *,
    description: str | None = None,
    snapshot_name: str = "v2_model_sequence",
    completion_message: str = "All requested V2 training runs completed.",
) -> int:
    args = parse_args(default_models, description=description)
    config_path = Path(args.config)
    model_paths = [Path(model) for model in args.models]

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(args.snapshot_root) / f"{snapshot_name}_{timestamp}" / "configs"
    if args.dry_run:
        frozen_config_path, frozen_model_paths = frozen_config_paths(snapshot_dir, config_path, model_paths)
        commands = [framework_train_command(frozen_config_path, model_path) for model_path in frozen_model_paths]
        print(f"frozen configs would be written to: {snapshot_dir}")
        for index, command in enumerate(commands, start=1):
            print(f"{index}: {' '.join(command)}")
        return 0

    frozen_config_path, frozen_model_paths = freeze_configs(config_path, model_paths, snapshot_dir)
    print(f"Frozen configs written to: {snapshot_dir}")

    commands = [framework_train_command(frozen_config_path, model_path) for model_path in frozen_model_paths]
    for command in commands:
        run_training(command)

    print(f"\n{completion_message}")
    return 0


def main() -> int:
    return run_sequence(DEFAULT_MODELS)


if __name__ == "__main__":
    raise SystemExit(main())
