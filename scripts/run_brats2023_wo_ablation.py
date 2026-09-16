"""Run Dice-only DG-CMFNet V2 ablations on the original BraTS2023 subject split."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli
import tomli_w

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = Path("configs/experiment.brats.toml")
MODEL_CONFIGS = {
    "wo_fdfm": Path("configs/model.dgcmfnet_V2.wo_fdfm.toml"),
    "wo_gfm": Path("configs/model.dgcmfnet_V2.wo_gfm.toml"),
    "wo_gim": Path("configs/model.dgcmfnet_V2.wo_gim.toml"),
    "wo_all": Path("configs/model.dgcmfnet_V2.wo_all.toml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT), help="Base BraTS2023 experiment config.")
    parser.add_argument("--gpu", default="2", help="CUDA device index written to each frozen experiment config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override [trainer].epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override [trainer].patience.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="Run only the selected variants, in the order given.",
    )
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for frozen configs and logs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without writing files.")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomli.load(file)


def write_toml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        tomli_w.dump(config, file)


def training_command(experiment_path: Path, model_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "framework.cli.main",
        "train",
        "--configs",
        str(experiment_path),
        "--model",
        str(model_path),
    ]


def configure_experiment(base: dict[str, Any], *, label: str, gpu: str, epochs: int | None, patience: int | None) -> dict[str, Any]:
    experiment = dict(base)

    experiment_section = dict(experiment.get("experiment", {}))
    base_name = str(experiment_section.get("name", "brats2023-ablation"))
    experiment_section["name"] = f"{base_name}-dice-{label.replace('_', '-')}"
    experiment_section["seed"] = 37
    experiment["experiment"] = experiment_section

    data = dict(experiment.get("data", {}))
    for fixed_list_key in ("train_ids_file", "val_ids_file", "test_ids_file"):
        data.pop(fixed_list_key, None)
    data["val_ratio"] = 0.20
    data["test_ratio"] = 0.0
    data["seed"] = 37
    experiment["data"] = data

    task = dict(experiment.get("task", {}))
    task["loss"] = "uahl"
    task["loss_lambda_alpha"] = 0.0
    task["loss_lambda_beta"] = 0.0
    experiment["task"] = task

    trainer = dict(experiment.get("trainer", {}))
    trainer["device"] = f"cuda:{gpu}"
    if epochs is not None:
        trainer["epochs"] = epochs
    if patience is not None:
        trainer["patience"] = patience
    experiment["trainer"] = trainer
    return experiment


def run_training(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print(f"Log: {log_path}")
    print("=" * 80)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Training failed with exit code {return_code}; see {log_path}")


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    experiment_source = Path(args.config)
    selected_models = [(label, MODEL_CONFIGS[label]) for label in args.only]
    missing = [path for path in (experiment_source, *(path for _, path in selected_models)) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(str(path) for path in missing))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_dir = Path(args.snapshot_root) / f"brats2023_wo_dice_ablation_gpu{args.gpu}_{timestamp}"
    config_dir = sequence_dir / "configs"
    log_dir = sequence_dir / "logs"

    run_paths: list[tuple[str, Path, Path]] = []
    for index, (label, model_source) in enumerate(selected_models, start=1):
        frozen_experiment = config_dir / f"{index:02d}_{label}.experiment.toml"
        frozen_model = config_dir / f"{index:02d}_{label}.model.toml"
        run_paths.append((label, frozen_experiment, frozen_model))

    if args.dry_run:
        print(f"Trainer device: cuda:{args.gpu}")
        print(f"Base experiment: {experiment_source}")
        print("Split: subject-wise, seed=37, val_ratio=0.20, test_ratio=0.0")
        print("Loss: Dice only (UAHL with lambda_alpha=0 and lambda_beta=0)")
        print(f"Frozen configs would be written to: {config_dir}")
        for label, frozen_experiment, frozen_model in run_paths:
            print(f"{label}: {' '.join(training_command(frozen_experiment, frozen_model))}")
        return 0

    base_experiment = load_toml(experiment_source)
    config_dir.mkdir(parents=True, exist_ok=False)

    for (label, model_source), (_, frozen_experiment, frozen_model) in zip(selected_models, run_paths):
        experiment = configure_experiment(
            base_experiment,
            label=label,
            gpu=args.gpu,
            epochs=args.epochs,
            patience=args.patience,
        )

        write_toml(experiment, frozen_experiment)
        shutil.copy2(model_source, frozen_model)

    print(f"Frozen configs written to: {config_dir}")
    for label, frozen_experiment, frozen_model in run_paths:
        run_training(training_command(frozen_experiment, frozen_model), log_dir / f"{label}.log")

    print(f"\nAll requested Dice-only BraTS2023 w/o ablations completed. Sequence artifacts: {sequence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
