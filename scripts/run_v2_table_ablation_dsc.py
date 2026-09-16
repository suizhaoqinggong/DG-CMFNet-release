"""Run the DG-CMFNet V2 table ablations and summarize validation DSC only.

Order:
1. w/o all, Dice-only
2. w/o FG-GIM, Dice-only
3. w/o CG-GFM, Dice-only
4. w/o FDFM, Dice-only
5. Ours, U-AHL

HD/HD95 is intentionally not computed here. Dice-only is implemented as U-AHL
with both auxiliary weights set to zero, matching the existing project config.
"""

from __future__ import annotations

import argparse
import csv
import json
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
import torch


@dataclass(frozen=True)
class LossSpec:
    label: str
    loss: str
    lambda_alpha: float
    lambda_beta: float
    gamma: float = 2.0


@dataclass(frozen=True)
class RunSpec:
    method: str
    slug: str
    loss_spec: LossSpec
    model_config: Path


DICE_ONLY = LossSpec("dice_only", "uahl", lambda_alpha=0.0, lambda_beta=0.0)
UAHL = LossSpec("uahl", "uahl", lambda_alpha=1.0, lambda_beta=0.05)

RUNS = (
    RunSpec("w/o all", "wo_all", DICE_ONLY, Path("configs/model.dgcmfnet_V2.wo_all.toml")),
    RunSpec("w/o FG-GIM", "wo_fg_gim", DICE_ONLY, Path("configs/model.dgcmfnet_V2.wo_gim.toml")),
    RunSpec("w/o CG-GFM", "wo_cg_gfm", DICE_ONLY, Path("configs/model.dgcmfnet_V2.wo_gfm.toml")),
    RunSpec("w/o FDFM", "wo_fdfm", DICE_ONLY, Path("configs/model.dgcmfnet_V2.wo_fdfm.toml")),
    RunSpec("Ours", "ours", UAHL, Path("configs/model.dgcmfnet_V2.toml")),
)

DICE_KEYS = ("wt", "tc", "et", "mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/experiment.brats2020.dice_only.toml",
        help="Experiment TOML used as the data/trainer template for every run.",
    )
    parser.add_argument("--device", default=None, help='Override [trainer].device, for example "cuda:3".')
    parser.add_argument("--epochs", type=int, default=None, help="Override [trainer].epochs for all runs.")
    parser.add_argument("--patience", type=int, default=None, help="Override [trainer].patience for all runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned commands only.")
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for this ablation sequence.")
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


def require_inputs(base_config: Path, runs: tuple[RunSpec, ...]) -> None:
    missing = [str(path) for path in [base_config, *(spec.model_config for spec in runs)] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(missing))


def build_experiment_config(
    base_config: dict[str, Any],
    spec: RunSpec,
    *,
    device: Optional[str],
    epochs: Optional[int],
    patience: Optional[int],
) -> dict[str, Any]:
    config = deepcopy(base_config)

    experiment_config = dict(config.get("experiment", {}))
    experiment_config["name"] = f"brats2020-ablation-{spec.slug}"
    config["experiment"] = experiment_config

    task_config = dict(config.get("task", {}))
    task_config["loss"] = spec.loss_spec.loss
    task_config["loss_lambda_alpha"] = spec.loss_spec.lambda_alpha
    task_config["loss_lambda_beta"] = spec.loss_spec.lambda_beta
    task_config["loss_gamma"] = spec.loss_spec.gamma
    config["task"] = task_config

    trainer_config = dict(config.get("trainer", {}))
    if device is not None:
        trainer_config["device"] = device
    if epochs is not None:
        trainer_config["epochs"] = epochs
    if patience is not None:
        trainer_config["patience"] = patience
    config["trainer"] = trainer_config

    return config


def run_training(command: list[str]) -> Path:
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
    run_dir: Optional[Path] = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        marker = "Run outputs saved to:"
        if marker in line:
            run_dir = Path(line.split(marker, 1)[1].strip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training command failed with exit code {return_code}: {' '.join(command)}")
    if run_dir is None:
        raise RuntimeError("Training finished but the run directory was not found in framework output")
    return run_dir


def load_best_metrics(run_dir: Path) -> tuple[int, dict[str, float]]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError(f"Checkpoint metrics must be a dict: {checkpoint_path}")
    return int(payload["epoch"]), {str(key): float(value) for key, value in metrics.items()}


def summarize_run(spec: RunSpec, run_dir: Path) -> dict[str, Any]:
    best_epoch, metrics = load_best_metrics(run_dir)
    dice_fraction = {key: metrics[f"brats_dice_{key}"] for key in DICE_KEYS}
    dice_percent = {key: value * 100.0 for key, value in dice_fraction.items()}
    return {
        "method": spec.method,
        "slug": spec.slug,
        "loss": spec.loss_spec.label,
        "model_config": str(spec.model_config),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "dice_fraction": dice_fraction,
        "dice_percent": dice_percent,
    }


def write_summary(sequence_dir: Path, rows: list[dict[str, Any]]) -> None:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    (sequence_dir / "ablation_dsc_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    csv_path = sequence_dir / "ablation_dsc_summary.csv"
    fieldnames = [
        "method",
        "loss",
        "best_epoch",
        "dsc_wt_percent",
        "dsc_tc_percent",
        "dsc_et_percent",
        "dsc_mean_percent",
        "run_dir",
        "model_config",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            dice = row["dice_percent"]
            writer.writerow(
                {
                    "method": row["method"],
                    "loss": row["loss"],
                    "best_epoch": row["best_epoch"],
                    "dsc_wt_percent": f"{dice['wt']:.4f}",
                    "dsc_tc_percent": f"{dice['tc']:.4f}",
                    "dsc_et_percent": f"{dice['et']:.4f}",
                    "dsc_mean_percent": f"{dice['mean']:.4f}",
                    "run_dir": row["run_dir"],
                    "model_config": row["model_config"],
                }
            )

    lines = [
        "# DG-CMFNet V2 Ablation DSC Summary",
        "",
        "HD/HD95 was not measured by this script.",
        "",
        "| Method | Loss | WT DSC (%) | TC DSC (%) | ET DSC (%) | Mean DSC (%) | Best Epoch | Run Dir |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        dice = row["dice_percent"]
        lines.append(
            "| {method} | {loss} | {wt:.2f} | {tc:.2f} | {et:.2f} | {mean:.2f} | {epoch} | `{run_dir}` |".format(
                method=row["method"],
                loss=row["loss"],
                wt=dice["wt"],
                tc=dice["tc"],
                et=dice["et"],
                mean=dice["mean"],
                epoch=row["best_epoch"],
                run_dir=row["run_dir"],
            )
        )
    (sequence_dir / "ablation_dsc_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_config_path = Path(args.base_config)
    require_inputs(base_config_path, RUNS)

    base_config = load_toml(base_config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_dir = Path(args.snapshot_root) / f"v2_table_ablation_dsc_{timestamp}"
    config_dir = sequence_dir / "configs"

    planned = []
    for index, spec in enumerate(RUNS, start=1):
        label = f"{index:02d}_{spec.slug}_{spec.loss_spec.label}"
        frozen_experiment = config_dir / f"{label}.experiment.toml"
        frozen_model = config_dir / f"{label}.model.toml"
        runtime_config = build_experiment_config(
            base_config,
            spec,
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
        )
        planned.append((spec, runtime_config, frozen_experiment, frozen_model))

    if args.dry_run:
        print(f"Sequence directory would be: {sequence_dir}")
        for spec, runtime_config, frozen_experiment, frozen_model in planned:
            command = framework_train_command(frozen_experiment, frozen_model)
            task = runtime_config["task"]
            trainer = runtime_config.get("trainer", {})
            print(
                f"{spec.method}: loss={spec.loss_spec.label} "
                f"alpha={task['loss_lambda_alpha']} beta={task['loss_lambda_beta']} "
                f"device={trainer.get('device')} model={spec.model_config}"
            )
            print("  " + " ".join(command))
        return 0

    rows: list[dict[str, Any]] = []
    for spec, runtime_config, frozen_experiment, frozen_model in planned:
        write_toml(runtime_config, frozen_experiment)
        shutil.copy2(spec.model_config, frozen_model)

        print(
            f"\n{'=' * 80}\n"
            f"{spec.method} ({spec.loss_spec.label})\n"
            f"experiment={frozen_experiment}\n"
            f"model={frozen_model}\n"
            f"{'=' * 80}"
        )
        run_dir = run_training(framework_train_command(frozen_experiment, frozen_model))
        rows.append(summarize_run(spec, run_dir))
        write_summary(sequence_dir, rows)
        print(f"Partial DSC summary written to: {sequence_dir}")

    print("\nAll DG-CMFNet V2 ablation DSC runs completed.")
    print(f"Summary: {sequence_dir / 'ablation_dsc_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
