"""Run the original-loss comparison models that have not been completed yet.

This is a narrow wrapper around scripts/run_original_loss_models.py. It keeps
the same config-freezing behavior and loss overrides, but only schedules the
remaining models from the first original-loss sequence:

- U-Net
- ResU-Net
- Dense U-Net
- TransBTS
- NestedFormer
- Slim UNETR
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from run_original_loss_models import (
    RUNS,
    RunSpec,
    build_experiment_config,
    framework_train_command,
    load_toml,
    require_inputs,
    run_training,
    write_toml,
)

REMAINING_LABELS = (
    "01_unet_original_dc_ce",
    "02_resunet_original_dc_ce",
    "03_dense_unet_original_dc_ce",
    "07_transbts_original_softmax_dice",
    "08_nestedformer_original_sigmoid_region_dice",
    "10_slim_unetr_original_focal_dice",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/experiment.brats2020.dice_only.toml",
        help="Experiment TOML used as the data/trainer template.",
    )
    parser.add_argument("--device", default=None, help='Override [trainer].device, for example "cuda:2".')
    parser.add_argument("--epochs", type=int, default=None, help="Override [trainer].epochs for every run.")
    parser.add_argument("--patience", type=int, default=None, help="Override [trainer].patience for every run.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without launching training.")
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for frozen configs.")
    parser.add_argument(
        "--only",
        choices=REMAINING_LABELS,
        nargs="+",
        default=None,
        help="Run an exact subset of the remaining specs.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip specs that already have a matching run with metrics through the requested epoch count.",
    )
    return parser.parse_args()


def remaining_runs(exact_labels: list[str] | None) -> tuple[RunSpec, ...]:
    selected_labels = set(exact_labels or REMAINING_LABELS)
    runs_by_label = {spec.label: spec for spec in RUNS}
    missing = [label for label in selected_labels if label not in runs_by_label]
    if missing:
        raise KeyError(f"Unknown run labels: {missing}")
    return tuple(runs_by_label[label] for label in REMAINING_LABELS if label in selected_labels)


def model_name_from_config(model_config: Path) -> str:
    config = load_toml(model_config)
    model = config.get("model", {})
    name = model.get("name")
    if not isinstance(name, str):
        raise ValueError(f"{model_config} does not define [model].name")
    return name


def last_epoch(metrics_path: Path) -> int | None:
    if not metrics_path.is_file():
        return None
    last_line = ""
    with metrics_path.open() as f:
        for line in f:
            if line.strip():
                last_line = line
    if not last_line:
        return None

    import json

    data = json.loads(last_line)
    epoch = data.get("epoch")
    return int(epoch) if isinstance(epoch, int) else None


def has_completed_run(spec: RunSpec, *, expected_epochs: int, output_root: Path) -> bool:
    slug = spec.label.split("_", 1)[1]
    model_name = model_name_from_config(spec.model_config)
    pattern = f"brats2020-{slug}_{model_name}_*"
    for run_dir in output_root.glob(pattern):
        epoch = last_epoch(run_dir / "metrics.jsonl")
        if epoch is not None and epoch >= expected_epochs:
            return True
    return False


def main() -> int:
    args = parse_args()
    runs = remaining_runs(args.only)
    if not runs:
        raise ValueError("No remaining runs selected")

    base_config = Path(args.base_config)
    require_inputs(base_config, runs)
    base = load_toml(base_config)
    expected_epochs = int(args.epochs or base.get("trainer", {}).get("epochs", 200))
    output_root = Path(base.get("output", {}).get("root_dir", "runs"))

    if args.skip_completed:
        runs = tuple(
            spec for spec in runs if not has_completed_run(spec, expected_epochs=expected_epochs, output_root=output_root)
        )
        if not runs:
            print("All selected remaining runs already have completed metrics.")
            return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(args.snapshot_root) / f"remaining_original_loss_model_sequence_{timestamp}" / "configs"
    planned = []
    for spec in runs:
        frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
        frozen_model = snapshot_dir / f"{spec.label}.model.toml"
        planned.append((spec, frozen_experiment, frozen_model))

    if args.dry_run:
        print(f"Frozen configs would be written to: {snapshot_dir}")
        for spec, frozen_experiment, frozen_model in planned:
            command = framework_train_command(frozen_experiment, frozen_model)
            print(f"{spec.label}: loss={spec.loss_spec.loss} model={spec.model_config} command={' '.join(command)}")
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

        print(f"\n{'=' * 80}\n{spec.label}\nloss={spec.loss_spec.loss}\nmodel={spec.model_config}\n{'=' * 80}")
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print("All remaining original-loss comparison runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
