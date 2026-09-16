"""Run FG-GIM-only with Dice loss, then full DG-CMFNet V2 with U-AHL."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from run_v2_models import framework_train_command, run_training


@dataclass(frozen=True)
class RunSpec:
    label: str
    experiment_config: Path
    model_config: Path


RUNS = (
    RunSpec(
        "01_gim_only_dice",
        Path("configs/experiment.brats2020.dice_only.toml"),
        Path("configs/model.dgcmfnet_V2.fggim_only.toml"),
    ),
    RunSpec(
        "02_ours_uahl",
        Path("configs/experiment.brats2020.toml"),
        Path("configs/model.dgcmfnet_V2.toml"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned commands only.")
    parser.add_argument("--snapshot-root", default="runs", help="Directory containing frozen configuration snapshots.")
    return parser.parse_args()


def require_inputs() -> None:
    missing = [
        str(path)
        for spec in RUNS
        for path in (spec.experiment_config, spec.model_config)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    require_inputs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(args.snapshot_root) / f"v2_gim_ours_sequence_{timestamp}" / "configs"

    if args.dry_run:
        print(f"Frozen configs would be written to: {snapshot_dir}")
        for spec in RUNS:
            frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
            frozen_model = snapshot_dir / f"{spec.label}.model.toml"
            command = framework_train_command(frozen_experiment, frozen_model)
            print(f"{spec.label}: {' '.join(command)}")
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    print(f"Frozen configs written to: {snapshot_dir}")

    for spec in RUNS:
        frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
        frozen_model = snapshot_dir / f"{spec.label}.model.toml"
        shutil.copy2(spec.experiment_config, frozen_experiment)
        shutil.copy2(spec.model_config, frozen_model)

        print(
            f"\n{'=' * 80}\n"
            f"{spec.label}\n"
            f"experiment={spec.experiment_config}\n"
            f"model={spec.model_config}\n"
            f"{'=' * 80}"
        )
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print("FG-GIM-only Dice and full DG-CMFNet V2 U-AHL training runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
