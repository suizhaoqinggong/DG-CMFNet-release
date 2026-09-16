"""Run the requested V2 ablations with their designated loss functions.

Order:
1. FG-GIM only, Dice only
2. FGFM only, Dice only
3. w/o FG-GIM, U-AHL
4. w/o CG-GFM, U-AHL
5. w/o FDFM, U-AHL

Every experiment and model TOML is copied to a timestamped snapshot before
training, so the recorded loss and architecture remain traceable.
"""

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


DICE_EXPERIMENT = Path("configs/experiment.brats2020.dice_only.toml")
UAHL_EXPERIMENT = Path("configs/experiment.brats2020.toml")

RUNS = (
    RunSpec("01_gim_only_dice", DICE_EXPERIMENT, Path("configs/model.dgcmfnet_V2.fggim_only.toml")),
    RunSpec("02_fgfm_only_dice", DICE_EXPERIMENT, Path("configs/model.dgcmfnet_V2.fgfm_only.toml")),
    RunSpec("03_wo_gim_uahl", UAHL_EXPERIMENT, Path("configs/model.dgcmfnet_V2.wo_gim.toml")),
    RunSpec("04_wo_gfm_uahl", UAHL_EXPERIMENT, Path("configs/model.dgcmfnet_V2.wo_gfm.toml")),
    RunSpec("05_wo_fdfm_uahl", UAHL_EXPERIMENT, Path("configs/model.dgcmfnet_V2.wo_fdfm.toml")),
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
    snapshot_dir = Path(args.snapshot_root) / f"v2_mixed_loss_ablation_{timestamp}" / "configs"

    if args.dry_run:
        print(f"Frozen configs would be written to: {snapshot_dir}")
        for spec in RUNS:
            print(f"{spec.label}: {spec.experiment_config} + {spec.model_config}")
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    print(f"Frozen configs written to: {snapshot_dir}")

    for spec in RUNS:
        frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
        frozen_model = snapshot_dir / f"{spec.label}.model.toml"
        shutil.copy2(spec.experiment_config, frozen_experiment)
        shutil.copy2(spec.model_config, frozen_model)

        print(f"\n{'=' * 80}\n{spec.label}\nexperiment={spec.experiment_config}\nmodel={spec.model_config}\n{'=' * 80}")
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print("All requested mixed-loss V2 ablations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
