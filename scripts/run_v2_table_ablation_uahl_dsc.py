"""Run the DG-CMFNet V2 table ablations with U-AHL loss for every model.

Order:
1. w/o all, U-AHL
2. w/o FG-GIM, U-AHL
3. w/o CG-GFM, U-AHL
4. w/o FDFM, U-AHL
5. Ours, U-AHL

HD/HD95 is intentionally not computed here. This script mirrors
run_v2_table_ablation_dsc.py, but forces every architecture to use the full
U-AHL loss settings: loss="uahl", loss_lambda_alpha=1.0,
loss_lambda_beta=0.05, loss_gamma=2.0.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import run_v2_table_ablation_dsc as table

RUNS = (
    table.RunSpec("w/o all", "wo_all", table.UAHL, Path("configs/model.dgcmfnet_V2.wo_all.toml")),
    table.RunSpec("w/o FG-GIM", "wo_fg_gim", table.UAHL, Path("configs/model.dgcmfnet_V2.wo_gim.toml")),
    table.RunSpec("w/o CG-GFM", "wo_cg_gfm", table.UAHL, Path("configs/model.dgcmfnet_V2.wo_gfm.toml")),
    table.RunSpec("w/o FDFM", "wo_fdfm", table.UAHL, Path("configs/model.dgcmfnet_V2.wo_fdfm.toml")),
    table.RunSpec("Ours", "ours", table.UAHL, Path("configs/model.dgcmfnet_V2.toml")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/experiment.brats2020.dice_only.toml",
        help=(
            "Experiment TOML used as the data/trainer template for every run. "
            "The task loss fields are overwritten to full U-AHL."
        ),
    )
    parser.add_argument("--device", default=None, help='Override [trainer].device, for example "cuda:3".')
    parser.add_argument("--epochs", type=int, default=None, help="Override [trainer].epochs for all runs.")
    parser.add_argument("--patience", type=int, default=None, help="Override [trainer].patience for all runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned commands only.")
    parser.add_argument("--snapshot-root", default="runs", help="Root directory for this ablation sequence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config_path = Path(args.base_config)
    table.require_inputs(base_config_path, RUNS)

    base_config = table.load_toml(base_config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_dir = Path(args.snapshot_root) / f"v2_table_ablation_uahl_dsc_{timestamp}"
    config_dir = sequence_dir / "configs"

    planned = []
    for index, spec in enumerate(RUNS, start=1):
        label = f"{index:02d}_{spec.slug}_{spec.loss_spec.label}"
        frozen_experiment = config_dir / f"{label}.experiment.toml"
        frozen_model = config_dir / f"{label}.model.toml"
        runtime_config = table.build_experiment_config(
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
            command = table.framework_train_command(frozen_experiment, frozen_model)
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
        table.write_toml(runtime_config, frozen_experiment)
        shutil.copy2(spec.model_config, frozen_model)

        print(
            f"\n{'=' * 80}\n"
            f"{spec.method} ({spec.loss_spec.label})\n"
            f"experiment={frozen_experiment}\n"
            f"model={frozen_model}\n"
            f"{'=' * 80}"
        )
        run_dir = table.run_training(table.framework_train_command(frozen_experiment, frozen_model))
        rows.append(table.summarize_run(spec, run_dir))
        table.write_summary(sequence_dir, rows)
        print(f"Partial DSC summary written to: {sequence_dir}")

    print("\nAll DG-CMFNet V2 U-AHL ablation DSC runs completed.")
    print(f"Summary: {sequence_dir / 'ablation_dsc_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
