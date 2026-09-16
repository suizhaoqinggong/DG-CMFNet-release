"""Run FG-GIM-only Dice ablations over FG-GIM edge modes sequentially.

Default setting:
- experiment: configs/experiment.brats2020.dice_only.toml
- model template: configs/model.dgcmfnet_V2.fggim_only.toml
- changed parameter: model.fg_gim_edge_mode
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w
from run_v2_models import framework_train_command, run_training

DEFAULT_EXPERIMENT = Path("configs/experiment.brats2020.dice_only.toml")
DEFAULT_MODEL = Path("configs/model.dgcmfnet_V2.fggim_only.toml")
DEFAULT_EDGE_MODES = ("all", "no_self", "cross_modal", "aligned_cross")
VALID_EDGE_MODES = frozenset(DEFAULT_EDGE_MODES)


@dataclass(frozen=True)
class RunSpec:
    label: str
    edge_mode: str
    experiment_config: Path
    model_config: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_EXPERIMENT),
        help="Dice-only experiment config TOML.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="FG-GIM-only model config TOML used as the template.",
    )
    parser.add_argument(
        "--edge-modes",
        nargs="+",
        default=list(DEFAULT_EDGE_MODES),
        choices=sorted(VALID_EDGE_MODES),
        help="FG-GIM edge_mode values to run in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned commands only.",
    )
    parser.add_argument(
        "--snapshot-root",
        default="runs",
        help="Directory containing frozen configuration snapshots.",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def write_toml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def require_inputs(experiment_config: Path, model_config: Path) -> None:
    missing = [str(path) for path in (experiment_config, model_config) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing configuration file(s): " + ", ".join(missing))

    experiment = load_toml(experiment_config)
    task = experiment.get("task", {})
    if task.get("loss") != "uahl" or float(task.get("loss_lambda_alpha", -1.0)) != 0.0 or float(
        task.get("loss_lambda_beta", -1.0)
    ) != 0.0:
        raise ValueError(
            "Expected Dice-only experiment config: task.loss='uahl', "
            "loss_lambda_alpha=0.0, loss_lambda_beta=0.0"
        )

    model = load_toml(model_config)
    model_section = model.get("model", {})
    if model_section.get("use_cg_gfm") is not False:
        raise ValueError("Expected GIM-only model config with model.use_cg_gfm=false")
    if not model_section.get("fg_gim_stages"):
        raise ValueError("Expected GIM-only model config with non-empty model.fg_gim_stages")
    if model_section.get("decoder_frequency_position") is not None:
        raise ValueError("Expected GIM-only model config without decoder frequency guidance")


def make_specs(experiment_config: Path, model_config: Path, edge_modes: list[str]) -> list[RunSpec]:
    return [
        RunSpec(
            label=f"{index:02d}_gimonly_dice_edge_{edge_mode}",
            edge_mode=edge_mode,
            experiment_config=experiment_config,
            model_config=model_config,
        )
        for index, edge_mode in enumerate(edge_modes, start=1)
    ]


def freeze_model_with_edge_mode(source: Path, destination: Path, edge_mode: str) -> None:
    model_config = load_toml(source)
    model_section = model_config.setdefault("model", {})
    model_section["fg_gim_edge_mode"] = edge_mode
    write_toml(destination, model_config)


def main() -> int:
    args = parse_args()
    experiment_config = Path(args.config)
    model_config = Path(args.model)
    edge_modes = list(args.edge_modes)

    require_inputs(experiment_config, model_config)
    specs = make_specs(experiment_config, model_config, edge_modes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(args.snapshot_root) / f"v2_gimonly_edge_mode_ablation_{timestamp}" / "configs"

    if args.dry_run:
        print(f"Frozen configs would be written to: {snapshot_dir}")
        for spec in specs:
            frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
            frozen_model = snapshot_dir / f"{spec.label}.model.toml"
            command = framework_train_command(frozen_experiment, frozen_model)
            print(f"{spec.label}: edge_mode={spec.edge_mode}")
            print("  " + " ".join(command))
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    print(f"Frozen configs written to: {snapshot_dir}")

    for spec in specs:
        frozen_experiment = snapshot_dir / f"{spec.label}.experiment.toml"
        frozen_model = snapshot_dir / f"{spec.label}.model.toml"
        shutil.copy2(spec.experiment_config, frozen_experiment)
        freeze_model_with_edge_mode(spec.model_config, frozen_model, spec.edge_mode)

        print(
            f"\n{'=' * 80}\n"
            f"{spec.label}\n"
            f"experiment={spec.experiment_config}\n"
            f"model_template={spec.model_config}\n"
            f"fg_gim_edge_mode={spec.edge_mode}\n"
            f"{'=' * 80}"
        )
        run_training(framework_train_command(frozen_experiment, frozen_model))

    print("All FG-GIM-only Dice edge-mode ablations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
