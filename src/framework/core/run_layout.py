"""Run directory layout and experiment tracking."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w


@dataclass(frozen=True)
class RunLayout:
    """Paths for a single experiment run."""

    run_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    plot_dir: Path


def layout_from_run_dir(run_dir: str | Path) -> RunLayout:
    """Create a RunLayout for an existing or preselected run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    plot_dir = run_dir / "plots"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    return RunLayout(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        plot_dir=plot_dir,
    )


def resolve_run_layout(
    experiment_name: str,
    model_name: str,
    root_dir: str | Path = "runs",
) -> RunLayout:
    """Create a run directory layout."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{model_name}_{timestamp}"
    return layout_from_run_dir(Path(root_dir) / run_name)


def snapshot_config(config: dict[str, Any], dest_dir: Path) -> None:
    """Write config snapshot to run directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = dest_dir / "experiment.snapshot.toml"
    with open(snapshot_path, "wb") as f:
        tomli_w.dump(config, f)


def _dataset_ids(dataset: Any) -> list[str]:
    ids = getattr(dataset, "case_ids", None)
    if ids is None:
        ids = getattr(dataset, "ids", None)
    if ids is None:
        return []
    return [str(sample_id) for sample_id in ids]


def snapshot_split_ids(data_adapter: Any, dest_dir: Path) -> None:
    """Write train/val/test sample ids to the run directory when available."""
    split_dir = dest_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, dataset in zip(
        ("train", "val", "test"),
        data_adapter.get_splits(),
    ):
        ids = _dataset_ids(dataset)
        text = "\n".join(ids)
        (split_dir / f"{split_name}_ids.txt").write_text(f"{text}\n" if text else "")


def backup_code(src_dir: Path, dest_dir: Path) -> None:
    """Backup source code to run directory."""
    if not src_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_dir / item.name)
