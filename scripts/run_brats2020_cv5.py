"""Run BraTS2020 5-fold cross-validation sequentially and summarize metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli
import tomli_w
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="configs/model.dgcmfnet.toml",
        help="Model config TOML passed to the framework train command.",
    )
    parser.add_argument(
        "--config",
        "--config-pattern",
        dest="config_pattern",
        default="configs/experiment.brats2020.cv5.toml",
        help="Experiment config TOML. Strings may contain {fold}; legacy paths may also contain {fold}.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Fold indices to run in order.",
    )
    parser.add_argument(
        "--summary-root",
        default="runs/cv5_summaries",
        help="Directory where cross-validation summaries will be written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Temporarily override [trainer].device for all folds, for example "cuda:2".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and validate config paths without launching training.",
    )
    return parser.parse_args()


def load_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)


def fold_config_path(pattern: str, fold: int) -> Path:
    if "{fold}" in pattern:
        return Path(pattern.format(fold=fold))
    return Path(pattern)


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


def format_fold_placeholders(value: Any, fold: int) -> Any:
    if isinstance(value, dict):
        return {key: format_fold_placeholders(item, fold) for key, item in value.items()}
    if isinstance(value, list):
        return [format_fold_placeholders(item, fold) for item in value]
    if isinstance(value, str) and "{fold}" in value:
        return value.replace("{fold}", str(fold))
    return value


def build_runtime_config(config: dict[str, Any], *, fold: int, device: str | None) -> dict[str, Any]:
    runtime_config = format_fold_placeholders(config, fold)
    if device is None:
        return runtime_config

    trainer_config = dict(runtime_config.get("trainer", {}))
    trainer_config["device"] = device
    runtime_config["trainer"] = trainer_config
    return runtime_config


def runtime_config_path(source_path: Path, summary_dir: Path, fold: int) -> Path:
    return summary_dir / "runtime_configs" / f"{source_path.stem}.runtime.fold{fold}.toml"


def write_runtime_config(runtime_config: dict[str, Any], runtime_path: Path) -> Path:
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_path, "wb") as f:
        tomli_w.dump(runtime_config, f)
    return runtime_path


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
    run_dir: Path | None = None
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


def resolve_monitor_value(metrics: dict[str, Any], monitor: str) -> float | None:
    value = metrics.get(monitor)
    if value is None and monitor.startswith("val_"):
        value = metrics.get(monitor[4:])
    if value is None:
        return None
    return float(value)


def load_best_checkpoint(run_dir: Path, monitor: str) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError(f"Checkpoint metrics must be a dict: {checkpoint_path}")

    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(payload["epoch"]),
        "monitor_value": resolve_monitor_value(metrics, monitor),
        "metrics": {str(key): float(value) for key, value in metrics.items() if isinstance(value, (float, int))},
    }


def metric_stats(values: list[float]) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std_population": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "variance_population": float(statistics.pvariance(values)) if len(values) > 1 else 0.0,
    }
    if len(values) > 1:
        result["std_sample"] = float(statistics.stdev(values))
        result["variance_sample"] = float(statistics.variance(values))
    else:
        result["std_sample"] = math.nan
        result["variance_sample"] = math.nan
    return result


def summarize_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = sorted({key for row in fold_results for key in row["metrics"]})
    metrics_summary = {}
    for key in metric_keys:
        values = [row["metrics"][key] for row in fold_results if key in row["metrics"]]
        metrics_summary[key] = metric_stats(values)

    monitor_values = [row["monitor_value"] for row in fold_results if row["monitor_value"] is not None]
    return {
        "folds": len(fold_results),
        "monitor": fold_results[0]["monitor"] if fold_results else None,
        "monitor_summary": metric_stats([float(value) for value in monitor_values]) if monitor_values else None,
        "metrics": metrics_summary,
        "fold_results": fold_results,
    }


def write_summary(summary_dir: Path, summary: dict[str, Any]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "cv5_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    fold_rows = []
    for row in summary["fold_results"]:
        flat_row = {
            "fold": row["fold"],
            "best_epoch": row["best_epoch"],
            "monitor": row["monitor"],
            "monitor_value": row["monitor_value"],
            "run_dir": row["run_dir"],
            "checkpoint": row["checkpoint"],
        }
        flat_row.update(row["metrics"])
        fold_rows.append(flat_row)

    if fold_rows:
        fieldnames = sorted({key for row in fold_rows for key in row})
        with open(summary_dir / "cv5_fold_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fold_rows)

    metric_fieldnames = ["metric", "count", "mean", "std_sample", "variance_sample", "std_population", "variance_population"]
    with open(summary_dir / "cv5_metric_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
        writer.writeheader()
        for metric, stats in summary["metrics"].items():
            writer.writerow({"metric": metric, **stats})


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_path}")

    model_name = str(load_toml(model_path)["model"]["name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path(args.summary_root) / f"brats2020_cv5_{model_name}_{timestamp}"
    runtime_device = args.device

    fold_results: list[dict[str, Any]] = []
    for fold in args.folds:
        config_path = fold_config_path(args.config_pattern, fold)
        if not config_path.exists():
            raise FileNotFoundError(f"Fold config not found: {config_path}")

        config = build_runtime_config(load_toml(config_path), fold=fold, device=runtime_device)
        checkpoint_config = config.get("checkpoint", {})
        monitor = str(checkpoint_config.get("monitor", "val_loss"))
        runtime_path = runtime_config_path(config_path, summary_dir, fold)
        if not args.dry_run:
            write_runtime_config(
                config,
                runtime_path,
            )
        command = framework_train_command(runtime_path, model_path)

        if args.dry_run:
            if args.device is not None:
                print(f"fold{fold}: requested device={args.device}, runtime trainer.device={runtime_device}")
            print(f"fold{fold}: source config={config_path}")
            print(f"fold{fold}: runtime config={runtime_path}")
            print(f"fold{fold}: experiment.name={config.get('experiment', {}).get('name')}")
            print(f"fold{fold}: trainer.device={config.get('trainer', {}).get('device')}")
            print(f"fold{fold}: train_ids_file={config.get('data', {}).get('train_ids_file')}")
            print(f"fold{fold}: {' '.join(command)}")
            continue

        run_dir = run_training(command)
        result = load_best_checkpoint(run_dir, monitor)
        result["fold"] = fold
        result["config"] = str(config_path)
        result["runtime_config"] = str(runtime_path)
        result["model"] = str(model_path)
        result["monitor"] = monitor
        fold_results.append(result)

        print(
            f"fold{fold} best: epoch={result['best_epoch']} "
            f"{monitor}={result['monitor_value']} run_dir={run_dir}"
        )

    if args.dry_run:
        print(f"summary would be written under: {summary_dir}")
        return 0

    summary = summarize_fold_results(fold_results)
    write_summary(summary_dir, summary)

    monitor = summary["monitor"]
    monitor_summary = summary["monitor_summary"]
    print("\n" + "=" * 80)
    print(f"CV summary written to: {summary_dir}")
    if monitor_summary is not None:
        print(
            f"{monitor}: mean={monitor_summary['mean']:.6f}, "
            f"std_sample={monitor_summary['std_sample']:.6f}, "
            f"variance_sample={monitor_summary['variance_sample']:.6f}"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
