#!/usr/bin/env python3
"""Compute paired Wilcoxon p-values from per-case segmentation metrics.

Each input CSV must contain a unique ``case_id`` column and the requested
metric columns.  The script refuses to compare different case sets so a
p-value cannot accidentally be computed from unpaired evaluations.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import t, wilcoxon


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    source: Path
    values: dict[str, dict[str, float]]


def parse_method(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --baseline.") from exc
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both NAME and PATH are required for --baseline.")
    return name.strip(), Path(path).expanduser()


def read_metrics(name: str, path: Path, metrics: Iterable[str]) -> MethodMetrics:
    if not path.is_file():
        raise FileNotFoundError(f"{name}: file not found: {path}")

    metric_names = tuple(metrics)
    values: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{name}: CSV has no header: {path}")
        missing = {"case_id", *metric_names}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)} in {path}")
        for row_number, row in enumerate(reader, start=2):
            case_id = (row["case_id"] or "").strip()
            if not case_id:
                raise ValueError(f"{name}: empty case_id at row {row_number}")
            if case_id in values:
                raise ValueError(f"{name}: duplicate case_id {case_id!r}")
            try:
                values[case_id] = {metric: float(row[metric]) for metric in metric_names}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: non-numeric value at row {row_number}") from exc
    if not values:
        raise ValueError(f"{name}: no cases found in {path}")
    return MethodMetrics(name=name, source=path, values=values)


def mean_sd_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=0))
    if len(values) < 2:
        return mean, sd, math.nan, math.nan
    half_width = float(t.ppf(0.975, df=len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, sd, mean - half_width, mean + half_width


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the same order as the input."""
    adjusted = [math.nan] * len(p_values)
    previous = 0.0
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    total = len(p_values)
    for rank, (index, p_value) in enumerate(ordered):
        corrected = min(1.0, (total - rank) * p_value)
        previous = max(previous, corrected)
        adjusted[index] = previous
    return adjusted


def format_number(value: float, digits: int = 6) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}g}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, ours: MethodMetrics, rows: list[dict[str, object]], alternative: str) -> None:
    lines = [
        "# Paired Wilcoxon signed-rank test",
        "",
        f"Reference method: `{ours.name}`  ",
        f"Reference file: `{ours.source}`  ",
        f"Alternative: `{alternative}`; zero differences: `wilcox`; p-values: unadjusted and Holm-adjusted.",
        "",
        "The test is paired by `case_id`. `ours_minus_baseline` is positive when the reference method has a larger metric.",
        "",
        "| Metric | Baseline | N | Ours mean ± SD | Baseline mean ± SD | Ours − baseline | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {baseline} | {n_cases} | {ours_mean} ± {ours_sd} | {baseline_mean} ± "
            "{baseline_sd} | {ours_minus_baseline} | {p_value_raw} | {p_value_holm} |".format(
                metric=row["metric"],
                baseline=row["baseline"],
                n_cases=row["n_cases"],
                ours_mean=format_number(float(row["ours_mean"])),
                ours_sd=format_number(float(row["ours_sd"])),
                baseline_mean=format_number(float(row["baseline_mean"])),
                baseline_sd=format_number(float(row["baseline_sd"])),
                ours_minus_baseline=format_number(float(row["ours_minus_baseline"])),
                p_value_raw=format_number(float(row["p_value_raw"])),
                p_value_holm=format_number(float(row["p_value_holm"])),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path, help="CSV for the reference/DG-CMFNet method.")
    parser.add_argument("--ours-name", default="DG-CMFNet", help="Display name for the reference method.")
    parser.add_argument(
        "--baseline", action="append", required=True, type=parse_method, metavar="NAME=CSV", help="Baseline CSV; repeat."
    )
    parser.add_argument("--metrics", nargs="+", default=["mean_dice", "mean_hd95"], help="Per-case metric columns.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and Markdown reports.")
    parser.add_argument(
        "--alternative", choices=("two-sided", "greater", "less"), default="two-sided", help="Wilcoxon alternative hypothesis."
    )
    args = parser.parse_args()

    baseline_names = [name for name, _ in args.baseline]
    if len(baseline_names) != len(set(baseline_names)):
        parser.error("Baseline names must be unique.")

    ours = read_metrics(args.ours_name, args.ours.expanduser(), args.metrics)
    baselines = [read_metrics(name, path, args.metrics) for name, path in args.baseline]
    ours_cases = set(ours.values)
    rows: list[dict[str, object]] = []

    for baseline in baselines:
        baseline_cases = set(baseline.values)
        if ours_cases != baseline_cases:
            missing_from_baseline = sorted(ours_cases - baseline_cases)
            missing_from_ours = sorted(baseline_cases - ours_cases)
            raise ValueError(
                f"Case-set mismatch for {baseline.name}: "
                f"missing from baseline={missing_from_baseline[:5]}, missing from ours={missing_from_ours[:5]}"
            )
        case_ids = sorted(ours_cases)
        for metric in args.metrics:
            ours_values = np.array([ours.values[case_id][metric] for case_id in case_ids], dtype=float)
            baseline_values = np.array([baseline.values[case_id][metric] for case_id in case_ids], dtype=float)
            if not (np.isfinite(ours_values).all() and np.isfinite(baseline_values).all()):
                raise ValueError(f"Non-finite values found for {baseline.name}, metric {metric}")
            try:
                statistic, p_value = wilcoxon(
                    ours_values,
                    baseline_values,
                    alternative=args.alternative,
                    zero_method="wilcox",
                    method="auto",
                )
            except ValueError as exc:
                raise ValueError(f"Wilcoxon failed for {baseline.name}, metric {metric}: {exc}") from exc
            ours_mean, ours_sd, ours_ci_low, ours_ci_high = mean_sd_ci(ours_values)
            baseline_mean, baseline_sd, baseline_ci_low, baseline_ci_high = mean_sd_ci(baseline_values)
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline.name,
                    "n_cases": len(case_ids),
                    "wilcoxon_statistic": float(statistic),
                    "p_value_raw": float(p_value),
                    "ours_mean": ours_mean,
                    "ours_sd": ours_sd,
                    "ours_ci95_low": ours_ci_low,
                    "ours_ci95_high": ours_ci_high,
                    "baseline_mean": baseline_mean,
                    "baseline_sd": baseline_sd,
                    "baseline_ci95_low": baseline_ci_low,
                    "baseline_ci95_high": baseline_ci_high,
                    "ours_minus_baseline": float(np.mean(ours_values - baseline_values)),
                }
            )

    adjusted = holm_adjust([float(row["p_value_raw"]) for row in rows])
    for row, p_value_holm in zip(rows, adjusted):
        row["p_value_holm"] = p_value_holm

    output_dir = args.output_dir.expanduser()
    write_csv(output_dir / "paired_wilcoxon_results.csv", rows)
    write_report(output_dir / "paired_wilcoxon_report.md", ours, rows, args.alternative)
    print(f"Wrote {output_dir / 'paired_wilcoxon_results.csv'}")
    print(f"Wrote {output_dir / 'paired_wilcoxon_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
