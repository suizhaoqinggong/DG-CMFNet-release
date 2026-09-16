"""Generate standard 5-fold BraTS2020 cross-validation split files.

All available BraTS2020 cases participate in cross-validation. Each fold uses
one validation partition, the remaining cases for training, and an empty test
split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        default="data/brats2020_uncompressed",
        help="BraTS2020 case directory containing one subdirectory per case.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/splits",
        help="Directory where split files will be written.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=37, help="Deterministic shuffle seed.")
    parser.add_argument("--expected-count", type=int, default=369, help="Expected number of cases.")
    return parser.parse_args()


def _case_ids(root_dir: Path) -> list[str]:
    if not root_dir.exists():
        raise FileNotFoundError(f"BraTS2020 directory not found: {root_dir}")
    case_ids = sorted(path.name for path in root_dir.iterdir() if path.is_dir())
    if not case_ids:
        raise ValueError(f"No case directories found in {root_dir}")
    return case_ids


def _build_folds(case_ids: list[str], *, folds: int, seed: int) -> list[list[str]]:
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)

    base_size, remainder = divmod(len(shuffled), folds)
    val_folds: list[list[str]] = []
    cursor = 0
    for fold in range(folds):
        fold_size = base_size + (1 if fold < remainder else 0)
        val_folds.append(sorted(shuffled[cursor : cursor + fold_size]))
        cursor += fold_size
    return val_folds


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + ("\n" if ids else ""))


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _case_ids(root_dir)
    if len(case_ids) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} cases in {root_dir}, found {len(case_ids)}")

    val_folds = _build_folds(case_ids, folds=args.folds, seed=args.seed)
    all_val_ids = [case_id for fold_ids in val_folds for case_id in fold_ids]
    if sorted(all_val_ids) != case_ids:
        raise RuntimeError("Validation folds do not cover each case exactly once")

    metadata: dict[str, object] = {
        "dataset": "Task001_BraTS2020",
        "total": len(case_ids),
        "folds": args.folds,
        "seed": args.seed,
        "method": (
            "standard 5-fold cross-validation over all sorted BraTS2020 case ids; "
            "case ids are shuffled with Python random.Random(seed), split as evenly as possible, "
            "and each fold uses the held-out partition for validation with an empty test split"
        ),
        "root_dir": str(root_dir.resolve()),
        "fold_details": [],
    }

    for fold, val_ids in enumerate(val_folds):
        val_set = set(val_ids)
        train_ids = [case_id for case_id in case_ids if case_id not in val_set]
        test_ids: list[str] = []

        train_path = output_dir / f"brats2020_cv5_fold{fold}_train_ids.txt"
        val_path = output_dir / f"brats2020_cv5_fold{fold}_val_ids.txt"
        test_path = output_dir / f"brats2020_cv5_fold{fold}_test_ids.txt"

        _write_ids(train_path, train_ids)
        _write_ids(val_path, val_ids)
        _write_ids(test_path, test_ids)

        metadata["fold_details"].append(
            {
                "fold": fold,
                "counts": {
                    "train": len(train_ids),
                    "val": len(val_ids),
                    "test": len(test_ids),
                },
                "files": {
                    "train": str(train_path),
                    "val": str(val_path),
                    "test": str(test_path),
                },
            }
        )

    metadata_path = output_dir / "brats2020_cv5_split.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for detail in metadata["fold_details"]:
        counts = detail["counts"]
        print(
            f"fold{detail['fold']}: "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
