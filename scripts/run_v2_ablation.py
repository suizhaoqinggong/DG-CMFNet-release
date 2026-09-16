"""Run the four BraTS2020 DG-CMFNet V2 ablation models sequentially."""

from __future__ import annotations

from pathlib import Path

from run_v2_models import run_sequence


DEFAULT_MODELS = [
    Path("configs/model.baseline_V2.toml"),
    Path("configs/model.dgcmfnet_V2.fggim_only.toml"),
    Path("configs/model.dgcmfnet_V2.gfm_only.toml"),
    Path("configs/model.dgcmfnet_V2.fgfm_only.toml"),
]


def main() -> int:
    return run_sequence(
        DEFAULT_MODELS,
        description=__doc__,
        snapshot_name="v2_ablation_sequence",
        completion_message="All four DG-CMFNet V2 ablation training runs completed.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
