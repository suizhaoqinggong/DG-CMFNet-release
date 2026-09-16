"""Run FG-GIM-only and CG-GFM-only DG-CMFNet V2 configs sequentially."""

from __future__ import annotations

from pathlib import Path

from run_v2_models import run_sequence


DEFAULT_MODELS = [
    Path("configs/model.dgcmfnet_V2.fggim_only.toml"),
    Path("configs/model.dgcmfnet_V2.gfm_only.toml"),
]


def main() -> int:
    return run_sequence(
        DEFAULT_MODELS,
        description=__doc__,
        snapshot_name="v2_gim_gfm_only_sequence",
        completion_message="All requested GIM/GFM-only V2 training runs completed.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
