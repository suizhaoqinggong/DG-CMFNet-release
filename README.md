# DG-CMFNet

Official training code for **DG-CMFNet**: Dual-Granularity Cross-Modal Fusion Network for brain tumor segmentation on BraTS.

This repository is a sanitized public snapshot of the paper implementation. It does **not** include BraTS images, checkpoints, or internal lab paths.

## Requirements

- Python 3.9
- CUDA GPU recommended for the paper training setup
- BraTS 2020 / BraTS 2023 data obtained from the official challenge organizers

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync --extra dev
```

## Data

BraTS volumes are **not** redistributed here. After you obtain the official training data:

1. Preprocess to compact arrays:

```bash
python scripts/preprocess_brats.py \
  --input /path/to/MICCAI_BraTS2020_TrainingData \
  --output data/brats2020
```

2. Point the experiment config at that directory. Paper configs use:

- `data/brats2020` for BraTS 2020
- `data/brats2023` for BraTS 2023
- split files under `configs/splits/`

The paper's BraTS 2020 train/val/holdout IDs are in `configs/splits/brats2020_fold0_*.txt`.

## Train the paper model

```bash
uv run framework train \
  --configs configs/experiment.brats2020.toml \
  --model configs/model.dgcmfnet_V2.toml
```

Ablation configs live next to the main model file:

- `configs/model.dgcmfnet_V2.wo_gim.toml`
- `configs/model.dgcmfnet_V2.wo_gfm.toml`
- `configs/model.dgcmfnet_V2.wo_fdfm.toml`
- `configs/model.dgcmfnet_V2.wo_all.toml`

## Full-volume validation

```bash
python scripts/validate_full_volume.py --run-dir runs/<your-run>
```

## Comparison models

Several baselines are wrapped as adapters and keep their original source trees **outside** this repository. Set the corresponding environment variable or `source_root` in the model TOML:

| Model | Environment variable |
| --- | --- |
| TransBTS | `TRANSBTS_SOURCE_ROOT` |
| NestedFormer | `NESTEDFORMER_SOURCE_ROOT` |
| Slim UNETR | `SLIM_UNETR_SOURCE_ROOT` |
| SegMamba | `SEGMAMBA_SOURCE_ROOT` |
| SegMamba-V2 | `SEGMAMBA_V2_SOURCE_ROOT` |
| nnMamba | `NNMAMBA_SOURCE_ROOT` |
| U-Mamba | `UMAMBA_SOURCE_ROOT` |
| VT-UNet | `VTUNET_SOURCE_ROOT` |

## Tests

```bash
uv run pytest
```

Adapter tests that need an upstream source tree are skipped when that tree is absent.

## What is not in this snapshot

- BraTS images, NIfTI exports, and checkpoints
- Internal GPU host scripts and absolute lab paths
- Unpublished manuscript sources and qualitative patient-slice figures

Place weights and large result tables in a data repository (for example Zenodo) and link the DOI here after upload.

## Citation

If you use this code, please cite the DG-CMFNet paper (update the bibliographic details after publication). See `CITATION.cff`.
