# DG-CMFNet

Official implementation of **DG-CMFNet**: Dual-Granularity Cross-Modal Fusion Network for brain tumor segmentation on BraTS.

This repository contains the model and the training code needed to run the paper method. It does **not** include BraTS images, checkpoints, comparison models, or ablation configs.

## Requirements

- Python 3.9
- A CUDA GPU is recommended for the paper training setup
- BraTS 2020 / BraTS 2023 training data from the official challenge organizers

Install with [uv](https://github.com/astral-sh/uv). Package downloads use the Tsinghua PyPI mirror by default:

```bash
uv sync
```

## Data

BraTS volumes are **not** redistributed here. After you obtain the official training data, preprocess it with the project environment:

```bash
uv run python scripts/preprocess_brats.py \
  --input /path/to/MICCAI_BraTS2020_TrainingData \
  --output data/brats2020
```

```bash
uv run python scripts/preprocess_brats.py \
  --input /path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --output data/brats2023
```

The preprocessor accepts both compressed `.nii.gz` and uncompressed `.nii` files (BraTS 2020 is often the latter). It writes one `.npz` file per case. The shipped experiment configs currently set `data_format = "npy"`. After using this preprocessor, change that field to `"npz"` in the experiment TOML, or convert the arrays to `.npy` directories yourself.

Paper configs expect:

- `data/brats2020` for BraTS 2020, with the fold-0 IDs in `configs/splits/brats2020_fold0_*.txt` (221 / 74 / 74 train / val / holdout)
- `data/brats2023` for BraTS 2023, which has no checked-in split files and uses `val_ratio = 0.20`

## Train DG-CMFNet

```bash
uv run framework train \
  --configs configs/experiment.brats2020.toml \
  --model configs/model.dgcmfnet_V2.toml
```

BraTS 2023 uses `configs/experiment.brats.toml` with the same model config.

