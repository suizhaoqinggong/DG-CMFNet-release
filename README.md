# DG-CMFNet

Official implementation of **DG-CMFNet**: Dual-Granularity Cross-Modal Fusion Network for brain tumor segmentation on BraTS.

This repository contains the model and the training code needed to run the paper method. It does **not** include BraTS images, checkpoints, comparison models, or ablation configs.

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

```bash
python scripts/preprocess_brats.py \
  --input /path/to/MICCAI_BraTS2020_TrainingData \
  --output data/brats2020
```

Paper configs expect:

- `data/brats2020` for BraTS 2020
- `data/brats2023` for BraTS 2023
- BraTS 2020 train/val/holdout IDs in `configs/splits/brats2020_fold0_*.txt`

## Train DG-CMFNet

```bash
uv run framework train \
  --configs configs/experiment.brats2020.toml \
  --model configs/model.dgcmfnet_V2.toml
```

BraTS 2023 uses `configs/experiment.brats.toml` with the same model config.

## Full-volume validation

```bash
python scripts/validate_full_volume.py --run-dir runs/<your-run>
```

## Tests

```bash
uv run pytest
```

## Citation

If you use this code, please cite the DG-CMFNet paper (update the bibliographic details after publication). See `CITATION.cff`.
