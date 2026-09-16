"""3D data augmentation transforms for brain tumor segmentation.

All transforms operate on numpy arrays in (M, D, H, W) and (D, H, W) format.
Spatial transforms are applied consistently to image and label.
"""

from __future__ import annotations

import numpy as np


class RandomCrop3D:
    """Randomly crop a fixed-size sub-volume from a larger volume."""

    def __init__(self, crop_size: tuple[int, int, int] = (128, 128, 128)):
        self.crop_size = crop_size

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cd, ch, cw = self.crop_size
        _, D, H, W = volume.shape
        d0 = np.random.randint(0, max(1, D - cd + 1))
        h0 = np.random.randint(0, max(1, H - ch + 1))
        w0 = np.random.randint(0, max(1, W - cw + 1))
        return volume[:, d0:d0+cd, h0:h0+ch, w0:w0+cw], seg[d0:d0+cd, h0:h0+ch, w0:w0+cw]


class CenterCrop3D:
    """Center-crop a fixed-size sub-volume."""

    def __init__(self, crop_size: tuple[int, int, int] = (128, 128, 128)):
        self.crop_size = crop_size

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cd, ch, cw = self.crop_size
        _, D, H, W = volume.shape
        d0 = max(0, (D - cd) // 2)
        h0 = max(0, (H - ch) // 2)
        w0 = max(0, (W - cw) // 2)
        return volume[:, d0:d0+cd, h0:h0+ch, w0:w0+cw], seg[d0:d0+cd, h0:h0+ch, w0:w0+cw]


class RandomFlip3D:
    """Randomly flip each spatial axis with independent probability."""

    def __init__(self, axes: tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
        self.axes = axes
        self.prob = prob

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for axis in self.axes:
            if np.random.random() < self.prob:
                volume = np.flip(volume, axis=axis + 1)
                seg = np.flip(seg, axis=axis)
        return volume, seg


class RandomRotate90:
    """Random 90° rotation in the axial plane (H-W axes)."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() >= self.prob:
            return volume, seg

        k = np.random.randint(1, 4)
        volume_rot = np.rot90(volume, k=k, axes=(-2, -1))
        seg_rot = np.rot90(seg, k=k, axes=(-2, -1))
        return np.ascontiguousarray(volume_rot), np.ascontiguousarray(seg_rot)


class RandomIntensityScale:
    """Multiply each modality by a random factor."""

    def __init__(self, scale_range: tuple[float, float] = (0.85, 1.15), prob: float = 0.5):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            for m in range(volume.shape[0]):
                factor = np.random.uniform(*self.scale_range)
                mask = volume[m] > 1e-6
                volume[m][mask] *= factor
        return volume, seg


class RandomIntensityShift:
    """Add a random offset to each modality."""

    def __init__(self, shift_std: float = 0.1, prob: float = 0.5):
        self.shift_std = shift_std
        self.prob = prob

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            for m in range(volume.shape[0]):
                v = volume[m]
                mask = v > 1e-6
                if mask.sum() == 0:
                    continue
                offset = np.random.normal(0, self.shift_std * v[mask].std())
                v[mask] += offset
        return volume, seg


class RandomGaussianNoise:
    """Add Gaussian noise to each modality."""

    def __init__(self, noise_std: float = 0.02, prob: float = 0.5):
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            for m in range(volume.shape[0]):
                mask = volume[m] > 1e-6
                noise = np.random.normal(0, self.noise_std, volume[m].shape).astype(np.float32)
                volume[m][mask] += noise[mask]
        return volume, seg


class RandomGamma:
    """Apply random gamma correction to each modality."""

    def __init__(self, gamma_range: tuple[float, float] = (0.7, 1.5), prob: float = 0.3):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            for m in range(volume.shape[0]):
                v = volume[m]
                mask = v > 1e-6
                if mask.sum() == 0:
                    continue
                v_min, v_max = v[mask].min(), v[mask].max()
                if v_max <= v_min:
                    continue
                v_norm = np.clip((v - v_min) / (v_max - v_min), 0.0, 1.0)
                gamma = np.random.uniform(*self.gamma_range)
                v_gamma = np.power(v_norm, gamma)
                volume[m] = v_gamma * (v_max - v_min) + v_min
        return volume, seg


class Compose:
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, volume: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            volume, seg = t(volume, seg)
        return volume, seg


def default_training_augmentation(crop_size: tuple[int, int, int] = (128, 128, 128)) -> Compose:
    return Compose(
        [
            RandomCrop3D(crop_size=crop_size),
            RandomFlip3D(axes=(0, 1, 2), prob=0.5),
            RandomRotate90(prob=0.5),
            RandomIntensityScale(scale_range=(0.85, 1.15), prob=0.3),
            RandomIntensityShift(shift_std=0.1, prob=0.3),
            RandomGaussianNoise(noise_std=0.02, prob=0.3),
            RandomGamma(gamma_range=(0.7, 1.5), prob=0.3),
        ]
    )


def default_validation_crop(crop_size: tuple[int, int, int] = (128, 128, 128)) -> Compose:
    return Compose([CenterCrop3D(crop_size=crop_size)])
