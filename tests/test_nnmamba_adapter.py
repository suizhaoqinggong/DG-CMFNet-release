from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from framework.contracts.types import Batch
from framework.registry.defaults import create_default_registries
from models.nnmamba_adapter import NnMambaAdapter, _resolve_source_file, _validate_spatial_shape


class _FakeNnMamba(nn.Module):
    def __init__(self, in_ch: int, number_classes: int, **kwargs: object) -> None:
        super().__init__()
        self.projection = nn.Conv3d(in_ch, number_classes, kernel_size=1)
        self.do_ds = True

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        logits = self.projection(x)
        return [logits, logits] if self.do_ds else logits


def _make_batch(channels: int = 2, size: int = 16) -> Batch:
    return Batch(
        signal=torch.randn(1, channels, size, size, size),
        label=torch.zeros(1, size, size, size).long(),
        id=["case-1"],
        meta=[{}],
    )


def _make_model(monkeypatch: pytest.MonkeyPatch) -> NnMambaAdapter:
    monkeypatch.setattr("models.nnmamba_adapter._resolve_source_file", lambda source_root: Path(__file__))
    monkeypatch.setattr("models.nnmamba_adapter._load_upstream_network_class", lambda source_file: _FakeNnMamba)
    return NnMambaAdapter(
        num_classes=3,
        num_modalities=2,
        source_root="external/nnMamba",
        img_size=(16, 16, 16),
    )


def test_nnmamba_returns_framework_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch).eval()

    with torch.no_grad():
        output = model(_make_batch())

    assert output.shape == (1, 3, 16, 16, 16)
    assert model.model.do_ds is False


def test_nnmamba_rejects_wrong_input_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch)

    with pytest.raises(ValueError, match="expects spatial shape"):
        model(_make_batch(size=32))


def test_nnmamba_requires_dimensions_divisible_by_16() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        _validate_spatial_shape((16, 16, 20))


def test_nnmamba_resolves_upstream_layout(tmp_path: Path) -> None:
    source_file = tmp_path / "nnunet" / "network_architecture" / "nnMamba.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()

    assert _resolve_source_file(tmp_path) == source_file


def test_nnmamba_is_registered() -> None:
    assert "nnmamba" in create_default_registries().models
