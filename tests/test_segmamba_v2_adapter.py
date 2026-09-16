from pathlib import Path

import pytest
import torch
import torch.nn as nn

from framework.contracts.types import Batch
from framework.registry.defaults import create_default_registries
from models.segmamba_v2_adapter import SegMambaV2Adapter, _resolve_source_file


class _FakeSegMamba(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, **kwargs: object) -> None:
        super().__init__()
        self.projection = nn.Conv3d(in_chans, out_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def _make_batch(channels: int = 2, size: int = 16) -> Batch:
    return Batch(
        signal=torch.randn(1, channels, size, size, size),
        label=torch.zeros(1, size, size, size).long(),
        id=["case-1"],
        meta=[{}],
    )


def _make_model(monkeypatch: pytest.MonkeyPatch) -> SegMambaV2Adapter:
    monkeypatch.setattr("models.segmamba_v2_adapter._resolve_source_file", lambda source_root: Path(__file__))
    monkeypatch.setattr("models.segmamba_v2_adapter._load_upstream_network_class", lambda source_file: _FakeSegMamba)
    return SegMambaV2Adapter(
        num_classes=3,
        num_modalities=2,
        source_root="external/SegMamba-V2",
        img_size=(16, 16, 16),
        depths=(1, 1, 1, 1),
    )


def test_segmamba_v2_returns_framework_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch).eval()

    with torch.no_grad():
        output = model(_make_batch())

    assert output.shape == (1, 3, 16, 16, 16)


def test_segmamba_v2_rejects_wrong_input_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch)

    with pytest.raises(ValueError, match="expects spatial shape"):
        model(_make_batch(size=32))


def test_segmamba_v2_resolves_upstream_layout(tmp_path: Path) -> None:
    source_file = tmp_path / "brats23" / "models_segmamba" / "segmambav2.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()

    assert _resolve_source_file(tmp_path) == source_file


def test_segmamba_v2_is_registered() -> None:
    assert "segmamba_v2" in create_default_registries().models
