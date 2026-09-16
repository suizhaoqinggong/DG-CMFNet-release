from pathlib import Path

import pytest
import torch
import torch.nn as nn

from framework.contracts.types import Batch
from framework.registry.defaults import create_default_registries
from models.segmamba_adapter import (
    SegMambaAdapter,
    _CausalConv1dCompat,
    _resolve_source_file,
    _validate_spatial_shape,
)


class _FakeSegMamba(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, **kwargs: object) -> None:
        super().__init__()
        self.projection = nn.Conv3d(in_chans, out_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def _make_batch(channels: int = 2, size: int = 32) -> Batch:
    return Batch(
        signal=torch.randn(1, channels, size, size, size),
        label=torch.zeros(1, size, size, size).long(),
        id=["case-1"],
        meta=[{}],
    )


def _make_model(monkeypatch: pytest.MonkeyPatch) -> SegMambaAdapter:
    monkeypatch.setattr("models.segmamba_adapter._resolve_source_file", lambda source_root: Path(__file__))
    monkeypatch.setattr("models.segmamba_adapter._load_upstream_network_class", lambda source_file: _FakeSegMamba)
    return SegMambaAdapter(
        num_classes=3,
        num_modalities=2,
        source_root="external/SegMamba",
        img_size=(32, 32, 32),
        depths=(1, 1, 1, 1),
    )


def test_segmamba_returns_framework_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch).eval()

    with torch.no_grad():
        output = model(_make_batch())

    assert output.shape == (1, 3, 32, 32, 32)


def test_segmamba_rejects_wrong_input_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch)

    with pytest.raises(ValueError, match="expects spatial shape"):
        model(_make_batch(size=64))


def test_segmamba_validates_bimamba_slice_shape() -> None:
    with pytest.raises(ValueError, match="encoder stage 3"):
        _validate_spatial_shape((16, 16, 16))


def test_segmamba_resolves_upstream_layout(tmp_path: Path) -> None:
    source_file = tmp_path / "model_segmamba" / "segmamba.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()

    assert _resolve_source_file(tmp_path) == source_file


def test_segmamba_is_registered() -> None:
    assert "segmamba" in create_default_registries().models


def test_segmamba_translates_current_causal_conv_extension_api() -> None:
    class _Backend:
        def causal_conv1d_fwd(self, *args: object) -> torch.Tensor:
            assert len(args) == 7
            return torch.ones(1)

        def causal_conv1d_bwd(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, None, None]:
            assert len(args) == 10
            return torch.ones(1), torch.ones(1), None, None

    compat = _CausalConv1dCompat(_Backend())
    tensor = torch.ones(1)

    assert compat.causal_conv1d_fwd(tensor, tensor, None, True).item() == 1
    assert len(compat.causal_conv1d_bwd(tensor, tensor, None, tensor, None, True)) == 3
