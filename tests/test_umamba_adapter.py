import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from framework.contracts.types import Batch
from framework.registry.defaults import create_default_registries
from models.umamba_adapter import UMambaAdapter


class _FakeMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def _install_fake_mamba(monkeypatch) -> None:
    module = types.ModuleType("mamba_ssm")
    module.Mamba = _FakeMamba
    monkeypatch.setitem(sys.modules, "mamba_ssm", module)


def _source_root() -> str:
    import os

    import pytest

    configured = os.environ.get("UMAMBA_SOURCE_ROOT")
    root = Path(configured) if configured else Path("external/U-Mamba")
    source_file = root / "umamba" / "nnunetv2" / "nets" / "UMambaEnc_3d.py"
    if not source_file.is_file():
        pytest.skip("External U-Mamba source tree is not available in this environment")
    return str(root)


def _make_batch() -> Batch:
    return Batch(
        signal=torch.randn(1, 2, 16, 16, 16),
        label=torch.zeros(1, 16, 16, 16).long(),
        id=["case-1"],
        meta=[{}],
    )


def _make_model(variant: str) -> UMambaAdapter:
    return UMambaAdapter(
        num_classes=3,
        num_modalities=2,
        variant=variant,
        source_root=_source_root(),
        img_size=(16, 16, 16),
        num_stages=4,
        features_per_stage=[4, 8, 16, 32],
        kernel_sizes=[(3, 3, 3)] * 4,
        strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        n_conv_per_stage=[1, 1, 1, 1],
        n_conv_per_stage_decoder=[1, 1, 1],
    )


def test_umamba_bot_returns_framework_logits(monkeypatch):
    _install_fake_mamba(monkeypatch)
    model = _make_model("bot").eval()

    with torch.no_grad():
        output = model(_make_batch())

    assert output.shape == (1, 3, 16, 16, 16)


def test_umamba_enc_returns_framework_logits(monkeypatch):
    _install_fake_mamba(monkeypatch)
    model = _make_model("enc").eval()

    with torch.no_grad():
        output = model(_make_batch())

    assert output.shape == (1, 3, 16, 16, 16)


def test_umamba_rejects_invalid_variant():
    try:
        UMambaAdapter(variant="unknown")
    except ValueError as exc:
        assert "variant must be 'enc' or 'bot'" in str(exc)
    else:
        raise AssertionError("invalid U-Mamba variant was accepted")


def test_umamba_is_registered():
    assert "umamba" in create_default_registries().models
