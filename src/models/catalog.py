"""Model registry catalog."""

from typing import cast

from framework.contracts.model import ModelAdapter
from framework.registry.registry import Registry

from .attention_unet import AttentionUNet
from .dgcmfnet import DGCMFNet
from .nnformer_adapter import NnFormerAdapter
from .nnmamba_adapter import NnMambaAdapter
from .nnunet_adapter import NnUNetAdapter
from .resunet_2d import SliceWiseResUNet2D
from .segformer3d_adapter import SegFormer3DAdapter
from .segmamba_adapter import SegMambaAdapter
from .segmamba_v2_adapter import SegMambaV2Adapter
from .shared_encoder_concat_unet import BaselineV2, DGCMFNetV2, SharedEncoderConcatUNet
from .simple_net import SimpleNet
from .swinbts import SwinBTSAdapter
from .swinunetr_adapter import SwinUNETRAdapter
from .umamba_adapter import UMambaAdapter
from .unet_2d import PaperUNet2D
from .unet_variants import (
    DenseUNet3D,
    NestedFormerAdapter,
    PaperUNet3D,
    ResUNet3D,
    SlimUNETRAdapter,
    TransBTSAdapter,
    UNet3D,
)
from .vtunet_adapter import VTUNetAdapter

ModelFactory = type[ModelAdapter]

MODEL_REGISTRY_TABLE: dict[str, ModelFactory] = {
    "simple_net": cast(ModelFactory, SimpleNet),
    "dgcmfnet": cast(ModelFactory, DGCMFNet),
    # Paper-faithful 64-1024 U-Net (current default configs/model.unet.toml).
    "unet": cast(ModelFactory, PaperUNet3D),
    # Configurable UNet3D used by original-loss BraTS2020 comparison runs
    # (snapshot keys: base_ch / channel_multipliers / norm / block_type).
    "unet3d": cast(ModelFactory, UNet3D),
    "unet_2d": cast(ModelFactory, PaperUNet2D),
    "resunet": cast(ModelFactory, ResUNet3D),
    "resunet_2d": cast(ModelFactory, SliceWiseResUNet2D),
    "dense_unet": cast(ModelFactory, DenseUNet3D),
    "attention_unet": cast(ModelFactory, AttentionUNet),
    "transbts": cast(ModelFactory, TransBTSAdapter),
    "nestedformer": cast(ModelFactory, NestedFormerAdapter),
    "slim_unetr": cast(ModelFactory, SlimUNETRAdapter),
    "nnformer": cast(ModelFactory, NnFormerAdapter),
    "nnmamba": cast(ModelFactory, NnMambaAdapter),
    "nnunet": cast(ModelFactory, NnUNetAdapter),
    "segformer3d": cast(ModelFactory, SegFormer3DAdapter),
    "segmamba": cast(ModelFactory, SegMambaAdapter),
    "segmamba_v2": cast(ModelFactory, SegMambaV2Adapter),
    "vtunet": cast(ModelFactory, VTUNetAdapter),
    "swinbts": cast(ModelFactory, SwinBTSAdapter),
    "swinunetr": cast(ModelFactory, SwinUNETRAdapter),
    "umamba": cast(ModelFactory, UMambaAdapter),
    "baseline_V2": cast(ModelFactory, BaselineV2),
    "dgcmfnet_V2": cast(ModelFactory, DGCMFNetV2),
    "shared_encoder_concat_unet": cast(ModelFactory, SharedEncoderConcatUNet),
}


def register_models(registry: Registry[ModelAdapter]) -> None:
    """Register all models into the given registry."""
    for name, factory in MODEL_REGISTRY_TABLE.items():
        registry.register(name, factory)
