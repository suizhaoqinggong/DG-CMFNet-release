"""Model registry catalog."""

from typing import cast

from framework.contracts.model import ModelAdapter
from framework.registry.registry import Registry

from .dgcmfnet import DGCMFNet
from .shared_encoder_concat_unet import DGCMFNetV2
from .simple_net import SimpleNet

ModelFactory = type[ModelAdapter]

MODEL_REGISTRY_TABLE: dict[str, ModelFactory] = {
    "simple_net": cast(ModelFactory, SimpleNet),
    "dgcmfnet": cast(ModelFactory, DGCMFNet),
    "dgcmfnet_V2": cast(ModelFactory, DGCMFNetV2),
}


def register_models(registry: Registry[ModelAdapter]) -> None:
    """Register all models into the given registry."""
    for name, factory in MODEL_REGISTRY_TABLE.items():
        registry.register(name, factory)
