"""Device resolution, AMP, and batch movement utilities."""

from typing import Optional

import torch

try:
    from torch.amp import GradScaler
except ImportError:  # torch<2.3 keeps GradScaler under torch.cuda.amp
    from torch.cuda.amp import GradScaler  # type: ignore[no-redef]

from framework.contracts.types import Batch


def resolve_device(device_spec: Optional[str]) -> torch.device:
    """Resolve device string to torch.device."""
    if device_spec is None or device_spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_spec)

    if device.type == "cuda" and not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True
    return device


def build_grad_scaler(enabled: bool) -> Optional[GradScaler]:
    """Build a GradScaler if AMP is enabled."""
    if enabled:
        return GradScaler()
    return None


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move all tensors in a batch to the given device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
