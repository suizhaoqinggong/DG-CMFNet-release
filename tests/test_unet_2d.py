import torch

from models.unet_2d import PaperUNet2D


def test_paper_unet_2d_preserves_valid_convolutions_and_paper_geometry():
    model = PaperUNet2D(channels=[2, 4, 8, 16, 32], bottleneck_dropout=0.0)
    convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    three_by_three = [module for module in convs if module.kernel_size == (3, 3)]
    assert PaperUNet2D.paper_channels == (64, 128, 256, 512, 1024)
    assert PaperUNet2D.paper_output_size(572) == 388
    assert PaperUNet2D.padded_input_size(128) == 316
    assert all(module.padding == (0, 0) for module in three_by_three)
    assert not any(isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)) for module in model.modules())


def test_paper_unet_2d_returns_slice_and_reassembled_volume_shapes():
    model = PaperUNet2D(
        num_classes=4,
        num_modalities=4,
        channels=[2, 4, 8, 16, 32],
        bottleneck_dropout=0.0,
        eval_slice_chunk_size=2,
    ).eval()
    with torch.no_grad():
        slice_logits = model({"signal": torch.randn(1, 4, 128, 128), "label": torch.zeros(1, 128, 128, dtype=torch.long), "id": [], "meta": []})
        volume_logits = model({"signal": torch.randn(1, 4, 3, 128, 128), "label": torch.zeros(1, 3, 128, 128, dtype=torch.long), "id": [], "meta": []})
    assert slice_logits.shape == (1, 4, 128, 128)
    assert volume_logits.shape == (1, 4, 3, 128, 128)
