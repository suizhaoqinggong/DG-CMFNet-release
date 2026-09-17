"""Integration tests for the framework template."""
import torch
import torch.nn as nn
from data_adapters.dummy_adapter import DummyDataAdapter
from framework.contracts.types import Batch
from framework.core.checkpoint import CheckpointManager
from framework.core.trainer import Trainer, TrainerSettings
from framework.logging.structured_logger import NullExperimentLogger
from framework.metrics.collection import MetricCollection
from framework.registry.defaults import create_default_registries
from framework.registry.factories import build_component_bundle, load_merged_experiment_config, load_toml
from models.simple_net import SimpleNet
from tasks.classification import ClassificationTask

def test_registry_creation():
    """Test that default registries are created and populated."""
    registries = create_default_registries()
    assert 'simple_net' in registries.models
    assert 'dgcmfnet' in registries.models
    assert 'dgcmfnet_V2' in registries.models
    assert 'classification' in registries.tasks
    assert 'dummy' in registries.data_adapters
    assert 'accuracy' in registries.metrics

def test_model_forward():
    """Test SimpleNet forward pass."""
    model = SimpleNet(num_classes=5, input_dim=128)
    batch = Batch(signal=torch.randn(4, 128), label=torch.randint(0, 2, (4, 5)).float(), id=['s1', 's2', 's3', 's4'], meta=[{}, {}, {}, {}])
    outputs = model(batch)
    assert outputs.shape == (4, 5)

def test_dgcmfnet_uses_six_unet_levels_and_fggim_on_every_encoder_stage():
    """DG-CMFNet should keep output size while using five encoder skips plus bottleneck."""
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_embed_dim=8, fg_gim_num_heads=2, cg_gfm_embed_dim=16, cg_gfm_num_heads=4, cg_gfm_num_layers=1, cg_gfm_patch_size=(1, 1, 1), cg_gfm_pos_grid_size=(1, 1, 1))
    model.eval()
    assert len(model.encoders[0].downs) == 5
    assert model.fg_gim_stages == [0, 1, 2, 3, 4]
    assert set(model.fg_gims.keys()) == {'0', '1', '2', '3', '4'}
    batch = Batch(signal=torch.randn(1, 2, 32, 32, 32), label=torch.zeros(1, 32, 32, 32).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 32, 32, 32)

def test_dgcmfnet_v2_returns_framework_logits_shape():
    """dgcmfnet_V2 should expose framework-native dense logits with all DG-CMFNet modules enabled."""
    from models.dgcmfnet import CoarseGrainedGlobalFusionModule, FineGrainedGraphInteractionModule
    from models.shared_encoder_concat_unet import DGCMFNetV2
    model = DGCMFNetV2(num_classes=3, num_modalities=2, base_ch=1, fg_gim_stages=[0, 4], fg_gim_embed_dim=8, fg_gim_num_heads=2, fg_gim_pool_size=(1, 1, 1), cg_gfm_embed_dim=8, cg_gfm_num_heads=2, cg_gfm_num_layers=1, cg_gfm_patch_size=(1, 1, 1), cg_gfm_pos_grid_size=(1, 1, 1))
    model.eval()
    assert set(model.fg_gims.keys()) == {'0', '4'}
    assert all((isinstance(module, FineGrainedGraphInteractionModule) for module in model.fg_gims.values()))
    assert isinstance(model.cg_gfm, CoarseGrainedGlobalFusionModule)
    assert model.decoder.frequency_stage_mapping == ((0, 3), (1, 4), (2, 5))
    assert set(model.decoder.frequency_guidance.keys()) == {'0', '1', '2'}
    batch = Batch(signal=torch.randn(1, 2, 33, 33, 33), label=torch.zeros(1, 33, 33, 33).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 33, 33, 33)

def test_dgcmfnet_v2_calls_fggim_cggfm_and_frequency_guidance():
    """dgcmfnet_V2 should route modality lists through all three DG-CMFNet module types."""
    from models.shared_encoder_concat_unet import DGCMFNetV2

    class RecordingListModule(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[list[tuple[int, ...]]] = []

        def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
            self.calls.append([tuple(feature.shape) for feature in features])
            return features

    class RecordingFrequencyModule(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def forward(self, boundary: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
            self.calls.append((tuple(boundary.shape), tuple(semantic.shape)))
            return boundary
    model = DGCMFNetV2(num_classes=3, num_modalities=2, base_ch=1, fg_gim_stages=[0, 5], fg_gim_embed_dim=8, fg_gim_num_heads=2, fg_gim_pool_size=(1, 1, 1), cg_gfm_embed_dim=8, cg_gfm_num_heads=2, cg_gfm_num_layers=1, cg_gfm_patch_size=(1, 1, 1), cg_gfm_pos_grid_size=(1, 1, 1))
    fg_stage0 = RecordingListModule()
    fg_stage5 = RecordingListModule()
    cg_gfm = RecordingListModule()
    frequency_recorders = [RecordingFrequencyModule() for _ in range(3)]
    model.fg_gims['0'] = fg_stage0
    model.fg_gims['5'] = fg_stage5
    model.cg_gfm = cg_gfm
    for (key, recorder) in zip(('0', '1', '2'), frequency_recorders):
        model.decoder.frequency_guidance[key] = recorder
    model.eval()
    batch = Batch(signal=torch.randn(1, 2, 33, 33, 33), label=torch.zeros(1, 33, 33, 33).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 33, 33, 33)
    assert fg_stage0.calls == [[(1, 1, 33, 33, 33), (1, 1, 33, 33, 33)]]
    assert fg_stage5.calls == [[(1, 32, 2, 2, 2), (1, 32, 2, 2, 2)]]
    assert cg_gfm.calls == [[(1, 32, 2, 2, 2), (1, 32, 2, 2, 2)]]
    assert [recorder.calls for recorder in frequency_recorders] == [[((1, 8, 9, 9, 9), (1, 64, 2, 2, 2))], [((1, 4, 17, 17, 17), (1, 32, 3, 3, 3))], [((1, 2, 33, 33, 33), (1, 16, 5, 5, 5))]]

def test_dgcmfnet_v2_can_use_compressed_fggim_auxiliary_skip_channels():
    """FG-GIM can preserve raw skip features without widening the decoder."""
    from models.shared_encoder_concat_unet import DGCMFNetV2
    model = DGCMFNetV2(num_classes=3, num_modalities=2, base_ch=1, fg_gim_stages=[0, 4], fg_gim_auxiliary_skip=True, fg_gim_embed_dim=8, fg_gim_num_heads=2, fg_gim_pool_size=(1, 1, 1), use_cg_gfm=False, decoder_type='standard')
    model.eval()
    assert model.fg_gim_auxiliary_skip is True
    assert model.fg_gim_auxiliary_stages == {0, 4}
    assert set(model.fg_gim_auxiliary_fusions.keys()) == {'0', '4'}
    assert [fusion.out_channels for fusion in model.level_fusions] == [2, 4, 8, 16, 32, 64]
    assert model.decoder.decoder_feature_channels == [64, 32, 16, 8, 4, 2]
    batch = Batch(signal=torch.randn(1, 2, 33, 33, 33), label=torch.zeros(1, 33, 33, 33).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 33, 33, 33)

def test_fggim_can_use_modality_embeddings_relation_bias_and_cross_modal_edges():
    """FG-GIM can expose modality identity and edge type to graph attention."""
    from models.dgcmfnet import FineGrainedGraphInteractionModule
    gim = FineGrainedGraphInteractionModule(in_channels=4, embed_dim=8, num_heads=2, pool_size=(2, 2, 1), window_size=(3, 3, 1), num_modalities=3, modality_aware=True, relation_bias=True, edge_mode='cross_modal')
    gim.eval()
    assert gim.modality_embedding is not None
    assert tuple(gim.modality_embedding.shape) == (3, 8)
    assert gim.relation_bias is not None
    assert tuple(gim.relation_bias.shape) == (3, 3, 2)
    mask = gim._build_graph_mask(3, torch.device('cpu'))
    assert mask.shape == (12, 12)
    assert not mask[:4, :4].any()
    assert mask[:4, 4:8].any()
    features = [torch.randn(1, 4, 5, 5, 3) for _ in range(3)]
    with torch.no_grad():
        outputs = gim(features)
    assert [output.shape for output in outputs] == [feature.shape for feature in features]

def test_dgcmfnet_v2_config_builds_registered_model():
    """The dgcmfnet_V2 TOML config should instantiate the enhanced model through the registry."""
    model_config = load_toml('configs/model.dgcmfnet_V2.toml')['model']
    model_name = model_config.pop('name')
    model = create_default_registries().models.create(model_name, **model_config)
    assert model_name == 'dgcmfnet_V2'
    assert model.fg_gim_stages == [2, 3, 4]
    assert model.fg_gim_auxiliary_skip is True
    assert model.fg_gim_auxiliary_stages == {2, 3, 4}
    assert set(model.fg_gim_auxiliary_fusions.keys()) == {'2', '3', '4'}
    assert model.fg_gims['4'].embed_dim == 256
    assert model.fg_gims['4'].pool_size == (8, 8, 8)
    assert model.use_cg_gfm is True
    assert model.skip_fusion == 'concat'
    assert model.unet_bottleneck_dropout == 0.2
    assert model.fg_gims['2'].modality_aware is True
    assert model.fg_gims['2'].use_relation_bias is True
    assert model.fg_gims['2'].edge_mode == 'all'
    assert tuple(model.fg_gims['2'].modality_embedding.shape) == (4, 64)
    assert tuple(model.fg_gims['2'].relation_bias.shape) == (4, 4, 4)
    assert model.decoder_frequency_position == 'decoder_feature'
    assert model.decoder.decoder_feature_channels == [1024, 512, 256, 128, 64, 32]
    assert model.decoder.frequency_stage_mapping == ((0, 3), (1, 4), (2, 5))
    assert model.decoder.frequency_guidance['0'].gate_mode == 'residual'

def test_shared_concat_decoder_can_apply_frequency_guidance_before_fusion():
    """Before-fusion FDFM should refine each selected skip before concatenation and convolution."""
    from models.shared_encoder_concat_unet import SharedConcatDecoder
    decoder = SharedConcatDecoder(num_classes=4, num_modalities=2, base_ch=1, decoder_type='frequency_guided', frequency_position='before_fusion', frequency_gate_mode='residual')
    recorders = []
    for key in ('0', '1', '2'):
        recorder = _RecordingGuidance()
        decoder.frequency_guidance[key] = recorder
        recorders.append(recorder)
    decoder_input = [torch.randn(1, 64, 1, 1, 1), torch.randn(1, 32, 2, 2, 2), torch.randn(1, 16, 4, 4, 4), torch.randn(1, 8, 8, 8, 8), torch.randn(1, 4, 16, 16, 16), torch.randn(1, 2, 32, 32, 32)]
    decoder.eval()
    with torch.no_grad():
        logits = decoder(decoder_input)
    assert decoder.frequency_position == 'before_fusion'
    assert decoder.frequency_stage_mapping == ((0, 2), (1, 3), (2, 4))
    assert logits.shape == (1, 4, 32, 32, 32)
    assert [recorder.calls for recorder in recorders] == [[((1, 8, 8, 8, 8), (1, 64, 1, 1, 1))], [((1, 4, 16, 16, 16), (1, 32, 2, 2, 2))], [((1, 2, 32, 32, 32), (1, 16, 4, 4, 4))]]

def test_dgcmfnet_dropout_parameter_reaches_fggim_and_cggfm():
    """DG-CMFNet dropout config should activate MC-dropout-capable modules."""
    from torch import nn
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[0], fg_gim_embed_dim=8, fg_gim_num_heads=2, cg_gfm_embed_dim=16, cg_gfm_num_heads=4, cg_gfm_num_layers=1, cg_gfm_patch_size=(1, 1, 1), cg_gfm_pos_grid_size=(1, 1, 1), dropout=0.25)
    assert isinstance(model.fg_gims['0'].dropout, nn.Dropout)
    assert model.fg_gims['0'].dropout.p == 0.25
    block = model.cg_gfm.transformer_blocks[0]
    assert block.attn.dropout == 0.25
    assert [module.p for module in block.mlp if isinstance(module, nn.Dropout)] == [0.25, 0.25]

def test_dgcmfnet_unet_bottleneck_dropout_uses_dropout3d_independently():
    """UNet backbone MC dropout should use spatial 3D dropout separately from graph/transformer dropout."""
    from torch import nn
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[], cg_gfm_embed_dim=16, cg_gfm_num_heads=4, cg_gfm_num_layers=1, cg_gfm_patch_size=(1, 1, 1), cg_gfm_pos_grid_size=(1, 1, 1), dropout=0.25, unet_bottleneck_dropout=0.1)
    assert model.unet_bottleneck_dropout == 0.1
    assert all((isinstance(encoder.bottleneck_dropout, nn.Dropout3d) for encoder in model.encoders))
    assert [encoder.bottleneck_dropout.p for encoder in model.encoders] == [0.1, 0.1]
    block = model.cg_gfm.transformer_blocks[0]
    assert block.attn.dropout == 0.25
    assert [module.p for module in block.mlp if isinstance(module, nn.Dropout)] == [0.25, 0.25]
    default_model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[], cg_gfm_embed_dim=16, cg_gfm_num_heads=4, cg_gfm_num_layers=1, cg_gfm_patch_size=(1, 1, 1), cg_gfm_pos_grid_size=(1, 1, 1))
    assert all((isinstance(encoder.bottleneck_dropout, nn.Identity) for encoder in default_model.encoders))

def test_dgcmfnet_can_disable_cggfm_for_ablation():
    """CG-GFM can be removed entirely for ablation runs."""
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[], use_cg_gfm=False, decoder_type='standard')
    model.eval()
    assert model.use_cg_gfm is False
    assert model.cg_gfm is None
    assert not any((name.startswith('cg_gfm.') for (name, _) in model.named_parameters()))
    batch = Batch(signal=torch.randn(1, 2, 32, 32, 32), label=torch.zeros(1, 32, 32, 32).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 32, 32, 32)

def test_dgcmfnet_supports_concat_1x1_skip_fusion():
    """Same-stage modality skips can be fused by learnable concat + 1x1 conv."""
    from torch import nn
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[], use_cg_gfm=False, decoder_type='standard', skip_fusion='concat_1x1')
    model.eval()
    assert model.skip_fusion_mode == 'concat_1x1'
    assert [fusion.mode for fusion in model.skip_fusions] == ['concat_1x1'] * 5
    first_projection = model.skip_fusions[0].proj[0]
    assert isinstance(first_projection, nn.Conv3d)
    assert first_projection.in_channels == 4
    assert first_projection.out_channels == 2
    batch = Batch(signal=torch.randn(1, 2, 32, 32, 32), label=torch.zeros(1, 32, 32, 32).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 32, 32, 32)

def test_dgcmfnet_supports_raw_concat_skip_fusion():
    """DGCMFNet can keep baseline-style raw modality concatenation into the decoder."""
    from torch import nn
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[], use_cg_gfm=False, decoder_type='standard', skip_fusion='concat')
    model.eval()
    assert model.skip_fusion_mode == 'concat'
    assert [fusion.out_channels for fusion in model.skip_fusions] == [4, 8, 16, 32, 64]
    assert isinstance(model.bottleneck_fusion, nn.Identity)
    assert model.bottleneck_out_channels == 128
    assert model.decoder.decoder_feature_channels == [128, 64, 32, 16, 8, 4]
    batch = Batch(signal=torch.randn(1, 2, 32, 32, 32), label=torch.zeros(1, 32, 32, 32).long(), id=['s1'], meta=[{}])
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.shape == (1, 3, 32, 32, 32)

def test_dgcmfnet_rejects_unknown_skip_fusion():
    """Skip fusion config should fail fast for unsupported modes."""
    import pytest
    from models.dgcmfnet import DGCMFNet
    with pytest.raises(ValueError, match='skip_fusion'):
        DGCMFNet(num_classes=3, num_modalities=2, base_ch=2, fg_gim_stages=[], use_cg_gfm=False, decoder_type='standard', skip_fusion='attention')

def test_dgcmfnet_accepts_whiteboard_fggim_parameters():
    """Whiteboard FG-GIM setup uses C_l=8, N=512, and C'=32 at L1."""
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=4, num_modalities=4, base_ch=8, fg_gim_stages=[0], fg_gim_embed_dim=32, fg_gim_num_heads=4, fg_gim_pool_size=(8, 8, 8), cg_gfm_embed_dim=64, cg_gfm_num_heads=4, cg_gfm_num_layers=1)
    assert model.fg_gim_stages == [0]
    assert model.fg_gims['0'].in_channels == 8
    assert model.fg_gims['0'].embed_dim == 32
    assert model.fg_gims['0'].pool_size == (8, 8, 8)
    assert model.fg_gims['0'].num_nodes == 512

def test_dgcmfnet_accepts_recommended_stagewise_fggim_parameters():
    """Recommended FG-GIM setup uses stage-specific C' and pooled node counts."""
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(num_classes=4, num_modalities=4, base_ch=8, fg_gim_stages=[0, 1, 2, 3, 4, 5], fg_gim_embed_dims=[32, 64, 128, 128, 256, 256], fg_gim_num_heads=4, fg_gim_pool_sizes=[(8, 8, 8), (8, 8, 8), (4, 4, 4), (4, 4, 4), (2, 2, 2), (2, 2, 2)], cg_gfm_embed_dim=256, cg_gfm_num_heads=8, cg_gfm_num_layers=1)
    expected_channels = [8, 16, 32, 64, 128, 256]
    expected_embed_dims = [32, 64, 128, 128, 256, 256]
    expected_pool_sizes = [(8, 8, 8), (8, 8, 8), (4, 4, 4), (4, 4, 4), (2, 2, 2), (2, 2, 2)]
    assert model.fg_gim_stages == [0, 1, 2, 3, 4, 5]
    assert set(model.fg_gims.keys()) == {'0', '1', '2', '3', '4', '5'}
    assert [model.fg_gims[str(stage)].in_channels for stage in model.fg_gim_stages] == expected_channels
    assert [model.fg_gims[str(stage)].embed_dim for stage in model.fg_gim_stages] == expected_embed_dims
    assert [model.fg_gims[str(stage)].pool_size for stage in model.fg_gim_stages] == expected_pool_sizes
    assert [model.fg_gims[str(stage)].num_nodes for stage in model.fg_gim_stages] == [512, 512, 64, 64, 8, 8]

def test_data_adapter():
    """Test DummyDataAdapter produces valid data."""
    adapter = DummyDataAdapter(num_samples=100, num_classes=3, input_dim=64)
    adapter.prepare()
    (train_ds, val_ds, test_ds) = adapter.get_splits()
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    sample = train_ds[0]
    assert sample['signal'].shape == (64,)
    assert sample['label'].shape == (3,)
    batch = adapter.collate_fn([train_ds[0], train_ds[1]])
    assert batch['signal'].shape == (2, 64)
    assert batch['label'].shape == (2, 3)

def test_classification_task():
    """Test ClassificationTask computes loss correctly."""
    task = ClassificationTask(num_classes=5, loss='bce')
    logits = torch.randn(4, 5)
    labels = torch.randint(0, 2, (4, 5)).float()
    batch = Batch(signal=torch.randn(4, 128), label=labels, id=['s1', 's2', 's3', 's4'], meta=[{}, {}, {}, {}])
    loss = task.compute_loss(logits, batch)
    assert loss.item() >= 0
    assert task.infer_problem_type() == 'multilabel'

def test_focal_loss():
    """Test FocalLoss runs without errors."""
    from framework.losses.focal import FocalLoss
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0

def test_tensorboard_compat_installs_pkg_resources_shim():
    """TensorBoard compat entry should not require setuptools pkg_resources."""
    import sys
    from framework.cli.tensorboard_compat import install_pkg_resources_shim
    previous = sys.modules.pop('pkg_resources', None)
    try:
        install_pkg_resources_shim(force=True)
        shim = sys.modules['pkg_resources']
        assert list(shim.iter_entry_points('tensorboard_plugins')) == []
        assert shim.parse_version('2.0') > shim.parse_version('1.0')
    finally:
        if previous is None:
            sys.modules.pop('pkg_resources', None)
        else:
            sys.modules['pkg_resources'] = previous

def test_uahl_focal_term_uses_target_class_probabilities():
    """UAHL focal term should decrease when target-class probabilities improve."""
    from framework.losses.uahl import UncertaintyAwareHybridLoss
    loss_fn = UncertaintyAwareHybridLoss(num_classes=4, gamma=2.0)
    targets = torch.tensor([[[[0, 2]]]])
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()
    uniform_probs = torch.full((1, 4, 1, 1, 2), 0.25)
    confident_probs = torch.tensor([[[[[0.9, 0.02]]], [[[0.04, 0.03]]], [[[0.03, 0.92]]], [[[0.03, 0.03]]]]])
    assert loss_fn._focal_term(confident_probs, targets_onehot) < loss_fn._focal_term(uniform_probs, targets_onehot)

def test_softmax_focal_loss_matches_uahl_focal_component():
    """Focal-only ablations must use exactly the focal term from U-AHL."""
    from framework.losses.uahl import SoftmaxFocalLoss, UncertaintyAwareHybridLoss
    logits = torch.randn(2, 4, 1, 2, 2)
    targets = torch.randint(0, 4, (2, 1, 2, 2))
    focal = SoftmaxFocalLoss(num_classes=4, gamma=2.0)
    uahl = UncertaintyAwareHybridLoss(num_classes=4, gamma=2.0)
    focal_loss = focal(logits, targets)
    uahl(logits, targets)
    assert torch.isclose(focal_loss.detach(), uahl.get_last_components()['loss_focal'])
    assert set(focal.get_last_components()) == {'loss_focal'}

def test_uahl_entropy_term_uses_standard_predictive_entropy():
    """UAHL entropy should sum over classes before averaging voxels."""
    from framework.losses.uahl import UncertaintyAwareHybridLoss
    loss_fn = UncertaintyAwareHybridLoss(num_classes=4)
    uniform_probs = torch.full((1, 4, 1, 1, 2), 0.25)
    assert torch.isclose(loss_fn._entropy_term(uniform_probs), torch.log(torch.tensor(4.0)))

def test_uahl_records_loss_components_after_forward():
    """UAHL should expose the latest raw component values for logging."""
    from framework.losses.uahl import UncertaintyAwareHybridLoss
    loss_fn = UncertaintyAwareHybridLoss(num_classes=4, lambda_alpha=1.5, lambda_beta=0.25, gamma=2.0)
    logits = torch.randn(2, 4, 1, 2, 2)
    targets = torch.randint(0, 4, (2, 1, 2, 2))
    loss = loss_fn(logits, targets)
    components = loss_fn.get_last_components()
    assert set(components) == {'loss_dice', 'loss_focal', 'loss_entropy'}
    expected = components['loss_dice'] + 1.5 * components['loss_focal'] + 0.25 * components['loss_entropy']
    assert abs(loss.item() - expected) < 1e-06
    assert all((value >= 0 for value in components.values()))

def test_softmax_focal_loss_matches_uahl_focal_component_with_gradients():
    """Focal-only ablations should reuse the exact focal term reported by U-AHL."""
    from framework.losses.uahl import SoftmaxFocalLoss, UncertaintyAwareHybridLoss
    logits = torch.tensor([[[[[3.0, -1.0]]], [[[0.0, 0.5]]], [[[-1.0, 2.5]]], [[[-2.0, 0.0]]]]], requires_grad=True)
    targets = torch.tensor([[[[0, 2]]]])
    focal_only = SoftmaxFocalLoss(num_classes=4, gamma=2.0)
    uahl = UncertaintyAwareHybridLoss(num_classes=4, gamma=2.0)
    actual = focal_only(logits, targets)
    (probs, targets_onehot) = uahl._prepare_inputs(logits, targets)
    expected = uahl._focal_term(probs, targets_onehot)
    assert torch.allclose(actual, expected)
    assert set(focal_only.get_last_components()) == {'loss_focal'}
    actual.backward()
    assert logits.grad is not None

def test_brats_label_remap_preserves_gli_label_3_as_et():
    """BraTS GLI uses label 3 for ET, while older BraTS variants may use label 4."""
    import numpy as np
    from data_adapters.brats_adapter import BRATS_LABEL_REMAP
    from scripts.preprocess_brats import _remap_labels
    labels = np.array([0, 1, 2, 3, 4])
    assert _remap_labels(labels).tolist() == [0, 1, 2, 3, 3]
    assert BRATS_LABEL_REMAP[3] == 3

def test_brats_split_keeps_related_suffixes_together_and_allows_empty_test(tmp_path):
    """BraTS split should keep -000/-001 variants of the same subject in one split."""
    from data_adapters.brats_adapter import BraTSDataAdapter, brats_subject_id
    case_ids = ['BraTS-GLI-00001-000', 'BraTS-GLI-00001-001', 'BraTS-GLI-00002-000', 'BraTS-GLI-00002-001', 'BraTS-GLI-00003-000', 'BraTS-GLI-00003-001', 'BraTS-GLI-00004-000', 'BraTS-GLI-00004-001']
    for case_id in case_ids:
        (tmp_path / case_id).mkdir()
    adapter = BraTSDataAdapter(root_dir=str(tmp_path), val_ratio=0.5, test_ratio=0.0, seed=0, use_preprocessed=True, preprocessed_dir=str(tmp_path))
    adapter.prepare()
    (train_ds, val_ds, test_ds) = adapter.get_splits()
    train_subjects = {brats_subject_id(case_id) for case_id in train_ds.case_ids}
    val_subjects = {brats_subject_id(case_id) for case_id in val_ds.case_ids}
    assert len(test_ds) == 0
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects | val_subjects == {'BraTS-GLI-00001', 'BraTS-GLI-00002', 'BraTS-GLI-00003', 'BraTS-GLI-00004'}

def test_brats_split_uses_fixed_validation_ids_file(tmp_path):
    from data_adapters.brats_adapter import BraTSDataAdapter
    case_ids = ['BraTS20_Training_001', 'BraTS20_Training_002', 'BraTS20_Training_003', 'BraTS20_Training_004']
    for case_id in case_ids:
        (tmp_path / case_id).mkdir()
    val_ids_file = tmp_path / 'val_ids.txt'
    val_ids_file.write_text('BraTS20_Training_002\nBraTS20_Training_004\n')
    adapter = BraTSDataAdapter(root_dir=str(tmp_path), use_preprocessed=True, preprocessed_dir=str(tmp_path), val_ratio=0.0, test_ratio=0.0, val_ids_file=str(val_ids_file))
    adapter.prepare()
    (train_ds, val_ds, test_ds) = adapter.get_splits()
    assert train_ds.case_ids == ['BraTS20_Training_001', 'BraTS20_Training_003']
    assert val_ds.case_ids == ['BraTS20_Training_002', 'BraTS20_Training_004']
    assert test_ds.case_ids == []

def test_experiment_seed_controls_data_adapter_when_data_seed_is_missing():
    """The experiment seed should drive data splitting unless [data].seed overrides it."""
    config = {'experiment': {'name': 'seed-test', 'seed': 123, 'class_names': ['negative', 'positive']}, 'data': {'adapter': 'dummy', 'num_samples': 20, 'input_dim': 8, 'batch_size': 2, 'num_workers': 0}, 'model': {'name': 'simple_net', 'input_dim': 8}, 'task': {'name': 'classification'}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'metrics': {'params': {}}}
    bundle = build_component_bundle(config, create_default_registries())
    assert bundle.data_adapter.seed == 123

def test_data_seed_controls_data_adapter_when_experiment_seed_is_missing():
    """A fixed [data].seed should keep splits stable while training seed is unset."""
    config = {'experiment': {'name': 'seed-test', 'class_names': ['negative', 'positive']}, 'data': {'adapter': 'dummy', 'seed': 37, 'num_samples': 20, 'input_dim': 8, 'batch_size': 2, 'num_workers': 0}, 'model': {'name': 'simple_net', 'input_dim': 8}, 'task': {'name': 'classification'}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'metrics': {'params': {}}}
    bundle = build_component_bundle(config, create_default_registries())
    assert bundle.data_adapter.seed == 37

def test_trainer_seed_defaults_to_unset():
    """Training should not set global RNG state when [experiment].seed is omitted."""
    assert TrainerSettings().seed is None

def test_distributed_context_detects_torchrun_environment(monkeypatch):
    """torchrun environment variables should enable DDP context detection."""
    from framework.core.distributed import infer_distributed_context
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setenv('WORLD_SIZE', '2')
    context = infer_distributed_context()
    assert context.enabled
    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert not context.is_main_process

def test_component_bundle_uses_distributed_sampler_for_training_only():
    """DDP should shard training data while leaving validation/test loaders complete."""
    from torch.utils.data.distributed import DistributedSampler
    from framework.core.distributed import DistributedContext
    config = {'experiment': {'name': 'ddp-test', 'seed': 7, 'class_names': ['a', 'b']}, 'data': {'adapter': 'dummy', 'num_samples': 20, 'input_dim': 8, 'batch_size': 2, 'num_workers': 0}, 'model': {'name': 'simple_net', 'input_dim': 8}, 'task': {'name': 'classification'}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'metrics': {'params': {}}}
    context = DistributedContext(enabled=True, rank=1, local_rank=1, world_size=2, device=torch.device('cpu'))
    bundle = build_component_bundle(config, create_default_registries(), distributed_context=context)
    assert isinstance(bundle.train_loader.sampler, DistributedSampler)
    assert bundle.train_loader.sampler.rank == 1
    assert bundle.train_loader.sampler.num_replicas == 2
    assert not isinstance(bundle.val_loader.sampler, DistributedSampler)
    assert not isinstance(bundle.test_loader.sampler, DistributedSampler)

def test_missing_experiment_seed_uses_broadcast_sampler_seed_in_ddp(monkeypatch):
    """DDP sampler shuffling should stay consistent across ranks without a training seed."""
    from torch.utils.data.distributed import DistributedSampler
    import framework.registry.factories as factories
    from framework.core.distributed import DistributedContext
    monkeypatch.setattr(factories.random, 'randrange', lambda upper: 9876)
    monkeypatch.setattr(factories, 'broadcast_object', lambda value, context: value)
    config = {'experiment': {'name': 'ddp-test', 'class_names': ['a', 'b']}, 'data': {'adapter': 'dummy', 'seed': 37, 'num_samples': 20, 'input_dim': 8, 'batch_size': 2, 'num_workers': 0}, 'model': {'name': 'simple_net', 'input_dim': 8}, 'task': {'name': 'classification'}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'metrics': {'params': {}}}
    context = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2, device=torch.device('cpu'))
    bundle = build_component_bundle(config, create_default_registries(), distributed_context=context)
    assert bundle.data_adapter.seed == 37
    assert isinstance(bundle.train_loader.sampler, DistributedSampler)
    assert bundle.train_loader.sampler.seed == 9876

def test_average_metric_dict_returns_averaged_metrics(monkeypatch):
    """DDP metric averaging should return a metrics dict, not mutate and drop it."""
    import framework.core.distributed as distributed
    from framework.core.distributed import DistributedContext, average_metric_dict

    def fake_all_reduce(values, op=None):
        values.mul_(2)
    monkeypatch.setattr(distributed.dist, 'all_reduce', fake_all_reduce)
    context = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2, device=torch.device('cpu'))
    averaged = average_metric_dict({'loss': 0.5, 'accuracy': 0.25}, context)
    assert averaged == {'loss': 0.5, 'accuracy': 0.25}

def test_snapshot_split_ids_writes_dataset_case_ids(tmp_path):
    """Run directories should contain exact split membership for reproducibility."""
    from framework.core.run_layout import snapshot_split_ids

    class DatasetWithIds:

        def __init__(self, case_ids):
            self.case_ids = case_ids

    class AdapterWithSplits:

        def get_splits(self):
            return (DatasetWithIds(['train-a', 'train-b']), DatasetWithIds(['val-a']), DatasetWithIds([]))
    snapshot_split_ids(AdapterWithSplits(), tmp_path)
    assert (tmp_path / 'splits' / 'train_ids.txt').read_text().splitlines() == ['train-a', 'train-b']
    assert (tmp_path / 'splits' / 'val_ids.txt').read_text().splitlines() == ['val-a']
    assert (tmp_path / 'splits' / 'test_ids.txt').read_text().splitlines() == []

def test_cg_gfm_positional_embedding_is_registered_before_forward():
    """CG-GFM positional encoding must be optimized from the first step."""
    from models.dgcmfnet import DGCMFNet
    model = DGCMFNet(base_ch=4, fg_gim_embed_dim=16, fg_gim_num_heads=4, cg_gfm_embed_dim=32, cg_gfm_num_heads=4, cg_gfm_num_layers=1)
    params = dict(model.named_parameters())
    assert 'cg_gfm.pos_embed' in params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
    assert any((params['cg_gfm.pos_embed'] is p for p in optimizer_params))

def test_frequency_guidance_uses_semantic_low_to_enhance_boundary_high():
    """Deep low-frequency semantics should gate shallow high-frequency detail."""
    from models.dgcmfnet import FrequencyGuidance3d, _low_pass3d
    guidance = FrequencyGuidance3d(semantic_ch=1, boundary_ch=1, frequency_kernel_size=3, gate_mode='multiplicative')
    boundary = torch.arange(27, dtype=torch.float32).view(1, 1, 3, 3, 3)
    semantic = torch.ones(1, 1, 1, 1, 1)
    with torch.no_grad():
        enhanced = guidance(boundary, semantic)
    boundary_low = _low_pass3d(boundary, 3)
    boundary_high = boundary - boundary_low
    expected = boundary_low + boundary_high * 0.5
    assert torch.allclose(enhanced, expected)

def test_frequency_guidance_positive_gate_bias_preserves_more_boundary_high_frequency():
    """A positive multiplicative gate bias should start closer to identity high-frequency transfer."""
    from models.dgcmfnet import FrequencyGuidance3d, _low_pass3d
    guidance = FrequencyGuidance3d(semantic_ch=1, boundary_ch=1, frequency_kernel_size=3, gate_mode='multiplicative', gate_init_bias=2.0)
    boundary = torch.arange(27, dtype=torch.float32).view(1, 1, 3, 3, 3)
    semantic = torch.ones(1, 1, 1, 1, 1)
    with torch.no_grad():
        enhanced = guidance(boundary, semantic)
    boundary_low = _low_pass3d(boundary, 3)
    boundary_high = boundary - boundary_low
    expected = boundary_low + boundary_high * torch.sigmoid(torch.tensor(2.0))
    assert torch.allclose(enhanced, expected)

def test_frequency_guidance_residual_gate_matches_paper_formula():
    """Residual FDFM should reconstruct the shallow feature from its low-pass base and tanh-gated detail."""
    from models.dgcmfnet import FrequencyGuidance3d, _low_pass3d
    guidance = FrequencyGuidance3d(semantic_ch=1, boundary_ch=1, frequency_kernel_size=3, gate_mode='residual', gate_init_bias=0.75)
    boundary = torch.arange(27, dtype=torch.float32).view(1, 1, 3, 3, 3)
    semantic = torch.ones(1, 1, 1, 1, 1)
    with torch.no_grad():
        enhanced = guidance(boundary, semantic)
    boundary_low = _low_pass3d(boundary, 3)
    boundary_high = boundary - boundary_low
    expected = boundary_low + boundary_high * torch.tanh(torch.tensor(0.75))
    assert torch.allclose(enhanced, expected)

def test_shared_decoder_frequency_guidance_defaults_to_paper_residual_mode():
    """FDFM should default to the paper-defined tanh-gated low-pass reconstruction."""
    from models.dgcmfnet import SharedDecoder
    decoder = SharedDecoder(base_ch=2, out_ch=4, decoder_type='frequency_guided')
    assert len(decoder.frequency_guidance) > 0
    for guidance in decoder.frequency_guidance.values():
        assert guidance.gate_mode == 'residual'
        assert torch.allclose(guidance.semantic_gate.bias, torch.zeros_like(guidance.semantic_gate.bias))

def test_frequency_guided_decoder_uses_six_decoder_side_features_by_default():
    """Default FDFM treats the bottleneck as F1 and refines decoder outputs F4-F6."""
    (decoder, calls, logits) = _run_recording_decoder()
    assert decoder.frequency_stage_mapping == ((0, 3), (1, 4), (2, 5))
    assert decoder.frequency_position == 'decoder_feature'
    assert logits.shape == (1, 4, 32, 32, 32)
    assert calls == [[((1, 8, 8, 8, 8), (1, 64, 1, 1, 1))], [((1, 4, 16, 16, 16), (1, 32, 2, 2, 2))], [((1, 2, 32, 32, 32), (1, 16, 4, 4, 4))]]

def test_frequency_guided_decoder_uses_cross_layer_mapping():
    """Frequency-guided decoder should map deep semantic states to shallow skips."""
    from models.dgcmfnet import Up3d
    (decoder, calls, logits) = _run_recording_decoder('pre_concat_skip')
    assert len(decoder.ups) == 5
    assert all((isinstance(up, Up3d) for up in decoder.ups))
    assert decoder.frequency_stage_mapping == ((0, 2), (1, 3), (2, 4))
    assert decoder.frequency_position == 'pre_concat_skip'
    assert logits.shape == (1, 4, 32, 32, 32)
    assert calls == [[((1, 8, 8, 8, 8), (1, 64, 1, 1, 1))], [((1, 4, 16, 16, 16), (1, 32, 2, 2, 2))], [((1, 2, 32, 32, 32), (1, 16, 4, 4, 4))]]

def test_frequency_guided_decoder_can_run_after_concat_before_conv():
    """Frequency guidance can refine concat features before decoder convolution."""
    (decoder, calls, logits) = _run_recording_decoder('post_concat_pre_conv')
    assert decoder.frequency_position == 'post_concat_pre_conv'
    assert logits.shape == (1, 4, 32, 32, 32)
    assert calls == [[((1, 16, 8, 8, 8), (1, 64, 1, 1, 1))], [((1, 8, 16, 16, 16), (1, 32, 2, 2, 2))], [((1, 4, 32, 32, 32), (1, 16, 4, 4, 4))]]

def test_frequency_guided_decoder_can_run_after_decoder_conv():
    """Frequency guidance can refine each selected decoder block output."""
    (decoder, calls, logits) = _run_recording_decoder('post_conv')
    assert decoder.frequency_position == 'post_conv'
    assert logits.shape == (1, 4, 32, 32, 32)
    assert calls == [[((1, 8, 8, 8, 8), (1, 64, 1, 1, 1))], [((1, 4, 16, 16, 16), (1, 32, 2, 2, 2))], [((1, 2, 32, 32, 32), (1, 16, 4, 4, 4))]]

class _RecordingGuidance(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, boundary: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        self.calls.append((tuple(boundary.shape), tuple(semantic.shape)))
        return boundary

def test_shared_concat_decoder_can_guide_skips_before_fusion():
    """The V2 decoder should apply cross-layer guidance to skips before concatenation."""
    from models.shared_encoder_concat_unet import SharedConcatDecoder
    decoder = SharedConcatDecoder(num_classes=4, base_ch=1, level_channels=[1, 2, 4, 8, 16, 32], decoder_type='frequency_guided', frequency_position='before_fusion', frequency_stage_mapping=[[0, 2], [1, 3], [2, 4]], frequency_gate_mode='residual')
    recorders = []
    for key in ('0', '1', '2'):
        recorder = _RecordingGuidance()
        decoder.frequency_guidance[key] = recorder
        recorders.append(recorder)
    decoder_input = [torch.randn(1, 32, 1, 1, 1), torch.randn(1, 16, 2, 2, 2), torch.randn(1, 8, 4, 4, 4), torch.randn(1, 4, 8, 8, 8), torch.randn(1, 2, 16, 16, 16), torch.randn(1, 1, 32, 32, 32)]
    decoder.eval()
    with torch.no_grad():
        logits = decoder(decoder_input)
    assert decoder.frequency_position == 'before_fusion'
    assert decoder.frequency_stage_mapping == ((0, 2), (1, 3), (2, 4))
    assert logits.shape == (1, 4, 32, 32, 32)
    assert [recorder.calls for recorder in recorders] == [[((1, 4, 8, 8, 8), (1, 32, 1, 1, 1))], [((1, 2, 16, 16, 16), (1, 16, 2, 2, 2))], [((1, 1, 32, 32, 32), (1, 8, 4, 4, 4))]]

def _run_recording_decoder(frequency_position=None) -> tuple[torch.nn.Module, list[list[tuple[tuple[int, ...], tuple[int, ...]]]], torch.Tensor]:
    from models.dgcmfnet import SharedDecoder
    decoder_kwargs = {'base_ch': 2, 'out_ch': 4, 'decoder_type': 'frequency_guided'}
    if frequency_position is not None:
        decoder_kwargs['frequency_position'] = frequency_position
    decoder = SharedDecoder(**decoder_kwargs)
    recorders = []
    for key in ('0', '1', '2'):
        recorder = _RecordingGuidance()
        decoder.frequency_guidance[key] = recorder
        recorders.append(recorder)
    x = torch.randn(1, 64, 1, 1, 1)
    skips = [torch.randn(1, 2, 32, 32, 32), torch.randn(1, 4, 16, 16, 16), torch.randn(1, 8, 8, 8, 8), torch.randn(1, 16, 4, 4, 4), torch.randn(1, 32, 2, 2, 2)]
    decoder.eval()
    with torch.no_grad():
        logits = decoder(x, skips)
    return (decoder, [recorder.calls for recorder in recorders], logits)

def test_brats_region_dice_uses_wt_tc_et_regions():
    """BraTS Dice should report WT, TC, and ET composite regions."""
    from framework.metrics.brats import BraTSRegionDiceMetric
    targets = torch.tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    preds = torch.nn.functional.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()
    metric = BraTSRegionDiceMetric(num_classes=4)
    metric.update(preds, targets)
    results = metric.compute()
    assert results['wt'] == 1.0
    assert results['tc'] == 1.0
    assert results['et'] == 1.0
    assert results['mean'] == 1.0

def test_brats_region_hausdorff_is_zero_for_exact_match():
    """BraTS HD should be zero when WT, TC, and ET masks match exactly."""
    from framework.metrics.brats import BraTSRegionHausdorffMetric
    targets = torch.tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    preds = torch.nn.functional.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()
    metric = BraTSRegionHausdorffMetric(num_classes=4)
    metric.update(preds, targets)
    results = metric.compute()
    assert results['wt'] == 0.0
    assert results['tc'] == 0.0
    assert results['et'] == 0.0
    assert results['mean'] == 0.0

def test_metric_collection():
    """Test metric collection computes and resets."""
    from framework.metrics.accuracy import AccuracyMetric
    metric = AccuracyMetric(threshold=0.5)
    preds = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metric.update(preds, targets)
    results = metric.compute()
    assert 'accuracy' in results
    assert results['accuracy'] == 1.0
    metric.reset()
    results = metric.compute()
    assert results['accuracy'] == 0.0

def test_checkpoint_manager(tmp_path):
    """Test checkpoint save and load."""
    ckpt_dir = tmp_path / 'checkpoints'
    manager = CheckpointManager(ckpt_dir, monitor='loss', mode='min')
    state = {'weight': torch.tensor([1.0])}
    metrics = {'loss': 0.5}
    manager.save(state, epoch=1, metrics=metrics, is_best=True)
    assert (ckpt_dir / 'last.pt').exists()
    assert (ckpt_dir / 'best.pt').exists()
    payload = manager.load()
    assert payload['epoch'] == 1
    assert payload['metrics']['loss'] == 0.5

def test_checkpoint_manager_tracks_explicit_monitor_value(tmp_path):
    """Best checkpoint tracking should work when the saved metrics use an unprefixed key."""
    ckpt_dir = tmp_path / 'checkpoints'
    manager = CheckpointManager(ckpt_dir, monitor='val_score', mode='max')
    state = {'weight': torch.tensor([1.0])}
    first_is_best = manager.is_improved(0.5)
    manager.save(state, epoch=1, metrics={'score': 0.5}, is_best=first_is_best, monitor_value=0.5)
    second_is_best = manager.is_improved(0.4)
    manager.save(state, epoch=2, metrics={'score': 0.4}, is_best=second_is_best, monitor_value=0.4)
    assert manager.load()['epoch'] == 1
    assert manager.load(ckpt_dir / 'last.pt')['epoch'] == 2

def test_trainer_prints_and_logs_uahl_loss_components(tmp_path, capsys):
    """Each epoch should expose train/val UAHL components in console and logger output."""
    from tasks.segmentation import SegmentationTask

    class TinySegmentationModel(torch.nn.Module):

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, num_classes, 1, 1, 2))

        def forward(self, batch: Batch) -> torch.Tensor:
            batch_size = batch['label'].shape[0]
            return self.logits.expand(batch_size, -1, -1, -1, -1)

    class RecordingLogger:

        def __init__(self) -> None:
            self.records = []

        def start_run(self, run_name, config):
            pass

        def log_params(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            self.records.append((step, dict(metrics)))

        def log_histogram(self, name, values, step=None):
            pass

        def log_figure(self, name, figure, step=None):
            pass

        def end_run(self):
            pass

    def batch(values: list[int]) -> Batch:
        labels = torch.tensor(values, dtype=torch.long).reshape(1, 1, 1, 2)
        return Batch(signal=torch.zeros(1, 1), label=labels, id=['case'], meta=[{}])
    logger = RecordingLogger()
    model = TinySegmentationModel(num_classes=4)
    task = SegmentationTask(num_classes=4, loss='uahl', loss_lambda_alpha=1.0, loss_lambda_beta=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model=model, task=task, optimizer=optimizer, train_loader=[batch([0, 1]), batch([2, 3])], val_loader=[batch([1, 3])], test_loader=[batch([0, 3])], train_metrics=MetricCollection([]), val_metrics=MetricCollection([]), test_metrics=MetricCollection([]), settings=TrainerSettings(epochs=1, patience=5, device='cpu'), checkpoint_manager=CheckpointManager(tmp_path / 'ckpt', monitor='loss', mode='min'), logger=logger, run_dir=tmp_path)
    trainer.train()
    output = capsys.readouterr().out
    for key in ('loss_dice', 'loss_focal', 'loss_entropy'):
        assert f'train_{key}=' in output
        assert f'val_{key}=' in output
    logged_metrics = {}
    for (_, record) in logger.records:
        logged_metrics.update(record)
    for split in ('train', 'val'):
        for key in ('loss_dice', 'loss_focal', 'loss_entropy'):
            assert f'{split}/{key}' in logged_metrics

def test_trainer_runs_one_epoch(tmp_path):
    """Test that Trainer can run for one epoch without crashing."""
    model = SimpleNet(num_classes=3, input_dim=64)
    task = ClassificationTask(num_classes=3, loss='bce')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    adapter = DummyDataAdapter(num_samples=32, num_classes=3, input_dim=64)
    adapter.prepare()
    (train_ds, val_ds, test_ds) = adapter.get_splits()
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=8, collate_fn=adapter.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=adapter.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, collate_fn=adapter.collate_fn)
    from framework.metrics.accuracy import AccuracyMetric
    metrics = [AccuracyMetric(threshold=0.5)]
    train_mc = MetricCollection(metrics)
    val_mc = MetricCollection([AccuracyMetric(threshold=0.5)])
    test_mc = MetricCollection([AccuracyMetric(threshold=0.5)])
    settings = TrainerSettings(epochs=1, patience=5, device='cpu')
    ckpt = CheckpointManager(tmp_path / 'ckpt', monitor='loss', mode='min')
    trainer = Trainer(model=model, task=task, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, train_metrics=train_mc, val_metrics=val_mc, test_metrics=test_mc, settings=settings, checkpoint_manager=ckpt, logger=NullExperimentLogger(), run_dir=tmp_path)
    metrics = trainer.train()
    assert 'accuracy_accuracy' in metrics

def test_config_loading():
    """Test loading merged config."""
    import tempfile
    from pathlib import Path
    exp_toml = Path(tempfile.gettempdir()) / 'test_exp.toml'
    model_toml = Path(tempfile.gettempdir()) / 'test_model.toml'
    exp_toml.write_text('\n[experiment]\nname = "test"\nclass_names = ["A", "B"]\n\n[data]\nadapter = "dummy"\nbatch_size = 16\n')
    model_toml.write_text('\n[model]\nname = "simple_net"\ninput_dim = 64\n')
    config = load_merged_experiment_config(exp_toml, model_toml)
    assert config['experiment']['name'] == 'test'
    assert config['model']['name'] == 'simple_net'
    assert config['data']['batch_size'] == 16
    assert config['model']['input_dim'] == 64
    exp_toml.unlink()
    model_toml.unlink()

def test_dataloader_performance_options_are_configurable():
    """DataLoader throughput options from config are applied."""
    from framework.registry.defaults import create_default_registries
    from framework.registry.factories import build_component_bundle
    config = {'experiment': {'name': 'test', 'class_names': ['A', 'B']}, 'data': {'adapter': 'dummy', 'num_samples': 16, 'input_dim': 8, 'batch_size': 4, 'num_workers': 2, 'pin_memory': True, 'drop_last': True, 'persistent_workers': True, 'prefetch_factor': 4}, 'model': {'name': 'simple_net', 'input_dim': 8}, 'task': {'name': 'classification'}, 'metrics': {'params': {'accuracy': {}}}}
    bundle = build_component_bundle(config, create_default_registries())
    assert bundle.train_loader.pin_memory is True
    assert bundle.train_loader.drop_last is True
    assert bundle.train_loader.persistent_workers is True
    assert bundle.train_loader.prefetch_factor == 4
    assert bundle.val_loader.pin_memory is True
    assert bundle.val_loader.drop_last is False
    assert bundle.val_loader.persistent_workers is True
    assert bundle.test_loader.prefetch_factor == 4

def test_segmentation_overlap_metrics_do_not_sync_during_update(monkeypatch):
    """Dice/IoU update should avoid CPU copies and scalar syncs in the batch loop."""
    from framework.metrics.dice import DiceMetric
    from framework.metrics.iou import IoUMetric

    def fail_cpu(self):
        raise AssertionError('metric update should not force tensors to CPU')

    def fail_item(self):
        raise AssertionError('metric update should not call Tensor.item()')
    monkeypatch.setattr(torch.Tensor, 'cpu', fail_cpu)
    monkeypatch.setattr(torch.Tensor, 'item', fail_item)
    preds = torch.randn(1, 4, 2, 2, 2)
    targets = torch.randint(0, 4, (1, 2, 2, 2))
    dice = DiceMetric(num_classes=4)
    iou = IoUMetric(num_classes=4)
    dice.update(preds, targets)
    iou.update(preds, targets)

def test_full_volume_sliding_window_starts_cover_axis():
    """Sliding-window starts should cover the final voxel without overshooting."""
    from scripts.validate_full_volume import sliding_window_starts
    assert sliding_window_starts(length=240, window=128, overlap=0.5) == [0, 64, 112]
    assert sliding_window_starts(length=100, window=128, overlap=0.5) == [0]

def test_full_volume_region_metrics_report_dice_and_hd95():
    """Full-volume metrics should report WT/TC/ET Dice and HD95 values."""
    import numpy as np
    from scripts.validate_full_volume import compute_region_metrics
    target = np.zeros((8, 8, 8), dtype=np.int64)
    pred = np.zeros((8, 8, 8), dtype=np.int64)
    target[2:5, 2:5, 2:5] = 3
    pred[2:5, 2:5, 2:5] = 3
    metrics = compute_region_metrics(pred, target)
    assert metrics['wt_dice'] == 1.0
    assert metrics['tc_dice'] == 1.0
    assert metrics['et_dice'] == 1.0
    assert metrics['wt_hd95'] == 0.0
    assert metrics['tc_hd95'] == 0.0
    assert metrics['et_hd95'] == 0.0

def test_full_volume_metrics_ignore_small_empty_et_prediction_by_default():
    """Small ET false positives on GT-empty ET cases should not trigger ET empty-mask penalty."""
    import numpy as np
    from scripts.validate_full_volume import compute_region_metrics
    target = np.zeros((8, 8, 8), dtype=np.int64)
    pred = np.zeros((8, 8, 8), dtype=np.int64)
    target[2:5, 2:5, 2:5] = 1
    pred[2:5, 2:5, 2:5] = 1
    pred[0, 0, 0] = 3
    pred[0, 0, 1] = 3
    metrics = compute_region_metrics(pred, target)
    assert metrics['et_dice'] == 1.0
    assert metrics['et_hd95'] == 0.0

def test_full_volume_metrics_penalize_large_empty_et_prediction_with_brats_diagonal():
    """Large ET false positives on GT-empty ET cases should use the configured empty HD95 penalty."""
    import numpy as np
    from scripts.validate_full_volume import BRATS_EMPTY_HD95_PENALTY, compute_region_metrics
    target = np.zeros((8, 8, 8), dtype=np.int64)
    pred = np.zeros((8, 8, 8), dtype=np.int64)
    pred.ravel()[:51] = 3
    metrics = compute_region_metrics(pred, target)
    assert metrics['et_dice'] == 0.0
    assert abs(metrics['et_hd95'] - BRATS_EMPTY_HD95_PENALTY) < 1e-06

def test_full_volume_sliding_window_predict_stitches_complete_volume():
    """Sliding-window inference should return one label for every input voxel."""
    import torch
    from scripts.validate_full_volume import sliding_window_predict

    class ConstantClassOne(torch.nn.Module):

        def forward(self, batch):
            signal = batch['signal']
            logits = torch.zeros(signal.shape[0], 2, *signal.shape[2:], dtype=signal.dtype, device=signal.device)
            logits[:, 1] = 10.0
            return logits
    volume = torch.zeros(1, 4, 5, 6)
    pred = sliding_window_predict(ConstantClassOne(), volume, device=torch.device('cpu'), num_classes=2, window_size=(3, 3, 3), overlap=0.5)
    assert pred.shape == (4, 5, 6)
    assert torch.all(pred == 1)

def test_sliding_window_predict_proba_batches_windows():
    """Sliding-window inference should send multiple windows per forward pass."""
    import torch
    from scripts.validate_full_volume import sliding_window_predict_proba

    class RecordingModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, batch):
            signal = batch['signal']
            self.batch_sizes.append(signal.shape[0])
            logits = torch.zeros(signal.shape[0], 2, *signal.shape[2:], dtype=signal.dtype, device=signal.device)
            logits[:, 1] = 10.0
            return logits
    model = RecordingModel()
    volume = torch.zeros(1, 5, 5, 5)
    probs = sliding_window_predict_proba(model, volume, device=torch.device('cpu'), num_classes=2, window_size=(3, 3, 3), overlap=0.0, window_batch_size=3)
    assert probs.shape == (2, 5, 5, 5)
    assert model.batch_sizes == [3, 3, 2]

def test_sliding_window_predict_proba_uses_task_postprocess():
    """Region-logit models should be converted by the task before accumulation."""
    import torch
    from scripts.validate_full_volume import sliding_window_predict_proba

    class ThreeRegionModel(torch.nn.Module):

        def forward(self, batch):
            signal = batch['signal']
            return torch.zeros(signal.shape[0], 3, *signal.shape[2:])

    def four_class_postprocess(logits):
        (batch, _, depth, height, width) = logits.shape
        probs = torch.zeros(batch, 4, depth, height, width)
        probs[:, 2] = 1.0
        return probs
    probs = sliding_window_predict_proba(ThreeRegionModel(), torch.zeros(1, 4, 4, 4), device=torch.device('cpu'), num_classes=4, window_size=(4, 4, 4), postprocess_fn=four_class_postprocess)
    assert probs.shape == (4, 4, 4, 4)
    assert torch.all(torch.argmax(probs, dim=0) == 2)

def test_enable_mc_dropout_keeps_batchnorm_in_eval():
    """MC-dropout validation should enable dropout without switching BatchNorm to train."""
    from scripts.validate_full_volume import enable_mc_dropout
    model = torch.nn.Sequential(torch.nn.BatchNorm3d(1), torch.nn.Dropout3d(p=0.5))
    model.train()
    enabled = enable_mc_dropout(model)
    assert enabled == 1
    assert model.training is False
    assert model[0].training is False
    assert model[1].training is True

def test_mc_dropout_predict_returns_mean_probs_and_uncertainty():
    """MC prediction should aggregate stochastic probabilities and report uncertainty maps."""
    from scripts.validate_full_volume import mc_dropout_predict

    class AlternatingModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            signal = batch['signal']
            logits = torch.zeros(signal.shape[0], 2, *signal.shape[2:], dtype=signal.dtype, device=signal.device)
            logits[:, self.calls % 2] = 8.0
            return logits
    volume = torch.zeros(1, 4, 4, 4)
    result = mc_dropout_predict(AlternatingModel(), volume, device=torch.device('cpu'), num_classes=2, window_size=(4, 4, 4), overlap=0.5, mc_samples=2)
    assert result.pred.shape == (4, 4, 4)
    assert result.mean_probs.shape == (2, 4, 4, 4)
    assert result.aleatoric_uncertainty.shape == (4, 4, 4)
    assert result.epistemic_uncertainty.shape == (4, 4, 4)
    assert float(result.epistemic_uncertainty.mean()) > 0.2
    assert float(result.aleatoric_uncertainty.mean()) < 0.01
    assert result.predictive_entropy is result.aleatoric_uncertainty
    assert result.probability_variance is result.epistemic_uncertainty

def test_summarize_mc_sample_metrics_reports_mean_and_std():
    """Per-sample MC Dice should be summarized for stability reporting."""
    from scripts.validate_full_volume import summarize_mc_sample_metrics
    summary = summarize_mc_sample_metrics([{'mean_dice': 0.5, 'wt_dice': 1.0, 'tc_dice': 0.25}, {'mean_dice': 1.0, 'wt_dice': 0.0, 'tc_dice': 0.75}])
    assert summary['sample_mean_dice_mean'] == 0.75
    assert summary['sample_mean_dice_std'] == 0.25
    assert summary['sample_wt_dice_mean'] == 0.5
    assert summary['sample_tc_dice_std'] == 0.25

def test_summarize_uncertainty_reports_paper_aligned_names():
    """MC uncertainty summaries should expose U_ale and U_epi names."""
    from scripts.validate_full_volume import MCDropoutPrediction, summarize_uncertainty
    result = MCDropoutPrediction(pred=torch.zeros(1, 1, 1, dtype=torch.long), mean_probs=torch.full((2, 1, 1, 1), 0.5), epistemic_uncertainty=torch.full((1, 1, 1), 0.25), aleatoric_uncertainty=torch.full((1, 1, 1), 0.01), sample_count=2)
    summary = summarize_uncertainty(result)
    assert abs(summary['aleatoric_uncertainty_mean'] - 0.01) < 1e-08
    assert abs(summary['epistemic_uncertainty_mean'] - 0.25) < 1e-08
