import torch
from torch.utils.data import Dataset

from data_adapters.brats_adapter import BraTSDataAdapter, BraTSSliceDataset
from models.resunet_2d import SliceWiseResUNet2D


class _VolumeDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        label = torch.zeros(5, 32, 32, dtype=torch.long)
        label[2, 8:16, 8:16] = 3
        return {"signal": torch.randn(4, 5, 32, 32), "label": label, "id": "case-1", "meta": {}}


def test_slice_dataset_batches_2d_training_slices_without_losing_patient_split():
    dataset = BraTSSliceDataset(_VolumeDataset(), slices_per_case=4, foreground_slice_ratio=0.5)
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["signal"].shape == (4, 4, 32, 32)
    assert sample["label"].shape == (4, 32, 32)
    assert sample["meta"]["slice_wise_2d"] is True

    batch = BraTSDataAdapter().collate_fn([sample])
    assert batch["signal"].shape == (4, 4, 32, 32)
    assert batch["label"].shape == (4, 32, 32)


def test_reference_resunet_2d_accepts_slices_and_reassembles_volume_logits():
    model = SliceWiseResUNet2D(num_classes=4, num_modalities=4, filters=[4, 8, 16, 32], eval_slice_chunk_size=2).eval()
    with torch.no_grad():
        slice_logits = model({"signal": torch.randn(2, 4, 32, 32), "label": torch.zeros(2, 32, 32, dtype=torch.long), "id": [], "meta": []})
        volume_logits = model({"signal": torch.randn(1, 4, 5, 32, 32), "label": torch.zeros(1, 5, 32, 32, dtype=torch.long), "id": [], "meta": []})
    assert slice_logits.shape == (2, 4, 32, 32)
    assert volume_logits.shape == (1, 4, 5, 32, 32)
