"""Tests for dataset and preprocessing pipelines."""

from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.preprocessing import crop_fundus_area, apply_clahe, get_train_transforms, get_val_transforms
from src.data.dataset import FundusDataset, create_dataloaders


def test_crop_fundus_area():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    cropped = crop_fundus_area(img, threshold=10)
    assert cropped.shape[0] < 100 and cropped.shape[1] < 100
    assert cropped.shape[0] == 60 and cropped.shape[1] == 60


def test_apply_clahe():
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    enhanced = apply_clahe(img)
    assert enhanced.shape == (50, 50, 3)
    assert enhanced.dtype == np.uint8


def test_transforms_shape():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    train_tf = get_train_transforms(image_size=224)
    val_tf = get_val_transforms(image_size=224)

    out_train = train_tf(image=img)["image"]
    out_val = val_tf(image=img)["image"]

    assert out_train.shape == (3, 224, 224)
    assert out_val.shape == (3, 224, 224)
    assert isinstance(out_train, torch.Tensor)


def test_fundus_dataset_folder_loading():
    val_path = Path("data/val")
    if val_path.exists():
        dataset = FundusDataset(val_path, transform=get_val_transforms(224))
        assert len(dataset) > 0
        img, label = dataset[0]
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)
