"""Tests for dataset and preprocessing pipelines."""

import numpy as np
import pytest
import torch
import pandas as pd
from src.data.preprocessing import crop_fundus_area, apply_clahe, get_train_transforms, get_val_transforms


def test_crop_fundus_area():
    # Create black image with a bright circle in center
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
