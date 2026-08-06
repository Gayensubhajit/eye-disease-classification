"""Tests for model building and forward pass."""

import torch
from src.models.backbone import FundusClassifier, build_model


def test_fundus_classifier_forward():
    model = FundusClassifier(model_name="efficientnet_b0", num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_build_model_factory():
    config = {
        "model": {"name": "resnet18", "pretrained": False, "dropout": 0.1},
        "data": {"num_classes": 10},
    }
    model = build_model(config)
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (1, 10)
