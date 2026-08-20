"""Tests for model building, CBAM attention, multi-scale feature fusion, and forward pass."""

import torch
from src.models.backbone import FundusClassifier, build_model, CBAMModule


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


def test_biomedclip_factory():
    config = {
        "model": {"name": "biomedclip", "pretrained": False, "dropout": 0.2},
        "data": {"num_classes": 10},
    }
    model = build_model(config)
    assert model.is_biomedclip is True
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_cbam_module():
    cbam = CBAMModule(in_planes=64)
    x = torch.randn(2, 64, 14, 14)
    out = cbam(x)
    assert out.shape == (2, 64, 14, 14)


def test_biomedclip_cbam_fusion_factory():
    config = {
        "model": {"name": "biomedclip_cbam_fusion", "pretrained": False, "dropout": 0.3},
        "data": {"num_classes": 10},
    }
    model = build_model(config)
    assert model.is_biomedclip is True
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)
