"""Tests for loss functions."""

import torch
from src.losses.focal_loss import FocalLoss, get_loss_function


def test_focal_loss_forward():
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 10)
    targets = torch.tensor([0, 2, 5, 9])
    loss = criterion(logits, targets)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0


def test_get_loss_function_factory():
    config = {"loss": {"name": "focal", "gamma": 2.0}}
    criterion = get_loss_function(config)
    logits = torch.randn(2, 5)
    targets = torch.tensor([1, 4])
    loss = criterion(logits, targets)
    assert loss.item() > 0
