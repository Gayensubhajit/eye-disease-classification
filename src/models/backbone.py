"""Baseline model factory."""

import timm


def build_baseline(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
    """Create a timm classification model with the requested output head."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
