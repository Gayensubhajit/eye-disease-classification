"""Model architecture module wrapping timm vision backbones for fundus image classification."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import timm


class FundusClassifier(nn.Module):
    """Fundus image classification neural network wrapping timm backbones."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Args:
            model_name: Name of the timm backbone (e.g. 'efficientnet_b0', 'resnet50', 'convnext_tiny').
            num_classes: Number of disease target classes.
            pretrained: Whether to load ImageNet pre-trained weights.
            dropout: Dropout probability in output classifier head.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delivering logits tensor of shape (batch_size, num_classes)."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map before pooling for Grad-CAM."""
        return self.backbone.forward_features(x)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build model instance from configuration dictionary.

    Args:
        config: Loaded config dictionary.

    Returns:
        PyTorch nn.Module model.
    """
    model_cfg = config["model"]
    num_classes = config["data"]["num_classes"]

    return FundusClassifier(
        model_name=model_cfg.get("name", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
