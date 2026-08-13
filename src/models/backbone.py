"""Model architecture module supporting timm vision backbones and BiomedCLIP for fundus classification."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import timm


class BiomedCLIPClassifier(nn.Module):
    """Fundus image classification wrapper around Microsoft's BiomedCLIP vision encoder."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        import open_clip

        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        if pretrained:
            clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
        else:
            clip_model = open_clip.create_model(model_name)

        self.visual = clip_model.visual
        # Extract feature dimension (512 for BiomedCLIP ViT-B/16)
        in_features = getattr(self.visual, "output_dim", None)
        if in_features is None:
            if hasattr(self.visual, "proj") and self.visual.proj is not None:
                in_features = self.visual.proj.shape[1] if len(self.visual.proj.shape) > 1 else 512
            else:
                in_features = 512

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delivering logits tensor of shape (batch_size, num_classes)."""
        features = self.visual(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features for interpretability."""
        return self.visual(x)


class FundusClassifier(nn.Module):
    """Fundus image classification neural network wrapping timm or BiomedCLIP backbones."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if model_name.lower() in ["biomedclip", "biomed_clip", "microsoft/biomedclip"]:
            self.backbone = BiomedCLIPClassifier(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
            )
            self.is_biomedclip = True
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            self.is_biomedclip = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delivering logits tensor of shape (batch_size, num_classes)."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map before pooling for Grad-CAM."""
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        elif hasattr(self.backbone, "get_features"):
            return self.backbone.get_features(x)
        return self.backbone(x)


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
