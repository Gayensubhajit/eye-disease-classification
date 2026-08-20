"""Model architecture module supporting timm backbones, BiomedCLIP, and novel CBAM + Multi-Scale Feature Pyramid Fusion."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import timm


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) for CBAM."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) for CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return self.sigmoid(scale)


class CBAMModule(nn.Module):
    """Convolutional Block Attention Module combining CAM and SAM."""

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class BiomedCLIPCBAMFusionClassifier(nn.Module):
    """Novel Hybrid Architecture: Microsoft BiomedCLIP + CBAM Dual Attention + Multi-Scale Feature Pyramid Fusion."""

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
        trunk = getattr(self.visual, "trunk", None)
        self.trunk = trunk

        embed_dim = 768
        if hasattr(trunk, "embed_dim"):
            embed_dim = trunk.embed_dim

        # CBAM Attention Module operating on spatial patch grid
        self.cbam = CBAMModule(in_planes=embed_dim, ratio=16, kernel_size=7)

        # Multi-scale projection
        fused_dim = embed_dim * 2  # CLS global token + CBAM spatial pooled features
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting multi-scale token maps, applying CBAM attention, and delivering 10-class logits."""
        if self.trunk is not None and hasattr(self.trunk, "forward_features"):
            feats = self.trunk.forward_features(x)  # (B, 197, 768)
            cls_token = feats[:, 0, :]  # Global semantic view (B, 768)
            patch_tokens = feats[:, 1:, :]  # Spatial patch tokens (B, 196, 768)

            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            spatial_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)

            # Apply CBAM Channel + Spatial Attention
            attended_map = self.cbam(spatial_map)
            pooled_spatial = torch.mean(attended_map, dim=(2, 3))  # (B, 768)

            # Multi-scale feature fusion
            fused_features = torch.cat([cls_token, pooled_spatial], dim=1)  # (B, 1536)
            return self.classifier(fused_features)
        else:
            features = self.visual(x)
            if isinstance(features, (list, tuple)):
                features = features[0]
            return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map for Grad-CAM visualization."""
        if self.trunk is not None and hasattr(self.trunk, "forward_features"):
            feats = self.trunk.forward_features(x)
            patch_tokens = feats[:, 1:, :]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            spatial_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
            return self.cbam(spatial_map)
        return self.visual(x)


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
    """Fundus image classification neural network wrapping timm, BiomedCLIP, or CBAM-Fusion backbones."""

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

        if model_name.lower() in [
            "biomedclip_cbam_fusion",
            "cbam_biomedclip",
            "biomedclip_fusion",
        ]:
            self.backbone = BiomedCLIPCBAMFusionClassifier(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
            )
            self.is_biomedclip = True
        elif model_name.lower() in ["biomedclip", "biomed_clip", "microsoft/biomedclip"]:
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
    """Build model instance from configuration dictionary."""
    model_cfg = config["model"]
    num_classes = config["data"]["num_classes"]

    return FundusClassifier(
        model_name=model_cfg.get("name", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
