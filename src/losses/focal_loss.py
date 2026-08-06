"""Custom loss functions for imbalanced multi-class classification."""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor tensor per class of shape (num_classes,).
            gamma: Focusing parameter for hard examples (default 2.0).
            reduction: Specifies reduction: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model logits of shape (N, num_classes).
            targets: Ground truth class indices of shape (N,).
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    config: Dict[str, Any], class_counts: Optional[np.ndarray] = None
) -> nn.Module:
    """Factory function to build configured loss function.

    Args:
        config: Loaded config dictionary.
        class_counts: Array of sample counts per class for computing inverse-frequency weights.

    Returns:
        PyTorch nn.Module loss instance.
    """
    loss_config = config.get("loss", {})
    loss_name = loss_config.get("name", "cross_entropy").lower()

    weights = None
    if class_counts is not None and len(class_counts) > 0:
        total = np.sum(class_counts)
        weights_arr = total / (len(class_counts) * np.maximum(class_counts, 1).astype(np.float32))
        weights = torch.tensor(weights_arr, dtype=torch.float32)

    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_name == "focal":
        gamma = float(loss_config.get("gamma", 2.0))
        return FocalLoss(alpha=weights, gamma=gamma)
    else:
        return nn.CrossEntropyLoss()
