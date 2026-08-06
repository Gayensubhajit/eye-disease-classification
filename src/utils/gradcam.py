"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for model interpretability."""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM visual explanation generator for PyTorch CNN models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch classification model.
            target_layer: Convolutional layer to extract activation maps and gradients from.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap array normalized in [0.0, 1.0].

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            class_idx: Target class index. If None, uses predicted class with maximum score.

        Returns:
            Normalized 2D numpy heatmap array of shape (H, W).
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activation, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False
        )

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlay 2D normalized heatmap onto an RGB image.

    Args:
        image: Original RGB image array (H, W, 3) with values in range [0, 255].
        heatmap: Normalized 2D float heatmap array in range [0.0, 1.0].
        alpha: Blend ratio between heatmap and original image.
        colormap: OpenCV colormap constant.

    Returns:
        RGB image array with overlaid heatmap.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, colored_heatmap, alpha, 0)
    return overlaid
