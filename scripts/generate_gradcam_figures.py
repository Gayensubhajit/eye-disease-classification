"""Generate publication-ready Grad-CAM heatmaps showing model visual attention for fundus diagnoses."""

import argparse
import glob
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.preprocessing import crop_fundus_area, apply_clahe
from src.models.backbone import build_model
from src.utils.gradcam import GradCAM, overlay_heatmap


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for all 10 fundus disease classes")
    parser.add_argument(
        "--config",
        default="configs/efficientnet_b3_384_clahe.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/efficientnet_b3_384_clahe/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gradcam_figures",
        help="Directory to save Grad-CAM figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found at {args.checkpoint}. Cannot generate figures.")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device}...")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Find target convolutional layer for Grad-CAM
    target_layer = None
    if hasattr(model.backbone, "conv_head"):
        target_layer = model.backbone.conv_head
    elif hasattr(model.backbone, "act2"):
        target_layer = model.backbone.act2
    else:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                print(f"Using target conv layer: {name}")
                break

    if target_layer is None:
        print("Could not find a convolutional target layer for Grad-CAM.")
        return

    grad_cam = GradCAM(model, target_layer)

    test_dir = Path(config["data"]["test_dir"])
    class_names = config["data"]["class_names"]
    image_size = config["data"].get("image_size", 384)
    crop_fundus = config["data"].get("crop_fundus", True)
    use_clahe = config["data"].get("apply_clahe", False)

    print(f"Generating Grad-CAM at resolution {image_size}x{image_size} (CLAHE: {use_clahe})...")

    fig, axes = plt.subplots(len(class_names), 3, figsize=(12, 3.2 * len(class_names)))
    plt.subplots_adjust(wspace=0.15, hspace=0.35)

    for i, class_name in enumerate(class_names):
        class_folder = test_dir / class_name
        img_paths = sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")))
        if not img_paths:
            continue

        # Choose a representative sample from test set
        img_path = img_paths[0]
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if crop_fundus:
            rgb = crop_fundus_area(rgb)
        if use_clahe:
            rgb = apply_clahe(rgb)

        rgb_resized = cv2.resize(rgb, (image_size, image_size))

        # Normalize with standard ImageNet stats
        norm = (rgb_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

        heatmap = grad_cam.generate(tensor, class_idx=i)
        overlaid = overlay_heatmap(rgb_resized, heatmap, alpha=0.5)

        # Plot 1: Input Preprocessed Image
        axes[i, 0].imshow(rgb_resized)
        axes[i, 0].set_title(f"Class {i+1}: {class_name[:24]}", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        # Plot 2: Grad-CAM Activation Heatmap
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Spatial Heatmap", fontsize=9)
        axes[i, 1].axis("off")

        # Plot 3: Clinical Diagnostic Overlay
        axes[i, 2].imshow(overlaid)
        axes[i, 2].set_title("Anatomical Focus Overlay", fontsize=9, fontweight="bold")
        axes[i, 2].axis("off")

    fig_path = output_dir / f"gradcam_10classes_grid_{image_size}.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"\nGrad-CAM 10-disease figure saved successfully to: {fig_path.resolve()}")


if __name__ == "__main__":
    main()
