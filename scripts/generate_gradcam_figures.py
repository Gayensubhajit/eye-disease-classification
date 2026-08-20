"""Generate Grad-CAM heatmaps showing model visual attention for fundus diagnoses."""

import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.backbone import build_model
from src.utils.gradcam import GradCAM, overlay_heatmap
import yaml

def main():
    output_dir = "outputs/gradcam_figures"
    os.makedirs(output_dir, exist_ok=True)

    config_path = "configs/config.yaml"
    checkpoint_path = "outputs/best_model.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Skipping figure generation.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Get target layer for Grad-CAM
    target_layer = None
    if hasattr(model.backbone, "conv_head"):
        target_layer = model.backbone.conv_head
    elif hasattr(model.backbone, "act2"):
        target_layer = model.backbone.act2
    else:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        print("Could not find a convolutional target layer for Grad-CAM.")
        return

    grad_cam = GradCAM(model, target_layer)

    # Sample images from test set
    test_dir = config["data"]["test_dir"]
    class_names = config["data"]["class_names"]

    fig, axes = plt.subplots(len(class_names), 3, figsize=(10, 3 * len(class_names)))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    for i, class_name in enumerate(class_names):
        class_folder = os.path.join(test_dir, class_name)
        img_paths = glob.glob(os.path.join(class_folder, "*.jpg"))
        if not img_paths:
            continue

        img_path = img_paths[0]
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (224, 224))

        # Preprocess
        tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        heatmap = grad_cam.generate(tensor, class_idx=i)
        overlaid = overlay_heatmap(rgb_resized, heatmap, alpha=0.5)

        axes[i, 0].imshow(rgb_resized)
        axes[i, 0].set_title(f"Original: {class_name[:15]}...", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=8)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlaid)
        axes[i, 2].set_title("Overlaid Diagnosis", fontsize=8)
        axes[i, 2].axis("off")

    fig_path = os.path.join(output_dir, "gradcam_10classes_grid.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Grad-CAM figure saved successfully to {fig_path}")

if __name__ == "__main__":
    main()
