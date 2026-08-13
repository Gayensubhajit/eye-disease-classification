"""Standalone evaluation script to evaluate trained model checkpoints on untouched test sets."""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml

from src.data.dataset import FundusDataset, get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, plot_confusion_matrix
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fundus classification model checkpoint")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="Path to checkpoint .pth file")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default="outputs/test_eval", help="Output directory for test results")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {checkpoint_path} on device {device}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test dataset
    test_dir = config["data"]["test_dir"]
    image_size = config["data"]["image_size"]
    crop_fundus = config["data"].get("crop_fundus", True)
    apply_clahe_flag = config["data"].get("apply_clahe", False)
    class_names = config["data"]["class_names"]

    val_transforms = get_val_transforms(image_size)
    test_dataset = FundusDataset(
        test_dir,
        class_names=class_names,
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"Evaluating {len(test_dataset)} test samples...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    print(f"Macro ROC-AUC:      {metrics['macro_auc']:.4f}")
    print(f"Cohen's Kappa:      {metrics['kappa']:.4f}")
    print(f"Macro Sensitivity:  {metrics['macro_sensitivity']:.4f}")
    print(f"Macro Specificity:  {metrics['macro_specificity']:.4f}")

    if "per_class" in metrics:
        print("\n--- Per-Class Performance ---")
        df_per_class = pd.DataFrame(metrics["per_class"])
        print(df_per_class.to_string(index=False))

    # Save metrics JSON
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot & save confusion matrix
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        str(output_dir / "test_confusion_matrix.png"),
        title="Test Set Confusion Matrix",
    )
    print(f"\nEvaluation reports saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
