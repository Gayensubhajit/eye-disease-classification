"""Advanced Evaluation & Test-Time Augmentation (TTA) Ensemble for Fundus Disease Classification."""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FundusDataset, get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, plot_confusion_matrix


def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    use_tta: bool = True,
) -> torch.Tensor:
    """Predict class probabilities with optional multi-flip Test-Time Augmentation (TTA)."""
    if not use_tta:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            return torch.softmax(logits, dim=1)

    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        # TTA 1: Original
        p1 = torch.softmax(model(images), dim=1)
        # TTA 2: Horizontal Flip
        p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        # TTA 3: Vertical Flip
        p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
        # TTA 4: Both Flips (180 deg)
        p4 = torch.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)

    return (p1 + p2 + p3 + p4) / 4.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble with TTA on Test Set")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/config.yaml",
            "configs/biomedclip.yaml",
            "configs/biomedclip_cbam_fusion.yaml",
        ],
        help="List of model config files",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "outputs/best_model.pth",
            "outputs/biomedclip/best_model.pth",
            "outputs/biomedclip_cbam_fusion/best_model.pth",
        ],
        help="List of model checkpoint .pth files",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.40],
        help="Ensemble blending weights",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument(
        "--output-dir", default="outputs/ensemble_eval", help="Output directory for ensemble metrics"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ensemble Evaluation on device: {device}")

    # Load master config for dataset paths
    with open(args.configs[0], "r", encoding="utf-8") as f:
        master_cfg = yaml.safe_load(f)

    test_dir = master_cfg["data"]["test_dir"]
    class_names = master_cfg["data"]["class_names"]
    use_tta = not args.no_tta

    # Normalize weights
    weights = np.array(args.weights, dtype=np.float32)
    weights = weights / weights.sum()

    print(f"Ensemble Components: {len(args.checkpoints)} models with weights {weights.round(3).tolist()}")
    print(f"Test-Time Augmentation (TTA): {'Enabled (4-view averaging)' if use_tta else 'Disabled'}")

    model_probs_list = []
    ground_truth = None

    for idx, (cfg_path, ckpt_path) in enumerate(zip(args.configs, args.checkpoints)):
        if not Path(ckpt_path).exists():
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping...")
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        print(f"\n[{idx+1}/{len(args.checkpoints)}] Loading {cfg['model']['name']} from {ckpt_path}...")
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        image_size = cfg["data"].get("image_size", 224)
        crop_fundus = cfg["data"].get("crop_fundus", True)
        apply_clahe = cfg["data"].get("apply_clahe", False)

        test_transforms = get_val_transforms(image_size)
        test_dataset = FundusDataset(
            test_dir,
            class_names=class_names,
            transform=test_transforms,
            crop_fundus=crop_fundus,
            apply_clahe_flag=apply_clahe,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["training"].get("batch_size", 16),
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 4),
        )

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                probs = predict_with_tta(model, images, use_tta=use_tta)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.numpy())

        model_probs = np.concatenate(all_probs)
        model_probs_list.append(model_probs)

        if ground_truth is None:
            ground_truth = np.concatenate(all_targets)

        # Print individual model test metrics
        single_preds = np.argmax(model_probs, axis=1)
        single_metrics = compute_metrics(ground_truth, single_preds, model_probs, class_names=class_names)
        print(f"  -> Single Model Acc: {single_metrics['accuracy']:.4f} ({single_metrics['accuracy']*100:.2f}%), Macro F1: {single_metrics['macro_f1']:.4f}, ROC-AUC: {single_metrics['macro_auc']:.4f}")

        # Free memory before loading next model
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(model_probs_list) == 0:
        print("Error: No valid models were evaluated.")
        return

    # Weighted Ensemble Blending
    fused_probs = np.zeros_like(model_probs_list[0])
    valid_weights = weights[: len(model_probs_list)]
    valid_weights = valid_weights / valid_weights.sum()

    for p, w in zip(model_probs_list, valid_weights):
        fused_probs += w * p

    fused_preds = np.argmax(fused_probs, axis=1)
    ensemble_metrics = compute_metrics(ground_truth, fused_preds, fused_probs, class_names=class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("        🏆 ENSEMBLE + TTA FINAL TEST RESULTS 🏆        ")
    print("=======================================================")
    print(f"Test Accuracy:         {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy:     {ensemble_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1-Score:        {ensemble_metrics['macro_f1']:.4f}")
    print(f"Macro ROC-AUC:         {ensemble_metrics['macro_auc']:.4f}")
    print(f"Cohen's Kappa:         {ensemble_metrics['kappa']:.4f}")
    print(f"Macro Sensitivity:     {ensemble_metrics['macro_sensitivity']:.4f}")
    print(f"Macro Specificity:     {ensemble_metrics['macro_specificity']:.4f}")

    if "per_class" in ensemble_metrics:
        print("\n--- Per-Class Performance ---")
        df_per_class = pd.DataFrame(ensemble_metrics["per_class"])
        print(df_per_class.to_string(index=False))

    # Save metrics JSON
    with open(output_dir / "ensemble_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_metrics, f, indent=2)

    # Save Confusion Matrix
    plot_confusion_matrix(
        ground_truth,
        fused_preds,
        class_names,
        str(output_dir / "ensemble_test_confusion_matrix.png"),
        title="Ensemble + TTA Test Confusion Matrix",
    )
    print(f"\nEnsemble artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
