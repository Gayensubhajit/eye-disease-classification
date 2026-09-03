"""Multi-Scale Test-Time Augmentation (MS-TTA) Evaluation Engine.

Evaluates standalone models and heterogeneous ensembles across multi-scale
pyramids (e.g. 1.0x, 1.15x, 0.88x) and 4-view geometric symmetries to push
retinal disease classification accuracy beyond single-resolution limits.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FundusDataset, get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, plot_confusion_matrix


def predict_multiscale_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    scales: List[float] = [1.0, 1.15],
    use_flips: bool = True,
    is_vit: bool = False,
) -> torch.Tensor:
    """Predict class probabilities averaged across multiple scales and flip orientations.

    Args:
        model: Loaded PyTorch classification model in eval mode.
        images: Input tensor of shape (B, 3, H, W).
        scales: List of scale multipliers (e.g. [1.0, 1.15]).
        use_flips: If True, tests horizontal, vertical, and 180-deg flips.
        is_vit: If True (e.g. BiomedCLIP ViT), constrains to base size to avoid patch grid mismatch.

    Returns:
        Averaged probability tensor of shape (B, num_classes).
    """
    total_probs = None
    count = 0
    B, C, H, W = images.shape

    # For ViT models with rigid patch grids, only use scale 1.0
    active_scales = [1.0] if is_vit else scales

    with torch.no_grad():
        for scale in active_scales:
            if abs(scale - 1.0) < 1e-4:
                scaled_x = images
            else:
                # Snap height & width to multiples of 32 for ConvNeXt/ResNet/EffNet downsampling compatibility
                H_s = int(round(H * scale / 32.0) * 32)
                W_s = int(round(W * scale / 32.0) * 32)
                scaled_x = F.interpolate(
                    images, size=(H_s, W_s), mode="bilinear", align_corners=False
                )

            # Generate geometric views
            views = [scaled_x]
            if use_flips:
                views.append(torch.flip(scaled_x, dims=[3]))        # Horizontal flip
                views.append(torch.flip(scaled_x, dims=[2]))        # Vertical flip
                views.append(torch.flip(scaled_x, dims=[2, 3]))     # 180-degree rotation

            for v in views:
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(v)
                    probs = torch.softmax(logits, dim=1)

                if total_probs is None:
                    total_probs = probs
                else:
                    total_probs = total_probs + probs
                count += 1

    return total_probs / float(count)


def main():
    parser = argparse.ArgumentParser(description="Multi-Scale TTA Evaluation")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/convnext_small_384_clahe.yaml"],
        help="List of model config files",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["outputs/convnext_small_384_clahe/best_model.pth"],
        help="List of model checkpoint .pth files",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[1.0],
        help="Ensemble blending weights (if evaluating multiple models)",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[1.0, 1.15],
        help="Scale factors for multi-scale testing (e.g. 1.0 1.15 0.90)",
    )
    parser.add_argument(
        "--no-flips", action="store_true", help="Disable horizontal/vertical flips"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ms_tta_eval",
        help="Output directory for MS-TTA metrics and artifacts",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================")
    print(f"      🔬 MULTI-SCALE TEST-TIME AUGMENTATION (MS-TTA) 🔬        ")
    print(f"================================================================")
    print(f"Device:            {device}")
    if torch.cuda.is_available():
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
        free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        print(f"Free VRAM:         {free_vram:.2f} GB")
    print(f"Scales:            {args.scales}")
    print(f"Geometric Flips:   {'Disabled' if args.no_flips else 'Enabled (4 views per scale)'}")
    print(f"Models:            {len(args.checkpoints)}")
    print(f"Batch Size:        {args.batch_size}")

    # Load master config for data paths
    with open(args.configs[0], "r", encoding="utf-8") as f:
        master_cfg = yaml.safe_load(f)

    test_dir = master_cfg["data"]["test_dir"]
    class_names = master_cfg["data"]["class_names"]
    use_flips = not args.no_flips

    # Normalize weights
    weights = np.array(args.weights, dtype=np.float32)
    weights = weights / weights.sum()

    model_probs_list = []
    ground_truth = None

    for idx, (cfg_path, ckpt_path) in enumerate(zip(args.configs, args.checkpoints)):
        if not Path(ckpt_path).exists():
            print(f"\n[Error] Checkpoint {ckpt_path} not found! Skipping...")
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_name = cfg["model"]["name"]
        print(f"\n[{idx+1}/{len(args.checkpoints)}] Loading {model_name} from {ckpt_path}...")
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        is_vit = getattr(model, "is_biomedclip", False)

        image_size = cfg["data"].get("image_size", 384)
        crop_fundus = cfg["data"].get("crop_fundus", True)
        apply_clahe = cfg["data"].get("apply_clahe", False)

        print(f"  Config: Base Res={image_size}x{image_size}, CLAHE={apply_clahe}, CropFundus={crop_fundus}")
        if is_vit:
            print(f"  Note: ViT backbone detected. Constrained to native 224x224 patch grid + flips.")

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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 2),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        all_probs = []
        all_targets = []

        total_batches = len(test_loader)
        print(f"  Inference starting ({len(test_dataset)} test images across {total_batches} batches)...")

        with torch.no_grad():
            for b_idx, (images, targets) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                probs = predict_multiscale_tta(
                    model,
                    images,
                    scales=args.scales,
                    use_flips=use_flips,
                    is_vit=is_vit,
                )
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.numpy())

                if (b_idx + 1) % 25 == 0 or (b_idx + 1) == total_batches:
                    print(f"    Batch [{b_idx+1}/{total_batches}] completed")

        model_probs = np.concatenate(all_probs)
        model_probs_list.append(model_probs)

        if ground_truth is None:
            ground_truth = np.concatenate(all_targets)

        # Single model MS-TTA metrics
        single_preds = np.argmax(model_probs, axis=1)
        single_metrics = compute_metrics(
            ground_truth, single_preds, model_probs, class_names=class_names
        )
        print(f"  -> Model {idx+1} MS-TTA Acc: {single_metrics['accuracy']*100:.2f}%, "
              f"Macro F1: {single_metrics['macro_f1']*100:.2f}%, "
              f"ROC-AUC: {single_metrics['macro_auc']:.4f}, "
              f"Kappa: {single_metrics['kappa']:.4f}")

        # Explicit GPU memory cleanup
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(model_probs_list) == 0:
        print("[Error] No valid models were evaluated.")
        return

    # Ensemble Blending across models
    fused_probs = np.zeros_like(model_probs_list[0])
    valid_weights = weights[: len(model_probs_list)]
    valid_weights = valid_weights / valid_weights.sum()

    for p, w in zip(model_probs_list, valid_weights):
        fused_probs += w * p

    fused_preds = np.argmax(fused_probs, axis=1)
    ensemble_metrics = compute_metrics(
        ground_truth, fused_preds, fused_probs, class_names=class_names
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("        🏆 FINAL MULTI-SCALE TTA EVALUATION REPORT 🏆        ")
    print("=" * 65)
    print(f"Test Accuracy:         {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy:     {ensemble_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1-Score:        {ensemble_metrics['macro_f1']:.4f} ({ensemble_metrics['macro_f1']*100:.2f}%)")
    print(f"Macro ROC-AUC:         {ensemble_metrics['macro_auc']:.4f}")
    print(f"Cohen\s Kappa:         {ensemble_metrics['kappa']:.4f}")
    print(f"Macro Sensitivity:     {ensemble_metrics['macro_sensitivity']:.4f}")
    print(f"Macro Specificity:     {ensemble_metrics['macro_specificity']:.4f}")

    if "per_class" in ensemble_metrics:
        print("\n--- Per-Class Performance Breakdown (60 Test Images/Class) ---")
        rows = []
        for row in ensemble_metrics["per_class"]:
            rows.append({
                "Disease": row["class_name"],
                "Sensitivity": f"{row['sensitivity']*100:.1f}%",
                "Specificity": f"{row['specificity']*100:.1f}%",
                "F1-Score": f"{row['f1_score']*100:.1f}%",
            })
        df_per_class = pd.DataFrame(rows)
        print(df_per_class.to_string(index=False))

    # Save model probabilities
    np.savez(output_dir / "model_probs.npz", ground_truth=ground_truth, model_probs=np.array(model_probs_list))

    # Save metrics JSON
    with open(output_dir / "ms_tta_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_metrics, f, indent=2)

    # Save Confusion Matrix
    plot_confusion_matrix(
        ground_truth,
        fused_preds,
        class_names,
        str(output_dir / "ms_tta_confusion_matrix.png"),
        title="Multi-Scale TTA Test Confusion Matrix",
    )
    print(f"\nArtifacts successfully written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
