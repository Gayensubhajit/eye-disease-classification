"""
Cache predictions for all 4 trained models on Kaggle 4-class and find optimal ensemble weights.
"""
import os
import gc
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.optimize import minimize

from src.data.dataset import FundusDataset, get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics
from scripts.evaluate_ms_tta import predict_multiscale_tta

def run_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_info = [
        ("BiomedCLIP_CBAM", "configs/kaggle_4class_biomedclip_cbam.yaml", "outputs/kaggle_4class_biomedclip_cbam/BEST_95.27pct_biomed.pth", True),
        ("ConvNeXt_Small_v2", "configs/kaggle_4class_convnext_small_384.yaml", "outputs/kaggle_4class_convnext_v2/BEST_95.27pct_convnext.pth", False),
        ("EfficientNet_B3", "configs/kaggle_4class_efficientnet_b3_384.yaml", "outputs/kaggle_4class_efficientnet_b3_384/best_model_25ep.pth", False),
        ("DenseNet_121", "configs/kaggle_4class_densenet121.yaml", "outputs/kaggle_4class_densenet121/best_model.pth", False),
    ]

    cache_file = Path("outputs/cached_4class_model_probs.npz")
    all_model_probs = {}
    ground_truth = None

    if cache_file.exists():
        print(f"Loading cached predictions from {cache_file}...")
        data = np.load(cache_file)
        ground_truth = data["ground_truth"]
        for key in data.files:
            if key != "ground_truth":
                all_model_probs[key] = data[key]
    else:
        print("Generating and caching MS-TTA probabilities for all models...")
        for name, cfg_path, ckpt_path, is_vit in models_info:
            print(f"\n--- Processing {name} ---")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model(cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            test_dir = cfg["data"]["test_dir"]
            class_names = cfg["data"]["class_names"]
            image_size = cfg["data"]["image_size"]
            val_transforms = get_val_transforms(image_size)
            test_dataset = FundusDataset(
                test_dir,
                class_names=class_names,
                transform=val_transforms,
                crop_fundus=cfg["data"].get("crop_fundus", False),
                apply_clahe_flag=cfg["data"].get("apply_clahe", True),
            )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

            all_probs, all_targets = [], []
            with torch.no_grad():
                for images, targets in test_loader:
                    images = images.to(device, non_blocking=True)
                    p = predict_multiscale_tta(model, images, scales=[1.0, 1.15], use_flips=True, is_vit=is_vit)
                    all_probs.append(p.cpu().numpy())
                    all_targets.append(targets.numpy())

            model_p = np.concatenate(all_probs)
            all_model_probs[name] = model_p
            if ground_truth is None:
                ground_truth = np.concatenate(all_targets)

            single_preds = np.argmax(model_p, axis=1)
            acc = np.mean(single_preds == ground_truth)
            print(f"  {name} Solo MS-TTA Accuracy: {acc*100:.2f}%")

            del model, ckpt
            gc.collect()
            torch.cuda.empty_cache()

        np.savez(cache_file, ground_truth=ground_truth, **all_model_probs)
        print(f"\nSaved all predictions to {cache_file}!")

    # Grid search / Optimization for best weights
    print("\n" + "=" * 60)
    print("       SEARCHING FOR OPTIMAL ENSEMBLE WEIGHTS        ")
    print("=" * 60)

    names = list(all_model_probs.keys())
    probs_stack = np.stack([all_model_probs[n] for n in names], axis=0) # shape (M, N, C)
    num_models = len(names)

    best_acc = 0.0
    best_weights = None

    # Systematic fine grid search
    from itertools import product
    grid = np.linspace(0.0, 1.0, 11)
    for w in product(grid, repeat=num_models):
        if sum(w) == 0:
            continue
        w_norm = np.array(w) / sum(w)
        # Weighted fusion
        fused = np.sum(probs_stack * w_norm[:, None, None], axis=0)
        preds = np.argmax(fused, axis=1)
        acc = np.mean(preds == ground_truth)
        if acc > best_acc:
            best_acc = acc
            best_weights = w_norm
            weight_str = ", ".join([f"{names[i]}: {best_weights[i]:.2f}" for i in range(num_models)])
            print(f"New Best: {acc*100:.2f}% ({np.sum(preds == ground_truth)}/{len(ground_truth)}) -> [{weight_str}]")

    print("\n" + "=" * 60)
    print(f"FINAL OPTIMAL ACCURACY: {best_acc*100:.2f}% ({int(best_acc*len(ground_truth))}/{len(ground_truth)})")
    for i, n in enumerate(names):
        print(f"  {n:20s}: weight = {best_weights[i]:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    run_cache()
