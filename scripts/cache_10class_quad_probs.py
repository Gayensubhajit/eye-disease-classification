"""
Cache predictions for all 4 trained models on the 10-Class Dataset for both Validation and Test sets.
Models:
1. ConvNeXt-Small 384 (CLAHE)
2. EfficientNet-B3 384 (CLAHE)
3. ResNet-50d 384 (CLAHE)
4. BiomedCLIP-CBAM 224 (CLAHE)
"""
import os
import gc
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FundusDataset, get_val_transforms
from src.models.backbone import build_model
from scripts.evaluate_ms_tta import predict_multiscale_tta

def run_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_info = [
        ("convnext", "configs/convnext_small_384_clahe.yaml", "outputs/convnext_small_384_clahe/best_model.pth", False),
        ("effnet", "configs/efficientnet_b3_384_clahe.yaml", "outputs/efficientnet_b3_384_clahe/best_model.pth", False),
        ("resnet", "configs/resnet50d_384_clahe.yaml", "outputs/resnet50d_384_clahe/best_model.pth", False),
        ("biomed", "configs/biomedclip_cbam_fusion.yaml", "outputs/biomedclip_cbam_fusion/best_model.pth", True),
    ]

    cache_file = Path("outputs/cached_10class_quad_probs.npz")
    results = {}
    val_targets = None
    test_targets = None
    class_names = None

    for key, cfg_path, ckpt_path, is_vit in models_info:
        print(f"\n==========================================")
        print(f"Processing Model: {key.upper()} ({ckpt_path})")
        print(f"==========================================")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Is training complete?")
            
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if class_names is None:
            class_names = cfg["data"]["class_names"]

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        image_size = cfg["data"]["image_size"]
        crop_fundus = cfg["data"].get("crop_fundus", False)
        apply_clahe = cfg["data"].get("apply_clahe", True)
        val_transforms = get_val_transforms(image_size)

        # 1. Validation Set
        print(f"Evaluating on Validation Set (600 images)...")
        val_dataset = FundusDataset(
            cfg["data"]["val_dir"],
            class_names=class_names,
            transform=val_transforms,
            crop_fundus=crop_fundus,
            apply_clahe_flag=apply_clahe,
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

        val_probs, curr_val_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                p = predict_multiscale_tta(model, images, scales=[1.0, 1.15], use_flips=True, is_vit=is_vit)
                val_probs.append(p.cpu().numpy())
                curr_val_targets.append(targets.numpy())

        v_probs = np.concatenate(val_probs)
        results[f"val_{key}"] = v_probs
        if val_targets is None:
            val_targets = np.concatenate(curr_val_targets)
            results["val_targets"] = val_targets

        v_acc = np.mean(np.argmax(v_probs, axis=1) == val_targets)
        print(f"  -> {key.upper()} Val MS-TTA Accuracy: {v_acc*100:.2f}% ({int(v_acc*len(val_targets))}/{len(val_targets)})")

        # 2. Test Set
        print(f"Evaluating on Test Set (600 images)...")
        test_dataset = FundusDataset(
            cfg["data"]["test_dir"],
            class_names=class_names,
            transform=val_transforms,
            crop_fundus=crop_fundus,
            apply_clahe_flag=apply_clahe,
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

        test_probs, curr_test_targets = [], []
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                p = predict_multiscale_tta(model, images, scales=[1.0, 1.15], use_flips=True, is_vit=is_vit)
                test_probs.append(p.cpu().numpy())
                curr_test_targets.append(targets.numpy())

        t_probs = np.concatenate(test_probs)
        results[f"test_{key}"] = t_probs
        if test_targets is None:
            test_targets = np.concatenate(curr_test_targets)
            results["test_targets"] = test_targets

        t_acc = np.mean(np.argmax(t_probs, axis=1) == test_targets)
        print(f"  -> {key.upper()} Test MS-TTA Accuracy: {t_acc*100:.2f}% ({int(t_acc*len(test_targets))}/{len(test_targets)})")

        del model, ckpt
        gc.collect()
        torch.cuda.empty_cache()

    results["class_names"] = np.array(class_names)
    np.savez(cache_file, **results)
    print(f"\nSuccessfully saved all 4 models val/test predictions to {cache_file}!")

if __name__ == "__main__":
    run_cache()
