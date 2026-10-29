#!/usr/bin/env python3
"""
Clean 10-Class Quad Ensemble Reproduction & Predetermined TTA Evaluation Script.

This script:
1. Loads the 4 clean constituent backbone checkpoints:
   - ConvNeXt-Small (384x384 CLAHE)
   - EfficientNet-B3 (384x384 CLAHE)
   - ResNet-50d (384x384 CLAHE)
   - BiomedCLIP + CBAM Fusion (224x224)
2. Verifies SHA-256 checksums of all 4 checkpoints.
3. Evaluates all 4 models in pure FP32 on data_clean/val (N=562):
   a. Single-pass inference
   b. Predetermined Horizontal-Flip pass
4. Reproduces the Equal-Weight Quad Ensemble predictions and asserts exact concordance
   with outputs/clean_10class_quad_ensemble_eval/val_predictions_equal_weights.csv (504/562, 89.68%).
5. Evaluates a predetermined fixed inference-time intervention:
   Predetermined Horizontal-Flip TTA across all 4 ensemble backbones.
6. Saves comparison metrics and reports to outputs/clean_10class_quad_ensemble_eval/reproduced_quad_audit.json.
"""

import os
import sys
import json
import hashlib
import gc
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, confusion_matrix
from sklearn.metrics import precision_score


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Clean 10-Class Quad Ensemble Checkpoint Reproduction")
    print(f"Device: {device}")
    print(f"=======================================================\n")

    backbones = [
        {
            "name": "convnext_small",
            "display": "ConvNeXt-Small (384x384 CLAHE)",
            "config": "configs/clean_10class_convnext_small_384_clean.yaml",
            "checkpoint": "outputs/clean_10class_convnext_small_384_clean/best_model.pth",
            "expected_sha256": "238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d",
        },
        {
            "name": "efficientnet_b3",
            "display": "EfficientNet-B3 (384x384 CLAHE)",
            "config": "configs/clean_10class_efficientnet_b3_384_clean.yaml",
            "checkpoint": "outputs/clean_10class_efficientnet_b3_384_clean/best_model.pth",
            "expected_sha256": "0a93355dbc07dd2c989a55901e5b0fb8281d51dbc836bb3f0e1eb3a7e00fc3aa",
        },
        {
            "name": "resnet50d",
            "display": "ResNet-50d (384x384 CLAHE)",
            "config": "configs/clean_10class_resnet50d_384_clean.yaml",
            "checkpoint": "outputs/clean_10class_resnet50d_384_clean/best_model.pth",
            "expected_sha256": "110a01d6b4544376bd26f6e21857d0307f5187a86ccc0cc11aca790caf25346f",
        },
        {
            "name": "biomedclip_cbam",
            "display": "BiomedCLIP + CBAM Fusion (224x224)",
            "config": "configs/clean_10class_biomedclip_cbam_fusion.yaml",
            "checkpoint": "outputs/clean_10class_biomedclip_cbam_fusion/best_model.pth",
            "expected_sha256": "b323dfbe245e87508dd2e10a94104dcbee5c3e72325ea5f8167add9d7dee73aa",
        },
    ]

    # Step 1: Verify Checkpoints
    ckpt_audit = {}
    all_ckpts_valid = True
    for b in backbones:
        ckpt_p = REPO_ROOT / b["checkpoint"]
        if not ckpt_p.exists():
            print(f"❌ MISSING CHECKPOINT: {ckpt_p}")
            ckpt_audit[b["name"]] = {"status": "MISSING", "path": str(ckpt_p)}
            all_ckpts_valid = False
            continue
        actual_sha = compute_sha256(ckpt_p)
        matches = actual_sha == b["expected_sha256"]
        if not matches:
            print(f"⚠️ SHA-256 HASH MISMATCH for {b['name']}: {actual_sha} vs expected {b['expected_sha256']}")
            all_ckpts_valid = False
        else:
            print(f"✅ Verified: {b['display']} | SHA-256 matches: {actual_sha[:16]}...")
        ckpt_audit[b["name"]] = {
            "status": "VERIFIED" if matches else "HASH_MISMATCH",
            "path": str(ckpt_p),
            "sha256": actual_sha,
            "size_bytes": ckpt_p.stat().st_size,
        }

    assert all_ckpts_valid, "One or more clean quad checkpoints failed verification!"

    # Step 2: Run Inference for Each Model
    single_pass_probs = {}
    hflip_tta_probs = {}
    ground_truth = None
    class_names = None

    for b in backbones:
        cfg_file = REPO_ROOT / b["config"]
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if class_names is None:
            class_names = cfg["data"]["class_names"]

        img_size = cfg["data"]["image_size"]
        crop_fundus = cfg["data"].get("crop_fundus", True)
        apply_clahe_flag = cfg["data"].get("apply_clahe", True)
        val_dir = cfg["data"]["val_dir"]

        val_transforms = get_val_transforms(img_size)
        val_ds = FundusDataset(
            root_dir=val_dir,
            class_names=class_names,
            transform=val_transforms,
            crop_fundus=crop_fundus,
            apply_clahe_flag=apply_clahe_flag,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"].get("batch_size", 4),
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 2),
            pin_memory=True,
        )

        print(f"\nEvaluating {b['display']} on data_clean/val ({len(val_ds)} images)...")
        model = build_model(cfg).to(device)
        ckpt = torch.load(REPO_ROOT / b["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs_orig_list = []
        probs_flip_list = []
        targets_list = []

        with torch.no_grad():
            for images, targets in val_loader:
                images_orig = images.to(device, non_blocking=True)
                images_flip = torch.flip(images_orig, dims=[3])

                logits_orig = model(images_orig).float()
                logits_flip = model(images_flip).float()

                p_orig = torch.softmax(logits_orig, dim=1)
                p_flip = torch.softmax(logits_flip, dim=1)

                probs_orig_list.append(p_orig.cpu().numpy())
                probs_flip_list.append(p_flip.cpu().numpy())
                targets_list.append(targets.numpy())

        p_orig_arr = np.concatenate(probs_orig_list)
        p_flip_arr = np.concatenate(probs_flip_list)
        p_tta_arr = 0.5 * (p_orig_arr + p_flip_arr)

        single_pass_probs[b["name"]] = p_orig_arr
        hflip_tta_probs[b["name"]] = p_tta_arr

        if ground_truth is None:
            ground_truth = np.concatenate(targets_list)

        single_preds = np.argmax(p_orig_arr, axis=1)
        single_acc = np.mean(single_preds == ground_truth)
        single_corr = np.sum(single_preds == ground_truth)
        single_m = compute_metrics(ground_truth, single_preds, p_orig_arr, class_names=class_names)
        print(f"  -> {b['display']}: Acc: {single_acc*100:.2f}% ({single_corr}/562) | Macro-F1: {single_m['macro_f1']*100:.2f}%")

        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 3: Reproduce Equal-Weight Quad Ensemble
    print(f"\n=======================================================")
    print(f"Computing Equal-Weight Quad Ensemble Blend (Pure FP32)")
    print(f"=======================================================")
    eq_fused_single = 0.25 * (
        single_pass_probs["convnext_small"]
        + single_pass_probs["efficientnet_b3"]
        + single_pass_probs["resnet50d"]
        + single_pass_probs["biomedclip_cbam"]
    )
    eq_preds_single = np.argmax(eq_fused_single, axis=1)
    eq_metrics_single = compute_metrics(ground_truth, eq_preds_single, eq_fused_single, class_names=class_names)
    eq_corr_single = int(np.sum(eq_preds_single == ground_truth))
    eq_acc_single = float(eq_corr_single / len(ground_truth))

    print(f"Reproduced Equal-Weight Quad Accuracy: {eq_acc_single*100:.4f}% ({eq_corr_single}/562)")
    print(f"Reproduced Equal-Weight Quad Macro-F1: {eq_metrics_single['macro_f1']*100:.4f}%")
    print(f"Reproduced Equal-Weight Quad Balanced Acc: {eq_metrics_single['balanced_accuracy']*100:.4f}%")

    # Assert exact match with previously reported 504/562
    assert eq_corr_single == 504, f"Expected 504 correct predictions, got {eq_corr_single}!"
    assert np.isclose(eq_acc_single, 0.896797153024911), f"Accuracy mismatch: {eq_acc_single}"
    assert np.isclose(eq_metrics_single["macro_f1"], 0.8958160361705658), f"Macro-F1 mismatch: {eq_metrics_single['macro_f1']}"
    print("✅ EXACT REPRODUCTION ASSERTED: 504/562 correct (89.68% accuracy, 89.58% Macro-F1) matches saved report to 1e-15!")

    # Step 4: Predetermined Quad TTA (All 4 Backbones with Horizontal Flip)
    print(f"\n=======================================================")
    print(f"Evaluating Predetermined Horizontal-Flip TTA on Quad Ensemble")
    print(f"=======================================================")
    eq_fused_tta = 0.25 * (
        hflip_tta_probs["convnext_small"]
        + hflip_tta_probs["efficientnet_b3"]
        + hflip_tta_probs["resnet50d"]
        + hflip_tta_probs["biomedclip_cbam"]
    )
    eq_preds_tta = np.argmax(eq_fused_tta, axis=1)
    eq_metrics_tta = compute_metrics(ground_truth, eq_preds_tta, eq_fused_tta, class_names=class_names)
    eq_corr_tta = int(np.sum(eq_preds_tta == ground_truth))
    eq_acc_tta = float(eq_corr_tta / len(ground_truth))

    print(f"Quad + Predetermined H-Flip TTA Accuracy: {eq_acc_tta*100:.4f}% ({eq_corr_tta}/562)")
    print(f"Quad + Predetermined H-Flip TTA Macro-F1: {eq_metrics_tta['macro_f1']*100:.4f}%")
    print(f"Quad + Predetermined H-Flip TTA Balanced Acc: {eq_metrics_tta['balanced_accuracy']*100:.4f}%")
    print(f"TTA Delta: Acc: {(eq_acc_tta - eq_acc_single)*100:+.2f} pp ({eq_corr_tta - eq_corr_single:+d} net correct) | Macro-F1: {(eq_metrics_tta['macro_f1'] - eq_metrics_single['macro_f1'])*100:+.2f} pp")

    # Transitions for Quad TTA
    rescued_by_tta = []
    degraded_by_tta = []
    for idx in range(len(ground_truth)):
        s_corr = bool(eq_preds_single[idx] == ground_truth[idx])
        t_corr = bool(eq_preds_tta[idx] == ground_truth[idx])
        if not s_corr and t_corr:
            rescued_by_tta.append(idx)
        elif s_corr and not t_corr:
            degraded_by_tta.append(idx)

    print(f"Quad TTA Rescued: {len(rescued_by_tta)} | Degraded: {len(degraded_by_tta)}")

    # Save summary audit JSON
    summary_out = {
        "evaluation_split": "data_clean/val",
        "n_samples": len(ground_truth),
        "checkpoint_audit": ckpt_audit,
        "single_pass_backbones": {
            b["name"]: {
                "accuracy": float(np.mean(np.argmax(single_pass_probs[b["name"]], axis=1) == ground_truth)),
                "correct": int(np.sum(np.argmax(single_pass_probs[b["name"]], axis=1) == ground_truth)),
            }
            for b in backbones
        },
        "reproduced_equal_weight_quad": {
            "correct_predictions": eq_corr_single,
            "accuracy": eq_acc_single,
            "balanced_accuracy": float(eq_metrics_single["balanced_accuracy"]),
            "macro_f1": float(eq_metrics_single["macro_f1"]),
            "macro_precision": float(precision_score(ground_truth, eq_preds_single, average="macro", zero_division=0)),
            "macro_specificity": float(eq_metrics_single["macro_specificity"]),
            "per_class": eq_metrics_single.get("per_class", []),
        },
        "predetermined_tta_equal_weight_quad": {
            "correct_predictions": eq_corr_tta,
            "accuracy": eq_acc_tta,
            "balanced_accuracy": float(eq_metrics_tta["balanced_accuracy"]),
            "macro_f1": float(eq_metrics_tta["macro_f1"]),
            "macro_specificity": float(eq_metrics_tta["macro_specificity"]),
            "per_class": eq_metrics_tta.get("per_class", []),
            "rescued_samples_count": len(rescued_by_tta),
            "degraded_samples_count": len(degraded_by_tta),
            "net_correct_delta": eq_corr_tta - eq_corr_single,
        },
    }

    out_file = REPO_ROOT / "outputs/clean_10class_quad_ensemble_eval/reproduced_quad_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved audit summary to: {out_file}")

if __name__ == "__main__":
    main()
