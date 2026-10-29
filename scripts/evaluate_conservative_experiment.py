#!/usr/bin/env python3
"""
Evaluation and Comparison Script for the Controlled Conservative-Augmentation Experiment.

This script:
1. Reconciles validation predictions for ConvNeXt-Small (conservative augmentation) on data_clean/val (N=562).
2. Performs Evaluation A (Standalone):
   Compares ConvNeXt-Small (Conservative) against ConvNeXt-Small (Original Clean Baseline).
3. Performs Evaluation B (Updated Quad Ensemble):
   Replaces ConvNeXt-Small baseline probabilities with Conservative ConvNeXt-Small probabilities in the
   Equal-Weight Quad (weights [0.25, 0.25, 0.25, 0.25]).
   Compares Updated Quad against the Original Equal-Weight Quad Baseline (504/562, 89.68%).
4. Rigorously computes:
   - Overall Accuracy, Balanced Accuracy, Macro-F1, Macro-Precision (via precision_score, NOT F1), Macro-Specificity
   - Per-class Recall, Specificity, Precision, and F1
   - Glaucoma-specific metrics and Glaucoma <-> Healthy/Myopia confusion counts
   - Paired McNemar tests and itemized lists of rescued and degraded samples
5. Saves all artifacts to outputs/clean_10class_convnext_small_384_conservative_geom_only/
   without modifying any baseline files.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, plot_confusion_matrix


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def mcnemar_test(b_incorrect_a_correct: int, b_correct_a_incorrect: int) -> Dict[str, float]:
    b = b_incorrect_a_correct
    c = b_correct_a_incorrect
    total_disc = b + c
    if total_disc == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    stat = (abs(b - c) - 1.0) ** 2 / total_disc
    p_val = float(1.0 - chi2.cdf(stat, df=1))
    return {"statistic": float(stat), "p_value": p_val, "b": b, "c": c}


def run_evaluation(
    conservative_dir: str = "outputs/clean_10class_convnext_small_384_conservative_geom_only",
    config_path: str = "configs/clean_10class_convnext_small_384_conservative.yaml",
):
    cons_dir = Path(conservative_dir)
    cfg_file = Path(config_path)

    ckpt_path = cons_dir / "best_model.pth"
    assert ckpt_path.exists(), f"Conservative checkpoint missing: {ckpt_path}"
    assert cfg_file.exists(), f"Conservative config missing: {cfg_file}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Conservative Augmentation Experiment Evaluation")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"=======================================================\n")

    ckpt_sha = compute_sha256(ckpt_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    image_size = config["data"]["image_size"]
    crop_fundus = config["data"].get("crop_fundus", True)
    apply_clahe_flag = config["data"].get("apply_clahe", True)
    class_names = config["data"]["class_names"]
    num_classes = len(class_names)
    val_dir = config["data"]["val_dir"]

    val_transforms = get_val_transforms(image_size)
    val_dataset = FundusDataset(
        root_dir=val_dir,
        class_names=class_names,
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    n_samples = len(val_dataset)
    assert n_samples == 562, f"Expected 562 samples, got {n_samples}"

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=True,
    )

    # 1. Run Pure FP32 Inference for New Checkpoint
    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_targets = []
    all_probs_cons = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).float()
            probs = torch.softmax(logits, dim=1)
            all_probs_cons.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    y_prob_cons = np.concatenate(all_probs_cons)
    y_pred_cons = np.argmax(y_prob_cons, axis=1)

    # Save per-image predictions CSV
    pred_rows = []
    for idx in range(n_samples):
        img_path, lbl = val_dataset.samples[idx]
        rel_path = os.path.relpath(img_path, REPO_ROOT)
        row = {
            "image_index": idx,
            "image_path": rel_path,
            "filename": os.path.basename(img_path),
            "true_label": int(lbl),
            "true_class": class_names[lbl],
            "pred_label": int(y_pred_cons[idx]),
            "pred_class": class_names[y_pred_cons[idx]],
            "correct": bool(y_pred_cons[idx] == lbl),
            "max_probability": float(np.max(y_prob_cons[idx])),
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(y_prob_cons[idx, c_idx])
        pred_rows.append(row)

    df_preds_cons = pd.DataFrame(pred_rows)
    pred_csv_path = cons_dir / "val_predictions_reconciled.csv"
    df_preds_cons.to_csv(pred_csv_path, index=False)
    print(f"Saved: {pred_csv_path}")

    # Confusion matrix
    cm_cons = confusion_matrix(y_true, y_pred_cons, labels=list(range(num_classes)))
    df_cm_cons = pd.DataFrame(cm_cons, index=class_names, columns=class_names)
    cm_csv_path = cons_dir / "val_confusion_matrix_reconciled.csv"
    df_cm_cons.to_csv(cm_csv_path)
    print(f"Saved: {cm_csv_path}")

    cm_png_path = cons_dir / "best_val_confusion_matrix_reconciled.png"
    plot_confusion_matrix(y_true, y_pred_cons, class_names, str(cm_png_path), title="ConvNeXt-Small Conservative Aug Confusion Matrix")

    # Metrics for Standalone Conservative
    acc_cons = float(np.mean(y_pred_cons == y_true))
    corr_cons = int(np.sum(y_pred_cons == y_true))
    bal_acc_cons = float(balanced_accuracy_score(y_true, y_pred_cons))
    macro_f1_cons = float(f1_score(y_true, y_pred_cons, average="macro", zero_division=0))
    macro_prec_cons = float(precision_score(y_true, y_pred_cons, average="macro", zero_division=0))
    per_class_rec_cons = recall_score(y_true, y_pred_cons, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_prec_cons = precision_score(y_true, y_pred_cons, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1_cons = f1_score(y_true, y_pred_cons, average=None, labels=list(range(num_classes)), zero_division=0)

    # 2. Evaluation A: Compare Standalone Conservative vs Standalone Baseline
    baseline_csv = REPO_ROOT / "outputs/clean_10class_convnext_small_384_clean/val_predictions_reconciled.csv"
    df_base = pd.read_csv(baseline_csv)
    y_pred_base = df_base["pred_label"].values

    acc_base = float(np.mean(y_pred_base == y_true))
    corr_base = int(np.sum(y_pred_base == y_true))
    bal_acc_base = float(balanced_accuracy_score(y_true, y_pred_base))
    macro_f1_base = float(f1_score(y_true, y_pred_base, average="macro", zero_division=0))
    macro_prec_base = float(precision_score(y_true, y_pred_base, average="macro", zero_division=0))
    per_class_rec_base = recall_score(y_true, y_pred_base, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1_base = f1_score(y_true, y_pred_base, average=None, labels=list(range(num_classes)), zero_division=0)

    rescued_standalone = [idx for idx in range(n_samples) if y_pred_base[idx] != y_true[idx] and y_pred_cons[idx] == y_true[idx]]
    degraded_standalone = [idx for idx in range(n_samples) if y_pred_base[idx] == y_true[idx] and y_pred_cons[idx] != y_true[idx]]
    mcnemar_standalone = mcnemar_test(len(rescued_standalone), len(degraded_standalone))

    # Confusion Counts for Glaucoma <-> Healthy/Myopia
    glaucoma_idx = class_names.index("Glaucoma")
    healthy_idx = class_names.index("Healthy")
    myopia_idx = class_names.index("Myopia")

    cm_base = confusion_matrix(y_true, y_pred_base, labels=list(range(num_classes)))

    confusions_base = {
        "glaucoma_as_healthy": int(cm_base[glaucoma_idx, healthy_idx]),
        "healthy_as_glaucoma": int(cm_base[healthy_idx, glaucoma_idx]),
        "glaucoma_as_myopia": int(cm_base[glaucoma_idx, myopia_idx]),
        "myopia_as_glaucoma": int(cm_base[myopia_idx, glaucoma_idx]),
    }
    confusions_cons = {
        "glaucoma_as_healthy": int(cm_cons[glaucoma_idx, healthy_idx]),
        "healthy_as_glaucoma": int(cm_cons[healthy_idx, glaucoma_idx]),
        "glaucoma_as_myopia": int(cm_cons[glaucoma_idx, myopia_idx]),
        "myopia_as_glaucoma": int(cm_cons[myopia_idx, glaucoma_idx]),
    }

    # 3. Evaluation B: Updated Quad Ensemble
    # Load probabilities of EffNet, ResNet, BiomedCLIP
    df_eff = pd.read_csv(REPO_ROOT / "outputs/clean_10class_efficientnet_b3_384_clean/val_predictions_reconciled.csv")
    df_res = pd.read_csv(REPO_ROOT / "outputs/clean_10class_resnet50d_384_clean/val_predictions_reconciled.csv")
    df_bio = pd.read_csv(REPO_ROOT / "outputs/clean_10class_biomedclip_cbam_fusion/val_predictions_reconciled.csv")

    prob_cols = [f"prob_{c}" for c in class_names]
    p_eff = df_eff[prob_cols].values
    p_res = df_res[prob_cols].values
    p_bio = df_bio[prob_cols].values

    # Original Equal-Weight Quad
    p_conv_orig = df_base[prob_cols].values
    p_quad_orig = 0.25 * (p_conv_orig + p_eff + p_res + p_bio)
    y_pred_quad_orig = np.argmax(p_quad_orig, axis=1)

    # Updated Equal-Weight Quad (Replacing ConvNeXt)
    p_quad_updated = 0.25 * (y_prob_cons + p_eff + p_res + p_bio)
    y_pred_quad_updated = np.argmax(p_quad_updated, axis=1)

    acc_quad_orig = float(np.mean(y_pred_quad_orig == y_true))
    corr_quad_orig = int(np.sum(y_pred_quad_orig == y_true))
    macro_f1_quad_orig = float(f1_score(y_true, y_pred_quad_orig, average="macro", zero_division=0))
    bal_acc_quad_orig = float(balanced_accuracy_score(y_true, y_pred_quad_orig))
    macro_prec_quad_orig = float(precision_score(y_true, y_pred_quad_orig, average="macro", zero_division=0))
    per_class_rec_quad_orig = recall_score(y_true, y_pred_quad_orig, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1_quad_orig = f1_score(y_true, y_pred_quad_orig, average=None, labels=list(range(num_classes)), zero_division=0)

    acc_quad_upd = float(np.mean(y_pred_quad_updated == y_true))
    corr_quad_upd = int(np.sum(y_pred_quad_updated == y_true))
    macro_f1_quad_upd = float(f1_score(y_true, y_pred_quad_updated, average="macro", zero_division=0))
    bal_acc_quad_upd = float(balanced_accuracy_score(y_true, y_pred_quad_updated))
    macro_prec_quad_upd = float(precision_score(y_true, y_pred_quad_updated, average="macro", zero_division=0))
    per_class_rec_quad_upd = recall_score(y_true, y_pred_quad_updated, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1_quad_upd = f1_score(y_true, y_pred_quad_updated, average=None, labels=list(range(num_classes)), zero_division=0)

    rescued_quad = [idx for idx in range(n_samples) if y_pred_quad_orig[idx] != y_true[idx] and y_pred_quad_updated[idx] == y_true[idx]]
    degraded_quad = [idx for idx in range(n_samples) if y_pred_quad_orig[idx] == y_true[idx] and y_pred_quad_updated[idx] != y_true[idx]]
    mcnemar_quad = mcnemar_test(len(rescued_quad), len(degraded_quad))

    cm_quad_orig = confusion_matrix(y_true, y_pred_quad_orig, labels=list(range(num_classes)))
    cm_quad_upd = confusion_matrix(y_true, y_pred_quad_updated, labels=list(range(num_classes)))

    confusions_quad_orig = {
        "glaucoma_as_healthy": int(cm_quad_orig[glaucoma_idx, healthy_idx]),
        "healthy_as_glaucoma": int(cm_quad_orig[healthy_idx, glaucoma_idx]),
        "glaucoma_as_myopia": int(cm_quad_orig[glaucoma_idx, myopia_idx]),
        "myopia_as_glaucoma": int(cm_quad_orig[myopia_idx, glaucoma_idx]),
    }
    confusions_quad_upd = {
        "glaucoma_as_healthy": int(cm_quad_upd[glaucoma_idx, healthy_idx]),
        "healthy_as_glaucoma": int(cm_quad_upd[healthy_idx, glaucoma_idx]),
        "glaucoma_as_myopia": int(cm_quad_upd[glaucoma_idx, myopia_idx]),
        "myopia_as_glaucoma": int(cm_quad_upd[myopia_idx, glaucoma_idx]),
    }

    # Save Updated Quad Predictions CSV
    quad_upd_rows = []
    for idx in range(n_samples):
        img_path, lbl = val_dataset.samples[idx]
        rel_path = os.path.relpath(img_path, REPO_ROOT)
        row = {
            "image_index": idx,
            "image_path": rel_path,
            "filename": os.path.basename(img_path),
            "true_label": int(lbl),
            "true_class": class_names[lbl],
            "pred_label": int(y_pred_quad_updated[idx]),
            "pred_class": class_names[y_pred_quad_updated[idx]],
            "correct": bool(y_pred_quad_updated[idx] == lbl),
            "max_probability": float(np.max(p_quad_updated[idx])),
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(p_quad_updated[idx, c_idx])
        quad_upd_rows.append(row)

    df_quad_upd = pd.DataFrame(quad_upd_rows)
    quad_upd_csv_path = cons_dir / "val_predictions_updated_equal_weight_quad.csv"
    df_quad_upd.to_csv(quad_upd_csv_path, index=False)
    print(f"Saved: {quad_upd_csv_path}")

    # Build Comprehensive Comparison Artifact
    comparison_results = {
        "evaluation_split": "data_clean/val",
        "n_samples": n_samples,
        "checkpoint_sha256": ckpt_sha,
        "evaluation_A_standalone": {
            "model_name": "ConvNeXt-Small (384x384 CLAHE)",
            "baseline": {
                "checkpoint": "outputs/clean_10class_convnext_small_384_clean/best_model.pth",
                "accuracy": acc_base,
                "correct_samples": corr_base,
                "balanced_accuracy": bal_acc_base,
                "macro_f1": macro_f1_base,
                "macro_precision": macro_prec_base,
                "glaucoma_f1": float(per_class_f1_base[glaucoma_idx]),
                "glaucoma_recall": float(per_class_rec_base[glaucoma_idx]),
                "confusions": confusions_base,
            },
            "conservative": {
                "checkpoint": str(ckpt_path),
                "accuracy": acc_cons,
                "correct_samples": corr_cons,
                "balanced_accuracy": bal_acc_cons,
                "macro_f1": macro_f1_cons,
                "macro_precision": macro_prec_cons,
                "glaucoma_f1": float(per_class_f1_cons[glaucoma_idx]),
                "glaucoma_recall": float(per_class_rec_cons[glaucoma_idx]),
                "confusions": confusions_cons,
            },
            "delta": {
                "accuracy_pp": (acc_cons - acc_base) * 100,
                "net_correct": corr_cons - corr_base,
                "macro_f1_pp": (macro_f1_cons - macro_f1_base) * 100,
                "balanced_accuracy_pp": (bal_acc_cons - bal_acc_base) * 100,
                "glaucoma_f1_pp": (per_class_f1_cons[glaucoma_idx] - per_class_f1_base[glaucoma_idx]) * 100,
                "glaucoma_recall_pp": (per_class_rec_cons[glaucoma_idx] - per_class_rec_base[glaucoma_idx]) * 100,
            },
            "mcnemar": mcnemar_standalone,
            "rescued_count": len(rescued_standalone),
            "degraded_count": len(degraded_standalone),
        },
        "evaluation_B_quad_ensemble": {
            "model_name": "Equal-Weight Quad Ensemble (0.25 each)",
            "original_quad": {
                "accuracy": acc_quad_orig,
                "correct_samples": corr_quad_orig,
                "balanced_accuracy": bal_acc_quad_orig,
                "macro_f1": macro_f1_quad_orig,
                "macro_precision": macro_prec_quad_orig,
                "glaucoma_f1": float(per_class_f1_quad_orig[glaucoma_idx]),
                "glaucoma_recall": float(per_class_rec_quad_orig[glaucoma_idx]),
                "confusions": confusions_quad_orig,
            },
            "updated_quad": {
                "accuracy": acc_quad_upd,
                "correct_samples": corr_quad_upd,
                "balanced_accuracy": bal_acc_quad_upd,
                "macro_f1": macro_f1_quad_upd,
                "macro_precision": macro_prec_quad_upd,
                "glaucoma_f1": float(per_class_f1_quad_upd[glaucoma_idx]),
                "glaucoma_recall": float(per_class_rec_quad_upd[glaucoma_idx]),
                "confusions": confusions_quad_upd,
            },
            "delta": {
                "accuracy_pp": (acc_quad_upd - acc_quad_orig) * 100,
                "net_correct": corr_quad_upd - corr_quad_orig,
                "macro_f1_pp": (macro_f1_quad_upd - macro_f1_quad_orig) * 100,
                "balanced_accuracy_pp": (bal_acc_quad_upd - bal_acc_quad_orig) * 100,
                "glaucoma_f1_pp": (per_class_f1_quad_upd[glaucoma_idx] - per_class_f1_quad_orig[glaucoma_idx]) * 100,
                "glaucoma_recall_pp": (per_class_rec_quad_upd[glaucoma_idx] - per_class_rec_quad_orig[glaucoma_idx]) * 100,
            },
            "mcnemar": mcnemar_quad,
            "rescued_count": len(rescued_quad),
            "degraded_count": len(degraded_quad),
        },
        "per_class_breakdown": [
            {
                "class_name": class_names[i],
                "support": int(np.sum(y_true == i)),
                "convnext_baseline_f1": float(per_class_f1_base[i]),
                "convnext_conservative_f1": float(per_class_f1_cons[i]),
                "convnext_f1_delta_pp": float(per_class_f1_cons[i] - per_class_f1_base[i]) * 100,
                "quad_orig_f1": float(per_class_f1_quad_orig[i]),
                "quad_updated_f1": float(per_class_f1_quad_upd[i]),
                "quad_f1_delta_pp": float(per_class_f1_quad_upd[i] - per_class_f1_quad_orig[i]) * 100,
            }
            for i in range(num_classes)
        ]
    }

    report_json_path = cons_dir / "conservative_experiment_comparison.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved: {report_json_path}")

    print("\n--- STANDALONE CONVNEXT COMPARISON ---")
    print(f"Baseline:     Acc: {acc_base*100:.2f}% ({corr_base}/562) | Macro-F1: {macro_f1_base*100:.2f}% | Glaucoma F1: {per_class_f1_base[glaucoma_idx]*100:.2f}%")
    print(f"Conservative: Acc: {acc_cons*100:.2f}% ({corr_cons}/562) | Macro-F1: {macro_f1_cons*100:.2f}% | Glaucoma F1: {per_class_f1_cons[glaucoma_idx]*100:.2f}%")
    print(f"Delta:        Acc: {(acc_cons - acc_base)*100:+.2f} pp ({corr_cons - corr_base:+d}) | Macro-F1: {(macro_f1_cons - macro_f1_base)*100:+.2f} pp | Glaucoma F1: {(per_class_f1_cons[glaucoma_idx] - per_class_f1_base[glaucoma_idx])*100:+.2f} pp")

    print("\n--- EQUAL-WEIGHT QUAD ENSEMBLE COMPARISON ---")
    print(f"Original Quad: Acc: {acc_quad_orig*100:.2f}% ({corr_quad_orig}/562) | Macro-F1: {macro_f1_quad_orig*100:.2f}% | Glaucoma F1: {per_class_f1_quad_orig[glaucoma_idx]*100:.2f}%")
    print(f"Updated Quad:  Acc: {acc_quad_upd*100:.2f}% ({corr_quad_upd}/562) | Macro-F1: {macro_f1_quad_upd*100:.2f}% | Glaucoma F1: {per_class_f1_quad_upd[glaucoma_idx]*100:.2f}%")
    print(f"Delta:         Acc: {(acc_quad_upd - acc_quad_orig)*100:+.2f} pp ({corr_quad_upd - corr_quad_orig:+d}) | Macro-F1: {(macro_f1_quad_upd - macro_f1_quad_orig)*100:+.2f} pp | Glaucoma F1: {(per_class_f1_quad_upd[glaucoma_idx] - per_class_f1_quad_orig[glaucoma_idx])*100:+.2f} pp")
    print("--------------------------------------------\n")

    return comparison_results


if __name__ == "__main__":
    run_evaluation()
