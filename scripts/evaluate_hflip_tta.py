#!/usr/bin/env python3
"""
Predetermined Horizontal-Flip Test-Time Augmentation (TTA) Evaluation Script
for ConvNeXt-Small (384x384 CLAHE) Clean Benchmark Baseline.

This script executes Phase 1 of the Accuracy Improvement Phase:
1. Loads the canonical baseline checkpoint and configuration without modifications.
2. Asserts SHA-256 hash of the checkpoint matches the canonical record.
3. Loads the clean 10-class validation cohort (data_clean/val, N=562).
4. Performs pure FP32 inference for:
   a. Original image (single-pass verification)
   b. Horizontally flipped image (torch.flip along width axis)
   c. TTA ensemble: exact FP32 average of (P_orig + P_flip) / 2.0
5. Verifies 100% concordance of single-pass predictions against baseline reconciled records.
6. Computes comprehensive classification metrics (Accuracy, Balanced Accuracy, Macro-F1,
   Weighted-F1, Quadratic Cohen's Kappa, ROC-AUC, Per-Class Sensitivity/Recall, Specificity,
   Precision, and F1).
7. Computes McNemar's paired test and tracks all corrected and newly misclassified images.
8. Saves all artifacts to a separate experiment directory without overwriting baseline files.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, plot_confusion_matrix, confusion_matrix
from torch.utils.data import DataLoader


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def mcnemar_test(b_incorrect_tta_correct: int, b_correct_tta_incorrect: int) -> Dict[str, float]:
    """Calculate McNemar test with Edwards continuity correction."""
    b = b_incorrect_tta_correct  # Baseline 0, TTA 1
    c = b_correct_tta_incorrect  # Baseline 1, TTA 0
    total_discordant = b + c
    if total_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    stat = (abs(b - c) - 1.0) ** 2 / total_discordant
    p_val = float(1.0 - chi2.cdf(stat, df=1))
    return {"statistic": float(stat), "p_value": p_val, "b": b, "c": c}


def run_hflip_tta(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    config_file = Path(config_path)
    ckpt_file = Path(checkpoint_path)
    out_dir = Path(output_dir)

    assert config_file.exists(), f"Config file not found: {config_file}"
    assert ckpt_file.exists(), f"Checkpoint file not found: {ckpt_file}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Checkpoint Verification
    ckpt_sha256 = compute_sha256(ckpt_file)
    print(f"\n=======================================================")
    print(f"Predetermined Horizontal-Flip TTA Evaluation")
    print(f"Model Checkpoint: {ckpt_file}")
    print(f"Checkpoint SHA-256: {ckpt_sha256}")
    print(f"Output Directory: {out_dir}")
    print(f"=======================================================\n")

    device = torch.device(device_name)

    # 2. Dataset and Loader setup
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
    print(f"Validation cohort: {n_samples} samples across {num_classes} classes.")
    assert n_samples == 562, f"Expected exactly 562 validation samples, found {n_samples}"

    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["training"].get("num_workers", 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model Loading
    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Deterministic Dual-Pass Inference (Original + H-Flip)
    all_targets = []
    all_probs_orig = []
    all_probs_flip = []
    all_probs_tta = []

    print("Running FP32 deterministic original and horizontal-flip passes...")
    with torch.no_grad():
        for step, (images, targets) in enumerate(val_loader):
            images_orig = images.to(device, non_blocking=True)
            # Horizontal flip: reverse along width dimension (dim 3: B, C, H, W)
            images_flip = torch.flip(images_orig, dims=[3])

            # Forward passes in FP32
            logits_orig = model(images_orig).float()
            logits_flip = model(images_flip).float()

            probs_orig = torch.softmax(logits_orig, dim=1)
            probs_flip = torch.softmax(logits_flip, dim=1)
            probs_tta = 0.5 * (probs_orig + probs_flip)

            all_targets.append(targets.numpy())
            all_probs_orig.append(probs_orig.cpu().numpy())
            all_probs_flip.append(probs_flip.cpu().numpy())
            all_probs_tta.append(probs_tta.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_prob_orig = np.concatenate(all_probs_orig)
    y_prob_flip = np.concatenate(all_probs_flip)
    y_prob_tta = np.concatenate(all_probs_tta)

    y_pred_orig = np.argmax(y_prob_orig, axis=1)
    y_pred_flip = np.argmax(y_prob_flip, axis=1)
    y_pred_tta = np.argmax(y_prob_tta, axis=1)

    # 5. Baseline Re-verification Sanity Check
    baseline_csv_path = REPO_ROOT / "outputs/clean_10class_convnext_small_384_clean/val_predictions_reconciled.csv"
    if baseline_csv_path.exists():
        df_base = pd.read_csv(baseline_csv_path)
        assert len(df_base) == 562, f"Baseline length mismatch: {len(df_base)}"
        # Check alignment of ground truths
        assert np.array_equal(df_base["true_label"].values, y_true), "Ground truth alignment mismatch!"
        # Check single-pass predicted labels
        orig_match_rate = np.mean(df_base["pred_label"].values == y_pred_orig)
        print(f"Single-pass re-verification concordance with baseline CSV: {orig_match_rate * 100:.2f}%")
        assert orig_match_rate == 1.0, f"Expected 100% match with baseline predictions, got {orig_match_rate * 100:.2f}%"

        # Check maximum absolute difference in predicted probabilities
        prob_cols = [f"prob_{c}" for c in class_names]
        base_probs = df_base[prob_cols].values
        max_prob_diff = np.max(np.abs(base_probs - y_prob_orig))
        print(f"Max absolute probability difference from baseline record: {max_prob_diff:.2e}")
        assert max_prob_diff < 1e-4, f"Probability drift detected: {max_prob_diff}"

    # 6. Compute Comprehensive Metrics for Baseline, Flip, and TTA
    metrics_orig = compute_metrics(y_true, y_pred_orig, y_prob_orig, class_names=class_names)
    metrics_flip = compute_metrics(y_true, y_pred_flip, y_prob_flip, class_names=class_names)
    metrics_tta = compute_metrics(y_true, y_pred_tta, y_prob_tta, class_names=class_names)

    cm_orig = confusion_matrix(y_true, y_pred_orig, labels=list(range(num_classes)))
    cm_flip = confusion_matrix(y_true, y_pred_flip, labels=list(range(num_classes)))
    cm_tta = confusion_matrix(y_true, y_pred_tta, labels=list(range(num_classes)))

    # Assert mathematical identities for TTA
    total_samples_cm = int(np.sum(cm_tta))
    correct_cm = int(np.sum(np.diag(cm_tta)))
    acc_cm = correct_cm / total_samples_cm
    assert total_samples_cm == 562, f"TTA CM total samples mismatch: {total_samples_cm}"
    assert np.isclose(acc_cm, metrics_tta["accuracy"]), f"Accuracy identity failed: {acc_cm} vs {metrics_tta['accuracy']}"

    recalls_tta = []
    specificities_tta = []
    precisions_tta = []
    f1s_tta = []
    supports_tta = []

    for i in range(num_classes):
        tp = cm_tta[i, i]
        fn = np.sum(cm_tta[i, :]) - tp
        fp = np.sum(cm_tta[:, i]) - tp
        tn = np.sum(cm_tta) - (tp + fn + fp)
        supp = tp + fn

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls_tta.append(rec)
        specificities_tta.append(spec)
        precisions_tta.append(prec)
        f1s_tta.append(f1)
        supports_tta.append(supp)

    balanced_acc_tta = float(np.mean(recalls_tta))
    macro_f1_tta = float(np.mean(f1s_tta))
    macro_prec_tta = float(np.mean(precisions_tta))
    macro_spec_tta = float(np.mean(specificities_tta))

    assert np.isclose(balanced_acc_tta, metrics_tta["balanced_accuracy"]), "Balanced accuracy identity failed"
    assert np.isclose(macro_f1_tta, metrics_tta["macro_f1"]), "Macro-F1 identity failed"

    # 7. Compare TTA vs Baseline & Classify Sample Transitions
    corrected_indices = []
    newly_misclassified_indices = []
    both_incorrect_different_pred = []
    concordant_correct = []
    concordant_incorrect = []

    pred_rows = []
    for idx in range(n_samples):
        img_path, lbl = val_dataset.samples[idx]
        rel_path = os.path.relpath(img_path, REPO_ROOT)
        b_corr = bool(y_pred_orig[idx] == lbl)
        tta_corr = bool(y_pred_tta[idx] == lbl)

        status = ""
        if not b_corr and tta_corr:
            status = "CORRECTED_BY_TTA"
            corrected_indices.append(idx)
        elif b_corr and not tta_corr:
            status = "NEWLY_MISCLASSIFIED_BY_TTA"
            newly_misclassified_indices.append(idx)
        elif b_corr and tta_corr:
            status = "CONCORDANT_CORRECT"
            concordant_correct.append(idx)
        else:
            if y_pred_orig[idx] == y_pred_tta[idx]:
                status = "CONCORDANT_INCORRECT"
                concordant_incorrect.append(idx)
            else:
                status = "BOTH_INCORRECT_TRANSITION"
                both_incorrect_different_pred.append(idx)

        row = {
            "image_index": idx,
            "image_path": rel_path,
            "filename": os.path.basename(img_path),
            "true_label": int(lbl),
            "true_class": class_names[lbl],
            "baseline_pred_label": int(y_pred_orig[idx]),
            "baseline_pred_class": class_names[y_pred_orig[idx]],
            "baseline_correct": b_corr,
            "baseline_max_prob": float(np.max(y_prob_orig[idx])),
            "flip_pred_label": int(y_pred_flip[idx]),
            "flip_pred_class": class_names[y_pred_flip[idx]],
            "flip_correct": bool(y_pred_flip[idx] == lbl),
            "tta_pred_label": int(y_pred_tta[idx]),
            "tta_pred_class": class_names[y_pred_tta[idx]],
            "tta_correct": tta_corr,
            "tta_max_prob": float(np.max(y_prob_tta[idx])),
            "transition_status": status,
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_tta_{c_name}"] = float(y_prob_tta[idx, c_idx])
            row[f"prob_orig_{c_name}"] = float(y_prob_orig[idx, c_idx])
            row[f"prob_flip_{c_name}"] = float(y_prob_flip[idx, c_idx])

        pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)
    pred_csv_path = out_dir / "val_predictions_tta.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"Saved: {pred_csv_path}")

    # Confusion matrix CSV & Plot
    df_cm_tta = pd.DataFrame(cm_tta, index=class_names, columns=class_names)
    cm_csv_path = out_dir / "val_confusion_matrix_tta.csv"
    df_cm_tta.to_csv(cm_csv_path)
    print(f"Saved: {cm_csv_path}")

    cm_png_path = out_dir / "val_confusion_matrix_tta.png"
    plot_confusion_matrix(y_true, y_pred_tta, class_names, str(cm_png_path), title="ConvNeXt-Small H-Flip TTA Confusion Matrix")
    print(f"Saved: {cm_png_path}")

    # McNemar's test
    mcnemar_res = mcnemar_test(
        b_incorrect_tta_correct=len(corrected_indices),
        b_correct_tta_incorrect=len(newly_misclassified_indices),
    )

    # Detailed Per-Class Comparison
    per_class_comparison = []
    for i in range(num_classes):
        cname = class_names[i]
        orig_pc = metrics_orig["per_class"][i]
        tta_pc = metrics_tta["per_class"][i]
        flip_pc = metrics_flip["per_class"][i]

        tp_orig = cm_orig[i, i]
        tp_tta = cm_tta[i, i]
        tp_diff = int(tp_tta - tp_orig)

        # Baseline precisions
        tp_b = cm_orig[i, i]
        fp_b = np.sum(cm_orig[:, i]) - tp_b
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0

        per_class_comparison.append({
            "class_name": cname,
            "support": int(supports_tta[i]),
            "baseline_tp": int(tp_orig),
            "tta_tp": int(tp_tta),
            "tp_delta": tp_diff,
            "baseline_recall": float(orig_pc["sensitivity"]),
            "tta_recall": float(tta_pc["sensitivity"]),
            "recall_delta": float(tta_pc["sensitivity"] - orig_pc["sensitivity"]),
            "baseline_precision": float(prec_b),
            "tta_precision": float(precisions_tta[i]),
            "precision_delta": float(precisions_tta[i] - prec_b),
            "baseline_specificity": float(orig_pc["specificity"]),
            "tta_specificity": float(tta_pc["specificity"]),
            "specificity_delta": float(tta_pc["specificity"] - orig_pc["specificity"]),
            "baseline_f1": float(orig_pc["f1_score"]),
            "tta_f1": float(tta_pc["f1_score"]),
            "f1_delta": float(tta_pc["f1_score"] - orig_pc["f1_score"]),
        })

    # Summary Dictionaries
    comparison_summary = {
        "evaluation_split": "data_clean/val",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_sha256,
        "n_samples": n_samples,
        "baseline_correct": int(np.sum(np.diag(cm_orig))),
        "baseline_accuracy": float(metrics_orig["accuracy"]),
        "baseline_balanced_accuracy": float(metrics_orig["balanced_accuracy"]),
        "baseline_macro_f1": float(metrics_orig["macro_f1"]),
        "baseline_macro_precision": float(np.mean([p["baseline_precision"] for p in per_class_comparison])),
        "baseline_macro_specificity": float(metrics_orig["macro_specificity"]),
        "baseline_weighted_f1": float(metrics_orig["weighted_f1"]),
        "baseline_cohen_kappa": float(metrics_orig["kappa"]),
        "hflip_pass_only_correct": int(np.sum(np.diag(cm_flip))),
        "hflip_pass_only_accuracy": float(metrics_flip["accuracy"]),
        "hflip_pass_only_macro_f1": float(metrics_flip["macro_f1"]),
        "tta_correct": int(correct_cm),
        "tta_accuracy": float(acc_cm),
        "tta_balanced_accuracy": float(balanced_acc_tta),
        "tta_macro_f1": float(macro_f1_tta),
        "tta_macro_precision": float(macro_prec_tta),
        "tta_macro_specificity": float(macro_spec_tta),
        "tta_weighted_f1": float(metrics_tta["weighted_f1"]),
        "tta_cohen_kappa": float(metrics_tta["kappa"]),
        "accuracy_delta_percentage_points": float((acc_cm - metrics_orig["accuracy"]) * 100),
        "balanced_accuracy_delta_percentage_points": float((balanced_acc_tta - metrics_orig["balanced_accuracy"]) * 100),
        "macro_f1_delta_percentage_points": float((macro_f1_tta - metrics_orig["macro_f1"]) * 100),
        "net_correct_delta": int(correct_cm - np.sum(np.diag(cm_orig))),
        "num_corrected_by_tta": len(corrected_indices),
        "num_newly_misclassified_by_tta": len(newly_misclassified_indices),
        "num_both_incorrect_transition": len(both_incorrect_different_pred),
        "num_concordant_correct": len(concordant_correct),
        "num_concordant_incorrect": len(concordant_incorrect),
        "mcnemar_test": mcnemar_res,
        "corrected_samples": [
            {
                "image_index": int(idx),
                "image_path": str(df_preds.iloc[idx]["image_path"]),
                "filename": str(df_preds.iloc[idx]["filename"]),
                "true_class": str(df_preds.iloc[idx]["true_class"]),
                "baseline_pred_class": str(df_preds.iloc[idx]["baseline_pred_class"]),
                "baseline_prob": float(df_preds.iloc[idx]["baseline_max_prob"]),
                "tta_pred_class": str(df_preds.iloc[idx]["tta_pred_class"]),
                "tta_prob": float(df_preds.iloc[idx]["tta_max_prob"]),
            }
            for idx in corrected_indices
        ],
        "newly_misclassified_samples": [
            {
                "image_index": int(idx),
                "image_path": str(df_preds.iloc[idx]["image_path"]),
                "filename": str(df_preds.iloc[idx]["filename"]),
                "true_class": str(df_preds.iloc[idx]["true_class"]),
                "baseline_pred_class": str(df_preds.iloc[idx]["baseline_pred_class"]),
                "baseline_prob": float(df_preds.iloc[idx]["baseline_max_prob"]),
                "tta_pred_class": str(df_preds.iloc[idx]["tta_pred_class"]),
                "tta_prob": float(df_preds.iloc[idx]["tta_max_prob"]),
            }
            for idx in newly_misclassified_indices
        ],
        "per_class_breakdown": per_class_comparison,
        "confusion_matrix_baseline": cm_orig.tolist(),
        "confusion_matrix_tta": cm_tta.tolist(),
    }

    comp_json_path = out_dir / "tta_vs_baseline_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"Saved: {comp_json_path}")

    # Standard val_evaluation_reconciled.json compatible format
    val_eval_tta = {
        "evaluation_split": "data_clean/val",
        "method": "Test-Time Augmentation (Predetermined Horizontal Flip, FP32 Mean)",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_sha256,
        "val_samples": n_samples,
        "correct_predictions": correct_cm,
        "overall_accuracy": float(acc_cm),
        "balanced_accuracy": float(balanced_acc_tta),
        "macro_f1": float(macro_f1_tta),
        "macro_precision": float(macro_prec_tta),
        "macro_sensitivity": float(balanced_acc_tta),
        "macro_specificity": float(macro_spec_tta),
        "weighted_f1": float(metrics_tta["weighted_f1"]),
        "cohen_kappa_quadratic": float(metrics_tta["kappa"]),
        "class_names": class_names,
        "confusion_matrix": cm_tta.tolist(),
        "per_class_summary": [
            {
                "class_name": class_names[i],
                "support": int(supports_tta[i]),
                "true_positives": int(cm_tta[i, i]),
                "sensitivity_recall": float(recalls_tta[i]),
                "specificity": float(specificities_tta[i]),
                "precision": float(precisions_tta[i]),
                "f1_score": float(f1s_tta[i]),
            }
            for i in range(num_classes)
        ]
    }
    eval_json_path = out_dir / "val_evaluation_tta.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(val_eval_tta, f, indent=2)
    print(f"Saved: {eval_json_path}")

    print("\n--- RESULTS SUMMARY ---")
    print(f"Baseline:  Acc: {metrics_orig['accuracy']*100:.2f}% ({np.sum(np.diag(cm_orig))}/562) | Macro-F1: {metrics_orig['macro_f1']*100:.2f}% | Bal-Acc: {metrics_orig['balanced_accuracy']*100:.2f}%")
    print(f"Flip-Pass: Acc: {metrics_flip['accuracy']*100:.2f}% ({np.sum(np.diag(cm_flip))}/562) | Macro-F1: {metrics_flip['macro_f1']*100:.2f}% | Bal-Acc: {metrics_flip['balanced_accuracy']*100:.2f}%")
    print(f"TTA (H-Flip): Acc: {acc_cm*100:.2f}% ({correct_cm}/562) | Macro-F1: {macro_f1_tta*100:.2f}% | Bal-Acc: {balanced_acc_tta*100:.2f}%")
    print(f"Delta: Acc: {(acc_cm - metrics_orig['accuracy'])*100:+.2f} pp | Macro-F1: {(macro_f1_tta - metrics_orig['macro_f1'])*100:+.2f} pp | Bal-Acc: {(balanced_acc_tta - metrics_orig['balanced_accuracy'])*100:+.2f} pp")
    print(f"Corrected samples: {len(corrected_indices)} | Newly misclassified samples: {len(newly_misclassified_indices)} | Net delta: {correct_cm - np.sum(np.diag(cm_orig)):+d}")
    print(f"McNemar test: statistic = {mcnemar_res['statistic']:.4f}, p-value = {mcnemar_res['p_value']:.4f}")
    print("-----------------------\n")

    return comparison_summary


def main():
    parser = argparse.ArgumentParser(description="Run predetermined horizontal-flip TTA on clean ConvNeXt-Small.")
    parser.add_argument("--config", default="configs/clean_10class_convnext_small_384_clean.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="outputs/clean_10class_convnext_small_384_clean/best_model.pth", help="Path to checkpoint PTH")
    parser.add_argument("--output-dir", default="outputs/clean_10class_convnext_small_384_tta_hflip", help="Directory to save TTA evaluation artifacts")
    args = parser.parse_args()

    run_hflip_tta(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
