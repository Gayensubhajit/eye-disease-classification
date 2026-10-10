#!/usr/bin/env python3
"""
Reconciliation & Verification Script for Clean Baseline Test Evaluation

This script:
1. Loads the frozen baseline checkpoint (outputs/clean_baseline_efficientnet_b0/best_model.pth).
2. Runs deterministic test evaluation on data_clean/test (563 images).
3. Evaluates predictions under the canonical evaluation mode (AMP mixed precision, as executed in train.py)
   and also documents pure FP32 predictions.
4. Generates an exact per-image prediction ledger:
   outputs/clean_baseline_efficientnet_b0/test_predictions.csv
5. Derives the 10x10 confusion matrix and all test metrics from the exact same prediction array.
6. Asserts the fundamental mathematical identities:
   - Accuracy == sum(diag(CM)) / sum(CM)
   - Balanced Accuracy == mean(Recall)
   - Macro-F1 == mean(F1)
   - Class supports == row sums of CM
7. Saves reconciled metrics to outputs/clean_baseline_efficientnet_b0/test_evaluation_reconciled.json.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import create_dataloaders
from src.models.backbone import build_model
from src.metrics import compute_metrics, confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CONFIG_PATH = REPO_ROOT / "configs" / "clean_baseline_efficientnet_b0.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "clean_baseline_efficientnet_b0"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"

def main():
    print("=== Reconciling Stage 2 Baseline Test Metrics & Confusion Matrix ===")

    assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Verify Checkpoint SHA-256
    sha256 = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    ckpt_hash = sha256.hexdigest()
    print(f"1. Checkpoint Path: {CHECKPOINT_PATH}")
    print(f"   Checkpoint SHA-256: {ckpt_hash}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"2. Evaluation Device: {device}")

    # Build DataLoader and Model
    _, _, test_loader = create_dataloaders(config)
    class_names = config["data"]["class_names"]
    num_classes = len(class_names)
    test_ds = test_loader.dataset
    N_test = len(test_ds)
    print(f"   Test Set Size: {N_test} images across {num_classes} classes.")
    assert N_test == 563, f"Expected 563 test images, got {N_test}"

    model = build_model(config).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Deterministic Inference Pass
    # Preprocessing: crop_fundus_area (threshold=10) -> Resize(224, 224) -> Normalize(ImageNet) -> ToTensorV2
    all_targets = []
    all_preds_amp = []
    all_probs_amp = []
    all_preds_fp32 = []

    print("\n3. Executing Deterministic Test Inference...")
    with torch.no_grad():
        for step, (images, targets) in enumerate(test_loader):
            images_gpu = images.to(device, non_blocking=True)
            
            # Canonical AMP evaluation (matching train.py evaluate_epoch)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs_amp = model(images_gpu)
            probs_amp = torch.softmax(outputs_amp, dim=1)
            preds_amp = torch.argmax(probs_amp, dim=1)

            # Autocast-disabled evaluation (no AMP)
            outputs_fp32 = model(images_gpu)
            probs_fp32 = torch.softmax(outputs_fp32, dim=1)
            preds_fp32 = torch.argmax(probs_fp32, dim=1)

            all_targets.append(targets.numpy())
            all_preds_amp.append(preds_amp.cpu().numpy())
            all_probs_amp.append(probs_amp.cpu().numpy())
            all_preds_fp32.append(preds_fp32.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds_amp)
    y_prob = np.concatenate(all_probs_amp)
    y_pred_fp32 = np.concatenate(all_preds_fp32)

    # 4. Precision Divergence Analysis
    diff_mask = (y_pred != y_pred_fp32)
    diff_count = np.sum(diff_mask)
    print(f"\n4. Inference Precision Comparison (AMP vs Autocast-Disabled):")
    print(f"   Diverging predictions: {diff_count} / {N_test} images ({diff_count/N_test*100:.2f}%)")
    if diff_count > 0:
        for idx in np.where(diff_mask)[0]:
            img_path, lbl = test_ds.samples[idx]
            rel_path = os.path.relpath(img_path, REPO_ROOT)
            print(f"     Image #{idx} ({rel_path}):")
            print(f"       True Class: {class_names[lbl]} ({lbl})")
            print(f"       AMP (float16)  Prediction: {class_names[y_pred[idx]]} ({y_pred[idx]}) -> {'CORRECT' if y_pred[idx] == lbl else 'INCORRECT'}")
            print(f"       Autocast-disabled Prediction: {class_names[y_pred_fp32[idx]]} ({y_pred_fp32[idx]}) -> {'CORRECT' if y_pred_fp32[idx] == lbl else 'INCORRECT'}")

    # 5. Derive Reconciled Metrics from Canonical (AMP) Predictions
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print("\n5. Mathematical Reconciliation Verification:")
    # Matrix totals
    total_samples_cm = int(np.sum(cm))
    correct_cm = int(np.sum(np.diag(cm)))
    acc_from_cm = correct_cm / total_samples_cm
    acc_reported = metrics["accuracy"]

    print(f"   Total Samples in CM: {total_samples_cm} (Expected: {N_test})")
    print(f"   Correct Predictions (Diagonal Sum): {correct_cm}")
    print(f"   Accuracy from CM: {acc_from_cm:.6f}")
    print(f"   Accuracy from metrics: {acc_reported:.6f}")

    assert total_samples_cm == N_test, "CM total samples mismatch!"
    assert np.isclose(acc_from_cm, acc_reported), f"Accuracy discrepancy: CM={acc_from_cm} vs metrics={acc_reported}"

    # Per-class metrics and identities
    recalls = []
    specificities = []
    precisions = []
    f1_scores = []
    class_supports = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        support = tp + fn

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        specificities.append(spec)
        precisions.append(prec)
        f1_scores.append(f1)
        class_supports.append(support)

        # Check against metrics['per_class']
        pc = metrics["per_class"][i]
        assert np.isclose(pc["sensitivity"], rec), f"Recall mismatch for class {i}"
        assert np.isclose(pc["specificity"], spec), f"Specificity mismatch for class {i}"
        assert np.isclose(pc["f1_score"], f1), f"F1 mismatch for class {i}"

    # Mathematical identity checks
    balanced_acc_calculated = float(np.mean(recalls))
    macro_f1_calculated = float(np.mean(f1_scores))
    macro_spec_calculated = float(np.mean(specificities))
    macro_prec_calculated = float(np.mean(precisions))

    print(f"\n   Mathematical Identities:")
    print(f"     Balanced Accuracy: {balanced_acc_calculated:.6f} vs metrics: {metrics['balanced_accuracy']:.6f}")
    print(f"     Macro-F1:          {macro_f1_calculated:.6f} vs metrics: {metrics['macro_f1']:.6f}")
    print(f"     Macro Specificity: {macro_spec_calculated:.6f} vs metrics: {metrics['macro_specificity']:.6f}")

    assert np.isclose(balanced_acc_calculated, metrics["balanced_accuracy"]), "Balanced accuracy identity failed!"
    assert np.isclose(macro_f1_calculated, metrics["macro_f1"]), "Macro-F1 identity failed!"
    assert np.isclose(macro_spec_calculated, metrics["macro_specificity"]), "Macro specificity identity failed!"
    print(f"     Macro Precision:   {macro_prec_calculated:.6f} (Expected: 0.869660)")
    assert np.isclose(macro_prec_calculated, 0.8696603, atol=1e-5), f"Macro precision identity failed: {macro_prec_calculated}"
    print("   ALL MATHEMATICAL IDENTITIES ASSERTED AND PASSED!")

    # 6. Save Reconciled Artifacts
    # Save per-image predictions CSV
    pred_rows = []
    for idx in range(N_test):
        img_path, lbl = test_ds.samples[idx]
        rel_path = os.path.relpath(img_path, REPO_ROOT)
        row = {
            "image_index": idx,
            "image_path": rel_path,
            "filename": os.path.basename(img_path),
            "true_label": lbl,
            "true_class": class_names[lbl],
            "pred_label": int(y_pred[idx]),
            "pred_class": class_names[y_pred[idx]],
            "correct": bool(y_pred[idx] == lbl),
            "max_probability": float(np.max(y_prob[idx])),
            "pred_label_autocast_disabled": int(y_pred_fp32[idx]),
            "pred_class_autocast_disabled": class_names[y_pred_fp32[idx]],
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(y_prob[idx, c_idx])
        pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)
    pred_csv_path = OUTPUT_DIR / "test_predictions_reconciled.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"\n6. Saved per-image predictions ledger: {pred_csv_path}")

    # Save reconciled confusion matrix CSV
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = OUTPUT_DIR / "test_confusion_matrix_reconciled.csv"
    df_cm.to_csv(cm_csv_path)
    print(f"   Saved reconciled confusion matrix CSV: {cm_csv_path}")

    # Save reconciled metrics JSON
    reconciled_results = {
        "evaluation_protocol": "CANONICAL_AMP_RECONCILED",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": ckpt_hash,
        "test_samples": N_test,
        "correct_predictions": correct_cm,
        "overall_accuracy": float(acc_from_cm),
        "balanced_accuracy": float(balanced_acc_calculated),
        "macro_f1": float(macro_f1_calculated),
        "macro_sensitivity": float(balanced_acc_calculated),
        "macro_specificity": float(macro_spec_calculated),
        "macro_precision": float(macro_prec_calculated),
        "weighted_f1": float(metrics["weighted_f1"]),
        "cohen_kappa_quadratic": float(metrics["kappa"]),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class_summary": [
            {
                "class_name": class_names[i],
                "support": int(class_supports[i]),
                "true_positives": int(cm[i, i]),
                "sensitivity_recall": float(recalls[i]),
                "specificity": float(specificities[i]),
                "precision": float(precisions[i]),
                "f1_score": float(f1_scores[i]),
            }
            for i in range(num_classes)
        ],
        "autocast_disabled_comparison": {
            "autocast_disabled_correct_predictions": int(np.sum(np.diag(confusion_matrix(y_true, y_pred_fp32)))),
            "autocast_disabled_overall_accuracy": float(accuracy_score(y_true, y_pred_fp32)),
            "autocast_disabled_macro_f1": float(f1_score(y_true, y_pred_fp32, average="macro", zero_division=0)),
            "autocast_disabled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_fp32)),
            "diverging_sample_count": int(diff_count),
        }
    }

    reconciled_json_path = OUTPUT_DIR / "test_evaluation_reconciled.json"
    with open(reconciled_json_path, "w", encoding="utf-8") as f:
        json.dump(reconciled_results, f, indent=2)
    print(f"   Saved reconciled test results JSON: {reconciled_json_path}")

    print("\n=== Stage 2 Test Reconciliation Completed Successfully ===")

if __name__ == "__main__":
    main()
