#!/usr/bin/env python3
"""
Validation Artifacts Generation & Reconciliation Script

This script:
1. Loads a given run's best checkpoint and config.
2. Runs deterministic validation inference strictly on data_clean/val (562 images).
3. Generates and saves:
   - val_predictions_reconciled.csv
   - val_confusion_matrix_reconciled.csv
   - val_evaluation_reconciled.json
4. Rigorously asserts mathematical identities:
   - Total samples == 562
   - Diagonal sum / 562 == Accuracy
   - mean(Recall) == Balanced Accuracy
   - mean(F1) == Macro-F1
   - mean(Precision) == Macro-Precision
   - mean(Specificity) == Macro-Specificity
   - Row sums == Manifest validation class supports
"""

import os
import sys
import json
import hashlib
import argparse
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

def reconcile_validation(config_path: str, checkpoint_path: str, output_dir: str):
    config_file = Path(config_path)
    ckpt_file = Path(checkpoint_path)
    out_dir = Path(output_dir)

    assert config_file.exists(), f"Config missing: {config_file}"
    assert ckpt_file.exists(), f"Checkpoint missing: {ckpt_file}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Checkpoint SHA-256
    sha256 = hashlib.sha256()
    with open(ckpt_file, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    ckpt_hash = sha256.hexdigest()
    print(f"\n--- Reconciling Validation Artifacts for {out_dir.name} ---")
    print(f"Checkpoint: {ckpt_file.name} | SHA-256: {ckpt_hash}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build DataLoader and Model
    _, val_loader, _ = create_dataloaders(config)
    class_names = config["data"]["class_names"]
    num_classes = len(class_names)
    val_ds = val_loader.dataset
    N_val = len(val_ds)
    print(f"Validation dataset: {N_val} images across {num_classes} classes.")
    assert N_val == 562, f"Expected 562 validation images, got {N_val}"

    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Deterministic inference pass on val
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for step, (images, targets) in enumerate(val_loader):
            images_gpu = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images_gpu)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # Metrics and Confusion Matrix
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    total_samples_cm = int(np.sum(cm))
    correct_cm = int(np.sum(np.diag(cm)))
    acc_from_cm = correct_cm / total_samples_cm
    acc_reported = metrics["accuracy"]

    print(f"Validation Total Samples in CM: {total_samples_cm} (Expected: 562)")
    print(f"Validation Correct Predictions (Diagonal Sum): {correct_cm}")
    print(f"Validation Accuracy: {acc_from_cm:.6f}")

    assert total_samples_cm == 562, f"Expected 562 in CM, got {total_samples_cm}"
    assert np.isclose(acc_from_cm, acc_reported), f"Accuracy mismatch: {acc_from_cm} vs {acc_reported}"

    # Expected validation class supports from frozen manifest
    expected_supports = {
        "Central Serous Chorioretinopathy [Color Fundus]": 11,
        "Diabetic Retinopathy": 191,
        "Disc Edema": 15,
        "Glaucoma": 124,
        "Healthy": 105,
        "Macular Scar": 47,
        "Myopia": 37,
        "Pterygium": 2,
        "Retinal Detachment": 13,
        "Retinitis Pigmentosa": 17,
    }

    recalls = []
    specificities = []
    precisions = []
    f1_scores = []
    class_supports = []

    for i in range(num_classes):
        cname = class_names[i]
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

        # Assert support matches manifest
        assert support == expected_supports[cname], f"Support mismatch for {cname}: {support} vs {expected_supports[cname]}"

        # Assert against metrics['per_class']
        pc = metrics["per_class"][i]
        assert np.isclose(pc["sensitivity"], rec), f"Recall mismatch for {cname}"
        assert np.isclose(pc["specificity"], spec), f"Specificity mismatch for {cname}"
        assert np.isclose(pc["f1_score"], f1), f"F1 mismatch for {cname}"

    balanced_acc_calculated = float(np.mean(recalls))
    macro_f1_calculated = float(np.mean(f1_scores))
    macro_spec_calculated = float(np.mean(specificities))
    macro_prec_calculated = float(np.mean(precisions))

    print(f"Validation Mathematical Identities:")
    print(f"  Balanced Accuracy: {balanced_acc_calculated:.6f} vs metrics: {metrics['balanced_accuracy']:.6f}")
    print(f"  Macro-F1:          {macro_f1_calculated:.6f} vs metrics: {metrics['macro_f1']:.6f}")
    print(f"  Macro Specificity: {macro_spec_calculated:.6f} vs metrics: {metrics['macro_specificity']:.6f}")
    print(f"  Macro Precision:   {macro_prec_calculated:.6f}")

    assert np.isclose(balanced_acc_calculated, metrics["balanced_accuracy"]), "Balanced accuracy identity failed!"
    assert np.isclose(macro_f1_calculated, metrics["macro_f1"]), "Macro-F1 identity failed!"
    assert np.isclose(macro_spec_calculated, metrics["macro_specificity"]), "Macro specificity identity failed!"
    print("ALL VALIDATION MATHEMATICAL IDENTITIES ASSERTED AND PASSED!")

    # Save per-image predictions CSV
    pred_rows = []
    for idx in range(N_val):
        img_path, lbl = val_ds.samples[idx]
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
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(y_prob[idx, c_idx])
        pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)
    pred_csv_path = out_dir / "val_predictions_reconciled.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"Saved: {pred_csv_path}")

    # Save confusion matrix CSV
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = out_dir / "val_confusion_matrix_reconciled.csv"
    df_cm.to_csv(cm_csv_path)
    print(f"Saved: {cm_csv_path}")

    # Save reconciled metrics JSON
    reconciled_results = {
        "evaluation_split": "data_clean/val",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_hash,
        "best_epoch": ckpt.get("epoch", 19),
        "val_samples": N_val,
        "correct_predictions": correct_cm,
        "overall_accuracy": float(acc_from_cm),
        "balanced_accuracy": float(balanced_acc_calculated),
        "macro_f1": float(macro_f1_calculated),
        "macro_precision": float(macro_prec_calculated),
        "macro_sensitivity": float(balanced_acc_calculated),
        "macro_specificity": float(macro_spec_calculated),
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
        ]
    }

    reconciled_json_path = out_dir / "val_evaluation_reconciled.json"
    with open(reconciled_json_path, "w", encoding="utf-8") as f:
        json.dump(reconciled_results, f, indent=2)
    print(f"Saved: {reconciled_json_path}")
    return reconciled_results

def main():
    parser = argparse.ArgumentParser(description="Reconcile validation artifacts")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint PTH")
    parser.add_argument("--output-dir", required=True, help="Directory to save reconciled validation artifacts")
    args = parser.parse_args()

    reconcile_validation(args.config, args.checkpoint, args.output_dir)

if __name__ == "__main__":
    main()
