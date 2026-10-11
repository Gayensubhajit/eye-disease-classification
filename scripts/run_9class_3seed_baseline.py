#!/usr/bin/env python3
"""
Three-Seed Baseline Runner for the 9-Class Posterior-Segment Retinal Fundus Benchmark.

Architectural Protocol:
- Architecture: EfficientNet-B0 (ImageNet pre-trained)
- Task: 9-Class Posterior-Segment Color Fundus Classification (Excludes Pterygium)
- Seeds: 42, 1337, 2026
- Learning Rate: 3e-4 (AdamW)
- Weight Decay: 1e-4
- Loss: Standard Cross-Entropy
- Scheduler: Cosine Annealing (20 epochs)
- Batch Size: 16
- Resolution: 224x224
- Output Structure: outputs/clean_9class_baseline_effnet/seed_{seed}/
- Internal Test Set: STRICTLY QUARANTINED (no test inference)
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.models.backbone import build_model
from src.metrics import compute_metrics

SEEDS = [42, 1337, 2026]
OUTPUT_ROOT = ROOT / "outputs/clean_9class_baseline_effnet"

CLASS_NAMES = [
    "Central Serous Chorioretinopathy [Color Fundus]",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]

EXPECTED_VAL_SUPPORTS = {
    "Central Serous Chorioretinopathy [Color Fundus]": 11,
    "Diabetic Retinopathy": 191,
    "Disc Edema": 15,
    "Glaucoma": 124,
    "Healthy": 105,
    "Macular Scar": 47,
    "Myopia": 37,
    "Retinal Detachment": 13,
    "Retinitis Pigmentosa": 17,
}


def build_seed_config(seed: int, seed_dir: Path) -> Path:
    config_dict = {
        "seed": seed,
        "data": {
            "train_dir": "data_clean/train",
            "val_dir": "data_clean/val",
            "test_dir": "data_clean/test",
            "image_size": 224,
            "num_classes": 9,
            "class_names": CLASS_NAMES,
            "crop_fundus": True,
            "apply_clahe": False,
            "augmentation_mode": "conservative",
        },
        "model": {
            "name": "efficientnet_b0",
            "pretrained": True,
            "dropout": 0.2,
        },
        "loss": {
            "name": "cross_entropy",
        },
        "training": {
            "batch_size": 16,
            "epochs": 20,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "num_workers": 2,
            "device": "cuda",
            "use_weighted_sampler": False,
            "scheduler": "cosine",
            "evaluate_test": False,
        },
        "output": {
            "directory": str(seed_dir.resolve().relative_to(ROOT.resolve())) if seed_dir.resolve().is_relative_to(ROOT.resolve()) else str(seed_dir),
        },
    }

    config_path = seed_dir / "run_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    return config_path


def run_training_subprocess(config_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "src/train.py"),
        "--config",
        str(config_path),
        "--no-test-eval",
    ]
    env = os.environ.copy()
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"\n>>> Launching training: {' '.join(cmd)}")
    start = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    elapsed = time.time() - start
    print(f">>> Training completed in {elapsed/60:.2f} minutes.")


def evaluate_and_reconcile_seed(seed: int, seed_dir: Path, config_path: Path) -> Dict[str, Any]:
    print(f"\n=== Reconciling Validation Artifacts for Seed {seed} ===")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader (val strictly)
    _, val_loader, _ = create_dataloaders(config)
    val_ds = val_loader.dataset
    assert len(val_ds) == 560, f"Expected 560 val images, got {len(val_ds)}"

    # Load Model Checkpoint
    ckpt_path = seed_dir / "best_model.pth"
    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    best_epoch = ckpt["epoch"]
    val_loss_at_peak = ckpt["val_metrics"].get("val_loss", None)

    # Load training history to get exact train/val losses at best epoch
    with open(seed_dir / "training_history.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    best_hist_entry = next((h for h in history if h["epoch"] == best_epoch), None)
    train_loss_at_peak = best_hist_entry["train_loss"] if best_hist_entry else None
    val_loss_at_peak = best_hist_entry["val_loss"] if best_hist_entry else val_loss_at_peak

    # Run Deterministic Autocast-Disabled FP32 Inference
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)  # Pure FP32 pass
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # 1. Prediction Ledger
    pred_rows = []
    for idx, (path_obj, true_sample_label) in enumerate(val_ds.samples):
        assert int(y_true[idx]) == int(true_sample_label), f"Ground truth mismatch at sample {idx}: {y_true[idx]} vs {true_sample_label}"
        row = {
            "sample_index": idx,
            "image_path": str(path_obj.resolve().relative_to(ROOT.resolve())) if path_obj.resolve().is_relative_to(ROOT.resolve()) else str(path_obj),
            "true_label": int(y_true[idx]),
            "true_class": CLASS_NAMES[y_true[idx]],
            "pred_label": int(y_pred[idx]),
            "pred_class": CLASS_NAMES[y_pred[idx]],
            "is_correct": bool(y_true[idx] == y_pred[idx]),
            "max_probability": float(np.max(y_prob[idx])),
        }
        for c_idx, c_name in enumerate(CLASS_NAMES):
            row[f"prob_{c_name}"] = float(y_prob[idx, c_idx])
        pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(seed_dir / "val_predictions.csv", index=False)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_cm.to_csv(seed_dir / "val_confusion_matrix.csv")

    # 3. Metrics computation
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=CLASS_NAMES)

    total_samples = int(np.sum(cm))
    correct_samples = int(np.sum(np.diag(cm)))
    acc_cm = correct_samples / total_samples

    # Rigorous Mathematical Assertions
    assert total_samples == 560, f"Expected 560 samples in CM, got {total_samples}"
    assert np.isclose(acc_cm, metrics["accuracy"]), f"Accuracy mismatch: {acc_cm} vs {metrics['accuracy']}"
    assert correct_samples == sum(df_preds["is_correct"]), "Correct count mismatch!"

    # Verify per-class supports against manifest
    for c_idx, c_name in enumerate(CLASS_NAMES):
        obs_supp = int(np.sum(cm[c_idx, :]))
        exp_supp = EXPECTED_VAL_SUPPORTS[c_name]
        assert obs_supp == exp_supp, f"Class support mismatch for {c_name}: observed {obs_supp}, expected {exp_supp}"

    # Class-level metrics
    recalls = []
    precisions = []
    f1s = []
    specificities = []

    for c_idx in range(9):
        tp = int(cm[c_idx, c_idx])
        fn = int(np.sum(cm[c_idx, :]) - tp)
        fp = int(np.sum(cm[:, c_idx]) - tp)
        tn = int(total_samples - (tp + fn + fp))

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        specificities.append(spec)

    macro_f1 = float(np.mean(f1s))
    balanced_acc = float(np.mean(recalls))
    macro_prec = float(np.mean(precisions))
    macro_spec = float(np.mean(specificities))

    assert np.isclose(macro_f1, metrics["macro_f1"]), "Macro-F1 mismatch"
    assert np.isclose(balanced_acc, metrics["balanced_accuracy"]), "Balanced accuracy mismatch"

    # Build structured per-class dictionary
    per_class_dict = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tp_val = int(cm[c_idx, c_idx])
        fp_val = int(np.sum(cm[:, c_idx]) - tp_val)
        fn_val = int(np.sum(cm[c_idx, :]) - tp_val)
        tn_val = int(total_samples - (tp_val + fn_val + fp_val))
        per_class_dict[c_name] = {
            "validation_support": EXPECTED_VAL_SUPPORTS[c_name],
            "f1": float(f1s[c_idx]),
            "sensitivity": float(recalls[c_idx]),
            "recall": float(recalls[c_idx]),
            "precision": float(precisions[c_idx]),
            "specificity": float(specificities[c_idx]),
            "tp": tp_val,
            "fp": fp_val,
            "fn": fn_val,
            "tn": tn_val,
        }

    # Save reconciled metrics
    val_metrics_out = {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "total_samples": total_samples,
        "correct_count": correct_samples,
        "accuracy": float(acc_cm),
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_specificity": macro_spec,
        "train_loss_at_peak": float(train_loss_at_peak) if train_loss_at_peak is not None else None,
        "val_loss_at_peak": float(val_loss_at_peak) if val_loss_at_peak is not None else None,
        "checkpoint_sha256": ckpt_sha256,
        "checkpoint_path": str(ckpt_path.resolve().relative_to(ROOT.resolve())) if ckpt_path.resolve().is_relative_to(ROOT.resolve()) else str(ckpt_path),
        "per_class": per_class_dict,
    }

    with open(seed_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics_out, f, indent=2)

    with open(seed_dir / "best_model.pth.sha256", "w", encoding="utf-8") as f:
        f.write(f"{ckpt_sha256}  best_model.pth\n")

    print(f"Seed {seed} Reconciled Metrics: Val Macro-F1 = {macro_f1*100:.2f}%, "
          f"Balanced Acc = {balanced_acc*100:.2f}%, Accuracy = {acc_cm*100:.2f}% ({correct_samples}/560)")
    return val_metrics_out


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_seed_results = []

    print("=" * 70)
    print("STARTING 3-SEED 9-CLASS FUNDUS BASELINE (EFFICIENTNET-B0)")
    print(f"Seeds: {SEEDS}")
    print(f"Output Root: {OUTPUT_ROOT}")
    print("=" * 70)

    for seed in SEEDS:
        seed_dir = OUTPUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        config_path = build_seed_config(seed, seed_dir)

        val_metrics_file = seed_dir / "val_metrics.json"
        best_ckpt_file = seed_dir / "best_model.pth"
        if val_metrics_file.exists() and best_ckpt_file.exists():
            print(f"\n>>> Seed {seed} already completed. Loading existing reconciled metrics from {val_metrics_file}...")
            with open(val_metrics_file, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
            all_seed_results.append(metrics_dict)
            continue

        run_training_subprocess(config_path)
        metrics_dict = evaluate_and_reconcile_seed(seed, seed_dir, config_path)
        all_seed_results.append(metrics_dict)

    # Build Aggregate Results
    f1_list = [r["macro_f1"] for r in all_seed_results]
    bacc_list = [r["balanced_accuracy"] for r in all_seed_results]
    acc_list = [r["accuracy"] for r in all_seed_results]
    vloss_list = [r["val_loss_at_peak"] for r in all_seed_results if r["val_loss_at_peak"] is not None]

    # Per-class aggregates across seeds
    per_class_agg = {}
    for c_name in CLASS_NAMES:
        c_f1s = [r["per_class"][c_name]["f1"] for r in all_seed_results]
        c_recalls = [r["per_class"][c_name]["sensitivity"] for r in all_seed_results]
        c_precs = [r["per_class"][c_name]["precision"] for r in all_seed_results]
        c_supp = EXPECTED_VAL_SUPPORTS[c_name]

        per_class_agg[c_name] = {
            "validation_support": c_supp,
            "f1_mean": float(np.mean(c_f1s)),
            "f1_std": float(np.std(c_f1s, ddof=1)),
            "recall_mean": float(np.mean(c_recalls)),
            "recall_std": float(np.std(c_recalls, ddof=1)),
            "precision_mean": float(np.mean(c_precs)),
            "precision_std": float(np.std(c_precs, ddof=1)),
            "individual_seed_f1s": {f"seed_{SEEDS[i]}": float(c_f1s[i]) for i in range(len(SEEDS))},
        }

    aggregate_summary = {
        "experiment_title": "9-Class Posterior-Segment Retinal Baseline (EfficientNet-B0)",
        "protocol_version": "Research Protocol v1.1",
        "seeds_evaluated": SEEDS,
        "sample_count_validation": 560,
        "classes_count": 9,
        "methodological_scope": (
            "Variability across seeds 42, 1337, and 2026 reflects stochastic optimization "
            "and weight initialization dynamics under a fixed dataset split; it does not represent "
            "random-split or patient-level variability."
        ),
        "aggregate_metrics": {
            "macro_f1": {
                "mean": float(np.mean(f1_list)),
                "std": float(np.std(f1_list, ddof=1)),
                "min": float(np.min(f1_list)),
                "max": float(np.max(f1_list)),
                "values": f1_list,
            },
            "balanced_accuracy": {
                "mean": float(np.mean(bacc_list)),
                "std": float(np.std(bacc_list, ddof=1)),
                "min": float(np.min(bacc_list)),
                "max": float(np.max(bacc_list)),
                "values": bacc_list,
            },
            "overall_accuracy": {
                "mean": float(np.mean(acc_list)),
                "std": float(np.std(acc_list, ddof=1)),
                "min": float(np.min(acc_list)),
                "max": float(np.max(acc_list)),
                "values": acc_list,
            },
            "validation_loss_at_peak": {
                "mean": float(np.mean(vloss_list)) if vloss_list else None,
                "std": float(np.std(vloss_list, ddof=1)) if len(vloss_list) > 1 else None,
                "values": vloss_list,
            },
        },
        "individual_runs": {
            f"seed_{r['seed']}": {
                "best_epoch": r["best_epoch"],
                "macro_f1": r["macro_f1"],
                "balanced_accuracy": r["balanced_accuracy"],
                "accuracy": r["accuracy"],
                "correct_count": r["correct_count"],
                "val_loss": r["val_loss_at_peak"],
                "train_loss": r["train_loss_at_peak"],
                "checkpoint_sha256": r["checkpoint_sha256"],
            }
            for r in all_seed_results
        },
        "per_class_performance": per_class_agg,
    }

    agg_path = OUTPUT_ROOT / "aggregate_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("3-SEED BASELINE COMPLETE — AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"Validation Macro-F1:     {np.mean(f1_list)*100:.2f}% ± {np.std(f1_list, ddof=1)*100:.2f}% (Range: {np.min(f1_list)*100:.2f}% - {np.max(f1_list)*100:.2f}%)")
    print(f"Validation Balanced Acc: {np.mean(bacc_list)*100:.2f}% ± {np.std(bacc_list, ddof=1)*100:.2f}%")
    print(f"Validation Accuracy:     {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list, ddof=1)*100:.2f}%")
    print(f"Aggregate report written to: {agg_path}")
    print("Test set quarantine: ENFORCED (Zero test inferences executed).")


if __name__ == "__main__":
    main()
