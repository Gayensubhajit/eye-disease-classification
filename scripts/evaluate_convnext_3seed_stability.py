#!/usr/bin/env python3
"""
ConvNeXt-Small Multi-Seed Stability Evaluation Script (Milestone V8.1).
Strict Safeguards: Requires all three seeds (42, 1337, 2026) to be 100% complete with
valid checkpoints, training histories, validation summaries, and matching configurations.
If any checkpoint or artifact is missing, aborts immediately without writing incomplete metrics.
Strict Quarantine: Constructs a validation-only DataLoader. data_clean/test is NEVER loaded or accessed.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score
)
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbone import build_model
from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms

REQUIRED_SEEDS = {
    42: {
        "config": "configs/clean_10class_convnext_small_384_clean.yaml",
        "checkpoint": "outputs/clean_10class_convnext_small_384_clean/best_model.pth",
        "output_dir": "outputs/clean_10class_convnext_small_384_clean",
        "expected_sha": "238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d",
    },
    1337: {
        "config": "configs/clean_10class_convnext_small_384_seed1337.yaml",
        "checkpoint": "outputs/clean_10class_convnext_small_384_seed1337/best_model.pth",
        "output_dir": "outputs/clean_10class_convnext_small_384_seed1337",
        "expected_sha": None,
    },
    2026: {
        "config": "configs/clean_10class_convnext_small_384_seed2026.yaml",
        "checkpoint": "outputs/clean_10class_convnext_small_384_seed2026/best_model.pth",
        "output_dir": "outputs/clean_10class_convnext_small_384_seed2026",
        "expected_sha": None,
    },
}

def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def verify_configuration_equivalence(configs: Dict[int, Dict[str, Any]]):
    """Verify that all configurations share identical hyperparameters except seed and output dir."""
    ref_cfg = configs[42]
    
    # Check data section
    for key in ["train_dir", "val_dir", "image_size", "num_classes", "crop_fundus", "apply_clahe", "class_names"]:
        for s in [1337, 2026]:
            val_ref = ref_cfg["data"].get(key)
            val_cur = configs[s]["data"].get(key)
            assert val_ref == val_cur, f"Config mismatch in data.{key} between seed 42 and {s}: {val_ref} vs {val_cur}"

    # Check model section
    for key in ["name", "pretrained", "dropout"]:
        for s in [1337, 2026]:
            val_ref = ref_cfg["model"].get(key)
            val_cur = configs[s]["model"].get(key)
            assert val_ref == val_cur, f"Config mismatch in model.{key} between seed 42 and {s}: {val_ref} vs {val_cur}"

    # Check training section
    for key in ["batch_size", "epochs", "learning_rate", "weight_decay", "scheduler", "use_amp", "evaluate_test"]:
        for s in [1337, 2026]:
            val_ref = ref_cfg["training"].get(key)
            val_cur = configs[s]["training"].get(key)
            assert val_ref == val_cur, f"Config mismatch in training.{key} between seed 42 and {s}: {val_ref} vs {val_cur}"

    print("✅ Configuration equivalence verified across all three seeds.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ConvNeXt-Small Multi-Seed Stability Evaluator ===")
    print(f"Device: {device}")

    # PHASE 1: STRICT ARTIFACT & COMPLETION SAFEGUARD
    missing_artifacts = []
    loaded_configs = {}
    checkpoint_hashes = {}
    config_hashes = {}

    for s, info in REQUIRED_SEEDS.items():
        cfg_p = REPO_ROOT / info["config"]
        ckpt_p = REPO_ROOT / info["checkpoint"]
        out_dir = REPO_ROOT / info["output_dir"]
        hist_p = out_dir / "training_history.json"
        summ_p = out_dir / "val_evaluation_summary.json"

        if not cfg_p.exists():
            missing_artifacts.append(f"Seed {s}: Config missing ({cfg_p})")
        else:
            with open(cfg_p, "r") as f:
                loaded_configs[s] = yaml.safe_load(f)
            config_hashes[s] = compute_sha256(cfg_p)

        if not ckpt_p.exists():
            missing_artifacts.append(f"Seed {s}: Checkpoint missing ({ckpt_p})")
        else:
            h = compute_sha256(ckpt_p)
            checkpoint_hashes[s] = h
            if info["expected_sha"] and h != info["expected_sha"]:
                missing_artifacts.append(f"Seed {s}: Checkpoint SHA mismatch! Expected {info['expected_sha']}, got {h}")

        if not hist_p.exists():
            missing_artifacts.append(f"Seed {s}: Training history missing ({hist_p})")
        if not summ_p.exists():
            missing_artifacts.append(f"Seed {s}: Validation summary missing ({summ_p})")

    if missing_artifacts:
        print("\n❌ CRITICAL: ConvNeXt-Small 3-seed evaluation CANNOT proceed. Missing required artifacts:")
        for m in missing_artifacts:
            print(f"  - {m}")
        print("\nStopping with exit code 1 to prevent writing incomplete metrics.")
        sys.exit(1)

    print("✅ All three seed artifacts verified (checkpoints, configs, histories, summaries).")
    verify_configuration_equivalence(loaded_configs)

    # STRICT QUARANTINE: Construct validation-only DataLoader
    base_cfg = loaded_configs[42]
    assert base_cfg.get("training", {}).get("evaluate_test") is False, "Quarantine check: evaluate_test must be false!"

    image_size = base_cfg["data"]["image_size"]
    crop_fundus = base_cfg["data"].get("crop_fundus", True)
    apply_clahe_flag = base_cfg["data"].get("apply_clahe", True)
    class_names = base_cfg["data"]["class_names"]

    val_transforms = get_val_transforms(image_size)
    val_dataset = FundusDataset(
        base_cfg["data"]["val_dir"],
        class_names=class_names,
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    samples = val_dataset.samples
    assert len(samples) == 562, f"Expected 562 val samples, found {len(samples)}"

    out_stability_dir = REPO_ROOT / "outputs/clean_10class_convnext_small_3seed_stability"
    out_stability_dir.mkdir(parents=True, exist_ok=True)

    seed_results = {}
    all_seed_probs = {}
    all_seed_preds = {}

    for seed, info in REQUIRED_SEEDS.items():
        print(f"\n--- Evaluating Seed {seed} ---")
        cfg = loaded_configs[seed]
        ckpt_p = REPO_ROOT / info["checkpoint"]

        model = build_model(cfg).to(device)
        checkpoint = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        all_probs = []
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, dtype=torch.float32)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_targets.append(targets.numpy())
                all_preds.append(preds)

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_seed_probs[seed] = all_probs
        all_seed_preds[seed] = all_preds

        acc = float(accuracy_score(all_targets, all_preds))
        bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
        weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted"))
        kappa = float(cohen_kappa_score(all_targets, all_preds))
        try:
            auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"))
        except Exception:
            auc = 0.0

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
        
        per_class_metrics = []
        for c_idx, c_name in enumerate(class_names):
            supp = int(cm[c_idx, :].sum())
            rec = float(cm[c_idx, c_idx] / supp) if supp > 0 else 0.0
            col_sum = int(cm[:, c_idx].sum())
            prec = float(cm[c_idx, c_idx] / col_sum) if col_sum > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_class_metrics.append({
                "class_index": c_idx,
                "class_name": c_name,
                "support": supp,
                "recall": rec,
                "precision": prec,
                "f1_score": f1,
            })

        correct_count = int(np.sum(all_preds == all_targets))
        print(f"Seed {seed}: Acc = {acc*100:.2f}% ({correct_count}/{len(all_targets)}) | Macro-F1 = {macro_f1*100:.2f}% | BalAcc = {bal_acc*100:.2f}%")

        # Save per-image predictions
        pred_records = []
        for i, (img_path, true_label_idx) in enumerate(samples):
            pred_idx = int(all_preds[i])
            pred_records.append({
                "sample_index": i,
                "image_path": str(img_path),
                "true_label_idx": int(true_label_idx),
                "true_label_name": class_names[true_label_idx],
                "predicted_label_idx": pred_idx,
                "predicted_label_name": class_names[pred_idx],
                "is_correct": bool(pred_idx == true_label_idx),
                "confidence": float(all_probs[i, pred_idx]),
                "true_class_score": float(all_probs[i, true_label_idx]),
                "probabilities": {class_names[c]: float(all_probs[i, c]) for c in range(len(class_names))}
            })

        pred_json_p = out_stability_dir / f"per_image_predictions_seed{seed}.json"
        with open(pred_json_p, "w") as f:
            json.dump(pred_records, f, indent=2)

        pred_csv_p = out_stability_dir / f"per_image_predictions_seed{seed}.csv"
        df_preds = pd.DataFrame([{
            "sample_index": r["sample_index"],
            "image_path": r["image_path"],
            "true_label_idx": r["true_label_idx"],
            "true_label_name": r["true_label_name"],
            "predicted_label_idx": r["predicted_label_idx"],
            "predicted_label_name": r["predicted_label_name"],
            "is_correct": r["is_correct"],
            "confidence": r["confidence"],
            "true_class_score": r["true_class_score"]
        } for r in pred_records])
        df_preds.to_csv(pred_csv_p, index=False)

        seed_results[seed] = {
            "checkpoint_sha256": checkpoint_hashes[seed],
            "config_sha256": config_hashes[seed],
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "kappa": kappa,
            "macro_auc": auc,
            "correct_samples": correct_count,
            "total_samples": len(all_targets),
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist()
        }

    # Aggregate 3-seed statistics (ddof=1)
    accs = [seed_results[s]["accuracy"] for s in [42, 1337, 2026]]
    bal_accs = [seed_results[s]["balanced_accuracy"] for s in [42, 1337, 2026]]
    mf1s = [seed_results[s]["macro_f1"] for s in [42, 1337, 2026]]
    wf1s = [seed_results[s]["weighted_f1"] for s in [42, 1337, 2026]]
    kappas = [seed_results[s]["kappa"] for s in [42, 1337, 2026]]
    aucs = [seed_results[s]["macro_auc"] for s in [42, 1337, 2026]]

    agg_per_class = []
    for c_idx, c_name in enumerate(class_names):
        supp = seed_results[42]["per_class"][c_idx]["support"]
        recs = [seed_results[s]["per_class"][c_idx]["recall"] for s in [42, 1337, 2026]]
        precs = [seed_results[s]["per_class"][c_idx]["precision"] for s in [42, 1337, 2026]]
        f1s = [seed_results[s]["per_class"][c_idx]["f1_score"] for s in [42, 1337, 2026]]
        agg_per_class.append({
            "class_index": c_idx,
            "class_name": c_name,
            "support": supp,
            "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs, ddof=1))},
            "precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs, ddof=1))},
            "f1_score": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s, ddof=1))},
        })

    final_payload = {
        "model": "convnext_small",
        "resolution": 384,
        "preprocessing": "crop_fundus + clahe",
        "cohort_size": 562,
        "seeds": seed_results,
        "aggregate_headline": {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1))},
            "balanced_accuracy": {"mean": float(np.mean(bal_accs)), "std": float(np.std(bal_accs, ddof=1))},
            "macro_f1": {"mean": float(np.mean(mf1s)), "std": float(np.std(mf1s, ddof=1))},
            "weighted_f1": {"mean": float(np.mean(wf1s)), "std": float(np.std(wf1s, ddof=1))},
            "kappa": {"mean": float(np.mean(kappas)), "std": float(np.std(kappas, ddof=1))},
            "macro_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))},
        },
        "aggregate_per_class": agg_per_class,
    }

    summary_p = out_stability_dir / "convnext_stability_metrics.json"
    with open(summary_p, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n=======================================================")
    print(f"ConvNeXt-Small 3-Seed Aggregate Performance (N=3):")
    print(f"  Accuracy:          {final_payload['aggregate_headline']['accuracy']['mean']*100:.2f}% ± {final_payload['aggregate_headline']['accuracy']['std']*100:.2f}%")
    print(f"  Macro-F1:          {final_payload['aggregate_headline']['macro_f1']['mean']*100:.2f}% ± {final_payload['aggregate_headline']['macro_f1']['std']*100:.2f}%")
    print(f"  Balanced Accuracy: {final_payload['aggregate_headline']['balanced_accuracy']['mean']*100:.2f}% ± {final_payload['aggregate_headline']['balanced_accuracy']['std']*100:.2f}%")
    print(f"=======================================================")
    print(f"✅ Successfully saved verified 3-seed metrics to {summary_p.name}")

if __name__ == "__main__":
    main()
