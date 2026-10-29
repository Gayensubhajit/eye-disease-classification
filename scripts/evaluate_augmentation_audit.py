#!/usr/bin/env python3
"""
Milestone V8: Controlled Augmentation Experiment Evaluation Script.
Evaluates Arm A (No Augmentation) vs Arm B (Moderate Augmentation)
across Seeds 42, 1337, and 2026 on data_clean/val (N=562) under pure FP32.
Strict Quarantine: Constructs a validation-only DataLoader. data_clean/test is NEVER instantiated or accessed.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbone import build_model
from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms

EXPERIMENTS = {
    "arm_a_noaug": {
        "display": "Arm A (No Stochastic Augmentation)",
        "seeds": {
            42: {
                "config": "configs/clean_10class_effnet_b0_noaug_seed42.yaml",
                "checkpoint": "outputs/clean_10class_effnet_b0_noaug_seed42/best_model.pth",
                "output_dir": "outputs/clean_10class_effnet_b0_noaug_seed42",
            },
            1337: {
                "config": "configs/clean_10class_effnet_b0_noaug_seed1337.yaml",
                "checkpoint": "outputs/clean_10class_effnet_b0_noaug_seed1337/best_model.pth",
                "output_dir": "outputs/clean_10class_effnet_b0_noaug_seed1337",
            },
            2026: {
                "config": "configs/clean_10class_effnet_b0_noaug_seed2026.yaml",
                "checkpoint": "outputs/clean_10class_effnet_b0_noaug_seed2026/best_model.pth",
                "output_dir": "outputs/clean_10class_effnet_b0_noaug_seed2026",
            },
        }
    },
    "arm_b_aug": {
        "display": "Arm B (Moderate Stochastic Augmentation)",
        "seeds": {
            42: {
                "config": "configs/clean_10class_effnet_b0_aug_seed42.yaml",
                "checkpoint": "outputs/clean_10class_effnet_b0_aug_seed42/best_model.pth",
                "output_dir": "outputs/clean_10class_effnet_b0_aug_seed42",
            },
            1337: {
                "config": "configs/clean_10class_effnet_b0_aug_seed1337.yaml",
                "checkpoint": "outputs/clean_10class_effnet_b0_aug_seed1337/best_model.pth",
                "output_dir": "outputs/clean_10class_effnet_b0_aug_seed1337",
            },
            2026: {
                "config": "configs/clean_10class_effnet_b0_aug_seed2026.yaml",
                "checkpoint": "outputs/clean_10class_effnet_b0_aug_seed2026/best_model.pth",
                "output_dir": "outputs/clean_10class_effnet_b0_aug_seed2026",
            },
        }
    }
}

def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Milestone V8: Controlled Augmentation Evaluation ===")
    print(f"Device: {device}")

    base_cfg = yaml.safe_load(open(REPO_ROOT / EXPERIMENTS["arm_a_noaug"]["seeds"][42]["config"]))
    assert base_cfg.get("training", {}).get("evaluate_test") is False, "Quarantine check: evaluate_test must be false!"

    # Strict Quarantine: instantiate validation dataset ONLY
    image_size = base_cfg["data"]["image_size"]
    crop_fundus = base_cfg["data"].get("crop_fundus", True)
    apply_clahe_flag = base_cfg["data"].get("apply_clahe", False)
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
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    samples = val_dataset.samples
    assert len(samples) == 562, f"Expected 562 val samples, found {len(samples)}"
    y_true = np.array([int(s[1]) for s in samples])
    image_paths = [str(Path(s[0]).as_posix()) for s in samples]

    out_dir = REPO_ROOT / "outputs/clean_10class_effnet_b0_augmentation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for arm_key, arm_info in EXPERIMENTS.items():
        print(f"\n=======================================================")
        print(f"Evaluating {arm_info['display']}...")
        print(f"=======================================================")
        arm_results = {"seeds": {}, "predictions": {}}

        for seed, s_info in arm_info["seeds"].items():
            ckpt_p = REPO_ROOT / s_info["checkpoint"]
            cfg_p = REPO_ROOT / s_info["config"]
            if not ckpt_p.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_p}")
            
            ckpt_sha = compute_sha256(ckpt_p)
            cfg_sha = compute_sha256(cfg_p)
            
            with open(REPO_ROOT / s_info["output_dir"] / "val_evaluation_summary.json") as f:
                val_summary = json.load(f)
            with open(REPO_ROOT / s_info["output_dir"] / "training_history.json") as f:
                train_history = json.load(f)

            best_ep = val_summary["best_epoch"]
            best_hist = next(h for h in train_history if h["epoch"] == best_ep)

            # Load model and run FP32 inference
            cfg = yaml.safe_load(open(cfg_p))
            model = build_model(cfg).to(device)
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            probs_list = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    logits = model(imgs).float()
                    probs = torch.softmax(logits, dim=1)
                    probs_list.append(probs.cpu().numpy())

            p_arr = np.concatenate(probs_list, axis=0)
            assert p_arr.shape == (562, 10)
            preds = np.argmax(p_arr, axis=1)

            acc = float(accuracy_score(y_true, preds))
            bal_acc = float(balanced_accuracy_score(y_true, preds))
            macro_f1 = float(f1_score(y_true, preds, average="macro"))
            weighted_f1 = float(f1_score(y_true, preds, average="weighted"))
            kappa = float(cohen_kappa_score(y_true, preds))
            
            try:
                macro_auc = float(roc_auc_score(y_true, p_arr, multi_class="ovr", average="macro"))
            except Exception:
                macro_auc = float("nan")

            cm = confusion_matrix(y_true, preds, labels=list(range(10)))

            per_class_metrics = []
            for c_idx, c_name in enumerate(class_names):
                c_supp = int(np.sum(cm[c_idx, :]))
                c_tp = int(cm[c_idx, c_idx])
                c_fn = int(np.sum(cm[c_idx, :]) - c_tp)
                c_fp = int(np.sum(cm[:, c_idx]) - c_tp)
                c_tn = int(np.sum(cm) - c_tp - c_fn - c_fp)

                c_prec = float(precision_score(y_true == c_idx, preds == c_idx, zero_division=0))
                c_rec = float(recall_score(y_true == c_idx, preds == c_idx, zero_division=0))
                c_f1 = float(f1_score(y_true == c_idx, preds == c_idx, zero_division=0))

                per_class_metrics.append({
                    "class_index": c_idx,
                    "class_name": c_name,
                    "support": c_supp,
                    "tp": c_tp, "fp": c_fp, "fn": c_fn, "tn": c_tn,
                    "precision": c_prec,
                    "recall": c_rec,
                    "f1_score": c_f1,
                })

            # Save aligned per-image predictions
            per_image_recs = []
            for idx in range(len(samples)):
                p_lbl = int(preds[idx])
                t_lbl = int(y_true[idx])
                per_image_recs.append({
                    "sample_index": idx,
                    "image_path": image_paths[idx],
                    "true_label_idx": t_lbl,
                    "true_label_name": class_names[t_lbl],
                    "predicted_label_idx": p_lbl,
                    "predicted_label_name": class_names[p_lbl],
                    "is_correct": bool(p_lbl == t_lbl),
                    "confidence": round(float(p_arr[idx, p_lbl]), 6),
                    "true_class_score": round(float(p_arr[idx, t_lbl]), 6),
                })

            pred_save_p = out_dir / f"predictions_{arm_key}_seed{seed}.json"
            with open(pred_save_p, "w") as f:
                json.dump(per_image_recs, f, indent=2)

            arm_results["seeds"][seed] = {
                "config_sha256": cfg_sha,
                "checkpoint_sha256": ckpt_sha,
                "best_epoch": best_ep,
                "val_loss": float(best_hist["val_loss"]),
                "accuracy": acc,
                "correct_samples": int(np.sum(preds == y_true)),
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "weighted_f1": weighted_f1,
                "macro_auc": macro_auc,
                "kappa": kappa,
                "training_time_minutes": float(val_summary["training_time_minutes"]),
                "confusion_matrix": cm.tolist(),
                "per_class": per_class_metrics,
            }
            print(f"Seed {seed:4d}: Acc = {acc*100:.2f}% ({int(np.sum(preds == y_true))}/562) | Macro-F1 = {macro_f1*100:.2f}% | Val Loss = {best_hist['val_loss']:.4f} (Ep {best_ep})")

        # Aggregate across seeds
        headline_keys = ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "macro_auc", "kappa", "val_loss", "training_time_minutes"]
        agg_headline = {}
        for k in headline_keys:
            vals = [arm_results["seeds"][s][k] for s in [42, 1337, 2026]]
            agg_headline[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": {str(s): arm_results["seeds"][s][k] for s in [42, 1337, 2026]}
            }

        # Aggregate per class
        agg_per_class = []
        for c_idx, c_name in enumerate(class_names):
            supp = arm_results["seeds"][42]["per_class"][c_idx]["support"]
            recs = [arm_results["seeds"][s]["per_class"][c_idx]["recall"] for s in [42, 1337, 2026]]
            precs = [arm_results["seeds"][s]["per_class"][c_idx]["precision"] for s in [42, 1337, 2026]]
            f1s = [arm_results["seeds"][s]["per_class"][c_idx]["f1_score"] for s in [42, 1337, 2026]]
            agg_per_class.append({
                "class_index": c_idx,
                "class_name": c_name,
                "support": supp,
                "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs, ddof=1))},
                "precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs, ddof=1))},
                "f1_score": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s, ddof=1))},
            })

        arm_results["aggregate_headline"] = agg_headline
        arm_results["aggregate_per_class"] = agg_per_class
        all_results[arm_key] = arm_results

    # Comparative difference analysis
    diff_summary = {}
    for k in ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "val_loss"]:
        mean_a = all_results["arm_a_noaug"]["aggregate_headline"][k]["mean"]
        mean_b = all_results["arm_b_aug"]["aggregate_headline"][k]["mean"]
        delta = mean_b - mean_a
        diff_summary[k] = {
            "mean_noaug": mean_a,
            "mean_aug": mean_b,
            "delta": delta,
            "delta_pct_points": delta * 100.0 if k != "val_loss" else delta,
        }

    final_payload = {
        "dataset": "data_clean/val",
        "cohort_size": 562,
        "classes": class_names,
        "arms": all_results,
        "comparative_difference": diff_summary,
    }

    metrics_save_p = out_dir / "augmentation_audit_metrics.json"
    with open(metrics_save_p, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n✅ Successfully saved augmentation audit metrics to {metrics_save_p.name}")

if __name__ == "__main__":
    main()
