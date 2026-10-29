#!/usr/bin/env python3
"""
Milestone V7: Swin-Base Multi-Seed Stability Evaluation Script.
Evaluates Seeds 42, 1337, and 2026 on data_clean/val (N=562) under pure FP32.
Computes Mean +- SD across seeds, per-class recall stability, and confusion matrices.
Strictly quarantines data_clean/test.
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
from src.data.dataset import create_dataloaders
import yaml

SEEDS = {
    42: {
        "config": "configs/clean_10class_swin_base_patch4_window12_384.yaml",
        "checkpoint": "outputs/clean_10class_swin_base_384_clean/best_model.pth",
        "output_dir": "outputs/clean_10class_swin_base_384_clean",
        "expected_sha": "6d118211b29368d71e49ad1dbb5e439ba7f798e416bb4fa32a963e3eac8fbef2",
    },
    1337: {
        "config": "configs/clean_10class_swin_base_384_seed1337.yaml",
        "checkpoint": "outputs/clean_10class_swin_base_384_seed1337/best_model.pth",
        "output_dir": "outputs/clean_10class_swin_base_384_seed1337",
        "expected_sha": "38fc44a6c816f98490d21e5394caff6016abb43c71acd7e6db5dea728ddb5597",
    },
    2026: {
        "config": "configs/clean_10class_swin_base_384_seed2026.yaml",
        "checkpoint": "outputs/clean_10class_swin_base_384_seed2026/best_model.pth",
        "output_dir": "outputs/clean_10class_swin_base_384_seed2026",
        "expected_sha": "1b6c819e387a5b942ad5e832381f5e81e976760e173cf08b4aaf28a36b792b3e",
    },
}

def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Milestone V7: Swin-Base 3-Seed Reproducibility Audit ===")
    print(f"Device: {device}")

    # Checkpoint verification & config hashing
    print("\n--- 1. Checkpoint & Config Hash Verification ---")
    verified_info = {}
    for seed, info in SEEDS.items():
        ckpt_p = REPO_ROOT / info["checkpoint"]
        cfg_p = REPO_ROOT / info["config"]
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_p}")
        if not cfg_p.exists():
            raise FileNotFoundError(f"Missing config: {cfg_p}")
        
        actual_ckpt_sha = compute_sha256(ckpt_p)
        actual_cfg_sha = compute_sha256(cfg_p)
        assert actual_ckpt_sha == info["expected_sha"], f"SHA mismatch on seed {seed}: {actual_ckpt_sha} vs {info['expected_sha']}"
        print(f"Seed {seed:4d}:")
        print(f"  Config SHA-256:     {actual_cfg_sha}")
        print(f"  Checkpoint SHA-256: {actual_ckpt_sha} (VERIFIED)")
        
        with open(info["output_dir"] + "/val_evaluation_summary.json") as f:
            v_sum = json.load(f)
        with open(info["output_dir"] + "/training_history.json") as f:
            t_hist = json.load(f)

        best_ep = v_sum["best_epoch"]
        best_hist = next(h for h in t_hist if h["epoch"] == best_ep)
        verified_info[seed] = {
            "config_sha256": actual_cfg_sha,
            "checkpoint_sha256": actual_ckpt_sha,
            "best_epoch": best_ep,
            "val_loss": best_hist["val_loss"],
            "training_time_minutes": v_sum["training_time_minutes"],
        }

    # Load validation dataloader using seed 42 config (identical data splits)
    base_cfg_path = REPO_ROOT / SEEDS[42]["config"]
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    
    # Assert data_clean/test quarantine
    assert base_cfg.get("training", {}).get("evaluate_test") is False, "CRITICAL: evaluate_test must be false!"

    train_loader, val_loader, _ = create_dataloaders(base_cfg)
    class_names = base_cfg["data"]["class_names"]
    val_dataset = val_loader.dataset
    y_true = np.array([sample[1] for sample in val_dataset.samples])
    n_samples = len(y_true)
    assert n_samples == 562, f"Expected 562 val samples, found {n_samples}"
    print(f"\nLoaded validation dataset: {n_samples} samples across {len(class_names)} classes.")

    seed_metrics = {}
    seed_probs = {}
    seed_preds = {}
    seed_cms = {}

    for seed, info in SEEDS.items():
        print(f"\nEvaluating Seed {seed} Checkpoint...")
        cfg_p = REPO_ROOT / info["config"]
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model = build_model(cfg).to(device)
        ckpt = torch.load(REPO_ROOT / info["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs_list = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images).float()
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())

        p_arr = np.concatenate(probs_list, axis=0)
        assert p_arr.shape == (562, 10)
        preds = np.argmax(p_arr, axis=1)

        seed_probs[seed] = p_arr
        seed_preds[seed] = preds

        acc = float(accuracy_score(y_true, preds))
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        weighted_f1 = float(f1_score(y_true, preds, average="weighted"))
        kappa = float(cohen_kappa_score(y_true, preds))

        try:
            auc = float(roc_auc_score(y_true, p_arr, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_true, preds, labels=list(range(len(class_names))))
        seed_cms[seed] = cm

        # Per-class metrics
        per_class_list = []
        for i, cname in enumerate(class_names):
            tp = int(cm[i, i])
            fn = int(np.sum(cm[i, :]) - tp)
            fp = int(np.sum(cm[:, i]) - tp)
            tn = int(np.sum(cm) - tp - fn - fp)
            supp = int(np.sum(cm[i, :]))

            prec = float(precision_score(y_true == i, preds == i, zero_division=0))
            rec = float(recall_score(y_true == i, preds == i, zero_division=0))
            f1 = float(f1_score(y_true == i, preds == i, zero_division=0))
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            per_class_list.append({
                "class_index": i,
                "class_name": cname,
                "support": supp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1_score": f1,
            })

        seed_metrics[seed] = {
            "accuracy": acc,
            "correct_samples": int(np.sum(preds == y_true)),
            "total_samples": n_samples,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "weighted_f1": weighted_f1,
            "macro_auc": auc,
            "kappa": kappa,
            "best_epoch": verified_info[seed]["best_epoch"],
            "val_loss": verified_info[seed]["val_loss"],
            "training_time_minutes": verified_info[seed]["training_time_minutes"],
            "checkpoint_sha256": verified_info[seed]["checkpoint_sha256"],
            "config_sha256": verified_info[seed]["config_sha256"],
            "per_class": per_class_list,
        }

        print(f"  Accuracy:      {acc*100:.2f}% ({int(np.sum(preds == y_true))}/562)")
        print(f"  Macro-F1:      {macro_f1*100:.2f}%")
        print(f"  Balanced Acc:  {bal_acc*100:.2f}%")
        print(f"  Weighted F1:   {weighted_f1*100:.2f}%")
        print(f"  Macro AUC:     {auc:.4f}")
        print(f"  Val Loss:      {verified_info[seed]['val_loss']:.4f} (Epoch {verified_info[seed]['best_epoch']})")

    # Aggregate Statistics (Mean +- SD across N=3 seeds)
    headline_keys = ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "macro_auc", "kappa", "val_loss", "training_time_minutes"]
    aggregate_headline = {}
    for k in headline_keys:
        vals = [seed_metrics[s][k] for s in SEEDS.keys()]
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))  # sample standard deviation
        aggregate_headline[k] = {
            "mean": m,
            "std": sd,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": {str(s): seed_metrics[s][k] for s in SEEDS.keys()},
        }

    # Per-class aggregate statistics
    aggregate_per_class = []
    for i, cname in enumerate(class_names):
        supp = seed_metrics[42]["per_class"][i]["support"]
        recs = [seed_metrics[s]["per_class"][i]["recall"] for s in SEEDS.keys()]
        precs = [seed_metrics[s]["per_class"][i]["precision"] for s in SEEDS.keys()]
        f1s = [seed_metrics[s]["per_class"][i]["f1_score"] for s in SEEDS.keys()]

        aggregate_per_class.append({
            "class_index": i,
            "class_name": cname,
            "support": supp,
            "recall": {
                "mean": float(np.mean(recs)),
                "std": float(np.std(recs, ddof=1)),
                "min": float(np.min(recs)),
                "max": float(np.max(recs)),
                "by_seed": {str(s): recs[idx] for idx, s in enumerate(SEEDS.keys())},
            },
            "precision": {
                "mean": float(np.mean(precs)),
                "std": float(np.std(precs, ddof=1)),
                "by_seed": {str(s): precs[idx] for idx, s in enumerate(SEEDS.keys())},
            },
            "f1_score": {
                "mean": float(np.mean(f1s)),
                "std": float(np.std(f1s, ddof=1)),
                "by_seed": {str(s): f1s[idx] for idx, s in enumerate(SEEDS.keys())},
            },
        })

    # Output directory
    out_dir = REPO_ROOT / "outputs" / "clean_10class_swin_base_3seed_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_payload = {
        "dataset": "data_clean/val",
        "cohort_size": n_samples,
        "classes": class_names,
        "headline_aggregate": aggregate_headline,
        "per_class_aggregate": aggregate_per_class,
        "seed_metrics": seed_metrics,
        "confusion_matrices": {str(s): seed_cms[s].tolist() for s in SEEDS.keys()},
    }

    with open(out_dir / "three_seed_stability_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n✅ Successfully saved 3-seed stability metrics to {out_dir / 'three_seed_stability_metrics.json'}")

    # Print summary tables
    print("\n=========================================================================================================")
    print("                      SWIN-BASE 3-SEED STABILITY HEADLINE SUMMARY TABLE                                  ")
    print("=========================================================================================================")
    print(f"{'Metric':<22} | {'Seed 42':<10} | {'Seed 1337':<10} | {'Seed 2026':<10} | {'Mean +- SD (N=3)':<20}")
    print("-" * 85)
    for k in ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "macro_auc", "kappa", "val_loss"]:
        v42 = seed_metrics[42][k]
        v1337 = seed_metrics[1337][k]
        v2026 = seed_metrics[2026][k]
        m = aggregate_headline[k]["mean"]
        sd = aggregate_headline[k]["std"]
        if k in ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1"]:
            print(f"{k.replace('_', ' ').title():<22} | {v42*100:6.2f}%    | {v1337*100:6.2f}%    | {v2026*100:6.2f}%    | {m*100:6.2f}% +- {sd*100:4.2f}%")
        elif k == "val_loss":
            print(f"{'Val Loss':<22} | {v42:8.4f}   | {v1337:8.4f}   | {v2026:8.4f}   | {m:6.4f} +- {sd:6.4f}")
        else:
            print(f"{k.replace('_', ' ').title():<22} | {v42:8.4f}   | {v1337:8.4f}   | {v2026:8.4f}   | {m:6.4f} +- {sd:6.4f}")
    print("=" * 85)

    print("\n=========================================================================================================")
    print("                      PER-CLASS RECALL & STABILITY ACROSS SEEDS                                          ")
    print("=========================================================================================================")
    print(f"{'Class Name':<42} | {'Supp':<5} | {'S-42':<7} | {'S-1337':<7} | {'S-2026':<7} | {'Recall Mean +- SD':<20}")
    print("-" * 105)
    for item in aggregate_per_class:
        cname = item["class_name"]
        supp = item["support"]
        r42 = item["recall"]["by_seed"]["42"]
        r1337 = item["recall"]["by_seed"]["1337"]
        r2026 = item["recall"]["by_seed"]["2026"]
        rm = item["recall"]["mean"]
        rsd = item["recall"]["std"]
        print(f"{cname:<42} | {supp:<5} | {r42*100:5.1f}%  | {r1337*100:5.1f}%  | {r2026*100:5.1f}%  | {rm*100:5.1f}% +- {rsd*100:4.1f}%")
    print("=" * 105)

if __name__ == "__main__":
    main()
