#!/usr/bin/env python3
"""
No-Training Quad Ensemble Fusion Improvement Study.

This script rigorously evaluates two non-training fusion interventions for the clean 10-class quad ensemble:
1. Baseline Champion: Predeclared Equal-Weight Probability Averaging (504/562, 89.68%, 89.58% Macro-F1).
2. Experiment A — Rank Averaging: Equal-weight averaging of within-model class ranks (ties broken by raw probability).
3. Experiment B — Temperature Calibration:
   - Extracts exact FP32 raw logits from the 4 clean quad checkpoints (no guessing).
   - Fits scalar temperatures via 5-Fold Stratified Cross-Fitting (OOF, zero in-sample leakage).
   - Computes calibrated ensemble probabilities and evaluates out-of-fold predictions.
   - Compares with global (in-sample) temperatures and measures ECE and NLL reduction.
4. Paired Comparisons: McNemar test, rescued/degraded samples, per-class metrics, and Glaucoma specificity.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms
from src.models.backbone import build_model
from src.metrics import plot_confusion_matrix


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


def compute_nll(T: float, logits: np.ndarray, targets: np.ndarray) -> float:
    scaled = logits / max(float(T), 1e-4)
    max_s = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_s.squeeze(1) + np.log(np.sum(np.exp(scaled - max_s), axis=1))
    target_logits = scaled[np.arange(len(targets)), targets]
    return float(-np.mean(target_logits - log_sum_exp))


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = REPO_ROOT / "outputs/clean_10class_quad_fusion_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("Clean 10-Class Quad Ensemble Fusion Improvement Study")
    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")
    print("=======================================================\n")

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

    # Verify Checkpoints
    for b in backbones:
        ckpt_path = REPO_ROOT / b["checkpoint"]
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        sha = compute_sha256(ckpt_path)
        assert sha == b["expected_sha256"], f"Hash mismatch for {b[name]}: {sha} vs {b[expected_sha256]}"
        print(f"Verified {b['display']} [SHA-256: {sha[:16]}...]")

    # Extract Exact FP32 Raw Logits & Probs
    logits_dict = {}
    probs_dict = {}
    y_true = None
    class_names = None
    val_samples = None

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
        if val_samples is None:
            val_samples = val_ds.samples

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"].get("batch_size", 4),
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 2),
            pin_memory=True,
        )

        model = build_model(cfg).to(device)
        ckpt = torch.load(REPO_ROOT / b["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        b_logits = []
        b_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                l = model(images).float()
                b_logits.append(l.cpu().numpy())
                b_targets.append(targets.numpy())

        logits_arr = np.concatenate(b_logits)
        targets_arr = np.concatenate(b_targets)
        if y_true is None:
            y_true = targets_arr
        else:
            assert np.array_equal(y_true, targets_arr), "Target mismatch across models!"

        p_arr = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()
        logits_dict[b["name"]] = logits_arr
        probs_dict[b["name"]] = p_arr
        print(f"Computed exact logits and probabilities for {b['display']}")

    n_samples = len(y_true)
    num_classes = len(class_names)
    assert n_samples == 562, f"Expected 562 validation samples, got {n_samples}"

    # Save exact raw logits and probabilities
    npz_path = output_dir / "quad_backbone_logits_and_probs.npz"
    np.savez_compressed(
        npz_path,
        y_true=y_true,
        class_names=class_names,
        **{f"logits_{k}": v for k, v in logits_dict.items()},
        **{f"probs_{k}": v for k, v in probs_dict.items()}
    )
    print(f"\nSaved raw logits and probabilities to: {npz_path}")

    # --- 1. BASELINE CHAMPION: Equal-Weight Probability Averaging ---
    prob_list = [probs_dict[b["name"]] for b in backbones]
    mean_probs_base = np.mean(prob_list, axis=0)
    preds_base = np.argmax(mean_probs_base, axis=1)

    corr_base = int(np.sum(preds_base == y_true))
    acc_base = float(corr_base / n_samples)
    macro_f1_base = float(f1_score(y_true, preds_base, average="macro", zero_division=0))
    bal_acc_base = float(balanced_accuracy_score(y_true, preds_base))
    macro_prec_base = float(precision_score(y_true, preds_base, average="macro", zero_division=0))
    per_class_f1_base = f1_score(y_true, preds_base, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_rec_base = recall_score(y_true, preds_base, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_prec_base = precision_score(y_true, preds_base, average=None, labels=list(range(num_classes)), zero_division=0)

    cm_base = confusion_matrix(y_true, preds_base, labels=list(range(num_classes)))
    glaucoma_idx = class_names.index("Glaucoma")
    healthy_idx = class_names.index("Healthy")
    myopia_idx = class_names.index("Myopia")

    assert corr_base == 504, f"Expected baseline quad to have 504 correct, got {corr_base}!"
    print(f"\n[Baseline Champion Equal-Weight Quad] Correct: {corr_base}/{n_samples} ({acc_base:.4%}) | Macro-F1: {macro_f1_base:.4%}")

    # --- 2. EXPERIMENT A: Rank Averaging ---
    # Convert each model probability row into class ranks (1 = lowest prob, 10 = highest prob)
    ranks_list = []
    for p_mat in prob_list:
        r_mat = np.zeros_like(p_mat)
        for i in range(n_samples):
            r_mat[i] = rankdata(p_mat[i], method="average")
        ranks_list.append(r_mat)

    mean_ranks = np.mean(ranks_list, axis=0)
    preds_rank = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        max_r = np.max(mean_ranks[i])
        top_c = np.where(mean_ranks[i] == max_r)[0]
        if len(top_c) == 1:
            preds_rank[i] = top_c[0]
        else:
            # Tie-break by higher raw ensemble probability
            best_c = top_c[np.argmax(mean_probs_base[i, top_c])]
            preds_rank[i] = best_c

    corr_rank = int(np.sum(preds_rank == y_true))
    acc_rank = float(corr_rank / n_samples)
    macro_f1_rank = float(f1_score(y_true, preds_rank, average="macro", zero_division=0))
    bal_acc_rank = float(balanced_accuracy_score(y_true, preds_rank))
    macro_prec_rank = float(precision_score(y_true, preds_rank, average="macro", zero_division=0))
    per_class_f1_rank = f1_score(y_true, preds_rank, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_rec_rank = recall_score(y_true, preds_rank, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_prec_rank = precision_score(y_true, preds_rank, average=None, labels=list(range(num_classes)), zero_division=0)
    cm_rank = confusion_matrix(y_true, preds_rank, labels=list(range(num_classes)))

    # Paired McNemar against Baseline
    b_rank_rescued = int(np.sum((preds_rank == y_true) & (preds_base != y_true)))
    c_rank_degraded = int(np.sum((preds_rank != y_true) & (preds_base == y_true)))
    mcnemar_rank = mcnemar_test(b_rank_rescued, c_rank_degraded)

    rescued_rank_samples = []
    degraded_rank_samples = []
    for i in range(n_samples):
        img_p, lbl = val_samples[i]
        rel_p = os.path.relpath(img_p, REPO_ROOT)
        if preds_rank[i] == y_true[i] and preds_base[i] != y_true[i]:
            rescued_rank_samples.append({
                "image_index": i,
                "path": rel_p,
                "true_class": class_names[lbl],
                "base_pred": class_names[preds_base[i]],
                "rank_pred": class_names[preds_rank[i]],
            })
        elif preds_rank[i] != y_true[i] and preds_base[i] == y_true[i]:
            degraded_rank_samples.append({
                "image_index": i,
                "path": rel_p,
                "true_class": class_names[lbl],
                "base_pred": class_names[preds_base[i]],
                "rank_pred": class_names[preds_rank[i]],
            })

    # Save Rank Predictions CSV
    rank_rows = []
    for i in range(n_samples):
        img_p, lbl = val_samples[i]
        rel_p = os.path.relpath(img_p, REPO_ROOT)
        row = {
            "image_index": i,
            "image_path": rel_p,
            "filename": os.path.basename(img_p),
            "true_label": int(lbl),
            "true_class": class_names[lbl],
            "pred_label": int(preds_rank[i]),
            "pred_class": class_names[preds_rank[i]],
            "correct": bool(preds_rank[i] == lbl),
            "max_mean_rank": float(np.max(mean_ranks[i])),
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"mean_rank_{c_name}"] = float(mean_ranks[i, c_idx])
            row[f"prob_{c_name}"] = float(mean_probs_base[i, c_idx])
        rank_rows.append(row)
    pd.DataFrame(rank_rows).to_csv(output_dir / "val_predictions_rank_averaging.csv", index=False)
    pd.DataFrame(cm_rank, index=class_names, columns=class_names).to_csv(output_dir / "val_confusion_matrix_rank_averaging.csv")
    plot_confusion_matrix(y_true, preds_rank, class_names, str(output_dir / "val_confusion_matrix_rank_averaging.png"), title="Rank Averaged Quad Confusion Matrix")

    print(f"[Experiment A — Rank Averaging] Correct: {corr_rank}/{n_samples} ({acc_rank:.4%}) | Macro-F1: {macro_f1_rank:.4%}")
    print(f"  Delta: {acc_rank - acc_base:+.2%} ({corr_rank - corr_base:+d} samples) | Rescued: {b_rank_rescued}, Degraded: {c_rank_degraded}")

    # --- 3. EXPERIMENT B: Temperature Calibration ---
    # Global (In-Sample) Calibration for comparison
    global_temperatures = {}
    pre_calib_ece = {}
    post_calib_global_ece = {}
    pre_calib_nll = {}
    post_calib_global_nll = {}

    for b in backbones:
        b_name = b["name"]
        l_mat = logits_dict[b_name]
        p_mat = probs_dict[b_name]

        nll_orig = compute_nll(1.0, l_mat, y_true)
        ece_orig = compute_ece(p_mat, y_true)

        res = minimize_scalar(compute_nll, bounds=(0.05, 10.0), method="bounded", args=(l_mat, y_true))
        t_opt = float(res.x)
        global_temperatures[b_name] = t_opt

        p_calib_global = torch.softmax(torch.tensor(l_mat / t_opt), dim=1).numpy()
        nll_opt = compute_nll(t_opt, l_mat, y_true)
        ece_opt = compute_ece(p_calib_global, y_true)

        pre_calib_nll[b_name] = nll_orig
        post_calib_global_nll[b_name] = nll_opt
        pre_calib_ece[b_name] = ece_orig
        post_calib_global_ece[b_name] = ece_opt

    # 5-Fold Stratified Cross-Validation (Out-of-Fold Calibration)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_temperatures = {b["name"]: [] for b in backbones}
    oof_calibrated_probs = {b["name"]: np.zeros((n_samples, num_classes)) for b in backbones}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y_true)):
        y_train = y_true[train_idx]
        y_val = y_true[val_idx]

        for b in backbones:
            b_name = b["name"]
            l_train = logits_dict[b_name][train_idx]
            l_val = logits_dict[b_name][val_idx]

            res_fold = minimize_scalar(compute_nll, bounds=(0.05, 10.0), method="bounded", args=(l_train, y_train))
            t_fold = float(res_fold.x)
            fold_temperatures[b_name].append(t_fold)

            # Apply fold-fitted temperature strictly to held-out validation samples
            p_val_calib = torch.softmax(torch.tensor(l_val / t_fold), dim=1).numpy()
            oof_calibrated_probs[b_name][val_idx] = p_val_calib

    # Out-of-fold calibrated probabilities across the 4 models
    oof_prob_list = [oof_calibrated_probs[b["name"]] for b in backbones]
    mean_probs_calib_oof = np.mean(oof_prob_list, axis=0)
    preds_calib_oof = np.argmax(mean_probs_calib_oof, axis=1)

    corr_calib = int(np.sum(preds_calib_oof == y_true))
    acc_calib = float(corr_calib / n_samples)
    macro_f1_calib = float(f1_score(y_true, preds_calib_oof, average="macro", zero_division=0))
    bal_acc_calib = float(balanced_accuracy_score(y_true, preds_calib_oof))
    macro_prec_calib = float(precision_score(y_true, preds_calib_oof, average="macro", zero_division=0))
    per_class_f1_calib = f1_score(y_true, preds_calib_oof, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_rec_calib = recall_score(y_true, preds_calib_oof, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_prec_calib = precision_score(y_true, preds_calib_oof, average=None, labels=list(range(num_classes)), zero_division=0)
    cm_calib = confusion_matrix(y_true, preds_calib_oof, labels=list(range(num_classes)))

    # Paired McNemar against Baseline
    b_calib_rescued = int(np.sum((preds_calib_oof == y_true) & (preds_base != y_true)))
    c_calib_degraded = int(np.sum((preds_calib_oof != y_true) & (preds_base == y_true)))
    mcnemar_calib = mcnemar_test(b_calib_rescued, c_calib_degraded)

    rescued_calib_samples = []
    degraded_calib_samples = []
    for i in range(n_samples):
        img_p, lbl = val_samples[i]
        rel_p = os.path.relpath(img_p, REPO_ROOT)
        if preds_calib_oof[i] == y_true[i] and preds_base[i] != y_true[i]:
            rescued_calib_samples.append({
                "image_index": i,
                "path": rel_p,
                "true_class": class_names[lbl],
                "base_pred": class_names[preds_base[i]],
                "calib_pred": class_names[preds_calib_oof[i]],
            })
        elif preds_calib_oof[i] != y_true[i] and preds_base[i] == y_true[i]:
            degraded_calib_samples.append({
                "image_index": i,
                "path": rel_p,
                "true_class": class_names[lbl],
                "base_pred": class_names[preds_base[i]],
                "calib_pred": class_names[preds_calib_oof[i]],
            })

    # Save Calibrated Predictions CSV
    calib_rows = []
    for i in range(n_samples):
        img_p, lbl = val_samples[i]
        rel_p = os.path.relpath(img_p, REPO_ROOT)
        row = {
            "image_index": i,
            "image_path": rel_p,
            "filename": os.path.basename(img_p),
            "true_label": int(lbl),
            "true_class": class_names[lbl],
            "pred_label": int(preds_calib_oof[i]),
            "pred_class": class_names[preds_calib_oof[i]],
            "correct": bool(preds_calib_oof[i] == lbl),
            "max_probability": float(np.max(mean_probs_calib_oof[i])),
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(mean_probs_calib_oof[i, c_idx])
        calib_rows.append(row)
    pd.DataFrame(calib_rows).to_csv(output_dir / "val_predictions_temperature_calibrated_oof.csv", index=False)
    pd.DataFrame(cm_calib, index=class_names, columns=class_names).to_csv(output_dir / "val_confusion_matrix_temperature_calibrated_oof.csv")
    plot_confusion_matrix(y_true, preds_calib_oof, class_names, str(output_dir / "val_confusion_matrix_temperature_calibrated_oof.png"), title="Temperature-Calibrated Quad (OOF) Confusion Matrix")

    print(f"[Experiment B — Temperature Calibration (OOF)] Correct: {corr_calib}/{n_samples} ({acc_calib:.4%}) | Macro-F1: {macro_f1_calib:.4%}")
    print(f"  Delta: {acc_calib - acc_base:+.2%} ({corr_calib - corr_base:+d} samples) | Rescued: {b_calib_rescued}, Degraded: {c_calib_degraded}")

    # Build Comprehensive Comparison JSON
    per_class_breakdown = []
    for idx, c_name in enumerate(class_names):
        supp = int(np.sum(y_true == idx))
        per_class_breakdown.append({
            "class_name": c_name,
            "support": supp,
            "baseline_f1": float(per_class_f1_base[idx]),
            "baseline_recall": float(per_class_rec_base[idx]),
            "baseline_precision": float(per_class_prec_base[idx]),
            "rank_f1": float(per_class_f1_rank[idx]),
            "rank_recall": float(per_class_rec_rank[idx]),
            "rank_precision": float(per_class_prec_rank[idx]),
            "rank_f1_delta_pp": float((per_class_f1_rank[idx] - per_class_f1_base[idx]) * 100),
            "calib_f1": float(per_class_f1_calib[idx]),
            "calib_recall": float(per_class_rec_calib[idx]),
            "calib_precision": float(per_class_prec_calib[idx]),
            "calib_f1_delta_pp": float((per_class_f1_calib[idx] - per_class_f1_base[idx]) * 100),
        })

    temperature_summary = {}
    for b in backbones:
        b_name = b["name"]
        f_temps = fold_temperatures[b_name]
        temperature_summary[b_name] = {
            "display": b["display"],
            "global_in_sample_T": global_temperatures[b_name],
            "oof_5fold_mean_T": float(np.mean(f_temps)),
            "oof_5fold_std_T": float(np.std(f_temps)),
            "oof_5fold_values": [float(x) for x in f_temps],
            "pre_calib_nll": pre_calib_nll[b_name],
            "post_calib_global_nll": post_calib_global_nll[b_name],
            "pre_calib_ece": pre_calib_ece[b_name],
            "post_calib_global_ece": post_calib_global_ece[b_name],
        }

    comparison_report = {
        "evaluation_split": "data_clean/val",
        "n_samples": n_samples,
        "class_names": class_names,
        "baseline_champion": {
            "name": "Equal-Weight Probability Averaging (Baseline Quad)",
            "accuracy": acc_base,
            "correct_samples": corr_base,
            "macro_f1": macro_f1_base,
            "balanced_accuracy": bal_acc_base,
            "macro_precision": macro_prec_base,
            "glaucoma_f1": float(per_class_f1_base[glaucoma_idx]),
            "glaucoma_recall": float(per_class_rec_base[glaucoma_idx]),
            "glaucoma_precision": float(per_class_prec_base[glaucoma_idx]),
            "confusions": {
                "glaucoma_as_healthy": int(cm_base[glaucoma_idx, healthy_idx]),
                "healthy_as_glaucoma": int(cm_base[healthy_idx, glaucoma_idx]),
                "glaucoma_as_myopia": int(cm_base[glaucoma_idx, myopia_idx]),
                "myopia_as_glaucoma": int(cm_base[myopia_idx, glaucoma_idx]),
            }
        },
        "experiment_A_rank_averaging": {
            "name": "Equal-Weight Rank Averaging",
            "accuracy": acc_rank,
            "correct_samples": corr_rank,
            "macro_f1": macro_f1_rank,
            "balanced_accuracy": bal_acc_rank,
            "macro_precision": macro_prec_rank,
            "glaucoma_f1": float(per_class_f1_rank[glaucoma_idx]),
            "glaucoma_recall": float(per_class_rec_rank[glaucoma_idx]),
            "glaucoma_precision": float(per_class_prec_rank[glaucoma_idx]),
            "confusions": {
                "glaucoma_as_healthy": int(cm_rank[glaucoma_idx, healthy_idx]),
                "healthy_as_glaucoma": int(cm_rank[healthy_idx, glaucoma_idx]),
                "glaucoma_as_myopia": int(cm_rank[glaucoma_idx, myopia_idx]),
                "myopia_as_glaucoma": int(cm_rank[myopia_idx, glaucoma_idx]),
            },
            "delta_vs_baseline": {
                "accuracy_pp": float((acc_rank - acc_base) * 100),
                "net_correct": int(corr_rank - corr_base),
                "macro_f1_pp": float((macro_f1_rank - macro_f1_base) * 100),
                "glaucoma_f1_pp": float((per_class_f1_rank[glaucoma_idx] - per_class_f1_base[glaucoma_idx]) * 100),
                "glaucoma_recall_pp": float((per_class_rec_rank[glaucoma_idx] - per_class_rec_base[glaucoma_idx]) * 100),
            },
            "mcnemar": mcnemar_rank,
            "rescued_count": b_rank_rescued,
            "degraded_count": c_rank_degraded,
            "rescued_samples": rescued_rank_samples,
            "degraded_samples": degraded_rank_samples,
        },
        "experiment_B_temperature_calibration": {
            "name": "5-Fold Cross-Calibrated Probability Averaging (OOF)",
            "accuracy": acc_calib,
            "correct_samples": corr_calib,
            "macro_f1": macro_f1_calib,
            "balanced_accuracy": bal_acc_calib,
            "macro_precision": macro_prec_calib,
            "glaucoma_f1": float(per_class_f1_calib[glaucoma_idx]),
            "glaucoma_recall": float(per_class_rec_calib[glaucoma_idx]),
            "glaucoma_precision": float(per_class_prec_calib[glaucoma_idx]),
            "confusions": {
                "glaucoma_as_healthy": int(cm_calib[glaucoma_idx, healthy_idx]),
                "healthy_as_glaucoma": int(cm_calib[healthy_idx, glaucoma_idx]),
                "glaucoma_as_myopia": int(cm_calib[glaucoma_idx, myopia_idx]),
                "myopia_as_glaucoma": int(cm_calib[myopia_idx, glaucoma_idx]),
            },
            "delta_vs_baseline": {
                "accuracy_pp": float((acc_calib - acc_base) * 100),
                "net_correct": int(corr_calib - corr_base),
                "macro_f1_pp": float((macro_f1_calib - macro_f1_base) * 100),
                "glaucoma_f1_pp": float((per_class_f1_calib[glaucoma_idx] - per_class_f1_base[glaucoma_idx]) * 100),
                "glaucoma_recall_pp": float((per_class_rec_calib[glaucoma_idx] - per_class_rec_base[glaucoma_idx]) * 100),
            },
            "mcnemar": mcnemar_calib,
            "rescued_count": b_calib_rescued,
            "degraded_count": c_calib_degraded,
            "rescued_samples": rescued_calib_samples,
            "degraded_samples": degraded_calib_samples,
            "temperatures_by_model": temperature_summary,
        },
        "per_class_breakdown": per_class_breakdown,
    }

    report_json_path = output_dir / "quad_fusion_study_comparison.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    print(f"\nSaved comparison report to: {report_json_path}")


if __name__ == "__main__":
    main()
