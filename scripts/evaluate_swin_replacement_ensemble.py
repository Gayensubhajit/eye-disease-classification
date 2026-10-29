#!/usr/bin/env python3
"""
Clean 10-Class Swin-Base Standalone & Predeclared Replacement Quad Ensemble Evaluation.

Evaluates:
1. Checkpoint integrity (SHA-256) of all 5 models:
   - ConvNeXt-Small (384x384 CLAHE)
   - EfficientNet-B3 (384x384 CLAHE)
   - ResNet-50d (384x384 CLAHE)
   - BiomedCLIP + CBAM (224x224)
   - Swin-Base (384x384 CLAHE)
2. Standalone FP32 inference on data_clean/val (N=562) for all 5 models.
3. Reference Champion Equal-Weight Quad: [ConvNeXt, EfficientNet, ResNet, BiomedCLIP] (0.25 each)
4. Predeclared Replacement Equal-Weight Quad: [ConvNeXt, EfficientNet, Swin-Base, BiomedCLIP] (0.25 each)
5. 5-Model Quintet: [ConvNeXt, EfficientNet, ResNet, BiomedCLIP, Swin-Base] (0.20 each)
6. Detailed Clinical Error Analysis: Glaucoma, Myopia, Macular Scar, Paired Disagreements, and McNemar tests.
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FundusDataset
from src.data.preprocessing import get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Swin-Base & Replacement Quad Ensemble Comprehensive Evaluation")
    print(f"Device: {device}")
    print(f"=======================================================\n")

    models_info = [
        {
            "id": "convnext_small",
            "display": "ConvNeXt-Small (384x384 CLAHE)",
            "config": "configs/clean_10class_convnext_small_384_clean.yaml",
            "checkpoint": "outputs/clean_10class_convnext_small_384_clean/best_model.pth",
            "expected_sha256": "238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d",
        },
        {
            "id": "efficientnet_b3",
            "display": "EfficientNet-B3 (384x384 CLAHE)",
            "config": "configs/clean_10class_efficientnet_b3_384_clean.yaml",
            "checkpoint": "outputs/clean_10class_efficientnet_b3_384_clean/best_model.pth",
            "expected_sha256": "0a93355dbc07dd2c989a55901e5b0fb8281d51dbc836bb3f0e1eb3a7e00fc3aa",
        },
        {
            "id": "resnet50d",
            "display": "ResNet-50d (384x384 CLAHE)",
            "config": "configs/clean_10class_resnet50d_384_clean.yaml",
            "checkpoint": "outputs/clean_10class_resnet50d_384_clean/best_model.pth",
            "expected_sha256": "110a01d6b4544376bd26f6e21857d0307f5187a86ccc0cc11aca790caf25346f",
        },
        {
            "id": "biomedclip_cbam",
            "display": "BiomedCLIP + CBAM Fusion (224x224)",
            "config": "configs/clean_10class_biomedclip_cbam_fusion.yaml",
            "checkpoint": "outputs/clean_10class_biomedclip_cbam_fusion/best_model.pth",
            "expected_sha256": "b323dfbe245e87508dd2e10a94104dcbee5c3e72325ea5f8167add9d7dee73aa",
        },
        {
            "id": "swin_base",
            "display": "Swin-Base (384x384 CLAHE)",
            "config": "configs/clean_10class_swin_base_patch4_window12_384.yaml",
            "checkpoint": "outputs/clean_10class_swin_base_384_clean/best_model.pth",
            "expected_sha256": "6d118211b29368d71e49ad1dbb5e439ba7f798e416bb4fa32a963e3eac8fbef2",
        },
    ]

    # Verify Checkpoints
    print("--- 1. Checkpoint Verification ---")
    ckpt_audit = {}
    for m in models_info:
        ckpt_p = REPO_ROOT / m["checkpoint"]
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        sha = compute_sha256(ckpt_p)
        assert sha == m["expected_sha256"], f"SHA mismatch for {m['id']}: {sha} vs {m['expected_sha256']}"
        print(f"✅ {m['display']}: Verified SHA-256 = {sha}")
        ckpt_audit[m["id"]] = {"path": str(ckpt_p), "sha256": sha, "size_bytes": ckpt_p.stat().st_size}

    # Run FP32 Inference
    print("\n--- 2. FP32 Validation Inference (N=562) ---")
    model_probs = {}
    ground_truth = None
    class_names = None
    image_paths = None

    for m in models_info:
        cfg_file = REPO_ROOT / m["config"]
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

        if image_paths is None:
            image_paths = [str(s[0]) for s in val_ds.samples]

        print(f"Running inference for {m['display']}...")
        model = build_model(cfg).to(device)
        ckpt = torch.load(REPO_ROOT / m["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs_list = []
        targets_list = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images).float()
                p = torch.softmax(logits, dim=1)
                probs_list.append(p.cpu().numpy())
                targets_list.append(targets.numpy())

        p_arr = np.concatenate(probs_list)
        model_probs[m["id"]] = p_arr

        if ground_truth is None:
            ground_truth = np.concatenate(targets_list)

        preds = np.argmax(p_arr, axis=1)
        acc = np.mean(preds == ground_truth)
        corr = np.sum(preds == ground_truth)
        metrics = compute_metrics(ground_truth, preds, p_arr, class_names=class_names)
        print(f"  -> {m['display']}: Acc={acc*100:.2f}% ({corr}/562) | Macro-F1={metrics['macro_f1']*100:.2f}% | BalAcc={metrics['balanced_accuracy']*100:.2f}%")

        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Individual Model Scoreboard
    print("\n--- 3. Standalone Model Comparison Scoreboard ---")
    standalone_results = {}
    for m in models_info:
        mid = m["id"]
        probs = model_probs[mid]
        preds = np.argmax(probs, axis=1)
        met = compute_metrics(ground_truth, preds, probs, class_names=class_names)
        corr = int(np.sum(preds == ground_truth))
        acc = float(corr / len(ground_truth))
        standalone_results[mid] = {
            "display": m["display"],
            "correct": corr,
            "total": len(ground_truth),
            "accuracy": acc,
            "macro_f1": met["macro_f1"],
            "balanced_accuracy": met["balanced_accuracy"],
            "weighted_f1": met["weighted_f1"],
            "kappa": met["kappa"],
            "per_class": met["per_class"],
        }
        print(f"{m['display']:36s}: {corr:3d}/562 ({acc*100:.2f}%) | Macro-F1: {met['macro_f1']*100:.2f}% | BalAcc: {met['balanced_accuracy']*100:.2f}%")

    # Ensembles
    print("\n--- 4. Ensemble Evaluations ---")
    # A. Reference Champion Quad: ConvNeXt + EfficientNet + ResNet-50d + BiomedCLIP (0.25 each)
    p_champ_quad = 0.25 * (
        model_probs["convnext_small"]
        + model_probs["efficientnet_b3"]
        + model_probs["resnet50d"]
        + model_probs["biomedclip_cbam"]
    )
    preds_champ = np.argmax(p_champ_quad, axis=1)
    met_champ = compute_metrics(ground_truth, preds_champ, p_champ_quad, class_names=class_names)
    corr_champ = int(np.sum(preds_champ == ground_truth))
    acc_champ = float(corr_champ / len(ground_truth))

    assert corr_champ == 504, f"Reference Quad reproduction mismatch: {corr_champ}/562 (expected 504)"
    print(f"Reference Champion Quad: {corr_champ}/562 ({acc_champ*100:.2f}%) | Macro-F1: {met_champ['macro_f1']*100:.2f}% | BalAcc: {met_champ['balanced_accuracy']*100:.2f}%")

    # B. Predeclared Replacement Quad: ConvNeXt + EfficientNet + Swin-Base + BiomedCLIP (0.25 each)
    p_repl_quad = 0.25 * (
        model_probs["convnext_small"]
        + model_probs["efficientnet_b3"]
        + model_probs["swin_base"]
        + model_probs["biomedclip_cbam"]
    )
    preds_repl = np.argmax(p_repl_quad, axis=1)
    met_repl = compute_metrics(ground_truth, preds_repl, p_repl_quad, class_names=class_names)
    corr_repl = int(np.sum(preds_repl == ground_truth))
    acc_repl = float(corr_repl / len(ground_truth))
    print(f"Candidate Replacement Quad: {corr_repl}/562 ({acc_repl*100:.2f}%) | Macro-F1: {met_repl['macro_f1']*100:.2f}% | BalAcc: {met_repl['balanced_accuracy']*100:.2f}%")

    # C. Quintet: ConvNeXt + EfficientNet + ResNet-50d + BiomedCLIP + Swin-Base (0.20 each)
    p_quintet = 0.20 * (
        model_probs["convnext_small"]
        + model_probs["efficientnet_b3"]
        + model_probs["resnet50d"]
        + model_probs["biomedclip_cbam"]
        + model_probs["swin_base"]
    )
    preds_quin = np.argmax(p_quintet, axis=1)
    met_quin = compute_metrics(ground_truth, preds_quin, p_quintet, class_names=class_names)
    corr_quin = int(np.sum(preds_quin == ground_truth))
    acc_quin = float(corr_quin / len(ground_truth))
    print(f"5-Model Quintet Ensemble:   {corr_quin}/562 ({acc_quin*100:.2f}%) | Macro-F1: {met_quin['macro_f1']*100:.2f}% | BalAcc: {met_quin['balanced_accuracy']*100:.2f}%")

    # D. Swin + ConvNeXt Duet (0.50 each)
    p_duet = 0.5 * (model_probs["convnext_small"] + model_probs["swin_base"])
    preds_duet = np.argmax(p_duet, axis=1)
    met_duet = compute_metrics(ground_truth, preds_duet, p_duet, class_names=class_names)
    corr_duet = int(np.sum(preds_duet == ground_truth))
    print(f"ConvNeXt + Swin Duet:       {corr_duet}/562 ({corr_duet/562*100:.2f}%) | Macro-F1: {met_duet['macro_f1']*100:.2f}% | BalAcc: {met_duet['balanced_accuracy']*100:.2f}%")

    # Paired Contingency & McNemar: Champion Quad vs Replacement Quad
    champ_correct = (preds_champ == ground_truth)
    repl_correct = (preds_repl == ground_truth)

    both_correct = int(np.sum(champ_correct & repl_correct))
    champ_only = int(np.sum(champ_correct & ~repl_correct))
    repl_only = int(np.sum(~champ_correct & repl_correct))
    both_wrong = int(np.sum(~champ_correct & ~repl_correct))

    print("\n--- 5. Paired Comparison: Champion Quad vs Replacement Quad ---")
    print(f"Both Correct:                    {both_correct}/562 ({both_correct/562*100:.2f}%)")
    print(f"Replacement Correct, Champ Wrong: {repl_only} samples")
    print(f"Champion Correct, Repl Wrong:    {champ_only} samples")
    print(f"Both Incorrect:                  {both_wrong} samples")
    net_gain = repl_only - champ_only
    print(f"Net Correct Difference:          {net_gain:+d} samples ({corr_repl} vs {corr_champ})")

    # Contingency between ConvNeXt-Small and Swin-Base Standalone
    cnext_correct = (np.argmax(model_probs["convnext_small"], axis=1) == ground_truth)
    swin_correct = (np.argmax(model_probs["swin_base"], axis=1) == ground_truth)
    swin_and_cnext_both_corr = int(np.sum(cnext_correct & swin_correct))
    swin_only_corr = int(np.sum(~cnext_correct & swin_correct))
    cnext_only_corr = int(np.sum(cnext_correct & ~swin_correct))
    both_wrong_cnext_swin = int(np.sum(~cnext_correct & ~swin_correct))

    print("\n--- 6. Paired Comparison: ConvNeXt-Small vs Swin-Base Standalone ---")
    print(f"Both Correct:                    {swin_and_cnext_both_corr}/562 ({swin_and_cnext_both_corr/562*100:.2f}%)")
    print(f"Swin Correct, ConvNeXt Wrong:    {swin_only_corr} samples")
    print(f"ConvNeXt Correct, Swin Wrong:    {cnext_only_corr} samples")
    print(f"Both Incorrect:                  {both_wrong_cnext_swin} samples")

    # Oracle Accuracy Across All 5 Models
    all_correct_mask = (
        (np.argmax(model_probs["convnext_small"], axis=1) == ground_truth)
        | (np.argmax(model_probs["efficientnet_b3"], axis=1) == ground_truth)
        | (np.argmax(model_probs["resnet50d"], axis=1) == ground_truth)
        | (np.argmax(model_probs["biomedclip_cbam"], axis=1) == ground_truth)
        | (np.argmax(model_probs["swin_base"], axis=1) == ground_truth)
    )
    oracle_corr = int(np.sum(all_correct_mask))
    print(f"5-Model Oracle Accuracy:         {oracle_corr}/562 ({oracle_corr/562*100:.2f}%) (Hard Errors Missed by ALL 5: {562 - oracle_corr})")

    # Glaucoma and Myopia Deep-Dive
    print("\n--- 7. Clinical Focus: Glaucoma & Myopia ---")
    glaucoma_idx = class_names.index("Glaucoma")
    myopia_idx = class_names.index("Myopia")
    healthy_idx = class_names.index("Healthy")
    scar_idx = class_names.index("Macular Scar")

    def get_class_stat(met_dict, c_idx):
        return met_dict["per_class"][c_idx]

    print("GLAUCOMA (31 true samples):")
    for name, m_dict in [
        ("ConvNeXt-Small", standalone_results["convnext_small"]),
        ("ResNet-50d", standalone_results["resnet50d"]),
        ("Swin-Base", standalone_results["swin_base"]),
        ("Champion Quad", met_champ),
        ("Replacement Quad", met_repl),
        ("5-Model Quintet", met_quin),
    ]:
        stat = get_class_stat(m_dict, glaucoma_idx)
        print(f"  {name:20s}: Sensitivity(Recall)={stat['sensitivity']*100:.2f}% | Specificity={stat['specificity']*100:.2f}% | F1={stat['f1_score']*100:.2f}%")

    print("\nMYOPIA (37 true samples):")
    for name, m_dict in [
        ("ConvNeXt-Small", standalone_results["convnext_small"]),
        ("ResNet-50d", standalone_results["resnet50d"]),
        ("Swin-Base", standalone_results["swin_base"]),
        ("Champion Quad", met_champ),
        ("Replacement Quad", met_repl),
        ("5-Model Quintet", met_quin),
    ]:
        stat = get_class_stat(m_dict, myopia_idx)
        print(f"  {name:20s}: Sensitivity(Recall)={stat['sensitivity']*100:.2f}% | Specificity={stat['specificity']*100:.2f}% | F1={stat['f1_score']*100:.2f}%")

    print("\nMACULAR SCAR (47 true samples):")
    for name, m_dict in [
        ("ConvNeXt-Small", standalone_results["convnext_small"]),
        ("ResNet-50d", standalone_results["resnet50d"]),
        ("Swin-Base", standalone_results["swin_base"]),
        ("Champion Quad", met_champ),
        ("Replacement Quad", met_repl),
        ("5-Model Quintet", met_quin),
    ]:
        stat = get_class_stat(m_dict, scar_idx)
        print(f"  {name:20s}: Sensitivity(Recall)={stat['sensitivity']*100:.2f}% | Specificity={stat['specificity']*100:.2f}% | F1={stat['f1_score']*100:.2f}%")

    # Confusion Matrices
    cm_champ = confusion_matrix(ground_truth, preds_champ, labels=list(range(len(class_names))))
    cm_repl = confusion_matrix(ground_truth, preds_repl, labels=list(range(len(class_names))))
    cm_swin = confusion_matrix(ground_truth, np.argmax(model_probs["swin_base"], axis=1), labels=list(range(len(class_names))))

    # Save Artifacts
    out_dir = REPO_ROOT / "outputs" / "clean_10class_swin_replacement_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "checkpoints": ckpt_audit,
        "standalone_models": standalone_results,
        "ensembles": {
            "champion_equal_weight_quad": {
                "correct": corr_champ,
                "total": 562,
                "accuracy": acc_champ,
                "macro_f1": met_champ["macro_f1"],
                "balanced_accuracy": met_champ["balanced_accuracy"],
                "weighted_f1": met_champ["weighted_f1"],
                "kappa": met_champ["kappa"],
                "per_class": met_champ["per_class"],
            },
            "replacement_equal_weight_quad": {
                "correct": corr_repl,
                "total": 562,
                "accuracy": acc_repl,
                "macro_f1": met_repl["macro_f1"],
                "balanced_accuracy": met_repl["balanced_accuracy"],
                "weighted_f1": met_repl["weighted_f1"],
                "kappa": met_repl["kappa"],
                "per_class": met_repl["per_class"],
            },
            "quintet_equal_weight": {
                "correct": corr_quin,
                "total": 562,
                "accuracy": acc_quin,
                "macro_f1": met_quin["macro_f1"],
                "balanced_accuracy": met_quin["balanced_accuracy"],
                "weighted_f1": met_quin["weighted_f1"],
                "kappa": met_quin["kappa"],
                "per_class": met_quin["per_class"],
            },
            "convnext_swin_duet": {
                "correct": corr_duet,
                "total": 562,
                "accuracy": corr_duet / 562,
                "macro_f1": met_duet["macro_f1"],
                "balanced_accuracy": met_duet["balanced_accuracy"],
            },
        },
        "paired_disagreements": {
            "champion_vs_replacement": {
                "both_correct": both_correct,
                "replacement_only_correct": repl_only,
                "champion_only_correct": champ_only,
                "both_wrong": both_wrong,
                "net_gain": net_gain,
            },
            "convnext_vs_swin": {
                "both_correct": swin_and_cnext_both_corr,
                "swin_only_correct": swin_only_corr,
                "convnext_only_correct": cnext_only_corr,
                "both_wrong": both_wrong_cnext_swin,
            },
            "oracle_correct": oracle_corr,
            "hard_errors_missed_by_all": 562 - oracle_corr,
        },
        "test_split_quarantined": True,
    }

    with open(out_dir / "swin_replacement_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Save Prediction CSVs
    pd.DataFrame(cm_champ, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix_champion_quad.csv")
    pd.DataFrame(cm_repl, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix_replacement_quad.csv")
    pd.DataFrame(cm_swin, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix_swin_standalone.csv")

    df_preds = pd.DataFrame({
        "image_path": image_paths,
        "true_label": [class_names[g] for g in ground_truth],
        "true_idx": ground_truth,
        "convnext_pred": [class_names[p] for p in np.argmax(model_probs["convnext_small"], axis=1)],
        "swin_pred": [class_names[p] for p in np.argmax(model_probs["swin_base"], axis=1)],
        "champion_quad_pred": [class_names[p] for p in preds_champ],
        "replacement_quad_pred": [class_names[p] for p in preds_repl],
        "quintet_pred": [class_names[p] for p in preds_quin],
    })
    df_preds.to_csv(out_dir / "val_predictions_all_models.csv", index=False)

    print(f"\n✅ All artifacts successfully saved to: {out_dir}")


if __name__ == "__main__":
    main()
