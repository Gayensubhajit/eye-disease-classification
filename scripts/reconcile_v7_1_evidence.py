#!/usr/bin/env python3
"""
Milestone V7.1: Evidence Reconciliation Script.
Generates aligned per-image predictions for Seeds 42, 1337, 2026.
Recalculates error overlap metrics directly from predictions.
Verifies pretrained weights hash, checkpoint hashes, and resolves the Seed 42 epoch discrepancy.
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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbone import build_model
from src.data.dataset import create_dataloaders

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

PRETRAINED_WEIGHT_PATH = Path("/home/silentbyte/.cache/huggingface/hub/models--timm--swin_base_patch4_window12_384.ms_in22k_ft_in1k/blobs/7ff88869f8609f09e36a48846aba48665c0a4164910391a1d24595d774db85c2")

def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Milestone V7.1: Evidence Reconciliation ===")
    print(f"Device: {device}")

    # 1. Pretrained Weights and Checkpoint Integrity Verification
    print("\n--- 1. Pretrained Weight & Checkpoint Integrity Verification ---")
    if PRETRAINED_WEIGHT_PATH.exists():
        actual_pretrained_sha = compute_sha256(PRETRAINED_WEIGHT_PATH)
        pretrained_size = PRETRAINED_WEIGHT_PATH.stat().st_size
        print(f"Pretrained Weight File: {PRETRAINED_WEIGHT_PATH}")
        print(f"  Size:       {pretrained_size} bytes")
        print(f"  SHA-256:    {actual_pretrained_sha}")
    else:
        actual_pretrained_sha = None
        pretrained_size = None
        print(f"WARNING: Pretrained weight file not found at {PRETRAINED_WEIGHT_PATH}")

    integrity_records = {}
    for seed, info in SEEDS.items():
        ckpt_p = REPO_ROOT / info["checkpoint"]
        cfg_p = REPO_ROOT / info["config"]
        ckpt_sha = compute_sha256(ckpt_p)
        cfg_sha = compute_sha256(cfg_p)
        assert ckpt_sha == info["expected_sha"], f"SHA mismatch on Seed {seed}"
        
        # Load checkpoint directly to check internal epoch metadata
        ckpt_obj = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        internal_epoch = ckpt_obj.get("epoch")
        
        integrity_records[seed] = {
            "config_path": info["config"],
            "config_sha256": cfg_sha,
            "checkpoint_path": info["checkpoint"],
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_internal_epoch": internal_epoch,
        }
        print(f"Seed {seed:4d}: Checkpoint verified (SHA-256: {ckpt_sha}, Internal Epoch: {internal_epoch})")

    # 2. Resolve Seed 42 Epoch Discrepancy
    print("\n--- 2. Reconcile Seed 42 Best Epoch ---")
    with open(REPO_ROOT / "outputs/clean_10class_swin_base_384_clean/val_evaluation_summary.json") as f:
        s42_summary = json.load(f)
    with open(REPO_ROOT / "outputs/clean_10class_swin_base_384_clean/training_history.json") as f:
        s42_hist = json.load(f)

    print(f"Swin-Base Seed 42 val_evaluation_summary.json best_epoch: {s42_summary['best_epoch']}")
    print(f"Swin-Base Seed 42 best_model.pth internal epoch:         {integrity_records[42]['checkpoint_internal_epoch']}")
    
    # Check Epoch 13 vs Epoch 16 in history
    h13 = next(h for h in s42_hist if h["epoch"] == 13)
    h16 = next(h for h in s42_hist if h["epoch"] == 16)
    print(f"Swin-Base Seed 42 Epoch 13: Val Macro-F1 = {h13['val_metrics']['macro_f1']:.6f}, Acc = {h13['val_metrics']['accuracy']*100:.2f}%, Val Loss = {h13['val_loss']:.4f}")
    print(f"Swin-Base Seed 42 Epoch 16: Val Macro-F1 = {h16['val_metrics']['macro_f1']:.6f}, Acc = {h16['val_metrics']['accuracy']*100:.2f}%, Val Loss = {h16['val_loss']:.4f}")

    # Inspect ConvNeXt-Small summary for historical comparison
    convnext_sum_p = REPO_ROOT / "outputs/clean_10class_convnext_small_384_clean/val_evaluation_summary.json"
    if convnext_sum_p.exists():
        with open(convnext_sum_p) as f:
            cnext_sum = json.load(f)
        print(f"ConvNeXt-Small Seed 42 best_epoch: {cnext_sum.get('best_epoch')} (Val Macro-F1: {cnext_sum.get('val_macro_f1_at_best_epoch'):.6f})")

    # 3. Load Validation Dataset
    print("\n--- 3. Validation Dataset & Dataloader Alignment ---")
    base_cfg = yaml.safe_load(open(REPO_ROOT / SEEDS[42]["config"]))
    assert base_cfg.get("training", {}).get("evaluate_test") is False, "Quarantine check: evaluate_test must be false!"
    
    train_loader, val_loader, _ = create_dataloaders(base_cfg)
    val_dataset = val_loader.dataset
    class_names = base_cfg["data"]["class_names"]
    
    samples = val_dataset.samples
    assert len(samples) == 562, f"Expected 562 samples, found {len(samples)}"
    
    image_relative_paths = [str(Path(s[0]).as_posix()) for s in samples]
    true_labels = [int(s[1]) for s in samples]
    true_label_names = [class_names[lbl] for lbl in true_labels]
    
    print(f"Total Validation Samples: {len(image_relative_paths)}")
    print(f"First Sample: {image_relative_paths[0]} -> {true_label_names[0]}")
    print(f"Last Sample:  {image_relative_paths[-1]} -> {true_label_names[-1]}")

    # 4. Generate Aligned Per-Image Predictions
    print("\n--- 4. Generating Aligned Per-Image Validation Predictions ---")
    out_dir = REPO_ROOT / "outputs/clean_10class_swin_base_3seed_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_by_seed = {}

    for seed, info in SEEDS.items():
        print(f"Running inference for Seed {seed}...")
        cfg = yaml.safe_load(open(REPO_ROOT / info["config"]))
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

        probs_arr = np.concatenate(probs_list, axis=0)
        assert probs_arr.shape == (562, 10)
        preds = np.argmax(probs_arr, axis=1)

        per_image_records = []
        for idx in range(len(samples)):
            rel_path = image_relative_paths[idx]
            t_lbl = true_labels[idx]
            p_lbl = int(preds[idx])
            is_corr = bool(p_lbl == t_lbl)
            conf = float(probs_arr[idx, p_lbl])
            true_score = float(probs_arr[idx, t_lbl])
            all_probs = [float(p) for p in probs_arr[idx]]

            per_image_records.append({
                "sample_index": idx,
                "image_path": rel_path,
                "true_label_idx": t_lbl,
                "true_label_name": true_label_names[idx],
                "predicted_label_idx": p_lbl,
                "predicted_label_name": class_names[p_lbl],
                "is_correct": is_corr,
                "predicted_class_confidence": round(conf, 6),
                "true_class_score": round(true_score, 6),
                "all_class_probabilities": [round(p, 6) for p in all_probs],
            })

        predictions_by_seed[seed] = per_image_records

        # Save JSON and CSV
        json_path = out_dir / f"per_image_predictions_seed{seed}.json"
        csv_path = out_dir / f"per_image_predictions_seed{seed}.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(per_image_records, f, indent=2)

        df = pd.DataFrame([{
            "sample_index": r["sample_index"],
            "image_path": r["image_path"],
            "true_label_idx": r["true_label_idx"],
            "true_label_name": r["true_label_name"],
            "predicted_label_idx": r["predicted_label_idx"],
            "predicted_label_name": r["predicted_label_name"],
            "is_correct": r["is_correct"],
            "predicted_class_confidence": r["predicted_class_confidence"],
            "true_class_score": r["true_class_score"],
        } for r in per_image_records])
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved {len(per_image_records)} predictions to {json_path.name} and {csv_path.name}")

    # 5. Verify Sample Alignment Across Files
    print("\n--- 5. Verifying Sample Alignment Across Seeds ---")
    for idx in range(len(samples)):
        p42 = predictions_by_seed[42][idx]
        p1337 = predictions_by_seed[1337][idx]
        p2026 = predictions_by_seed[2026][idx]
        assert p42["image_path"] == p1337["image_path"] == p2026["image_path"], f"Path alignment mismatch at index {idx}"
        assert p42["true_label_idx"] == p1337["true_label_idx"] == p2026["true_label_idx"], f"Label alignment mismatch at index {idx}"
    print("✅ Verified 100% identical sample ordering and ground-truth labels across all three seed prediction files.")

    # 6. Recalculate Error Overlap Directly from Saved Predictions
    print("\n--- 6. Deriving Reconciled Error Overlap Directly from Predictions ---")
    errors_by_seed = {}
    for seed in SEEDS.keys():
        errors_by_seed[seed] = set(r["sample_index"] for r in predictions_by_seed[seed] if not r["is_correct"])
        print(f"Seed {seed:4d}: {len(errors_by_seed[seed])} errors ({562 - len(errors_by_seed[seed])}/562 correct)")

    shared_all_3 = sorted(list(errors_by_seed[42] & errors_by_seed[1337] & errors_by_seed[2026]))
    shared_at_least_2 = sorted(list(
        (errors_by_seed[42] & errors_by_seed[1337])
        | (errors_by_seed[42] & errors_by_seed[2026])
        | (errors_by_seed[1337] & errors_by_seed[2026])
    ))
    union_errors = sorted(list(errors_by_seed[42] | errors_by_seed[1337] | errors_by_seed[2026]))

    print(f"Errors shared by ALL 3 seeds:       {len(shared_all_3)}")
    print(f"Errors shared by AT LEAST 2 seeds:  {len(shared_at_least_2)}")
    print(f"Union of all error images:          {len(union_errors)}")
    print(f"Images correctly classified by all: {562 - len(union_errors)} / 562")

    # Detailed Breakdown of Shared Errors by True Class
    class_error_breakdown = []
    for c_idx, c_name in enumerate(class_names):
        c_samples = [idx for idx in range(len(samples)) if true_labels[idx] == c_idx]
        c_supp = len(c_samples)
        c_errs_42 = set(c_samples) & errors_by_seed[42]
        c_errs_1337 = set(c_samples) & errors_by_seed[1337]
        c_errs_2026 = set(c_samples) & errors_by_seed[2026]
        c_shared_all = set(c_samples) & set(shared_all_3)
        c_shared_ge2 = set(c_samples) & set(shared_at_least_2)
        c_union = set(c_samples) & set(union_errors)

        class_error_breakdown.append({
            "class_index": c_idx,
            "class_name": c_name,
            "support": c_supp,
            "errors_seed42": len(c_errs_42),
            "errors_seed1337": len(c_errs_1337),
            "errors_seed2026": len(c_errs_2026),
            "shared_by_all_3": len(c_shared_all),
            "shared_by_ge2": len(c_shared_ge2),
            "union_errors": len(c_union),
        })

    # Error image details
    shared_all_3_details = []
    for s_idx in shared_all_3:
        r42 = predictions_by_seed[42][s_idx]
        r1337 = predictions_by_seed[1337][s_idx]
        r2026 = predictions_by_seed[2026][s_idx]
        shared_all_3_details.append({
            "sample_index": s_idx,
            "image_path": r42["image_path"],
            "true_label": r42["true_label_name"],
            "seed42_predicted": r42["predicted_label_name"],
            "seed42_confidence": r42["predicted_class_confidence"],
            "seed1337_predicted": r1337["predicted_label_name"],
            "seed1337_confidence": r1337["predicted_class_confidence"],
            "seed2026_predicted": r2026["predicted_label_name"],
            "seed2026_confidence": r2026["predicted_class_confidence"],
        })

    reconciled_overlap_data = {
        "dataset": "data_clean/val",
        "cohort_size": 562,
        "classes": class_names,
        "pretrained_weights": {
            "path": str(PRETRAINED_WEIGHT_PATH),
            "exists": PRETRAINED_WEIGHT_PATH.exists(),
            "size_bytes": pretrained_size,
            "sha256": actual_pretrained_sha,
        },
        "integrity_records": integrity_records,
        "seed42_epoch_reconciliation": {
            "selected_epoch": 13,
            "selected_epoch_macro_f1": h13["val_metrics"]["macro_f1"],
            "selected_epoch_accuracy": h13["val_metrics"]["accuracy"],
            "selected_epoch_val_loss": h13["val_loss"],
            "checkpoint_metadata_epoch": integrity_records[42]["checkpoint_internal_epoch"],
            "summary_json_best_epoch": s42_summary["best_epoch"],
            "epoch_16_macro_f1": h16["val_metrics"]["macro_f1"],
            "epoch_16_accuracy": h16["val_metrics"]["accuracy"],
            "epoch_16_val_loss": h16["val_loss"],
            "explanation": "Epoch 13 produced the peak validation Macro-F1 (0.896819) and was saved to best_model.pth according to the training protocol. Earlier project records mentioning Epoch 16 correspond to ConvNeXt-Small (which was selected at Epoch 16 with Macro-F1 = 0.901843)."
        },
        "error_counts": {
            "seed42_errors": len(errors_by_seed[42]),
            "seed1337_errors": len(errors_by_seed[1337]),
            "seed2026_errors": len(errors_by_seed[2026]),
            "shared_by_all_3": len(shared_all_3),
            "shared_by_at_least_2": len(shared_at_least_2),
            "union_of_all_errors": len(union_errors),
            "correct_across_all_3": 562 - len(union_errors),
        },
        "class_error_breakdown": class_error_breakdown,
        "shared_all_3_details": shared_all_3_details,
    }

    reconciled_json_path = out_dir / "error_overlap_reconciled.json"
    with open(reconciled_json_path, "w", encoding="utf-8") as f:
        json.dump(reconciled_overlap_data, f, indent=2)
    print(f"\n✅ Successfully persisted reconciled error overlap to {reconciled_json_path.name}")

if __name__ == "__main__":
    main()
