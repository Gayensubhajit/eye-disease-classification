#!/usr/bin/env python3
"""
Three-Seed Baseline Runner for the 10-Class Benchmark (Track A / EXP-02).

Architectural Protocol:
- Architecture: EfficientNet-B0 (ImageNet pre-trained)
- Task: 10-Class Benchmark (Preserves all 10 nominal categories, including audited Pterygium)
- Seeds: 42, 1337, 2026
- Learning Rate: 3e-4 (AdamW)
- Weight Decay: 1e-4
- Loss: Standard Unweighted Cross-Entropy
- Scheduler: Cosine Annealing (20 epochs)
- Batch Size: 16
- Resolution: 224x224
- Output Structure: outputs/clean_10class_baseline_effnet/seed_{seed}/
- Internal Test Set: STRICTLY QUARANTINED (no test inference until explicit protocol authorization)
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import math
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
OUTPUT_ROOT = ROOT / "outputs/clean_10class_baseline_effnet"

CLASS_NAMES_10 = [
    "Central Serous Chorioretinopathy [Color Fundus]",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]

EXPECTED_VAL_SUPPORTS_10 = {
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


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    """Compute Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p_hat = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p_hat + z**2 / (2 * total)) / denominator
    adjusted_std_error = (z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)) / denominator
    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)
    return lower, upper


def compute_high_specificity_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    target_specificities: List[float] = [0.90, 0.95],
    eps: float = 1e-6,
) -> Dict[str, Dict[str, Any]]:
    """Select One-vs-Rest operating thresholds targeting >= 90% and >= 95% specificity on validation data."""
    results = {}
    for c_idx, c_name in enumerate(class_names):
        pos_mask = (y_true == c_idx)
        neg_mask = (y_true != c_idx)
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        c_probs = y_prob[:, c_idx]
        candidate_thresholds = np.unique(np.concatenate([[0.0], c_probs, [1.0 + eps]]))

        c_results = {}
        for target_spec in target_specificities:
            feasible = []
            for t in candidate_thresholds:
                spec = np.sum(c_probs[neg_mask] < t) / n_neg if n_neg > 0 else 1.0
                sens = np.sum(c_probs[pos_mask] >= t) / n_pos if n_pos > 0 else 0.0
                if spec >= target_spec:
                    feasible.append((t, sens, spec))

            assert len(feasible) > 0, f"Feasible set empty for class {c_name} at target {target_spec}"

            max_sens = max(item[1] for item in feasible)
            top_sens = [item for item in feasible if item[1] == max_sens]
            selected_t, achieved_sens, achieved_spec = min(top_sens, key=lambda x: x[0])

            c_results[f"spec_{int(target_spec * 100)}"] = {
                "target_specificity": target_spec,
                "selected_threshold": float(selected_t),
                "achieved_sensitivity": float(achieved_sens),
                "achieved_specificity": float(achieved_spec),
                "positive_support": n_pos,
                "negative_support": n_neg,
            }
        results[c_name] = c_results
    return results


def build_seed_config(seed: int, seed_dir: Path) -> Path:
    """Construct seed-specific configuration file based on clean_baseline_efficientnet_b0.yaml."""
    base_cfg_path = ROOT / "configs/clean_baseline_efficientnet_b0.yaml"
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = seed
    cfg["output"]["directory"] = str(seed_dir)
    cfg["training"]["device"] = "cuda"
    cfg["training"]["epochs"] = 20
    cfg["training"]["use_amp"] = False  # Pure FP32 determinism
    cfg["training"]["evaluate_test"] = False  # Quarantine internal test set

    config_path = seed_dir / f"config_seed_{seed}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return config_path


def run_training_subprocess(config_path: Path) -> None:
    """Execute training script as isolated subprocess ensuring determinism flags."""
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        str(ROOT / "src/train.py"),
        "--config", str(config_path),
    ]

    print(f"\n--- Spawning isolated training process: {' '.join(cmd)} ---")
    start_t = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    elapsed = time.time() - start_t

    if result.returncode != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {result.returncode} after {elapsed:.1f}s")
    print(f"--- Training process finished successfully in {elapsed / 60:.2f} minutes ---")


@torch.no_grad()
def evaluate_and_reconcile_seed(seed: int, seed_dir: Path, config_path: Path) -> Dict[str, Any]:
    """Load best checkpoint, evaluate validation set, and extract metrics."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt_path = seed_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Compute checkpoint SHA-256
    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    ckpt_sha256 = sha256.hexdigest()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_epoch = checkpoint["epoch"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loader, val_loader, test_loader = create_dataloaders(config)
    del test_loader  # Quarantine test loader immediately

    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in val_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    assert len(y_true) == 562, f"Validation samples count mismatch: expected 562, got {len(y_true)}"

    # Operating thresholds
    operating_thresholds = compute_high_specificity_thresholds(
        y_true, y_prob, CLASS_NAMES_10, target_specificities=[0.90, 0.95]
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES_10))))
    correct_samples = int(np.trace(cm))
    acc_cm = correct_samples / len(y_true)

    # Per-class metrics
    per_class_dict = {}
    f1_scores = []
    recalls = []
    precisions = []
    specificities = []

    for idx, c_name in enumerate(CLASS_NAMES_10):
        tp = int(cm[idx, idx])
        fn = int(np.sum(cm[idx, :]) - tp)
        fp = int(np.sum(cm[:, idx]) - tp)
        tn = int(len(y_true) - (tp + fn + fp))
        supp = int(tp + fn)

        assert supp == EXPECTED_VAL_SUPPORTS_10[c_name], (
            f"Validation support mismatch for class {c_name}: expected {EXPECTED_VAL_SUPPORTS_10[c_name]}, got {supp}"
        )

        rec = tp / supp if supp > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        specificities.append(spec)
        f1_scores.append(f1)

        entry = {
            "class_index": idx,
            "validation_support": supp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "sensitivity": float(rec),
            "specificity": float(spec),
            "precision": float(prec),
            "f1": float(f1),
        }
        if c_name == "Pterygium":
            ci_low, ci_high = wilson_score_interval(tp, supp)
            entry["wilson_95_ci_sensitivity"] = [float(ci_low), float(ci_high)]

        per_class_dict[c_name] = entry

    macro_f1 = float(np.mean(f1_scores))
    balanced_acc = float(np.mean(recalls))
    macro_prec = float(np.mean(precisions))
    macro_spec = float(np.mean(specificities))

    # Retrieve history loss
    history_file = seed_dir / "training_history.json"
    train_loss_at_peak = None
    val_loss_at_peak = None
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            hist = json.load(f)
        for ep_data in hist:
            if ep_data.get("epoch") == best_epoch:
                train_loss_at_peak = ep_data.get("train_loss")
                val_loss_at_peak = ep_data.get("val_loss")
                break

    val_metrics_out = {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "validation_samples": len(y_true),
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
        "operating_thresholds": operating_thresholds,
        "per_class": per_class_dict,
    }

    with open(seed_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics_out, f, indent=2)

    with open(seed_dir / "best_model.pth.sha256", "w", encoding="utf-8") as f:
        f.write(f"{ckpt_sha256}  best_model.pth\n")

    print(f"Seed {seed} Reconciled Metrics: Val Macro-F1 = {macro_f1*100:.2f}%, "
          f"Balanced Acc = {balanced_acc*100:.2f}%, Accuracy = {acc_cm*100:.2f}% ({correct_samples}/562)")
    return val_metrics_out


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_seed_results = []

    print("=" * 70)
    print("STARTING 3-SEED 10-CLASS FULL BENCHMARK (EFFICIENTNET-B0 / TRACK A)")
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
    for c_name in CLASS_NAMES_10:
        c_f1s = [r["per_class"][c_name]["f1"] for r in all_seed_results]
        c_recalls = [r["per_class"][c_name]["sensitivity"] for r in all_seed_results]
        c_precs = [r["per_class"][c_name]["precision"] for r in all_seed_results]
        c_supp = EXPECTED_VAL_SUPPORTS_10[c_name]

        entry = {
            "validation_support": c_supp,
            "f1_mean": float(np.mean(c_f1s)),
            "f1_std": float(np.std(c_f1s, ddof=1)),
            "recall_mean": float(np.mean(c_recalls)),
            "recall_std": float(np.std(c_recalls, ddof=1)),
            "precision_mean": float(np.mean(c_precs)),
            "precision_std": float(np.std(c_precs, ddof=1)),
            "individual_seed_f1s": {f"seed_{SEEDS[i]}": float(c_f1s[i]) for i in range(len(SEEDS))},
        }
        if c_name == "Pterygium":
            entry["contingency_note"] = "Support N=2. Sensitivities restricted to {0.0, 0.5, 1.0}."
        per_class_agg[c_name] = entry

    aggregate_summary = {
        "experiment_title": "10-Class Benchmark Baseline (EfficientNet-B0 / Track A)",
        "protocol_version": "EXP_10CLASS_BENCHMARK_PROTOCOL v1.0",
        "seeds_evaluated": SEEDS,
        "sample_count_validation": 562,
        "classes_count": 10,
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
    print("3-SEED 10-CLASS BASELINE COMPLETE — AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"Validation Macro-F1:     {np.mean(f1_list)*100:.2f}% ± {np.std(f1_list, ddof=1)*100:.2f}%")
    print(f"Validation Balanced Acc: {np.mean(bacc_list)*100:.2f}% ± {np.std(bacc_list, ddof=1)*100:.2f}%")
    print(f"Validation Accuracy:     {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list, ddof=1)*100:.2f}%")
    print(f"Aggregate report written to: {agg_path}")
    print("Test set quarantine: ENFORCED (Zero test inferences executed).")


if __name__ == "__main__":
    main()
